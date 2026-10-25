"""Durable session journal, interruption budget and atomic restart checkpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
from typing import Any, Callable, Mapping
import uuid


class SessionPaused(RuntimeError):
    """A planned interruption, never a capability failure."""


class TrainingBudgetComplete(RuntimeError):
    """Resume entered with the complete optimizer budget; export without a step."""


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class RunSession:
    def __init__(self, output_dir: Path, *, max_seconds: float = 0,
                 clock: Callable = time.monotonic) -> None:
        self.output_dir = Path(output_dir)
        if max_seconds < 0:
            raise ValueError("session limit must be non-negative")
        self.session_id = uuid.uuid4().hex
        self.clock = clock
        self.started = clock()
        self.max_seconds = max_seconds
        self.pause_reason: str | None = None
        self._handlers: dict[int, Any] = {}

    def expired(self) -> bool:
        return self.pause_reason is not None or (self.max_seconds > 0 and
                                                 self.clock() - self.started >= self.max_seconds)

    def check(self) -> None:
        if self.expired():
            raise SessionPaused(self.pause_reason or "session time budget reached")

    def install_handlers(self) -> None:
        for number in (signal.SIGTERM, signal.SIGINT):
            self._handlers[number] = signal.getsignal(number)
            signal.signal(number, lambda signum, frame: setattr(self, "pause_reason", signal.Signals(signum).name))

    def close(self) -> None:
        for number, handler in self._handlers.items():
            signal.signal(number, handler)
        self._handlers.clear()

    def event(self, event: str, **details: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        value = {"session_id": self.session_id, "event": event,
                 "utc": datetime.now(timezone.utc).isoformat(), **details}
        with (self.output_dir / "sessions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class RestartCheckpointStore:
    """Separate restart snapshots from capability-certified rollback targets."""

    def __init__(self, root: Path, *, keep_last: int = 3) -> None:
        if keep_last < 2:
            raise ValueError("keep_last must be at least 2")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.skipped: list[str] = []

    def save(self, state: Mapping[str, Any], artifact_writer: Callable[[Path], None]) -> Path:
        name = f"resume-{time.time_ns()}-step-{int(state['global_step']):08d}"
        temporary = self.root / ("." + name + ".tmp")
        destination = self.root / name
        temporary.mkdir()
        try:
            artifact_writer(temporary)
            artifacts = {}
            for path in sorted(temporary.rglob("*")):
                if path.is_file():
                    with path.open("rb") as handle:
                        os.fsync(handle.fileno())
                    artifacts[str(path.relative_to(temporary))] = file_hash(path)
            if not artifacts:
                raise ValueError("restart checkpoint has no training artifacts")
            atomic_json(temporary / "restart_state.json", {"version": 1, **dict(state), "artifacts": artifacts})
            atomic_json(temporary / "COMMITTED.json", {"state_sha256": file_hash(temporary / "restart_state.json")})
            temporary.replace(destination)
            atomic_json(self.root / "latest.json", {"checkpoint": name, "global_step": state["global_step"]})
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        # Publish first; pruning must never remove the only complete checkpoint.
        for path in sorted(self.root.glob("resume-*"), reverse=True)[self.keep_last:]:
            if path.is_dir():
                shutil.rmtree(path)
        return destination

    def latest(self) -> tuple[Path, dict[str, Any]] | None:
        self.skipped = []
        # A kill after directory publication but before latest.json is recoverable.
        for path in sorted(self.root.glob("resume-*"), reverse=True):
            try:
                committed = json.loads((path / "COMMITTED.json").read_text(encoding="utf-8"))
                if committed["state_sha256"] != file_hash(path / "restart_state.json"):
                    raise ValueError("restart protocol checksum mismatch")
                state = json.loads((path / "restart_state.json").read_text(encoding="utf-8"))
                if state.get("version") != 1 or not state.get("artifacts"):
                    raise ValueError("invalid restart state")
                for name, expected in state["artifacts"].items():
                    artifact = Path(name)
                    if artifact.is_absolute() or ".." in artifact.parts or file_hash(path / artifact) != expected:
                        raise ValueError("training artifact checksum mismatch")
                return path, state
            except (OSError, ValueError, KeyError):
                self.skipped.append(path.name)
        return None

    def discard_incomplete(self) -> int:
        """Call only under the run lock; partial snapshots are not recoverable."""
        count = 0
        for path in self.root.glob(".resume-*.tmp"):
            if path.is_dir():
                shutil.rmtree(path)
                count += 1
        return count


def restore_control_state(loop: Any, payload: Mapping[str, Any]) -> None:
    if loop.pid is not None:
        loop.pid.load_state_dict(payload["pid_state"])
    loop.guard.load_state_dict(payload["guard_state"])
    loop.load_state_dict(payload["loop_state"])
    if loop.stop_requested:
        raise RuntimeError("a safety-stopped trajectory cannot be resumed as a normal session")
    known = {record.checkpoint_id for record in loop.checkpoint_manager.records}
    if not set(loop.active_checkpoint_ids) <= known:
        raise ValueError("restart references missing safety checkpoints")
