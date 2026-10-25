"""Lightweight checkpoint and rollback state machine.

This round persists JSON-compatible protocol state, not model weights or optimizer
tensors. Callers may provide lightweight trainer/optimizer state mappings. Future
TRL integration can implement the same capture/restore protocol with real state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import uuid
from typing import Any, Callable, Mapping, Protocol


class Stateful(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    optimizer_step: int
    control_step: int
    capability_score: float
    domain_validation_score: float | None
    safety_verified: bool
    path: str
    created_at_utc: str


@dataclass(frozen=True)
class RollbackRequest:
    requested: bool
    reason: str
    target_checkpoint_id: str | None
    created_at_utc: str


class CheckpointManager:
    MANIFEST = "manifest.json"
    STATE = "state.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records: list[CheckpointRecord] = []
        self.rollback_request: RollbackRequest | None = None
        self._load_manifest_if_present()

    @property
    def latest_safe(self) -> CheckpointRecord | None:
        safe = [record for record in self.records if record.safety_verified]
        return max(safe, key=lambda item: item.optimizer_step, default=None)

    @property
    def best_feasible(self) -> CheckpointRecord | None:
        eligible = [
            record
            for record in self.records
            if record.safety_verified and record.domain_validation_score is not None
        ]
        return max(eligible, key=lambda item: item.domain_validation_score, default=None)

    def best_feasible_among(self, checkpoint_ids: set[str]) -> CheckpointRecord | None:
        return max(
            (
                record for record in self.records
                if record.checkpoint_id in checkpoint_ids
                and record.safety_verified
                and record.domain_validation_score is not None
                and math.isfinite(record.domain_validation_score)
            ),
            key=lambda record: (record.domain_validation_score, record.optimizer_step),
            default=None,
        )

    def latest_rollback_candidate(
        self, minimum_score: float, checkpoint_ids: set[str]
    ) -> CheckpointRecord | None:
        return max(
            (
                record for record in self.records
                if record.checkpoint_id in checkpoint_ids
                and record.safety_verified
                and record.capability_score >= minimum_score
            ),
            key=lambda record: record.optimizer_step,
            default=None,
        )

    def save(
        self,
        checkpoint_id: str,
        *,
        optimizer_step: int,
        control_step: int,
        capability_score: float,
        domain_validation_score: float | None,
        safety_verified: bool,
        pid_state: Mapping[str, Any],
        safety_state: Mapping[str, Any],
        trainer_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        artifact_writer: Callable[[Path], None] | None = None,
    ) -> CheckpointRecord:
        if not checkpoint_id or "/" in checkpoint_id or ".." in checkpoint_id:
            raise ValueError("checkpoint_id must be a safe single path component")
        if optimizer_step < 0 or control_step < 0:
            raise ValueError("step counters must be non-negative")
        directory = self.root / checkpoint_id
        temporary = self.root / f".{checkpoint_id}.tmp"
        if directory.exists() or temporary.exists():
            raise FileExistsError(f"checkpoint already exists: {checkpoint_id}")
        temporary.mkdir(parents=False, exist_ok=False)
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id, optimizer_step=optimizer_step, control_step=control_step,
            capability_score=capability_score, domain_validation_score=domain_validation_score,
            safety_verified=safety_verified, path=str(directory), created_at_utc=self._now(),
        )
        payload = {
            "schema_version": 1,
            "checkpoint_id": checkpoint_id,
            "checkpoint_record": asdict(record),
            "pid_state": dict(pid_state),
            "safety_state": dict(safety_state),
            "trainer_state": dict(trainer_state),
            "optimizer_state": dict(optimizer_state),
            "metadata": dict(metadata or {}),
        }
        try:
            self._write_json_atomic(temporary / self.STATE, payload)
            if artifact_writer is not None:
                artifact_writer(temporary)
            temporary.replace(directory)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        self.records.append(record)
        self._write_manifest()
        return record

    def recover_interrupted_writes(self) -> None:
        """Call under the run's exclusive lock after a process restart."""
        known = {record.checkpoint_id for record in self.records}
        changed = False
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            if directory.name.startswith("."):
                if directory.name.endswith(".tmp"):
                    # Do not mistake a half-written checkpoint for a safe point.
                    directory.rename(self.root / (".abandoned-" + uuid.uuid4().hex))
                continue
            if directory.name in known:
                continue
            payload = json.loads((directory / self.STATE).read_text(encoding="utf-8"))
            raw_record = payload.get("checkpoint_record")
            if raw_record is None:
                raise RuntimeError("unindexed legacy checkpoint requires manual inspection")
            record = CheckpointRecord(**raw_record)
            if record.checkpoint_id != directory.name or Path(record.path).resolve() != directory.resolve():
                raise ValueError("published checkpoint identity mismatch")
            self.records.append(record)
            known.add(record.checkpoint_id)
            changed = True
        if changed:
            self._write_manifest()

    def read_payload(self, checkpoint_id: str) -> dict[str, Any]:
        record = next((item for item in self.records if item.checkpoint_id == checkpoint_id), None)
        if record is None:
            raise KeyError(f"unknown checkpoint: {checkpoint_id}")
        payload = json.loads((Path(record.path) / self.STATE).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported checkpoint schema")
        return payload

    def clear_rollback_request(self) -> None:
        self.rollback_request = None
        self._write_manifest()

    def request_rollback(
        self, reason: str, *, minimum_score: float | None = None,
        checkpoint_ids: set[str] | None = None,
    ) -> RollbackRequest:
        target = (
            self.latest_safe
            if minimum_score is None or checkpoint_ids is None
            else self.latest_rollback_candidate(minimum_score, checkpoint_ids)
        )
        request = RollbackRequest(
            requested=True,
            reason=reason,
            target_checkpoint_id=target.checkpoint_id if target else None,
            created_at_utc=self._now(),
        )
        self.rollback_request = request
        self._write_manifest()
        return request

    def restore(
        self,
        checkpoint_id: str,
        *,
        pid: Stateful,
        safety: Stateful,
        trainer: Stateful,
        optimizer: Stateful,
    ) -> CheckpointRecord:
        record = next((item for item in self.records if item.checkpoint_id == checkpoint_id), None)
        if record is None:
            raise KeyError(f"unknown checkpoint: {checkpoint_id}")
        if not record.safety_verified:
            raise PermissionError("rollback is allowed only to a safety-verified checkpoint")
        payload = self.read_payload(checkpoint_id)
        pid.load_state_dict(payload["pid_state"])
        safety.load_state_dict(payload["safety_state"])
        trainer.load_state_dict(payload["trainer_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        self.clear_rollback_request()
        return record

    def _load_manifest_if_present(self) -> None:
        path = self.root / self.MANIFEST
        if not path.exists():
            return
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported checkpoint manifest schema")
        self.records = [CheckpointRecord(**item) for item in manifest.get("records", [])]
        raw_request = manifest.get("rollback_request")
        self.rollback_request = RollbackRequest(**raw_request) if raw_request else None

    def _write_manifest(self) -> None:
        self._write_json_atomic(
            self.root / self.MANIFEST,
            {
                "schema_version": 1,
                "records": [asdict(item) for item in self.records],
                "rollback_request": asdict(self.rollback_request) if self.rollback_request else None,
                "latest_safe_id": self.latest_safe.checkpoint_id if self.latest_safe else None,
                "best_feasible_id": self.best_feasible.checkpoint_id if self.best_feasible else None,
            },
        )

    @staticmethod
    def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
