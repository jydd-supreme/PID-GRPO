"""Persist only frozen, selected benchmark rows across training restarts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from src.evaluation.online_data import digest, normalize_row


class SelectedSourceCache:
    def __init__(self, root: str | Path, *, expected_manifest: Mapping[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.expected_manifest = expected_manifest

    def _check_frozen_selection(self, task: str,
                                selected: Mapping[str, list[dict[str, Any]]]) -> None:
        if self.expected_manifest is None:
            return
        for role, samples in self.expected_manifest["roles"].items():
            expected = [item for item in samples if item["id"].startswith(f"{task}:")]
            if expected:
                actual = [{"id": row["id"], "content_hash": row["content_hash"]}
                          for row in selected[role]]
                if actual != expected:
                    raise ValueError(f"{task}/{role}: selection differs from frozen manifest")

    def _path(self, suite: Mapping[str, Any], task: str) -> Path:
        if not re.fullmatch(r"[a-z0-9_]+", task):
            raise ValueError(f"invalid cache task name: {task}")
        return self.root / digest(suite) / f"{task}.json"

    def load(self, suite: Mapping[str, Any], task: str,
             source: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]] | None:
        path = self._path(suite, task)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload["version"] != 1 or payload["suite_sha256"] != digest(suite)
                or payload["source_sha256"] != digest(source)):
                raise ValueError("selection identity mismatch")
            selected = payload["selected"]
            allocations = source["allocations"]
            if set(selected) != set(allocations):
                raise ValueError("selection roles differ")
            for role, rows in selected.items():
                count = allocations[role]
                if not isinstance(rows, list) or (count is not None and len(rows) != count):
                    raise ValueError(f"invalid {role} count")
                if count is None and source.get("expected_rows") is not None:
                    if len(rows) != source["expected_rows"]:
                        raise ValueError(f"invalid full {role} count")
                for row in rows:
                    if row != normalize_row(task, source, row["raw"]):
                        raise ValueError("cached row content mismatch")
            self._check_frozen_selection(task, selected)
            return selected
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid selected-row cache {path}: {exc}") from exc

    def save(self, suite: Mapping[str, Any], task: str, source: Mapping[str, Any],
             selected: Mapping[str, list[dict[str, Any]]]) -> None:
        self._check_frozen_selection(task, selected)
        path = self._path(suite, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "suite_sha256": digest(suite),
                   "source_sha256": digest(source), "selected": selected}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             prefix=f".{task}-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
