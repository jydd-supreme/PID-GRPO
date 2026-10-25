"""Local JSONL dataset loading and split isolation checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.capability_evaluator import (
    DatasetDescriptor,
    DatasetRole,
    validate_dataset_isolation,
)


REQUIRED_FIELDS = {"id", "prompt", "answer"}


def load_local_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
        sample_id = str(row["id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample id in {path}: {sample_id}")
        seen.add(sample_id)
        rows.append(row)
    return rows


def build_local_splits(
    train_path: str | Path,
    control_path: str | Path,
    final_test_path: str | Path,
) -> dict[DatasetRole, list[dict[str, Any]]]:
    paths = {
        DatasetRole.TRAIN: Path(train_path),
        DatasetRole.CONTROL: Path(control_path),
        DatasetRole.FINAL_TEST: Path(final_test_path),
    }
    splits = {role: load_local_jsonl(path) for role, path in paths.items()}
    descriptors = [
        DatasetDescriptor(path.name, role, frozenset(str(row["id"]) for row in splits[role]))
        for role, path in paths.items()
    ]
    validate_dataset_isolation(descriptors)
    return splits
