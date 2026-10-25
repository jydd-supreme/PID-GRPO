"""Reuse completed evaluation rows only for identical model weights and inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.evaluation.online_data import digest


def model_fingerprint(model: Any) -> str:
    import torch
    result = hashlib.sha256()
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not parameters:
        parameters = list(model.named_parameters())
    for name, value in parameters:
        result.update(f"{name}:{value.shape}:{value.dtype}".encode())
        result.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    # Buffers can affect inference even when they are not optimizer parameters.
    for name, value in model.named_buffers():
        result.update(name.encode())
        result.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


class EvaluationProgress:
    def __init__(self, root: Path, model: Any, rows: list[dict[str, Any]], identity: Any) -> None:
        key = digest([identity, model_fingerprint(model), [row["id"] for row in rows]])
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{key}.jsonl"
        self.results: dict[str, Any] = {}
        if self.path.exists():
            # Drop a partial last write caused by SIGKILL. Earlier records remain.
            valid_end = 0
            with self.path.open("rb") as handle:
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    try:
                        record = json.loads(line)
                        if not line.endswith(b"\n"):
                            break
                        self.results[record["id"]] = record["result"]
                    except (ValueError, KeyError):
                        break
                    valid_end = handle.tell()
            with self.path.open("r+b") as handle:
                handle.truncate(valid_end)

    def append(self, sample_id: str, result: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"id": sample_id, "result": result}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.results[sample_id] = result
