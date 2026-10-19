#!/usr/bin/env python3
"""Prepare deterministic, mutually exclusive GSM8K smoke-test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rewards.math_reward import extract_answer
from src.utils.config import load_config
from src.utils.storage import configure_runtime_storage


DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
SEED = 42
COUNTS = {"train": 128, "control": 64, "domain_validation": 16, "final_test": 16}


def _record(source_split: str, source_index: int, row: dict[str, Any]) -> dict[str, Any]:
    question = str(row["question"]).strip()
    return {
        "id": f"gsm8k-{source_split}-{source_index:05d}",
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"Solve this grade-school math problem. Show concise reasoning, then end "
                    f"with the final numeric answer.\n\n{question}"
                ),
            }
        ],
        "answer": extract_answer(str(row["answer"])),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_qwen_pid.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    storage = configure_runtime_storage(config)

    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
        cache_dir=storage["HF_DATASETS_CACHE"],
    )
    train_indices = list(range(len(dataset["train"])))
    test_indices = list(range(len(dataset["test"])))
    random.Random(SEED).shuffle(train_indices)
    random.Random(SEED).shuffle(test_indices)

    train_rows = [
        _record("train", index, dict(dataset["train"][index]))
        for index in train_indices[: COUNTS["train"]]
    ]
    cursor = 0
    evaluation_rows: dict[str, list[dict[str, Any]]] = {}
    for name in ("control", "domain_validation", "final_test"):
        count = COUNTS[name]
        chosen = test_indices[cursor : cursor + count]
        evaluation_rows[name] = [
            _record("test", index, dict(dataset["test"][index])) for index in chosen
        ]
        cursor += count

    paths = {
        "train": Path(config["data"]["train_file"]),
        "control": Path(config["data"]["control_file"]),
        "domain_validation": Path(config["data"]["domain_validation_file"]),
        "final_test": Path(config["data"]["final_test_file"]),
    }
    splits = {"train": train_rows, **evaluation_rows}
    hashes = {name: _write_jsonl(paths[name], rows) for name, rows in splits.items()}
    all_ids = [row["id"] for rows in splits.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("prepared splits are not mutually exclusive")

    manifest = {
        "purpose": "engineering smoke test; not a formal benchmark result",
        "dataset": DATASET_NAME,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "license": "MIT",
        "seed": SEED,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "sha256": hashes,
    }
    manifest_path = Path(config["storage"]["root_dir"]) / "data" / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
