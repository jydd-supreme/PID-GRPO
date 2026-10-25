"""Data boundary shared by experiment setup and the GRPO training backend."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.data.dataset_builder import load_local_jsonl
from src.evaluation.online_data import check_manifest, digest, load_suite, materialize_roles


@dataclass
class ExperimentData:
    splits: dict[str, list[dict[str, Any]]]
    manifest: dict[str, Any] | None = None


def repository_path(path: str, repository: Path | None = None) -> Path:
    return (repository or Path(__file__).resolve().parents[2]) / path


def training_roles(config: Mapping[str, Any]) -> list[str]:
    roles = ["control", "domain_validation"]
    if config["data"].get("backend", "local_jsonl") == "online_suite":
        roles.insert(0, "train")
    return roles


def training_record(row: Mapping[str, Any]) -> dict[str, Any]:
    if row["kind"] != "math":
        raise ValueError("GRPO math training requires math samples")
    return {"id": row["id"], "prompt": [{"role": "user", "content": row["prompt"]}],
            "answer": row["answer"], "question_hash": row["question_hash"]}


def load_experiment_data(config: Mapping[str, Any], *, repository: Path | None = None,
                         row_loader: Any = None) -> ExperimentData:
    data = config["data"]
    evaluation = config["evaluation"]
    if evaluation.get("backend", "local_exact_match") != "online_suite":
        names = ("train_file", "control_file", "domain_validation_file", "final_test_file")
        splits = {name: load_local_jsonl(data[name]) for name in names}
        seen = set()
        for rows in splits.values():
            for row in rows:
                if str(row["id"]) in seen:
                    raise ValueError(f"dataset leakage: duplicate id {row['id']}")
                seen.add(str(row["id"]))
        return ExperimentData(splits)

    suite = load_suite(repository_path(evaluation["suite_config"], repository))
    expected = json.loads(repository_path(evaluation["manifest_file"], repository).read_text(encoding="utf-8"))
    # Reject configuration mismatch before any remote request.
    if expected["suite_sha256"] != digest(suite):
        raise ValueError("suite configuration differs from frozen manifest")
    kwargs = {"row_loader": row_loader} if row_loader is not None else {}
    online, manifest = materialize_roles(suite, training_roles(config), **kwargs)
    check_manifest(expected, manifest)
    online_training = data.get("backend", "local_jsonl") == "online_suite"
    train = ([training_record(row) for row in online["train"]] if online_training
             else load_local_jsonl(data["train_file"]))
    evaluation_hashes = {row["question_hash"] for role, rows in online.items()
                         if role != "train" for row in rows}
    for row in train:
        prompt = row["prompt"]
        if isinstance(prompt, list):
            prompt = "\n".join(str(message.get("content", "")) for message in prompt)
        key = row.get("question_hash", digest(" ".join(str(prompt).casefold().split())))
        if key in evaluation_hashes:
            raise ValueError("training question overlaps evaluation")
    if not train:
        raise ValueError("empty training split")
    return ExperimentData({"train_file": train, "control_file": online["control"],
                           "domain_validation_file": online["domain_validation"]}, manifest)
