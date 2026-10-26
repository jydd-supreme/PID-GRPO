"""Pinned streaming benchmark selection without writing dataset contents to disk."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
from typing import Any, Callable, Iterable, Mapping

import yaml

FINAL_ROLES = frozenset({"forgetting", "final_math", "instruction"})
ROLES = frozenset({"train", "domain_validation", "control"}) | FINAL_ROLES


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def load_suite(path: str | Path) -> dict[str, Any]:
    suite = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if suite.get("version") != 1 or suite.get("aggregation") != "sample_weighted":
        raise ValueError("unsupported suite version or aggregation")
    for name, source in suite["sources"].items():
        if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", ""))):
            raise ValueError(f"{name}: immutable dataset revision required")
        allocations = source["allocations"]
        if not allocations or not set(allocations) <= ROLES:
            raise ValueError(f"{name}: invalid allocations")
        if any(n is not None and (type(n) is not int or n < 1) for n in allocations.values()):
            raise ValueError(f"{name}: counts must be positive integers or null")
        if None in allocations.values() and len(allocations) != 1:
            raise ValueError(f"{name}: full split must belong to one role")
        if source["kind"] not in {"math", "mmlu", "arc", "hellaswag", "ifeval"}:
            raise ValueError(f"{name}: unsupported scorer")
        if source["selection"] not in {"buffered", "reservoir", "stratified"}:
            raise ValueError(f"{name}: unsupported selection")
    return suite


def remote_rows(source: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    # Lazy import: manifest/unit tests do not need the training dependency set.
    from datasets import load_dataset

    # Push projection into Parquet, rather than dropping long solution columns
    # only after the remote row has already been read.
    kwargs = {"columns": source["columns"]} if source.get("columns") else {}
    stream = load_dataset(source["path"], name=source.get("name"),
                          revision=source["revision"], split=source["split"],
                          streaming=True, **kwargs)
    return stream


def _reservoir(rows: Iterable[dict[str, Any]], count: int | None,
               rng: random.Random) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if count is None or index < count:
            selected.append(dict(row))
        else:
            target = rng.randrange(index + 1)
            if target < count:
                selected[target] = dict(row)
    if count is not None and len(selected) != count:
        raise ValueError(f"source too small: need {count}, got {len(selected)}")
    rng.shuffle(selected)
    return selected


def _buffered(rows: Iterable[dict[str, Any]], count: int, size: int,
              rng: random.Random) -> list[dict[str, Any]]:
    if size < 1:
        raise ValueError("buffer_size must be positive")
    iterator = iter(rows)
    buffer = list(itertools.islice(iterator, size))
    result = []
    while buffer and len(result) < count:
        index = rng.randrange(len(buffer))
        result.append(dict(buffer[index]))
        replacement = next(iterator, None)
        if replacement is None:
            buffer.pop(index)
        else:
            buffer[index] = replacement
    if len(result) != count:
        raise ValueError(f"source too small: need {count}, got {len(result)}")
    return result


def _stratified(rows: Iterable[dict[str, Any]], allocations: Mapping[str, int],
                rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    # MMLU is small: scan in memory to guarantee coverage of every subject.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["subject"])].append(dict(row))
    subjects = sorted(groups)
    if not subjects:
        raise ValueError("empty stratified source")
    rng.shuffle(subjects)
    for group in groups.values():
        rng.shuffle(group)
    offsets = defaultdict(int)
    result = {}
    for role, count in allocations.items():
        chosen = []
        for index, subject in enumerate(subjects):
            quota = count // len(subjects) + (index < count % len(subjects))
            start = offsets[subject]
            values = groups[subject][start:start + quota]
            if len(values) != quota:
                raise ValueError(f"insufficient {subject} samples for {role}")
            chosen.extend(values)
            offsets[subject] += quota
        rng.shuffle(chosen)
        result[role] = chosen
    return result


def _clean_hellaswag(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def normalize_row(task: str, source: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    kind = source["kind"]
    row: dict[str, Any] = {"task": task, "kind": kind, "raw": dict(raw)}
    if kind == "mmlu":
        options = list(raw["choices"])
        prompt = f"The following is a multiple choice question about {raw['subject'].replace('_', ' ')}.\n\n{raw['question']}\n"
        prompt += "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(options))
        row.update(prompt=prompt + "\nAnswer:", choices=[chr(65+i) for i in range(len(options))],
                   gold=int(raw["answer"]), metric="acc")
    elif kind == "arc":
        labels = list(raw["choices"]["label"])
        row.update(prompt="Question: " + raw["question"] + "\nAnswer:",
                   choices=list(raw["choices"]["text"]),
                   gold=labels.index(str(raw["answerKey"])), metric="acc_norm")
    elif kind == "hellaswag":
        context = raw["ctx_a"] + " " + raw["ctx_b"].capitalize()
        row.update(prompt=_clean_hellaswag(raw["activity_label"] + ": " + context),
                   choices=[_clean_hellaswag(x) for x in raw["endings"]],
                   gold=int(raw["label"]), metric="acc_norm")
    elif kind == "math":
        answer = str(raw.get("final_answer", raw.get("answer", "")))
        if task == "gsm8k":
            if "####" not in answer:
                raise ValueError("GSM8K reference lacks final answer marker")
            answer = answer.rsplit("####", 1)[1].strip()
        prompt = str(raw.get("problem", raw.get("question", "")))
        if not prompt.strip() or not answer.strip():
            raise ValueError(f"{task}: empty question or reference")
        row.update(prompt=prompt + "\nShow your reasoning and put the final answer in \\boxed{}.",
                   answer=answer, metric="math_verify")
    else:
        row.update(prompt=str(raw["prompt"]), metric="prompt_level_strict_acc")
    if "choices" in row and (not row["choices"] or not 0 <= row["gold"] < len(row["choices"])):
        raise ValueError(f"{task}: invalid multiple-choice label")
    # Source content, rather than row position, survives stream implementation changes.
    row["id"] = f"{task}:{digest(raw)}"
    row["content_hash"] = digest(raw)
    question = raw.get("question", raw.get("problem", raw.get("prompt", row["prompt"])))
    row["question_hash"] = digest(" ".join(str(question).casefold().split()))
    return row


def _unique_questions(rows: Iterable[dict[str, Any]], task: str,
                      source: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    """Skip repeated source questions before sampling a fixed-size split."""
    seen: set[str] = set()
    for raw in rows:
        key = normalize_row(task, source, raw)["question_hash"]
        if key not in seen:
            seen.add(key)
            yield raw


def materialize_roles(suite: Mapping[str, Any], roles: Iterable[str], *,
                      row_loader: Callable = remote_rows) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    requested = set(roles)
    if not requested or not requested <= ROLES:
        raise ValueError("unknown or empty evaluation roles")
    result: dict[str, list[dict[str, Any]]] = {role: [] for role in sorted(requested)}
    manifest: dict[str, Any] = {"version": 1, "suite_sha256": digest(suite),
                               "protocol": suite["protocol"], "roles": {}}
    seen: dict[str, str] = {}
    for task, source in suite["sources"].items():
        allocations = source["allocations"]
        if not requested.intersection(allocations):
            continue
        rng = random.Random(int(digest([suite["seed"], task])[:16], 16))
        rows = row_loader(source)
        # Fixed-size splits must contain the requested number of distinct
        # questions. Full final splits keep every row and retain the duplicate
        # guard below so an advertised full-set count cannot shrink silently.
        if None not in allocations.values():
            rows = _unique_questions(rows, task, source)
        if source["selection"] == "stratified":
            selected = _stratified(rows, allocations, rng)
        else:
            total = None if None in allocations.values() else sum(allocations.values())
            pool = (_buffered(rows, total, source["buffer_size"], rng)
                    if source["selection"] == "buffered" else _reservoir(rows, total, rng))
            if source.get("expected_rows") is not None and len(pool) != source["expected_rows"]:
                raise ValueError(f"{task}: source count changed, got {len(pool)}")
            selected = {}
            cursor = 0
            for role, count in allocations.items():
                selected[role] = pool[cursor:cursor + count] if count is not None else pool
                cursor += count or len(pool)
        # Check all allocations, even when only one role was requested.
        for role, values in selected.items():
            for raw in values:
                row = normalize_row(task, source, raw)
                key = row["question_hash"]
                if key in seen:
                    raise ValueError(f"duplicate question in {seen[key]} and {role}: {row['id']}")
                seen[key] = role
                if role in requested:
                    row["role"] = role
                    result[role].append(row)
    for role, values in result.items():
        if not values:
            raise ValueError(f"empty role: {role}")
        manifest["roles"][role] = [{"id": r["id"], "content_hash": r["content_hash"]} for r in values]
    return result, manifest


def check_manifest(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    if expected["suite_sha256"] != actual["suite_sha256"]:
        raise ValueError("suite configuration differs from frozen manifest")
    for role, samples in actual["roles"].items():
        if expected["roles"].get(role) != samples:
            raise ValueError(f"sample identity/content changed for {role}")
