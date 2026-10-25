"""YAML configuration loading, inheritance, and semantic validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.controllers.pid_controller import PIDConfig
from src.safety.thresholds import ThresholdConfig


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    config = _merge(load_config(path.parent / parent), raw) if parent else raw
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    for section in ("safety", "control", "training", "data", "evaluation", "storage"):
        if section not in config:
            raise ValueError(f"missing config section: {section}")
    safety = config["safety"]
    ThresholdConfig(
        max_degradation_ratio=float(safety["max_degradation_ratio"]),
        safe_ratio=float(safety["safe_ratio"]),
        warn_ratio=float(safety["warn_ratio"]),
    )
    control = config["control"]
    mode = control.get("mode")
    if mode not in {"no_kl", "fixed_kl", "pid_kl"}:
        raise ValueError("control.mode must be no_kl, fixed_kl, or pid_kl")
    beta = float(control["beta"])
    if mode == "no_kl" and beta != 0.0:
        raise ValueError("no_kl requires beta=0")
    if mode in {"fixed_kl", "pid_kl"} and beta <= 0:
        raise ValueError(f"{mode} requires beta > 0")
    if int(control["eval_interval_optimizer_steps"]) < 1:
        raise ValueError("eval interval must be >= 1 optimizer update")
    PIDConfig(**control["pid"])
    recovery = safety["recovery"]
    if int(recovery["max_retries"]) < 0 or int(recovery["hold_evaluations"]) < 1:
        raise ValueError("invalid recovery retry or hold setting")
    if float(recovery["beta_backoff"]) < 1:
        raise ValueError("recovery.beta_backoff must be >= 1")
    if not 0 < float(recovery["lr_backoff"]) < 1:
        raise ValueError("recovery.lr_backoff must be in (0, 1)")
    if not 0 < float(recovery["interval_backoff"]) < 1:
        raise ValueError("recovery.interval_backoff must be in (0, 1)")
    training = config["training"]
    if not isinstance(training.get("enabled"), bool):
        raise ValueError("training.enabled must be a boolean")
    if training["enabled"] and (
        not safety["hard_stop_enabled"] or not safety["rollback_enabled"]
    ):
        raise ValueError("real training requires Hard stop and rollback protection")
    if int(training["max_steps"]) < 1:
        raise ValueError("max_steps must be positive")
    if int(training["per_device_train_batch_size"]) < 1:
        raise ValueError("per_device_train_batch_size must be positive")
    if int(training["gradient_accumulation_steps"]) < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    generations = int(training["num_generations"])
    if generations < 2:
        raise ValueError("GRPO num_generations must be at least 2")
    if int(training["per_device_train_batch_size"]) % generations:
        raise ValueError("single-GPU batch size must be divisible by num_generations")
    data_backend = config["data"].get("backend", "local_jsonl")
    evaluation = config["evaluation"]
    eval_backend = evaluation.get("backend", "local_exact_match")
    if data_backend not in {"local_jsonl", "online_suite"} or eval_backend not in {"local_exact_match", "online_suite"}:
        raise ValueError("unsupported data/evaluation backend")
    if training.get("reward_backend", "numeric_exact_match") not in {"numeric_exact_match", "math_verify"}:
        raise ValueError("unsupported training reward_backend")
    if data_backend == "online_suite":
        if eval_backend != "online_suite" or training.get("reward_backend") != "math_verify":
            raise ValueError("online DeepMath requires online evaluation and math_verify reward")
    required_paths = [] if data_backend == "online_suite" else ["train_file"]
    if eval_backend == "local_exact_match":
        required_paths.extend(["control_file", "domain_validation_file", "final_test_file"])
    for name in required_paths:
        if not config["data"].get(name):
            raise ValueError(f"missing data path: {name}")
    if eval_backend == "online_suite":
        if not evaluation.get("suite_config") or not evaluation.get("manifest_file"):
            raise ValueError("online evaluation requires suite_config and manifest_file")
        if int(evaluation.get("max_context_length", 0)) <= int(evaluation["max_new_tokens"]):
            raise ValueError("evaluation context must exceed generation budget")
    if int(evaluation["batch_size"]) < 1 or int(evaluation["max_new_tokens"]) < 1:
        raise ValueError("evaluation sizes must be positive")
    storage = config["storage"]
    if not storage.get("root_dir") or not storage.get("cache_dir"):
        raise ValueError("storage root_dir and cache_dir are required")
    if float(storage["min_free_gb"]) < 0:
        raise ValueError("storage.min_free_gb must be non-negative")
