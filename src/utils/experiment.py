"""Experiment protocol inspection without model or dataset downloads."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from src.data.experiment_data import repository_path, training_roles
from src.evaluation.online_data import digest, load_suite


def runtime_config(config: Mapping[str, Any], *, seed: int | None = None,
                   root_dir: str | None = None) -> dict[str, Any]:
    result = deepcopy(dict(config))
    if root_dir is not None:
        old_root = Path(result["storage"]["root_dir"])
        root = Path(root_dir).expanduser().resolve()
        def relocated(value: str) -> str:
            return str(root / Path(value).relative_to(old_root))
        result["project"]["output_dir"] = relocated(result["project"]["output_dir"])
        result["storage"]["cache_dir"] = relocated(result["storage"]["cache_dir"])
        result["storage"]["root_dir"] = str(root)
        for name, value in result["data"].items():
            if name.endswith("_file"):
                result["data"][name] = relocated(value)
    if seed is not None:
        result["project"]["seed"] = seed
    # Run seeds change model sampling, not frozen benchmark membership.
    result["project"]["output_dir"] = str(Path(result["project"]["output_dir"]) /
                                           f"seed-{result['project']['seed']}")
    return result


def experiment_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    suite = load_suite(repository_path(config["evaluation"]["suite_config"]))
    training = config["training"]
    count = training["per_device_train_batch_size"] * training["gradient_accumulation_steps"]
    return {
        "model": training["model_name"], "model_revision": training["model_revision"],
        "mode": config["control"]["mode"], "initial_beta": config["control"]["beta"],
        "reward_backend": training["reward_backend"], "seed": config["project"]["seed"],
        "max_optimizer_updates": training["max_steps"],
        "eval_interval_optimizer_updates": config["control"]["eval_interval_optimizer_steps"],
        "completions_per_update": count, "questions_per_update": count // training["num_generations"],
        "max_sampled_completions": count * training["max_steps"],
        "suite_sha256": digest(suite), "manifest": config["evaluation"]["manifest_file"],
        "training_roles": training_roles(config), "final_roles": ["forgetting", "final_math", "instruction"],
        "allocations": {task: source["allocations"] for task, source in suite["sources"].items()},
        "output_dir": config["project"]["output_dir"],
        "restart_policy": config.get("checkpointing", {"interval_optimizer_steps": 25,
                                                        "interval_seconds": 600, "keep_last": 3,
                                                        "session_max_seconds": 0}),
    }
