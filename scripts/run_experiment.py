#!/usr/bin/env python3
"""Stage the DeepMath -> GRPO -> benchmark -> PID experiment on a server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.experiment_data import repository_path, training_roles
from src.evaluation.online_data import digest, load_suite, materialize_roles
from src.utils.config import load_config, validate_config
from src.utils.experiment import experiment_plan, runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "freeze", "preflight", "train"])
    parser.add_argument("--config", default="configs/experiments/deepmath_pid.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root-dir", help="server data disk; remaps cache/data/output paths together")
    parser.add_argument("--enable-real-training", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue the same mode/seed/output after restart")
    parser.add_argument("--session-max-seconds", type=float, help="session budget; 0 disables planned pausing")
    args = parser.parse_args()
    config = runtime_config(load_config(args.config), seed=args.seed, root_dir=args.root_dir)
    validate_config(config)
    plan = experiment_plan(config)
    if args.stage == "plan":
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if args.stage == "train" and (not args.enable_real_training or not config["training"]["enabled"]):
        parser.error("train requires --enable-real-training and training.enabled=true")
    path = repository_path(config["evaluation"]["manifest_file"])
    suite = load_suite(repository_path(config["evaluation"]["suite_config"]))
    if args.stage == "freeze":
        if path.exists():
            parser.error("manifest exists; reuse it or select a new manifest path")
        from src.utils.storage import configure_runtime_storage
        configure_runtime_storage(config)
        _, manifest = materialize_roles(suite, training_roles(config))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": str(path), "counts": {r: len(v) for r, v in manifest["roles"].items()}}, indent=2))
        return
    if not path.is_file():
        parser.error("missing data manifest; run freeze first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["suite_sha256"] != digest(suite) or not set(training_roles(config)) <= manifest["roles"].keys():
        parser.error("manifest does not match this experiment")
    if args.stage == "preflight":
        from src.trainers.real_grpo import verify_training_environment
        from src.rewards.verified_math_reward import require_math_verifier
        from src.utils.storage import configure_runtime_storage
        configure_runtime_storage(config)
        verify_training_environment()
        require_math_verifier()
        print(json.dumps({"status": "ready_for_smoke", "plan": plan}, indent=2))
        return
    from src.trainers.real_grpo import run_real_training
    result = run_real_training(config, resume=args.resume, session_max_seconds=args.session_max_seconds)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
