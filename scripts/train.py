#!/usr/bin/env python3
"""Guarded real GRPO training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.storage import configure_runtime_storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pid_kl.yaml")
    parser.add_argument("--enable-real-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--session-max-seconds", type=float)
    args = parser.parse_args()
    config = load_config(args.config)
    if not args.enable_real_training or not config["training"]["enabled"]:
        raise SystemExit(
            "Real training is disabled. Both --enable-real-training and training.enabled=true "
            "are required."
        )
    configure_runtime_storage(config)
    from src.trainers.real_grpo import run_real_training

    print(json.dumps(run_real_training(config, resume=args.resume,
                                       session_max_seconds=args.session_max_seconds), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
