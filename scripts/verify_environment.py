#!/usr/bin/env python3
"""Fail-fast verification for the pinned CUDA and Hugging Face runtime."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()
    import torch

    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append(f"expected Python 3.12, found {sys.version.split()[0]}")
    if torch.__version__ != "2.5.1+cu124":
        errors.append(f"expected torch 2.5.1+cu124, found {torch.__version__}")
    if torch.version.cuda != "12.4":
        errors.append(f"expected CUDA build 12.4, found {torch.version.cuda}")
    if not args.base_only:
        expected = {
            "transformers": "4.49.0",
            "trl": "0.15.2",
            "peft": "0.14.0",
            "datasets": "3.3.2",
            "accelerate": "1.4.0",
        }
        for package, wanted in expected.items():
            try:
                actual = version(package)
            except PackageNotFoundError:
                errors.append(f"{package} is not installed")
            else:
                if actual != wanted:
                    errors.append(f"expected {package} {wanted}, found {actual}")
        if not errors:
            from trl import GRPOConfig, GRPOTrainer

            if GRPOConfig().beta != 0.04 or GRPOTrainer is None:
                errors.append("TRL GRPO API smoke check failed")
    if errors:
        raise SystemExit("\n".join(errors))
    print("environment verified")


if __name__ == "__main__":
    main()
