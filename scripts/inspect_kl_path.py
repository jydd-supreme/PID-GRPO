#!/usr/bin/env python3
"""Inspect an already-installed TRL package without installing dependencies."""

from __future__ import annotations

import importlib.util
import inspect


def main() -> None:
    if importlib.util.find_spec("trl") is None:
        raise SystemExit("TRL is not installed; no package was downloaded or installed.")
    from trl import GRPOConfig, GRPOTrainer

    print(f"GRPOConfig.beta default: {inspect.signature(GRPOConfig).parameters['beta'].default}")
    source = inspect.getsource(GRPOTrainer)
    for needle in ("self.beta = args.beta", "if self.beta == 0.0", "self.beta * per_token_kl"):
        print(f"{needle!r}: {needle in source}")


if __name__ == "__main__":
    main()
