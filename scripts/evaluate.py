#!/usr/bin/env python3
"""Run only a labeled mock evaluator in round one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.capability_evaluator import MockCapabilityEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", nargs="+", type=float)
    args = parser.parse_args()
    evaluator = MockCapabilityEvaluator(args.scores)
    for _ in args.scores:
        print(json.dumps(evaluator.evaluate(None, None), sort_keys=True))


if __name__ == "__main__":
    main()
