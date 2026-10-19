#!/usr/bin/env python3
"""Plot recorded CSV metrics, visibly watermarking mock/synthetic inputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="outputs/figures/metrics.png")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Plotting requires the optional matplotlib dependency") from exc
    with Path(args.csv_path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("metrics CSV contains no rows")
    steps = [int(row["global_step"]) for row in rows]
    fields = ["general_capability_raw", "safe_threshold", "warn_threshold", "hard_threshold", "beta_new"]
    figure, axes = plt.subplots(2, 1, sharex=True)
    for field in fields[:4]:
        axes[0].plot(steps, [float(row[field]) for row in rows], label=field)
    axes[1].plot(steps, [float(row["beta_new"]) for row in rows], label="beta")
    axes[0].legend()
    axes[1].legend()
    if any(row.get("data_source") in {"mock", "synthetic"} for row in rows):
        figure.text(0.5, 0.5, "MOCK / SYNTHETIC — NOT AN EXPERIMENT", ha="center", alpha=0.25)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)


if __name__ == "__main__":
    main()
