"""Stable CSV metric schema with an optional TensorBoard sink."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Protocol


METRIC_FIELDS = [
    "global_step", "control_step", "domain_reward", "domain_accuracy",
    "general_capability_raw", "general_capability_ema", "initial_capability",
    "degradation_ratio", "safe_threshold", "warn_threshold", "hard_threshold",
    "capability_normalization_scale", "normalized_error",
    "risk_weight", "safety_state", "p_term", "i_term", "d_term",
    "control_output", "beta_old", "beta_new", "measured_kl", "rollback_count",
    "best_feasible_score", "learning_rate", "data_source",
]


class MetricsSink(Protocol):
    def log(self, step: int, metrics: Mapping[str, Any]) -> None: ...


class CSVMetricLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writeheader()

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        unknown = set(metrics) - set(METRIC_FIELDS)
        if unknown:
            raise ValueError(f"unknown metric fields: {sorted(unknown)}")
        row = {field: metrics.get(field, "") for field in METRIC_FIELDS}
        row["global_step"] = step
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writerow(row)


class TensorBoardSink:
    """Optional extension; importing it never installs TensorBoard."""

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("TensorBoard sink requires torch and tensorboard") from exc
        self.writer = SummaryWriter(str(log_dir))

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.writer.add_scalar(name, value, step)
