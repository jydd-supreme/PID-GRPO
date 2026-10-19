"""Continuous risk gating for PID proportional and integral action."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .thresholds import SafetyThresholds


@dataclass(frozen=True)
class PiecewiseLinearRiskGate:
    """Continuous monotone gate: 0 at SAFE, warn_weight at WARN, 1 at HARD."""

    warn_weight: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 < self.warn_weight < 1.0:
            raise ValueError("warn_weight must be in (0, 1)")

    def __call__(self, score: float, thresholds: SafetyThresholds) -> float:
        if not math.isfinite(score):
            return 1.0
        if score >= thresholds.safe:
            return 0.0
        if score > thresholds.warn:
            fraction = (thresholds.safe - score) / (thresholds.safe - thresholds.warn)
            return self.warn_weight * fraction
        if score > thresholds.hard:
            fraction = (thresholds.warn - score) / (thresholds.warn - thresholds.hard)
            return self.warn_weight + (1.0 - self.warn_weight) * fraction
        return 1.0
