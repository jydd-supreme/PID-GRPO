"""Capability thresholds derived from an initial baseline measurement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math


class SafetyState(str, Enum):
    SAFE = "SAFE"
    BUFFER = "BUFFER"
    WARNING = "WARNING"
    HARD = "HARD"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ThresholdConfig:
    max_degradation_ratio: float
    safe_ratio: float = 0.50
    warn_ratio: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 < self.max_degradation_ratio < 1.0:
            raise ValueError("max_degradation_ratio must be in (0, 1)")
        if not 0.0 < self.safe_ratio < self.warn_ratio < 1.0:
            raise ValueError("ratios must satisfy 0 < safe_ratio < warn_ratio < 1")


@dataclass(frozen=True)
class SafetyThresholds:
    initial_capability: float
    safe: float
    warn: float
    hard: float
    config: ThresholdConfig

    @classmethod
    def from_baseline(
        cls, initial_capability: float, config: ThresholdConfig
    ) -> "SafetyThresholds":
        if not math.isfinite(initial_capability) or initial_capability <= 0.0:
            raise ValueError("initial_capability must be finite and > 0")
        d_max = config.max_degradation_ratio
        return cls(
            initial_capability=initial_capability,
            safe=(1.0 - config.safe_ratio * d_max) * initial_capability,
            warn=(1.0 - config.warn_ratio * d_max) * initial_capability,
            hard=(1.0 - d_max) * initial_capability,
            config=config,
        )

    def classify(self, score: float) -> SafetyState:
        if not math.isfinite(score):
            return SafetyState.INVALID
        if score >= self.safe:
            return SafetyState.SAFE
        if score > self.warn:
            return SafetyState.BUFFER
        if score > self.hard:
            return SafetyState.WARNING
        return SafetyState.HARD

    def degradation_ratio(self, score: float) -> float:
        if not math.isfinite(score):
            return math.nan
        return (self.initial_capability - score) / self.initial_capability

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        return result
