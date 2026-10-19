"""Raw-measurement safety guard, evaluated before the PID controller."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .thresholds import SafetyState, SafetyThresholds


@dataclass(frozen=True)
class GuardDecision:
    allow_training: bool
    hard_violation: bool
    rollback_required: bool
    state: str
    reason: str
    raw_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafetyGuard:
    def __init__(
        self,
        thresholds: SafetyThresholds,
        *,
        hard_stop_enabled: bool = True,
        rollback_enabled: bool = True,
        min_eval_samples: int = 1,
    ) -> None:
        if min_eval_samples < 1:
            raise ValueError("min_eval_samples must be >= 1")
        self.thresholds = thresholds
        self.hard_stop_enabled = hard_stop_enabled
        self.rollback_enabled = rollback_enabled
        self.min_eval_samples = min_eval_samples
        self.last_decision: GuardDecision | None = None

    def evaluate(self, result: Mapping[str, Any]) -> GuardDecision:
        raw = result.get("overall_score")
        valid = result.get("valid") is True
        samples = result.get("num_samples", 0)
        invalid = (
            not valid
            or not isinstance(samples, int)
            or samples < self.min_eval_samples
            or not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
        )
        if invalid:
            decision = GuardDecision(
                allow_training=False,
                hard_violation=False,
                rollback_required=self.rollback_enabled,
                state=SafetyState.INVALID.value,
                reason="invalid, non-finite, or undersized capability evaluation",
                raw_score=float(raw) if isinstance(raw, (int, float)) else None,
            )
        else:
            score = float(raw)
            state = self.thresholds.classify(score)
            hard = state is SafetyState.HARD
            decision = GuardDecision(
                allow_training=not (hard and self.hard_stop_enabled),
                hard_violation=hard,
                rollback_required=hard and self.rollback_enabled,
                state=state.value,
                reason="raw score reached hard threshold" if hard else "evaluation accepted",
                raw_score=score,
            )
        self.last_decision = decision
        return decision

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "hard_stop_enabled": self.hard_stop_enabled,
            "rollback_enabled": self.rollback_enabled,
            "min_eval_samples": self.min_eval_samples,
            "last_decision": self.last_decision.to_dict() if self.last_decision else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported safety guard state version")
        if state.get("hard_stop_enabled") != self.hard_stop_enabled:
            raise ValueError("hard_stop_enabled mismatch")
        if state.get("rollback_enabled") != self.rollback_enabled:
            raise ValueError("rollback_enabled mismatch")
        if state.get("min_eval_samples") != self.min_eval_samples:
            raise ValueError("min_eval_samples mismatch")
        raw = state.get("last_decision")
        self.last_decision = GuardDecision(**raw) if raw else None
