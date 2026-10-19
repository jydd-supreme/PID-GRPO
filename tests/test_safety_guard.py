import pytest

from src.safety.safety_guard import SafetyGuard
from src.safety.thresholds import SafetyThresholds, ThresholdConfig


def make_guard() -> SafetyGuard:
    thresholds = SafetyThresholds.from_baseline(100, ThresholdConfig(0.02))
    return SafetyGuard(thresholds, min_eval_samples=2)


def test_hard_raw_measurement_stops_before_ema_can_hide_it() -> None:
    decision = make_guard().evaluate(
        {"overall_score": 98.0, "valid": True, "num_samples": 10}
    )
    assert decision.hard_violation
    assert not decision.allow_training
    assert decision.rollback_required


@pytest.mark.parametrize(
    "result",
    [
        {"overall_score": float("nan"), "valid": True, "num_samples": 10},
        {"overall_score": 100, "valid": False, "num_samples": 10},
        {"overall_score": 100, "valid": True, "num_samples": 1},
    ],
)
def test_invalid_evaluation_is_not_safe(result) -> None:
    decision = make_guard().evaluate(result)
    assert not decision.allow_training
    assert decision.state == "INVALID"
