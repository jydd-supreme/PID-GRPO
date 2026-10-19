import pytest

from src.safety.risk_gate import PiecewiseLinearRiskGate
from src.safety.thresholds import SafetyThresholds, ThresholdConfig


def test_gate_is_continuous_bounded_and_monotone() -> None:
    thresholds = SafetyThresholds.from_baseline(100, ThresholdConfig(0.10))
    gate = PiecewiseLinearRiskGate(warn_weight=0.8)
    scores = [thresholds.safe, 93.5, thresholds.warn, 91, thresholds.hard]
    weights = [gate(score, thresholds) for score in scores]
    assert weights == sorted(weights)
    assert weights[0] == 0
    assert weights[2] == pytest.approx(0.8)
    assert weights[-1] == 1
    epsilon = 1e-8
    assert gate(thresholds.warn + epsilon, thresholds) == pytest.approx(
        gate(thresholds.warn - epsilon, thresholds), abs=1e-7
    )


def test_invalid_score_is_maximum_risk() -> None:
    thresholds = SafetyThresholds.from_baseline(1, ThresholdConfig(0.1))
    assert PiecewiseLinearRiskGate()(float("nan"), thresholds) == 1
