import pytest

from src.safety.thresholds import SafetyState, SafetyThresholds, ThresholdConfig


def test_thresholds_scale_with_budget() -> None:
    narrow = SafetyThresholds.from_baseline(100, ThresholdConfig(0.02))
    wide = SafetyThresholds.from_baseline(100, ThresholdConfig(0.05))
    assert narrow.safe > narrow.warn > narrow.hard
    assert wide.safe > wide.warn > wide.hard
    assert wide.hard < narrow.hard
    assert narrow.classify(narrow.hard) is SafetyState.HARD
    assert narrow.classify(narrow.warn) is SafetyState.WARNING
    assert narrow.classify(narrow.safe) is SafetyState.SAFE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_degradation_ratio": 0},
        {"max_degradation_ratio": 1},
        {"max_degradation_ratio": 0.02, "safe_ratio": 0.9, "warn_ratio": 0.8},
    ],
)
def test_invalid_threshold_config(kwargs) -> None:
    with pytest.raises(ValueError):
        ThresholdConfig(**kwargs)


def test_baseline_must_be_positive() -> None:
    with pytest.raises(ValueError):
        SafetyThresholds.from_baseline(0, ThresholdConfig(0.02))
