import pytest

from src.controllers.pid_controller import PIDConfig, PIDController
from src.safety.thresholds import SafetyThresholds, ThresholdConfig


def make_controller(*, baseline=1.0, **overrides) -> PIDController:
    values = dict(
        kp=1.0, ki=1.0, kd=1.0, beta_min=0.01, beta_max=0.5,
        max_beta_multiplier=1.5, ema_alpha=1.0, derivative_alpha=1.0,
        integral_max=0.02, safe_integral_leak=0.0, safe_release_rate=0.5,
        warmup_control_steps=0,
    )
    values.update(overrides)
    thresholds = SafetyThresholds.from_baseline(baseline, ThresholdConfig(0.1))
    return PIDController(PIDConfig(**values), thresholds, initial_beta=0.1)


def test_decline_raises_beta_and_derivative_is_positive() -> None:
    controller = make_controller()
    controller.update(0.96)
    update = controller.update(0.93)
    assert update.d_term > 0
    assert update.beta_new > update.beta_old


def test_integral_saturates_and_leaks_in_safe_region() -> None:
    controller = make_controller(kd=0)
    for _ in range(10):
        controller.update(0.91)
    assert controller.integral_error == pytest.approx(0.02)
    controller.update(1.0)
    assert controller.integral_error == 0


def test_bounds_and_rate_limit() -> None:
    controller = make_controller(kp=100, ki=0, kd=0)
    update = controller.update(0.91)
    assert update.beta_new <= update.beta_old * 1.5
    for _ in range(10):
        controller.update(0.91)
    assert controller.beta <= 0.5


def test_warmup_preserves_immediate_p_action_and_releases_kl_above_warn() -> None:
    controller = make_controller(warmup_control_steps=1, kp=10, kd=0)
    warmup = controller.update(0.91)
    assert warmup.beta_new > warmup.beta_old
    controller.update(0.91)
    raised = controller.beta
    released = controller.update(1.0)
    assert raised > controller.base_beta
    assert controller.beta < raised
    assert released.released_kl


def test_buffer_p_releases_kl_and_falling_d_can_brake() -> None:
    controller = make_controller(kp=0.2, ki=0.5, kd=0, safe_release_rate=0.25)
    controller.update(0.99)
    update = controller.update(0.94)
    assert update.safety_state == "BUFFER"
    assert update.p_term < 0
    assert update.i_term == 0
    assert update.beta_new < update.beta_old
    predictive = make_controller(kp=0.2, ki=0, kd=2.0)
    predictive.update(0.99)
    brake = predictive.update(0.94)
    assert brake.safety_state == "BUFFER"
    assert brake.d_term > 0
    assert brake.beta_new > brake.beta_old


def test_integral_only_accumulates_below_warn_and_not_at_beta_max() -> None:
    controller = make_controller(kp=0, kd=0, ki=1.0)
    controller.update(0.94)
    assert controller.integral_error == 0
    controller.update(0.91)
    assert controller.integral_error > 0
    controller.beta = controller.config.beta_max
    previous = controller.integral_error
    controller.update(0.91)
    assert controller.integral_error == previous


def test_state_round_trip() -> None:
    first = make_controller()
    first.update(0.94)
    state = first.state_dict()
    second = make_controller()
    second.load_state_dict(state)
    assert second.state_dict() == state


def test_legacy_raw_unit_state_is_rejected() -> None:
    controller = make_controller()
    state = controller.state_dict()
    state["version"] = 1
    with pytest.raises(ValueError, match="unsupported PID state version"):
        controller.load_state_dict(state)


def test_invalid_measurement_rejected() -> None:
    with pytest.raises(ValueError):
        make_controller().update(float("nan"))


def test_pid_terms_are_invariant_to_capability_score_scale() -> None:
    unit_scale = make_controller(kp=0.2, ki=0.1, kd=0.3)
    percent_scale = make_controller(baseline=100.0, kp=0.2, ki=0.1, kd=0.3)

    unit_scale.update(0.96)
    percent_scale.update(96.0)
    unit_update = unit_scale.update(0.93)
    percent_update = percent_scale.update(93.0)

    assert unit_update.normalization_scale == pytest.approx(0.02)
    assert percent_update.normalization_scale == pytest.approx(2.0)
    for field in ("error", "p_term", "i_term", "d_term", "control_output", "beta_new"):
        assert getattr(unit_update, field) == pytest.approx(getattr(percent_update, field))


def test_error_uses_warning_to_hard_interval_as_one_unit() -> None:
    controller = make_controller(kp=0, ki=0, kd=0)
    update = controller.update(controller.thresholds.hard)
    assert update.error == pytest.approx(1.0)


def test_recovery_holds_kl_floor_until_warn_is_repeatedly_verified() -> None:
    controller = make_controller(kp=0.2, ki=0.1, kd=0, safe_release_rate=0.25)
    controller.begin_recovery(failed_beta=0.14, hold_evaluations=2)
    floor = controller.recovery_beta_floor
    assert floor == pytest.approx(0.14)
    controller.update(0.94)
    assert controller.beta >= floor
    assert controller.recovery_beta_floor == floor
    controller.update(0.91)
    assert controller.recovery_safe_streak == 0
    controller.update(0.94)
    assert controller.recovery_beta_floor == floor
    controller.update(0.94)
    assert controller.recovery_beta_floor is None
    assert controller.update(0.94).beta_new < floor
