"""Risk-aware PID controller for multiplicative KL coefficient updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from src.safety.risk_gate import PiecewiseLinearRiskGate
from src.safety.thresholds import SafetyThresholds


@dataclass(frozen=True)
class PIDConfig:
    kp: float = 0.2
    ki: float = 0.05
    kd: float = 0.2
    beta_min: float = 0.001
    beta_max: float = 1.0
    max_beta_multiplier: float = 2.0
    ema_alpha: float = 0.4
    derivative_alpha: float = 0.5
    deadband: float = 0.0
    integral_min: float = 0.0
    integral_max: float = 0.10
    safe_integral_leak: float = 0.5
    safe_release_rate: float = 0.25
    warmup_control_steps: int = 1

    def __post_init__(self) -> None:
        if min(self.kp, self.ki, self.kd) < 0:
            raise ValueError("PID gains must be non-negative")
        if not 0 < self.beta_min <= self.beta_max:
            raise ValueError("beta bounds must satisfy 0 < beta_min <= beta_max")
        if self.max_beta_multiplier < 1.0:
            raise ValueError("max_beta_multiplier must be >= 1")
        if not 0 < self.ema_alpha <= 1 or not 0 < self.derivative_alpha <= 1:
            raise ValueError("EMA alphas must be in (0, 1]")
        if self.deadband < 0 or self.integral_min > self.integral_max:
            raise ValueError("invalid deadband or integral bounds")
        if not 0 <= self.safe_integral_leak <= 1:
            raise ValueError("safe_integral_leak must be in [0, 1]")
        if not 0 <= self.safe_release_rate < 1:
            raise ValueError("safe_release_rate must be in [0, 1)")
        if self.warmup_control_steps < 0:
            raise ValueError("warmup_control_steps must be >= 0")


@dataclass(frozen=True)
class PIDUpdate:
    beta_old: float
    beta_new: float
    capability_raw: float
    capability_ema: float
    normalization_scale: float
    error: float
    risk_weight: float
    p_term: float
    i_term: float
    d_term: float
    control_output: float
    safety_state: str
    warmup: bool
    released_kl: bool


class PIDController:
    """Track Warn with signed P, conditional I and an early trend brake."""

    def __init__(
        self,
        config: PIDConfig,
        thresholds: SafetyThresholds,
        initial_beta: float,
        base_beta: float | None = None,
        risk_gate: PiecewiseLinearRiskGate | None = None,
    ) -> None:
        if initial_beta <= 0:
            raise ValueError("PID mode requires initial_beta > 0")
        self.config = config
        self.thresholds = thresholds
        self.base_beta = initial_beta if base_beta is None else base_beta
        if not config.beta_min <= self.base_beta <= config.beta_max:
            raise ValueError("base_beta must be inside beta bounds")
        self.beta = min(max(initial_beta, config.beta_min), config.beta_max)
        self.risk_gate = risk_gate or PiecewiseLinearRiskGate()
        self.integral_error = 0.0
        self.capability_ema: float | None = None
        self.previous_ema: float | None = None
        self.filtered_derivative = 0.0
        self.control_step = 0
        self.recovery_beta_floor: float | None = None
        self.recovery_safe_streak = 0
        self.recovery_hold_evaluations = 0

    def begin_recovery(self, failed_beta: float, hold_evaluations: int = 2) -> None:
        """Keep stronger KL temporarily after a verified rollback."""
        if hold_evaluations < 1 or not math.isfinite(failed_beta) or failed_beta <= 0:
            raise ValueError("invalid recovery settings")
        self.recovery_beta_floor = min(
            self.config.beta_max,
            self.beta * self.config.max_beta_multiplier,
            max(self.beta, failed_beta),
        )
        self.beta = max(self.beta, self.recovery_beta_floor)
        self.integral_error = 0.0
        self.previous_ema = None
        self.capability_ema = None
        self.filtered_derivative = 0.0
        self.recovery_safe_streak = 0
        self.recovery_hold_evaluations = hold_evaluations

    def update(self, capability: float) -> PIDUpdate:
        if not math.isfinite(capability):
            raise ValueError("PID cannot update from an invalid capability measurement")
        cfg = self.config
        old_beta = self.beta
        previous = self.capability_ema
        ema = capability if previous is None else cfg.ema_alpha * capability + (1 - cfg.ema_alpha) * previous
        self.previous_ema = previous
        self.capability_ema = ema
        # Express error and trend in units of the WARNING-to-HARD interval. This
        # makes PID gains, the deadband, and the integral bounds independent of
        # whether a benchmark reports capability on a 0-1 or 0-100 scale.
        normalization_scale = self.thresholds.warn - self.thresholds.hard
        normalized_error = (self.thresholds.warn - ema) / normalization_scale
        error = 0.0 if abs(normalized_error) <= cfg.deadband else normalized_error
        # PID regions use the filtered signal; the independent guard already
        # enforces hard safety on the unfiltered raw measurement.
        state = self.thresholds.classify(ema)

        old_integral = self.integral_error
        self.control_step += 1
        warmup = self.control_step <= cfg.warmup_control_steps
        if error <= 0.0:
            self.integral_error *= cfg.safe_integral_leak
        elif not warmup and self.beta < cfg.beta_max:
            candidate = self.integral_error + error
            self.integral_error = min(max(candidate, cfg.integral_min), cfg.integral_max)

        derivative = (
            0.0
            if previous is None
            else (previous - ema) / normalization_scale
        )
        self.filtered_derivative = (
            cfg.derivative_alpha * derivative
            + (1 - cfg.derivative_alpha) * self.filtered_derivative
        )
        p_term = cfg.kp * error
        i_term = cfg.ki * self.integral_error
        d_term = cfg.kd * self.filtered_derivative
        risk = self.risk_gate(ema, self.thresholds)
        # Above Warn, negative P releases KL so training can use the allowed
        # degradation budget. A falling trend can override that release.
        output = risk * (max(p_term, 0.0) + i_term) + (1.0 - risk) * min(p_term, 0.0) + d_term

        # Warm-up freezes I/D but never suppresses a first P safety action.
        if warmup:
            output -= risk * i_term + d_term
        log_rate_limit = math.log(cfg.max_beta_multiplier)
        # The release limit is a per-evaluation fraction, independent of the
        # positive emergency rate limit.
        log_release_limit = -math.log1p(-cfg.safe_release_rate)
        rate_limited_output = min(max(output, -min(log_rate_limit, log_release_limit)), log_rate_limit)
        new_beta = min(max(old_beta * math.exp(rate_limited_output), cfg.beta_min), cfg.beta_max)
        if new_beta >= cfg.beta_max and self.integral_error > old_integral:
            self.integral_error = old_integral
            i_term = cfg.ki * self.integral_error
            output = risk * (max(p_term, 0.0) + i_term) + (1.0 - risk) * min(p_term, 0.0) + d_term
            if warmup:
                output -= risk * i_term + d_term
            rate_limited_output = min(
                max(output, -min(log_rate_limit, log_release_limit)), log_rate_limit
            )
            new_beta = min(
                max(old_beta * math.exp(rate_limited_output), cfg.beta_min), cfg.beta_max
            )
        if self.recovery_beta_floor is not None:
            new_beta = max(new_beta, self.recovery_beta_floor)
            if capability >= self.thresholds.warn:
                self.recovery_safe_streak += 1
                if self.recovery_safe_streak >= self.recovery_hold_evaluations:
                    self.recovery_beta_floor = None
            else:
                self.recovery_safe_streak = 0
        released = new_beta < old_beta
        self.beta = new_beta
        return PIDUpdate(
            beta_old=old_beta,
            beta_new=new_beta,
            capability_raw=capability,
            capability_ema=ema,
            normalization_scale=normalization_scale,
            error=error,
            risk_weight=risk,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            control_output=output,
            safety_state=state.value,
            warmup=warmup,
            released_kl=released,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            # Version 3 adds signed P and recovery state. Older controller
            # states are intentionally rejected rather than silently migrated.
            "version": 3,
            "config": asdict(self.config),
            "beta": self.beta,
            "base_beta": self.base_beta,
            "integral_error": self.integral_error,
            "capability_ema": self.capability_ema,
            "previous_ema": self.previous_ema,
            "filtered_derivative": self.filtered_derivative,
            "control_step": self.control_step,
            "recovery_beta_floor": self.recovery_beta_floor,
            "recovery_safe_streak": self.recovery_safe_streak,
            "recovery_hold_evaluations": self.recovery_hold_evaluations,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 3:
            raise ValueError("unsupported PID state version")
        if state.get("config") != asdict(self.config):
            raise ValueError("PID state config does not match controller config")
        beta = float(state["beta"])
        if not self.config.beta_min <= beta <= self.config.beta_max:
            raise ValueError("restored beta is outside configured bounds")
        self.beta = beta
        self.base_beta = float(state["base_beta"])
        self.integral_error = float(state["integral_error"])
        self.capability_ema = state["capability_ema"]
        self.previous_ema = state["previous_ema"]
        self.filtered_derivative = float(state["filtered_derivative"])
        self.control_step = int(state["control_step"])
        self.recovery_beta_floor = state["recovery_beta_floor"]
        self.recovery_safe_streak = int(state["recovery_safe_streak"])
        self.recovery_hold_evaluations = int(state["recovery_hold_evaluations"])
