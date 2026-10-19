"""Transformers callback binding optimizer updates to the PID control plane."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.safety.hf_checkpoint import HFTrainingStateIO
from src.trainers.pid_grpo_trainer import ControlEvent, PIDGRPOControlLoop
from src.utils.metrics_logger import CSVMetricLogger


class PIDControlCallback:
    """Callback implementation loaded lazily to keep controller imports light."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from transformers import TrainerCallback

        class _Callback(TrainerCallback):
            def __init__(self, *inner_args: Any, **inner_kwargs: Any) -> None:
                self._impl = _PIDControlCallbackImpl(*inner_args, **inner_kwargs)

            def on_train_begin(self, args, state, control, **callback_kwargs):
                return self._impl.on_train_begin(args, state, control, **callback_kwargs)

            def on_step_end(self, args, state, control, **callback_kwargs):
                return self._impl.on_step_end(args, state, control, **callback_kwargs)

        return _Callback(*args, **kwargs)


class _PIDControlCallbackImpl:
    def __init__(
        self,
        *,
        trainer: Any,
        loop: PIDGRPOControlLoop,
        state_io: HFTrainingStateIO,
        baseline_score: float,
        domain_evaluator: Any | None = None,
        domain_dataset: Any | None = None,
        metrics_path: str | Path | None = None,
        data_source: str = "real",
    ) -> None:
        self.trainer = trainer
        self.loop = loop
        self.state_io = state_io
        self.baseline_score = float(baseline_score)
        self.domain_evaluator = domain_evaluator
        self.domain_dataset = domain_dataset
        self.metrics = CSVMetricLogger(metrics_path) if metrics_path else None
        self.data_source = data_source
        self.last_global_step = 0

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        if not self.loop.checkpoint_manager.records:
            self.loop.save_initial_checkpoint(self.baseline_score)
        self.last_global_step = int(state.global_step)
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        global_step = int(state.global_step)
        if global_step <= self.last_global_step:
            return control
        # Transformers increments global_step only after optimizer.step(), so
        # one callback event corresponds to one optimizer update even with
        # gradient accumulation.
        self.last_global_step = global_step
        domain_score = None
        if (
            global_step % self.loop.interval == 0
            and self.domain_evaluator is not None
            and self.domain_dataset is not None
        ):
            try:
                domain_result = self.domain_evaluator.evaluate(
                    self.trainer.model, self.domain_dataset
                )
            except Exception:
                domain_result = {"valid": False}
            if domain_result.get("valid") is True:
                domain_score = float(domain_result["overall_score"])
        event = self.loop.after_optimizer_step(
            model=self.trainer.model,
            trainer_state={"global_step": global_step},
            optimizer_state={"external_hf_state": True},
            domain_validation_score=domain_score,
        )
        if event.evaluated:
            self._log_event(event)
        if event.stop_requested:
            control.should_training_stop = True
        return control

    def _log_event(self, event: ControlEvent) -> None:
        if self.metrics is None:
            return
        guard = event.guard or {}
        pid = event.pid or {}
        thresholds = self.loop.guard.thresholds
        raw = guard.get("raw_score")
        self.metrics.log(
            event.optimizer_step,
            {
                "control_step": self.loop.control_step,
                "general_capability_raw": raw,
                "general_capability_ema": pid.get("capability_ema"),
                "initial_capability": thresholds.initial_capability,
                "degradation_ratio": (
                    thresholds.degradation_ratio(raw) if isinstance(raw, (int, float)) else ""
                ),
                "safe_threshold": thresholds.safe,
                "warn_threshold": thresholds.warn,
                "hard_threshold": thresholds.hard,
                "capability_normalization_scale": pid.get("normalization_scale"),
                "normalized_error": pid.get("error"),
                "risk_weight": pid.get("risk_weight"),
                "safety_state": guard.get("state"),
                "p_term": pid.get("p_term"),
                "i_term": pid.get("i_term"),
                "d_term": pid.get("d_term"),
                "control_output": pid.get("control_output"),
                "beta_old": pid.get("beta_old", event.beta),
                "beta_new": event.beta,
                "rollback_count": self.loop.rollback_count,
                "data_source": self.data_source,
            },
        )
