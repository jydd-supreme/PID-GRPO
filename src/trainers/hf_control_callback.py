"""Transformers callback binding optimizer updates to the PID control plane."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from src.safety.hf_checkpoint import HFTrainingStateIO
from src.trainers.pid_grpo_trainer import ControlEvent, PIDGRPOControlLoop
from src.utils.metrics_logger import CSVMetricLogger
from src.utils.run_session import SessionPaused, TrainingBudgetComplete, atomic_json


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

            def on_log(self, args, state, control, logs=None, **callback_kwargs):
                return self._impl.on_log(args, state, control, logs=logs, **callback_kwargs)

            def on_pre_optimizer_step(self, args, state, control, **callback_kwargs):
                return self._impl.on_pre_optimizer_step(args, state, control, **callback_kwargs)

            def configure_recovery(self, *, lr_multiplier: float, interval: int) -> None:
                self._impl.configure_recovery(lr_multiplier=lr_multiplier, interval=interval)

            def load_restart_state(self, state, rng_path=None) -> None:
                self._impl.load_restart_state(state, rng_path=rng_path)

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
        restart_store: Any = None,
        session: Any = None,
        restart_identity: dict[str, Any] | None = None,
        checkpoint_steps: int = 25,
        checkpoint_seconds: float = 600,
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
        self.lr_multiplier = 1.0
        self.effective_learning_rate: float | None = None
        self.restart_store = restart_store
        self.session = session
        self.restart_identity = restart_identity or {}
        self.checkpoint_steps = checkpoint_steps
        self.checkpoint_seconds = checkpoint_seconds
        self.last_snapshot_step = 0
        self.last_snapshot_time = session.clock() if session else 0
        self.pending_optimizer_step: int | None = None
        self.resume_rng_path = None

    def load_restart_state(self, state: dict[str, Any], rng_path=None) -> None:
        self.pending_optimizer_step = state["pending_optimizer_step"]
        self.lr_multiplier = float(state["lr_multiplier"])
        self.effective_learning_rate = state.get("effective_learning_rate")
        self.last_global_step = int(state["global_step"])
        self.last_snapshot_step = self.last_global_step
        self.resume_rng_path = rng_path

    def _snapshot(self, *, force: bool = False) -> None:
        if self.restart_store is None:
            return
        step = int(self.trainer.state.global_step)
        now = self.session.clock()
        if not force and step - self.last_snapshot_step < self.checkpoint_steps and now - self.last_snapshot_time < self.checkpoint_seconds:
            return
        state = {
            **self.restart_identity, "global_step": step,
            "pending_optimizer_step": self.pending_optimizer_step,
            "lr_multiplier": self.lr_multiplier, "effective_learning_rate": self.effective_learning_rate,
            "pid_state": self.loop.pid.state_dict() if self.loop.pid else None,
            "guard_state": self.loop.guard.state_dict(), "loop_state": self.loop.state_dict(),
        }
        path = self.restart_store.save(state, self.state_io.save)
        self.last_snapshot_step = step
        self.last_snapshot_time = self.session.clock()
        self.session.event("checkpoint", global_step=step, checkpoint=str(path),
                           pending_evaluation=self.pending_optimizer_step is not None)
        atomic_json(self.session.output_dir / "progress.json", {
            "status": "running", "session_id": self.session.session_id,
            "global_step": step, "control_step": self.loop.control_step,
            "beta": self.loop.beta_adapter.beta, "checkpoint": str(path),
            "pending_evaluation": self.pending_optimizer_step is not None,
        })

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **_: Any) -> Any:
        if self.metrics is not None and logs:
            path = self.metrics.path.with_name("grpo.jsonl")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"optimizer_step": int(state.global_step),
                                         "session_id": self.session.session_id if self.session else None,
                                         "beta_for_next_update": self.loop.beta_adapter.beta,
                                         "metrics": logs}) + "\n")
        return control

    def configure_recovery(self, *, lr_multiplier: float, interval: int) -> None:
        if not 0.0 < lr_multiplier <= 1.0 or interval < 1:
            raise ValueError("invalid recovery learning-rate multiplier or interval")
        self.lr_multiplier = lr_multiplier
        self.loop.interval = interval

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        optimizer = self.trainer.optimizer
        if optimizer is None:
            raise RuntimeError("optimizer is unavailable at the update boundary")
        # Trainer's scheduler writes the unscaled scheduled LR after each step.
        # Apply the recovery factor immediately before optimizer.step(), so it
        # does not compound across steps or get lost when the scheduler advances.
        for group in optimizer.param_groups:
            group["lr"] *= self.lr_multiplier
        self.effective_learning_rate = optimizer.param_groups[0]["lr"]
        return control

    def on_train_begin(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        if self.resume_rng_path is not None:
            # Trainer restores RNG only after on_train_begin. Restore it now as
            # well, before pending evaluation or publishing another snapshot.
            self.trainer._load_rng_state(str(self.resume_rng_path))
            self.resume_rng_path = None
        if not self.loop.checkpoint_manager.records:
            self.loop.save_initial_checkpoint(self.baseline_score)
        self.last_global_step = int(state.global_step)
        if self.pending_optimizer_step is not None:
            if self.pending_optimizer_step != self.last_global_step or self.loop.optimizer_step != self.last_global_step - 1:
                raise ValueError("pending checkpoint optimizer counters disagree")
            self._finish_step(args, state, control)
        elif self.loop.optimizer_step != self.last_global_step:
            raise ValueError("restored trainer and controller optimizer counters disagree")
        if self.restart_store is not None:
            self._snapshot(force=True)
        if self.last_global_step >= int(args.max_steps):
            raise TrainingBudgetComplete()
        if self.session:
            self.session.check()
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        global_step = int(state.global_step)
        if global_step <= self.last_global_step:
            return control
        # Transformers increments global_step only after optimizer.step(), so
        # one callback event corresponds to one optimizer update even with
        # gradient accumulation.
        self.last_global_step = global_step
        self.pending_optimizer_step = global_step
        # Publish optimizer state BEFORE a potentially long benchmark. A restart
        # must finish this boundary without taking another optimizer step.
        if global_step % self.loop.interval == 0 or global_step >= int(args.max_steps):
            self._snapshot(force=True)
        return self._finish_step(args, state, control)

    def _finish_step(self, args: Any, state: Any, control: Any) -> Any:
        global_step = int(state.global_step)
        expected_id = (f"attempt-{self.loop.attempt:02d}-step-{global_step:08d}"
                       f"-control-{self.loop.control_step + 1:05d}")
        # A kill may occur after a certified checkpoint was published but before
        # the post-evaluation restart snapshot. Do not apply the PID twice.
        if any(record.checkpoint_id == expected_id for record in self.loop.checkpoint_manager.records):
            self.loop.restore_protocol_state(expected_id)
            self.pending_optimizer_step = None
            if self.session:
                self.session.event("reconciled_evaluation", global_step=global_step)
            self._snapshot(force=True)
            return control
        domain_score = None
        if (
            (global_step % self.loop.interval == 0 or global_step >= int(args.max_steps))
            and self.domain_evaluator is not None
            and self.domain_dataset is not None
        ):
            try:
                domain_result = self.domain_evaluator.evaluate(
                    self.trainer.model, self.domain_dataset
                )
            except SessionPaused:
                raise
            except Exception:
                domain_result = {"valid": False}
            if domain_result.get("valid") is True:
                domain_score = float(domain_result["overall_score"])
        event = self.loop.after_optimizer_step(
            model=self.trainer.model,
            trainer_state={"global_step": global_step},
            optimizer_state={"external_hf_state": True},
            domain_validation_score=domain_score,
            force_evaluate=global_step >= int(args.max_steps),
        )
        if event.evaluated:
            self._log_event(event)
        if event.stop_requested:
            control.should_training_stop = True
        self.pending_optimizer_step = None
        if not event.stop_requested:
            should_pause = self.session is not None and self.session.expired()
            self._snapshot(force=event.evaluated or should_pause)
            if should_pause and global_step < int(args.max_steps):
                self.session.check()
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
                "learning_rate": self.effective_learning_rate,
                "data_source": self.data_source,
            },
        )
