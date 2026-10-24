"""TRL-facing beta adapter and framework-independent control-loop coordinator.

No training is started by this module. The adapter targets the current TRL design
where GRPOTrainer copies ``args.beta`` to ``self.beta`` and its loss reads
``self.beta``. Runtime compatibility is checked explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from src.controllers.pid_controller import PIDController, PIDUpdate
from src.evaluation.capability_evaluator import CapabilityEvaluator
from src.safety.checkpoint_manager import CheckpointManager
from src.safety.safety_guard import GuardDecision, SafetyGuard


class KLMode(str, Enum):
    NO_KL = "no_kl"
    FIXED_KL = "fixed_kl"
    PID_KL = "pid_kl"


class DynamicBetaAdapter:
    """Owns KL-mode invariants and updates the loss-consumed trainer attribute."""

    def __init__(self, trainer: Any, mode: KLMode | str, configured_beta: float) -> None:
        self.trainer = trainer
        self.mode = KLMode(mode)
        self.configured_beta = float(configured_beta)
        if not hasattr(trainer, "beta"):
            raise RuntimeError(
                "TRL adapter expected trainer.beta; inspect the installed TRL version before training"
            )
        runtime_beta = float(trainer.beta)
        if self.mode is KLMode.NO_KL:
            if self.configured_beta != 0.0 or runtime_beta != 0.0:
                raise ValueError("no_kl requires configured and trainer beta to be zero")
        elif self.configured_beta <= 0.0 or runtime_beta <= 0.0:
            raise ValueError(
                "fixed_kl/pid_kl must start with beta > 0 so TRL initializes KL reference state"
            )
        if self.mode is KLMode.FIXED_KL and runtime_beta != self.configured_beta:
            raise ValueError("fixed_kl trainer beta differs from configured beta")

    @property
    def beta(self) -> float:
        return float(self.trainer.beta)

    def set_beta(self, beta: float) -> float:
        beta = float(beta)
        if self.mode is KLMode.NO_KL:
            if beta != 0.0:
                raise ValueError("no_kl beta is immutable at zero")
            return 0.0
        if self.mode is KLMode.FIXED_KL:
            if beta != self.configured_beta:
                raise ValueError("fixed_kl beta cannot change")
            return self.beta
        if beta <= 0.0:
            raise ValueError("pid_kl beta must remain > 0")
        self.trainer.beta = beta
        # Mirror for readable config/checkpoints only; the current loss reads trainer.beta.
        if hasattr(self.trainer, "args") and hasattr(self.trainer.args, "beta"):
            self.trainer.args.beta = beta
        return self.beta

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "mode": self.mode.value,
            "configured_beta": self.configured_beta,
            "beta": self.beta,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1 or state.get("mode") != self.mode.value:
            raise ValueError("incompatible beta adapter state")
        if float(state["configured_beta"]) != self.configured_beta:
            raise ValueError("configured beta mismatch")
        self.set_beta(float(state["beta"]))


@dataclass(frozen=True)
class ControlEvent:
    optimizer_step: int
    evaluated: bool
    stop_requested: bool
    guard: dict[str, Any] | None
    pid: dict[str, Any] | None
    beta: float
    checkpoint_id: str | None
    rollback_target: str | None


class PIDGRPOControlLoop:
    """Coordinates discrete evaluation after actual optimizer updates.

    The caller must invoke :meth:`after_optimizer_step` exactly once after a
    successful optimizer.step(), never once per gradient-accumulation microbatch.
    """

    def __init__(
        self,
        *,
        evaluator: CapabilityEvaluator,
        control_dataset: Any,
        guard: SafetyGuard,
        beta_adapter: DynamicBetaAdapter,
        checkpoint_manager: CheckpointManager,
        eval_interval_optimizer_steps: int,
        pid: PIDController | None = None,
        checkpoint_artifact_writer: Callable[[Path], None] | None = None,
    ) -> None:
        if eval_interval_optimizer_steps < 1:
            raise ValueError("evaluation interval must be >= 1 optimizer update")
        if beta_adapter.mode is KLMode.PID_KL and pid is None:
            raise ValueError("pid_kl mode requires a PIDController")
        if beta_adapter.mode is not KLMode.PID_KL and pid is not None:
            raise ValueError("PIDController is only valid in pid_kl mode")
        self.evaluator = evaluator
        self.control_dataset = control_dataset
        self.guard = guard
        self.beta_adapter = beta_adapter
        self.checkpoint_manager = checkpoint_manager
        self.interval = eval_interval_optimizer_steps
        self.pid = pid
        self.checkpoint_artifact_writer = checkpoint_artifact_writer
        self.optimizer_step = 0
        self.control_step = 0
        self.stop_requested = False
        self.rollback_count = 0
        self.attempt = 0
        self.active_checkpoint_ids: list[str] = []

    def after_optimizer_step(
        self,
        *,
        model: Any,
        trainer_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
        domain_validation_score: float | None = None,
        force_evaluate: bool = False,
    ) -> ControlEvent:
        if self.stop_requested:
            raise RuntimeError("control loop is stopped; restore or create a new trajectory")
        self.optimizer_step += 1
        if not force_evaluate and self.optimizer_step % self.interval:
            return ControlEvent(
                self.optimizer_step, False, False, None, None,
                self.beta_adapter.beta, None, None,
            )
        try:
            result = self.evaluator.evaluate(model, self.control_dataset)
        except Exception as exc:
            result = {
                "overall_score": None,
                "subscores": {},
                "num_samples": 0,
                "valid": False,
                "source": "evaluator_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        self.control_step += 1
        decision = self.guard.evaluate(result)
        if not decision.allow_training:
            self.stop_requested = True
            rollback_target = None
            if decision.rollback_required:
                request = self.checkpoint_manager.request_rollback(
                    decision.reason,
                    minimum_score=self.guard.thresholds.warn,
                    checkpoint_ids=set(self.active_checkpoint_ids),
                )
                rollback_target = request.target_checkpoint_id
                self.rollback_count += 1
            return self._event(decision, None, None, rollback_target)

        pid_update: PIDUpdate | None = None
        if self.beta_adapter.mode is KLMode.PID_KL:
            assert self.pid is not None
            pid_update = self.pid.update(float(result["overall_score"]))
            self.beta_adapter.set_beta(pid_update.beta_new)

        # A disabled hard stop may permit the trajectory to continue, but a raw
        # HARD observation must never be certified as a safe rollback target.
        if decision.hard_violation:
            return self._event(decision, pid_update, None, None)

        checkpoint_id = (
            f"attempt-{self.attempt:02d}-step-{self.optimizer_step:08d}"
            f"-control-{self.control_step:05d}"
        )
        self.active_checkpoint_ids.append(checkpoint_id)
        try:
            self.checkpoint_manager.save(
                checkpoint_id,
                optimizer_step=self.optimizer_step,
                control_step=self.control_step,
                capability_score=float(result["overall_score"]),
                domain_validation_score=domain_validation_score,
                safety_verified=True,
                pid_state=self.pid.state_dict() if self.pid else {"version": 1, "mode": "disabled"},
                safety_state=self.guard.state_dict(),
                trainer_state=dict(trainer_state),
                optimizer_state=dict(optimizer_state),
                metadata={
                    "evaluation": dict(result),
                    "beta_adapter": self.beta_adapter.state_dict(),
                    "control_loop_state": self.state_dict(),
                    "evaluator_state": (
                        self.evaluator.state_dict() if hasattr(self.evaluator, "state_dict") else None
                    ),
                },
                artifact_writer=self.checkpoint_artifact_writer,
            )
        except BaseException:
            self.active_checkpoint_ids.pop()
            raise
        return self._event(decision, pid_update, checkpoint_id, None)

    def save_initial_checkpoint(self, capability_score: float) -> str:
        """Persist the pre-training model as the first safe rollback target."""
        checkpoint_id = "attempt-00-step-00000000-control-00000"
        self.active_checkpoint_ids.append(checkpoint_id)
        try:
            self.checkpoint_manager.save(
                checkpoint_id,
                optimizer_step=0,
                control_step=0,
                capability_score=float(capability_score),
                domain_validation_score=None,
                safety_verified=True,
                pid_state=self.pid.state_dict() if self.pid else {"version": 1, "mode": "disabled"},
                safety_state=self.guard.state_dict(),
                trainer_state={"external_hf_state": True},
                optimizer_state={"external_hf_state": True},
                metadata={
                    "evaluation": {"overall_score": float(capability_score), "source": "baseline"},
                    "beta_adapter": self.beta_adapter.state_dict(),
                    "control_loop_state": self.state_dict(),
                    "evaluator_state": (
                        self.evaluator.state_dict() if hasattr(self.evaluator, "state_dict") else None
                    ),
                },
                artifact_writer=self.checkpoint_artifact_writer,
            )
        except BaseException:
            self.active_checkpoint_ids.pop()
            raise
        return checkpoint_id

    def restore_protocol_state(self, checkpoint_id: str) -> None:
        payload = self.checkpoint_manager.read_payload(checkpoint_id)
        if self.pid is not None:
            self.pid.load_state_dict(payload["pid_state"])
        self.guard.load_state_dict(payload["safety_state"])
        metadata = payload["metadata"]
        self.load_state_dict(metadata["control_loop_state"])
        evaluator_state = metadata.get("evaluator_state")
        if evaluator_state is not None and hasattr(self.evaluator, "load_state_dict"):
            self.evaluator.load_state_dict(evaluator_state)
        self.stop_requested = False

    def begin_recovery(
        self, failed_beta: float, rollback_count: int, attempt: int,
        hold_evaluations: int,
    ) -> None:
        """Fork from a verified checkpoint with stronger temporary KL."""
        if self.pid is None:
            raise RuntimeError("automatic recovery is supported only for pid_kl")
        if attempt <= self.attempt:
            raise ValueError("recovery attempt must advance")
        self.attempt = attempt
        self.rollback_count = rollback_count
        self.pid.begin_recovery(failed_beta, hold_evaluations)
        self.beta_adapter.set_beta(self.pid.beta)
        self.stop_requested = False

    def _event(
        self,
        decision: GuardDecision,
        update: PIDUpdate | None,
        checkpoint_id: str | None,
        rollback_target: str | None,
    ) -> ControlEvent:
        return ControlEvent(
            optimizer_step=self.optimizer_step,
            evaluated=True,
            stop_requested=self.stop_requested,
            guard=decision.to_dict(),
            pid=asdict(update) if update else None,
            beta=self.beta_adapter.beta,
            checkpoint_id=checkpoint_id,
            rollback_target=rollback_target,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "optimizer_step": self.optimizer_step,
            "control_step": self.control_step,
            "stop_requested": self.stop_requested,
            "rollback_count": self.rollback_count,
            "attempt": self.attempt,
            "active_checkpoint_ids": list(self.active_checkpoint_ids),
            "beta_adapter": self.beta_adapter.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported control-loop state")
        self.optimizer_step = int(state["optimizer_step"])
        self.control_step = int(state["control_step"])
        self.stop_requested = bool(state["stop_requested"])
        self.rollback_count = int(state["rollback_count"])
        self.attempt = int(state.get("attempt", 0))
        self.active_checkpoint_ids = list(state.get("active_checkpoint_ids", []))
        self.beta_adapter.load_state_dict(state["beta_adapter"])
