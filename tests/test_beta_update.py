from pathlib import Path
from types import SimpleNamespace

import pytest

from src.controllers.pid_controller import PIDConfig, PIDController
from src.evaluation.capability_evaluator import MockCapabilityEvaluator
from src.safety.checkpoint_manager import CheckpointManager
from src.safety.safety_guard import SafetyGuard
from src.safety.thresholds import SafetyThresholds, ThresholdConfig
from src.trainers.hf_control_callback import _PIDControlCallbackImpl
from src.trainers.pid_grpo_trainer import (
    DynamicBetaAdapter, KLMode, PIDGRPOControlLoop,
)


class Trainer:
    def __init__(self, beta):
        self.beta = beta
        self.args = SimpleNamespace(beta=beta)


def components(tmp_path: Path, scores, interval=2, *, rollback_enabled=True, hard_stop_enabled=True):
    thresholds = SafetyThresholds.from_baseline(1.0, ThresholdConfig(0.1))
    trainer = Trainer(0.1)
    adapter = DynamicBetaAdapter(trainer, KLMode.PID_KL, 0.1)
    pid = PIDController(
        PIDConfig(kp=5, ki=0, kd=0, warmup_control_steps=0),
        thresholds, initial_beta=0.1,
    )
    loop = PIDGRPOControlLoop(
        evaluator=MockCapabilityEvaluator(scores), control_dataset=None,
        guard=SafetyGuard(
            thresholds,
            rollback_enabled=rollback_enabled,
            hard_stop_enabled=hard_stop_enabled,
        ), beta_adapter=adapter,
        checkpoint_manager=CheckpointManager(tmp_path),
        eval_interval_optimizer_steps=interval, pid=pid,
    )
    return trainer, loop


def test_modes_enforce_beta_invariants() -> None:
    no_kl = Trainer(0.0)
    adapter = DynamicBetaAdapter(no_kl, "no_kl", 0.0)
    assert adapter.set_beta(0.0) == 0.0
    with pytest.raises(ValueError):
        adapter.set_beta(0.1)
    fixed = DynamicBetaAdapter(Trainer(0.1), "fixed_kl", 0.1)
    with pytest.raises(ValueError):
        fixed.set_beta(0.2)


def test_interval_counts_optimizer_updates_and_pid_is_dynamic(tmp_path: Path) -> None:
    trainer, loop = components(tmp_path, [0.91])
    first = loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={})
    second = loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={})
    assert not first.evaluated
    assert second.evaluated
    assert trainer.beta > 0.1


def test_hard_guard_stops_before_pid_and_requests_safe_rollback(tmp_path: Path) -> None:
    trainer, loop = components(tmp_path, [0.99, 0.89], interval=1)
    safe = loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={})
    beta_before = trainer.beta
    hard = loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={})
    assert safe.checkpoint_id is not None
    assert hard.stop_requested
    assert hard.rollback_target == safe.checkpoint_id
    assert trainer.beta == beta_before


def test_beta_adapter_state_restores() -> None:
    trainer = Trainer(0.1)
    adapter = DynamicBetaAdapter(trainer, "pid_kl", 0.1)
    state = adapter.state_dict()
    adapter.set_beta(0.2)
    adapter.load_state_dict(state)
    assert trainer.beta == 0.1


def test_rollback_disabled_stops_without_request(tmp_path: Path) -> None:
    _, loop = components(tmp_path, [0.89], interval=1, rollback_enabled=False)
    event = loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={})
    assert event.stop_requested
    assert event.rollback_target is None
    assert loop.checkpoint_manager.rollback_request is None


def test_disabled_hard_stop_never_certifies_hard_checkpoint(tmp_path: Path) -> None:
    _, loop = components(tmp_path, [0.89], interval=1, hard_stop_enabled=False)
    event = loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={})
    assert not event.stop_requested
    assert event.guard["hard_violation"]
    assert event.checkpoint_id is None
    assert loop.checkpoint_manager.latest_safe is None


def test_evaluator_exception_becomes_invalid_stop(tmp_path: Path) -> None:
    class BrokenEvaluator:
        def evaluate(self, model, dataset):
            raise TimeoutError("benchmark timed out")

    _, loop = components(tmp_path, [0.99], interval=1)
    loop.evaluator = BrokenEvaluator()
    event = loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={})
    assert event.stop_requested
    assert event.guard["state"] == "INVALID"


def test_forced_last_step_evaluation_and_recovery_lineage(tmp_path: Path) -> None:
    trainer, loop = components(tmp_path, [0.99, 0.89, 0.98], interval=5)
    loop.save_initial_checkpoint(1.0)
    first = loop.after_optimizer_step(
        model=None, trainer_state={}, optimizer_state={}, force_evaluate=True,
        domain_validation_score=0.6,
    )
    assert first.checkpoint_id is not None
    hard = loop.after_optimizer_step(
        model=None, trainer_state={}, optimizer_state={}, force_evaluate=True,
    )
    assert hard.stop_requested
    assert hard.rollback_target == first.checkpoint_id
    failed_beta = trainer.beta
    loop.restore_protocol_state(first.checkpoint_id)
    loop.begin_recovery(failed_beta * 1.5, rollback_count=1, attempt=1, hold_evaluations=2)
    loop.evaluator = MockCapabilityEvaluator([0.98])
    assert trainer.beta >= failed_beta
    assert loop.active_checkpoint_ids[-1] == first.checkpoint_id
    continued = loop.after_optimizer_step(
        model=None, trainer_state={}, optimizer_state={}, force_evaluate=True,
        domain_validation_score=0.7,
    )
    assert continued.checkpoint_id.startswith("attempt-01-")
    assert loop.checkpoint_manager.best_feasible_among(
        set(loop.active_checkpoint_ids)
    ).checkpoint_id == continued.checkpoint_id


def test_recovery_learning_rate_backoff_is_applied_once_per_optimizer_step(tmp_path: Path) -> None:
    _, loop = components(tmp_path, [0.99], interval=5)
    trainer = SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 1e-5}]))
    callback = _PIDControlCallbackImpl(
        trainer=trainer, loop=loop, state_io=None, baseline_score=1.0,
    )
    callback.configure_recovery(lr_multiplier=0.5, interval=2)
    callback.on_pre_optimizer_step(None, None, None)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(5e-6)
    assert loop.interval == 2
    trainer.optimizer.param_groups[0]["lr"] = 8e-6  # Scheduler writes its next step.
    callback.on_pre_optimizer_step(None, None, None)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(4e-6)
