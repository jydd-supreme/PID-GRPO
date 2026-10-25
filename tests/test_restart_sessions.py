import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.controllers.pid_controller import PIDConfig, PIDController
from src.safety.checkpoint_manager import CheckpointManager
from src.safety.safety_guard import SafetyGuard
from src.safety.thresholds import SafetyThresholds, ThresholdConfig
from src.trainers.hf_control_callback import _PIDControlCallbackImpl
from src.trainers.pid_grpo_trainer import DynamicBetaAdapter, PIDGRPOControlLoop
from src.utils.run_session import (RestartCheckpointStore, RunSession, SessionPaused,
                                   TrainingBudgetComplete, restore_control_state)


def writer(path):
    (path / "weights.bin").write_bytes(b"complete weights")


def test_atomic_restart_store_falls_back_after_corruption_and_ignores_temporary_files(tmp_path):
    store = RestartCheckpointStore(tmp_path)
    first = store.save({"global_step": 1}, writer)
    second = store.save({"global_step": 2}, writer)
    (second / "weights.bin").write_bytes(b"truncated")
    (tmp_path / ".resume-999999.tmp").mkdir()
    path, state = store.latest()
    assert path == first and state["global_step"] == 1
    assert second.name in store.skipped


def test_failed_writer_does_not_publish_or_remove_previous_checkpoint(tmp_path):
    store = RestartCheckpointStore(tmp_path, keep_last=2)
    original = store.save({"global_step": 1}, writer)
    def broken(path):
        writer(path)
        raise RuntimeError("disk failed")
    with pytest.raises(RuntimeError):
        store.save({"global_step": 2}, broken)
    assert store.latest()[0] == original
    for step in (3, 4, 5):
        store.save({"global_step": step}, writer)
    assert len(list(tmp_path.glob("resume-*"))) == 2
    assert store.latest()[1]["global_step"] == 5


def test_protocol_checksum_and_partial_snapshot_cleanup(tmp_path):
    store = RestartCheckpointStore(tmp_path)
    first = store.save({"global_step": 1}, writer)
    second = store.save({"global_step": 2}, writer)
    state_path = second / "restart_state.json"
    state = json.loads(state_path.read_text())
    state["global_step"] = 999
    state_path.write_text(json.dumps(state))
    assert store.latest()[0] == first
    partial = tmp_path / ".resume-interrupted.tmp"
    partial.mkdir()
    (partial / "weights.bin").write_bytes(b"partial")
    assert store.discard_incomplete() == 1
    assert first.is_dir() and not partial.exists()


def test_safe_checkpoint_published_before_manifest_is_recovered(tmp_path):
    trainer, loop, _, callback = components(tmp_path)
    args = SimpleNamespace(max_steps=10)
    callback.on_train_begin(args, trainer.state, SimpleNamespace())
    manifest = loop.checkpoint_manager.root / "manifest.json"
    before = manifest.read_text()
    for step in (1, 2):
        trainer.state.global_step = step
        callback.on_step_end(args, trainer.state, SimpleNamespace())
    # Simulate a kill between atomic directory publication and manifest update.
    manifest.write_text(before)
    manager = CheckpointManager(loop.checkpoint_manager.root)
    manager.recover_interrupted_writes()
    assert {record.optimizer_step for record in manager.records} == {0, 2}


class ConstantEvaluator:
    def __init__(self):
        self.calls = 0
    def evaluate(self, model, dataset):
        self.calls += 1
        return {"valid": True, "overall_score": .99, "num_samples": 1000}


def components(root, *, session=None, evaluator=None):
    thresholds = SafetyThresholds.from_baseline(1., ThresholdConfig(.1))
    trainer = SimpleNamespace(beta=.04, args=SimpleNamespace(beta=.04),
                              state=SimpleNamespace(global_step=0), model=None)
    loop = PIDGRPOControlLoop(evaluator=evaluator or ConstantEvaluator(), control_dataset=[],
                              guard=SafetyGuard(thresholds),
                              beta_adapter=DynamicBetaAdapter(trainer, "pid_kl", .04),
                              checkpoint_manager=CheckpointManager(root / "safe"),
                              eval_interval_optimizer_steps=2,
                              pid=PIDController(PIDConfig(), thresholds, initial_beta=.04))
    store = RestartCheckpointStore(root / "restart")
    callback = _PIDControlCallbackImpl(trainer=trainer, loop=loop, state_io=SimpleNamespace(save=writer),
                                      baseline_score=1., restart_store=store,
                                      session=session or RunSession(root), checkpoint_steps=1)
    loop.checkpoint_artifact_writer = writer
    return trainer, loop, store, callback


def test_pending_evaluation_resumes_without_an_extra_update_or_pid_action(tmp_path):
    class PausingEvaluator(ConstantEvaluator):
        def evaluate(self, model, dataset):
            raise SessionPaused("session ended during control")
    trainer, loop, store, callback = components(tmp_path, evaluator=PausingEvaluator())
    args = SimpleNamespace(max_steps=4)
    callback.on_train_begin(args, trainer.state, SimpleNamespace())
    trainer.state.global_step = 1
    callback.on_step_end(args, trainer.state, SimpleNamespace())
    trainer.state.global_step = 2
    with pytest.raises(SessionPaused):
        callback.on_step_end(args, trainer.state, SimpleNamespace())
    _, state = store.latest()
    assert state["global_step"] == 2 and state["loop_state"]["optimizer_step"] == 1
    assert state["pending_optimizer_step"] == 2
    resumed, second_loop, _, second_callback = components(tmp_path)
    restore_control_state(second_loop, state)
    second_callback.load_restart_state(state)
    resumed.state.global_step = 2
    second_callback.on_train_begin(args, resumed.state, SimpleNamespace())
    assert second_loop.optimizer_step == 2
    assert second_loop.control_step == second_loop.pid.control_step == 1
    assert second_callback.pending_optimizer_step is None
    assert resumed.beta != .04


def test_completed_safe_evaluation_is_reconciled_instead_of_applying_pid_twice(tmp_path):
    trainer, loop, store, callback = components(tmp_path)
    args = SimpleNamespace(max_steps=2)
    callback.on_train_begin(args, trainer.state, SimpleNamespace())
    trainer.state.global_step = 1
    callback.on_step_end(args, trainer.state, SimpleNamespace())
    trainer.state.global_step = 2
    callback.pending_optimizer_step = 2
    callback._snapshot(force=True)
    _, pending = store.latest()
    loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={})
    expected_beta = trainer.beta
    resumed, new_loop, _, new_callback = components(tmp_path)
    restore_control_state(new_loop, pending)
    new_callback.load_restart_state(pending)
    resumed.state.global_step = 2
    with pytest.raises(TrainingBudgetComplete):
        new_callback.on_train_begin(args, resumed.state, SimpleNamespace())
    assert resumed.beta == expected_beta
    assert new_loop.pid.control_step == new_loop.control_step == 1
    assert new_loop.evaluator.calls == 0


def test_time_budget_pauses_at_a_complete_update_and_keeps_pid_state(tmp_path):
    elapsed = [0.]
    session = RunSession(tmp_path, max_seconds=10, clock=lambda: elapsed[0])
    trainer, loop, store, callback = components(tmp_path, session=session)
    args = SimpleNamespace(max_steps=10)
    callback.on_train_begin(args, trainer.state, SimpleNamespace())
    elapsed[0] = 11
    trainer.state.global_step = 1
    with pytest.raises(SessionPaused):
        callback.on_step_end(args, trainer.state, SimpleNamespace())
    _, state = store.latest()
    assert state["global_step"] == state["loop_state"]["optimizer_step"] == 1
    assert state["pending_optimizer_step"] is None
    assert state["pid_state"]["control_step"] == 0


def test_epoch_sampler_reconstructs_pinned_trl_permutation_sequence():
    torch = pytest.importorskip("torch")
    from src.data.resumable_sampler import ResumableRepeatSampler
    generator = torch.Generator().manual_seed(42)
    for epoch in range(4):
        expected = [i for i in torch.randperm(11, generator=generator).tolist() for _ in range(4)]
        sampler = ResumableRepeatSampler(list(range(11)), 4, 42)
        sampler.set_epoch(epoch)
        assert list(sampler) == expected


def test_interrupted_evaluation_reuses_rows_only_for_identical_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from src.evaluation.benchmark_evaluator import OnlineBenchmarkEvaluator
    model = torch.nn.Linear(1, 1)
    model.register_buffer("scalar", torch.tensor(1))
    session = RunSession(tmp_path)
    class Scorer:
        def __init__(self, interrupt=False):
            self.calls = 0
            self.interrupt = interrupt
        def prepare(self, rows):
            pass
        def score(self, model, row):
            self.calls += 1
            if self.interrupt and self.calls == 2:
                session.pause_reason = "shutdown"
            return {"score": 1.}
    rows = [{"id": str(i), "task": "mmlu"} for i in range(3)]
    first = Scorer(True)
    evaluator = OnlineBenchmarkEvaluator(first, progress_dir=tmp_path, progress_identity="same-run", session=session)
    with pytest.raises(SessionPaused):
        evaluator.evaluate(model, rows)
    assert model.training and first.calls == 2
    second = Scorer()
    resumed = OnlineBenchmarkEvaluator(second, progress_dir=tmp_path, progress_identity="same-run")
    assert resumed.evaluate(model, rows)["overall_score"] == 1
    assert second.calls == 1
    with torch.no_grad():
        model.weight.add_(1)
    resumed.evaluate(model, rows)
    assert second.calls == 4
