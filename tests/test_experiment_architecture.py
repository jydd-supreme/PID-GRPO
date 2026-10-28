import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.data.experiment_data import load_experiment_data
from src.evaluation.online_data import materialize_roles
from src.utils.config import load_config, validate_config
from src.utils.experiment import experiment_plan, runtime_config


def test_three_experiments_share_model_data_budget_and_100_update_protocol():
    configs = [load_config(f"configs/experiments/deepmath_{mode}.yaml") for mode in ("pid", "fixed", "no_kl")]
    for section in ("training", "evaluation", "data", "safety"):
        assert configs[0][section] == configs[1][section] == configs[2][section]
    for config in configs:
        assert config["training"]["model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"
        assert config["control"]["eval_interval_optimizer_steps"] == 100
        assert config["training"]["reward_backend"] == "math_verify"
        plan = experiment_plan(runtime_config(config))
        assert plan["questions_per_update"] == 8
        assert plan["completions_per_update"] == 32
    assert [config["control"]["beta"] for config in configs] == [.04, .04, 0]


def test_seed_and_server_path_changes_leave_frozen_dataset_identity_unchanged(tmp_path):
    original = load_config("configs/experiments/deepmath_pid.yaml")
    remapped = runtime_config(original, seed=43, root_dir=str(tmp_path))
    assert original["project"]["seed"] == 42
    assert remapped["project"]["seed"] == 43
    assert Path(remapped["project"]["output_dir"]).is_relative_to(tmp_path)
    assert remapped["project"]["output_dir"].endswith("seed-43")
    assert experiment_plan(remapped)["suite_sha256"] == experiment_plan(original)["suite_sha256"]


def test_deepmath_cannot_silently_use_legacy_numeric_reward():
    config = load_config("configs/experiments/deepmath_pid.yaml")
    config["training"]["reward_backend"] = "numeric_exact_match"
    with pytest.raises(ValueError, match="math_verify"):
        validate_config(config)


def test_online_data_enters_training_without_reference_solution_leakage(tmp_path):
    suite = {"version": 1, "seed": 42, "aggregation": "sample_weighted", "protocol": "pid-grpo-zero-shot-v1",
             "sources": {
                 "deepmath": {"path": "fake/math", "kind": "math", "selection": "buffered", "buffer_size": 2,
                              "revision": "a" * 40, "allocations": {"train": 4, "domain_validation": 2}},
                 "mmlu": {"path": "fake/mmlu", "kind": "mmlu", "selection": "reservoir",
                          "revision": "b" * 40, "allocations": {"control": 3, "forgetting": 3}},
                 "final": {"path": "do-not-read", "kind": "math", "selection": "reservoir",
                           "revision": "c" * 40, "allocations": {"final_math": None}}}}
    def loader(source):
        assert source["path"] != "do-not-read"
        if source["kind"] == "math":
            return [{"question": f"math question {i}", "final_answer": str(10000+i),
                     "r1_solution_1": "SECRET SOLUTION"} for i in range(15)]
        return [{"question": f"mmlu question {i}", "subject": "science", "choices": ["yes", "no"],
                 "answer": 0} for i in range(15)]
    _, manifest = materialize_roles(suite, ["train", "control", "domain_validation"], row_loader=loader)
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite, sort_keys=False))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    config = load_config("configs/experiments/deepmath_pid.yaml")
    config["evaluation"].update(suite_config="suite.yaml", manifest_file="manifest.json")
    config["storage"]["cache_dir"] = str(tmp_path / "cache")
    prepared = load_experiment_data(config, repository=tmp_path, row_loader=loader)
    assert len(prepared.splits["train_file"]) == 4
    assert "final_test_file" not in prepared.splits
    for row in prepared.splits["train_file"]:
        assert "SECRET" not in str(row)
        assert row["answer"] not in str(row["prompt"])
        assert row["prompt"][0]["role"] == "user"
    # Configuration corruption fails before a network read.
    suite["seed"] = 99
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite, sort_keys=False))
    with pytest.raises(ValueError, match="configuration differs"):
        load_experiment_data(config, repository=tmp_path,
                             row_loader=lambda _: pytest.fail("must not request network"))


def test_pid_beta_changes_at_100_updates_and_not_at_microstep_boundaries(tmp_path):
    from src.controllers.pid_controller import PIDConfig, PIDController
    from src.evaluation.capability_evaluator import MockCapabilityEvaluator
    from src.safety.checkpoint_manager import CheckpointManager
    from src.safety.safety_guard import SafetyGuard
    from src.safety.thresholds import SafetyThresholds, ThresholdConfig
    from src.trainers.pid_grpo_trainer import DynamicBetaAdapter, PIDGRPOControlLoop

    thresholds = SafetyThresholds.from_baseline(1., ThresholdConfig(.1))
    trainer = SimpleNamespace(beta=.04, args=SimpleNamespace(beta=.04))
    pid = PIDController(PIDConfig(kp=.2, ki=0, kd=0), thresholds, initial_beta=.04)
    loop = PIDGRPOControlLoop(evaluator=MockCapabilityEvaluator([.91]), control_dataset=None,
                              guard=SafetyGuard(thresholds), beta_adapter=DynamicBetaAdapter(trainer, "pid_kl", .04),
                              checkpoint_manager=CheckpointManager(tmp_path),
                              eval_interval_optimizer_steps=100, pid=pid)
    for _ in range(99):
        assert not loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={}).evaluated
        assert trainer.beta == .04
    event = loop.after_optimizer_step(model=None, trainer_state={}, optimizer_state={})
    assert event.evaluated and trainer.beta > .04


def test_benchmark_history_preserves_step_role_and_subscores(tmp_path):
    torch = pytest.importorskip("torch")
    from src.evaluation.benchmark_evaluator import OnlineBenchmarkEvaluator
    scorer = SimpleNamespace(prepare=lambda _: None, score=lambda model, row: {"score": 1.})
    history = tmp_path / "benchmarks.jsonl"
    evaluator = OnlineBenchmarkEvaluator(scorer, history_path=history, step_provider=lambda: 100)
    evaluator.evaluate(torch.nn.Linear(1, 1), [{"id": "x", "task": "mmlu", "role": "control"}])
    saved = json.loads(history.read_text())
    assert saved["optimizer_step"] == 100 and saved["role"] == "control"
    assert saved["subscores"]["mmlu"] == 1


def test_callback_retains_trl_training_metrics_at_their_optimizer_step(tmp_path):
    from src.trainers.hf_control_callback import _PIDControlCallbackImpl
    loop = SimpleNamespace(beta_adapter=SimpleNamespace(beta=.04))
    callback = _PIDControlCallbackImpl(trainer=None, loop=loop, state_io=None,
                                      baseline_score=.6, metrics_path=tmp_path / "training.csv")
    logs = {"kl": .002, "reward_std": .4, "reward": .25, "completion_length": 1400}
    callback.on_log(None, SimpleNamespace(global_step=100), None, logs=logs)
    saved = json.loads((tmp_path / "grpo.jsonl").read_text())
    assert saved["optimizer_step"] == 100
    assert saved["metrics"] == logs
