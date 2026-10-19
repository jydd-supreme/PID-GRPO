"""Pinned TRL 0.15.2 single-GPU GRPO integration."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

from src.controllers.pid_controller import PIDConfig, PIDController
from src.data.dataset_builder import load_local_jsonl
from src.evaluation.capability_evaluator import RealCapabilityEvaluator
from src.rewards.math_reward import compute_reward
from src.safety.checkpoint_manager import CheckpointManager
from src.safety.hf_checkpoint import HFTrainingStateIO
from src.safety.safety_guard import SafetyGuard
from src.safety.thresholds import SafetyThresholds, ThresholdConfig
from src.trainers.hf_control_callback import PIDControlCallback
from src.trainers.pid_grpo_trainer import (
    DynamicBetaAdapter,
    KLMode,
    PIDGRPOControlLoop,
)
from src.utils.seed import set_seed
from src.utils.storage import configure_runtime_storage


PINNED_VERSIONS = {
    "torch": "2.5.1",
    "transformers": "4.49.0",
    "trl": "0.15.2",
    "peft": "0.14.0",
    "datasets": "3.3.2",
    "accelerate": "1.4.0",
}


def verify_training_environment() -> None:
    import torch

    installed = {name: version(name) for name in PINNED_VERSIONS}
    for name, expected in PINNED_VERSIONS.items():
        actual = installed[name]
        if name == "torch":
            if actual.split("+")[0] != expected:
                raise RuntimeError(f"torch must be {expected}+cu124, found {actual}")
        elif actual != expected:
            raise RuntimeError(f"{name} must be {expected}, found {actual}")
    if torch.version.cuda != "12.4":
        raise RuntimeError(f"PyTorch must be built for CUDA 12.4, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("real training requires an available CUDA device")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size() != 1:
            raise RuntimeError("P0 integration is single-GPU only; distributed beta sync is not validated")


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, Mapping):
            return str(last.get("content", ""))
    return str(completion)


def math_accuracy_reward(completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
    return [
        compute_reward(_completion_text(completion), str(reference))
        for completion, reference in zip(completions, answer)
    ]


def _load_and_check_splits(data_config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    names = ("train_file", "control_file", "domain_validation_file", "final_test_file")
    splits = {name: load_local_jsonl(data_config[name]) for name in names}
    seen: dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            sample_id = str(row["id"])
            if sample_id in seen:
                raise ValueError(
                    f"dataset leakage: id {sample_id!r} occurs in {seen[sample_id]} and {split_name}"
                )
            seen[sample_id] = split_name
    return splits


def run_real_training(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run one safe single-GPU trajectory and return its terminal summary."""
    storage_paths = configure_runtime_storage(config)
    verify_training_environment()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    project = config["project"]
    training = config["training"]
    control = config["control"]
    safety = config["safety"]
    evaluation = config["evaluation"]
    set_seed(int(project["seed"]))
    splits = _load_and_check_splits(config["data"])

    model_name = str(training["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=str(training["model_revision"]),
        cache_dir=storage_paths["HF_HUB_CACHE"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    mode = KLMode(control["mode"])
    beta = float(control["beta"])
    output_dir = Path(project["output_dir"])
    run_output = output_dir / "trainer"
    args = GRPOConfig(
        output_dir=str(run_output),
        overwrite_output_dir=bool(training["overwrite_output_dir"]),
        max_steps=int(training["max_steps"]),
        num_train_epochs=float(training["num_train_epochs"]),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        logging_steps=int(training["logging_steps"]),
        save_strategy="no",
        report_to=[],
        bf16=bool(training["bf16"]),
        fp16=False,
        tf32=bool(training["tf32"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=int(project["seed"]),
        data_seed=int(project["seed"]),
        remove_unused_columns=False,
        max_prompt_length=int(training["max_prompt_length"]),
        max_completion_length=int(training["max_completion_length"]),
        num_generations=int(training["num_generations"]),
        temperature=float(training["temperature"]),
        beta=beta,
        model_init_kwargs={
            "revision": str(training["model_revision"]),
            "cache_dir": storage_paths["HF_HUB_CACHE"],
            "torch_dtype": torch.bfloat16 if training["bf16"] else torch.float32,
            "attn_implementation": "sdpa",
            "use_cache": False,
            "trust_remote_code": False,
        },
    )
    peft_config = None
    if training["use_lora"]:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(training["lora_r"]),
            lora_alpha=int(training["lora_alpha"]),
            lora_dropout=float(training["lora_dropout"]),
            target_modules=list(training["lora_target_modules"]),
        )
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=math_accuracy_reward,
        args=args,
        train_dataset=Dataset.from_list(splits["train_file"]),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    evaluator = RealCapabilityEvaluator(
        tokenizer,
        batch_size=int(evaluation["batch_size"]),
        max_new_tokens=int(evaluation["max_new_tokens"]),
    )
    control_dataset = splits["control_file"]
    baseline = evaluator.evaluate(trainer.model, control_dataset)
    if (
        baseline.get("valid") is not True
        or int(baseline.get("num_samples", 0)) < int(safety["min_eval_samples"])
        or float(baseline["overall_score"]) <= 0
    ):
        raise RuntimeError(f"invalid initial control baseline: {baseline}")
    thresholds = SafetyThresholds.from_baseline(
        float(baseline["overall_score"]),
        ThresholdConfig(
            max_degradation_ratio=float(safety["max_degradation_ratio"]),
            safe_ratio=float(safety["safe_ratio"]),
            warn_ratio=float(safety["warn_ratio"]),
        ),
    )
    guard = SafetyGuard(
        thresholds,
        hard_stop_enabled=bool(safety["hard_stop_enabled"]),
        rollback_enabled=bool(safety["rollback_enabled"]),
        min_eval_samples=int(safety["min_eval_samples"]),
    )
    adapter = DynamicBetaAdapter(trainer, mode, beta)
    pid = (
        PIDController(PIDConfig(**control["pid"]), thresholds, initial_beta=beta)
        if mode is KLMode.PID_KL
        else None
    )
    manager = CheckpointManager(output_dir / "checkpoints")
    if manager.records:
        raise RuntimeError(
            "checkpoint directory already contains a run; choose a new output_dir "
            "or implement an explicit reviewed resume"
        )
    loop = PIDGRPOControlLoop(
        evaluator=evaluator,
        control_dataset=control_dataset,
        guard=guard,
        beta_adapter=adapter,
        checkpoint_manager=manager,
        eval_interval_optimizer_steps=int(control["eval_interval_optimizer_steps"]),
        pid=pid,
    )
    state_io = HFTrainingStateIO(trainer, tokenizer)
    loop.checkpoint_artifact_writer = state_io.save
    callback = PIDControlCallback(
        trainer=trainer,
        loop=loop,
        state_io=state_io,
        baseline_score=float(baseline["overall_score"]),
        domain_evaluator=evaluator,
        domain_dataset=splits["domain_validation_file"],
        metrics_path=output_dir / "metrics" / "training.csv",
    )
    trainer.add_callback(callback)
    trainer.train()

    rollback = manager.rollback_request
    if loop.stop_requested and rollback is None:
        raise RuntimeError("training stopped for safety and rollback is disabled")
    if rollback is not None:
        if rollback.target_checkpoint_id is None:
            raise RuntimeError("safety stop requested but no safe checkpoint exists")
        record = next(r for r in manager.records if r.checkpoint_id == rollback.target_checkpoint_id)
        state_io.restore(record.path)
        loop.restore_protocol_state(record.checkpoint_id)
        manager.clear_rollback_request()
        recovered = evaluator.evaluate(trainer.model, control_dataset)
        if guard.evaluate(recovered).allow_training is not True:
            raise RuntimeError(f"rollback re-evaluation did not recover capability: {recovered}")
        return {
            "status": "rolled_back",
            "checkpoint_id": record.checkpoint_id,
            "control": recovered,
        }

    best = manager.best_feasible or manager.latest_safe
    if best is None:
        raise RuntimeError("training completed without a safety-verified checkpoint")
    state_io.restore(best.path)
    loop.restore_protocol_state(best.checkpoint_id)
    final_result = evaluator.evaluate(trainer.model, splits["final_test_file"])
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return {
        "status": "complete",
        "checkpoint_id": best.checkpoint_id,
        "final_test": final_result,
    }
