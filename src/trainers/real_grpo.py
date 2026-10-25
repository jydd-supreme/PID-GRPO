"""Pinned TRL 0.15.2 single-GPU GRPO integration."""

from __future__ import annotations

from importlib.metadata import version
import json
import math
from pathlib import Path
from typing import Any, Mapping

from src.controllers.pid_controller import PIDConfig, PIDController
from src.data.experiment_data import load_experiment_data
from src.evaluation.capability_evaluator import RealCapabilityEvaluator
from src.rewards.math_reward import compute_reward
from src.rewards.verified_math_reward import select_reward
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


def _record_metadata(record: Any, beta: float) -> dict[str, Any]:
    return {
        "checkpoint_id": record.checkpoint_id,
        "optimizer_step": record.optimizer_step,
        "capability_score": record.capability_score,
        "domain_validation_score": record.domain_validation_score,
        "beta": beta,
        "checkpoint_path": record.path,
    }


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
    online_suite = evaluation.get("backend") == "online_suite"
    reward_func = select_reward(training.get("reward_backend", "numeric_exact_match"), math_accuracy_reward)
    prepared_data = load_experiment_data(config)
    splits = prepared_data.splits
    manifest = prepared_data.manifest
    output_dir = Path(project["output_dir"])
    if (output_dir / "checkpoints" / "manifest.json").exists():
        raise RuntimeError("output directory already contains a run; choose a new output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(dict(config), indent=2), encoding="utf-8")
    if manifest is not None:
        (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

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
        reward_funcs=reward_func,
        args=args,
        train_dataset=Dataset.from_list(splits["train_file"]),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if online_suite:
        from src.evaluation.benchmark_evaluator import HFBenchmarkScorer, OnlineBenchmarkEvaluator

        scorer = HFBenchmarkScorer(
            tokenizer, max_new_tokens=int(evaluation["max_new_tokens"]),
            max_context_length=int(evaluation["max_context_length"]),
        )
        scorer.prepare(splits["control_file"] + splits["domain_validation_file"])
        evaluator = OnlineBenchmarkEvaluator(
            scorer, history_path=output_dir / "metrics" / "benchmarks.jsonl",
            step_provider=lambda: int(trainer.state.global_step),
        )
    else:
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
    (output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
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
    recovery = safety["recovery"]
    initial_interval = loop.interval
    attempts = 0
    resume_path: Path | None = None
    max_steps = int(training["max_steps"])
    while True:
        if resume_path is None:
            trainer.train()
        else:
            trainer.train(resume_from_checkpoint=str(resume_path))
        rollback = manager.rollback_request
        if rollback is None:
            if loop.stop_requested:
                raise RuntimeError("training stopped for safety and rollback is disabled")
            if int(trainer.state.global_step) != max_steps or loop.optimizer_step != max_steps:
                raise RuntimeError("training ended before the requested optimizer-step budget")
            break

        if rollback.target_checkpoint_id is None:
            return {"status": "stopped_hard", "reason": "no eligible rollback checkpoint"}
        failed_beta = loop.beta_adapter.beta
        rollback_count = loop.rollback_count
        candidate_id = rollback.target_checkpoint_id
        recovered = None
        record = None
        while candidate_id is not None:
            record = next(r for r in manager.records if r.checkpoint_id == candidate_id)
            state_io.restore(record.path)
            loop.restore_protocol_state(record.checkpoint_id)
            try:
                recovered = evaluator.evaluate(trainer.model, control_dataset)
            except Exception as exc:
                recovered = {"valid": False, "error": f"{type(exc).__name__}: {exc}"}
            decision = guard.evaluate(recovered)
            if decision.allow_training and float(recovered["overall_score"]) >= thresholds.warn:
                break
            earlier_ids = {
                checkpoint_id for checkpoint_id in loop.active_checkpoint_ids
                if next(r for r in manager.records if r.checkpoint_id == checkpoint_id).optimizer_step
                < record.optimizer_step
            }
            earlier = manager.latest_rollback_candidate(thresholds.warn, earlier_ids)
            candidate_id = earlier.checkpoint_id if earlier is not None else None
        if candidate_id is None or record is None:
            return {"status": "stopped_hard", "reason": "rollback re-evaluation failed"}
        manager.clear_rollback_request()
        if attempts >= int(recovery["max_retries"]):
            return {
                "status": "stopped_hard",
                "reason": "recovery retry limit reached",
                "restored_checkpoint_id": record.checkpoint_id,
                "control": recovered,
            }
        if mode is not KLMode.PID_KL:
            return {
                "status": "stopped_hard",
                "reason": "automatic continuation requires pid_kl",
                "restored_checkpoint_id": record.checkpoint_id,
            }
        attempts += 1
        loop.begin_recovery(
            failed_beta * float(recovery["beta_backoff"]),
            rollback_count, attempts, int(recovery["hold_evaluations"]),
        )
        new_interval = max(1, math.ceil(
            initial_interval * float(recovery["interval_backoff"]) ** attempts
        ))
        callback.configure_recovery(
            lr_multiplier=float(recovery["lr_backoff"]) ** attempts,
            interval=new_interval,
        )
        resume_path = state_io.resume_path(record.path)

    active_ids = set(loop.active_checkpoint_ids)
    last = next(
        (
            record for record in manager.records
            if record.checkpoint_id in active_ids and record.optimizer_step == max_steps
            and record.safety_verified
        ),
        None,
    )
    if last is None:
        raise RuntimeError("final optimizer step has no verified capability checkpoint")
    best = manager.best_feasible_among(active_ids)
    if best is None:
        raise RuntimeError("training completed without a valid domain-validation checkpoint")

    # The live model is the last optimizer step. Export it before selecting
    # best-feasible; the old path overwrote the final weights by restoring best.
    last_beta = loop.beta_adapter.beta
    last_dir = output_dir / "last_model"
    trainer.save_model(str(last_dir))
    tokenizer.save_pretrained(str(last_dir))
    state_io.restore(best.path)
    loop.restore_protocol_state(best.checkpoint_id)
    if online_suite:
        final_result = {"status": "not_evaluated", "reason": "run online_benchmarks.py evaluate --allow-final after tuning is frozen"}
    else:
        final_result = evaluator.evaluate(trainer.model, splits["final_test_file"])
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    summary = {
        "status": "complete",
        "attempts": attempts,
        "best_feasible": _record_metadata(best, loop.beta_adapter.beta),
        "last_step": _record_metadata(last, last_beta),
        "final_test_best": final_result,
        "best_model_path": str(best_dir),
        "last_model_path": str(last_dir),
    }
    if online_suite:
        summary["evaluation_suite_sha256"] = manifest["suite_sha256"]
        summary["evaluation_protocol"] = manifest["protocol"]
    metadata_path = output_dir / "model_selection.json"
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(metadata_path)
    return {
        **summary,
        "selection_metadata": str(metadata_path),
    }
