"""Pinned TRL 0.15.2 single-GPU GRPO integration."""

from __future__ import annotations

from importlib.metadata import version
import fcntl
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
from src.utils.run_session import (RunSession, RestartCheckpointStore, SessionPaused,
                                   TrainingBudgetComplete, atomic_json, restore_control_state)
from src.evaluation.online_data import digest


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


def run_real_training(config: Mapping[str, Any], *, resume: bool = False,
                      session_max_seconds: float | None = None) -> dict[str, Any]:
    configure_runtime_storage(config)
    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = config.get("checkpointing", {})
    limit = settings.get("session_max_seconds", 0) if session_max_seconds is None else session_max_seconds
    session = RunSession(output_dir, max_seconds=float(limit))
    with (output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another training process owns this output directory") from exc
        session.install_handlers()
        try:
            session.event("resume_requested" if resume else "start_requested")
            result = _run_real_training(config, resume=resume, session=session)
            session.event(result["status"], global_step=result.get("global_step"))
            atomic_json(output_dir / "run_result.json", result)
            if result["status"] == "complete":
                atomic_json(output_dir / "progress.json", {"status": "complete", "session_id": session.session_id,
                                                           "global_step": result.get("global_step", config["training"]["max_steps"])})
            return result
        except SessionPaused as exc:
            latest = RestartCheckpointStore(output_dir / "restart_checkpoints").latest()
            result = {"status": "paused", "reason": str(exc), "session_id": session.session_id,
                      "global_step": latest[1]["global_step"] if latest else 0,
                      "checkpoint": str(latest[0]) if latest else None,
                      "pending_evaluation": latest[1]["pending_optimizer_step"] is not None if latest else False}
            atomic_json(output_dir / "run_result.json", result)
            atomic_json(output_dir / "progress.json", result)
            session.event("paused", **{key: value for key, value in result.items() if key not in {"status", "session_id"}})
            return result
        except Exception as exc:
            session.event("error", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            session.close()


def _run_real_training(config: Mapping[str, Any], *, resume: bool, session: RunSession) -> dict[str, Any]:
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
    run_config_path = output_dir / "run_config.json"
    if resume:
        if not run_config_path.is_file() or json.loads(run_config_path.read_text(encoding="utf-8")) != dict(config):
            raise ValueError("resume requires the identical saved run configuration, including seed and max_steps")
        result_path = output_dir / "run_result.json"
        if result_path.is_file():
            previous_result = json.loads(result_path.read_text(encoding="utf-8"))
            if previous_result.get("status") == "complete":
                return previous_result
            if previous_result.get("status") == "stopped_hard":
                raise RuntimeError("this run stopped for safety; --resume cannot bypass Hard/Invalid")
    else:
        if run_config_path.exists() or (output_dir / "checkpoints" / "manifest.json").exists():
            raise RuntimeError("output directory already contains a run; pass --resume to continue it")
        atomic_json(run_config_path, dict(config))
    data_signature = digest(manifest if manifest is not None else splits)
    if resume and (output_dir / "data_identity.json").is_file():
        if json.loads((output_dir / "data_identity.json").read_text())["sha256"] != data_signature:
            raise ValueError("training/evaluation data changed since the original run")
    atomic_json(output_dir / "data_identity.json", {"sha256": data_signature})
    restart_identity = {"run_signature": digest(config), "data_signature": data_signature}
    settings = config.get("checkpointing", {})
    restart_store = RestartCheckpointStore(output_dir / "restart_checkpoints",
                                            keep_last=int(settings.get("keep_last", 3)))
    if resume:
        discarded = restart_store.discard_incomplete()
        if discarded:
            session.event("discarded_incomplete_snapshots", count=discarded)
    restart = restart_store.latest() if resume else None
    if restart and any(restart[1].get(key) != value for key, value in restart_identity.items()):
        raise ValueError("restart checkpoint identity does not match this run/data")
    if manifest is not None:
        atomic_json(output_dir / "data_manifest.json", manifest)

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
    from src.data.resumable_sampler import ResumableRepeatSampler

    class RestartableGRPOTrainer(GRPOTrainer):
        def _get_train_sampler(self):
            return ResumableRepeatSampler(self.train_dataset, self.num_generations, self.args.seed)

    trainer = RestartableGRPOTrainer(
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
            progress_dir=output_dir / "evaluation_progress",
            progress_identity=restart_identity,
            session=session,
        )
    else:
        evaluator = RealCapabilityEvaluator(
            tokenizer,
            batch_size=int(evaluation["batch_size"]),
            max_new_tokens=int(evaluation["max_new_tokens"]),
        )
    control_dataset = splits["control_file"]
    baseline_path = output_dir / "baseline.json"
    baseline = (json.loads(baseline_path.read_text(encoding="utf-8"))
                if resume and baseline_path.is_file() else evaluator.evaluate(trainer.model, control_dataset))
    if (
        baseline.get("valid") is not True
        or int(baseline.get("num_samples", 0)) < int(safety["min_eval_samples"])
        or float(baseline["overall_score"]) <= 0
    ):
        raise RuntimeError(f"invalid initial control baseline: {baseline}")
    atomic_json(baseline_path, baseline)
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
    if resume:
        manager.recover_interrupted_writes()
    if manager.records and not restart:
        raise RuntimeError("no intact restart checkpoint; do not restart this trajectory from the base model")
    if manager.rollback_request is not None:
        raise RuntimeError("pending safety rollback must be resolved; --resume cannot bypass it")
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
        restart_store=restart_store,
        session=session,
        restart_identity=restart_identity,
        checkpoint_steps=int(settings.get("interval_optimizer_steps", 25)),
        checkpoint_seconds=float(settings.get("interval_seconds", 600)),
    )
    trainer.add_callback(callback)
    recovery = safety["recovery"]
    initial_interval = loop.interval
    resume_path: Path | None = None
    if restart:
        directory, restored = restart
        restore_control_state(loop, restored)
        resume_path = state_io.resume_path(directory)
        callback.load_restart_state(restored, rng_path=resume_path)
        session.event("restored", global_step=restored["global_step"],
                       checkpoint=str(directory), skipped_corrupt=restart_store.skipped,
                       pending_evaluation=restored["pending_optimizer_step"] is not None)
    attempts = loop.attempt
    max_steps = int(training["max_steps"])
    while True:
        try:
            if resume_path is None:
                trainer.train()
            else:
                trainer.train(resume_from_checkpoint=str(resume_path))
        except TrainingBudgetComplete:
            pass
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
        "global_step": max_steps,
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
