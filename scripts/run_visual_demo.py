#!/usr/bin/env python3
"""Run a tiny real-TRL/Mock-capability demo and render audit figures."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/demo")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    repository = Path(__file__).resolve().parents[1]
    allowed = (repository / "outputs").resolve()
    if not root.is_relative_to(allowed):
        raise SystemExit(f"output must stay inside {allowed}")
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace this demo only")
        shutil.rmtree(root)
    root.mkdir(parents=True)

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        GPT2Config,
        GPT2LMHeadModel,
        PreTrainedTokenizerFast,
        TrainerCallback,
        set_seed,
    )
    from trl import GRPOConfig, GRPOTrainer

    from src.controllers.pid_controller import PIDConfig, PIDController
    from src.evaluation.capability_evaluator import MockCapabilityEvaluator
    from src.safety.checkpoint_manager import CheckpointManager
    from src.safety.hf_checkpoint import HFTrainingStateIO
    from src.safety.safety_guard import SafetyGuard
    from src.safety.thresholds import SafetyThresholds, ThresholdConfig
    from src.trainers.hf_control_callback import PIDControlCallback
    from src.trainers.pid_grpo_trainer import DynamicBetaAdapter, PIDGRPOControlLoop

    set_seed(42)
    backend = Tokenizer(WordLevel(
        {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "question": 3, "answer": 4},
        unk_token="[UNK]",
    ))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=5,
        n_layer=1,
        n_head=1,
        n_embd=16,
        n_positions=32,
        pad_token_id=0,
        eos_token_id=1,
    ))
    training_args = GRPOConfig(
        output_dir=str(root / "trainer"),
        max_steps=8,
        per_device_train_batch_size=2,
        num_generations=2,
        max_prompt_length=8,
        max_completion_length=3,
        learning_rate=5e-5,
        beta=0.04,
        report_to=[],
        save_strategy="no",
        logging_steps=1,
        use_cpu=True,
        seed=42,
    )

    def toy_reward(completions, **kwargs):
        # Deliberately synthetic: just creates non-constant reward groups for
        # exercising a real GRPO optimizer step without external data/models.
        return [float(sum(map(ord, str(text))) % 7) / 6.0 for text in completions]

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=toy_reward,
        train_dataset=Dataset.from_list([
            {"prompt": "question", "answer": "answer"},
            {"prompt": "question answer", "answer": "answer"},
            {"prompt": "answer question", "answer": "answer"},
            {"prompt": "question question", "answer": "answer"},
        ]),
        processing_class=tokenizer,
    )

    scores = [0.990, 0.965, 0.945, 0.925, 0.915, 0.930, 0.960, 0.990]
    thresholds = SafetyThresholds.from_baseline(1.0, ThresholdConfig(0.10))
    pid = PIDController(
        PIDConfig(
            kp=1.2, ki=0.1, kd=0.3, beta_min=0.001, beta_max=1.0,
            max_beta_multiplier=1.5, ema_alpha=0.7, derivative_alpha=0.7,
            integral_max=0.1, warmup_control_steps=1,
        ),
        thresholds,
        initial_beta=0.04,
    )
    manager = CheckpointManager(root / "checkpoints")
    loop = PIDGRPOControlLoop(
        evaluator=MockCapabilityEvaluator(scores),
        control_dataset=None,
        guard=SafetyGuard(thresholds),
        beta_adapter=DynamicBetaAdapter(trainer, "pid_kl", 0.04),
        checkpoint_manager=manager,
        eval_interval_optimizer_steps=1,
        pid=pid,
    )
    state_io = HFTrainingStateIO(trainer, tokenizer)
    loop.checkpoint_artifact_writer = state_io.save
    control_csv = root / "control_metrics.csv"
    trainer.add_callback(PIDControlCallback(
        trainer=trainer,
        loop=loop,
        state_io=state_io,
        baseline_score=1.0,
        metrics_path=control_csv,
        data_source="mock",
    ))

    training_logs: list[dict[str, float]] = []

    class LogCapture(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                training_logs.append({"step": int(state.global_step), **logs})

    trainer.add_callback(LogCapture())
    trainer.train()

    training_csv = root / "training_metrics.csv"
    fields = sorted({key for row in training_logs for key in row})
    with training_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(training_logs)

    with control_csv.open(encoding="utf-8") as handle:
        control_rows = list(csv.DictReader(handle))
    step_rows = [row for row in training_logs if "loss" in row]
    train_steps = [int(row["step"]) for row in step_rows]
    control_steps = [int(row["global_step"]) for row in control_rows]
    numeric = lambda name: [float(row[name]) for row in control_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(
        train_steps, [float(row["kl"]) for row in step_rows], marker="o", label="measured KL"
    )
    axes[0, 0].set(title="Actual tiny-GRPO optimization signals", xlabel="optimizer step", ylabel="KL")
    reward_axis = axes[0, 0].twinx()
    reward_axis.plot(
        train_steps, [float(row["reward"]) for row in step_rows],
        color="tab:orange", marker="s", alpha=0.75, label="reward",
    )
    reward_axis.set_ylabel("synthetic reward")
    lines = axes[0, 0].get_lines() + reward_axis.get_lines()
    axes[0, 0].legend(lines, [line.get_label() for line in lines], fontsize=8)
    axes[0, 1].plot(control_steps, numeric("general_capability_raw"), marker="o", label="raw")
    axes[0, 1].plot(control_steps, numeric("general_capability_ema"), marker="s", label="EMA")
    for field, style in (("safe_threshold", "--"), ("warn_threshold", ":"), ("hard_threshold", "-.")):
        axes[0, 1].plot(control_steps, numeric(field), style, label=field.replace("_threshold", ""))
    axes[0, 1].set(title="Mock capability feedback", xlabel="optimizer step", ylabel="score")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(control_steps, numeric("beta_new"), color="tab:red", marker="o")
    axes[1, 0].set(title="PID-controlled KL beta", xlabel="optimizer step", ylabel="beta")
    axes[1, 1].plot(control_steps, numeric("risk_weight"), marker="o", label="risk gate")
    axes[1, 1].plot(control_steps, numeric("p_term"), label="P")
    axes[1, 1].plot(control_steps, numeric("i_term"), label="I")
    axes[1, 1].plot(control_steps, numeric("d_term"), label="D")
    axes[1, 1].set(title="Controller internals", xlabel="optimizer step")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("PID-GRPO P0 integration demo", fontsize=15)
    fig.text(
        0.5, 0.5, "TOY MODEL + MOCK CAPABILITY — NOT AN EXPERIMENTAL RESULT",
        ha="center", va="center", fontsize=18, alpha=0.12, rotation=22,
    )
    figure_path = root / "toy_training_process.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(14, 3.8))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 4)
    axis.axis("off")
    labels = [
        "optimizer.step", "TRL on_step_end", "D_control evaluate",
        "Raw SafetyGuard", "PID update beta", "Atomic checkpoint",
    ]
    colors = ["#dbeafe", "#e0e7ff", "#fef3c7", "#fee2e2", "#dcfce7", "#f3e8ff"]
    positions = [(0.2, 2.0), (2.5, 2.0), (4.8, 2.0), (7.1, 2.0), (9.4, 2.0), (11.7, 2.0)]
    for index, ((x, y), label, color) in enumerate(zip(positions, labels, colors)):
        box = FancyBboxPatch((x, y), 1.8, 0.8, boxstyle="round,pad=0.08", fc=color, ec="#334155")
        axis.add_patch(box)
        axis.text(x + 0.9, y + 0.4, label, ha="center", va="center", fontsize=9)
        if index < len(positions) - 1:
            axis.annotate("", xy=(positions[index + 1][0], y + 0.4), xytext=(x + 1.8, y + 0.4),
                          arrowprops={"arrowstyle": "->", "color": "#334155", "lw": 1.5})
    axis.annotate("HARD / INVALID: stop + restore latest-safe + re-evaluate",
                  xy=(8.0, 2.0), xytext=(7.0, 0.6), ha="center", fontsize=10,
                  arrowprops={"arrowstyle": "->", "color": "#dc2626", "lw": 1.5}, color="#991b1b")
    axis.set_title("P0 test and training control flow", fontsize=15)
    flow_path = root / "control_flow.png"
    fig.savefig(flow_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"training metrics: {training_csv}")
    print(f"control metrics:  {control_csv}")
    print(f"training figure:  {figure_path}")
    print(f"flow figure:      {flow_path}")
    print(f"safe checkpoints: {len(manager.records)}")


if __name__ == "__main__":
    main()
