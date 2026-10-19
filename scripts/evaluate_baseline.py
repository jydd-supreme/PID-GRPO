#!/usr/bin/env python3
"""Evaluate the frozen base model before a real smoke-training trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_builder import load_local_jsonl
from src.utils.config import load_config
from src.utils.storage import configure_runtime_storage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke_qwen_pid.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    storage = configure_runtime_storage(config)

    from src.trainers.real_grpo import verify_training_environment

    verify_training_environment()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluation.capability_evaluator import RealCapabilityEvaluator

    training = config["training"]
    evaluation = config["evaluation"]
    model_name = str(training["model_name"])
    revision = str(training["model_revision"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=storage["HF_HUB_CACHE"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        cache_dir=storage["HF_HUB_CACHE"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).to("cuda")
    evaluator = RealCapabilityEvaluator(
        tokenizer,
        batch_size=int(evaluation["batch_size"]),
        max_new_tokens=int(evaluation["max_new_tokens"]),
    )
    rows = load_local_jsonl(config["data"]["control_file"])
    result = evaluator.evaluate(model, rows)
    result.update(
        {
            "model": model_name,
            "model_revision": revision,
            "split": "control",
            "data_source": "real_gsm8k_smoke",
        }
    )
    if result.get("valid") is not True or float(result["overall_score"]) <= 0:
        raise RuntimeError(f"invalid or zero baseline: {result}")
    output = Path(config["project"]["output_dir"]) / "metrics" / "baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
