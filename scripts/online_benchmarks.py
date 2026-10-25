#!/usr/bin/env python3
"""Freeze streaming sample identities, or evaluate a local base/LoRA model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.online_data import FINAL_ROLES, check_manifest, digest, load_suite, materialize_roles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "freeze", "evaluate"])
    parser.add_argument("--suite", default="configs/online_benchmarks.yaml")
    parser.add_argument("--roles", nargs="+", default=["control"])
    parser.add_argument("--manifest", default="benchmark_manifests/online_v1.json")
    parser.add_argument("--allow-final", action="store_true", help="explicit final/heldout evaluation")
    parser.add_argument("--model", help="local or Hugging Face BASE model")
    parser.add_argument("--revision", help="immutable base-model revision")
    parser.add_argument("--adapter", help="optional saved LoRA adapter directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-context-length", type=int, default=4096)
    parser.add_argument("--output", default="outputs/online_evaluation.json")
    parser.add_argument("--baseline-results", help="prior baseline output with the identical frozen suite")
    args = parser.parse_args()
    suite = load_suite(args.suite)
    if args.action == "plan":
        print(json.dumps(suite, indent=2, ensure_ascii=False))
        return
    if args.action == "evaluate" and FINAL_ROLES.intersection(args.roles) and not args.allow_final:
        parser.error("heldout/final evaluation requires --allow-final after tuning is frozen")
    if args.action == "evaluate" and (not args.model or not args.revision):
        parser.error("evaluate requires --model and --revision")
    manifest_path = Path(args.manifest)
    if args.action == "freeze" and manifest_path.exists():
        parser.error("manifest already exists; use a new name rather than overwrite frozen samples")
    if args.action == "evaluate" and not manifest_path.is_file():
        parser.error("freeze the requested roles before evaluating")
    if args.action == "evaluate" and Path(args.output).exists():
        parser.error("output exists; choose a new output path")
    baseline = None
    if args.baseline_results:
        baseline = json.loads(Path(args.baseline_results).read_text(encoding="utf-8"))
    roles, manifest = materialize_roles(suite, args.roles)
    if args.action == "freeze":
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"manifest": str(manifest_path),
                          "counts": {key: len(rows) for key, rows in roles.items()}}, indent=2))
        return
    check_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)
    if baseline and baseline["suite_sha256"] != manifest["suite_sha256"]:
        parser.error("baseline uses a different suite")
    sample_hashes = {role: digest(samples) for role, samples in manifest["roles"].items()}
    if baseline and any(baseline.get("sample_hashes", {}).get(role) != value
                        for role, value in sample_hashes.items()):
        parser.error("baseline sample identities differ")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from src.evaluation.benchmark_evaluator import HFBenchmarkScorer, OnlineBenchmarkEvaluator

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    scorer = HFBenchmarkScorer(tokenizer, max_new_tokens=args.max_new_tokens,
                               max_context_length=args.max_context_length)
    for rows in roles.values():
        scorer.prepare(rows)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=False,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    evaluator = OnlineBenchmarkEvaluator(scorer)
    output = Path(args.output)
    # Different roles remain separate; final metrics never enter control aggregation.
    results = {role: evaluator.evaluate(model, rows) for role, rows in roles.items()}
    if baseline:
        for role, result in results.items():
            for task, metrics in result["tasks"].items():
                old = baseline["results"][role]["tasks"][task]
                if old["num_samples"] != metrics["num_samples"]:
                    raise ValueError("baseline sample count differs")
                metrics["baseline_score"] = old["score"]
                metrics["delta"] = metrics["score"] - old["score"]
                metrics["relative_degradation"] = ((old["score"] - metrics["score"]) / old["score"]
                                                   if old["score"] > 0 else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": args.model, "revision": args.revision,
                                  "adapter": args.adapter, "manifest": str(manifest_path),
                                  "sample_hashes": sample_hashes,
                                  "suite_sha256": manifest["suite_sha256"], "results": results},
                                 indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    main()
