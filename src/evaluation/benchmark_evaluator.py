"""Opt-in benchmark scorer for in-memory rows obtained from online_data."""

from __future__ import annotations

from collections import defaultdict
import math
import json
from pathlib import Path
from typing import Any, Iterable

from src.evaluation.capability_evaluator import CapabilityEvaluator


class HFBenchmarkScorer:
    """Zero-shot raw prompts for MC; chat template and greedy decoding for generation.

    MC uses summed continuation log-probabilities and character-normalized scores
    for ARC/HellaSwag. No training/evaluation dependencies are imported eagerly.
    This protocol is project-specific, not a claimed leaderboard reproduction.
    """

    def __init__(self, tokenizer: Any, *, max_new_tokens: int = 2048,
                 max_context_length: int = 4096) -> None:
        if max_new_tokens < 1 or max_context_length < 2:
            raise ValueError("invalid generation/context lengths")
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.max_context_length = max_context_length

    def prepare(self, rows: Iterable[dict[str, Any]]) -> None:
        kinds = {row["kind"] for row in rows}
        if "math" in kinds:
            try:
                import math_verify  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("Math evaluation requires math-verify; see docs/online_evaluation.md") from exc
        if "ifeval" in kinds:
            try:
                from lm_eval.tasks.ifeval.utils import process_results  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("IFEval requires lm-eval[ifeval]==0.4.8 and its resources") from exc

    def _loglikelihood(self, model: Any, prompt: str, choice: str) -> float:
        import torch

        continuation = " " + choice
        # Same boundary convention as encode_pair: move trailing prompt spaces.
        stripped = prompt.rstrip()
        continuation = prompt[len(stripped):] + continuation
        context_ids = self.tokenizer.encode(stripped, add_special_tokens=False)
        all_ids = self.tokenizer.encode(stripped + continuation, add_special_tokens=False)
        if not context_ids:
            context_ids = [self.tokenizer.eos_token_id]
            all_ids = context_ids + self.tokenizer.encode(continuation, add_special_tokens=False)
        start = len(context_ids)
        if len(all_ids) <= start or len(all_ids) > self.max_context_length:
            raise ValueError("empty continuation or context limit exceeded; no silent truncation")
        device = next(model.parameters()).device
        input_ids = torch.tensor([all_ids], device=device)
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                       use_cache=False).logits[0, start - 1:-1].float()
        targets = input_ids[0, start:]
        return float(logits.log_softmax(-1).gather(-1, targets[:, None]).sum().item())

    def _generate(self, model: Any, prompt: str) -> tuple[str, bool]:
        tokenizer = self.tokenizer
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        width = inputs["input_ids"].shape[1]
        if width + self.max_new_tokens > self.max_context_length:
            raise ValueError("prompt plus generation budget exceeds configured context length")
        inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}
        output = model.generate(**inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id, use_cache=True)
        completion = output[0, width:]
        eos = tokenizer.eos_token_id
        truncated = len(completion) >= self.max_new_tokens and (eos is None or int(completion[-1]) != eos)
        return tokenizer.decode(completion, skip_special_tokens=True), truncated

    def score(self, model: Any, row: dict[str, Any]) -> dict[str, Any]:
        if "choices" in row:
            scores = [self._loglikelihood(model, row["prompt"], choice) for choice in row["choices"]]
            if not all(math.isfinite(x) for x in scores):
                raise ValueError("non-finite candidate log-probability")
            normalized = [score / max(1, len(choice)) for score, choice in zip(scores, row["choices"])]
            raw_correct = max(range(len(scores)), key=scores.__getitem__) == row["gold"]
            norm_correct = max(range(len(scores)), key=normalized.__getitem__) == row["gold"]
            return {"score": float(raw_correct if row["metric"] == "acc" else norm_correct),
                    "acc": float(raw_correct), "acc_norm": float(norm_correct)}
        response, truncated = self._generate(model, row["prompt"])
        if row["kind"] == "math":
            from src.rewards.verified_math_reward import verify_math_answer
            return {**verify_math_answer(response, row["answer"]), "truncated": truncated}
        from lm_eval.tasks.ifeval.utils import process_results

        metrics = process_results(row["raw"], [response])
        return {"score": float(metrics["prompt_level_strict_acc"]),
                "truncated": truncated, **metrics}


class OnlineBenchmarkEvaluator(CapabilityEvaluator):
    """Evaluate the SAME fixed in-memory samples at every control update."""

    def __init__(self, scorer: Any, *, history_path: str | Path | None = None,
                 step_provider: Any = None, progress_dir: str | Path | None = None,
                 progress_identity: Any = None, session: Any = None) -> None:
        self.scorer = scorer
        self.history_path = Path(history_path) if history_path else None
        self.step_provider = step_provider
        self.progress_dir = Path(progress_dir) if progress_dir else None
        self.progress_identity = progress_identity
        self.session = session

    def evaluate(self, model: Any, dataset: Any) -> dict[str, Any]:
        import torch

        rows = list(dataset)
        if not rows:
            raise ValueError("empty benchmark role")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate benchmark sample")
        self.scorer.prepare(rows)
        if self.session:
            self.session.check()
        progress = None
        if self.progress_dir:
            from src.evaluation.progress import EvaluationProgress
            progress = EvaluationProgress(self.progress_dir, model, rows, self.progress_identity)
        was_training = model.training
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        model.eval()
        try:
            with torch.inference_mode():
                for row in rows:
                    if self.session:
                        self.session.check()
                    scored = progress.results.get(row["id"]) if progress else None
                    if scored is None:
                        scored = self.scorer.score(model, row)
                        if not math.isfinite(float(scored["score"])) or not 0 <= scored["score"] <= 1:
                            raise ValueError("invalid benchmark score")
                        if progress:
                            progress.append(row["id"], scored)
                    if not math.isfinite(float(scored["score"])) or not 0 <= scored["score"] <= 1:
                        raise ValueError("invalid benchmark score")
                    by_task[row["task"]].append(scored)
        finally:
            model.train(was_training)
        tasks = {}
        for task, values in by_task.items():
            details: dict[str, Any] = {"score": sum(x["score"] for x in values) / len(values),
                                       "num_samples": len(values)}
            for metric in ("acc", "acc_norm", "truncated", "parse_failed",
                           "prompt_level_strict_acc", "prompt_level_loose_acc"):
                if metric in values[0]:
                    details[metric] = sum(float(x[metric]) for x in values) / len(values)
            for metric in ("inst_level_strict_acc", "inst_level_loose_acc"):
                if metric in values[0]:
                    flags = [flag for value in values for flag in value[metric]]
                    details[metric] = sum(flags) / len(flags)
            tasks[task] = details
        result = {"overall_score": sum(x["score"] for values in by_task.values() for x in values) / len(rows),
                "subscores": {name: value["score"] for name, value in tasks.items()},
                "tasks": tasks, "num_samples": len(rows), "valid": True,
                "source": "pinned_online_benchmarks", "protocol": "pid-grpo-zero-shot-v1"}
        if self.history_path:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"optimizer_step": self.step_provider() if self.step_provider else None,
                                         "session_id": self.session.session_id if self.session else None,
                                         "role": rows[0].get("role"), **result}) + "\n")
        return result
