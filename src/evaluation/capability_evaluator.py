"""Capability evaluator contracts and deterministic mock implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Iterable, Sequence

from src.rewards.math_reward import is_correct


class DatasetRole(str, Enum):
    TRAIN = "train"
    CONTROL = "control"
    FINAL_TEST = "final_test"


@dataclass(frozen=True)
class DatasetDescriptor:
    name: str
    role: DatasetRole
    sample_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class CapabilityResult:
    overall_score: float
    subscores: dict[str, float]
    num_samples: int
    valid: bool
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_dataset_isolation(datasets: Iterable[DatasetDescriptor]) -> None:
    items = list(datasets)
    roles = {item.role for item in items}
    if roles != {DatasetRole.TRAIN, DatasetRole.CONTROL, DatasetRole.FINAL_TEST}:
        raise ValueError("exactly the train, control, and final_test roles are required")
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            overlap = left.sample_ids & right.sample_ids
            if overlap:
                examples = sorted(overlap)[:3]
                raise ValueError(
                    f"dataset leakage between {left.role.value} and {right.role.value}: {examples}"
                )


class CapabilityEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model: Any, dataset: Any) -> dict[str, Any]:
        """Return overall_score, subscores, num_samples, and valid."""


class MockCapabilityEvaluator(CapabilityEvaluator):
    """Consumes preset results. Outputs are synthetic and never real evidence."""

    def __init__(self, results: Sequence[float | CapabilityResult | dict[str, Any]]) -> None:
        if not results:
            raise ValueError("at least one mock result is required")
        self._results = list(results)
        self._index = 0

    def evaluate(self, model: Any = None, dataset: Any = None) -> dict[str, Any]:
        if self._index >= len(self._results):
            raise IndexError("mock capability sequence exhausted")
        raw = self._results[self._index]
        self._index += 1
        if isinstance(raw, CapabilityResult):
            result = raw.to_dict()
        elif isinstance(raw, dict):
            result = dict(raw)
        else:
            score = float(raw)
            result = CapabilityResult(
                overall_score=score,
                subscores={},
                num_samples=1,
                valid=math.isfinite(score),
                source="mock",
            ).to_dict()
        result["source"] = "mock"
        return result

    def state_dict(self) -> dict[str, int]:
        return {"version": 1, "index": self._index}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported mock evaluator state")
        index = int(state["index"])
        if not 0 <= index <= len(self._results):
            raise ValueError("invalid mock evaluator index")
        self._index = index


class RealCapabilityEvaluator(CapabilityEvaluator):
    """Deterministic local exact-match evaluator for causal language models.

    Imports torch lazily so the lightweight controller tests do not require the
    training dependency set. The dataset must expose ``prompt`` and ``answer``.
    """

    def __init__(
        self,
        processing_class: Any,
        *,
        batch_size: int = 4,
        max_new_tokens: int = 128,
    ) -> None:
        if batch_size < 1 or max_new_tokens < 1:
            raise ValueError("batch_size and max_new_tokens must be positive")
        self.processing_class = processing_class
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def _format_prompt(self, prompt: Any) -> str:
        if isinstance(prompt, list) and hasattr(self.processing_class, "apply_chat_template"):
            return self.processing_class.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        return str(prompt)

    def evaluate(self, model: Any, dataset: Any) -> dict[str, Any]:
        import torch

        rows = list(dataset)
        if not rows:
            return CapabilityResult(0.0, {}, 0, False, "local_exact_match").to_dict()
        if any("prompt" not in row or "answer" not in row for row in rows):
            raise ValueError("evaluation rows require prompt and answer fields")

        tokenizer = self.processing_class
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        was_training = model.training
        model.eval()
        predictions: list[str] = []
        try:
            with torch.inference_mode():
                for start in range(0, len(rows), self.batch_size):
                    batch = rows[start : start + self.batch_size]
                    prompts = [self._format_prompt(row["prompt"]) for row in batch]
                    encoded = tokenizer(
                        prompts, return_tensors="pt", padding=True, truncation=True
                    )
                    encoded = {name: value.to(device) for name, value in encoded.items()}
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    prompt_width = encoded["input_ids"].shape[1]
                    predictions.extend(
                        tokenizer.batch_decode(
                            generated[:, prompt_width:], skip_special_tokens=True
                        )
                    )
        finally:
            if was_training:
                model.train()

        correct = sum(
            is_correct(prediction, str(row["answer"]))
            for prediction, row in zip(predictions, rows)
        )
        score = correct / len(rows)
        return CapabilityResult(
            overall_score=score,
            subscores={"exact_match": score},
            num_samples=len(rows),
            valid=len(predictions) == len(rows),
            source="local_exact_match",
        ).to_dict()
