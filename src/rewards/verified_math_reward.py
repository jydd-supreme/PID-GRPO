"""Opt-in symbolic math scoring for the approved DeepMath experiment."""

from __future__ import annotations

from typing import Any, Mapping


def require_math_verifier() -> None:
    try:
        import math_verify  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("DeepMath requires math-verify[antlr4_11_0]==0.7.0") from exc


def verify_math_answer(prediction: str, reference: str) -> dict[str, Any]:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    gold = parse("\\boxed{" + reference + "}", extraction_config=[LatexExtractionConfig()])
    if not gold:
        raise ValueError("unparseable math reference; cannot silently train on incorrect labels")
    parsed = parse(prediction, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
    return {"score": float(bool(parsed) and verify(gold, parsed)), "parse_failed": not bool(parsed)}


def _text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], Mapping):
        return str(completion[-1].get("content", ""))
    raise TypeError("unsupported completion format")


def verified_math_reward(completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
    if len(completions) != len(answer):
        raise ValueError("completion/reference lengths differ")
    return [verify_math_answer(_text(completion), str(reference))["score"]
            for completion, reference in zip(completions, answer)]


def select_reward(backend: str, legacy_reward: Any) -> Any:
    if backend == "numeric_exact_match":
        return legacy_reward
    if backend == "math_verify":
        require_math_verifier()
        return verified_math_reward
    raise ValueError(f"unsupported reward backend: {backend}")
