"""Conservative string-based math answer reward (no symbolic equivalence)."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re


_BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_HASH = re.compile(r"####\s*([^\n]+)")
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def extract_answer(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    hashed = _HASH.findall(text)
    if hashed:
        return hashed[-1].strip()
    numbers = _NUMBER.findall(text)
    return numbers[-1].strip() if numbers else text.strip()


def normalize_answer(answer: str) -> str:
    value = extract_answer(answer).strip()
    value = re.sub(r"[\s,]+", "", value)
    value = value.rstrip(".")
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value.casefold()
    if not number.is_finite():
        return value.casefold()
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", "+0"} else normalized


def is_correct(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def compute_reward(prediction: str, reference: str) -> float:
    return 1.0 if is_correct(prediction, reference) else 0.0
