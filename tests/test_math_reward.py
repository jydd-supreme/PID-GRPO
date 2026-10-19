import pytest

from src.rewards.math_reward import compute_reward, extract_answer, normalize_answer


@pytest.mark.parametrize("text", ["42", "42.", "#### 42", r"\boxed{42}"])
def test_supported_answer_forms(text: str) -> None:
    assert compute_reward(text, "42") == 1.0


def test_last_numeric_answer_and_incorrect_reward() -> None:
    assert extract_answer("first 7, finally 42") == "42"
    assert compute_reward("41", "42") == 0.0


def test_decimal_normalization_not_symbolic_math() -> None:
    assert normalize_answer("42.00") == "42"
    assert compute_reward("1/2", "0.5") == 0.0
