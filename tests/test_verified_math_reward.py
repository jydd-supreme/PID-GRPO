import pytest

pytest.importorskip("math_verify")

from src.rewards.verified_math_reward import select_reward, verified_math_reward, verify_math_answer


def test_fractions_are_verified_symbolically_instead_of_using_last_digit():
    assert verify_math_answer(r"\boxed{\frac{1}{2}}", "0.5")["score"] == 1.
    assert verify_math_answer(r"\boxed{\frac{1}{2}}", "2")["score"] == 0.
    assert verify_math_answer(r"\boxed{x+x}", "2x")["score"] == 1.


def test_chat_rewards_match_string_rewards_and_validate_alignment():
    completions = [[{"role": "assistant", "content": r"\boxed{42}"}], r"\boxed{7}"]
    assert verified_math_reward(completions, ["42", "8"]) == [1., 0.]
    with pytest.raises(ValueError, match="lengths differ"):
        verified_math_reward(completions, ["42"])
    legacy = object()
    assert select_reward("numeric_exact_match", legacy) is legacy
