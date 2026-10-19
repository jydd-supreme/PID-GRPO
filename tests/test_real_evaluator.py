import pytest

torch = pytest.importorskip("torch")

from src.evaluation.capability_evaluator import RealCapabilityEvaluator


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "[EOS]"

    def __call__(self, prompts, **kwargs):
        return {
            "input_ids": torch.tensor([[3, 4]] * len(prompts)),
            "attention_mask": torch.ones((len(prompts), 2), dtype=torch.long),
        }

    def batch_decode(self, values, **kwargs):
        return ["42" if int(row[0]) == 5 else "wrong" for row in values]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, attention_mask, **kwargs):
        answer = torch.full((input_ids.shape[0], 1), 5, device=input_ids.device)
        return torch.cat([input_ids, answer], dim=1)


def test_real_evaluator_generates_and_scores_exact_match() -> None:
    model = Model()
    model.train()
    result = RealCapabilityEvaluator(Tokenizer(), batch_size=2, max_new_tokens=1).evaluate(
        model,
        [
            {"prompt": "one", "answer": "42"},
            {"prompt": "two", "answer": "41"},
        ],
    )
    assert result["valid"]
    assert result["num_samples"] == 2
    assert result["overall_score"] == 0.5
    assert result["source"] == "local_exact_match"
    assert model.training
