from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

from src.safety.hf_checkpoint import HFTrainingStateIO
from src.controllers.pid_controller import PIDConfig, PIDController
from src.evaluation.capability_evaluator import MockCapabilityEvaluator
from src.safety.checkpoint_manager import CheckpointManager
from src.safety.safety_guard import SafetyGuard
from src.safety.thresholds import SafetyThresholds, ThresholdConfig
from src.trainers.hf_control_callback import PIDControlCallback
from src.trainers.pid_grpo_trainer import (
    DynamicBetaAdapter, PIDGRPOControlLoop,
)


class IdentityAccelerator:
    def gather_for_metrics(self, value):
        return value


def test_dynamic_beta_changes_pinned_trl_loss_numerically() -> None:
    trainer = object.__new__(trl.GRPOTrainer)
    trainer.beta = 0.1
    trainer.args = SimpleNamespace(beta=0.1)
    trainer.accelerator = IdentityAccelerator()
    trainer._metrics = defaultdict(list)

    def logps(self, model, input_ids, attention_mask, logits_to_keep):
        return torch.zeros((1, logits_to_keep), dtype=torch.float32, requires_grad=True)

    trainer._get_per_token_logps = MethodType(logps, trainer)
    inputs = {
        "prompt_ids": torch.tensor([[1]]),
        "prompt_mask": torch.tensor([[1]]),
        "completion_ids": torch.tensor([[2, 3]]),
        "completion_mask": torch.tensor([[1.0, 1.0]]),
        "ref_per_token_logps": torch.ones((1, 2)),
        "advantages": torch.zeros(1),
    }
    adapter = DynamicBetaAdapter(trainer, "pid_kl", 0.1)
    first = trainer.compute_loss(None, inputs).item()
    adapter.set_beta(0.2)
    second = trainer.compute_loss(None, inputs).item()
    assert second == pytest.approx(2 * first)
    assert trainer.args.beta == 0.2


def test_hf_checkpoint_restores_model_optimizer_scheduler_and_state(tmp_path: Path) -> None:
    from transformers import TrainerState

    class Processing:
        def save_pretrained(self, path):
            Path(path, "tokenizer.marker").write_text("ok", encoding="utf-8")

    class Trainer:
        def __init__(self):
            self.model = torch.nn.Linear(2, 1)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1)
            self.state = TrainerState(global_step=3)

        def save_model(self, path):
            target = Path(path)
            target.mkdir(parents=True)
            torch.save(self.model.state_dict(), target / "pytorch_model.bin")

        def _load_from_checkpoint(self, path):
            self.model.load_state_dict(
                torch.load(Path(path) / "pytorch_model.bin", weights_only=True)
            )

    trainer = Trainer()
    expected = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}
    io = HFTrainingStateIO(trainer, Processing())
    io.save(tmp_path)
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(10)
    trainer.state.global_step = 99
    io.restore(tmp_path)
    assert trainer.state.global_step == 3
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(value, expected[name])


def test_pinned_trl_runs_one_real_optimizer_step_with_pid_callback(tmp_path: Path) -> None:
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel(
        {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "question": 3, "answer": 4},
        unk_token="[UNK]",
    ))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=5, n_layer=1, n_head=1, n_embd=16, n_positions=32,
        pad_token_id=0, eos_token_id=1,
    ))
    args = trl.GRPOConfig(
        output_dir=str(tmp_path / "trainer"), max_steps=1,
        per_device_train_batch_size=2, num_generations=2,
        max_prompt_length=8, max_completion_length=2,
        beta=0.1, report_to=[], save_strategy="no", logging_strategy="no",
        use_cpu=True,
    )

    def reward(completions, **kwargs):
        return [0.0] * len(completions)

    trainer = trl.GRPOTrainer(
        model=model, args=args, reward_funcs=reward,
        train_dataset=Dataset.from_list([
            {"prompt": "question", "answer": "answer"},
            {"prompt": "question", "answer": "answer"},
        ]),
        processing_class=tokenizer,
    )
    thresholds = SafetyThresholds.from_baseline(1.0, ThresholdConfig(0.1))
    pid = PIDController(
        PIDConfig(kp=5, ki=0, kd=0, warmup_control_steps=0),
        thresholds, initial_beta=0.1,
    )
    manager = CheckpointManager(tmp_path / "safe")
    loop = PIDGRPOControlLoop(
        evaluator=MockCapabilityEvaluator([0.91]), control_dataset=None,
        guard=SafetyGuard(thresholds),
        beta_adapter=DynamicBetaAdapter(trainer, "pid_kl", 0.1),
        checkpoint_manager=manager, eval_interval_optimizer_steps=1, pid=pid,
    )
    state_io = HFTrainingStateIO(trainer, tokenizer)
    loop.checkpoint_artifact_writer = state_io.save
    trainer.add_callback(PIDControlCallback(
        trainer=trainer, loop=loop, state_io=state_io, baseline_score=1.0,
    ))
    trainer.train()
    assert loop.optimizer_step == 1
    assert trainer.beta > 0.1
    assert len(manager.records) == 2
