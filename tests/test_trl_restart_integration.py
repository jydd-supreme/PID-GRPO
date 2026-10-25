"""No-download CPU validation against the pinned TRL/Transformers APIs."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

from src.controllers.pid_controller import PIDConfig, PIDController
from src.data.resumable_sampler import ResumableRepeatSampler
from src.safety.checkpoint_manager import CheckpointManager
from src.safety.hf_checkpoint import HFTrainingStateIO
from src.safety.safety_guard import SafetyGuard
from src.safety.thresholds import SafetyThresholds, ThresholdConfig
from src.trainers.hf_control_callback import PIDControlCallback
from src.trainers.pid_grpo_trainer import DynamicBetaAdapter, PIDGRPOControlLoop
from src.utils.run_session import (RestartCheckpointStore, RunSession, SessionPaused,
                                   TrainingBudgetComplete, restore_control_state)


def make_training(root: Path, *, stop_at=None, lora=False, interrupt_evaluation=False):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TrainerCallback

    torch.manual_seed(42)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2,
                                   "q0": 3, "q1": 4, "q2": 5, "q3": 6, "answer": 7}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=8, n_layer=1, n_head=1, n_embd=8,
                                      n_positions=16, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
                                      eos_token_id=1, pad_token_id=0))
    class Trainer(trl.GRPOTrainer):
        def _get_train_sampler(self):
            return ResumableRepeatSampler(self.train_dataset, self.num_generations, self.args.seed)

    args = trl.GRPOConfig(output_dir=str(root / "trainer"), max_steps=5,
                           per_device_train_batch_size=2, gradient_accumulation_steps=2,
                           num_generations=2, max_prompt_length=2, max_completion_length=2,
                           beta=.04, learning_rate=1e-3, report_to=[], save_strategy="no",
                           logging_strategy="no", use_cpu=True, disable_tqdm=True)
    peft_config = None
    if lora:
        from peft import LoraConfig
        peft_config = LoraConfig(r=2, lora_alpha=4, lora_dropout=0., target_modules=["c_attn"], task_type="CAUSAL_LM")
    trainer = Trainer(model=model, args=args, processing_class=tokenizer,
                      train_dataset=Dataset.from_list([{"prompt": f"q{i}"} for i in range(4)]),
                      reward_funcs=lambda completions, **kwargs: [float(i % 2) for i in range(len(completions))],
                      peft_config=peft_config)
    class Evaluator:
        def evaluate(self, model, dataset):
            if interrupt_evaluation and int(trainer.state.global_step) == stop_at:
                raise SessionPaused("test interrupted control evaluation")
            return {"valid": True, "overall_score": .99, "num_samples": 1}
    thresholds = SafetyThresholds.from_baseline(1., ThresholdConfig(.1))
    loop = PIDGRPOControlLoop(evaluator=Evaluator(), control_dataset=[], guard=SafetyGuard(thresholds),
                              beta_adapter=DynamicBetaAdapter(trainer, "pid_kl", .04),
                              checkpoint_manager=CheckpointManager(root / "safe"), eval_interval_optimizer_steps=2,
                              pid=PIDController(PIDConfig(), thresholds, initial_beta=.04))
    state_io = HFTrainingStateIO(trainer, tokenizer)
    loop.checkpoint_artifact_writer = state_io.save
    store = RestartCheckpointStore(root / "restart")
    session = RunSession(root)
    if stop_at is not None:
        class Stop(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == stop_at:
                    session.pause_reason = "test interruption"
        trainer.add_callback(Stop())
    callback = PIDControlCallback(trainer=trainer, loop=loop, state_io=state_io, baseline_score=1.,
                                  session=session, restart_store=store, checkpoint_steps=1)
    trainer.add_callback(callback)
    return trainer, loop, store, callback


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("stop_at,interrupt_evaluation", [(3, False), (2, True)])
def test_real_optimizer_restart_matches_uninterrupted_training_across_epochs(tmp_path, lora, stop_at, interrupt_evaluation):
    uninterrupted, complete_loop, _, _ = make_training(tmp_path / "continuous", lora=lora)
    uninterrupted.train()
    expected = {name: value.detach().clone() for name, value in uninterrupted.model.state_dict().items()}
    first, first_loop, store, _ = make_training(tmp_path / "restarted", stop_at=stop_at,
                                               lora=lora, interrupt_evaluation=interrupt_evaluation)
    with pytest.raises(SessionPaused):
        first.train()
    directory, saved = store.latest()
    assert saved["global_step"] == stop_at
    assert (saved["pending_optimizer_step"] is not None) == interrupt_evaluation
    resumed, loop, _, callback = make_training(tmp_path / "restarted", lora=lora)
    restore_control_state(loop, saved)
    path = HFTrainingStateIO.resume_path(directory)
    callback.load_restart_state(saved, rng_path=path)
    resumed.train(resume_from_checkpoint=str(path))
    assert resumed.state.global_step == loop.optimizer_step == 5
    assert loop.pid.state_dict() == complete_loop.pid.state_dict()
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=1e-6, atol=1e-7)
    # A kill after the last snapshot but before export must not take step six.
    directory, saved = store.latest()
    at_budget, budget_loop, _, budget_callback = make_training(tmp_path / "restarted", lora=lora)
    restore_control_state(budget_loop, saved)
    path = HFTrainingStateIO.resume_path(directory)
    budget_callback.load_restart_state(saved, rng_path=path)
    with pytest.raises(TrainingBudgetComplete):
        at_budget.train(resume_from_checkpoint=str(path))
    assert at_budget.state.global_step == 5
    for name, value in at_budget.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=1e-6, atol=1e-7)
