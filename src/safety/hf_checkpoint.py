"""Hugging Face training artifact persistence for safe control checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any


class HFTrainingStateIO:
    """Save and restore model, optimizer, scheduler, trainer and RNG state.

    The surrounding :class:`CheckpointManager` supplies atomic directory
    publication. This class deliberately targets a single Trainer process;
    distributed launch is rejected by the real-training entry point until a
    multi-rank integration test is added.
    """

    ARTIFACT_DIR = "hf_training"

    def __init__(self, trainer: Any, processing_class: Any) -> None:
        self.trainer = trainer
        self.processing_class = processing_class

    def save(self, checkpoint_directory: Path) -> None:
        import numpy as np
        import torch

        root = checkpoint_directory / self.ARTIFACT_DIR
        model_dir = root / "model"
        root.mkdir(parents=True, exist_ok=False)
        self.trainer.save_model(str(model_dir))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(model_dir))
        if self.trainer.optimizer is None:
            raise RuntimeError("optimizer must exist before a real checkpoint is saved")
        torch.save(self.trainer.optimizer.state_dict(), root / "optimizer.pt")
        if self.trainer.lr_scheduler is not None:
            torch.save(self.trainer.lr_scheduler.state_dict(), root / "scheduler.pt")
        self.trainer.state.save_to_json(str(root / "trainer_state.json"))
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            root / "rng_state.pt",
        )
        (root / "runtime.json").write_text(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "python_checkpoint_format": 1,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def restore(self, checkpoint_directory: str | Path) -> None:
        import numpy as np
        import torch
        from transformers import TrainerState

        root = Path(checkpoint_directory) / self.ARTIFACT_DIR
        model_dir = root / "model"
        if not root.is_dir():
            raise FileNotFoundError(root)
        # Trainer owns the PEFT/full-model distinction and applies the correct
        # adapter or model state loader for the pinned Transformers release.
        self.trainer._load_from_checkpoint(str(model_dir))
        if self.trainer.optimizer is None:
            raise RuntimeError("optimizer must exist before checkpoint restore")
        self.trainer.optimizer.load_state_dict(
            torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False)
        )
        scheduler_path = root / "scheduler.pt"
        if scheduler_path.exists() and self.trainer.lr_scheduler is not None:
            self.trainer.lr_scheduler.load_state_dict(
                torch.load(scheduler_path, map_location="cpu", weights_only=False)
            )
        self.trainer.state = TrainerState.load_from_json(str(root / "trainer_state.json"))
        rng = torch.load(root / "rng_state.pt", map_location="cpu", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
