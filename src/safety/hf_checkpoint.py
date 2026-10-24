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
        root.mkdir(parents=True, exist_ok=False)
        # Match the pinned Transformers Trainer checkpoint layout so train()
        # can resume its global step, dataloader position and scheduler itself.
        self.trainer.save_model(str(root))
        if self.processing_class is not None:
            self.processing_class.save_pretrained(str(root))
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
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
            root / "rng_state.pth",
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
        if not root.is_dir():
            raise FileNotFoundError(root)
        model_dir = root / "model" if (root / "model").is_dir() else root
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
        rng_path = root / "rng_state.pth"
        if not rng_path.exists():
            rng_path = root / "rng_state.pt"
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng.get("cpu", rng.get("torch_cpu")))
        if torch.cuda.is_available():
            if rng_path.name == "rng_state.pt" and rng.get("torch_cuda"):
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
            elif rng.get("cuda") is not None:
                torch.cuda.set_rng_state(rng["cuda"])

    @classmethod
    def resume_path(cls, checkpoint_directory: str | Path) -> Path:
        root = Path(checkpoint_directory) / cls.ARTIFACT_DIR
        if not (root / "trainer_state.json").is_file():
            raise FileNotFoundError(root / "trainer_state.json")
        if (root / "model").exists():
            raise RuntimeError("legacy checkpoint layout cannot resume Trainer.train")
        return root
