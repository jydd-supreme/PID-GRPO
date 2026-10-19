from pathlib import Path

import pytest

from src.evaluation.capability_evaluator import (
    DatasetDescriptor, DatasetRole, validate_dataset_isolation,
)
from src.utils.config import load_config


@pytest.mark.parametrize(
    "name",
    [
        "base.yaml",
        "no_kl.yaml",
        "fixed_kl.yaml",
        "pid_kl.yaml",
        "real_pid_kl.yaml",
        "smoke_qwen_pid.yaml",
    ],
)
def test_shipped_configs_are_valid(name: str) -> None:
    config = load_config(Path("configs") / name)
    assert isinstance(config["training"]["enabled"], bool)


def test_split_leakage_is_rejected() -> None:
    descriptors = [
        DatasetDescriptor("train", DatasetRole.TRAIN, frozenset({"a"})),
        DatasetDescriptor("control", DatasetRole.CONTROL, frozenset({"a"})),
        DatasetDescriptor("final", DatasetRole.FINAL_TEST, frozenset({"z"})),
    ]
    with pytest.raises(ValueError, match="leakage"):
        validate_dataset_isolation(descriptors)
