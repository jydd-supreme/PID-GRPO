import os
from pathlib import Path
import tempfile

from src.utils.storage import configure_runtime_storage


def test_runtime_storage_routes_caches_to_data_root(tmp_path: Path) -> None:
    old_tempdir = tempfile.tempdir
    names = (
        "HF_HOME", "HF_HUB_CACHE", "HF_ASSETS_CACHE", "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
        "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
        "PIP_CACHE_DIR", "WANDB_DIR", "TMPDIR",
    )
    old_environment = {name: os.environ.get(name) for name in names}
    config = {
        "project": {"output_dir": str(tmp_path / "outputs")},
        "storage": {
            "root_dir": str(tmp_path),
            "cache_dir": str(tmp_path / ".cache"),
            "min_free_gb": 0,
        },
        "data": {
            "train_file": str(tmp_path / "data" / "train.jsonl"),
            "control_file": str(tmp_path / "data" / "control.jsonl"),
        },
    }
    try:
        paths = configure_runtime_storage(config)
        assert Path(paths["HF_HOME"]).is_dir()
        assert Path(paths["TMPDIR"]).is_dir()
        assert Path(os.environ["TORCH_HOME"]).is_relative_to(tmp_path)
        assert Path(tempfile.gettempdir()).is_relative_to(tmp_path)
    finally:
        tempfile.tempdir = old_tempdir
        for name, value in old_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_runtime_storage_rejects_system_disk_path(tmp_path: Path) -> None:
    config = {
        "project": {"output_dir": "/var/tmp/pid-grpo-output"},
        "storage": {
            "root_dir": str(tmp_path),
            "cache_dir": str(tmp_path / ".cache"),
            "min_free_gb": 0,
        },
        "data": {"train_file": str(tmp_path / "train.jsonl")},
    }
    try:
        configure_runtime_storage(config)
    except ValueError as exc:
        assert "inside data-disk root" in str(exc)
    else:
        raise AssertionError("system-disk output path was accepted")
