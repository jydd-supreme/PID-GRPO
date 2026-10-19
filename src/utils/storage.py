"""Keep training artifacts and caches on the configured data disk."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping


def configure_runtime_storage(config: Mapping[str, Any]) -> dict[str, str]:
    storage = config["storage"]
    root = Path(storage["root_dir"]).expanduser().resolve()
    cache = Path(storage["cache_dir"]).expanduser().resolve()
    output = Path(config["project"]["output_dir"]).expanduser().resolve()
    data_paths = [Path(value).expanduser().resolve() for value in config["data"].values()]
    for name, path in [("cache_dir", cache), ("output_dir", output)]:
        if not path.is_relative_to(root):
            raise ValueError(f"{name} must be inside data-disk root {root}: {path}")
    for path in data_paths:
        if not path.is_relative_to(root):
            raise ValueError(f"data path must be inside data-disk root {root}: {path}")

    directories = {
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "HF_ASSETS_CACHE": cache / "huggingface" / "assets",
        "HF_DATASETS_CACHE": cache / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": cache / "huggingface" / "transformers",
        "TORCH_HOME": cache / "torch",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda",
        "PIP_CACHE_DIR": cache / "pip",
        "WANDB_DIR": output / "wandb",
        "TMPDIR": cache / "tmp",
    }
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    for name, path in directories.items():
        os.environ[name] = str(path)
    # tempfile caches its chosen directory, so update both the environment and
    # the module-level value before Transformers/Datasets are imported.
    tempfile.tempdir = str(directories["TMPDIR"])

    free_gb = shutil.disk_usage(root).free / (1024 ** 3)
    minimum = float(storage["min_free_gb"])
    if free_gb < minimum:
        raise RuntimeError(
            f"data disk has only {free_gb:.1f} GiB free; at least {minimum:.1f} GiB is required"
        )
    return {name: str(path) for name, path in directories.items()}
