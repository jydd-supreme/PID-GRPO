"""Best-effort deterministic seeding without requiring heavy dependencies."""

from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
