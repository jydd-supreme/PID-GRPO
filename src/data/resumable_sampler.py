"""The pinned TRL permutation sequence, reconstructible from seed and epoch."""

from __future__ import annotations


class ResumableRepeatSampler:
    def __init__(self, data_source, repeat_count: int, seed: int) -> None:
        self.size = len(data_source)
        self.repeat_count = repeat_count
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self):
        return self.size * self.repeat_count

    def __iter__(self):
        import torch
        generator = torch.Generator().manual_seed(self.seed)
        # Replaying earlier permutations reproduces RepeatRandomSampler exactly,
        # without depending on a generator hidden inside a previous process.
        for _ in range(self.epoch + 1):
            indices = torch.randperm(self.size, generator=generator).tolist()
        return iter([index for index in indices for _ in range(self.repeat_count)])
