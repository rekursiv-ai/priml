"""Device housekeeping that only a CUDA placement needs."""

from __future__ import annotations

import gc

import torch


def synchronize(device: torch.device) -> None:
    """Wait for queued work on a CUDA device; any other device has nothing queued."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def release_memory(device: torch.device) -> None:
    """Collect garbage and return cached CUDA blocks after a stage on ``device``."""
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
