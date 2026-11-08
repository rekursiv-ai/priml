"""A host tensor that reports CUDA, for testing device handling without a GPU."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from torch import Tensor

import torch


if TYPE_CHECKING:
    import numpy as np


class OnCuda(Tensor):
    """A host tensor reporting ``cuda:0``, as a GPU-resident one would.

    Ops on it return ``OnCuda`` too, so a value derived from a CUDA input still
    reports CUDA. ``cpu()`` hands back a plain host tensor and ``numpy()``
    refuses unless forced to copy, as both do for CUDA memory. A test wraps its
    input in one to check that code moves a CUDA value to the host where it
    must; nothing runs on a GPU, so device-mismatch errors and kernels are out
    of its reach.
    """

    @property
    @override
    def device(self) -> torch.device:
        return torch.device("cuda", 0)

    @override
    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> Tensor:
        return self.as_subclass(Tensor).clone(memory_format=memory_format)

    @override
    def numpy(self, *, force: bool = False) -> np.ndarray:
        if force:
            return self.cpu().numpy()
        raise TypeError(
            "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to "
            "copy the tensor to host memory first.",
        )
