from typing import Protocol, runtime_checkable

import torch

"""Protocols for some functionalities in tensor subclasses"""

@runtime_checkable
class SupportsActivationPreScaling(Protocol):
    act_pre_scale: torch.Tensor | None
