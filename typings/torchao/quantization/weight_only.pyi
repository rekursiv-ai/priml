import torch

__all__ = ["WeightOnlyInt8QuantLinear"]

class WeightOnlyInt8QuantLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor: ...
    @classmethod
    def from_float(cls, mod: torch.nn.Linear):  # -> Self:
        ...
