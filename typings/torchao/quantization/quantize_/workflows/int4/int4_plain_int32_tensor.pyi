from torchao.utils import TorchAOBaseTensor

import torch

__all__ = ["Int4PlainInt32Tensor"]
aten = ...

class Int4PlainInt32Tensor(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    optional_tensor_data_names = ...
    def __new__(
        cls,
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
        act_pre_scale: torch.Tensor | None = ...,
    ):  # -> Self:
        ...
    def __init__(
        self,
        qdata,
        scale,
        zero_point,
        block_size,
        shape,
        act_pre_scale: torch.Tensor | None = ...,
    ) -> None: ...
    @classmethod
    def from_hp(
        cls, w: torch.Tensor, block_size: list[int]
    ):  # -> Int4PlainInt32Tensor:
        ...

implements = ...

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):  # -> Any:
    ...
