from torchao.utils import TorchAOBaseTensor

import torch

__all__ = ["Int4Tensor"]
aten = ...

class Int4Tensor(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    optional_tensor_data_names = ...
    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: list[int],
        shape: torch.Size,
        act_pre_scale: torch.Tensor | None = ...,
    ):  # -> Self:
        ...
    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: list[int],
        shape: torch.Size,
        act_pre_scale: torch.Tensor | None = ...,
    ) -> None: ...
    @classmethod
    def from_hp(cls, w: torch.Tensor, block_size: list[int]):  # -> Int4Tensor:
        ...

implements = ...

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):  # -> Any:
    ...
@implements(torch.bmm)
def _(func, types, args, kwargs):  # -> Any:
    ...
@implements(aten.slice.Tensor)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.cat.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.transpose.int)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.view.default)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.squeeze.dim)
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
