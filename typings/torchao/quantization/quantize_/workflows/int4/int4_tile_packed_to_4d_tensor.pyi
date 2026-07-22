from torchao.utils import TorchAOBaseTensor

import torch

from .int4_choose_qparams_algorithm import Int4ChooseQParamsAlgorithm

__all__ = ["Int4TilePackedTo4dTensor"]
aten = ...

class Int4TilePackedTo4dTensor(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    optional_tensor_data_names = ...
    def __new__(
        cls,
        qdata: torch.Tensor,
        scale_and_zero: torch.Tensor,
        block_size: list[int],
        shape: torch.Size,
        act_pre_scale: torch.Tensor | None = ...,
    ):  # -> Self:
        ...
    def __init__(
        self,
        qdata: torch.Tensor,
        scale_and_zero: torch.Tensor,
        block_size: list[int],
        shape: torch.Size,
        act_pre_scale: torch.Tensor | None = ...,
    ) -> None: ...
    @classmethod
    def from_hp(
        cls,
        hp_tensor: torch.Tensor,
        block_size: list[int],
        int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = ...,
    ):  # -> Int4TilePackedTo4dTensor:
        ...

implements = ...

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):  # -> Tensor | Any:
    ...
@implements(aten.slice.Tensor)
def _(func, _types, args, _kwargs):  # -> Int4TilePackedTo4dTensor:
    ...
