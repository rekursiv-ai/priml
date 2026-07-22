from torchao.utils import TorchAOBaseTensor

import torch

from .int4_choose_qparams_algorithm import Int4ChooseQParamsAlgorithm

__all__ = ["Int4OpaqueTensor"]
aten = ...

class Int4OpaqueTensor(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    optional_tensor_data_names = ...
    def __new__(
        cls,
        qdata,
        scale_and_zero,
        block_size,
        shape,
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
        w: torch.Tensor,
        block_size: list[int],
        int4_choose_qparams_algorithm: Int4ChooseQParamsAlgorithm = ...,
    ):  # -> Int4OpaqueTensor:
        ...

implements = ...

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):  # -> Any:
    ...
