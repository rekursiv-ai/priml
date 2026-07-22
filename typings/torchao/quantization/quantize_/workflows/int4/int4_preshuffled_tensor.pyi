import importlib.util

from fbgemm_gpu.experimental.gen_ai.quantize import (
    quantize_fp8_row,
    quantize_int4_preshuffle,
)
from torchao.quantization.quantize_.workflows.int4.int4_tensor import Int4Tensor
from torchao.utils import TorchAOBaseTensor

import torch

__all__ = ["Int4PreshuffledTensor"]
aten = ...
if (
    importlib.util.find_spec("fbgemm_gpu") is None
    or importlib.util.find_spec("fbgemm_gpu.experimental") is None
):
    quantize_int4_preshuffle = ...
    quantize_fp8_row = ...
    pack_int4 = ...
else: ...

class Int4PreshuffledTensor(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    optional_tensor_data_names = ...
    def __new__(
        cls,
        qdata: torch.Tensor,
        group_scale: torch.Tensor,
        block_size: list[int],
        shape: list[int],
        group_zero: torch.Tensor | None = ...,
        row_scale: torch.Tensor | None = ...,
    ):  # -> Self:
        ...
    def __init__(
        self,
        qdata: torch.Tensor,
        group_scale: torch.Tensor,
        block_size: list[int],
        shape: list[int],
        group_zero: torch.Tensor | None = ...,
        row_scale: torch.Tensor | None = ...,
    ) -> None: ...
    @classmethod
    def from_hp(
        cls, w: torch.Tensor, block_size: list[int], activation_dtype: torch.dtype = ...
    ):  # -> Int4PreshuffledTensor:
        ...
    @classmethod
    def from_int4_tensor(cls, tensor: Int4Tensor):  # -> Int4PreshuffledTensor:
        ...

implements = ...

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):  # -> Any:
    ...
@implements(torch.bmm)
def _(func, types, args, kwargs):  # -> Tensor | Any:
    ...
