from torchao.quantization.quantize_.workflows.intx.intx_packing_format import (
    IntxPackingFormat,
)
from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
    IntxUnpackedToInt8Tensor,
)
from torchao.utils import TorchAOBaseTensor

import torch

__all__ = ["IntxOpaqueTensor"]
aten = ...

class IntxOpaqueTensor(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    def __new__(
        cls,
        packed_weights,
        bit_width,
        block_size,
        shape,
        dtype,
        packed_weights_has_zeros,
        packed_weights_has_bias,
        intx_packing_format,
    ):  # -> Self:
        ...
    def __init__(
        self,
        packed_weights,
        bit_width,
        block_size,
        shape,
        dtype,
        packed_weights_has_zeros,
        packed_weights_has_bias,
        intx_packing_format,
    ) -> None: ...
    def to(self, *args, **kwargs): ...
    @classmethod
    def from_intx_unpacked_to_int8_tensor(
        cls,
        tensor: IntxUnpackedToInt8Tensor,
        *,
        bias: torch.Tensor | None = ...,
        intx_packing_format: IntxPackingFormat = ...,
    ):  # -> Self:
        ...

implements = ...

@implements([torch.nn.functional.linear, aten.linear.default])
def _(func, types, args, kwargs):  # -> Any:
    ...
@implements([torch.nn.functional.embedding, aten.embedding.default])
def _(func, types, args, kwargs):  # -> Any:
    ...
