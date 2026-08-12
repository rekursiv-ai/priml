from dataclasses import dataclass

from torchao.prototype.mx_formats.config import MXGemmKernelChoice, ScaleCalculationMode
from torchao.quantization.quantize_.common import QuantizeTensorKwargs
from torchao.utils import TorchAOBaseTensor

import torch

"""
Defines the tensor subclasses to represent the MX format spec from
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

Exponent E8M0 encoding details (OCP spec section 5.4.1):
  * bias: 127
  * supported exponent range: -127 to 127
  * infinities: N/A
  * NaN: 11111111
  * Zeros: N/A
"""
aten = ...

@dataclass
class QuantizeTensorToMXKwargs(QuantizeTensorKwargs):
    elem_dtype: torch.dtype | str = ...
    block_size: int = ...
    scaling_mode: ScaleCalculationMode = ...
    gemm_kernel_choice: MXGemmKernelChoice = ...
    pack_fp6: bool = ...

def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: torch.dtype | str,
    block_size: int,
    scaling_mode: ScaleCalculationMode = ...,
    pack_fp6: bool = ...,
):  # -> tuple[Tensor, Tensor | Any]:
    ...
def get_fp_scale(scale_e8m0):  # -> Tensor:
    ...
def to_dtype(
    data_lp, scale_e8m0, elem_dtype, block_size, target_dtype, pack_fp6: bool = ...
):  # -> Tensor:
    ...
def tensor_size_hp_to_fp4x2(orig_size, is_contiguous):  # -> list[Any]:
    ...
def tensor_size_fp4x2_to_hp(orig_size, is_contiguous):  # -> list[Any]:
    ...
def tensor_size_hpx3_to_fp6x4(orig_size, is_contiguous):  # -> list[Any]:
    ...
def tensor_size_fp6x4_to_hpx3(orig_size, is_contiguous):  # -> list[Any]:
    ...

class MXTensor(TorchAOBaseTensor):
    tensor_data_names = ...
    tensor_attribute_names = ...
    def __new__(
        cls,
        qdata,
        scale_e8m0_bits,
        elem_dtype,
        block_size,
        orig_dtype,
        gemm_kernel_choice,
        pack_fp6,
        act_quant_kwargs,
    ):  # -> Self:
        ...
    def __repr__(self):  # -> str:
        ...
    def to_dtype(self, target_dtype):  # -> Tensor:
        ...
    @staticmethod
    @torch._dynamo.allow_in_graph
    def to_mx(
        data_hp: torch.Tensor,
        elem_dtype: torch.dtype | str,
        block_size: int = ...,
        scaling_mode: ScaleCalculationMode = ...,
        gemm_kernel_choice: MXGemmKernelChoice = ...,
        pack_fp6: bool = ...,
        act_quant_kwargs: QuantizeTensorToMXKwargs | None = ...,
    ):  # -> DTensor | MXTensor:
        ...

implements = ...

@implements([aten.detach.default, aten.alias.default])
def _(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements([aten.mm.default, aten.matmul.default])
def mx_mm(func, types, args, kwargs):  # -> Tensor:
    ...
@implements([aten.addmm.default])
def mx_addmm(func, types, args, kwargs):  # -> Tensor:
    ...
@implements([aten.t.default])
def mx_t(func, types, args, kwargs):  # -> MXTensor:
    ...
@implements([aten.sum.dim_IntList])
def mx_cast_up_op(func, types, args, kwargs): ...
@implements([aten.view.default])
def mx_view_op(func, types, args, kwargs):  # -> MXTensor:
    ...
@implements([aten.slice.Tensor])
def mx_slice(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements([aten.clone.default])
def mx_clone(func, types, args, kwargs): ...
@implements([aten.select.int])
def mx_select(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
