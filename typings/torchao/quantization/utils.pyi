from torch.utils._python_dispatch import TorchDispatchMode

import torch

from .granularity import Granularity

__all__ = [
    "_quant_int8_dynamic_per_token_linear",
    "_quantize_activation_per_token_absmax",
    "compute_error",
    "dequantize_per_channel",
    "dequantize_per_tensor",
    "dynamically_quantize_per_channel",
    "get_group_qparams_symmetric",
    "get_groupwise_affine_qparams",
    "groupwise_affine_dequantize_tensor",
    "groupwise_affine_dequantize_tensor_from_qparams",
    "groupwise_affine_quantize_tensor",
    "groupwise_affine_quantize_tensor_from_qparams",
    "pack_tinygemm_scales_and_zeros",
    "per_token_dynamic_quant",
    "recommended_inductor_config_setter",
    "unpack_tinygemm_scales_and_zeros",
]

def compute_error(x, y):  # -> Tensor:
    ...

_cur_fqn: str | None = ...
_fqn_to_op_to_shape_to_count: dict[
    str | None, dict[str | None, dict[str | None, int]]
] = ...

class LoggingTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=..., kwargs=...): ...

class _MultiInput:
    def __init__(self, inputs) -> None: ...
    def add_input(self, input):  # -> Self:
        ...
    def __getitem__(self, slice):  # -> _MultiInput:
        ...
    def cuda(self):  # -> None:
        ...
    def xpu(self):  # -> None:
        ...

def dynamically_quantize_per_channel(
    x, quant_min, quant_max, target_dtype
):  # -> tuple[Tensor, Tensor, Tensor]:
    ...
def dequantize_per_tensor(int_repr, scale, zero_point, out_dtype=...):  # -> Tensor:
    ...
def dequantize_per_channel(int_repr, scales, zero_points, out_dtype=...):  # -> Tensor:
    ...
def get_groupwise_affine_qparams(
    w,
    n_bit=...,
    groupsize=...,
    dtype=...,
    zero_point_domain=...,
    preserve_zero=...,
    eps=...,
):  # -> tuple[Tensor, Tensor]:
    ...
def pack_tinygemm_scales_and_zeros(scales, zeros, dtype=...):  # -> Tensor:
    ...
def unpack_tinygemm_scales_and_zeros(scales_and_zeros):  # -> tuple[Tensor, ...]:
    ...
def groupwise_affine_quantize_tensor_from_qparams(
    w, scales, zeros, n_bit=..., groupsize=..., zero_point_domain=...
):  # -> Tensor:
    ...
def groupwise_affine_dequantize_tensor_from_qparams(
    w_int4x8, scales, zeros, n_bit=..., groupsize=..., zero_point_domain=...
):  # -> Tensor:
    ...
def groupwise_affine_quantize_tensor(
    w, n_bit=..., groupsize=..., dtype=..., zero_point_domain=..., preserve_zero=...
):  # -> tuple[Tensor, Tensor]:
    ...
def groupwise_affine_dequantize_tensor(
    w_int4x8, scales_and_zeros, n_bit=..., groupsize=...
):  # -> Tensor:
    ...
def get_group_qparams_symmetric(
    w, n_bit=..., groupsize=..., precision=..., mapping_type=..., eps=...
):  # -> tuple[Tensor, Tensor]:
    ...
def group_quantize_tensor_symmetric(
    w, n_bit=..., group_size=..., precision=..., mapping_type=...
):  # -> tuple[Any, Tensor, Tensor]:
    ...
def per_token_dynamic_quant(
    input: torch.Tensor,
    scale_dtype: torch.dtype = ...,
    zero_point_dtype: torch.dtype = ...,
    eps: float | None = ...,
) -> torch.Tensor: ...
def recommended_inductor_config_setter():  # -> None:
    ...
def get_block_size(
    input_shape: tuple[int, ...], granularity: Granularity
) -> tuple[int, ...]: ...
