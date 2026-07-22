from enum import Enum

import torch

__all__ = [
    "MappingType",
    "TorchAODType",
    "ZeroPointDomain",
    "_choose_qparams_affine_dont_preserve_zero",
    "_choose_qparams_affine_floatx",
    "_choose_qparams_affine_tinygemm",
    "_choose_qparams_and_quantize_affine_hqq",
    "_choose_qparams_and_quantize_affine_qqq",
    "_choose_qparams_and_quantize_scale_only_hqq",
    "_choose_qparams_gguf",
    "_choose_scale_float8",
    "_dequantize_affine_float8",
    "_dequantize_affine_floatx",
    "_dequantize_affine_no_zero_point",
    "_dequantize_affine_qqq",
    "_dequantize_affine_tinygemm",
    "_dequantize_gguf",
    "_fake_quantize_affine",
    "_fake_quantize_affine_cachemask",
    "_quantize_affine_float8",
    "_quantize_affine_floatx",
    "_quantize_affine_no_zero_point",
    "_quantize_affine_tinygemm",
    "_quantize_gguf",
    "choose_qparams_affine",
    "choose_qparams_affine_with_min_max",
    "dequantize_affine",
    "quantize_affine",
]

class MappingType(Enum):
    SYMMETRIC = ...
    SYMMETRIC_NO_CLIPPING_ERR = ...
    ASYMMETRIC = ...

class ZeroPointDomain(Enum):
    INT = ...
    FLOAT = ...
    NONE = ...

class TorchAODType(Enum):
    INT1 = ...
    INT2 = ...
    INT3 = ...
    INT4 = ...
    INT5 = ...
    INT6 = ...
    INT7 = ...

FP8_TYPES = ...
_DTYPE_TO_QVALUE_BOUNDS: dict[torch.dtype | TorchAODType, tuple[int, int]] = ...
_DTYPE_TO_BIT_WIDTH: dict[torch.dtype | TorchAODType, tuple[int, int]] = ...
_SUB_BYTE_UINT_BOUNDS: dict[torch.dtype | TorchAODType, tuple[int, int]] = ...
_SUB_BYTE_INT_BOUNDS: dict[torch.dtype | TorchAODType, tuple[int, int]] = ...
_SUB_BYTE_UINT_BOUNDS = ...
_GGUF_QK_K = ...
_ONES_TABLE = ...
quant_lib = ...
register_custom_op = ...

class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor: ...
    @staticmethod
    def backward(ctx, gy: torch.Tensor) -> torch.Tensor: ...

class _RoundToFloat8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, float8_dtype: torch.dtype) -> torch.Tensor: ...
    @staticmethod
    def backward(ctx, gy: torch.Tensor) -> torch.Tensor: ...

@torch.no_grad()
def quantize_affine(
    input: torch.Tensor,
    block_size: tuple[int, ...],
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    output_dtype: torch.dtype,
    quant_min: float | None = ...,
    quant_max: float | None = ...,
) -> torch.Tensor: ...
def dequantize_affine(
    input: torch.Tensor,
    block_size: tuple[int, ...],
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    input_dtype: torch.dtype,
    quant_min: float | None = ...,
    quant_max: float | None = ...,
    *,
    output_dtype: torch.dtype = ...,
) -> torch.Tensor: ...
@torch.no_grad()
def choose_qparams_affine(
    input: torch.Tensor,
    mapping_type: MappingType,
    block_size: tuple[int],
    target_dtype: torch.dtype,
    quant_min: float | None = ...,
    quant_max: float | None = ...,
    eps: float | None = ...,
    scale_dtype: torch.dtype | None = ...,
    zero_point_dtype: torch.dtype | None = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def choose_qparams_affine_with_min_max(
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    mapping_type: MappingType,
    block_size: tuple[int, ...],
    target_dtype: torch.dtype,
    quant_min: int | None = ...,
    quant_max: int | None = ...,
    eps: float | None = ...,
    scale_dtype: torch.dtype | None = ...,
    zero_point_dtype: torch.dtype | None = ...,
    preserve_zero: bool = ...,
    zero_point_domain: ZeroPointDomain = ...,
) -> tuple[torch.Tensor, torch.Tensor]: ...
@torch.inference_mode()
def optimize_weights_proximal_legacy(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    min_max: list,
    axis: int = ...,
    dtype: torch.dtype | None = ...,
    device: str | None = ...,
    verbose: bool = ...,
    opt_params: dict = ...,
) -> tuple: ...
