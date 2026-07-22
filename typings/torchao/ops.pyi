import functools

from torch import Tensor

import torch

lib = ...

def register_custom_op(name):  # -> Callable[..., Any | Callable[..., Any]]:
    ...
def register_custom_op_impl(name):  # -> Callable[..., CustomOpDef]:
    ...
@functools.lru_cache
def cached_compute_capability(): ...
def quant_llm_linear(
    EXPONENT: int,
    MANTISSA: int,
    _in_feats: Tensor,
    _weights: Tensor,
    _scales: Tensor,
    splitK: int = ...,
) -> Tensor: ...
@register_custom_op("torchao::quant_llm_linear")
def _(
    EXPONENT: int,
    MANTISSA: int,
    _in_feats: Tensor,
    _weights: Tensor,
    _scales: Tensor,
    splitK: int = ...,
) -> Tensor: ...
def qscaled_dot_product(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = ...,
    dropout_p: float = ...,
    is_causal: bool = ...,
    scale: float | None = ...,
    q_scale: float = ...,
    q_zp: int = ...,
    k_scale: float = ...,
    k_zp: int = ...,
    v_scale: float = ...,
    v_zp: int = ...,
    a_scale: float = ...,
    a_zp: int = ...,
    o_scale: float = ...,
    o_zp: int = ...,
) -> Tensor: ...
@register_custom_op("torchao::qscaled_dot_product")
def _(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = ...,
    dropout_p: float = ...,
    is_causal: bool = ...,
    scale: float | None = ...,
    q_scale: float = ...,
    q_zp: int = ...,
    k_scale: float = ...,
    k_zp: int = ...,
    v_scale: float = ...,
    v_zp: int = ...,
    a_scale: float = ...,
    a_zp: int = ...,
    o_scale: float = ...,
    o_zp: int = ...,
) -> Tensor: ...
def unpack_tensor_core_tiled_layout(packed_w: Tensor, inner_k_tiles: int) -> Tensor: ...
@register_custom_op("torchao::unpack_tensor_core_tiled_layout")
def _(packed_w: Tensor, inner_k_tiles: int) -> Tensor: ...
def dequantize_tensor_core_tiled_layout(
    packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int
) -> Tensor: ...
@register_custom_op("torchao::dequantize_tensor_core_tiled_layout")
def _(
    packed_w: Tensor, scales_and_zeros: Tensor, group_size: int, inner_k_tiles: int
) -> Tensor: ...
def marlin_24_gemm(
    x: Tensor,
    weight_marlin: Tensor,
    meta: Tensor,
    s: Tensor,
    workspace: Tensor,
    bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor: ...
@register_custom_op("torchao::marlin_24_gemm")
def _(
    x: Tensor,
    weight_marlin: Tensor,
    meta: Tensor,
    s: Tensor,
    workspace: Tensor,
    bits: int,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor: ...
def marlin_qqq_gemm(
    x: Tensor,
    weight_marlin: Tensor,
    s_tok: Tensor,
    s_ch: Tensor,
    s_group: Tensor,
    workspace: Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor: ...
@register_custom_op("torchao::marlin_qqq_gemm")
def _(
    x: Tensor,
    weight_marlin: Tensor,
    s_tok: Tensor,
    s_ch: Tensor,
    s_group: Tensor,
    workspace: Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> Tensor: ...
def rowwise_scaled_linear_cutlass_s8s4(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
) -> Tensor: ...
@register_custom_op("torchao::rowwise_scaled_linear_cutlass_s8s4")
def _(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
) -> Tensor: ...
def rowwise_scaled_linear_cutlass_s4s4(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
) -> Tensor: ...
@register_custom_op("torchao::rowwise_scaled_linear_cutlass_s4s4")
def _(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
) -> Tensor: ...
def rowwise_scaled_linear_sparse_cutlass_f8f8(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_meta: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
) -> Tensor: ...
@register_custom_op("torchao::rowwise_scaled_linear_sparse_cutlass_f8f8")
def _(
    input: Tensor,
    input_scale: Tensor,
    weight: Tensor,
    weight_meta: Tensor,
    weight_scale: Tensor,
    bias: Tensor | None = ...,
    out_dtype: torch.dtype | None = ...,
) -> Tensor: ...
def to_sparse_semi_structured_cutlass_sm9x_f8(weight: Tensor) -> (Tensor, Tensor): ...
@register_custom_op("torchao::to_sparse_semi_structured_cutlass_sm9x_f8")
def _(weight: Tensor) -> (Tensor, Tensor): ...
def sparse24_sm90_sparsify(
    input_tensor: Tensor,
    metadata_format: str,
    activation: str,
    algorithm: str,
    dtype=...,
    scale=...,
) -> (Tensor, Tensor): ...
@register_custom_op("torchao::sparse24_sm90_sparsify")
def _(
    input_tensor: Tensor,
    metadata_format: str,
    activation: str,
    algorithm: str,
    dtype=...,
    scale=...,
):  # -> tuple[Tensor, Tensor]:
    ...
def sparse24_fp8_sm90_cutlass_gemm(
    a: Tensor,
    meta: Tensor,
    b: Tensor,
    a_scale: Tensor | None = ...,
    b_scale: Tensor | None = ...,
    swizzle_size: int = ...,
    swizzle_axis: str = ...,
    sm_count: int = ...,
) -> Tensor: ...
@register_custom_op("torchao::sparse24_fp8_sm90_cutlass_gemm")
def _(
    a: Tensor,
    meta: Tensor,
    b: Tensor,
    a_scale: Tensor | None = ...,
    b_scale: Tensor | None = ...,
    swizzle_size: int = ...,
    swizzle_axis: str = ...,
    sm_count: int = ...,
):  # -> Tensor:
    ...
def swizzle_mm(
    mat1: Tensor, mat2: Tensor, mat1_is_swizzled: bool, mat2_is_swizzled: bool
) -> Tensor: ...
@register_custom_op("torchao::swizzle_mm")
def _(
    mat1: Tensor, mat2: Tensor, mat1_is_swizzled: bool, mat2_is_swizzled: bool
) -> Tensor: ...
def swizzle_scaled_mm(
    mat1: Tensor,
    mat2: Tensor,
    mat1_is_swizzled: bool,
    mat2_is_swizzled: bool,
    scale_a: Tensor,
    scale_b: Tensor,
    bias: Tensor | None,
    scale_result: Tensor | None,
    out_dtype: torch.dtype | None,
) -> Tensor: ...
@register_custom_op("torchao::swizzle_scaled_mm")
def _(
    mat1: Tensor,
    mat2: Tensor,
    mat1_is_swizzled: bool,
    mat2_is_swizzled: bool,
    scale_a: Tensor,
    scale_b: Tensor,
    bias: Tensor | None,
    scale_result: Tensor | None,
    out_dtype: torch.dtype | None,
) -> Tensor: ...
@register_custom_op("torchao::mx_fp8_bf16")
def meta_mx_fp8_bf16(
    A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor
):  # -> Tensor:
    ...
def mx_fp4_bf16(A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor):  # -> Any:
    ...
@register_custom_op("torchao::mx_fp4_bf16")
def meta_mx_fp4_bf16(
    A: Tensor, B: Tensor, A_scale: Tensor, B_scale: Tensor
):  # -> Tensor:
    ...
def da8w4_linear_prepack_cpu(
    weight: Tensor, scales: Tensor, qzeros: Tensor
) -> Tensor: ...
@register_custom_op("torchao::da8w4_linear_prepack_cpu")
def _(weight: Tensor, scales: Tensor, qzeros: Tensor) -> Tensor: ...
def da8w4_linear_cpu(
    input: Tensor,
    input_scales: Tensor,
    input_qzeros: Tensor,
    weight: Tensor,
    weight_scales: Tensor,
    weight_qzeros: Tensor,
    compensation: Tensor,
    bias: Tensor | None,
    out_dtype: torch.dtype,
):  # -> Any:
    ...
@register_custom_op("torchao::da8w4_linear_cpu")
def _(
    input: Tensor,
    input_scales: Tensor,
    input_qzeros: Tensor,
    weight: Tensor,
    weight_scales: Tensor,
    weight_qzeros: Tensor,
    compensation: Tensor,
    bias: Tensor | None,
    out_dtype: torch.dtype,
) -> Tensor: ...
@register_custom_op("torchao::_scaled_embedding_bag")
def _(
    qweight: Tensor,
    indices: Tensor,
    offsets: Tensor,
    w_scales: Tensor,
    o_scale: float,
    mode: int,
    include_last_offset: bool,
) -> Tensor: ...
