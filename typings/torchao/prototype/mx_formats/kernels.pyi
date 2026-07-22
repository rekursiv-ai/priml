from torch.distributed.tensor.experimental import register_sharding
from torch.library import triton_op
from torch.utils._triton import has_triton
from torchao.utils import is_sm_at_least_100, torch_version_at_least

import torch
import triton
import triton.language as tl

logger = ...

def get_bits(x: torch.Tensor) -> str: ...

SIGN_MASK_F4 = ...
MANTISSA_MASK_F4 = ...
SIGN_MASK_F6_E2M3 = ...
MANTISSA_MASK_F6_E2M3 = ...
SIGN_MASK_F6_E3M2 = ...
MANTISSA_MASK_F6_E3M2 = ...
ZERO_BITS_F32 = ...
ZERO_POINT_FIVE_BITS_F32 = ...

def f32_to_f4_unpacked(x):  # -> Tensor:
    ...
def f32_to_f6_e2m3_unpacked(x):  # -> Tensor:
    ...
def f32_to_f6_e3m2_unpacked(x):  # -> Tensor:
    ...
def f4_unpacked_to_f32(x: torch.Tensor):  # -> Tensor:
    ...
def f6_e2m3_unpacked_to_f32(x: torch.Tensor):  # -> Tensor:
    ...
def f6_e3m2_unpacked_to_f32(x: torch.Tensor):  # -> Tensor:
    ...

if has_triton():
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_IN": 2}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 4}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 8}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 16}, num_warps=1),
        ],
        key=["n_mx_blocks"],
    )
    @triton.jit
    def triton_f6_to_bf16_kernel(
        x_ptr,
        output_ptr,
        n_mx_blocks,
        mx_block_size: tl.constexpr,
        packed_mx_block_size: tl.constexpr,
        sign_mask_f6: tl.constexpr,
        mbits_f6: tl.constexpr,
        f6_exp_bias: tl.constexpr,
        mbits_f32: tl.constexpr,
        f32_exp_bias: tl.constexpr,
        BLOCK_SIZE_IN: tl.constexpr,
    ): ...
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_IN": 2}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 4}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 8}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 16}, num_warps=1),
        ],
        key=["n_mx_blocks"],
    )
    @triton.jit
    def triton_f6_to_scaled_bf16_kernel(
        x_ptr,
        s_ptr,
        output_ptr,
        n_mx_blocks,
        mx_block_size: tl.constexpr,
        packed_mx_block_size: tl.constexpr,
        sign_mask_f6: tl.constexpr,
        mbits_f6: tl.constexpr,
        f6_exp_bias: tl.constexpr,
        mbits_f32: tl.constexpr,
        f32_exp_bias: tl.constexpr,
        e8m0_exponent_bias: tl.constexpr,
        e8m0_exponent_nan_val: tl.constexpr,
        BLOCK_SIZE_IN: tl.constexpr,
    ): ...
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_IN": 2}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 4}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 8}, num_warps=1),
            triton.Config({"BLOCK_SIZE_IN": 16}, num_warps=1),
        ],
        key=["n_mx_blocks"],
    )
    @triton.jit
    def triton_pack_uint6_kernel(
        input_ptr,
        output_ptr,
        n_mx_blocks,
        MX_BLOCK_SIZE: tl.constexpr,
        PACKED_MX_BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE_IN: tl.constexpr,
    ):  # -> None:
        ...

else:
    def triton_f6_to_bf16_kernel(
        x_ptr,
        output_ptr,
        n_elements_in,
        sign_mask_f6,
        mbits_f6,
        f6_exp_bias,
        mbits_f32,
        f32_exp_bias,
        BLOCK_SIZE_IN,
    ): ...
    def triton_f6_to_scaled_bf16_kernel(
        x_ptr,
        s_ptr,
        output_ptr,
        n_elements_in,
        mx_block_size,
        sign_mask_f6,
        mbits_f6,
        f6_exp_bias,
        mbits_f32,
        f32_exp_bias,
        e8m0_exponent_bias,
        e8m0_exponent_nan_val,
        BLOCK_SIZE_IN,
    ): ...
    def triton_pack_uint6_kernel(
        input_ptr,
        output_ptr,
        n_mx_blocks,
        MX_BLOCK_SIZE,
        PACKED_MX_BLOCK_SIZE,
        BLOCK_SIZE,
    ): ...

def triton_f6_e2m3_to_bf16(x: torch.Tensor) -> torch.Tensor: ...
def triton_f6_e3m2_to_bf16(x: torch.Tensor) -> torch.Tensor: ...
@torch.library.custom_op("ao::triton_f6_e2m3_to_scaled_bf16", mutates_args=())
def triton_f6_e2m3_to_scaled_bf16(
    x: torch.Tensor, s_e8m0: torch.Tensor, mx_block_size: int
) -> torch.Tensor: ...
@torch.library.custom_op("ao::triton_f6_e3m2_to_scaled_bf16", mutates_args=())
def triton_f6_e3m2_to_scaled_bf16(
    x: torch.Tensor, s_e8m0: torch.Tensor, mx_block_size: int
) -> torch.Tensor: ...
@triton_f6_e3m2_to_scaled_bf16.register_fake
def _(x, s_e8m0, mx_block_size):  # -> Tensor:
    ...
@triton_f6_e2m3_to_scaled_bf16.register_fake
def _(x, s_e8m0, mx_block_size):  # -> Tensor:
    ...
def down_size(size):  # -> tuple[*tuple[Any, ...], Any]:
    ...
def up_size(size):  # -> tuple[*tuple[Any, ...], Any]:
    ...
def unpack_uint4(uint8_data) -> torch.Tensor: ...
def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor: ...
def pack_uint6_pytorch(uint8_data: torch.Tensor) -> torch.Tensor: ...
@torch.library.custom_op("ao::pack_uint6", mutates_args=())
def pack_uint6(uint8_data: torch.Tensor) -> torch.Tensor: ...
@pack_uint6.register_fake
def _(uint8_data):  # -> Tensor:
    ...

if torch_version_at_least("2.7.0") and has_triton():
    @triton.autotune(
        configs=_get_mxfp8_dim1_kernel_autotune_configs(),
        key=["n_rows", "n_cols", "INNER_BLOCK_SIZE"],
    )
    @triton.jit
    def to_mxfp8_dim1_kernel(
        x_ptr,
        output_col_major_ptr,
        col_scale_ptr,
        n_rows,
        n_cols,
        ROW_TILE_SIZE: tl.constexpr,
        COL_TILE_SIZE: tl.constexpr,
        INNER_BLOCK_SIZE: tl.constexpr,
    ): ...
    @triton_op("torchao::triton_to_mxfp8_dim1", mutates_args={})
    def triton_to_mxfp8_dim1(
        x: torch.Tensor, inner_block_size: int = ...
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @register_sharding(torch.ops.torchao.triton_to_mxfp8_dim1.default)
    def custom_triton_to_mxfp8_dim1_sharding(
        x, inner_block_size=...
    ):  # -> list[tuple[list[Replicate], list[Replicate | None]] | tuple[list[Shard], list[Shard | None]]]:
        ...
    def triton_to_mxfp8_dim1_reference(
        x_hp: torch.Tensor, block_size
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @triton.jit
    def triton_scale_swizzle(
        scale_ptr,
        scale_rows,
        scale_cols,
        output_ptr,
        input_row_stride,
        input_col_stride,
        output_block_stride,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):  # -> None:
        ...
    @torch.library.custom_op("torchao::triton_mx_block_rearrange", mutates_args=())
    def triton_mx_block_rearrange(scale_tensor: torch.Tensor) -> torch.Tensor: ...
    @triton.jit
    def convert_fp32_to_fp4_packed(x_pairs):  # -> tensor | tuple:
        ...
    @triton.jit
    def quantize_nvfp4_triton_kernel(
        x_ptr,
        tensor_scale_ptr,
        q_ptr,
        s_ptr,
        stride_xm,
        stride_xn,
        M,
        N,
        USE_TENSOR_SCALE: tl.constexpr,
        MASK_SCALES: tl.constexpr,
    ): ...
    @torch.library.custom_op("ao::triton_quantize_nvfp4", mutates_args=())
    def triton_quantize_nvfp4(
        x: torch.Tensor, per_tensor_scale: torch.Tensor | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @triton_quantize_nvfp4.register_fake
    def _(x, per_tensor_scale=...):  # -> tuple[Tensor, Tensor]:
        ...
    @triton_mx_block_rearrange.register_fake
    def _(scale_tensor): ...

else:
    def triton_to_mxfp8_dim1(
        x, inner_block_size=...
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def triton_to_mxfp8_dim1_reference(
        x_hp: torch.Tensor, block_size
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def triton_mx_block_rearrange(scale_tensor: torch.Tensor) -> torch.Tensor: ...
    def triton_quantize_nvfp4(
        x: torch.Tensor, tensor_scale: torch.Tensor | None = ...
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

mxfp8_cuda_extension_available = ...
if is_sm_at_least_100():
    mxfp8_cuda_extension_available = ...
if mxfp8_cuda_extension_available:
    @torch.library.custom_op("torchao::mxfp8_quantize_cuda", mutates_args=())
    def mxfp8_quantize_cuda(
        x: torch.Tensor,
        rowwise: bool = ...,
        colwise: bool = ...,
        scaling_mode: str = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @mxfp8_quantize_cuda.register_fake
    def _(
        x: torch.Tensor,
        rowwise: bool = ...,
        colwise: bool = ...,
        scaling_mode: str = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
    @register_sharding(torch.ops.torchao.mxfp8_quantize_cuda.default)
    def custom_mxfp8_quantize_cuda_dim1_sharding(
        x: torch.Tensor,
        rowwise: bool = ...,
        colwise: bool = ...,
        scaling_mode: str = ...,
    ):  # -> list[tuple[list[Replicate | None], list[Replicate | None]] | tuple[list[Shard | None], list[Shard | None]]]:
        ...

else:
    def mxfp8_quantize_cuda(
        x: torch.Tensor,
        rowwise: bool = ...,
        colwise: bool = ...,
        scaling_mode: str = ...,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...
