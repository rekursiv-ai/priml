from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch.distributed.device_mesh import DeviceMesh

import torch
import torch.nn.functional as F

aten = ...
c10d_functional = ...

def nf4_all_gather_into_tensor(func, *args, **kwargs):  # -> NF4Tensor:
    ...
def scatter_nf4tensor(
    func, *args, **kwargs
):  # -> tuple[Any | list[Any], Any | list[Any]]:
    ...

NF4_OPS_TABLE: dict[Any, Any] = ...
_INNER_TENSOR_NAMES_FOR_SHARDING = ...
CHUNK_SIZE = ...

def same_metadata(a: NF4Tensor, b: NF4Tensor):  # -> bool:
    ...
def implements(aten_ops):  # -> Callable[..., Any]:
    ...
def construct_nf4_args(
    nf4tensor: NF4Tensor, kwargs: dict[str, Any] | None = ...
):  # -> tuple[SubclassTensorArgs, Any, Any, Any, Any, Any, Any, Any, Any]:
    ...
def apply_to_inner_tensors(
    nf4tensor: NF4Tensor, aten_op, args, kwargs
):  # -> dict[Any, Any]:
    ...
def call_from_inner_tensors(
    nf4tensor: NF4Tensor, method_name: str, args, kwargs
):  # -> dict[Any, Any]:
    ...

class CompareOp(Enum):
    EQ = ...
    LT = ...

def expect_num_of_args(
    op: CompareOp, num: int, msg: str
):  # -> Callable[..., _Wrapped[..., Any, ..., Any]]:
    ...
def expect_arg_value_at_k(
    k: int, op: CompareOp, value: Any, msg: str
):  # -> Callable[..., _Wrapped[..., Any, ..., Any]]:
    ...
def expect_args_len_at_k(
    k: int, op: CompareOp, value: Any, msg: str
):  # -> Callable[..., _Wrapped[..., Any, ..., Any]]:
    ...
@implements([torch.ops.aten.detach])
def noop_detach(func, *args, **kwargs): ...
@implements([torch.ops.aten.clone.default])
def clone(func, *args, **kwargs):  # -> NF4Tensor:
    ...
@implements([aten.detach.default])
def nf4_detach(aten_op, args, kwargs=...):  # -> NF4Tensor:
    ...
@implements([aten.empty_like.default])
def nf4_empty_like(aten_op, args, kwargs=...):  # -> NF4Tensor:
    ...
@implements([aten.split.Tensor])
def nf4_split(aten_op, args, kwargs=...):  # -> list[Any]:
    ...
@implements([aten.new_zeros.default])
@expect_args_len_at_k(1, CompareOp.LT, 3, "aten.view(NF4Tensor) with len(size)=")
def nf4_new_zeros(aten_op, args, kwargs=...):  # -> NF4Tensor:
    ...
@implements([aten.slice.Tensor])
@expect_num_of_args(CompareOp.LT, 5, "aten.slice(NF4Tensor) with customized step")
@expect_arg_value_at_k(1, CompareOp.EQ, 0, "aten.slice(NF4Tensor) with dim=")
@expect_arg_value_at_k(2, CompareOp.EQ, 0, "aten.slice(NF4Tensor) with start=")
def nf4_slice(aten_op, args, kwargs=...):  # -> NF4Tensor:
    ...
@implements([aten.view.default])
@expect_args_len_at_k(1, CompareOp.LT, 3, "aten.view(NF4Tensor) with len(size)=")
def nf4_view(aten_op, args, kwargs=...):  # -> NF4Tensor:
    ...
@implements([aten.as_strided.default])
@expect_args_len_at_k(1, CompareOp.LT, 3, ...)
def nf4_as_strided(aten_op, args, kwargs=...):  # -> NF4Tensor:
    ...
@implements([torch.ops.aten.to.dtype])
def to_dtype(func, *args, **kwargs):  # -> Any:
    ...
@implements([torch.ops.aten.t.default])
def t_default(func, *args, **kwargs):  # -> NF4Tensor:
    ...
@implements([torch.ops.aten.mm.default])
def mm_default(func, *args, **kwargs):  # -> Tensor:
    ...
@implements([aten.copy_.default])
def copy_(func, *args, **kwargs):  # -> Tensor | None:
    ...
@implements([aten.is_pinned.default])
def nf4_is_pinned(aten_op, args, kwargs=...):  # -> bool:
    ...
@implements([aten._pin_memory.default])
def nf4_pin_memory(aten_op, args, kwargs=...):  # -> NF4Tensor:
    ...
@implements([aten.cat.default])
def nf4_cat(aten_op: torch._ops.OpOverload, args, kwargs=...):  # -> Any:
    ...
@implements([torch.ops._c10d_functional.wait_tensor.default])
def wait_tensor(func, *args, **kwargs):  # -> NF4Tensor:
    ...

@dataclass(frozen=True)
class SubclassTensorArgs:
    original_shape: torch.Size
    original_strides: tuple
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool

def get_block_absmax(input_tensor: torch.Tensor, block_size: int) -> torch.Tensor: ...

class NF4Tensor(torch.Tensor):
    @torch._dynamo.disable
    def __new__(
        cls,
        tensor_meta: SubclassTensorArgs,
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        nf4: torch.Tensor,
    ):  # -> Self:
        ...
    @torch._dynamo.disable
    def __init__(
        self,
        tensor_meta: SubclassTensorArgs,
        block_size: int,
        n_blocks: int,
        scaler_block_size: int,
        quantized_scalers: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_mean: torch.Tensor,
        quantized_data: torch.Tensor,
        nf4: torch.Tensor,
    ) -> None: ...
    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls, input_tensor: torch.Tensor, block_size: int, scaler_block_size: int
    ):  # -> Self:
        ...
    @staticmethod
    def double_quantize_scalers(
        input_tensor: torch.Tensor, block_size: int, scaler_block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def dequantize_scalers(
        self,
        input_tensor: torch.Tensor,
        quantization_factor: torch.Tensor,
        scaler_block_size: int,
    ) -> torch.Tensor: ...
    @staticmethod
    def convert_to_norm_float_weight(
        input_tensor: torch.Tensor, n_blocks: int, block_size: int, nf4: torch.Tensor
    ) -> torch.Tensor: ...
    def get_original_weight(self) -> torch.Tensor: ...
    @staticmethod
    def quantize_tensor_nearest(
        value: torch.Tensor, nf4: torch.Tensor
    ) -> torch.Tensor: ...
    @staticmethod
    def dequantize(value: torch.Tensor, nf4: torch.Tensor) -> torch.Tensor: ...
    def __tensor_flatten__(
        self,
    ):  # -> tuple[list[str], dict[str, int | SubclassTensorArgs]]:
        ...
    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict, metadata, outer_size, outer_stride
    ):  # -> NF4Tensor:
        ...
    @classmethod
    @torch._dynamo.disable
    def __torch_dispatch__(cls, func, types, args, kwargs=...):  # -> Any:
        ...
    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=...): ...
    def fsdp_pre_all_gather(
        self, mesh: DeviceMesh
    ) -> tuple[tuple[torch.Tensor, ...], Any]: ...
    def fsdp_post_all_gather(
        self,
        all_gather_outputs: tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: torch.Tensor | None = ...,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]] | None: ...

class LinearNF4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: NF4Tensor):  # -> Tensor:
        ...
    @staticmethod
    def backward(ctx, grad_output):  # -> tuple[Any, None]:
        ...

def linear_nf4(input: torch.Tensor, weight: NF4Tensor) -> torch.Tensor: ...
def to_nf4(
    tensor, block_size: int = ..., scaler_block_size: int = ...
):  # -> NF4Tensor:
    ...

NF4_TORCH_FUNCTIONS = ...

def implements_torch_function(torch_function):  # -> Callable[..., Any]:
    ...
@implements_torch_function(torch.Tensor.to)
def function_to_dtype(*args, **kwargs):  # -> NF4Tensor:
    ...
@implements_torch_function(torch.Tensor.cpu)
def function_cpu(*args, **kwargs): ...
@implements_torch_function(torch.Tensor.cuda)
def function_cuda(*args, **kwargs):  # -> NF4Tensor:
    ...
@implements_torch_function(F.linear)
def _(*args, **kwargs):  # -> None:
    ...
@torch._dynamo.allow_in_graph
def nf4_constructor(
    tensor_meta: SubclassTensorArgs,
    block_size: int,
    n_blocks: int,
    scaler_block_size: int,
    quantized_scalers: torch.Tensor,
    quantization_factor: torch.Tensor,
    scaler_mean: torch.Tensor,
    quantized_data: torch.Tensor,
    nf4: torch.Tensor,
):  # -> NF4Tensor:
    ...
