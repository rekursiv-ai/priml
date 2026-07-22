from typing import Any

from torchao.float8.float8_training_tensor import Float8TrainingTensor

import torch

aten = ...
c10d_functional = ...
_c10d_functional = ...
FLOAT8_OPS_TABLE: dict[Any, Any] = ...

def addmm_float8_unwrapped(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,
    output_scale: torch.Tensor | None = ...,
    bias: torch.Tensor | None = ...,
    use_fast_accum: bool = ...,
) -> torch.Tensor: ...
def implements(aten_ops):  # -> Callable[..., Any]:
    ...
@implements(
    [
        aten._unsafe_view.default,
        aten.as_strided.default,
        aten.clone.default,
        aten.slice.Tensor,
        aten.fill_.Scalar,
        aten.reshape.default,
    ]
)
def float8_desugar_op(aten_op, args, kwargs=...):  # -> Float8TrainingTensor:
    ...
@implements([aten.detach.default])
def float8_desugar_data_and_scale_op(
    aten_op, args, kwargs=...
):  # -> Float8TrainingTensor:
    ...
@implements([aten.t.default, aten.transpose.int])
def float8_transpose(aten_op, args, kwargs=...):  # -> Float8TrainingTensor:
    ...
@implements([aten.view.default])
def float8_view(aten_op, args, kwargs=...):  # -> Float8TrainingTensor:
    ...
@implements([aten.split.Tensor])
def float8_split(aten_op, args, kwargs=...):  # -> list[Float8TrainingTensor]:
    ...
@implements([aten.cat.default])
def float8_cat(aten_op, args, kwargs=...):  # -> Float8TrainingTensor:
    ...
@implements([aten.sum.dim_IntList])
def float8_cast_up_op(aten_op, args, kwargs=...): ...
def preprocess_addmm(
    a: Float8TrainingTensor, b: Float8TrainingTensor
):  # -> tuple[Tensor, Tensor, Tensor, Tensor]:
    ...
@implements([aten.mm.default, aten.matmul.default])
def float8_mm(aten_op, args, kwargs=...):  # -> Tensor:
    ...
@implements([aten.addmm.default])
def float8_addmm(aten_op, args, kwargs=...):  # -> Tensor:
    ...
@implements([aten.is_same_size.default])
def float8_is_same_size(aten_op, args, kwargs=...): ...
@implements([aten._to_copy.default])
def autocast_to_copy(aten_op, args, kwargs=...):  # -> Float8TrainingTensor:
    ...
@implements(
    [
        c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor.default,
    ]
)
def allgather_fp8(aten_op, args, kwargs=...):  # -> Float8TrainingTensor:
    ...
@implements([c10d_functional.wait_tensor.default, _c10d_functional.wait_tensor.default])
def wait_tensor_fp8(aten_op, args, kwargs=...):  # -> Float8TrainingTensor:
    ...
@implements([aten.index_put_.default])
def index_put_fp8(aten_op, args, kwargs=...):  # -> Float8TrainingTensor:
    ...
@implements([aten.copy_.default])
def copy_fp8(aten_op, args, kwargs=...):  # -> Float8TrainingTensor:
    ...
