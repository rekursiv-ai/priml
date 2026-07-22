from torchao.ops import register_custom_op, register_custom_op_impl
from torchao.utils import TorchAOBaseTensor

import torch

aten = ...

@register_custom_op_impl("blocksparse::bsr_to_dense")
def bsr_to_dense(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    M: int,
    K: int,
) -> torch.Tensor: ...
@register_custom_op("blocksparse::bsr_to_dense")
def bsr_to_dense_abstract(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    M: int,
    K: int,
) -> torch.Tensor: ...
@register_custom_op_impl("blocksparse::int_addmm")
def blocksparse_int_addmm(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    A: torch.Tensor,
    left_alpha: torch.Tensor,
    right_alpha: torch.Tensor,
) -> torch.Tensor: ...
@register_custom_op("blocksparse::int_addmm")
def blocksparse_int_addmm_abstract(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    A: torch.Tensor,
    left_alpha: torch.Tensor,
    right_alpha: torch.Tensor,
) -> torch.Tensor: ...
@register_custom_op_impl("blocksparse::addmm")
def blocksparse_addmm(
    x_padded: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    M: int,
    K: int,
    bias: torch.Tensor,
) -> torch.Tensor: ...
@register_custom_op("blocksparse::addmm")
def blocksparse_addmm_abstract(
    x_padded: torch.Tensor,
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
    M: int,
    K: int,
    bias: torch.Tensor,
) -> torch.Tensor: ...

class BlockSparseTensor(TorchAOBaseTensor):
    bsr_crow_indices: torch.Tensor | None
    bsr_col_indices: torch.Tensor | None
    bsr_values: torch.Tensor | None
    blocksize: int
    __slots__ = ...
    @staticmethod
    def __new__(
        cls,
        shape: torch.Size,
        blocksize: int,
        bsr_crow_indices: torch.Tensor | None,
        bsr_col_indices: torch.Tensor | None,
        bsr_values: torch.Tensor | None,
        requires_grad: bool = ...,
    ): ...
    def __tensor_flatten__(self) -> tuple[list[str], tuple[torch.Size, bool, int]]: ...
    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: tuple[torch.Size, bool, int],
        outer_size,
        outer_stride,
    ) -> torch.Tensor: ...
    @classmethod
    def from_dense(cls, dense_tensor, blocksize):  # -> Self:
        ...
    def apply_fn_to_shard(self, func):  # -> BlockSparseTensor:
        ...

implements = ...

@implements(aten.detach.default)
def block_sparse_detach(func, types, args, kwargs):  # -> tuple[Any, ...] | Any:
    ...
@implements(aten.unsqueeze.default)
def block_sparse_unsqueeze(func, types, args, kwargs):  # -> BlockSparseTensor:
    ...
@implements(aten.mul.Tensor)
def block_sparse_mul(func, types, args, kwargs):  # -> BlockSparseTensor:
    ...
@implements(aten.sum.dim_IntList)
def block_sparse_sum(func, types, args, kwargs):  # -> Any:
    ...
@implements(aten.values.default)
def block_sparse_values(func, types, args, kwargs): ...
@implements(aten.crow_indices.default)
def block_sparse_crow_indices(func, types, args, kwargs): ...
@implements(aten.col_indices.default)
def block_sparse_col_indices(func, types, args, kwargs): ...
@implements(aten._nnz.default)
def block_sparse__nnz(func, types, args, kwargs): ...
@implements(torch.nn.functional.linear)
def block_sparse_linear(func, types, args, kwargs):  # -> Any:
    ...
