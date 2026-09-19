from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
    ffi as ffi,
)
from jax._src.interpreters import mlir as mlir
from jax._src.lib import (
    cpu_sparse as cpu_sparse,
    gpu_sparse as gpu_sparse,
    has_cpu_sparse as has_cpu_sparse,
)

SUPPORTED_DATA_DTYPES: Incomplete
SUPPORTED_INDEX_DTYPES: Incomplete
coo_spmv_p: Incomplete
coo_spmm_p: Incomplete
csr_spmv_p: Incomplete
csr_spmm_p: Incomplete

def coo_todense_gpu_lowering(ctx, data, row, col, *, shape, target_name_prefix): ...
def coo_fromdense_gpu_lowering(ctx, mat, *, nnz, index_dtype, target_name_prefix): ...
def csr_todense_gpu_lowering(
    ctx,
    data,
    indices,
    indptr,
    *,
    shape,
    target_name_prefix,
): ...
def csr_fromdense_gpu_lowering(ctx, mat, *, nnz, index_dtype, target_name_prefix): ...
