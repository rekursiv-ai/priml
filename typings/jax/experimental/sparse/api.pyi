from _typeshed import Incomplete
from jax import tree_util as tree_util
from jax._src import (
    core as core,
    dtypes as dtypes,
)
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
)
from jax._src.typing import (
    Array as Array,
    DTypeLike as DTypeLike,
    Shape as Shape,
)
from jax.experimental.sparse._base import JAXSparse as JAXSparse
from jax.experimental.sparse.bcoo import BCOO as BCOO
from jax.experimental.sparse.bcsr import BCSR as BCSR
from jax.experimental.sparse.coo import COO as COO
from jax.experimental.sparse.csr import (
    CSC as CSC,
    CSR as CSR,
)
from jax.interpreters import mlir as mlir

todense_p: Incomplete

def todense(arr: JAXSparse | Array) -> Array: ...
def empty(
    shape: Shape,
    dtype: DTypeLike | None = None,
    index_dtype: DTypeLike = "int32",
    sparse_format: str = "bcoo",
    **kwds,
) -> JAXSparse: ...
def eye(
    N: int,
    M: int | None = None,
    k: int = 0,
    dtype: DTypeLike | None = None,
    index_dtype: DTypeLike = "int32",
    sparse_format: str = "bcoo",
    **kwds,
) -> JAXSparse: ...
