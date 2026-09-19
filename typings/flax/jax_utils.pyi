from collections.abc import Generator
from typing import Any, TypeVar

_T = TypeVar("_T")

def replicate(tree: _T, devices: Any = None) -> _T: ...
def unreplicate(tree: _T) -> _T: ...
def pmean(xs: Any, axis_name: Any) -> Any: ...
def partial_eval_by_shape(
    fn: Any,
    input_spec: Any,
    *args: Any,
    **kwargs: Any,
) -> Any: ...
def prefetch_to_device(
    iterator: Any,
    size: int,
    devices: Any = None,
) -> Generator[Any]: ...
def scan_in_dim(
    body_fn: Any,
    init: Any,
    xs: Any,
    axis: Any = (0,),
    unroll: Any = (1,),
    keepdims: bool = False,
) -> Any: ...
def pad_shard_unpad(
    wrapped: Any,
    static_argnums: Any = (0,),
    static_argnames: Any = (),
    static_return: bool = False,
) -> Any: ...
