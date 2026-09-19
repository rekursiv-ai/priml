from collections.abc import Callable
from typing import Any

from _typeshed import Incomplete

pvary: Incomplete

def cond[OperandT, ResultT](
    pred: object,
    true_fun: Callable[[OperandT], ResultT],
    false_fun: Callable[[OperandT], ResultT],
    *,
    operand: OperandT,
) -> ResultT: ...
def scan(
    f: Any,
    init: Any,
    xs: Any = ...,
    length: Any = ...,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any]: ...
def top_k(operand: Any, k: int) -> tuple[Any, Any]: ...
def pmean(x: Any, axis_name: str, *args: Any, **kwargs: Any) -> Any: ...
def dynamic_update_slice(operand: Any, update: Any, start_indices: Any) -> Any: ...
def dynamic_slice(operand: Any, start_indices: Any, slice_sizes: Any) -> Any: ...
def stop_gradient(x: Any) -> Any: ...
def complex(x: Any, y: Any) -> Any: ...
