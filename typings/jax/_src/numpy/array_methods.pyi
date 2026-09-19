from collections.abc import Callable

from jax._src.lax import slicing as lax_slicing
from jax._src.ops import scatter
from jax._src.pjit import PartitionSpec
from jax._src.sharding_impls import NamedSharding
from jax._src.typing import Array, ArrayLike

__all__ = ["register_jax_array_methods"]

class _IndexUpdateHelper:
    array: Array
    def __init__(self, array: Array) -> None: ...
    def __getitem__(self, index: scatter.Index) -> _IndexUpdateRef: ...

class _IndexUpdateRef:
    array: Array
    index: scatter.Index
    def __init__(self, array: Array, index: scatter.Index) -> None: ...
    def get(
        self,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        fill_value: ArrayLike | None = None,
        out_sharding: NamedSharding | PartitionSpec | None = None,
        wrap_negative_indices: bool = True,
    ): ...
    def set(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        out_sharding: NamedSharding | PartitionSpec | None = None,
        wrap_negative_indices: bool = True,
    ) -> None: ...
    def apply(
        self,
        func: Callable[[ArrayLike], Array],
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        wrap_negative_indices: bool = True,
    ) -> Array: ...
    def add(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        out_sharding: NamedSharding | PartitionSpec | None = None,
        wrap_negative_indices: bool = True,
    ) -> Array: ...
    def subtract(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        wrap_negative_indices: bool = True,
    ) -> Array: ...
    def multiply(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        wrap_negative_indices: bool = True,
    ) -> Array: ...
    mul = multiply
    def divide(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        wrap_negative_indices: bool = True,
    ) -> Array: ...
    def power(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        wrap_negative_indices: bool = True,
    ) -> Array: ...
    def min(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        wrap_negative_indices: bool = True,
    ) -> Array: ...
    def max(
        self,
        values: ArrayLike,
        *,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        mode: str | lax_slicing.GatherScatterMode | None = None,
        wrap_negative_indices: bool = True,
    ) -> Array: ...

def register_jax_array_methods() -> None: ...
