from _typeshed import Incomplete
from jax._src.typing import Array

__all__ = ["c_", "index_exp", "mgrid", "ogrid", "r_", "s_"]

class _Mgrid:
    def __getitem__(self, key: slice | tuple[slice, ...]) -> Array: ...

mgrid: Incomplete

class _Ogrid:
    def __getitem__(self, key: slice | tuple[slice, ...]) -> Array | list[Array]: ...

ogrid: Incomplete

class _AxisConcat:
    axis: int
    ndmin: int
    trans1d: int
    op_name: str
    def __getitem__(self, key: _IndexType | tuple[_IndexType, ...]) -> Array: ...
    def __len__(self) -> int: ...

class RClass(_AxisConcat):
    axis: int
    ndmin: int
    trans1d: int
    op_name: str

r_: Incomplete

class CClass(_AxisConcat):
    axis: int
    ndmin: int
    trans1d: int
    op_name: str

c_: Incomplete
s_: Incomplete
index_exp: Incomplete
