from collections.abc import Sequence
from typing import Literal

from numba.core.typing.templates import Signature

class Type:
    name: str
    # A type called on argument types is the signature returning it.
    def __call__(self, *args: Type) -> Signature: ...
    # A scalar type sliced `[::1]` is its contiguous array type.
    def __getitem__(self, layout: slice | tuple[slice, ...], /) -> Array: ...

class Number(Type): ...

class Integer(Number):
    bitwidth: int
    signed: bool

class Float(Number):
    bitwidth: int

class NoneType(Type): ...
class Boolean(Type): ...

class Record(Type):
    size: int

class Array(Type):
    dtype: Type
    ndim: int
    layout: Literal["C", "F", "A"]
    mutable: bool
    def __init__(
        self,
        dtype: Type,
        ndim: int,
        layout: Literal["C", "F", "A"],
        readonly: bool = False,
        name: str | None = None,
        aligned: bool = True,
    ) -> None: ...

class StructRef(Type):
    def __init__(self, fields: Sequence[tuple[str, Type]]) -> None: ...

boolean: Boolean
none: NoneType
int32: Integer
int64: Integer
intp: Integer
uint8: Integer
uint64: Integer
float32: Float
float64: Float
void: NoneType
