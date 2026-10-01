from collections.abc import Callable, Sequence

__all__ = [
    "Dispatcher",
    "StructRef",
    "StructRefProxy",
    "define_proxy",
    "float64",
    "int32",
    "int64",
    "new",
    "njit",
    "prange",
    "register",
    "uint8",
]

class Type:
    # Subscripting a scalar type with [::1] gives its contiguous array type.

    def __getitem__(self, layout: object, /) -> Type: ...

class StructRef(Type):
    def __init__(self, fields: Sequence[tuple[str, Type]]) -> None: ...

class StructRefProxy: ...

class Dispatcher:
    @property
    def signatures(self) -> list[tuple[Type, ...]]: ...
    @property
    def py_func(self) -> Callable[..., object]: ...

float64: Type
int32: Type
int64: Type
uint8: Type

def njit[F: Callable[..., object]](
    *,
    cache: bool,
    error_model: str,
    parallel: bool = ...,
) -> Callable[[F], F]: ...
def prange(stop: int, /) -> range: ...
def register[T: type[StructRef]](struct_type: T, /) -> T: ...
def new(struct_type: StructRef, /) -> StructRefProxy: ...
def define_proxy(
    proxy_cls: type[StructRefProxy],
    typeclass: type[StructRef],
    fields: Sequence[str],
    /,
) -> None: ...
