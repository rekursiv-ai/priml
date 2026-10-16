from collections.abc import Sequence
from typing import TypeVar

from numba.core.types import StructRef

_T = TypeVar("_T", bound=type[StructRef])

class StructRefProxy: ...

def register(struct_type: _T, /) -> _T: ...
def new(struct_type: StructRef, /) -> StructRefProxy: ...
def define_proxy(
    proxy_cls: type[StructRefProxy],
    typeclass: type[StructRef],
    fields: Sequence[str],
    /,
) -> None: ...
