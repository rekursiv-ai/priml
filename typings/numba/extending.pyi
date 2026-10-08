from collections.abc import Callable
from typing import Generic, TypeVar

_F_co = TypeVar("_F_co", bound=Callable[..., object], covariant=True)
_F = TypeVar("_F", bound=Callable[..., object])

# Callable from compiled code only; its kernel-side signature is whatever the
# typing function returns, so each caller states it with a cast.
class _Intrinsic(Generic[_F_co]): ...

def intrinsic(func: _F, /) -> _Intrinsic[_F]: ...
