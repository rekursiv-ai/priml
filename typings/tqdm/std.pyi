from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import types

from _typeshed import Incomplete

from .utils import Comparable

__all__ = [
    "TqdmDeprecationWarning",
    "TqdmExperimentalWarning",
    "TqdmKeyError",
    "TqdmMonitorWarning",
    "TqdmTypeError",
    "TqdmWarning",
    "tqdm",
    "trange",
]

class TqdmTypeError(TypeError): ...
class TqdmKeyError(KeyError): ...

class TqdmWarning(Warning):
    def __init__(self, msg, fp_write=None) -> None: ...

class TqdmExperimentalWarning(TqdmWarning, FutureWarning): ...
class TqdmDeprecationWarning(TqdmWarning, DeprecationWarning): ...
class TqdmMonitorWarning(TqdmWarning, RuntimeWarning): ...

class TqdmDefaultWriteLock:
    th_lock: Incomplete
    locks: Incomplete
    def __init__(self) -> None: ...
    def acquire(self, *a, **k) -> None: ...
    def release(self) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc) -> None: ...
    @classmethod
    def create_mp_lock(cls) -> None: ...
    @classmethod
    def create_th_lock(cls) -> None: ...

class Bar:
    ASCII: str
    UTF: Incomplete
    BLANK: str
    COLOUR_RESET: str
    COLOUR_RGB: str
    COLOURS: Incomplete
    frac: Incomplete
    default_len: Incomplete
    charset: Incomplete
    def __init__(
        self, frac, default_len: int = 10, charset=..., colour=None
    ) -> None: ...
    @property
    def colour(self): ...
    @colour.setter
    def colour(self, value) -> None: ...
    def __format__(self, format_spec) -> str: ...

class EMA:
    alpha: Incomplete
    last: int
    calls: int
    def __init__(self, smoothing: float = 0.3) -> None: ...
    def __call__(self, x=None): ...

class tqdm(Comparable):
    monitor_interval: int
    monitor: Incomplete
    @staticmethod
    def format_sizeof(num, suffix: str = "", divisor: int = 1000): ...
    @staticmethod
    def format_interval(t): ...
    @staticmethod
    def format_num(n): ...
    @staticmethod
    def status_printer(file): ...
    @staticmethod
    def format_meter(
        n,
        total,
        elapsed,
        ncols=None,
        prefix: str = "",
        ascii: bool = False,
        unit: str = "it",
        unit_scale: bool = False,
        rate=None,
        bar_format=None,
        postfix=None,
        unit_divisor: int = 1000,
        initial: int = 0,
        colour=None,
        **extra_kwargs,
    ): ...
    def __new__(cls, *_, **__): ...
    @classmethod
    def write(
        cls, s: str, file: Any = None, end: str = "\n", nolock: bool = False
    ) -> None: ...
    @classmethod
    @contextmanager
    def external_write_mode(
        cls, file=None, nolock: bool = False
    ) -> Generator[None]: ...
    @classmethod
    def set_lock(cls, lock) -> None: ...
    @classmethod
    def get_lock(cls): ...
    @classmethod
    def pandas(cls, **tqdm_kwargs): ...
    iterable: Incomplete
    disable: Incomplete
    pos: Incomplete
    n: Incomplete
    total: Incomplete
    leave: Incomplete
    desc: Incomplete
    fp: Incomplete
    ncols: Incomplete
    nrows: Incomplete
    mininterval: Incomplete
    maxinterval: Incomplete
    miniters: Incomplete
    dynamic_miniters: Incomplete
    ascii: Incomplete
    unit: Incomplete
    unit_scale: Incomplete
    unit_divisor: Incomplete
    initial: Incomplete
    lock_args: Incomplete
    delay: Incomplete
    gui: Incomplete
    dynamic_ncols: Incomplete
    smoothing: Incomplete
    bar_format: Incomplete
    postfix: Incomplete
    colour: Incomplete
    last_print_n: Incomplete
    sp: Incomplete
    last_print_t: Incomplete
    start_t: Incomplete
    def __init__(
        self,
        iterable: Any = None,
        desc: str | None = None,
        total: int | None = None,
        leave: bool = True,
        file: Any = None,
        ncols: int | None = None,
        mininterval: float = 0.1,
        maxinterval: float = 10.0,
        miniters: Any = None,
        ascii: Any = None,
        disable: bool = False,
        unit: str = "it",
        unit_scale: bool = False,
        dynamic_ncols: bool = False,
        smoothing: float = 0.3,
        bar_format: str | None = None,
        initial: int = 0,
        position: int | None = None,
        postfix: Any = None,
        unit_divisor: int = 1000,
        write_bytes: bool = False,
        lock_args: Any = None,
        nrows: int | None = None,
        colour: str | None = None,
        delay: float = 0.0,
        gui: bool = False,
        **kwargs: Any,
    ) -> None: ...
    def __bool__(self) -> bool: ...
    def __len__(self) -> int: ...
    def __reversed__(self) -> Any: ...
    def __contains__(self, item: Any) -> bool: ...
    def __enter__(self) -> tqdm[Any]: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __hash__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def update(self, n: int = 1) -> bool | None: ...
    def close(self) -> None: ...
    def clear(self, nolock: bool = False) -> None: ...
    def refresh(self, nolock: bool = False, lock_args: Any = None) -> bool: ...
    def unpause(self) -> None: ...
    def reset(self, total: Any = None) -> None: ...
    def set_description(
        self, desc: str | None = None, refresh: bool = True
    ) -> None: ...
    def set_description_str(
        self, desc: str | None = None, refresh: bool = True
    ) -> None: ...
    def set_postfix(
        self, ordered_dict: Any = None, refresh: bool = True, **kwargs: Any
    ) -> None: ...
    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None: ...
    def moveto(self, n: int) -> None: ...
    @property
    def format_dict(self): ...
    def display(self, msg=None, pos=None): ...
    @classmethod
    @contextmanager
    def wrapattr(
        cls, stream, method, total=None, bytes: bool = True, **tqdm_kwargs
    ) -> Generator[Incomplete]: ...

def trange(*args, **kwargs): ...
