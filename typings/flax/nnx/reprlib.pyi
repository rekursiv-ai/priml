from collections.abc import Generator

import dataclasses
import threading
import typing as tp

from _typeshed import Incomplete

A = tp.TypeVar("A")
B = tp.TypeVar("B")

def supports_color() -> bool: ...

class Color(tp.NamedTuple):
    TYPE: str
    ATTRIBUTE: str
    SEP: str
    PAREN: str
    COMMENT: str
    INT: str
    STRING: str
    FLOAT: str
    BOOL: str
    NONE: str
    END: str

NO_COLOR: Incomplete
COLOR: Incomplete
COLOR = NO_COLOR

@dataclasses.dataclass
class ReprContext(threading.local):
    current_color: Color = ...
    depth: int = ...

REPR_CONTEXT: Incomplete

def colorized(x, /): ...

@dataclasses.dataclass
class Object:
    type: str | type
    start: str = ...
    end: str = ...
    kv_sep: str = ...
    indent: str = ...
    empty_repr: str = ...
    comment: str = ...
    same_line: bool = ...
    @property
    def elem_sep(self): ...

@dataclasses.dataclass
class Attr:
    key: str
    value: str | tp.Any
    start: str = ...
    end: str = ...
    use_raw_value: bool = ...
    use_raw_key: bool = ...

class Representable:
    def __nnx_repr__(self) -> tp.Iterator[Object | Attr]: ...

def get_repr(obj: Representable) -> str: ...

class MappingReprMixin(Representable):
    def __nnx_repr__(self) -> Generator[Incomplete]: ...

@dataclasses.dataclass(repr=False)
class PrettyMapping(Representable):
    mapping: tp.Mapping
    def __nnx_repr__(self) -> Generator[Incomplete]: ...

@dataclasses.dataclass(repr=False)
class SequenceReprMixin(Representable):
    def __nnx_repr__(self) -> Generator[Incomplete]: ...

@dataclasses.dataclass(repr=False)
class PrettySequence(Representable):
    sequence: tp.Sequence
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
