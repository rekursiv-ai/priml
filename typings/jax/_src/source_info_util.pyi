from collections.abc import Iterator
from typing import NamedTuple

import contextlib
import dataclasses
import functools
import threading
import types

from _typeshed import Incomplete
from jax._src import traceback_util as traceback_util
from jax._src.lib import xla_client as xla_client

Traceback: Incomplete

class Frame(NamedTuple):
    file_name: str
    function_name: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int

def register_exclusion(path: str): ...
def register_inclusion(path: str): ...

class Scope(NamedTuple):
    name: str
    def wrap(self, stack: list[str]): ...

class Transform(NamedTuple):
    name: str
    def wrap(self, stack: list[str]): ...

@dataclasses.dataclass(frozen=True)
class NameStack:
    stack: tuple[Scope | Transform, ...] = ...
    def extend(self, name: str) -> NameStack: ...
    def transform(self, transform_name: str) -> NameStack: ...
    def __getitem__(self, idx: slice) -> NameStack: ...
    def __len__(self) -> int: ...
    def __add__(self, other: NameStack) -> NameStack: ...
    def __radd__(self, other: NameStack) -> NameStack: ...

def new_name_stack(name: str = "") -> NameStack: ...

class SourceInfo:
    traceback: Traceback | None
    name_stack: NameStack
    def __init__(self, traceback: Traceback | None, name_stack: NameStack) -> None: ...
    def replace(
        self,
        *,
        traceback: Traceback | None = None,
        name_stack: NameStack | None = None,
    ) -> SourceInfo: ...

def new_source_info() -> SourceInfo: ...
@functools.cache
def is_user_filename(filename: str) -> bool: ...
def raw_frame_to_frame(code: types.CodeType, lasti: int) -> Frame: ...
def user_frames(traceback: Traceback | None) -> Iterator[Frame]: ...
def user_frame(traceback: Traceback | None) -> Frame | None: ...
def summarize(source_info: SourceInfo, num_frames: int = 1) -> str: ...

class _SourceInfoContext(threading.local):
    context: SourceInfo
    def __init__(self) -> None: ...

def current() -> SourceInfo: ...

class JaxStackTraceBeforeTransformation(Exception): ...

def has_user_context(e): ...

class UserContextManager:
    traceback: Incomplete
    name_stack: Incomplete
    def __init__(
        self,
        traceback: Traceback | None,
        *,
        name_stack: NameStack | None = None,
    ) -> None: ...
    prev: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

user_context = UserContextManager

def current_name_stack() -> NameStack: ...

class ExtendNameStackContextManager(contextlib.ContextDecorator):
    name: Incomplete
    def __init__(self, name: str) -> None: ...
    prev: Incomplete
    def __enter__(self): ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

extend_name_stack = ExtendNameStackContextManager

class SetNameStackContextManager(contextlib.ContextDecorator):
    name_stack: Incomplete
    def __init__(self, name_stack: NameStack) -> None: ...
    prev: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

set_name_stack = SetNameStackContextManager

@contextlib.contextmanager
def reset_name_stack() -> Iterator[None]: ...

class TransformNameStackContextManager(contextlib.ContextDecorator):
    name: Incomplete
    def __init__(self, name: str) -> None: ...
    prev: Incomplete
    def __enter__(self): ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

transform_name_stack = TransformNameStackContextManager
