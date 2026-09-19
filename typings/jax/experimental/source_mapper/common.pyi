from collections.abc import Generator, Sequence
from typing import Any, Protocol

import contextlib
import dataclasses

from jax._src import sourcemap as sourcemap

@dataclasses.dataclass(frozen=True)
class SourceMapDump:
    source_map: sourcemap.SourceMap
    generated_code: str
    pass_name: str

class CompileFn(Protocol):
    def __call__(self, work_dir, fn, f_args, f_kwargs, **kwargs) -> Any: ...

class GenerateDumpFn(Protocol):
    def __call__(self, compile_result: Any, **kwargs) -> SourceMapDump: ...

@dataclasses.dataclass(frozen=True)
class Pass:
    name: str
    compile_fn: CompileFn
    generate_dump: GenerateDumpFn

def register_pass(pass_: Pass): ...
def all_passes() -> Sequence[Pass]: ...
def filter_passes(regex: str) -> Sequence[Pass]: ...
@contextlib.contextmanager
def flag_env(**kwargs) -> Generator[None]: ...
def compile_with_env(f, f_args, f_kwargs, env_flags, compiler_flags): ...
