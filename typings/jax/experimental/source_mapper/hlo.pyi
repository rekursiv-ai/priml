from typing import Any

import enum

from _typeshed import Incomplete
from jax._src import sourcemap as sourcemap
from jax.experimental.source_mapper import (
    common as common,
    mlir as mlir,
)

class HloPass(enum.Enum):
    STABLE_HLO = "hlo:stable-hlo"
    ORIGINAL = "hlo:original"
    OPTIMIZED = "hlo:optimized"

METADATA_REGEX: Incomplete

def parse_hlo_dump(text: str) -> sourcemap.SourceMap: ...
def trace_and_lower(work_dir, f, f_args, f_kwargs, **_): ...
def stable_hlo_generate_dump(args: tuple[Any, str], **_) -> common.SourceMapDump: ...
def original_hlo_generate_dump(args: tuple[Any, str], **_) -> common.SourceMapDump: ...
def optimized_generate_dump(
    args: tuple[Any, str],
    xla_compiler_flags: dict[str, Any] | None = None,
    **_,
) -> common.SourceMapDump: ...
