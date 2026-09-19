from typing import NamedTuple

from _typeshed import Incomplete
from jax._src import sourcemap as sourcemap

LOC_REGEX: Incomplete
SRC_REGEX: Incomplete
SCOPED_REGEX: Incomplete
CALLSITE_REGEX: Incomplete

class Location(NamedTuple):
    file: Incomplete
    line: Incomplete
    col: Incomplete

class Redirect(NamedTuple):
    tgt_id: Incomplete

def create_mlir_sourcemap(mlir_dump: str) -> sourcemap.SourceMap: ...
def parse_mlir_locations(
    mlir_dump: list[str],
) -> tuple[dict[int, sourcemap.Segment], list[str]]: ...
