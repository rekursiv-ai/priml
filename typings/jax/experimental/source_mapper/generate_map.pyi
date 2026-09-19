from collections.abc import Sequence
from typing import Protocol

from jax.experimental.source_mapper import common as common

class SourceMapGeneratorFn(Protocol):
    def __call__(self, *args, **kwargs) -> Sequence[common.SourceMapDump]: ...

def generate_sourcemaps(
    f,
    passes: Sequence[common.Pass],
    **pass_kwargs,
) -> SourceMapGeneratorFn: ...
