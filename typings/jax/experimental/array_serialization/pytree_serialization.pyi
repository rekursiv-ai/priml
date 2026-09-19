from os import PathLike
from typing import Any

import jax.experimental.array_serialization.pytree_serialization_utils as utils

__all__ = ["load", "load_pytreedef", "nonblocking_load", "nonblocking_save", "save"]

type PyTreeT = Any

def save(
    data: PyTreeT,
    directory: str | PathLike[str],
    *,
    overwrite: bool = True,
    ts_specs: PyTreeT | None = None,
) -> None: ...
def load_pytreedef(directory: str | PathLike[str]) -> PyTreeT: ...
def load(
    directory: str | PathLike[str],
    shardings: PyTreeT,
    *,
    mask: PyTreeT | None = None,
    ts_specs: PyTreeT | None = None,
) -> PyTreeT: ...
def nonblocking_save(
    data: PyTreeT,
    directory: str | PathLike[str],
    *,
    overwrite: bool = True,
    ts_specs: PyTreeT | None = None,
) -> utils.PyTreeFuture: ...
def nonblocking_load(
    directory: str | PathLike[str],
    shardings: PyTreeT,
    *,
    mask: PyTreeT | None = None,
    ts_specs: PyTreeT | None = None,
) -> utils.PyTreeFuture: ...
