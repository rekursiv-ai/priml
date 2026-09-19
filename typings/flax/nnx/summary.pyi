from collections import defaultdict

import dataclasses
import typing as tp

from _typeshed import Incomplete
from flax import (
    nnx as nnx,
    typing as typing,
)
from flax.nnx import (
    graphlib as graphlib,
    statelib as statelib,
    variablelib as variablelib,
)

import jax
import numpy as np
import yaml

in_ipython: Incomplete

class NoneDumper(yaml.SafeDumper): ...
class SizeBytes(typing.SizeBytes): ...

class ObjectInfo(tp.NamedTuple):
    path: statelib.PathParts
    stats: dict[type[variablelib.Variable], SizeBytes]
    variable_groups: defaultdict[
        type[variablelib.Variable],
        defaultdict[typing.Key, variablelib.Variable],
    ]

type NodeStats = dict[int, ObjectInfo | None]

@dataclasses.dataclass(frozen=True, repr=False)
class ArrayRepr:
    shape: tuple[int, ...]
    dtype: tp.Any
    @classmethod
    def from_array(cls, x: jax.Array | np.ndarray): ...

@dataclasses.dataclass
class CallInfo:
    call_order: int
    object_id: int
    type: type
    path: statelib.PathParts
    inputs_repr: str
    outputs: tp.Any
    flops: int | None
    vjp_flops: int | None

class SimpleObjectRepr:
    type: Incomplete
    def __init__(self, obj: tp.Any) -> None: ...

def filter_rng_streams(row: CallInfo): ...
def tabulate(
    obj,
    *input_args,
    depth: int | None = None,
    method: str = "__call__",
    row_filter: tp.Callable[[CallInfo], bool] = ...,
    table_kwargs: tp.Mapping[str, tp.Any] = ...,
    column_kwargs: tp.Mapping[str, tp.Any] = ...,
    console_kwargs: tp.Mapping[str, tp.Any] = ...,
    compute_flops: bool = False,
    compute_vjp_flops: bool = False,
    **input_kwargs,
) -> str: ...
