from collections.abc import (
    Callable as Callable,
    Mapping,
    Sequence,
)
from typing import Any, NotRequired, TypedDict, overload

import dataclasses

from _typeshed import Incomplete
from jax._src import (
    core as core,
    dispatch as dispatch,
    effects as effects,
    util as util,
    xla_bridge as xla_bridge,
)
from jax._src.frozen_dict import FrozenDict as FrozenDict
from jax._src.hashable_array import HashableArray as HashableArray
from jax._src.interpreters import (
    ad as ad,
    batching as batching,
    mlir as mlir,
)
from jax._src.layout import Layout as Layout
from jax._src.lib import (
    jaxlib as jaxlib,
    xla_client as xla_client,
)
from jax._src.lib.mlir import ir as ir
from jax._src.typing import (
    Array as Array,
    ArrayLike as ArrayLike,
    DeprecatedArg as DeprecatedArg,
    DuckTypedArray as DuckTypedArray,
    Shape as Shape,
)

map: Incomplete
unsafe_map: Incomplete
type FfiLayoutOptions = Sequence[int] | Layout | None

def register_ffi_target(
    name: str,
    fn: Any,
    platform: str = "cpu",
    api_version: int = 1,
    **kwargs: Any,
) -> None: ...

class TypeRegistration(TypedDict):
    type_id: Any
    type_info: NotRequired[Any]

def register_ffi_type_id(name: str, obj: Any, platform: str = "cpu") -> None: ...
def register_ffi_type(
    name: str,
    type_registration: TypeRegistration,
    platform: str = "cpu",
) -> None: ...
def register_ffi_target_as_batch_partitionable(name: str) -> None: ...
def pycapsule(funcptr): ...
def include_dir() -> str: ...
def build_ffi_lowering_function(
    call_target_name: str,
    *,
    operand_layouts: Sequence[FfiLayoutOptions] | None = None,
    result_layouts: Sequence[FfiLayoutOptions] | None = None,
    backend_config: Mapping[str, ir.Attribute] | str | None = None,
    skip_ffi_layout_processing: bool = False,
    **lowering_args: Any,
) -> Callable[..., ir.Operation]: ...
def ffi_lowering(
    call_target_name: str,
    *,
    operand_layouts: Sequence[FfiLayoutOptions] | None = None,
    result_layouts: Sequence[FfiLayoutOptions] | None = None,
    backend_config: Mapping[str, ir.Attribute] | str | None = None,
    skip_ffi_layout_processing: bool = False,
    **lowering_args: Any,
) -> mlir.LoweringRule: ...

ResultMetadata: Incomplete

@overload
def ffi_call(
    target_name: str,
    result_shape_dtypes: ResultMetadata,
    *,
    has_side_effect: bool = ...,
    vmap_method: str | None = ...,
    input_layouts: Sequence[FfiLayoutOptions] | None = ...,
    output_layouts: FfiLayoutOptions | Sequence[FfiLayoutOptions] | None = ...,
    input_output_aliases: dict[int, int] | None = ...,
    custom_call_api_version: int = ...,
    legacy_backend_config: str | None = ...,
    vectorized: bool | DeprecatedArg | None = ...,
) -> Callable[..., Array]: ...
@overload
def ffi_call(
    target_name: str,
    result_shape_dtypes: Sequence[ResultMetadata],
    *,
    has_side_effect: bool = ...,
    vmap_method: str | None = ...,
    input_layouts: Sequence[FfiLayoutOptions] | None = ...,
    output_layouts: FfiLayoutOptions | Sequence[FfiLayoutOptions] | None = ...,
    input_output_aliases: dict[int, int] | None = ...,
    custom_call_api_version: int = ...,
    legacy_backend_config: str | None = ...,
    vectorized: bool | DeprecatedArg | None = ...,
) -> Callable[..., Sequence[Array]]: ...

@dataclasses.dataclass(frozen=True)
class FfiEffect(effects.Effect): ...

def ffi_call_abstract_eval(
    *avals_in,
    result_avals: tuple[core.AbstractValue, ...],
    has_side_effect: bool,
    **_,
): ...
def ffi_call_jvp(*args, target_name, **_) -> None: ...
def ffi_call_transpose(*args, target_name, **_) -> None: ...
def ffi_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *operands: ir.Value,
    target_name: str,
    has_side_effect: bool,
    input_layouts: Sequence[Sequence[int]],
    output_layouts: Sequence[Sequence[int]],
    input_output_aliases: Sequence[tuple[int, int]],
    custom_call_api_version: int,
    legacy_backend_config: str | None,
    attributes: Sequence[tuple[str, Any]],
    **_,
) -> Sequence[ir.Value]: ...
def ffi_batching_rule(
    prim,
    args,
    dims,
    *,
    vmap_method: str | None,
    result_avals: Sequence[core.ShapedArray],
    **kwargs: Any,
): ...

ffi_call_p: Incomplete
