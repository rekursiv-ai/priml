from collections.abc import Generator
from typing import Any

import dataclasses
import threading
import types
import typing as tp

from _typeshed import Incomplete
from flax import (
    config as config,
    errors as errors,
)
from flax.nnx import (
    reprlib as reprlib,
    tracers as tracers,
    visualization as visualization,
)
from flax.typing import (
    MISSING as MISSING,
    BaseConfigContext as BaseConfigContext,
    Missing as Missing,
    SizeBytes as SizeBytes,
)
from jax.experimental import (
    MutableArray as Ref,
    hijax as hjx,
)

import jax

A = tp.TypeVar("A")
B = tp.TypeVar("B")
C = tp.TypeVar("C")
F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])
P = tp.TypeVar("P", bound=property)
V = tp.TypeVar("V", bound=Variable[Any])
GetValueHook: Incomplete
SetValueHook: Incomplete
CreateValueHook: Incomplete
AxisName = str
AxisIndex = int
type AddAxisHook[V: Variable[Any]] = tp.Callable[[V, AxisIndex, AxisName | None], None]
type RemoveAxisHook[V: Variable[Any]] = tp.Callable[
    [V, AxisIndex, AxisName | None],
    None,
]

@dataclasses.dataclass
class VariableContext(threading.local):
    variable_hijax_stack: list[bool] = dataclasses.field(default_factory=list)
    variable_ref_stack: list[bool] = dataclasses.field(default_factory=list)
    eager_shard_stack: list[bool] = dataclasses.field(default_factory=list)

VARIABLE_CONTEXT: Incomplete

class use_eager_sharding(BaseConfigContext):
    get_default: Incomplete
    get_stack: Incomplete

def using_eager_sharding() -> bool: ...

@dataclasses.dataclass(frozen=True)
class VarDefaults(tp.Mapping[str, tp.Any]):
    hijax: bool
    ref: bool
    def __getitem__(self, key: str) -> tp.Any: ...
    def __iter__(self) -> tp.Iterator[str]: ...
    def __len__(self) -> int: ...

@tp.overload
def var_defaults() -> VarDefaults: ...
@tp.overload
def var_defaults(
    *,
    hijax: bool | None = None,
    ref: bool | None = None,
) -> VarDefaultsContext: ...

class VarDefaultsContext:
    hijax_prev: Incomplete
    hijax_new: Incomplete
    ref_prev: Incomplete
    ref_new: Incomplete
    def __init__(
        self,
        *,
        hijax_prev: bool | None,
        hijax_new: bool | None,
        ref_prev: bool | None,
        ref_new: bool | None,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
    def __call__(self, f: F) -> F: ...

def is_array_ref(x) -> tp.TypeGuard[Ref]: ...

@dataclasses.dataclass
class VariableMetadata(tp.Generic[A]):
    raw_value: A
    set_value_hooks: tuple[SetValueHook[A], ...] = ...
    get_value_hooks: tuple[GetValueHook[A], ...] = ...
    create_value_hooks: tuple[CreateValueHook[A], ...] = ...
    add_axis_hooks: tuple[AddAxisHook[Variable[A]], ...] = ...
    remove_axis_hooks: tuple[RemoveAxisHook[Variable[A]], ...] = ...
    metadata: tp.Mapping[str, tp.Any] = dataclasses.field(default_factory=dict)

type PyTreeDef = tp.Any
type Leaf = tp.Any

@dataclasses.dataclass(frozen=True)
class VariableQDD:
    leaf_avals: tuple[hjx.AbstractValue, ...]
    treedef: PyTreeDef
    var_type: type[Variable[Any]]
    def to_tangent_qdd(self): ...
    def normalize(self): ...

class VariableEffect(jax.core.Effect): ...

variable_effect: Incomplete

class NewVariable(hjx.HiPrimitive):
    def is_high(self, *leaves, treedef, var_type, has_qdd, ref) -> bool: ...
    def impl(self, *leaves, treedef, var_type, has_qdd, ref): ...
    def abstract_eval(self, *leaves, treedef, var_type, has_qdd, ref): ...
    def to_lojax(self, *leaves, treedef, var_type, has_qdd, ref): ...
    def jvp(self, primals, tangents, *, treedef, var_type, has_qdd, ref): ...
    def transpose(
        self,
        out_var: HijaxVariable,
        *input_leaves,
        treedef,
        var_type,
        has_qdd,
        ref,
    ): ...

new_variable_p: Incomplete

class SetVariable(hjx.HiPrimitive):
    multiple_results: bool
    def is_high(self, *leaf_avals, treedef, var_type) -> bool: ...
    def impl(self, hijax_var: HijaxVariable, *leaves, treedef, var_type): ...
    def abstract_eval(
        self,
        aval_mutable_qdd: hjx.AvalMutableQDD,
        *leaf_avals,
        treedef,
        var_type,
    ): ...
    def to_lojax(self, hijax_var: HijaxVariable, *leaves, treedef, var_type): ...
    def jvp(self, primals, tangents, *, treedef, var_type): ...
    def transpose(self, *args, treedef, var_type) -> None: ...

set_variable_p: Incomplete

class GetVariable(hjx.HiPrimitive):
    multiple_results: bool
    def impl(self, hijax_var: HijaxVariable, *, treedef, avals, var_type, has_qdd): ...
    def abstract_eval(self, abstract_var, *, treedef, avals, var_type, has_qdd): ...
    def to_lojax(
        self,
        hijax_var: HijaxVariable,
        *,
        treedef,
        avals,
        var_type,
        has_qdd,
    ): ...
    def jvp(self, primals, tangents, *, treedef, avals, var_type, has_qdd): ...
    def transpose(self, out, hijax_var, *, treedef, avals, var_type, has_qdd): ...

get_variable_p: Incomplete

class HijaxVariableMeta(type):
    def __instancecheck__(cls, instance): ...

class HijaxVariable(reprlib.Representable, tp.Generic[A], metaclass=HijaxVariableMeta):
    has_qdd: bool
    __init__: Incomplete
    @property
    def value(self) -> A: ...
    @value.setter
    def value(self, new_value: A): ...
    @property
    def var_type(self) -> type[Variable[A]]: ...
    __getattr__: Incomplete
    __setattr__: Incomplete
    __delattr__: Incomplete
    type: Incomplete
    hijax: Incomplete
    @property
    def ref(self) -> bool: ...
    get_metadata: Incomplete
    set_metadata: Incomplete
    def copy_from(self, other: Variable[A] | HijaxVariable[A]) -> None: ...
    def update_from_state(self, variable_state: Variable[A] | HijaxVariable[A]): ...
    get_raw_value: Incomplete
    set_raw_value: Incomplete
    set_value: Incomplete
    get_value: Incomplete
    create_value: Incomplete
    add_axis: Incomplete
    remove_axis: Incomplete
    copy: Incomplete
    replace: Incomplete
    to_state: Incomplete
    @classmethod
    def from_metadata(cls, value: A, metadata: dict[str, tp.Any]): ...
    __nnx_repr__: Incomplete
    __treescope_repr__: Incomplete
    __jax_array__: Incomplete
    __getitem__: Incomplete
    __setitem__: Incomplete
    __delitem__: Incomplete
    __call__: Incomplete
    __len__: Incomplete
    __iter__: Incomplete
    __contains__: Incomplete
    __add__: Incomplete
    __sub__: Incomplete
    __mul__: Incomplete
    __matmul__: Incomplete
    __truediv__: Incomplete
    __floordiv__: Incomplete
    __mod__: Incomplete
    __divmod__: Incomplete
    __pow__: Incomplete
    __lshift__: Incomplete
    __rshift__: Incomplete
    __and__: Incomplete
    __xor__: Incomplete
    __or__: Incomplete
    __radd__: Incomplete
    __rsub__: Incomplete
    __rmul__: Incomplete
    __rmatmul__: Incomplete
    __rtruediv__: Incomplete
    __rfloordiv__: Incomplete
    __rmod__: Incomplete
    __rdivmod__: Incomplete
    __rpow__: Incomplete
    __rlshift__: Incomplete
    __rrshift__: Incomplete
    __rand__: Incomplete
    __rxor__: Incomplete
    __ror__: Incomplete
    __iadd__: Incomplete
    __isub__: Incomplete
    __imul__: Incomplete
    __imatmul__: Incomplete
    __itruediv__: Incomplete
    __ifloordiv__: Incomplete
    __imod__: Incomplete
    __ipow__: Incomplete
    __ilshift__: Incomplete
    __irshift__: Incomplete
    __iand__: Incomplete
    __ixor__: Incomplete
    __ior__: Incomplete
    __neg__: Incomplete
    __pos__: Incomplete
    __abs__: Incomplete
    __invert__: Incomplete
    __complex__: Incomplete
    __int__: Incomplete
    __float__: Incomplete
    __index__: Incomplete
    __round__: Incomplete
    __trunc__: Incomplete
    __floor__: Incomplete
    __ceil__: Incomplete
    def cur_qdd(self): ...
    def type_state(self): ...

class AbstractVariable(hjx.MutableHiType, tp.Generic[A]):
    has_qdd: bool
    @property
    def ref(self) -> bool: ...
    @property
    def hijax(self): ...
    def __init__(
        self,
        var_type: type[Variable[A]],
        treedef: PyTreeDef | None,
        leaves: tuple[hjx.AbstractValue, ...] | None,
        has_qdd: bool,
        *,
        ref: bool = False,
    ) -> None: ...
    @property
    def dtype(self) -> None: ...
    @property
    def ndim(self) -> None: ...
    @property
    def size(self) -> None: ...
    @property
    def shape(self) -> None: ...
    def __getattr__(self, name: str): ...
    type: Incomplete
    get_metadata: Incomplete
    set_metadata: Incomplete
    copy_from: Incomplete
    update_from_state: Incomplete
    get_raw_value: Incomplete
    set_raw_value: Incomplete
    set_value: Incomplete
    get_value: Incomplete
    create_value: Incomplete
    add_axis: Incomplete
    remove_axis: Incomplete
    replace: Incomplete
    @hjx.aval_method
    def from_metadata(self, value, metadata: dict[str, tp.Any]): ...
    copy: Incomplete
    to_state: Incomplete
    @hjx.aval_method
    def __treescope_repr__(self, path, subtree_renderer) -> None: ...
    __jax_array__: Incomplete
    cur_qdd: Incomplete
    def __hash__(self): ...
    def __eq__(self, other): ...
    def str_short(self, short_dtypes: bool = False, **_) -> str: ...
    def lo_ty_qdd(self, variable_state: VariableQDD) -> list: ...
    def new_from_loval(
        self,
        variable_state: VariableQDD,
        *lo_vals,
    ) -> HijaxVariable: ...
    def read_loval(self, variable_state: VariableQDD, variable) -> list: ...
    def update_from_loval(self, box_state: VariableQDD, variable, *lo_vals) -> None: ...
    def to_tangent_aval(self): ...

class VariableMeta(type):
    def __new__(cls, cls_name, bases, attrs): ...
    def __instancecheck__(cls, instance): ...

class Variable(reprlib.Representable, tp.Generic[A], metaclass=VariableMeta):
    required_metadata: Incomplete
    @property
    def var_type(self): ...
    @property
    def hijax(self) -> bool: ...
    @property
    def ref(self) -> bool: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def sharding_names(self): ...
    def __init__(
        self,
        value: A | VariableMetadata[A],
        *,
        hijax: bool | None = None,
        ref: bool | None = None,
        eager_sharding: bool | None = None,
        **metadata: tp.Any,
    ) -> None: ...
    def __getattr__(self, name: str) -> tp.Any: ...
    def __setattr__(self, name: str, value: tp.Any): ...
    def __delattr__(self, name: str): ...
    @property
    def type(self): ...
    @tp.overload
    def get_metadata(self, *, exclude_required: bool = False) -> dict[str, tp.Any]: ...
    @tp.overload
    def get_metadata(self, name: str, default: tp.Any = ...) -> tp.Any: ...
    @tp.overload
    def set_metadata(self, metadata: dict[str, tp.Any], /) -> None: ...
    @tp.overload
    def set_metadata(self, name: str, value: tp.Any, /) -> None: ...
    @tp.overload
    def set_metadata(self, **metadata: tp.Any) -> None: ...
    def has_metadata(self, name: str) -> bool: ...
    def del_metadata(self, name: str) -> None: ...
    def copy_from(self, other: Variable[A]) -> None: ...
    def update_from_state(self, variable_state: Variable[A]): ...
    @tp.final
    def get_raw_value(self) -> A: ...
    def set_raw_value(self, value: A, *, _unsafe_bypass_check: bool = False): ...
    @property
    def raw_value(self) -> A: ...
    @raw_value.setter
    def raw_value(self, value: A): ...
    @property
    def value(self) -> A: ...
    @value.setter
    def value(self, value: A): ...
    def create_value(self, value: A): ...
    def get_value(self, *, index: tp.Any = ...) -> A: ...
    def set_value(self, value: A, *, index: tp.Any = ...): ...
    def add_axis(self, axis_index: AxisIndex, axis_name: AxisName | None): ...
    def remove_axis(self, axis_index: AxisIndex, axis_name: AxisName | None): ...
    @tp.overload
    def copy(self, value: B, **kwargs) -> Variable[B]: ...
    @tp.overload
    def copy(self, **kwargs) -> Variable[A]: ...
    @classmethod
    def from_metadata(cls, value: A, attributes: dict[str, tp.Any]) -> Variable[A]: ...
    replace = copy
    to_state = copy
    def __nnx_repr__(self) -> Generator[Incomplete]: ...
    def __treescope_repr__(self, path, subtree_renderer): ...
    def on_get_value(self, value: A) -> A: ...
    def on_set_value(self, value: A) -> A: ...
    def on_create_value(self, value: A) -> A: ...
    def on_add_axis(self, axis_index: AxisIndex, axis_name: AxisName | None) -> V: ...
    def on_remove_axis(
        self,
        axis_index: AxisIndex,
        axis_name: AxisName | None,
    ) -> V: ...
    def __jax_array__(self): ...
    @tp.overload
    def __getitem__(self, key) -> jax.Array: ...
    @tp.overload
    def __getitem__(self, key) -> B: ...
    @tp.overload
    def __getitem__(self, key: int) -> B: ...
    @tp.overload
    def __getitem__(self, key: int) -> B: ...
    @tp.overload
    def __getitem__(self, key) -> tp.Any: ...
    def __setitem__(self, key, value) -> None: ...
    def __delitem__(self, key) -> None: ...
    def __call__(self, *args, **kwargs) -> tp.Any: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> tp.Iterator: ...
    def __contains__(self, item) -> bool: ...
    __add__: Incomplete
    __sub__: Incomplete
    __mul__: Incomplete
    __matmul__: Incomplete
    __truediv__: Incomplete
    __floordiv__: Incomplete
    __mod__: Incomplete
    __pow__: Incomplete
    __lshift__: Incomplete
    __rshift__: Incomplete
    __and__: Incomplete
    __xor__: Incomplete
    __or__: Incomplete
    __radd__: Incomplete
    __rsub__: Incomplete
    __rmul__: Incomplete
    __rmatmul__: Incomplete
    __rtruediv__: Incomplete
    __rfloordiv__: Incomplete
    __rmod__: Incomplete
    __rpow__: Incomplete
    __rlshift__: Incomplete
    __rrshift__: Incomplete
    __rand__: Incomplete
    __rxor__: Incomplete
    __ror__: Incomplete
    def __eq__(self, other) -> bool: ...
    def __iadd__(self, other) -> V: ...
    def __isub__(self, other) -> V: ...
    def __imul__(self, other) -> V: ...
    def __imatmul__(self, other) -> V: ...
    def __itruediv__(self, other) -> V: ...
    def __ifloordiv__(self, other) -> V: ...
    def __imod__(self, other) -> V: ...
    def __ipow__(self, other) -> V: ...
    def __ilshift__(self, other) -> V: ...
    def __irshift__(self, other) -> V: ...
    def __iand__(self, other) -> V: ...
    def __ixor__(self, other) -> V: ...
    def __ior__(self, other) -> V: ...
    __neg__: Incomplete
    __pos__: Incomplete
    __abs__: Incomplete
    __invert__: Incomplete
    __complex__: Incomplete
    __int__: Incomplete
    __float__: Incomplete
    __index__: Incomplete
    __trunc__: Incomplete
    __floor__: Incomplete
    __ceil__: Incomplete
    def __round__(self, ndigits: int = 0) -> A: ...
    def __init_subclass__(cls) -> None: ...

VariableState = Variable

class Param(Variable[A]): ...
class BatchStat(Variable[A]): ...
class Cache(Variable[A]): ...
class Intermediate(Variable[A]): ...
class Perturbation(Intermediate[A]): ...

def with_metadata(
    initializer: F,
    set_value_hooks: SetValueHook[A] | tp.Sequence[SetValueHook[A]] = (),
    get_value_hooks: SetValueHook[A] | tp.Sequence[SetValueHook[A]] = (),
    create_value_hooks: CreateValueHook[A] | tp.Sequence[CreateValueHook[A]] = (),
    add_axis_hooks: AddAxisHook[Variable[A]]
    | tp.Sequence[AddAxisHook[Variable[A]]] = (),
    remove_axis_hooks: RemoveAxisHook[Variable[A]]
    | tp.Sequence[RemoveAxisHook[Variable[A]]] = (),
    **metadata: tp.Any,
) -> F: ...

VariableTypeCache: dict[str, type[Variable[tp.Any]]]

def variable_type_from_name(
    name: str,
    /,
    *,
    base: type[Variable[tp.Any]] = ...,
    allow_register: bool = False,
) -> type[Variable[tp.Any]]: ...
def variable_name_from_type(
    typ: type[Variable[tp.Any]],
    /,
    *,
    allow_register: bool = False,
) -> str: ...
@tp.overload
def register_variable_name(
    name: str,
    typ: type[Variable[tp.Any]],
    *,
    overwrite: bool = False,
) -> type[Variable[tp.Any]]: ...
@tp.overload
def register_variable_name(
    name: str,
    *,
    overwrite: bool = False,
) -> tp.Callable[[type[Variable[tp.Any]]], type[Variable[tp.Any]]]: ...
