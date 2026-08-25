import importlib.metadata
import importlib.util
import typing

from jax import Array as Array
from jax.tree_util import PyTreeDef as PyTreeDef
from jax.typing import (
    ArrayLike as ArrayLike,
    DTypeLike as DTypeLike,
)

from ._array_types import (
    AbstractArray as AbstractArray,
    AbstractDtype as AbstractDtype,
    get_array_name_format as get_array_name_format,
    make_numpy_struct_dtype as make_numpy_struct_dtype,
    set_array_name_format as set_array_name_format,
)
from ._config import config as config
from ._decorator import jaxtyped as jaxtyped
from ._errors import (
    AnnotationError as AnnotationError,
    TypeCheckError as TypeCheckError,
)
from ._import_hook import install_import_hook as install_import_hook
from ._indirection import (
    BFloat16 as BFloat16,
    Bool as Bool,
    Complex as Complex,
    Complex64 as Complex64,
    Complex128 as Complex128,
    Float as Float,
    Float8e4m3b11fnuz as Float8e4m3b11fnuz,
    Float8e4m3fn as Float8e4m3fn,
    Float8e4m3fnuz as Float8e4m3fnuz,
    Float8e5m2 as Float8e5m2,
    Float8e5m2fnuz as Float8e5m2fnuz,
    Float16 as Float16,
    Float32 as Float32,
    Float64 as Float64,
    Inexact as Inexact,
    Int as Int,
    Int2 as Int2,
    Int4 as Int4,
    Int8 as Int8,
    Int16 as Int16,
    Int32 as Int32,
    Int64 as Int64,
    Integer as Integer,
    Key as Key,
    Num as Num,
    PRNGKeyArray as PRNGKeyArray,
    Real as Real,
    Scalar as Scalar,
    ScalarLike as ScalarLike,
    Shaped as Shaped,
    UInt as UInt,
    UInt2 as UInt2,
    UInt4 as UInt4,
    UInt8 as UInt8,
    UInt16 as UInt16,
    UInt32 as UInt32,
    UInt64 as UInt64,
)
from ._ipython_extension import load_ipython_extension as load_ipython_extension
from ._storage import print_bindings as print_bindings

if typing.TYPE_CHECKING:
    type PyTree = getattr(typing, "foo" + "bar")
else: ...
if importlib.util.find_spec("typeguard") is not None: ...
__version__ = ...
