from typing import Any, TypeVar

import typing_extensions as tpe

from _typeshed import Incomplete

M = TypeVar("M", bound=flax.linen.Module)
FieldName = str
type Annotation = Any
type Default = Any

class _KwOnlyType: ...

KW_ONLY: Incomplete

def field(*, metadata=None, kw_only=..., **kwargs): ...
@tpe.dataclass_transform(field_specifiers=(field,))
def dataclass(cls=None, extra_fields=None, **kwargs): ...
