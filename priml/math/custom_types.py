"""Tensor type aliases and the dtype-coercion entry points.

Lives under ``priml.math`` because every consumer is a math/data-
processing module. Carved out of ``priml.lib.custom_types`` so callers
that only need the torch-free pieces (sentinels, sequence aliases,
checkpoint/job Protocols) don't pay the ~1.2s torch + jaxtyping import
on startup.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import functools

from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

import numpy as np

# AUTHORIZED STYLE DEVIATION -- STYLE.md forbids a non-``__init__`` module
# re-exporting a name it does not define. Granted for ``convert_to_tensor``
# ALONE, because it and ``Tensorable`` are one concept: the alias exists to
# name what the converter accepts, so a caller importing the type almost
# always wants the converter in the same breath. The aliasing predicates
# sitting beside it in :mod:`priml.memory` have no such tie and are
# deliberately not re-exported -- import those from the module defining them.
from priml.memory import convert_to_tensor


__all__ = [
    "Numeric",
    "TensorFn",
    "TensorNest",
    "Tensorable",
    "TensorableFn",
    "TensorableNest",
    "convert_to_tensor",
    "jaxtypechecked",
]


jaxtypechecked = functools.partial(jaxtyped, typechecker=typechecked)


Numeric = bool | int | float | complex | np.number | np.bool_

# "Tensorable" follows Python's -able convention (Callable, Hashable, Iterable)
# and avoids collision with torch._prims_common.TensorLike.
Tensorable = Sequence["Tensorable"] | np.ndarray[Any, Any] | Tensor | Numeric
TensorableNest = (
    Sequence["TensorableNest"] | Mapping[str, "TensorableNest"] | Tensorable
)
TensorNest = Sequence["TensorNest"] | Mapping[str, "TensorNest"] | Tensor


type TensorFn = Callable[[Tensor], Tensor]
"""TITO -- tensor in, tensor out.

Mathematically an *endomorphism* (a map from a set into itself), or more
precisely an endofunction, since the set here is one of values rather than an
arbitrary category. Both words are avoided: they are fancier than the thing they
name, and a reader who looks one up learns nothing that ``Tensor -> Tensor`` did
not already say.
"""


type TensorableFn = Callable[[Tensorable], Tensor]
"""The usual ``priml.math`` shape: accept anything coercible, return a Tensor.

Not TITO: the domain (lists, arrays, scalars, tensors) is wider than the
codomain, so this is not a :data:`TensorFn`. Such a function composes with
itself only because ``Tensor`` is one of the things ``Tensorable`` admits. It IS
assignable where a ``TensorFn`` is wanted, since parameters are contravariant;
the reverse is rejected.
"""
