from typing import NamedTuple

from optax._src import base as base

import chex

class SplitRealAndImaginaryArrays(NamedTuple):
    real: chex.Array
    imaginary: chex.Array

class SplitRealAndImaginaryState(NamedTuple):
    inner_state: base.OptState

def split_real_and_imaginary(
    inner: base.GradientTransformation,
) -> base.GradientTransformation: ...
