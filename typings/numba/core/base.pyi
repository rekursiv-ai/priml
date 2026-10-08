from llvmlite.ir import IRBuilder, Value
from numba.core.types import Array

import numpy as np

class BaseContext:
    def get_dummy_value(self) -> Value: ...
    def make_constant_array(
        self,
        builder: IRBuilder,
        typ: Array,
        ary: np.ndarray,
    ) -> Value: ...
