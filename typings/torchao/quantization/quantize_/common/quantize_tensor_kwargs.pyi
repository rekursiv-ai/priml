from typing import ClassVar

import abc

__all__ = ["QuantizeTensorKwargs", "_choose_quant_func_and_quantize_tensor"]

class QuantizeTensorKwargs(abc.ABC):
    VERSION: ClassVar[int] = ...
