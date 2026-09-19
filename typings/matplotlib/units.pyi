from typing import Any

from numpy.typing import NDArray

class ConversionError(TypeError): ...

class AxisInfo:
    def __init__(
        self,
        majloc=...,
        minloc=...,
        majfmt=...,
        minfmt=...,
        label=...,
        default_limits=...,
    ) -> None: ...

class ConversionInterface:
    @staticmethod
    def axisinfo(unit, axis) -> None: ...
    @staticmethod
    def default_units(x, axis) -> None: ...
    @staticmethod
    def convert(obj, unit, axis): ...

class DecimalConverter(ConversionInterface):
    @staticmethod
    def convert(value, unit, axis) -> float | _MaskedArray[Any] | NDArray[Any]: ...

class Registry(dict):
    def get_converter(self, x) -> None: ...

registry = ...
