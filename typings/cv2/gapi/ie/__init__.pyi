import typing as _typing

from cv2.gapi.ie import detail as detail

import cv2.typing

__all__: list[str] = ...
TraitAs_TENSOR: int
TRAIT_AS_TENSOR: int
TraitAs_IMAGE: int
TRAIT_AS_IMAGE: int
TraitAs = int
Sync: int
SYNC: int
Async: int
ASYNC: int
InferMode = int

class PyParams:
    @_typing.overload
    def __init__(self) -> None: ...
    @_typing.overload
    def __init__(self, tag: str, model: str, weights: str, device: str) -> None: ...
    @_typing.overload
    def __init__(self, tag: str, model: str, device: str) -> None: ...
    def constInput(
        self,
        layer_name: str,
        data: cv2.typing.MatLike,
        hint: TraitAs = ...,
    ) -> PyParams: ...
    def cfgNumRequests(self, nireq: int) -> PyParams: ...
    def cfgBatchSize(self, size: int) -> PyParams: ...

@_typing.overload
def params(tag: str, model: str, weights: str, device: str) -> PyParams: ...
@_typing.overload
def params(tag: str, model: str, device: str) -> PyParams: ...
