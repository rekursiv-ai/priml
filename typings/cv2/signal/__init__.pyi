import typing as _typing

import cv2
import cv2.typing

__all__: list[str] = ...

@_typing.overload
def resampleSignal(
    inputSignal: cv2.typing.MatLike,
    inFreq: int,
    outFreq: int,
    outSignal: cv2.typing.MatLike | None = ...,
) -> cv2.typing.MatLike: ...
@_typing.overload
def resampleSignal(
    inputSignal: cv2.UMat,
    inFreq: int,
    outFreq: int,
    outSignal: cv2.UMat | None = ...,
) -> cv2.UMat: ...
