from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt

__all__ = ["imiter", "imread", "imwrite"]

def imread(
    uri: str | bytes | Path | BinaryIO,
    *,
    index: int | None = ...,
    plugin: str | None = ...,
    extension: str | None = ...,
    format_hint: str | None = ...,
    **kwargs: Any,
) -> npt.NDArray[np.uint8]: ...
def imwrite(
    uri: str | bytes | Path | BinaryIO,
    image: npt.ArrayLike,
    *,
    plugin: str | None = ...,
    extension: str | None = ...,
    format_hint: str | None = ...,
    **kwargs: Any,
) -> None: ...
def imiter(
    uri: str | bytes | Path | BinaryIO,
    *,
    plugin: str | None = ...,
    extension: str | None = ...,
    format_hint: str | None = ...,
    **kwargs: Any,
) -> Any: ...
