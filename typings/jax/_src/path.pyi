from typing import Protocol

import os
import pathlib

__all__ = ["Path"]

class PathProtocol(Protocol):
    def __call__(self, *pathsegments: str | os.PathLike) -> pathlib.Path: ...

Path: PathProtocol
Path = pathlib.Path
