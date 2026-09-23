import os
import typing as _typing

import cv2
import cv2.typing

__all__: list[str] = ...
FLANN_INDEX_TYPE_8U: int
FLANN_INDEX_TYPE_8S: int
FLANN_INDEX_TYPE_16U: int
FLANN_INDEX_TYPE_16S: int
FLANN_INDEX_TYPE_32S: int
FLANN_INDEX_TYPE_32F: int
FLANN_INDEX_TYPE_64F: int
FLANN_INDEX_TYPE_STRING: int
FLANN_INDEX_TYPE_BOOL: int
FLANN_INDEX_TYPE_ALGORITHM: int
LAST_VALUE_FLANN_INDEX_TYPE: int
FlannIndexType = int

class Index:
    @_typing.overload
    def __init__(self) -> None: ...
    @_typing.overload
    def __init__(
        self,
        features: cv2.typing.MatLike,
        params: cv2.typing.IndexParams,
        distType: int = ...,
    ) -> None: ...
    @_typing.overload
    def __init__(
        self,
        features: cv2.UMat,
        params: cv2.typing.IndexParams,
        distType: int = ...,
    ) -> None: ...
    @_typing.overload
    def build(
        self,
        features: cv2.typing.MatLike,
        params: cv2.typing.IndexParams,
        distType: int = ...,
    ) -> None: ...
    @_typing.overload
    def build(
        self,
        features: cv2.UMat,
        params: cv2.typing.IndexParams,
        distType: int = ...,
    ) -> None: ...
    @_typing.overload
    def knnSearch(
        self,
        query: cv2.typing.MatLike,
        knn: int,
        indices: cv2.typing.MatLike | None = ...,
        dists: cv2.typing.MatLike | None = ...,
        params: cv2.typing.SearchParams = ...,
    ) -> tuple[cv2.typing.MatLike, cv2.typing.MatLike]: ...
    @_typing.overload
    def knnSearch(
        self,
        query: cv2.UMat,
        knn: int,
        indices: cv2.UMat | None = ...,
        dists: cv2.UMat | None = ...,
        params: cv2.typing.SearchParams = ...,
    ) -> tuple[cv2.UMat, cv2.UMat]: ...
    @_typing.overload
    def radiusSearch(
        self,
        query: cv2.typing.MatLike,
        radius: float,
        maxResults: int,
        indices: cv2.typing.MatLike | None = ...,
        dists: cv2.typing.MatLike | None = ...,
        params: cv2.typing.SearchParams = ...,
    ) -> tuple[int, cv2.typing.MatLike, cv2.typing.MatLike]: ...
    @_typing.overload
    def radiusSearch(
        self,
        query: cv2.UMat,
        radius: float,
        maxResults: int,
        indices: cv2.UMat | None = ...,
        dists: cv2.UMat | None = ...,
        params: cv2.typing.SearchParams = ...,
    ) -> tuple[int, cv2.UMat, cv2.UMat]: ...
    def save(self, filename: str | os.PathLike[str]) -> None: ...
    @_typing.overload
    def load(
        self,
        features: cv2.typing.MatLike,
        filename: str | os.PathLike[str],
    ) -> bool: ...
    @_typing.overload
    def load(self, features: cv2.UMat, filename: str | os.PathLike[str]) -> bool: ...
    def release(self) -> None: ...
    def getDistance(self) -> int: ...
    def getAlgorithm(self) -> int: ...
