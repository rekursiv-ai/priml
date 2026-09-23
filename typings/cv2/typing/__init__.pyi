import typing as _typing

import cv2
import cv2.dnn
import cv2.gapi.wip.draw
import cv2.mat_wrapper
import numpy as np

__all__ = [
    "DescriptorExtractor",
    "ExtractArgsCallback",
    "ExtractMetaCallback",
    "FeatureDetector",
    "FeatureExtractor",
    "GMetaArg",
    "GOptRunArg",
    "GProtoArg",
    "GProtoInputArgs",
    "GProtoOutputArgs",
    "GRunArg",
    "GTypeInfo",
    "IndexParams",
    "IntPointer",
    "LayerId",
    "LayerParams",
    "MatLike",
    "MatShape",
    "Matx33d",
    "Matx33f",
    "Matx44d",
    "Matx44f",
    "Moments",
    "Point",
    "Point2d",
    "Point2f",
    "Point2i",
    "Point3d",
    "Point3f",
    "Point3i",
    "Prim",
    "Range",
    "Rect",
    "Rect2d",
    "Rect2f",
    "Rect2i",
    "RotatedRect",
    "Scalar",
    "SearchParams",
    "Size",
    "Size2f",
    "TermCriteria",
    "Vec2d",
    "Vec2f",
    "Vec2i",
    "Vec3d",
    "Vec3f",
    "Vec3i",
    "Vec4d",
    "Vec4f",
    "Vec4i",
    "Vec6f",
    "map_int_and_double",
    "map_string_and_int",
    "map_string_and_string",
    "map_string_and_vector_float",
    "map_string_and_vector_size_t",
]
type NumPyArrayNumeric = np.ndarray[
    tuple[int, ...],
    np.dtype[np.integer[_typing.Any] | np.floating[_typing.Any]],
]
type NumPyArrayFloat32 = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type NumPyArrayFloat64 = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
TermCriteria_Type = cv2.TermCriteria_Type
IntPointer = int
type MatLike = cv2.mat_wrapper.Mat | NumPyArrayNumeric
type MatShape = _typing.Sequence[int]
type Size = _typing.Sequence[int]
type Size2f = _typing.Sequence[float]
type Scalar = _typing.Sequence[float] | float
type Point = _typing.Sequence[int]
Point2i = Point
type Point2f = _typing.Sequence[float]
type Point2d = _typing.Sequence[float]
type Point3i = _typing.Sequence[int]
type Point3f = _typing.Sequence[float]
type Point3d = _typing.Sequence[float]
type Range = _typing.Sequence[int]
type Rect = _typing.Sequence[int]
type Rect2i = _typing.Sequence[int]
type Rect2f = _typing.Sequence[float]
type Rect2d = _typing.Sequence[float]
type Moments = dict[str, float]
type RotatedRect = tuple[Point2f, Size2f, float]
type TermCriteria = tuple[TermCriteria_Type, int, float]
type Vec2i = _typing.Sequence[int]
type Vec2f = _typing.Sequence[float]
type Vec2d = _typing.Sequence[float]
type Vec3i = _typing.Sequence[int]
type Vec3f = _typing.Sequence[float]
type Vec3d = _typing.Sequence[float]
type Vec4i = _typing.Sequence[int]
type Vec4f = _typing.Sequence[float]
type Vec4d = _typing.Sequence[float]
type Vec6f = _typing.Sequence[float]
FeatureDetector = cv2.Feature2D
DescriptorExtractor = cv2.Feature2D
FeatureExtractor = cv2.Feature2D
Matx33f = NumPyArrayFloat32
Matx33d = NumPyArrayFloat64
Matx44f = NumPyArrayFloat32
Matx44d = NumPyArrayFloat64
LayerId = cv2.dnn.DictValue
type LayerParams = dict[str, int | float | str]
type IndexParams = _typing.Mapping[str, bool | int | float | str]
type SearchParams = _typing.Mapping[str, bool | int | float | str]
type map_string_and_string = dict[str, str]
type map_string_and_int = dict[str, int]
type map_string_and_vector_size_t = dict[str, _typing.Sequence[int]]
type map_string_and_vector_float = dict[str, _typing.Sequence[float]]
type map_int_and_double = dict[int, float]
type GProtoArg = Scalar | cv2.GMat | cv2.GOpaqueT | cv2.GArrayT
type GProtoInputArgs = _typing.Sequence[GProtoArg]
type GProtoOutputArgs = _typing.Sequence[GProtoArg]
type GRunArg = (
    MatLike | Scalar | cv2.GOpaqueT | cv2.GArrayT | _typing.Sequence[_typing.Any] | None
)
type GOptRunArg = GRunArg | None
type GMetaArg = cv2.GMat | Scalar | cv2.GOpaqueT | cv2.GArrayT
type Prim = (
    cv2.gapi.wip.draw.Text
    | cv2.gapi.wip.draw.Circle
    | cv2.gapi.wip.draw.Image
    | cv2.gapi.wip.draw.Line
    | cv2.gapi.wip.draw.Rect
    | cv2.gapi.wip.draw.Mosaic
    | cv2.gapi.wip.draw.Poly
)
type GTypeInfo = cv2.GMat | Scalar | cv2.GOpaqueT | cv2.GArrayT
type ExtractArgsCallback = _typing.Callable[
    [_typing.Sequence[GTypeInfo]],
    _typing.Sequence[GRunArg],
]
type ExtractMetaCallback = _typing.Callable[
    [_typing.Sequence[GTypeInfo]],
    _typing.Sequence[GMetaArg],
]
