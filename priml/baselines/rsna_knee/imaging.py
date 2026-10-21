"""Physical DICOM preparation: geometry, side, anatomy, fluid, rendering and windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, cast, override

import csv
import math
import zlib

from configgle import Fig
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.rsna_knee.data import FINDINGS
from priml.data.na_values import PANDAS_NA_VALUES
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL import Image

    import cv2

    from priml.math.custom_types import TensorFn
else:
    from wrapt import lazy_import

    cv2 = lazy_import("cv2")
    Image = lazy_import("PIL.Image")


@dataclass(kw_only=True, slots=True)
class AnatomicalSeries:
    """Source geometry and complete four-tissue probability maps for one series."""

    series: str
    plane: str
    shape: tuple[int, int]
    affines: NDArray[np.float64] | None
    aligned: bool
    fluid: int | None
    probabilities: Sequence[tuple[int, NDArray[np.float32]]]
    reason: str = "ok"


@dataclass(kw_only=True, slots=True)
class ProjectedAnchor:
    """Joint and optional patella coordinates in fractions of a native image."""

    joint: tuple[float, float]
    in_field: bool
    distance_mm: float
    patella: tuple[float, float] | None


@dataclass(kw_only=True, slots=True)
class AnchorRecord:
    """Study-level patient-space consensus, diagnostics, and per-series projections."""

    joint: NDArray[np.float64] | None = None
    patella: NDArray[np.float64] | None = None
    prior: NDArray[np.float64] | None = None
    method: str | None = None
    reasons: list[str] = field(default_factory=list)
    series: dict[str, ProjectedAnchor] = field(default_factory=dict)


class AnchorConsensus:
    """Choose and project the released seven-slice tissue-centroid consensus."""

    class Config(Fig["AnchorConsensus"]):
        agreement_mm: float = 20.0
        """Maximum distance to the heaviest candidate for averaging."""

        prior_mm: float = 35.0
        """Maximum candidate distance from the orthogonal-stack slab prior."""

        mass_min: int = 250
        """Minimum summed confident joint-tissue pixels across selected slices."""

        tissue_threshold: float = 0.5
        """Probability above which a pixel counts toward a tissue centroid."""

        tissue_min_pixels: int = 40
        """Fewest confident pixels for a slice to propose a centroid."""

        patella_slice_min_pixels: int = 60
        """Fewest confident patellar pixels for a slice to count."""

        patella_mass_min: int = 120
        """Minimum summed confident patellar pixels across selected slices."""

        patella_margin_mm: float = 10.0
        """How far behind the joint, in patient depth, the patella may sit."""

        field_min: float = 0.15
        """Smallest plausible joint position as a fraction of either image axis."""

        field_max: float = 0.85
        """Largest plausible joint position as a fraction of either image axis."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, series: Sequence[AnatomicalSeries]) -> AnchorRecord:
        """Resolve candidates and project the anatomical consensus into every series.

        Args:
          series: Complete acquisitions in their original recipe order.

        Returns:
          record: Patient-space anchors, fallback reasons, and series projections.

        """
        record = AnchorRecord(prior=_slab_prior(series))
        candidates: list[_Candidate] = []
        sagittal = sorted(
            (
                (index, item)
                for index, item in enumerate(series)
                if item.plane == "Sagittal"
            ),
            key=lambda pair: (-(pair[1].fluid or 0), pair[0]),
        )
        for _, item in sagittal:
            if item.affines is None or not item.aligned:
                record.reasons.append(
                    "no_geo" if item.affines is None else "not_aligned",
                )
                continue
            try:
                candidate, reason = self._candidate(item, affines=item.affines)
            except (
                ArithmeticError,
                LookupError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as error:
                candidate, reason = None, "error:" + repr(error)[:80]
            if candidate is None:
                record.reasons.append(reason)
            elif (
                record.prior is not None
                and np.linalg.norm(candidate.joint - record.prior)
                > self.config.prior_mm
            ):
                record.reasons.append(
                    f"far_from_prior({np.linalg.norm(candidate.joint - record.prior):.0f}mm)",
                )
            else:
                candidates.append(candidate)
        if candidates:
            candidates = [
                item
                for _, item in sorted(
                    enumerate(candidates),
                    key=lambda pair: (-pair[1].mass, pair[0]),
                )
            ]
            best = candidates[0]
            close = [
                item
                for item in candidates
                if np.linalg.norm(item.joint - best.joint) <= self.config.agreement_mm
            ]
            if len(close) >= 2:
                record.joint = _weighted(
                    [item.joint for item in close],
                    weights=[item.mass for item in close],
                )
                record.method = f"skm_tea_joint_avg{len(close)}"
            else:
                record.joint, record.method = best.joint, "skm_tea_joint"
                if len(candidates) >= 2:
                    record.reasons.append(
                        f"disagree({np.linalg.norm(candidates[1].joint - best.joint):.0f}mm)",
                    )
            patella = [item for item in close if item.patella is not None] or [
                item for item in candidates if item.patella is not None
            ]
            if patella:
                record.patella = max(
                    enumerate(patella),
                    key=lambda pair: (pair[1].patella_mass, -pair[0]),
                )[1].patella
        elif record.prior is not None:
            record.joint, record.method = record.prior, "slab_prior"
        else:
            record.reasons.append("no_geometry")
        if record.joint is not None:
            for item in series:
                if item.affines is None:
                    continue
                _, column, row, distance = _project(item.affines, point=record.joint)
                height, width = item.shape
                patella_point = None
                if record.patella is not None:
                    _, pc, pr, _ = _project(item.affines, point=record.patella)
                    patella_point = (pr / max(1, height - 1), pc / max(1, width - 1))
                record.series[item.series] = ProjectedAnchor(
                    joint=(row / max(1, height - 1), column / max(1, width - 1)),
                    in_field=0 <= row < height and 0 <= column < width,
                    distance_mm=round(distance, 1),
                    patella=patella_point,
                )
        return record

    def _candidate(
        self,
        series: AnatomicalSeries,
        affines: NDArray[np.float64],
    ) -> tuple[_Candidate | None, str]:
        if series.reason != "ok":
            return None, series.reason
        points: list[NDArray[np.float64]] = []
        weights: list[int] = []
        patella_points: list[NDArray[np.float64]] = []
        patella_weights: list[int] = []
        for index, probability in series.probabilities:
            affine = cast("NDArray[np.float64]", affines[index])
            center, count = self._centroid(np.max(probability[1:4], axis=0))
            if center is not None:
                points.append(
                    affine @ np.array([center[1], center[0], 1.0]),
                )
                weights.append(count)
            center, count = self._centroid(cast("NDArray[np.float32]", probability[0]))
            if center is not None and count >= self.config.patella_slice_min_pixels:
                patella_points.append(
                    affine @ np.array([center[1], center[0], 1.0]),
                )
                patella_weights.append(count)
        mass, patella_mass = sum(weights), sum(patella_weights)
        if mass < self.config.mass_min:
            return None, f"mass<{self.config.mass_min}({mass})"
        joint = _weighted(points, weights=weights)
        patella = (
            _weighted(patella_points, weights=patella_weights)
            if patella_mass >= self.config.patella_mass_min
            else None
        )
        if patella is not None:
            patella = (
                patella
                if patella.item(1) < joint.item(1)
                and patella.item(2) > joint.item(2) - self.config.patella_margin_mm
                else None
            )
        _, column, row, _ = _project(affines, point=joint)
        height, width = series.shape
        row_fraction, column_fraction = (
            row / max(1, height - 1),
            column / max(1, width - 1),
        )
        if (
            self.config.field_min <= row_fraction <= self.config.field_max
            and self.config.field_min <= column_fraction <= self.config.field_max
        ):
            return _Candidate(
                joint=joint,
                patella=patella,
                mass=mass,
                patella_mass=patella_mass,
            ), "ok"
        return None, f"implausible(rf={row_fraction:.2f},cf={column_fraction:.2f})"

    def _centroid(
        self,
        probability: NDArray[np.float32],
    ) -> tuple[tuple[float, float] | None, int]:
        mask = probability > self.config.tissue_threshold
        count = int(np.count_nonzero(mask))
        if count < self.config.tissue_min_pixels:
            return None, 0
        rows, columns = cast(
            "tuple[NDArray[np.int64], NDArray[np.int64]]",
            np.nonzero(mask),
        )
        weights = probability[mask]
        return (
            float((rows * weights).sum() / weights.sum()),
            float((columns * weights).sum() / weights.sum()),
        ), count


@dataclass(kw_only=True, slots=True)
class _Candidate:
    """Retain one patient-space joint proposal and its confident tissue masses."""

    joint: NDArray[np.float64]
    patella: NDArray[np.float64] | None
    mass: int
    patella_mass: int


def _weighted(
    points: Sequence[NDArray[np.float64]],
    weights: Sequence[int],
) -> NDArray[np.float64]:
    return np.average(
        np.array(points),
        axis=0,
        weights=np.array(weights, dtype=np.float64),
    )


def _project(
    affines: NDArray[np.float64],
    point: NDArray[np.float64],
) -> tuple[int, float, float, float]:
    origins = affines[:, :, 2]
    normal = np.cross(affines[0, :, 0], affines[0, :, 1])
    normal /= np.linalg.norm(normal)
    distances = (origins - point[None]) @ normal
    index = int(np.argmin(np.abs(distances)))
    coordinates: NDArray[np.float64] = np.linalg.lstsq(
        affines[index, :, :2],
        point - cast("NDArray[np.float64]", origins[index]),
        rcond=None,
    )[0]
    return (
        index,
        coordinates.item(0),
        coordinates.item(1),
        float(abs(distances.item(index))),
    )


def _slab_prior(series: Sequence[AnatomicalSeries]) -> NDArray[np.float64] | None:
    centers: dict[str, list[NDArray[np.float64]]] = {}
    for item in series:
        if item.affines is None or not item.aligned:
            continue
        height, width = item.shape
        point = cast(
            "NDArray[np.float64]",
            item.affines[len(item.affines) // 2],
        ) @ cast(
            "NDArray[np.float64]",
            np.array([(width - 1) / 2, (height - 1) / 2, 1.0]),
        )
        centers.setdefault(item.plane, []).append(point)
    if not centers:
        return None
    prior = np.zeros(3)
    for axis, planes in enumerate(
        (("Sagittal",), ("Coronal",), ("Sagittal", "Coronal")),
    ):
        selected = [point for plane in planes for point in centers.get(plane, [])] or [
            point for points in centers.values() for point in points
        ]
        prior[axis] = np.median([point.item(axis) for point in selected])
    return prior


PLANES: Final = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
"""Axis of the slice normal for each anatomical plane."""


@dataclass(kw_only=True, slots=True)
class FrameGeometry:
    """Physical metadata decoded from one original DICOM frame."""

    orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    origin: tuple[float, ...] = (0.0, 0.0, 0.0)
    spacing: tuple[float, float] = (1.0, 1.0)
    plane: str = "Axial"
    side: str | None = None
    slope: float = 1.0
    intercept: float = 0.0
    monochrome1: bool = False


@dataclass(frozen=True, kw_only=True, slots=True)
class CanonicalFrame:
    """Canonical source pixels and their column/row/origin mapping into LPS mm."""

    image: NDArray[np.float32]
    spacing: tuple[float, float]
    affine: NDArray[np.float64]
    basis: NDArray[np.float64]


@dataclass(frozen=True, kw_only=True, slots=True)
class EncodedSeries:
    """Every native pixel encoded with one full-series linear uint16 scale."""

    volume: NDArray[np.uint16]
    minimum: float
    maximum: float
    error_bound: float
    window: tuple[float, float]


def canonical_frame(
    pixels: NDArray[np.float32],
    geometry: FrameGeometry,
) -> CanonicalFrame:
    """Reorient a decoded frame without interpolation and preserve its LPS map.

    Args:
        pixels: Original two-dimensional decoded pixel values.
        geometry: Original orientation, spacing, modality transform, and study side.

    Returns:
        frame: All canonical pixels and their exact physical affine.

    """
    if geometry.plane not in PLANES:
        raise ValueError("Unsupported anatomical plane.")
    if len(geometry.orientation) != 6 or len(geometry.origin) != 3:
        raise ValueError("DICOM orientation and origin vectors are incomplete.")
    if (
        min(geometry.spacing) <= 0
        or not all(
            math.isfinite(value)
            for value in (*geometry.spacing, geometry.slope, geometry.intercept)
        )
        or geometry.slope == 0
    ):
        raise ValueError("DICOM spacing and modality transform are invalid.")
    source_first = np.asarray(geometry.orientation[:3])
    source_second = np.asarray(geometry.orientation[3:])
    first, second = unit(unit(source_first)), unit(unit(source_second))
    if abs(float(cast(np.float64, np.dot(first, second)))) > 1e-3:
        raise ValueError("DICOM orientation axes are not orthogonal.")
    normal = unit(np.cross(unit(first), unit(second)))
    axis = PLANES[geometry.plane]
    if int(np.argmax(np.abs(normal))) != axis:
        raise ValueError("Declared plane disagrees with the slice orientation.")
    signs: NDArray[np.bool_] = np.signbit(normal)
    negative = signs.item(axis)
    normal = -normal if negative else normal
    basis: NDArray[np.float64] = np.eye(3)
    desired = cast("NDArray[np.float64]", basis[1 if axis == 0 else 0])
    desired = -desired if axis == 0 else desired
    target_first = unit(
        desired - normal * float(cast(np.float64, np.dot(desired, normal))),
    )
    target_second = unit(np.cross(normal, target_first))
    transforms = (
        (False, False, False),
        (True, True, False),
        (False, True, True),
        (True, False, True),
        (False, False, True),
        (False, True, False),
        (True, False, False),
        (True, True, True),
    )
    candidates: list[tuple[float, tuple[bool, bool, bool]]] = []
    for swap, flip_row, flip_column in transforms:
        if swap ^ flip_row ^ flip_column != negative:
            continue
        candidate_first, candidate_second = (second, first) if swap else (first, second)
        candidate_first = unit(-candidate_first if flip_column else candidate_first)
        candidate_second = unit(-candidate_second if flip_row else candidate_second)
        score = float(cast(np.float64, np.dot(candidate_first, target_first))) + float(
            cast(np.float64, np.dot(candidate_second, target_second)),
        )
        candidates.append((score, (swap, flip_row, flip_column)))
    _, (swap, flip_row, flip_column) = max(
        candidates,
        key=lambda candidate: candidate[0],
    )
    image = pixels.astype(np.float32) * geometry.slope + geometry.intercept
    if np.count_nonzero(np.isfinite(image)) != image.size:
        raise ValueError("DICOM modality-scaled pixels are nonfinite.")
    if geometry.monochrome1:
        image = (float(image.min()) + float(image.max())) - image
    height, width = cast(tuple[int, int], pixels.shape)
    corners: NDArray[np.int64] = np.array(
        [[0, width - 1], [(height - 1) * width, height * width - 1]],
    )
    if swap:
        image, corners = image.T, corners.T
    if flip_row:
        image, corners = image[::-1], corners[::-1]
    if flip_column:
        image, corners = image[:, ::-1], corners[:, ::-1]
    if geometry.plane != "Sagittal" and geometry.side == "R":
        image, corners = image[:, ::-1], corners[:, ::-1]
    sy, sx = geometry.spacing
    spacing = (sx, sy) if swap else (sy, sx)
    output_height, output_width = (width, height) if swap else (height, width)
    origin = np.asarray(geometry.origin)
    indices = tuple(corners.item(index) for index in range(3))
    p0, pc, pr = (
        origin
        + (index % width) * sx * source_first
        + (index // width) * sy * source_second
        for index in indices
    )
    affine = np.stack(
        [
            (pc - p0) / max(1, output_width - 1),
            (pr - p0) / max(1, output_height - 1),
            p0,
        ],
        axis=1,
    )
    return CanonicalFrame(
        image=np.ascontiguousarray(image),
        spacing=spacing,
        affine=affine,
        basis=np.stack([target_first, target_second, normal], axis=1),
    )


def quantize_series(images: Sequence[NDArray[np.float32]]) -> EncodedSeries:
    """Encode the complete observed range and retain its display percentiles.

    Args:
        images: All physically ordered canonical slices of one acquisition.

    Returns:
        encoded: Complete uint16 volume, scale, error bound, and 1/99 window.

    """
    volume = np.stack(images).astype(np.float32)
    minimum, maximum = float(volume.min()), float(volume.max())
    span = max(maximum - minimum, 1e-6)
    encoded = (
        np.rint((volume - minimum) / span * 65_535).clip(0, 65_535).astype(np.uint16)
    )
    window: NDArray[np.float64] = np.percentile(encoded[:, ::4, ::4], [1, 99])
    return EncodedSeries(
        volume=encoded,
        minimum=minimum,
        maximum=maximum,
        error_bound=span / 131_070,
        window=(window.item(0), window.item(1)),
    )


def study_side(tags: Sequence[str], sagittal_positions: Sequence[float]) -> str | None:
    """Resolve explicit study tags before the sagittal-position rendering fallback.

    Args:
        tags: Independently normalized laterality evidence from every series.
        sagittal_positions: Finite sagittal-series median IPP x coordinates.

    Returns:
        side: L/R, or unavailable when tags conflict or no evidence exists.

    """
    explicit = set(tags) & {"L", "R"}
    if len(explicit) == 1:
        return next(iter(explicit))
    if len(explicit) > 1:
        return None
    positions = [value for value in sagittal_positions if math.isfinite(value)]
    if positions:
        return "L" if np.median(positions) >= 0 else "R"
    return None


def unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a vector scaled to length one.

    Args:
        vector: A finite, nonzero orientation axis.

    Returns:
        axis: The unit vector.

    Raises:
        ValueError: The vector is zero or nonfinite.

    """
    length: np.float64 = np.linalg.norm(vector)
    if not math.isfinite(float(length)) or length < 1e-8:
        raise ValueError("DICOM orientation contains a zero or nonfinite axis.")
    return vector / length


@dataclass(kw_only=True, slots=True)
class DicomFrame:
    """Encoded frame data with its native transfer syntax and consumed metadata."""

    transfer_syntax: str
    values: dict[int, bytes]
    pixel_data: bytes
    little_endian: bool
    _pixels: NDArray[np.generic] | None = field(default=None, init=False, repr=False)

    def text(self, tag: int) -> str:
        """Decode a consumed textual header field without its DICOM padding.

        Args:
            tag: Complete group/element identifier.

        Returns:
            value: Decoded text, or an empty string for an absent field.

        """
        encoding = (
            "utf-8"
            if self.values.get(0x00080005, b"").strip() == b"ISO_IR 192"
            else "latin-1"
        )
        return self.values.get(tag, b"").decode(encoding).strip(" \0")

    def numbers(self, tag: int) -> tuple[float, ...]:
        """Decode the complete decimal-string vector in a header field."""
        value = self.text(tag)
        return tuple(float(part) for part in value.split("\\")) if value else ()

    def integer(self, tag: int, default: int = 0) -> int:
        """Decode an unsigned-short header value in the dataset byte order.

        Args:
            tag: Complete group/element identifier.
            default: Value for absent or empty fields.

        Returns:
            value: Decoded unsigned integer.

        """
        value = self.values.get(tag)
        return (
            int.from_bytes(value, "little" if self.little_endian else "big")
            if value is not None and len(value) > 0
            else default
        )

    @property
    def instance_number(self) -> int | None:
        """Original acquisition instance number when present."""
        value = self.text(0x00200013)
        return int(value) if value else None

    @property
    def shape(self) -> tuple[int, int]:
        """Original row and column counts."""
        shape = self.integer(0x00280010), self.integer(0x00280011)
        if min(shape) <= 0:
            raise ValueError("DICOM image dimensions must be positive.")
        return shape

    @property
    def patient_id(self) -> str:
        """Original patient identity used only for grouping and consistency."""
        return self.text(0x00100020)

    @property
    def slope(self) -> float:
        """Modality rescale slope, one when absent."""
        return float(self.text(0x00281053) or "1")

    @property
    def intercept(self) -> float:
        """Modality rescale intercept, zero when absent."""
        return float(self.text(0x00281052) or "0")

    @property
    def monochrome1(self) -> bool:
        """Whether the photometric interpretation inverts intensity."""
        return self.text(0x00280004).upper() == "MONOCHROME1"

    @property
    def laterality(self) -> str | None:
        """Normalized side after independently checking both DICOM side tags."""
        sides: set[str] = set()
        for tag in (0x00200060, 0x00200062):
            value = self.text(tag).upper()
            if value in {"L", "LEFT"}:
                sides.add("L")
            elif value in {"R", "RIGHT"}:
                sides.add("R")
        if len(sides) > 1:
            raise ValueError("DICOM laterality tags conflict.")
        return next(iter(sides)) if sides else None

    @property
    def pixel_array(self) -> NDArray[np.generic]:
        """Every decoded native sample with unused-bit/sign correction."""
        if self._pixels is not None:
            return self._pixels
        if (
            self.integer(0x00280002, default=1) != 1
            or int(self.text(0x00280008) or "1") != 1
        ):
            raise ValueError("The MRI recipe requires one monochrome frame per file.")
        bits = self.integer(0x00280100)
        stored = self.integer(0x00280101, default=bits)
        if bits not in {8, 16, 32, 64} or stored <= 0 or stored > bits:
            raise ValueError("DICOM pixel bit allocation is unsupported.")
        signed = self.integer(0x00280103) == 1
        dtype = np.dtype(f"{'i' if signed else 'u'}{bits // 8}")
        rows, columns = self.shape
        if self.transfer_syntax == "1.2.840.10008.1.2.5":
            raw = _rle(self.pixel_data, count=rows * columns, width=bits // 8)
            values = np.frombuffer(raw, dtype=dtype.newbyteorder("<")).astype(dtype)
        elif self.transfer_syntax.startswith("1.2.840.10008.1.2.4."):
            with Image.open(BytesIO(self.pixel_data)) as image:
                values = np.asarray(image).astype(dtype).reshape(-1)
        else:
            encoded_dtype = dtype.newbyteorder("<" if self.little_endian else ">")
            values = np.frombuffer(
                self.pixel_data,
                dtype=encoded_dtype,
                count=rows * columns,
            ).astype(dtype)
        if values.size != rows * columns:
            raise ValueError("DICOM pixel payload does not match its dimensions.")
        if stored < bits:
            values <<= bits - stored
            values >>= bits - stored
        self._pixels = values.reshape(rows, columns)
        return self._pixels

    def geometry(self, plane: str, side: str | None) -> FrameGeometry:
        """Return complete source geometry and the R7 modality transform.

        Args:
            plane: Declared anatomical acquisition plane.
            side: Study laterality used by canonical orientation.

        Returns:
            geometry: Physical axes, position, spacing, and modality transform.

        """
        orientation, origin, spacing = (
            self.numbers(tag) for tag in (0x00200037, 0x00200032, 0x00280030)
        )
        if len(orientation) != 6 or len(origin) != 3 or len(spacing) != 2:
            raise ValueError("DICOM physical frame metadata is incomplete.")
        if not all(math.isfinite(value) for value in (*orientation, *origin, *spacing)):
            raise ValueError("DICOM physical frame metadata is nonfinite.")
        if self.text(0x00280004).upper() not in {"MONOCHROME1", "MONOCHROME2"}:
            raise ValueError("The MRI recipe requires monochrome photometry.")
        slope, intercept = self.slope, self.intercept
        if not math.isfinite(slope) or not math.isfinite(intercept) or slope == 0:
            raise ValueError("DICOM modality transform is nonfinite or has zero slope.")
        return FrameGeometry(
            orientation=orientation,
            origin=origin,
            spacing=(spacing[0], spacing[1]),
            plane=plane,
            side=side,
            slope=slope,
            intercept=intercept,
            monochrome1=self.monochrome1,
        )

    def modality_pixels(self) -> NDArray[np.float32]:
        """Apply the float32 rescale, inverting MONOCHROME1 around its maximum.

        Returns:
            image: Complete modality-scaled two-dimensional float32 pixels.

        """
        image = self.pixel_array.astype(np.float32) * np.float32(self.slope)
        image += np.float32(self.intercept)
        return image.max() - image if self.monochrome1 else image

    def raptor_pixels(self) -> NDArray[np.float32]:
        """Apply Raptor's float64 rescale and max-only MONOCHROME1 inversion.

        Returns:
            image: Complete modality-scaled two-dimensional float32 pixels.

        """
        pixels = self.pixel_array
        if 0x00281053 in self.values and 0x00281052 in self.values:
            transformed = pixels.astype(np.float64) * float(self.text(0x00281053))
            transformed += float(self.text(0x00281052))
            image = transformed.astype(np.float32)
        else:
            image = pixels.astype(np.float32)
        if self.monochrome1:
            image = image.max() - image
        return image


def read_dicom(payload: bytes) -> DicomFrame:
    """Parse a complete encoded frame and retain its lazy pixel decoder.

    Args:
        payload: The complete original file bytes, including file meta information.

    Returns:
        frame: All consumed header fields and every original encoded pixel.

    """
    if len(payload) < 132 or payload[128:132] != b"DICM":
        raise ValueError("DICOM file meta information is missing.")
    position = 132
    transfer_syntax = "1.2.840.10008.1.2"
    while position + 8 <= len(payload):
        tag, _, start, length = _element(
            payload,
            position=position,
            little=True,
            explicit=True,
        )
        if tag >> 16 != 2:
            break
        if tag == 0x00020010:
            transfer_syntax = (
                payload[start : start + length].decode("ascii").strip(" \0")
            )
        position = start + length
    if transfer_syntax == "1.2.840.10008.1.2.1.99":
        try:
            payload = zlib.decompress(payload[position:], -15)
        except zlib.error as error:
            raise ValueError("Invalid DICOM deflate payload.") from error
        position = 0
    little = transfer_syntax != "1.2.840.10008.1.2.2"
    explicit = transfer_syntax != "1.2.840.10008.1.2"
    values: dict[int, bytes] = {}
    pixels = b""
    while position + 8 <= len(payload):
        tag, _, start, length = _element(
            payload,
            position=position,
            little=little,
            explicit=explicit,
        )
        if length == 0xFFFFFFFF:
            if tag == 0x7FE00010:
                pixels, position = _fragments(payload, position=start)
            else:
                position = _skip_undefined(
                    payload,
                    position=start,
                    little=little,
                    explicit=explicit,
                )
            continue
        end = start + length
        if end > len(payload):
            raise ValueError("DICOM element extends beyond its file.")
        if tag == 0x7FE00010:
            pixels = payload[start:end]
        else:
            values[tag] = payload[start:end]
        position = end
    return DicomFrame(
        transfer_syntax=transfer_syntax,
        values=values,
        pixel_data=pixels,
        little_endian=little,
    )


def dicom_files(series_dir: Path) -> list[Path]:
    """List a series' DICOM members in the acquisition selection order.

    Args:
        series_dir: One selected series directory, which may not exist yet.

    Returns:
        files: Lexically sorted members with a case-insensitive DICOM suffix.

    """
    return sorted(path for path in series_dir.glob("*") if is_dicom_member(path))


def is_dicom_member(path: Path) -> bool:
    """Accept the case-insensitive DICOM suffix in archive and local inventories."""
    return path.suffix.lower() == ".dcm"


def _rle(payload: bytes, count: int, width: int) -> bytes:
    if len(payload) < 64 or int.from_bytes(payload[:4], "little") != width:
        raise ValueError(
            "DICOM RLE segment count does not match its monochrome samples.",
        )
    output: NDArray[np.uint8] = np.empty((count, width), dtype=np.uint8)
    offsets = [
        int.from_bytes(payload[4 + 4 * index : 8 + 4 * index], "little")
        for index in range(width)
    ]
    for index, offset in enumerate(offsets):
        start = offset
        end = offsets[index + 1] if index + 1 < width else len(payload)
        segment = bytearray()
        while start < end and len(segment) < count:
            control = payload[start]
            start += 1
            if control < 128:
                length = control + 1
                segment.extend(payload[start : start + length])
                start += length
            elif control > 128:
                if start >= end:
                    raise ValueError("DICOM RLE repeated byte is missing.")
                segment.extend(bytes([payload[start]]) * (257 - control))
                start += 1
        if len(segment) != count:
            raise ValueError("DICOM RLE segment does not match its image dimensions.")
        output[:, width - index - 1] = np.frombuffer(segment, dtype=np.uint8)
    return output.tobytes()


EXTENT_SCALE: Final = 130.0
"""Millimeters dividing a view's field extent in the metadata extent column."""


class MetadataColumn(IntEnum):
    """Columns of the per-slice metadata tensor, in storage order."""

    BLOCK = 0
    SLOT = 1
    PLANE = 2
    VIEW = 3
    RELATIVE = 4
    LOG_SPACING = 5
    EXTENT = 6
    FLUID = 7
    FAT = 8
    SLICE = 9
    DEPTH = 10
    ACQUISITION = 11


@dataclass(kw_only=True, slots=True)
class NativeSeries:
    """One complete canonical acquisition and its consumed rendering metadata."""

    series: str
    pixels: NDArray[np.uint16]
    positions: NDArray[np.float64]
    spacing: NDArray[np.float64]
    plane: str
    slot: int
    side: str | None
    fluid: int | None
    fat: int | None
    window: tuple[float, float]
    joint: tuple[float, float] | None = None
    patella: tuple[float, float] | None = None


@dataclass(kw_only=True, slots=True)
class SliceBlock:
    """A retained physical acquisition and every rendered view's first row."""

    series: str
    plane: str
    n_slices: int
    views: list[tuple[int, int]]


@dataclass(kw_only=True, slots=True)
class RenderedStudy:
    """Every rendered slice, its twelve metadata columns, and duplicate count."""

    pixels: Tensor
    metadata: Tensor
    blocks: list[SliceBlock]
    removed: int


class R7Render:
    """Render the complete v4 view set after physical ordering and slab averaging."""

    class Config(Fig["R7Render"]):
        image_size: int = 384
        """Square letterbox side length in pixels."""

        local_mm: float = 100.0
        """Physical extent of every joint, patellar, or posterior local crop."""

        thin_mm: float = 1.75
        """Acquisitions thinner than this threshold use averaged physical slabs."""

        slab_mm: float = 3.5
        """Target slab thickness before selecting three-slice model windows."""

        view_offsets_mm: dict[str, list[tuple[int, float, float]]] = field(
            default_factory=lambda: {
                "Sagittal": [(1, 0.0, 0.0), (4, -25.0, 40.0), (5, 5.0, -35.0)],
                "Coronal": [(2, 0.0, -30.0), (3, 0.0, 30.0)],
                "Axial": [(1, 0.0, 0.0), (4, -40.0, 0.0), (5, 35.0, 0.0)],
            },
        )
        """Per plane, each local view as (view id, row mm, column mm) from the joint."""

        display_floor: float = 0.02
        """Display curve value at the lower window edge when it is above zero."""

        display_knee: float = 0.9
        """Display curve value at the upper window edge when it is below full range."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, series: Sequence[NativeSeries]) -> RenderedStudy:
        """Render all retained acquisitions with the released intensity curve."""
        retained: list[tuple[NativeSeries, tuple[tuple[int, int, int, int], ...]]] = []
        pixels: list[NDArray[np.uint8]] = []
        metadata: list[list[float]] = []
        blocks: list[SliceBlock] = []
        for real_index, acquisition in enumerate(series):
            volume, positions, representatives, spacing = self._slabs(acquisition)
            view_inputs = self._views(
                acquisition,
                representative=representatives.item(0),
            )
            crops = tuple(
                _crop_geometry(
                    cast(tuple[int, int], acquisition.pixels.shape[-2:]),
                    spacing=(
                        acquisition.spacing.item(representatives.item(index), 0),
                        acquisition.spacing.item(representatives.item(index), 1),
                    ),
                    crop_mm=crop,
                    center=center,
                )
                for _, crop, center in view_inputs
                for index in range(len(representatives))
            )
            if any(
                _same_input(acquisition, second=previous) and crops == previous_crops
                for previous, previous_crops in retained
            ):
                continue
            retained.append((acquisition, crops))
            views: list[tuple[int, int]] = []
            for view, crop, center in view_inputs:
                views.append((view, len(pixels)))
                for index in range(len(positions)):
                    native = representatives.item(index)
                    output, extent = _render_slice(
                        cast("NDArray[np.uint16]", volume[index]),
                        spacing=(
                            acquisition.spacing.item(native, 0),
                            acquisition.spacing.item(native, 1),
                        ),
                        size=self.config.image_size,
                        crop_mm=crop,
                        center=center,
                        window=acquisition.window,
                        floor=self.config.display_floor,
                        knee=self.config.display_knee,
                    )
                    pixels.append(output)
                    relative = (
                        2
                        * (positions.item(index) - positions.item(0))
                        / max(positions.item(-1) - positions.item(0), 1e-6)
                        - 1
                    )
                    metadata.append(
                        [
                            len(blocks),
                            acquisition.slot,
                            PLANES[acquisition.plane],
                            view,
                            float(relative),
                            float(cast(np.float64, np.log1p(abs(spacing)))),
                            extent / EXTENT_SCALE,
                            -1 if acquisition.fluid is None else acquisition.fluid,
                            -1 if acquisition.fat is None else acquisition.fat,
                            index,
                            len(positions),
                            real_index,
                        ],
                    )
            blocks.append(
                SliceBlock(
                    series=acquisition.series,
                    plane=acquisition.plane,
                    n_slices=len(positions),
                    views=views,
                ),
            )
        size = self.config.image_size
        if not pixels:
            return RenderedStudy(
                pixels=torch.zeros((0, size, size), dtype=torch.uint8),
                metadata=torch.zeros((0, 12), dtype=torch.float32),
                blocks=blocks,
                removed=0,
            )
        return RenderedStudy(
            pixels=torch.from_numpy(np.stack(pixels)),
            metadata=torch.tensor(metadata, dtype=torch.float32),
            blocks=blocks,
            removed=len(series) - len(retained),
        )

    def _slabs(
        self,
        acquisition: NativeSeries,
    ) -> tuple[NDArray[np.uint16], NDArray[np.float64], NDArray[np.int64], float]:
        positions = acquisition.positions
        if (
            acquisition.pixels.ndim != 3
            or len(positions) != len(acquisition.pixels)
            or not len(positions)
            or np.count_nonzero(np.isfinite(positions)) != len(positions)
        ):
            raise ValueError("Native volume and slice positions are incomplete.")
        if acquisition.plane == "Sagittal" and acquisition.side == "R":
            positions = -positions
        order = np.argsort(positions, kind="stable")
        positions = positions[order]
        volume = acquisition.pixels[order]
        spacing = float(np.median(np.diff(positions))) if len(positions) > 1 else 0.0
        if 0 < spacing < self.config.thin_mm:
            group = max(1, round(self.config.slab_mm / spacing))
            count = len(positions) // group + int(
                len(positions) % group >= max(1, group // 2),
            )
            if count >= 2:
                ends = [
                    (index * group, min(len(volume), (index + 1) * group))
                    for index in range(count)
                ]
                volume = np.stack(
                    [
                        np.rint(volume[start:stop].astype(np.float32).mean(0)).astype(
                            np.uint16,
                        )
                        for start, stop in ends
                    ],
                )
                positions = np.array(
                    [positions[start:stop].mean() for start, stop in ends],
                )
                order = np.array([order[start] for start, _ in ends])
                spacing = float(np.median(np.diff(positions)))
        return volume, positions, order, spacing

    def _views(
        self,
        acquisition: NativeSeries,
        representative: int,
    ) -> list[tuple[int, float | None, tuple[float, float]]]:
        anchor = acquisition.joint if acquisition.joint is not None else (0.5, 0.5)
        height, width = cast(tuple[int, int], acquisition.pixels.shape[-2:])
        sy, sx = (
            acquisition.spacing.item(representative, 0),
            acquisition.spacing.item(representative, 1),
        )
        result: list[tuple[int, float | None, tuple[float, float]]] = [
            (0, None, (0.5, 0.5)),
        ]
        for view, row_mm, column_mm in self.config.view_offsets_mm[acquisition.plane]:
            center = (
                min(1.0, max(0.0, anchor[0] + row_mm / (sy * max(1, height - 1)))),
                min(1.0, max(0.0, anchor[1] + column_mm / (sx * max(1, width - 1)))),
            )
            if view == 4 and acquisition.patella is not None:
                center = acquisition.patella
            result.append((view, self.config.local_mm, center))
        return result


def evaluation_rows(
    study: RenderedStudy,
    *,
    stride: int,
    slab_mm: float,
) -> NDArray[np.int64]:
    """Return every file/center/three-slice row at the released physical stride.

    Args:
        study: Complete physically ordered rendered views and float32 metadata.
        stride: Center stride measured in physical neighbor steps.
        slab_mm: Physical distance between a window's neighboring slices.

    Returns:
        rows: Every file-zero, center, previous, center, next index tuple.

    """
    rows: list[tuple[int, ...]] = []
    for block in study.blocks:
        depth = block.n_slices
        for _, start in block.views:
            spacing = math.expm1(
                float(study.metadata[start, MetadataColumn.LOG_SPACING]),
            )
            step = max(1, round(slab_mm / max(spacing, 0.1)))
            centers = (
                list(range(depth))
                if stride * step <= 1
                else list(range(min(1, depth - 1), depth, stride * step))
            )
            if stride * step > 1 and centers[-1] < depth - 2:
                centers.append(depth - 1)
            rows.extend(
                (
                    0,
                    start + center,
                    *(
                        start + min(depth - 1, max(0, center + offset * step))
                        for offset in (-1, 0, 1)
                    ),
                )
                for center in centers
            )
    return np.array(rows, dtype=np.int64).reshape(-1, 5)


def _same_input(first: NativeSeries, second: NativeSeries) -> bool:
    return (
        (
            first.plane,
            first.slot,
            first.side,
            first.fluid,
            first.fat,
            first.window,
        )
        == (
            second.plane,
            second.slot,
            second.side,
            second.fluid,
            second.fat,
            second.window,
        )
        and np.array_equal(first.positions, second.positions)
        and np.array_equal(first.spacing, second.spacing)
        and np.array_equal(first.pixels, second.pixels)
    )


def _render_slice(
    pixels: NDArray[np.uint16],
    spacing: tuple[float, float],
    size: int,
    *,
    crop_mm: float | None,
    center: tuple[float, float],
    window: tuple[float, float],
    floor: float,
    knee: float,
) -> tuple[NDArray[np.uint8], float]:
    if size < 1:
        raise ValueError("Invalid native image geometry.")
    crop_height, crop_width, y, x = _crop_geometry(
        cast(tuple[int, int], pixels.shape),
        spacing=spacing,
        crop_mm=crop_mm,
        center=center,
    )
    sy, sx = spacing
    field_y, field_x = crop_height * sy, crop_width * sx
    extent = max(field_y, field_x)
    oh = max(1, round(size * field_y / extent))
    ow = max(1, round(size * field_x / extent))
    resized = cv2.resize(
        pixels[y : y + crop_height, x : x + crop_width].astype(np.float32),
        (ow, oh),
        interpolation=cv2.INTER_AREA,
    )
    low, high = window
    if (
        not all(math.isfinite(value) for value in window)
        or low < 0
        or low > high
        or high > 65_535
    ):
        raise ValueError("Invalid uint16 display window.")
    knots = [(0.0, 0.0)]
    if high - low < 1:
        knots.append((65_535.0, 1.0))
    else:
        if low > 0:
            knots.append((low, floor))
        knots.append((high, knee if high < 65_535 else 1.0))
        if high < 65_535:
            knots.append((65_535.0, 1.0))
    points = np.array(knots, dtype=np.float64)
    normalized = np.interp(resized, points[:, 0], points[:, 1]).astype(np.float32)
    output = np.zeros((size, size), dtype=np.uint8)
    py, px = (size - oh) // 2, (size - ow) // 2
    output[py : py + oh, px : px + ow] = np.rint(
        np.clip(normalized, 0, 1) * 255,
    ).astype(np.uint8)
    return output, extent


def _crop_geometry(
    shape: tuple[int, int],
    *,
    spacing: tuple[float, float],
    crop_mm: float | None,
    center: tuple[float, float],
) -> tuple[int, int, int, int]:
    height, width = shape
    sy, sx = spacing
    if (
        min(height, width) < 1
        or min(spacing) <= 0
        or not all(math.isfinite(value) for value in spacing)
    ):
        raise ValueError("Invalid native image geometry.")
    crop_height, crop_width, y, x = height, width, 0, 0
    if crop_mm is not None:
        if not math.isfinite(crop_mm) or crop_mm <= 0:
            raise ValueError("Invalid physical crop extent.")
        extent = min(crop_mm, height * sy, width * sx)
        crop_height = min(height, max(1, round(extent / sy)))
        crop_width = min(width, max(1, round(extent / sx)))
        y = min(
            height - crop_height,
            max(0, round(center[0] * (height - 1) - (crop_height - 1) / 2)),
        )
        x = min(
            width - crop_width,
            max(0, round(center[1] * (width - 1) - (crop_width - 1) / 2)),
        )
    return crop_height, crop_width, y, x


REGION_VIEWS: Final = {"eff_sag_p99_open": 4, "bak_sag_p99_open": 5}
"""Rendered view id of each regional fluid volume, keyed by its stacker feature."""


@dataclass(kw_only=True, slots=True)
class FluidResult:
    """Regional volumes and the selected sagittal acquisition's measurements."""

    effusion: float | None = None
    baker: float | None = None
    count: int = 0
    median: float | None = None
    slices: int | None = None
    spacing: float | None = None
    series: str | None = None
    status: str = "ok"

    def by_region(self) -> dict[str, float | None]:
        """Return the regional volumes under their stacker feature names."""
        return dict(zip(REGION_VIEWS, (self.effusion, self.baker), strict=True))


class FluidMeasure:
    """Measure opened bright-fluid masks in milliliters with missing-data fallback."""

    class Config(Fig["FluidMeasure"]):
        threshold: int = 229
        """Series-companded uint8 intensity corresponding to native p99."""

        opening_mm: float = 1.5
        """Elliptical morphological opening radius before measuring volume."""

        foreground_min: int = 10
        """Rendered intensity above which a pixel is tissue for the series median."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, study: RenderedStudy) -> FluidResult:
        """Choose the lowest-median fluid-sensitive sagittal series and both locals."""
        result = FluidResult()
        try:
            metadata: NDArray[np.float32] = np.asarray(
                study.metadata.numpy(),
                dtype=np.float32,
            )
            pixels: NDArray[np.uint8] = np.asarray(study.pixels.numpy(), dtype=np.uint8)
            size = cast(int, pixels.shape[-1])
            best: _FluidCandidate | None = None
            for block in study.blocks:
                if block.plane != "Sagittal":
                    continue
                views: dict[int, tuple[int, NDArray[np.float64]]] = {}
                first: NDArray[np.float32] | None = None
                for view, start in block.views:
                    if first is None:
                        first = cast("NDArray[np.float32]", metadata[start])
                    extent = (
                        np.asarray(
                            metadata[
                                start : start + block.n_slices,
                                MetadataColumn.EXTENT,
                            ],
                            dtype=np.float64,
                        )
                        * EXTENT_SCALE
                    )
                    views[view] = (start, extent / size)
                if first is None or int(first.item(MetadataColumn.FLUID)) != 1:
                    continue
                result.count += 1
                start, _ = views[0]
                full = pixels[start : start + block.n_slices]
                foreground = full[full > self.config.foreground_min]
                median = float(np.median(foreground)) if foreground.size else 255.0
                if best is None or median < best.median:
                    best = _FluidCandidate(
                        median=median,
                        series=block.series,
                        views=views,
                        slices=block.n_slices,
                        spacing=float(
                            cast(
                                np.float32,
                                np.expm1(
                                    np.float32(first.item(MetadataColumn.LOG_SPACING)),
                                ),
                            ),
                        ),
                    )
            if best is None:
                result.status = "no fluid-sensitive sagittal series"
                return result
            result.median, result.series, result.slices, result.spacing = (
                best.median,
                best.series,
                best.slices,
                best.spacing,
            )
            missing: list[str] = []
            for label, view in REGION_VIEWS.items():
                if view not in best.views:
                    missing.append(label)
                    continue
                start, spacing = best.views[view]
                mask = (
                    pixels[start : start + best.slices] >= self.config.threshold
                ).astype(np.uint8)
                opened = np.zeros_like(mask)
                for index in range(len(mask)):
                    plane = cast("NDArray[np.uint8]", mask[index])
                    if np.count_nonzero(plane):
                        radius = max(
                            1,
                            round(self.config.opening_mm / spacing.item(index)),
                        )
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (2 * radius + 1, 2 * radius + 1),
                        )
                        opened[index] = cv2.morphologyEx(
                            plane,
                            cv2.MORPH_OPEN,
                            kernel,
                        )
                volume = float(
                    (opened.reshape(len(opened), -1).sum(1) * spacing**2).sum()
                    * best.spacing
                    / 1_000.0,
                )
                if view == REGION_VIEWS["eff_sag_p99_open"]:
                    result.effusion = volume
                else:
                    result.baker = volume
            if missing:
                result.status = "no region view for " + ",".join(missing)
        except (
            ArithmeticError,
            LookupError,
            ValueError,
            TypeError,
            RuntimeError,
            MemoryError,
            cv2.error,
        ) as error:
            result.status = "error: " + repr(error)[:200]
            result.effusion = result.baker = None
        return result


@dataclass(kw_only=True, slots=True)
class _FluidCandidate:
    median: float
    series: str
    views: dict[int, tuple[int, NDArray[np.float64]]]
    slices: int
    spacing: float


@dataclass(kw_only=True, slots=True)
class RaptorSeries:
    """One original acquisition in CSV order, including unreadable file records."""

    series: str
    plane: str
    fluid: float | str | None
    frames: Sequence[DicomFrame | None]


@dataclass(kw_only=True, slots=True)
class RaptorBag:
    """All 96 original image slots, their nonzero-image mask and slot layout."""

    pixels: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    slot_sizes: tuple[int, ...]
    sagittal_slots: tuple[int, ...]


class RaptorPrepare:
    """Select complete raw acquisitions and render the released physical crops."""

    class Config(Fig["RaptorPrepare"]):
        image_size: int = 384
        """Side length of every cached image before arm-specific windows."""

        crop_mm: float = 140.0
        """Physical center-square width using only original row spacing."""

        slots: list[tuple[str, int, int]] = field(
            default_factory=lambda: [
                ("Sagittal", 1, 26),
                ("Sagittal", 0, 22),
                ("Coronal", 1, 18),
                ("Coronal", 0, 12),
                ("Axial", -1, 18),
            ],
        )
        """Ordered anatomical plane, fluid preference, and original slice count."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, series: Sequence[RaptorSeries]) -> RaptorBag:
        """Build every original raw slice slot and retain failed slices as zero."""
        count = sum(slot[2] for slot in self.config.slots)
        pixels: NDArray[np.uint8] = np.zeros(
            (count, self.config.image_size, self.config.image_size),
            dtype=np.uint8,
        )
        used: set[str] = set()
        offset = 0
        for plane, fluid, samples in self.config.slots:
            candidates = [
                item
                for item in series
                if item.plane == plane and item.series not in used
            ]
            preferred = (
                [item for item in candidates if int(item.fluid or 0) == fluid]
                if fluid in {0, 1}
                else []
            )
            selected = preferred or candidates
            if selected:
                item = selected[0]
                used.add(item.series)
                self._fill(pixels[offset : offset + samples], frames=item.frames)
            offset += samples
        return RaptorBag(
            pixels=pixels,
            mask=(pixels.reshape(count, -1).sum(1) > 0).astype(np.uint8),
            slot_sizes=tuple(slot[2] for slot in self.config.slots),
            sagittal_slots=tuple(
                index
                for index, slot in enumerate(self.config.slots)
                if slot[0] == "Sagittal"
            ),
        )

    def _fill(
        self,
        output: NDArray[np.uint8],
        frames: Sequence[DicomFrame | None],
    ) -> None:
        if not frames:
            return
        headers = [_raw_header(frame) for frame in frames]
        ordered = sorted(range(len(frames)), key=lambda index: headers[index][0])
        low, high = int(len(frames) * 0.02), int(len(frames) * 0.98) - 1
        high = max(high, low)
        picks = np.linspace(low, high, len(output)).round().astype(np.int64)
        selected = [ordered[picks.item(index)] for index in range(len(picks))]
        decoded: dict[int, NDArray[np.float32] | None] = {}
        for index in dict.fromkeys(selected):
            frame = frames[index]
            try:
                decoded[index] = None if frame is None else frame.raptor_pixels()
            except (
                ArithmeticError,
                LookupError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                decoded[index] = None
        images = [image for index in selected if (image := decoded[index]) is not None]
        if not images:
            return
        window = cast(
            "NDArray[np.float64]",
            np.percentile(
                np.concatenate([image.ravel() for image in images]),
                [2.0, 98.0],
            ),
        )
        for output_index, index in enumerate(selected):
            image = decoded[index]
            if image is None:
                continue
            spacing = headers[index][1]
            physical = (
                spacing
                if spacing > 0
                else float(np.median([value for _, value, valid in headers if valid]))
            )
            output[output_index] = raptor_slice(
                image,
                spacing=physical,
                window=window,
                image_size=self.config.image_size,
                crop_mm=self.config.crop_mm,
            )


def raptor_slice(
    image: NDArray[np.float32],
    *,
    spacing: float,
    window: NDArray[np.float64],
    image_size: int,
    crop_mm: float,
) -> NDArray[np.uint8]:
    """Render one raw slice with its complete selected-series intensity window.

    Args:
        image: Original modality-scaled pixels before anatomical reorientation.
        spacing: Original row spacing, or the valid-header median fallback.
        window: Float64 lower and upper percentiles over all selected full slices.
        image_size: Square output image width.
        crop_mm: Physical center-square extent.

    Returns:
        pixels: Complete resized crop after truncating uint8 conversion.

    """
    height, width = cast(tuple[int, int], image.shape)
    if (
        not math.isfinite(spacing)
        or not math.isfinite(crop_mm)
        or crop_mm <= 0
        or min(height, width, image_size) <= 0
    ):
        raise ValueError("Raptor crop geometry must be finite and nonempty.")
    side = min(round(crop_mm / max(spacing, 1e-3)), height, width)
    if side < 1:
        raise ValueError("Raptor crop contains no native pixels.")
    row, column = (height - side) // 2, (width - side) // 2
    low, high = cast(np.float64, window[0]), cast(np.float64, window[1])
    normalized = np.clip(
        (image[row : row + side, column : column + side] - low) / (high - low + 1e-6),
        0,
        1,
    )
    resized = cv2.resize(
        normalized,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    return (resized * 255).astype(np.uint8)


def window_centers(
    mask: NDArray[np.uint8],
    depth: int,
    count: int,
) -> NDArray[np.int64]:
    """Cover the valid slice interval with repeated, rounded three-slice centers.

    Args:
        mask: Nonzero rendered slices in the fixed study volume.
        depth: Number of slices, including unavailable series slots.
        count: Number of windows scored by the model.

    Returns:
        centers: The complete ordered center-index sequence.

    """
    valid = np.flatnonzero(np.greater(mask, 0))
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    first, last = valid.min().item(), valid.max().item()
    candidates = np.arange(first + 1, last, dtype=np.int64)
    if not len(candidates):
        candidates = np.asarray(
            [max(1, min((first + last) // 2, depth - 2))],
            dtype=np.int64,
        )
    picks = np.linspace(0, len(candidates) - 1, count).round().astype(np.int64)
    return candidates[picks]


class ReorientFn(Protocol):
    """Map gathered slice triplets to one test-time view of the study."""

    def __call__(self, triplets: Tensor, sagittal: Tensor, /) -> Tensor:
        """Return the triplets as seen from this view.

        Args:
            triplets: Window-by-channel-by-height-by-width uint8 slices.
            sagittal: Per window, whether its center lies in a sagittal slot.

        """
        ...


def unchanged(triplets: Tensor, sagittal: Tensor) -> Tensor:
    """Keep the acquired orientation."""
    del sagittal
    return triplets


def anatomical_mirror(triplets: Tensor, sagittal: Tensor) -> Tensor:
    """Reverse slice channels of sagittal windows and image columns of the rest."""
    return torch.where(sagittal, triplets.flip(1), triplets.flip(-1))


class RaptorWindows:
    """Render all selected triplets with the released crop convention."""

    class Config(Fig["RaptorWindows"]):
        windows: int = 94
        """Number of scored triplets across the complete 96-slice bag."""

        crop_size: int = 320
        """Side of the centered square cut from the cached study volume."""

        mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
        """ImageNet normalization offset per triplet channel."""

        std: tuple[float, float, float] = (0.229, 0.224, 0.225)
        """ImageNet normalization scale per triplet channel."""

    def __init__(self, config: Config) -> None:
        if min(config.windows, config.crop_size) < 1:
            raise ValueError("Raptor crop and window geometry is invalid.")
        if any(not math.isfinite(value) or value <= 0 for value in config.std):
            raise ValueError("Raptor normalization scales must be finite and positive.")
        self.config = config
        self.mean = torch.tensor(config.mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(config.std, dtype=torch.float32).view(3, 1, 1)
        values = torch.from_numpy(
            np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0,
        )
        self.intensity = values
        self.normalized = (
            (
                (
                    torch.stack([values, values, values], 0).view(3, 256, 1)
                    - self.mean.view(3, 1, 1)
                )
                / self.std.view(3, 1, 1)
            )
            .view(3, 256)
            .contiguous()
        )

    def __call__(
        self,
        bag: RaptorBag,
        *,
        output_size: int,
        device: str | torch.device = "cpu",
        reorient: ReorientFn = unchanged,
    ) -> Tensor:
        """Gather and normalize windows on the scoring device.

        Args:
            bag: Complete fixed-slot uint8 slice volume, its mask and slot layout.
            output_size: Side length consumed by the model; a smaller or larger
                crop is resized to it.
            device: Destination for gathered windows and normalization tables.
            reorient: Test-time view applied to the gathered triplets.

        Returns:
            windows: All normalized triplets in their original scoring order.

        """
        config = self.config
        volume, mask = bag.pixels, bag.mask
        depth, image_size = len(volume), cast(int, volume.shape[-1])
        if depth != sum(bag.slot_sizes):
            raise ValueError("Raptor slots must sum to the complete volume depth.")
        if volume.shape[1:] != (image_size, image_size) or mask.shape != (depth,):
            raise ValueError(
                "Raptor input pixels and slice mask must match their geometry.",
            )
        if output_size < 1 or config.crop_size > image_size:
            raise ValueError("Raptor crop and window geometry is invalid.")
        centers = window_centers(mask, depth=depth, count=config.windows)
        triplets = torch.from_numpy(
            np.ascontiguousarray(
                volume[np.stack([centers - 1, centers, centers + 1], 1)],
            ),
        ).to(device)
        boundaries = np.cumsum([0, *bag.slot_sizes])
        slots = np.searchsorted(boundaries, centers, side="right") - 1
        sagittal = (
            torch.from_numpy(np.isin(slots, bag.sagittal_slots))
            .to(device)
            .view(-1, 1, 1, 1)
        )
        triplets = reorient(triplets, sagittal)
        offset = (image_size - config.crop_size) // 2
        triplets = triplets[
            :,
            :,
            offset : offset + config.crop_size,
            offset : offset + config.crop_size,
        ]
        if config.crop_size != output_size:
            windows = self.intensity.to(device)[triplets.long()]
            windows = functional.interpolate(
                windows,
                size=(output_size, output_size),
                mode="bilinear",
            )
            return (windows - self.mean.to(device)) / self.std.to(device)
        table = self.normalized.to(device)
        return torch.stack(
            [table[channel][triplets[:, channel].long()] for channel in range(3)],
            1,
        )


class AnatomySampler:
    """Apply original anatomical normalization, orientation, resize, and padding."""

    class Config(Fig["AnatomySampler"]):
        slices: int = 7
        """Odd maximum number of physical slices centered in each acquisition."""

        span_mm: float = 4.0
        """Distance between anatomical model slices in patient space."""

        pixel_mm: float = 0.3125
        """Anatomical model's target in-plane pixel size."""

        pad_multiple: int = 16
        """Minimum model side and replication-padding multiple."""

        oblique_cos: float = 0.85
        """Minimum sagittal alignment with the AP and SI axes."""

    def __init__(self, config: Config) -> None:
        self.config = config

    @torch.no_grad()
    def __call__(
        self,
        volume: NDArray[np.uint16],
        spacing: NDArray[np.float64],
        affines: NDArray[np.float64],
        model: TensorFn,
        *,
        device: str | torch.device = "cpu",
    ) -> tuple[list[tuple[int, NDArray[np.float32]]] | None, str]:
        """Sample all selected slices and return native-sized tissue probabilities."""
        config = self.config
        if (
            abs(affines.item(0, 1, 0)) / np.linalg.norm(affines[0, :, 0])
            < config.oblique_cos
            or abs(affines.item(0, 2, 1)) / np.linalg.norm(affines[0, :, 1])
            < config.oblique_cos
        ):
            return None, "oblique"
        mean, std = float(volume.mean()), max(float(volume.std()), 1.0)
        flip_x, flip_y = affines.item(0, 1, 0) < 0, affines.item(0, 2, 1) > 0
        distance = (
            float(np.linalg.norm(affines[1, :, 2] - affines[0, :, 2]))
            if len(affines) > 1
            else 4.0
        )
        step = max(1, round(config.span_mm / max(distance, 1e-3)))
        middle, radius = len(volume) // 2, config.slices // 2
        selected = sorted(
            {
                min(len(volume) - 1, max(0, middle + offset * step))
                for offset in range(-radius, radius + 1)
            },
        )
        result: list[tuple[int, NDArray[np.float32]]] = []
        for index in selected:
            image = (
                cast("NDArray[np.uint16]", volume[index]).astype(np.float32) - mean
            ) / std
            if flip_x:
                image = image[:, ::-1]
            if flip_y:
                image = image[::-1]
            height, width = cast(tuple[int, int], image.shape)
            resized = _resize_input(
                image,
                size=(
                    max(
                        config.pad_multiple,
                        round(height * spacing.item(index, 0) / config.pixel_mm),
                    ),
                    max(
                        config.pad_multiple,
                        round(width * spacing.item(index, 1) / config.pixel_mm),
                    ),
                ),
            )
            inputs = torch.from_numpy(resized)[None, None].to(device)
            height, width = inputs.shape[-2:]
            inputs = functional.pad(
                inputs,
                (0, (-width) % config.pad_multiple, 0, (-height) % config.pad_multiple),
                mode="replicate",
            )
            probabilities = model(inputs).sigmoid()[..., :height, :width]
            restored = (
                functional.interpolate(
                    probabilities,
                    size=volume.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
                .float()
                .cpu()
                .numpy()
            )
            if flip_x:
                restored = restored[:, :, ::-1]
            if flip_y:
                restored = restored[:, ::-1]
            result.append((index, np.ascontiguousarray(restored)))
        return result, "ok"


def _resize_input(
    image: NDArray[np.float32],
    *,
    size: tuple[int, int],
) -> NDArray[np.float32]:
    """Resize bilinearly at half-pixel centers in float64, rounding once to float32."""
    height, width = cast(tuple[int, int], image.shape)
    x = (np.arange(size[1]) + 0.5) * (width / size[1]) - 0.5
    y = (np.arange(size[0]) + 0.5) * (height / size[0]) - 0.5
    left, top = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - left, (y - top)[:, None]
    pixels = image.astype(np.float64)
    left_pixels = pixels[:, np.maximum(left, 0)]
    right_pixels = pixels[:, np.minimum(left + 1, width - 1)]
    rows = left_pixels + fx * (right_pixels - left_pixels)
    top_rows = rows[np.maximum(top, 0)]
    bottom_rows = rows[np.minimum(top + 1, height - 1)]
    return (top_rows + fy * (bottom_rows - top_rows)).astype(np.float32)


@dataclass(kw_only=True, slots=True)
class NamedFrame:
    """A decoded DICOM frame with its original acquisition filename."""

    name: str
    frame: DicomFrame


@dataclass(kw_only=True, slots=True)
class SeriesOrder:
    """Every frame sorted by its positive cardinal patient-space projection."""

    frames: list[NamedFrame]
    positions: NDArray[np.float64]
    plane: str
    sides: tuple[str, ...]
    median_x: float


@dataclass(kw_only=True, slots=True)
class AlignmentQuality:
    """All alignment measurements consumed by anatomical anchor selection."""

    aligned: bool
    drift_mm: float
    axis_difference: float
    normal: NDArray[np.float64]
    depth: NDArray[np.float64]


@dataclass(kw_only=True, slots=True)
class PreparedSeries:
    """Complete canonical intensity volume, physical geometry, and slice failures."""

    native: NativeSeries
    files: list[str]
    affines: NDArray[np.float64]
    quality: AlignmentQuality
    intensity: tuple[float, ...]
    errors: list[tuple[str, str]]


def order_frames(frames: Sequence[NamedFrame], plane: str) -> SeriesOrder:
    """Validate every original header before physically ordering a series.

    Args:
        frames: Every frame of one acquisition, in its original enumeration order.
        plane: Declared anatomical plane.

    Returns:
        order: Complete stable physical ordering and explicit side evidence.

    """
    if not frames:
        raise ValueError("The acquisition contains no DICOM frames.")
    axis = PLANES[plane]
    geometry = [item.frame.geometry(plane, side=None) for item in frames]
    if any(item.frame.instance_number is None for item in frames):
        raise ValueError("Every frame requires an integer acquisition instance.")
    if len({item.frame.shape for item in frames}) != 1:
        raise ValueError("Pixel matrix changes within the acquisition.")
    spacing: NDArray[np.float64] = np.array([item.spacing for item in geometry])
    if np.any(spacing <= 0) or float(np.max(np.ptp(spacing, axis=0))) > 1e-5:
        raise ValueError("Pixel spacing is invalid or changes within the acquisition.")
    if len({item.monochrome1 for item in geometry}) != 1:
        raise ValueError("Photometric interpretation changes within the acquisition.")
    first, second = _directions(geometry[0])
    if abs(float(cast(np.float64, np.dot(first, second)))) > 1e-3:
        raise ValueError("DICOM orientation axes are not orthogonal.")
    normal = np.cross(unit(first), unit(second))
    normal = unit(normal)
    if int(np.argmax(np.abs(normal))) != axis:
        raise ValueError("Declared anatomical plane disagrees with its slice normal.")
    if normal.item(axis) < 0:
        normal = -normal
    positions: list[float] = []
    for item in geometry:
        row, column = _directions(item)
        candidate = unit(np.cross(row, column))
        if abs(float(cast(np.float64, np.dot(candidate, normal)))) < 0.995:
            raise ValueError("Slice normals vary within the acquisition.")
        if (
            float(cast(np.float64, np.dot(row, first))) < 0.995
            or float(cast(np.float64, np.dot(column, second))) < 0.995
        ):
            raise ValueError("In-plane orientation axes vary within the acquisition.")
        positions.append(
            float(cast(np.float64, np.dot(np.asarray(item.origin), normal))),
        )
    projections: NDArray[np.float64] = np.array(positions)
    indices = np.argsort(projections, kind="stable")
    ordered = projections[indices]
    if np.any(np.diff(ordered) <= 1e-3):
        raise ValueError("Physical slice projections are duplicate or non-increasing.")
    return SeriesOrder(
        frames=[frames[indices.item(index)] for index in range(len(indices))],
        positions=ordered,
        plane=plane,
        sides=tuple(
            sorted(
                {
                    side
                    for item in frames
                    if (side := item.frame.laterality) is not None
                },
            ),
        ),
        median_x=float(np.median([item.origin[0] for item in geometry])),
    )


def prepare_series(
    order: SeriesOrder,
    *,
    series: str,
    side: str | None,
    fat: int,
    fluid: int,
) -> PreparedSeries:
    """Canonicalize all usable slices and quantize their complete observed range.

    Args:
        order: Validated complete-header physical order.
        series: Original acquisition identity.
        side: Study rendering side resolved across all acquisitions.
        fat: Fat-suppression category, including unknown -1.
        fluid: Fluid-sensitive category, including unknown -1.

    Returns:
        prepared: Every retained pixel, position, spacing, affine, and slice error.

    """
    images: list[NDArray[np.float32]] = []
    positions: list[float] = []
    spacing: list[tuple[float, float]] = []
    affines: list[NDArray[np.float64]] = []
    names: list[str] = []
    errors: list[tuple[str, str]] = []
    for index, item in enumerate(order.frames):
        try:
            frame = canonical_frame(
                item.frame.pixel_array.astype(np.float32),
                geometry=item.frame.geometry(order.plane, side=side),
            )
        except (
            ArithmeticError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            errors.append((item.name, str(error)[:250]))
            continue
        if images and (
            frame.image.shape != images[0].shape
            or not np.allclose(frame.spacing, spacing[0], rtol=1e-4)
        ):
            errors.append(
                (item.name, "Native matrix or spacing changes within the acquisition."),
            )
            continue
        images.append(frame.image)
        positions.append(order.positions.item(index))
        spacing.append(frame.spacing)
        affines.append(frame.affine)
        names.append(item.name)
    if not images:
        raise ValueError("The acquisition contains no usable native slices.")
    encoded = quantize_series(images)
    matrices = np.stack(affines)
    axis = PLANES[order.plane]
    slot = axis * 2 + (1 if fat == 1 else 2) if fat in {0, 1} else 7 + axis
    return PreparedSeries(
        native=NativeSeries(
            series=series,
            pixels=encoded.volume,
            positions=np.array(positions),
            spacing=np.array(spacing),
            plane=order.plane,
            slot=slot,
            side=side,
            fluid=fluid,
            fat=fat if fat in {0, 1} else -1,
            window=encoded.window,
        ),
        files=names,
        affines=matrices,
        quality=_alignment(matrices),
        intensity=(
            encoded.minimum,
            encoded.maximum,
            encoded.error_bound,
            *encoded.window,
        ),
        errors=errors,
    )


def _alignment(affines: NDArray[np.float64]) -> AlignmentQuality:
    normal = np.cross(affines[0, :, 0], affines[0, :, 1])
    normal /= np.linalg.norm(normal)
    origins = affines[:, :, 2]
    relative = origins - affines[0, :, 2]
    drift = np.linalg.norm(relative - (relative @ normal)[:, None] * normal, axis=1)
    axis_difference = float(
        cast(np.float64, np.max(np.abs(affines[:, :, :2] - affines[0, :, :2]))),
    )
    return AlignmentQuality(
        aligned=bool(axis_difference < 1e-3 and drift.max() < 1.0),
        drift_mm=float(drift.max()),
        axis_difference=axis_difference,
        normal=normal,
        depth=origins @ normal,
    )


@dataclass(kw_only=True, slots=True)
class SideNote:
    """Original side evidence, rounded diagnostics, and an applied correction."""

    side: str | None
    status: str
    x: float | None
    score: float | None = None
    flipped: bool = False
    new_side: str | None = None
    tag_contradicted_by_position: bool = False


class SideCheck:
    """Respect explicit evidence and check ambiguous or contradicted orientations."""

    class Config(Fig["SideCheck"]):
        image_size: int = 160
        """Square classifier input side."""

        fractions: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6])
        """Rounded middle-slice fractions, deduplicated before preprocessing."""

        threshold: float = 2.0
        """Flip only when the float32 mean logit is below the negative threshold."""

        max_x: float = math.inf
        """A sagittal position this far from the scanner center skips the check."""

        position_mm: float = 20.0
        """Minimum sagittal position contradicting an explicit side tag."""

        window_percentiles: tuple[float, float] = (1.0, 99.0)
        """Percentiles of the subsampled slice that set its intensity window."""

        percentile_stride: int = 4
        """Row and column step of the subsample the window percentiles read."""

        mean: float = 0.45
        """Value subtracted from classifier pixels scaled to [0, 1]."""

        std: float = 0.25
        """Divisor of the centered classifier pixels."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def inputs(self, volume: NDArray[np.uint16]) -> NDArray[np.uint8]:
        """Select and preprocess every distinct original classifier slice.

        Args:
            volume: Complete canonical native uint16 acquisition.

        Returns:
            images: All selected square uint8 classifier inputs.

        """
        size = self.config.image_size
        if len(volume) == 0:
            return np.zeros((0, size, size), dtype=np.uint8)
        indices = sorted(
            {round(fraction * (len(volume) - 1)) for fraction in self.config.fractions},
        )
        images: list[NDArray[np.uint8]] = []
        for index in indices:
            image = cast("NDArray[np.uint16]", volume[index]).astype(np.float32)
            window = cast(
                "NDArray[np.float64]",
                np.percentile(
                    image[
                        :: self.config.percentile_stride,
                        :: self.config.percentile_stride,
                    ],
                    list(self.config.window_percentiles),
                ),
            )
            low, high = cast(np.float64, window[0]), cast(np.float64, window[1])
            normalized = np.clip((image - low) / max(float(high - low), 1e-6), 0, 1)
            resized = cv2.resize(normalized, (size, size), interpolation=cv2.INTER_AREA)
            images.append(np.rint(resized * 255).astype(np.uint8))
        return np.stack(images)

    def __call__(
        self,
        series: Sequence[NativeSeries],
        statuses: set[str],
        model: TensorFn | None,
        *,
        device: torch.device | str,
        rerender: Callable[[str], set[str | None]],
    ) -> SideNote | None:
        """Check the study and rebuild every dependent representation after a flip.

        Args:
            series: All successfully prepared acquisitions in original CSV order.
            statuses: Study-wide explicit-laterality status from each geometry record.
            model: Loaded orientation classifier, or None when checks are disabled.
            device: Classifier placement after CPU float32 image normalization.
            rerender: Rebuild from original frames and return every resulting side.

        Returns:
            note: Complete evidence and correction diagnostics, or disabled None.

        """
        if model is None:
            return None
        sides = {item.side for item in series}
        current = next(iter(sides)) if len(sides) == 1 else None
        sagittal = [
            float(np.median(item.positions))
            for item in series
            if item.plane == "Sagittal" and len(item.positions)
        ]
        x = float(np.median(sagittal)) if sagittal else None
        note = SideNote(
            side=current,
            status="|".join(sorted(statuses)),
            x=None if x is None else round(x, 1),
        )
        if statuses == {"explicit_study_tags"} and current in {"L", "R"}:
            position = (
                None
                if x is None or abs(x) < self.config.position_mm
                else "L"
                if x >= 0
                else "R"
            )
            if position is None or position == current:
                return note
            note.tag_contradicted_by_position = True
        if current in {"L", "R"} and x is not None and abs(x) >= self.config.max_x:
            return note
        candidates = [
            self.inputs(item.pixels)
            for item in series
            if item.plane in {"Coronal", "Axial"}
        ]
        images = [value for value in candidates if len(value)]
        if not images:
            return note
        batch = torch.from_numpy(np.concatenate(images)).float().div_(255.0)[:, None]
        with torch.inference_mode():
            logits = (
                model(((batch - self.config.mean) / self.config.std).to(device))
                .float()
                .squeeze(1)
            )
        score = float(logits.mean())
        note.score = round(score, 3)
        if score < -self.config.threshold:
            new = "R" if current in {None, "L"} else "L"
            if rerender(new) != {new}:
                raise ValueError(
                    "Forced study side was not applied to every rebuilt acquisition.",
                )
            note.flipped, note.new_side = True, new
        return note


MODEL_COLUMNS: Final = (
    MetadataColumn.SLOT,
    MetadataColumn.VIEW,
    MetadataColumn.RELATIVE,
    MetadataColumn.LOG_SPACING,
    MetadataColumn.EXTENT,
    MetadataColumn.FLUID,
)
"""Rendered metadata columns that become the model's first six metadata columns."""

EXTENT_COLUMNS: Final = (
    MetadataColumn.BLOCK,
    MetadataColumn.PLANE,
    MetadataColumn.DEPTH,
)
"""Rendered metadata columns that become the model's series, plane and depth."""


@dataclass(kw_only=True, slots=True)
class Acquisition:
    """Complete original acquisition headers in their filesystem enumeration order."""

    series: str
    plane: str
    frames: Sequence[NamedFrame]
    fat: float | str | None
    fluid: float | str | None
    error: str | None = None


@dataclass(kw_only=True, slots=True)
class PreparedStudy:
    """Every physical representation, correction, anchor, fluid volume, and window."""

    acquisitions: list[PreparedSeries]
    images: torch.Tensor
    metadata: torch.Tensor
    rendered: RenderedStudy
    anchors: AnchorRecord
    side: SideNote | None
    fluid: FluidResult
    errors: list[tuple[str, str]]


class StudyPreparation:
    """Prepare physical acquisitions and rebuild them when laterality changes."""

    class Config(Fig["StudyPreparation"]):
        eval_stride: int = 1
        """Released R7 center stride, multiplied by the physical neighbor step."""

        side: SideCheck.Config = field(default_factory=SideCheck.Config)
        """Laterality classifier preprocessing and correction thresholds."""

        anatomy: AnatomySampler.Config = field(default_factory=AnatomySampler.Config)
        """Physical anatomical slice normalization and restoration."""

        anchors: AnchorConsensus.Config = field(default_factory=AnchorConsensus.Config)
        """Patient-space anatomical consensus and slab fallback."""

        render: R7Render.Config = field(default_factory=R7Render.Config)
        """Complete physical v4 views and duplicate removal."""

        fluid: FluidMeasure.Config = field(default_factory=FluidMeasure.Config)
        """D2 fluid-volume measurement on the rendered rows."""

    def __init__(self, config: Config) -> None:
        self.eval_stride = config.eval_stride
        self.side = config.side.make()
        self.anatomy = config.anatomy.make()
        self.anchors = config.anchors.make()
        self.render = config.render.make()
        self.fluid = config.fluid.make()

    def __call__(
        self,
        acquisitions: Sequence[Acquisition],
        *,
        anatomy: TensorFn,
        side: TensorFn | None,
        device: str = "cpu",
    ) -> PreparedStudy:
        """Produce all three-slice windows after the complete physical rebuild.

        Args:
            acquisitions: Original CSV-ordered acquisitions and complete headers.
            anatomy: Loaded four-tissue model, called on physical sagittal slices.
            side: Loaded laterality model, or None to disable orientation checking.
            device: Placement of the two auxiliary models.

        Returns:
            study: Complete intermediate arrays and every consumed model input.

        """
        physical, current = _order_study(acquisitions)
        physical.rebuild(current)
        note = self.side(
            [item.native for item in physical.prepared],
            statuses=physical.statuses(),
            model=side,
            device=device,
            rerender=physical.rebuild,
        )
        anchors = self.anchors(
            [
                _anatomical(item, sampler=self.anatomy, model=anatomy, device=device)
                for item in physical.prepared
            ],
        )
        for item in physical.prepared:
            anchor = anchors.series.get(item.native.series)
            if anchor is not None and anchor.in_field:
                item.native.joint, item.native.patella = anchor.joint, anchor.patella
        rendered = self.render([item.native for item in physical.prepared])
        fluid = self.fluid(rendered)
        rows = evaluation_rows(
            rendered,
            stride=self.eval_stride,
            slab_mm=self.render.config.slab_mm,
        )
        if not len(rows):
            raise ValueError("The study contains no usable windows.")
        pixels, metadata = rendered.pixels.numpy(), rendered.metadata.numpy()
        images = np.stack([pixels[rows[:, 2 + offset]] for offset in range(3)], 1)
        selected = metadata[rows[:, 1]]
        columns = np.stack([selected[:, index] for index in MODEL_COLUMNS], 1)
        extent = np.stack([selected[:, index] for index in EXTENT_COLUMNS], 1)
        return PreparedStudy(
            acquisitions=physical.prepared,
            images=torch.from_numpy(images),
            metadata=torch.from_numpy(np.concatenate([columns, extent], 1)),
            rendered=rendered,
            anchors=anchors,
            side=note,
            fluid=fluid,
            errors=physical.errors,
        )


@dataclass(kw_only=True, slots=True)
class StudyInputs:
    """Independent native inputs for Raptor slots and physical R7 preparation."""

    raptor: list[RaptorSeries]
    physical: list[Acquisition]


class StudyData:
    """Read every selected study in CSV order and retain acquisition enumeration."""

    class Config(Fig["StudyData"]):
        base_dir: Path | str | None = None
        """Resource root beneath which `input_dir` resolves; the owner supplies it."""

        input_dir: Path | str = "/datasets/rsna-knee/raw"
        """Logical directory of the competition CSV files and series trees."""

        split: str = "test"
        """Competition split used for both CSV and DICOM directory names."""

        @override
        def finalize(self) -> Self:
            self.input_dir = resolve_working_dir(self.base_dir, self.input_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.input_dir = Path(config.input_dir)
        self.ids = [
            row["StudyInstanceUID"] or ""
            for row in _csv_rows(self.input_dir / f"{config.split}.csv")
        ]
        with (self.input_dir / "sample_submission.csv").open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            self.sample_ids = [row[0] for row in reader]
        self.labels = header[1:]
        if (
            len(set(self.ids)) != len(self.ids)
            or len(set(self.sample_ids)) != len(self.sample_ids)
            or set(self.sample_ids) != set(self.ids)
        ):
            raise ValueError("Competition and sample row identity differs or repeats.")
        if len(self.labels) != len(FINDINGS) or set(self.labels) != set(FINDINGS):
            raise ValueError("Sample finding columns differ from the released model.")
        rows = _csv_rows(self.input_dir / f"{config.split}_series.csv")
        fat, fluid = (
            _categories(rows, column="Fat_Suppression"),
            _categories(rows, column="Fluid_Sensitive"),
        )
        self.rows: dict[str, list[_SeriesRow]] = {uid: [] for uid in self.ids}
        for index, row in enumerate(rows):
            uid = row["StudyInstanceUID"] or ""
            if uid in self.rows:
                self.rows[uid].append(
                    _SeriesRow(
                        series=row["SeriesInstanceUID"] or "",
                        plane=row["Anatomical_Plane"] or "",
                        fat=fat[index],
                        fluid=fluid[index],
                    ),
                )

    def load(self, uid: str) -> StudyInputs:
        """Read all complete files for one study without sorting their enumeration.

        Args:
            uid: Study identity declared by the competition split CSV.

        Returns:
            study: Full raw slots and physical series with explicit header failures.

        """
        physical: list[Acquisition] = []
        raptor: list[RaptorSeries] = []
        for row in self.rows[uid]:
            directory = series_tree(self.input_dir, self.config.split, uid, row.series)
            named: list[NamedFrame] = []
            frames: list[DicomFrame | None] = []
            errors: list[str] = []
            for path in directory.glob("*"):
                if not is_dicom_member(path):
                    continue
                try:
                    frame = read_dicom(path.read_bytes())
                except (FileNotFoundError, EOFError):
                    raise
                except (OSError, ValueError, LookupError, ArithmeticError) as error:
                    frames.append(None)
                    errors.append(str(error)[:250])
                    continue
                frames.append(frame)
                named.append(NamedFrame(name=path.name, frame=frame))
            physical.append(
                Acquisition(
                    series=row.series,
                    plane=row.plane,
                    fat=row.fat,
                    fluid=row.fluid,
                    frames=named,
                    error="; ".join(errors) if errors else None,
                ),
            )
            raptor.append(
                RaptorSeries(
                    series=row.series,
                    plane=row.plane,
                    fluid=row.fluid,
                    frames=frames,
                ),
            )
        return StudyInputs(raptor=raptor, physical=physical)


def series_tree(input_dir: Path, split: str, *parts: str) -> Path:
    """Locate a path under the split's series tree, else under its image tree.

    Args:
        input_dir: Directory holding the split's extracted DICOM trees.
        split: Competition split naming both trees.
        *parts: Study and series identifiers beneath the tree.

    Returns:
        path: The `<split>_series` location when it exists, else `<split>_images`.

    """
    primary = input_dir / f"{split}_series" / Path(*parts)
    return primary if primary.is_dir() else input_dir / f"{split}_images" / Path(*parts)


@dataclass(frozen=True, kw_only=True, slots=True)
class _SeriesRow:
    series: str
    plane: str
    fat: float | str | None
    fluid: float | str | None


def _csv_rows(path: Path) -> list[dict[str, str | None]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return from_plain(list(csv.DictReader(handle)), list[dict[str, str | None]])


def _categories(
    rows: list[dict[str, str | None]],
    column: str,
) -> list[float | str | None]:
    values = [row.get(column) for row in rows]
    missing = PANDAS_NA_VALUES
    present = [value for value in values if value is not None and value not in missing]
    if present and all(value.casefold() in {"true", "false"} for value in present):
        return [
            float("nan")
            if value is None or value in missing
            else float(value.casefold() == "true")
            for value in values
        ]
    try:
        return [
            float("nan") if value is None or value in missing else float(value)
            for value in values
        ]
    except ValueError:
        return [
            float("nan") if value is None or value in missing else value
            for value in values
        ]


def _element(
    payload: bytes,
    position: int,
    *,
    little: bool,
    explicit: bool,
) -> tuple[int, bytes, int, int]:
    order = "little" if little else "big"
    group = int.from_bytes(payload[position : position + 2], order)
    element = int.from_bytes(payload[position + 2 : position + 4], order)
    tag = group << 16 | element
    vr = payload[position + 4 : position + 6] if explicit and group != 0xFFFE else b""
    if vr in {
        b"OB",
        b"OF",
        b"OL",
        b"OV",
        b"OW",
        b"SQ",
        b"SV",
        b"UC",
        b"UN",
        b"UR",
        b"UT",
        b"UV",
    }:
        start = position + 12
        length = int.from_bytes(payload[position + 8 : start], order)
    elif vr:
        start = position + 8
        length = int.from_bytes(payload[position + 6 : start], order)
    else:
        start = position + 8
        length = int.from_bytes(payload[position + 4 : start], order)
    if start > len(payload):
        raise ValueError("DICOM element header is truncated.")
    return tag, vr, start, length


def _skip_undefined(
    payload: bytes,
    position: int,
    *,
    little: bool,
    explicit: bool,
) -> int:
    while position + 8 <= len(payload):
        tag, _, start, length = _element(
            payload,
            position=position,
            little=little,
            explicit=explicit,
        )
        if tag in {0xFFFEE00D, 0xFFFEE0DD}:
            if length != 0:
                raise ValueError("DICOM sequence delimiter has a nonzero length.")
            return start
        position = (
            _skip_undefined(payload, position=start, little=little, explicit=explicit)
            if length == 0xFFFFFFFF
            else start + length
        )
    raise ValueError("DICOM sequence delimiter is missing.")


def _fragments(payload: bytes, position: int) -> tuple[bytes, int]:
    fragments: list[bytes] = []
    first = True
    while position + 8 <= len(payload):
        tag, _, start, length = _element(
            payload,
            position=position,
            little=True,
            explicit=False,
        )
        if tag == 0xFFFEE0DD:
            return b"".join(fragments), start + length
        if tag != 0xFFFEE000 or start + length > len(payload):
            raise ValueError("DICOM encapsulated pixel fragment is invalid.")
        if first:
            first = False
        else:
            fragments.append(payload[start : start + length])
        position = start + length
    raise ValueError("DICOM encapsulated pixel delimiter is missing.")


def _raw_header(frame: DicomFrame | None) -> tuple[float, float, bool]:
    if frame is None:
        return 0.0, 0.5, False
    try:
        orientation, origin, spacing = (
            frame.numbers(tag) for tag in (0x00200037, 0x00200032, 0x00280030)
        )
        if len(orientation) == 6 and origin:
            normal = np.cross(np.asarray(orientation[:3]), np.asarray(orientation[3:]))
            position = float(cast(np.float64, np.dot(np.asarray(origin), normal)))
        else:
            position = float(frame.instance_number or 0)
        return position, spacing[0] if spacing else 0.5, True
    except (ArithmeticError, LookupError, TypeError, ValueError):
        return 0.0, 0.5, False


def _directions(
    geometry: FrameGeometry,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return unit(np.asarray(geometry.orientation[:3])), unit(
        np.asarray(geometry.orientation[3:]),
    )


def _order_study(
    acquisitions: Sequence[Acquisition],
) -> tuple[_PhysicalStudy, str | None]:
    physical = _PhysicalStudy()
    patients: set[str] = set()
    for item in acquisitions:
        if item.plane not in PLANES:
            continue
        if item.error is not None:
            physical.header_errors.append((item.series, item.error))
            continue
        try:
            order = order_frames(item.frames, plane=item.plane)
        except (
            ArithmeticError,
            LookupError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            physical.header_errors.append((item.series, str(error)[:250]))
            continue
        physical.ordered.append((item, order))
        patients.update(frame.frame.patient_id for frame in item.frames)
    patients.discard("")
    if len(patients) > 1:
        raise ValueError("Conflicting patients occur within one study.")
    return physical, study_side(
        [side for _, order in physical.ordered for side in order.sides],
        sagittal_positions=[
            order.median_x for _, order in physical.ordered if order.plane == "Sagittal"
        ],
    )


@dataclass(kw_only=True, slots=True)
class _PhysicalStudy:
    ordered: list[tuple[Acquisition, SeriesOrder]] = field(default_factory=list)
    prepared: list[PreparedSeries] = field(default_factory=list)
    header_errors: list[tuple[str, str]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def rebuild(self, side: str | None) -> set[str | None]:
        self.prepared = []
        self.errors = self.header_errors.copy()
        for acquisition, order in self.ordered:
            fat, fluid = _category(acquisition.fat), _category(acquisition.fluid)
            try:
                prepared = prepare_series(
                    order,
                    series=acquisition.series,
                    side=side,
                    fat=fat,
                    fluid=fluid,
                )
            except ValueError as error:
                self.errors.append((acquisition.series, str(error)[:250]))
                continue
            self.prepared.append(prepared)
        return {item.native.side for item in self.prepared}

    def statuses(self) -> set[str]:
        retained = {item.native.series: set(item.files) for item in self.prepared}
        sides = {
            side
            for acquisition, order in self.ordered
            if acquisition.series in retained
            for item in order.frames
            if item.name in retained[acquisition.series]
            if (side := item.frame.laterality) is not None
        }
        status = (
            "conflicting_study_tags"
            if len(sides) > 1
            else "explicit_study_tags"
            if sides
            else "unknown_no_explicit_tags"
        )
        return {status} if retained else set()


def _category(value: float | str | None) -> int:
    return (
        -1
        if value is None or (isinstance(value, float) and math.isnan(value))
        else int(value)
    )


def _anatomical(
    item: PreparedSeries,
    sampler: AnatomySampler,
    model: TensorFn,
    device: str,
) -> AnatomicalSeries:
    native = item.native
    result = AnatomicalSeries(
        series=native.series,
        plane=native.plane,
        shape=native.pixels.shape[-2:],
        affines=item.affines,
        aligned=item.quality.aligned,
        fluid=native.fluid,
        probabilities=[],
    )
    if native.plane == "Sagittal" and item.quality.aligned:
        try:
            probabilities, result.reason = sampler(
                native.pixels,
                spacing=native.spacing,
                affines=item.affines,
                model=model,
                device=device,
            )
        except (
            ArithmeticError,
            LookupError,
            ValueError,
            TypeError,
            RuntimeError,
        ) as error:
            probabilities, result.reason = None, "error:" + repr(error)[:80]
        result.probabilities = probabilities or []
    return result
