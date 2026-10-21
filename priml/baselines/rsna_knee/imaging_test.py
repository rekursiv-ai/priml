"""Check DICOM decoding and physical preparation on hand-checkable inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

import math
import re

from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

import numpy as np
import pytest
import torch

from priml.baselines.rsna_knee.data import FINDINGS
from priml.baselines.rsna_knee.imaging import (
    EXTENT_COLUMNS,
    MODEL_COLUMNS,
    Acquisition,
    AnatomicalSeries,
    AnatomySampler,
    AnchorConsensus,
    FluidMeasure,
    FrameGeometry,
    MetadataColumn,
    NamedFrame,
    NativeSeries,
    ProjectedAnchor,
    R7Render,
    RaptorBag,
    RaptorPrepare,
    RaptorSeries,
    RaptorWindows,
    RenderedStudy,
    SideCheck,
    SideNote,
    SliceBlock,
    StudyPreparation,
    _categories,
    _csv_rows,
    _PhysicalStudy,
    _render_slice,
    _resize_input,
    anatomical_mirror,
    canonical_frame,
    dicom_files,
    evaluation_rows,
    order_frames,
    prepare_series,
    quantize_series,
    raptor_slice,
    read_dicom,
    study_side,
    unchanged,
    window_centers,
)
from priml.baselines.rsna_knee.support_test import (
    STUDY_PLANES,
    corrupt_deflate,
    dicom_file,
    study_inputs,
    study_series,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray
    from torch._ops import OpOverload

    from priml.baselines.rsna_knee.imaging import DicomFrame


def test_two_agreeing_candidates_average_by_tissue_mass() -> None:
    first = _synthetic_series(shape=(8, 10))
    assert first.affines is not None
    translated = first.affines.copy()
    translated[:, 0, 2] += 10
    lighter = []
    for index, probability in first.probabilities:
        trimmed = probability.copy()
        trimmed[1:4, -2:] = 0
        lighter.append((index, trimmed))
    second = replace(first, series="second", affines=translated, probabilities=lighter)
    result = AnchorConsensus.Config().make()([first, second])
    assert result.method == "skm_tea_joint_avg2"
    assert result.reasons == []
    assert result.joint is not None
    # Seven slices of 80 and of 60 confident pixels weight the two joints.
    assert np.array_equal(
        result.joint,
        [(13 * 560 + 23 * 420) / 980, 4.5, (-3.5 * 560 - 2.5 * 420) / 980],
    )


def test_rejected_series_and_a_far_candidate_fall_back_to_the_slab_prior() -> None:
    joint = _synthetic_series(shape=(8, 10))
    assert joint.affines is not None
    shifted = joint.affines.copy()
    shifted[:, 1, 2] += 40
    coronal = replace(
        joint,
        series="coronal",
        plane="Coronal",
        affines=shifted,
        probabilities=[],
    )
    result = AnchorConsensus.Config().make()(
        [
            replace(joint, series="unaligned", aligned=False),
            replace(joint, series="missing", affines=None),
            joint,
            coronal,
        ],
    )
    assert result.reasons == ["not_aligned", "no_geo", "far_from_prior(40mm)"]
    assert result.method == "slab_prior"
    assert result.prior is not None
    assert np.array_equal(result.prior, [13.0, 44.5, -3.5])
    assert result.joint is result.prior
    # Every series with geometry, before or after the missing one, is projected.
    assert list(result.series) == ["unaligned", "synthetic", "coronal"]
    assert all(anchor.patella is None for anchor in result.series.values())


def test_candidates_at_exactly_the_prior_and_agreement_limits_are_kept() -> None:
    joint = _synthetic_series(shape=(8, 10))
    assert joint.affines is not None
    shifted = joint.affines.copy()
    shifted[:, 1, 2] += 35
    coronal = replace(
        joint,
        series="coronal",
        plane="Coronal",
        affines=shifted,
        probabilities=[],
    )
    result = AnchorConsensus.Config().make()([joint, coronal])
    assert (result.method, result.reasons) == ("skm_tea_joint", [])
    translated = joint.affines.copy()
    translated[:, 0, 2] += 20
    result = AnchorConsensus.Config().make()(
        [joint, replace(joint, series="second", affines=translated)],
    )
    assert result.method == "skm_tea_joint_avg2"
    assert result.joint is not None
    assert np.array_equal(result.joint, [23.0, 4.5, -3.5])


def test_disagreeing_candidates_keep_the_heaviest_and_report_their_distance() -> None:
    first = _synthetic_series(shape=(8, 10))
    assert first.affines is not None
    translated = first.affines.copy()
    translated[:, 0, 2] += 30
    result = AnchorConsensus.Config().make()(
        [first, replace(first, series="far", affines=translated)],
    )
    assert result.method == "skm_tea_joint"
    assert result.reasons == ["disagree(30mm)"]
    assert result.joint is not None
    assert np.array_equal(result.joint, [13.0, 4.5, -3.5])


def test_series_without_geometry_leave_no_joint_or_projection() -> None:
    result = AnchorConsensus.Config().make()(
        [replace(_synthetic_series(), affines=None)],
    )
    assert result.joint is None
    assert result.method is None
    assert result.reasons == ["no_geo", "no_geometry"]
    assert result.series == {}


def test_consensus_projects_the_joint_and_an_anterior_patella_into_each_series() -> (
    None
):
    series = _synthetic_series(shape=(16, 20))
    for _, probability in series.probabilities:
        probability[0] = 0
        probability[0, :4, :16] = 0.75
    result = AnchorConsensus.Config().make()([series])
    assert result.method == "skm_tea_joint"
    assert result.joint is not None
    assert result.patella is not None
    assert np.array_equal(result.joint, [13.0, 9.5, -7.5])
    assert np.array_equal(result.patella, [13.0, 7.5, -1.5])
    assert result.series == {
        "synthetic": ProjectedAnchor(
            joint=(0.5, 0.5),
            in_field=True,
            distance_mm=0.0,
            patella=(1.5 / 15, 7.5 / 19),
        ),
    }


def test_absent_small_or_failed_tissue_predictions_retain_the_slab_prior() -> None:
    series = _synthetic_series(shape=(8, 10))
    for _, probability in series.probabilities:
        probability.fill(0)
    result = AnchorConsensus.Config().make()([series])
    assert result.method == "slab_prior"
    assert result.reasons == ["mass<250(0)"]
    assert result.patella is None
    series.reason = "oblique"
    assert AnchorConsensus.Config().make()([series]).reasons == ["oblique"]
    series.reason = "ok"
    # AnchorConsensus._candidate requires channels 1:4; one channel is malformed.
    series.probabilities = [(0, np.ones((1, 8, 10), dtype=np.float32))]
    result = AnchorConsensus.Config().make()([series])
    assert result.method == "slab_prior"
    assert len(result.reasons) == 1
    assert result.reasons[0].startswith("error:ValueError(")


def test_unlocalized_joint_and_posterior_or_small_patella_are_rejected() -> None:
    series = _synthetic_series()
    result = AnchorConsensus.Config().make()([series])
    assert result.method == "skm_tea_joint"
    assert result.patella is None
    for _, probability in series.probabilities:
        probability[0] = 0
        probability[0, :5] = 0.75
    assert AnchorConsensus.Config().make()([series]).patella is None
    series = _synthetic_series(shape=(20, 20))
    for _, probability in series.probabilities:
        probability.fill(0)
        probability[1:4, :2] = 0.75
    result = AnchorConsensus.Config().make()([series])
    assert result.method == "slab_prior"
    assert result.reasons == ["implausible(rf=0.03,cf=0.50)"]


def test_projection_ties_choose_first_slice_and_single_pixel_fields_are_valid() -> None:
    series = _synthetic_series(shape=(16, 16))
    assert series.affines is not None
    series.affines = series.affines[:2].copy()
    series.affines[1, 1, 2] = 2
    series.probabilities = series.probabilities[:2]
    result = AnchorConsensus.Config().make()([series])
    assert result.series["synthetic"].joint == (0.5, 8.5 / 15)
    assert result.series["synthetic"].distance_mm == 0.5
    series = _synthetic_series(shape=(1, 1))
    series.plane = "Axial"
    series.probabilities = []
    result = AnchorConsensus.Config().make()([series])
    assert result.method == "slab_prior"
    assert result.series["synthetic"].joint == (0.0, 0.0)
    assert result.series["synthetic"].in_field


def test_equal_fluid_and_tissue_mass_keep_original_acquisition_order() -> None:
    first = _synthetic_series(shape=(8, 10))
    first.series = "z"
    assert first.affines is not None
    translated = first.affines.copy()
    translated[:, 0, 2] += 50
    second = replace(first, series="a", affines=translated)
    consensus = AnchorConsensus.Config().make()
    first_joint = consensus([first, second]).joint
    second_joint = consensus([second, first]).joint
    assert first_joint is not None
    assert second_joint is not None
    assert np.array_equal(first_joint, [13, 4.5, -3.5])
    assert np.array_equal(second_joint, [63, 4.5, -3.5])


def test_fluid_sensitive_sagittal_series_are_considered_first() -> None:
    series = _synthetic_series()
    result = AnchorConsensus.Config().make()(
        [
            replace(series, series="plain", fluid=0, aligned=False),
            replace(series, series="fluid", affines=None),
        ],
    )
    assert result.reasons == ["no_geo", "not_aligned", "no_geometry"]


def test_the_heaviest_candidate_leads_and_a_disagreeing_one_lends_its_patella() -> None:
    lighter = _synthetic_series(shape=(16, 20))
    for _, probability in lighter.probabilities:
        probability[0] = 0
        probability[0, :4, :16] = 0.75
        probability[1:4, :, -2:] = 0
    assert lighter.affines is not None
    translated = lighter.affines.copy()
    translated[:, 0, 2] += 30
    heavier = replace(
        _synthetic_series(shape=(16, 20)),
        series="heavier",
        affines=translated,
    )
    for _, probability in heavier.probabilities:
        probability[0] = 0
    result = AnchorConsensus.Config().make()([lighter, heavier])
    assert (result.method, result.reasons) == ("skm_tea_joint", ["disagree(30mm)"])
    assert result.joint is not None
    assert result.patella is not None
    # 320 tissue pixels per slice outweigh 288; only the lighter has a patella.
    assert np.array_equal(result.joint, [43.0, 9.5, -7.5])
    assert np.array_equal(result.patella, [13.0, 7.5, -1.5])


@pytest.mark.parametrize(
    ("columns", "patella"),
    [(18, [23.0, 8.5, -1.5]), (16, [13.0, 7.5, -1.5])],
)
def test_agreeing_candidates_take_the_heaviest_patella_and_the_first_on_ties(
    columns: int,
    patella: list[float],
) -> None:
    first = _synthetic_series(shape=(16, 20))
    for _, probability in first.probabilities:
        probability[0] = 0
        probability[0, :4, :16] = 0.75
    assert first.affines is not None
    translated = first.affines.copy()
    translated[:, 0, 2] += 10
    second = replace(
        _synthetic_series(shape=(16, 20)),
        series="second",
        affines=translated,
    )
    for _, probability in second.probabilities:
        probability[0] = 0
        probability[0, :4, :columns] = 0.75
    result = AnchorConsensus.Config().make()([first, second])
    assert result.method == "skm_tea_joint_avg2"
    assert result.patella is not None
    assert np.array_equal(result.patella, patella)


def test_candidate_errors_are_reported_in_eighty_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AnchorConsensus, "_candidate", _long_failure)
    result = AnchorConsensus.Config().make()([_synthetic_series()])
    # The repr opens with the 14 characters RuntimeError(', leaving 66 x's.
    assert result.reasons == ["error:RuntimeError('" + "x" * 66]


def _long_failure(*args: object, **kwargs: object) -> None:
    del args, kwargs
    raise RuntimeError("x" * 100)


def test_projections_scale_by_each_series_extent_and_check_both_bounds() -> None:
    series = _synthetic_series(shape=(17, 21))
    for _, probability in series.probabilities:
        probability[0] = 0
        probability[0, :4, :16] = 0.75
    assert series.affines is not None
    shifted = series.affines.copy()
    shifted[:, 0, 2] += 0.25
    others = [
        replace(series, series=name, plane="Axial", shape=shape, probabilities=[])
        for name, shape in (("rows", (8, 21)), ("columns", (17, 10)))
    ]
    tiny = replace(others[0], series="tiny", shape=(2, 2), affines=shifted)
    result = AnchorConsensus.Config().make()([series, *others, tiny])
    # The joint sits at row 8, column 10 and the patella at 1.5, 7.5 of every
    # grid; the tiny stack's nearest slice is 0.25 mm away.
    assert result.series == {
        "synthetic": ProjectedAnchor(
            joint=(0.5, 0.5),
            in_field=True,
            distance_mm=0.0,
            patella=(1.5 / 16, 7.5 / 20),
        ),
        "rows": ProjectedAnchor(
            joint=(8 / 7, 0.5),
            in_field=False,
            distance_mm=0.0,
            patella=(1.5 / 7, 7.5 / 20),
        ),
        "columns": ProjectedAnchor(
            joint=(0.5, 10 / 9),
            in_field=False,
            distance_mm=0.0,
            patella=(1.5 / 16, 7.5 / 9),
        ),
        "tiny": ProjectedAnchor(
            joint=(8.0, 10.0),
            in_field=False,
            distance_mm=0.2,
            patella=(1.5, 7.5),
        ),
    }


def _synthetic_series(*, shape: tuple[int, int] = (8, 8)) -> AnatomicalSeries:
    # AnchorConsensus._project maps column, row, origin into three LPS coordinates.
    affines = np.zeros((7, 3, 3), dtype=np.float64)
    affines[:, 1, 0] = 1
    affines[:, 2, 1] = -1
    affines[:, 0, 2] = 10 + np.arange(7)
    return AnatomicalSeries(
        series="synthetic",
        plane="Sagittal",
        shape=shape,
        affines=affines,
        aligned=True,
        fluid=1,
        probabilities=[
            (index, np.full((4, *shape), 0.75, dtype=np.float32)) for index in range(7)
        ],
    )


_PLANE_AXES: Final = {
    "Sagittal": ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    "Coronal": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "Axial": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}
"""Canonical column, row and slice-normal directions of each plane in LPS."""


@pytest.mark.parametrize("side", ["L", "R"])
@pytest.mark.parametrize("transform", range(8))
@pytest.mark.parametrize("plane", list(_PLANE_AXES))
def test_canonical_frame_maps_every_d4_orientation_onto_the_plane_axes(
    plane: str,
    transform: int,
    side: str,
) -> None:
    first, second, normal = (np.array(axis) for axis in _PLANE_AXES[plane])
    row, column = (second, first) if transform & 4 else (first, second)
    row = -row if transform & 1 else row
    column = -column if transform & 2 else column
    origin = np.array([1.0, 2.0, 3.0])
    frame = canonical_frame(
        np.arange(15, dtype=np.float32).reshape(3, 5),
        FrameGeometry(
            orientation=(*row.tolist(), *column.tolist()),
            origin=(1.0, 2.0, 3.0),
            spacing=(0.5, 0.25),
            plane=plane,
            side=side,
        ),
    )
    # Non-sagittal right knees are mirrored so their columns run toward the right.
    mirror = -1.0 if side == "R" and plane != "Sagittal" else 1.0
    assert np.array_equal(frame.affine[:, 0], frame.spacing[1] * mirror * first)
    assert np.array_equal(frame.affine[:, 1], frame.spacing[0] * second)
    assert np.array_equal(frame.basis, np.stack([first, second, normal], axis=1))
    # Every canonical pixel keeps its unique source value and physical position.
    rows, columns = np.indices(frame.image.shape)
    source_rows, source_columns = np.divmod(frame.image.astype(np.int64), 5)
    assert np.array_equal(
        frame.affine @ np.stack([columns.ravel(), rows.ravel(), np.ones(15)]),
        origin[:, None]
        + np.outer(row, 0.25 * source_columns.ravel())
        + np.outer(column, 0.5 * source_rows.ravel()),
    )


def test_d4_score_ties_choose_the_first_orientation() -> None:
    cosine = 2**-0.5
    geometry = FrameGeometry(orientation=(cosine, cosine, 0.0, -cosine, cosine, 0.0))
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert np.array_equal(canonical_frame(image, geometry).image, image)


def test_canonical_pixels_apply_the_modality_transform_and_monochrome1_inversion() -> (
    None
):
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    geometry = FrameGeometry(slope=2.0, intercept=-1.0)
    assert np.array_equal(canonical_frame(image, geometry).image, image * 2 - 1)
    geometry.monochrome1 = True
    # The scaled range -1..9 inverts around its own extremes.
    assert np.array_equal(canonical_frame(image, geometry).image, 9 - image * 2)


def test_full_range_encoding_scales_every_sample_and_keeps_the_display_window() -> None:
    # 101 constant slices put the 1st and 99th percentiles on exact samples.
    encoded = quantize_series(
        [np.full((2, 3), level, dtype=np.float32) for level in range(101)],
    )
    assert encoded.volume.dtype == np.uint16
    assert encoded.volume[[0, 1, 50, 99, 100], 0, 0].tolist() == [
        0,
        655,
        32_768,
        64_880,
        65_535,
    ]
    assert (encoded.minimum, encoded.maximum) == (0.0, 100.0)
    assert encoded.error_bound == 100 / 131_070
    assert encoded.window == (655.0, 64_880.0)
    constant = quantize_series([np.full((2, 3), 7.0, dtype=np.float32)])
    assert not constant.volume.any()
    assert constant.error_bound == 1e-6 / 131_070


@pytest.mark.parametrize(
    ("tags", "positions", "side"),
    [
        (["L"], [-10.0], "L"),
        (["R"], [10.0], "R"),
        (["L", "R"], [10.0], None),
        ([], [-10.0], "R"),
        ([], [10.0, float("nan")], "L"),
        # Each median disagrees in sign with the mean and with one extreme.
        ([], [-3.0, -2.0, 10.0], "R"),
        ([], [-10.0, 2.0, 3.0], "L"),
        ([], [], None),
    ],
)
def test_study_side_trusts_one_explicit_tag_before_the_sagittal_position(
    tags: list[str],
    positions: list[float],
    side: str | None,
) -> None:
    assert study_side(tags, positions) == side


def test_unsupported_physical_geometry_fails_before_pixel_interpolation() -> None:
    geometry = FrameGeometry()
    image = np.zeros((3, 5), dtype=np.float32)
    geometry.plane = "unknown"
    with pytest.raises(ValueError, match=r"^Unsupported anatomical plane\.$"):
        canonical_frame(image, geometry)


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (
            FrameGeometry(orientation=(1.0,)),
            "DICOM orientation and origin vectors are incomplete.",
        ),
        (
            FrameGeometry(origin=(0.0,)),
            "DICOM orientation and origin vectors are incomplete.",
        ),
        (
            FrameGeometry(spacing=(0.0, 1.0)),
            "DICOM spacing and modality transform are invalid.",
        ),
        (
            FrameGeometry(spacing=(-1.0, 1.0)),
            "DICOM spacing and modality transform are invalid.",
        ),
        (
            FrameGeometry(spacing=(float("nan"), 1.0)),
            "DICOM spacing and modality transform are invalid.",
        ),
        (FrameGeometry(slope=0.0), "DICOM spacing and modality transform are invalid."),
        (
            FrameGeometry(intercept=float("inf")),
            "DICOM spacing and modality transform are invalid.",
        ),
        (
            FrameGeometry(orientation=(1.0, 0.0, 0.0, 1.0, 1.0, 0.0)),
            "DICOM orientation axes are not orthogonal.",
        ),
        (
            FrameGeometry(plane="Sagittal"),
            "Declared plane disagrees with the slice orientation.",
        ),
    ],
)
def test_incomplete_or_inconsistent_metadata_is_rejected(
    geometry: FrameGeometry,
    message: str,
) -> None:
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$"):
        canonical_frame(np.zeros((2, 3), dtype=np.float32), geometry)


@pytest.mark.parametrize("shape", [(2, 3), (3, 2)])
def test_nonfinite_pixels_and_two_pixel_affine_denominators(
    shape: tuple[int, int],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^DICOM modality-scaled pixels are nonfinite\.$",
    ):
        canonical_frame(
            np.full(shape, float("nan"), dtype=np.float32),
            FrameGeometry(),
        )
    assert np.array_equal(
        canonical_frame(np.zeros(shape, dtype=np.float32), FrameGeometry()).affine,
        np.diag([1.0, 1.0, 0.0]),
    )
    orientation = (1.0, 0.0, 0.0, 1e-3, (1 - 1e-6) ** 0.5, 0.0)
    actual = canonical_frame(
        np.zeros(shape, dtype=np.float32),
        FrameGeometry(orientation=orientation),
    )
    assert math.isfinite(actual.affine.sum().item())
    with pytest.raises(
        ValueError,
        match=r"^DICOM orientation contains a zero or nonfinite axis\.$",
    ):
        canonical_frame(
            np.zeros(shape, dtype=np.float32),
            FrameGeometry(orientation=(0.0,) * 6),
        )


_SAMPLES: Final = np.array(
    [
        [0, 1, 127, 255, 1023],
        [2047, 2048, 4095, 32_767, 32_768],
        [65_535, 592, 17_000, 65_520, 42],
    ],
    dtype=np.uint16,
)
"""Stored samples whose 8, 10, 12 and 16-bit readings all differ."""


@pytest.mark.parametrize(
    "syntax",
    [
        pytest.param("1.2.840.10008.1.2.1", id="explicit-little"),
        pytest.param("1.2.840.10008.1.2", id="implicit-little"),
        pytest.param("1.2.840.10008.1.2.2", id="explicit-big"),
        pytest.param("1.2.840.10008.1.2.1.99", id="deflated"),
        pytest.param("1.2.840.10008.1.2.5", id="rle"),
        pytest.param("1.2.840.10008.1.2.4.90", id="jpeg2000-lossless"),
    ],
)
def test_lossless_transfer_syntaxes_round_trip_samples_and_header(syntax: str) -> None:
    frame = read_dicom(
        dicom_file(
            _SAMPLES,
            position=(13.7, -27.1, 8.2),
            orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            spacing=(0.3, 0.7),
            instance=7,
            laterality="R",
            syntax=syntax,
            slope="1.25",
            intercept="-3",
        ),
    )
    assert frame.transfer_syntax == syntax
    assert frame.little_endian is (syntax != "1.2.840.10008.1.2.2")
    assert frame.pixel_array is frame.pixel_array
    assert frame.pixel_array.dtype == np.uint16
    assert np.array_equal(frame.pixel_array, _SAMPLES)
    assert (frame.instance_number, frame.shape) == (7, (3, 5))
    assert (frame.patient_id, frame.laterality) == ("numeric-patient", "R")
    # Parsed decimal header values are exact binary64, not arithmetic results.
    assert frame.geometry("Axial", None) == FrameGeometry(
        orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        origin=(13.7, -27.1, 8.2),
        spacing=(0.3, 0.7),
        slope=1.25,
        intercept=-3.0,
    )


@pytest.mark.parametrize(
    ("samples", "stored", "expected"),
    [
        pytest.param(_SAMPLES, 12, _SAMPLES & 0xFFF, id="unsigned-12"),
        pytest.param(_SAMPLES, 10, _SAMPLES & 0x3FF, id="unsigned-10"),
        # Bit 11 carries the sign: 0x800 is -2048, 0xFFF is -1 and 0xFF0 is -16.
        pytest.param(
            _SAMPLES.view(np.int16),
            12,
            np.array(
                [
                    [0, 1, 127, 255, 1023],
                    [2047, -2048, -1, -1, 0],
                    [-1, 592, 616, -16, 42],
                ],
                dtype=np.int16,
            ),
            id="signed-12",
        ),
        pytest.param(
            _SAMPLES.view(np.int16),
            16,
            _SAMPLES.view(np.int16),
            id="signed-16",
        ),
        pytest.param(
            _SAMPLES.astype(np.uint8),
            8,
            _SAMPLES.astype(np.uint8),
            id="byte",
        ),
    ],
)
def test_stored_bits_mask_unsigned_and_sign_extend_signed_samples(
    samples: NDArray[np.integer],
    stored: int,
    expected: NDArray[np.integer],
) -> None:
    payload = dicom_file(samples, position=(0.0, 0.0, 0.0), stored=stored)
    actual = read_dicom(payload).pixel_array
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)


def test_baseline_jpeg_decodes_a_flat_byte_matrix_exactly() -> None:
    flat = np.full((3, 5), 200, dtype=np.uint8)
    frame = read_dicom(
        dicom_file(flat, position=(0.0, 0.0, 0.0), syntax="1.2.840.10008.1.2.4.50"),
    )
    # A flat block holds only its DC term, 8 * (200 - 128), which the default
    # quantizer of 8 divides, so lossy compression keeps every sample.
    assert frame.pixel_array.dtype == np.uint8
    assert np.array_equal(frame.pixel_array, flat)


@pytest.mark.parametrize(
    ("pixels", "photometric", "slope", "intercept", "expected"),
    [
        # Both tags rescale; MONOCHROME1 then inverts around the maximum alone.
        ([[0, 1, 2], [3, 4, 6]], "MONOCHROME1", "0.5", "4", [[3, 2.5, 2], [1.5, 1, 0]]),
        # A slope without an intercept leaves the stored samples unscaled.
        ([[0, 1, 2], [3, 4, 6]], "MONOCHROME2", "0.5", "", [[0, 1, 2], [3, 4, 6]]),
        # 65535 * 1.23456789 - 0.137 = 80907.26967115 rounds once to 80907.2734375;
        # rounding the float32 product first lands one 2**-7 spacing lower.
        (
            [[65_535] * 3] * 2,
            "MONOCHROME2",
            "1.23456789",
            "-0.137",
            [[80907.2734375] * 3] * 2,
        ),
    ],
)
def test_raptor_pixels_rescale_once_in_float64_and_invert_by_the_maximum(
    pixels: list[list[int]],
    photometric: str,
    slope: str,
    intercept: str,
    expected: list[list[float]],
) -> None:
    frame = read_dicom(
        dicom_file(
            np.array(pixels, dtype=np.uint16),
            position=(0.0, 0.0, 0.0),
            photometric=photometric,
            slope=slope,
            intercept=intercept,
        ),
    )
    actual = frame.raptor_pixels()
    assert actual.dtype == np.float32
    assert np.array_equal(actual, np.array(expected, dtype=np.float32))


def test_r7_rescales_round_the_float32_product_before_adding_the_intercept() -> None:
    frame = read_dicom(
        dicom_file(
            np.full((2, 3), 65_535, dtype=np.uint16),
            position=(0.0, 0.0, 0.0),
            slope="1.23456789",
            intercept="-0.137",
        ),
    )
    canonical = canonical_frame(
        frame.pixel_array.astype(np.float32),
        frame.geometry("Sagittal", None),
    )
    # 65535 * 1.23456789 rounds to 80907.40625 in float32; adding -0.137 lands on
    # 80907.265625, one 2**-7 spacing below the once-rounded 80907.2734375.
    for image in (canonical.image, frame.modality_pixels()):
        assert image.dtype == np.float32
        assert np.array_equal(image, np.full((2, 3), 80907.265625, dtype=np.float32))


def test_modality_pixels_rescale_in_float32_and_invert_monochrome1() -> None:
    frame = _axial(_SAMPLES, z=0.0)
    plain = {
        tag: value
        for tag, value in frame.values.items()
        if tag not in {0x00281053, 0x00281052}
    }
    base = torch.from_numpy(frame.pixel_array).float()
    cases = (
        ({0x00280004: b"MONOCHROME2 "}, 1.0, 0.0, False),
        (
            {0x00280004: b"MONOCHROME2 ", 0x00281053: b"2.5 ", 0x00281052: b"-3 "},
            2.5,
            -3.0,
            False,
        ),
        (
            {0x00280004: b"monochrome1", 0x00281053: b"0.5 ", 0x00281052: b"4 "},
            0.5,
            4.0,
            True,
        ),
    )
    for values, slope, intercept, inverted in cases:
        variant = replace(frame, values={**plain, **values})
        assert variant.slope == slope
        assert variant.intercept == intercept
        assert variant.monochrome1 is inverted
        scaled = base * slope + intercept
        expected_pixels = scaled.max() - scaled if inverted else scaled
        actual = variant.modality_pixels()
        assert actual.dtype == np.float32
        assert torch.equal(torch.from_numpy(actual), expected_pixels)


def test_nested_undefined_sequences_preserve_every_top_level_field() -> None:
    payload = dicom_file(_SAMPLES, position=(0.0, 0.0, 0.0))
    frame = read_dicom(payload)
    # Two sequence items: one defined-length, one with nested undefined lengths.
    item = bytes.fromhex("feff00e0")
    item_end = bytes.fromhex("feff0de000000000")
    sequence_end = bytes.fromhex("feffdde000000000")
    sequence = bytes.fromhex("0800111153510000ffffffff")
    defined = item + (4).to_bytes(4, "little") + b"data"
    nested = item + bytes.fromhex("ffffffff") + sequence + sequence_end + item_end
    injected = payload + sequence + defined + nested + sequence_end
    actual = read_dicom(injected)
    assert actual.values == frame.values
    assert actual.pixel_data == frame.pixel_data
    with pytest.raises(ValueError, match=r"^DICOM sequence delimiter is missing\.$"):
        read_dicom(payload + sequence + nested)
    with pytest.raises(
        ValueError,
        match=r"^DICOM sequence delimiter has a nonzero length\.$",
    ):
        read_dicom(payload + sequence + sequence_end[:4] + (4).to_bytes(4, "little"))


def test_header_validation_and_optional_fields() -> None:
    frame = _axial(_SAMPLES, z=0.0)
    values = dict(frame.values)
    values.update(
        {
            0x00080005: b"ISO_IR 192",
            0x00100020: "patient\N{LATIN SMALL LETTER E WITH ACUTE}".encode(),
        },
    )
    changed = replace(frame, values=values)
    assert changed.patient_id == "patient\N{LATIN SMALL LETTER E WITH ACUTE}"
    for value in (b"L", b"LEFT"):
        values.update({0x00200060: value, 0x00200062: value})
        assert changed.laterality == "L"
    values.update({0x00200060: b"", 0x00200062: b"", 0x00200013: b"", 0x11110000: b""})
    assert changed.laterality is None
    assert changed.instance_number is None
    assert changed.integer(0x11110000, 7) == 7
    values.update({0x00200060: b"R", 0x00200062: b"L"})
    with pytest.raises(ValueError, match="laterality"):
        _ = changed.laterality
    for tag, value, message in (
        (0x00280010, b"\0\0", "dimensions"),
        (0x00280002, b"\x03\0", "one monochrome"),
        (0x00280100, b"\x03\0", "bit allocation"),
        (0x00280101, b"\x11\0", "bit allocation"),
    ):
        invalid = replace(frame, values={**frame.values, tag: value})
        with pytest.raises(ValueError, match=message):
            _ = invalid.pixel_array
    for tag, value, message in (
        (0x00200037, b"1\\0", "incomplete"),
        (0x00200032, b"nan\\0\\1", "nonfinite"),
        (0x00280004, b"RGB", "photometry"),
        (0x00281053, b"nan", "modality transform"),
        (0x00281052, b"inf", "modality transform"),
        (0x00281053, b"0", "modality transform"),
    ):
        invalid = replace(frame, values={**frame.values, tag: value})
        with pytest.raises(ValueError, match=message):
            invalid.geometry("Axial", None)
    jpeg = read_dicom(
        dicom_file(
            _SAMPLES.astype(np.uint8),
            position=(0.0, 0.0, 0.0),
            syntax="1.2.840.10008.1.2.4.50",
        ),
    )
    with pytest.raises(ValueError, match="payload"):
        _ = replace(jpeg, values={**jpeg.values, 0x00280010: b"\x01\0"}).pixel_array


def test_malformed_element_and_fragment_boundaries() -> None:
    payload = dicom_file(_SAMPLES, position=(0.0, 0.0, 0.0))
    for appended, message in (
        (bytes.fromhex("0800111153510000"), "header is truncated"),
        (bytes.fromhex("080010104c4f1000"), "element extends"),
        (bytes.fromhex("e07f10004f420000ffffffff"), "pixel delimiter"),
        (
            bytes.fromhex("e07f10004f420000ffffffff080010104c4f1000"),
            "fragment is invalid",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            read_dicom(payload + appended)
    with pytest.raises(ValueError, match="meta information"):
        read_dicom(b"DICM")


def test_rle_packbits_noop_repeat_and_invalid_segments() -> None:
    frame = read_dicom(
        dicom_file(_SAMPLES, position=(0.0, 0.0, 0.0), syntax="1.2.840.10008.1.2.5"),
    )
    values = {
        **frame.values,
        0x00280010: b"\x03\0",
        0x00280011: b"\x02\0",
        0x00280100: b"\x08\0",
        0x00280101: b"\x08\0",
    }
    header = (1).to_bytes(4, "little") + (64).to_bytes(4, "little") + bytes(56)
    repeated = replace(frame, values=values, pixel_data=header + bytes([128, 251, 7]))
    assert np.array_equal(repeated.pixel_array, np.full((3, 2), 7, dtype=np.uint8))
    for encoded, message in (
        (b"", "segment count"),
        (header + bytes([254]), "repeated byte"),
        (header + bytes([0, 7]), "dimensions"),
    ):
        with pytest.raises(ValueError, match=message):
            _ = replace(frame, values=values, pixel_data=encoded).pixel_array


def test_render_maps_each_view_crop_through_the_display_curve() -> None:
    # _render_slice draws each view on a square canvas; these levels are the full view.
    levels = np.arange(1, 17).reshape(4, 4)
    # Each 2x2 block varies by one around its level, so halving by area averages it.
    image = np.kron(levels, np.ones((2, 2), dtype=np.int64)) + np.tile(
        [[1, -1], [-1, 1]],
        (4, 4),
    )
    series = NativeSeries(
        series="7",
        # 257 maps each stored level onto one display level.
        pixels=(257 * np.stack([image, image + 1])).astype(np.uint16),
        positions=np.array([0.0, 4.0]),
        spacing=np.ones((2, 2)),
        plane="Sagittal",
        slot=2,
        side="L",
        fluid=1,
        fat=0,
        window=(0.0, 65_535.0),
    )
    study = R7Render.Config(image_size=4, local_mm=4.0).make()([series])
    # The full view, then the joint, patellar and posterior 4 mm crops.
    crops = [levels, image[2:6, 2:6], image[0:4, 4:8], image[4:8, 0:4]]
    assert torch.equal(
        study.pixels,
        torch.from_numpy(
            np.stack([crop + offset for crop in crops for offset in (0, 1)]).astype(
                np.uint8,
            ),
        ),
    )
    assert torch.equal(
        study.metadata,
        torch.tensor(
            [
                [
                    0,
                    2,
                    0,
                    view,
                    relative,
                    math.log1p(4.0),
                    extent / 130,
                    1,
                    0,
                    index,
                    2,
                    0,
                ]
                for view, extent in zip((0, 1, 4, 5), (8.0, 4.0, 4.0, 4.0), strict=True)
                for index, relative in enumerate((-1.0, 1.0))
            ],
            dtype=torch.float32,
        ),
    )
    assert study.blocks == [
        SliceBlock(
            series="7",
            plane="Sagittal",
            n_slices=2,
            views=[(0, 0), (1, 2), (4, 4), (5, 6)],
        ),
    ]
    assert study.removed == 0
    assert evaluation_rows(study, stride=1, slab_mm=3.5).tolist() == [
        row
        for start in (0, 2, 4, 6)
        for row in (
            [0, start, start, start, start + 1],
            [0, start + 1, start, start + 1, start + 1],
        )
    ]


def test_evaluation_rows_step_by_the_physical_slab_and_keep_the_last_slice() -> None:
    metadata = torch.zeros(6, 12)
    metadata[:, MetadataColumn.LOG_SPACING] = math.log1p(1.0)
    study = RenderedStudy(
        pixels=torch.zeros(6, 2, 3, dtype=torch.uint8),
        metadata=metadata,
        blocks=[SliceBlock(series="s", plane="Axial", n_slices=6, views=[(0, 0)])],
        removed=0,
    )
    # A 3 mm slab over 1 mm slices steps 3; stride 2 jumps 6 and appends slice 5.
    assert evaluation_rows(study, stride=2, slab_mm=3.0).tolist() == [
        [0, 1, 0, 1, 4],
        [0, 5, 2, 5, 5],
    ]


def test_thin_acquisitions_average_physical_slabs_before_rendering() -> None:
    levels = 2 * np.arange(8, dtype=np.uint16)
    series = NativeSeries(
        series="thin",
        pixels=np.broadcast_to(257 * levels[:, None, None], (8, 2, 3)).copy(),
        positions=np.arange(8.0),
        spacing=np.ones((8, 2)),
        plane="Sagittal",
        slot=0,
        side="R",
        fluid=None,
        fat=None,
        window=(0.0, 65_535.0),
    )
    study = R7Render.Config(image_size=3).make()([series])
    # A right sagittal series runs backward; four 1 mm slices make a 3.5 mm slab.
    full = np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    local = np.ones((3, 3), dtype=np.uint8)
    assert torch.equal(
        study.pixels,
        torch.from_numpy(
            np.stack(
                [
                    view * level
                    for view in (full, local, local, local)
                    for level in (11, 3)
                ],
            ),
        ),
    )
    assert study.metadata[:2, MetadataColumn.RELATIVE].tolist() == [-1.0, 1.0]
    assert torch.equal(
        study.metadata[:, MetadataColumn.LOG_SPACING],
        torch.full((8,), math.log1p(4.0)),
    )


def test_slab_grouping_and_spacing_read_the_median_slice_gap() -> None:
    series = NativeSeries(
        series="gapped",
        pixels=np.zeros((16, 2, 3), dtype=np.uint16),
        positions=np.array([*range(15), 29.0]),
        spacing=np.ones((16, 2)),
        plane="Axial",
        slot=0,
        side="L",
        fluid=None,
        fat=None,
        window=(0.0, 65_535.0),
    )
    study = R7Render.Config(image_size=3).make()([series])
    # The median 1 mm gap groups four slices; the mean gap, 29/15 mm, is not thin.
    assert [block.n_slices for block in study.blocks] == [4]
    # Slab centers 1.5, 5.5, 9.5 and 17 leave a median gap of 4 mm, a maximum of 7.5.
    assert torch.equal(
        study.metadata[:, MetadataColumn.LOG_SPACING],
        torch.full((16,), math.log1p(4.0)),
    )


def test_empty_render_preserves_typed_zero_rows_and_blocks() -> None:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        study = R7Render.Config(image_size=3).make()([])
        rendered = R7Render.Config(image_size=3).make()([_small_series()])
    finally:
        torch.set_default_dtype(previous_dtype)
    assert rendered.metadata.dtype == torch.float32
    assert study.pixels.shape == (0, 3, 3)
    assert study.pixels.dtype == torch.uint8
    assert study.metadata.shape == (0, 12)
    assert study.metadata.dtype == torch.float32
    assert study.blocks == []
    assert study.removed == 0
    assert evaluation_rows(study, stride=3, slab_mm=3.5).shape == (0, 5)


@pytest.mark.parametrize("anchor", ["joint", "patella"])
def test_distinct_rendering_anchors_survive_deduplication(anchor: str) -> None:
    first = NativeSeries(
        series="first",
        pixels=np.arange(2 * 5 * 7, dtype=np.uint16).reshape(2, 5, 7),
        positions=np.array([0.0, 3.5]),
        spacing=np.ones((2, 2)),  # NativeSeries requires two spacing axes per slice.
        plane="Sagittal",
        slot=1,
        side="L",
        fluid=1,
        fat=1,
        window=(0.0, 69.0),
        joint=(0.1, 0.1),
        patella=(0.1, 0.1),
    )
    second = replace(
        first,
        series="second",
        joint=(0.9, 0.9) if anchor == "joint" else first.joint,
        patella=(0.9, 0.9) if anchor == "patella" else first.patella,
    )
    config = R7Render.Config()
    config.image_size, config.local_mm = 3, 2.0
    render = config.make()
    left, right = render([first]), render([second])
    assert not torch.equal(left.pixels, right.pixels)
    combined = render([first, second, replace(first, series="duplicate")])
    assert combined.removed == 1
    assert [block.series for block in combined.blocks] == ["first", "second"]
    assert torch.equal(combined.pixels, torch.cat([left.pixels, right.pixels]))
    equivalent = replace(
        first,
        series="equivalent",
        joint=(0.11, 0.11),
        patella=(0.11, 0.11),
    )
    assert torch.equal(render([equivalent]).pixels, left.pixels)
    assert render([first, equivalent]).removed == 1


@pytest.mark.parametrize("difference", ["metadata", "positions", "spacing", "pixels"])
def test_only_identical_physical_inputs_are_deduplicated(difference: str) -> None:
    first = _small_series()
    second = replace(first, series="other")
    if difference == "metadata":
        second.slot += 1
    elif difference == "positions":
        second.positions = second.positions + 0.01
    elif difference == "spacing":
        second.spacing = second.spacing * 2
    else:
        second.pixels = second.pixels + 1
    render = R7Render.Config(image_size=3).make()
    left, right = render([first]), render([second])
    # A removed duplicate must not end the scan before the distinct series.
    combined = render([first, replace(first, series="duplicate"), second])
    assert combined.removed == 1
    assert [block.series for block in combined.blocks] == ["small", "other"]
    assert torch.equal(combined.pixels, torch.cat([left.pixels, right.pixels]))


def test_short_physical_span_and_single_frame_metadata_are_finite() -> None:
    series = _small_series()
    study = R7Render.Config(image_size=3).make()([series])
    assert torch.equal(study.metadata[:2, 4], torch.tensor([-1.0, 1.0]))
    assert torch.isfinite(study.metadata).all()
    series.pixels = series.pixels[:1]
    series.positions = series.positions[:1]
    series.spacing = series.spacing[:1]
    study = R7Render.Config(image_size=3).make()([series])
    assert torch.equal(study.metadata[:, 4], torch.full((4,), -1.0))
    assert torch.equal(study.metadata[:, 5], torch.zeros(4))


def test_invalid_position_geometry_crop_and_window_are_rejected() -> None:
    series = _small_series()
    series.positions = np.array([float("nan"), 1.0])
    with pytest.raises(
        ValueError,
        match=r"^Native volume and slice positions are incomplete\.$",
    ):
        R7Render.Config().make()([series])
    for config, message in (
        (R7Render.Config(image_size=0), "Invalid native image geometry."),
        (R7Render.Config(local_mm=-1), "Invalid physical crop extent."),
    ):
        with pytest.raises(ValueError, match="^" + message.replace(".", r"\.") + "$"):
            config.make()([_small_series()])
    series = _small_series()
    # R7Render._slabs and _views require two spacing axes per physical slice.
    series.spacing = -np.ones((2, 2))
    with pytest.raises(ValueError, match=r"^Invalid native image geometry\.$"):
        R7Render.Config().make()([series])
    series = _small_series()
    series.window = (float("nan"), 10.0)
    with pytest.raises(ValueError, match=r"^Invalid uint16 display window\.$"):
        R7Render.Config().make()([series])


def test_a_window_under_one_level_maps_the_full_range_without_floor_or_knee() -> None:
    levels = np.array([[0, 51, 102], [153, 204, 255]])
    rendered, extent = _render_slice(
        (257 * levels).astype(np.uint16),
        (1.0, 1.0),
        3,
        crop_mm=None,
        center=(0.5, 0.5),
        window=(7.0, 7.5),
        floor=0.02,
        knee=0.9,
    )
    # 0-65535 maps linearly onto 0-1; the 2x3 field fills the top two rows.
    assert (rendered.tolist(), extent) == ([*levels.tolist(), [0, 0, 0]], 3.0)


def test_each_slice_letterboxes_with_its_own_row_and_column_spacing() -> None:
    config = R7Render.Config(image_size=6)
    config.view_offsets_mm = {"Axial": []}
    series = replace(
        _small_series(),
        pixels=np.full((2, 3, 4), 65_535, dtype=np.uint16),
        positions=np.array([0.0, 5.0]),
        spacing=np.array([[1.0, 2.0], [3.0, 4.0]]),
        plane="Axial",
    )
    study = config.make()([series])
    # 3x4 pixels of 1x2 mm span 3x8 mm, of 3x4 mm 9x16 mm: 2 and 3 of 6 rows.
    expected = np.zeros((2, 6, 6), dtype=np.uint8)
    expected[0, 2:4] = expected[1, 1:4] = 255
    assert torch.equal(study.pixels, torch.from_numpy(expected))
    assert torch.equal(
        study.metadata[:, MetadataColumn.EXTENT],
        torch.tensor([8 / 130, 16 / 130]),
    )
    # Unknown fluid and fat categories are stored as -1.
    for column in (MetadataColumn.FLUID, MetadataColumn.FAT):
        assert study.metadata[:, column].tolist() == [-1.0, -1.0]


@pytest.mark.parametrize(
    "spacing",
    [[[1.0, 2.0], [2.0, 1.0]], [[1.0, 1.0], [2.0, 1.0]]],
)
def test_deduplication_compares_each_slice_crop_at_its_own_spacing(
    spacing: list[list[float]],
) -> None:
    config = R7Render.Config(image_size=3, local_mm=6.0)
    config.view_offsets_mm = {"Axial": [(1, 0.0, 0.0)]}
    first = replace(
        _small_series(),
        pixels=np.zeros((2, 3, 5), dtype=np.uint16),
        positions=np.array([0.0, 5.0]),
        spacing=np.array(spacing),
        plane="Axial",
        joint=(0.1, 0.1),
    )
    # Only the second slice's 2 mm rows crop two of three rows, which the joint
    # rows 0.1 and 0.9 then place differently.
    study = config.make()([first, replace(first, series="moved", joint=(0.9, 0.1))])
    assert (study.removed, [block.series for block in study.blocks]) == (
        0,
        ["small", "moved"],
    )


def _small_series() -> NativeSeries:
    return NativeSeries(
        series="small",
        pixels=np.arange(30, dtype=np.uint16).reshape(2, 3, 5),
        positions=np.array([0.0, 0.1]),
        # R7Render._slabs and _views require two spacing axes per physical slice.
        spacing=np.ones((2, 2)),
        plane="Sagittal",
        slot=0,
        side="L",
        fluid=None,
        fat=None,
        window=(0.0, 65_535.0),
    )


def test_fluid_volume_opens_bright_masks_in_the_lowest_median_sagittal_series() -> None:
    # FluidMeasure squares the last-dim pixel size of square R7Render slices.
    pixels = torch.zeros(8, 6, 6, dtype=torch.uint8)
    # 37 of series 9's 72 pixels are 50, the rest 240: its median stays below
    # series 8's 100 while its mean and maximum exceed it.
    pixels[0:2] = 240
    pixels[0] = 50
    pixels[1, 0, 0] = 50
    pixels[2:4, 1:5, 1:5] = 240
    pixels[4:6, 3, 3] = 255
    pixels[6:8] = 100
    metadata = torch.zeros(8, 12)
    metadata[:, MetadataColumn.FLUID] = 1
    metadata[:, MetadataColumn.LOG_SPACING] = 0.75
    # 0.09375 * 130 mm over 6 pixels is 2.03125 mm, so the opening radius is one.
    metadata[:, MetadataColumn.EXTENT] = 0.09375
    blocks = [
        SliceBlock(
            series="9",
            plane="Sagittal",
            n_slices=2,
            views=[(0, 0), (4, 2), (5, 4)],
        ),
        SliceBlock(series="8", plane="Sagittal", n_slices=2, views=[(0, 6)]),
        SliceBlock(series="2", plane="Coronal", n_slices=2, views=[(0, 0)]),
    ]
    result = FluidMeasure.Config().make()(
        RenderedStudy(pixels=pixels, metadata=metadata, blocks=blocks, removed=0),
    )
    slice_mm = float(np.float32(np.expm1(np.float32(0.75))))
    assert result.status == "ok"
    assert (result.series, result.count, result.slices, result.median) == (
        "9",
        2,
        2,
        50.0,
    )
    assert result.spacing == slice_mm
    # Opening a 4x4 square with the 3x3 ellipse keeps all but its four corners.
    assert result.effusion == 2 * 12 * 2.03125**2 * slice_mm / 1_000.0
    assert result.baker == 0.0
    assert result.by_region() == {
        "eff_sag_p99_open": result.effusion,
        "bak_sag_p99_open": 0.0,
    }


@pytest.mark.parametrize(
    ("views", "fluid", "status"),
    [
        ((0, 4, 5), 0.0, "no fluid-sensitive sagittal series"),
        ((0, 5), 1.0, "no region view for eff_sag_p99_open"),
        ((4, 5), 1.0, "error: KeyError(0)"),
    ],
)
def test_fluid_measurement_reports_each_missing_input(
    views: tuple[int, ...],
    fluid: float,
    status: str,
) -> None:
    metadata = torch.zeros(6, 12)
    metadata[:, MetadataColumn.FLUID] = fluid
    metadata[:, MetadataColumn.EXTENT] = 0.09375
    block = SliceBlock(
        series="9",
        plane="Sagittal",
        n_slices=2,
        views=[(view, 2 * index) for index, view in enumerate(views)],
    )
    result = FluidMeasure.Config().make()(
        RenderedStudy(
            # FluidMeasure squares the last-dim pixel size of square R7Render slices.
            pixels=torch.full((6, 8, 8), 50, dtype=torch.uint8),
            metadata=metadata,
            blocks=[block],
            removed=0,
        ),
    )
    assert result.status == status
    assert result.effusion is None


def _axial(
    pixels: NDArray[np.uint16],
    *,
    z: float,
    spacing: tuple[float, float] = (0.5, 0.5),
) -> DicomFrame:
    return read_dicom(
        dicom_file(
            pixels,
            position=(0.0, 0.0, z),
            orientation=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            spacing=spacing,
        ),
    )


def _constant(level: int, *, z: float) -> DicomFrame:
    return _axial(np.full((2, 3), level, dtype=np.uint16), z=z)


def _ramp() -> DicomFrame:
    return _axial(10 * np.arange(6, dtype=np.uint16).reshape(2, 3), z=0.0)


def test_raptor_slots_prefer_fluid_series_pick_ordered_frames_and_truncate() -> None:
    prepare = RaptorPrepare.Config(
        image_size=2,
        slots=[("Sagittal", 1, 3), ("Sagittal", 0, 2), ("Coronal", 1, 2)],
    ).make()
    bag = prepare(
        [
            RaptorSeries(
                series="a",
                plane="Sagittal",
                fluid=0.0,
                frames=[
                    _constant(100, z=2.0),
                    _constant(0, z=0.0),
                    _constant(50, z=1.0),
                ],
            ),
            RaptorSeries(
                series="b",
                plane="Sagittal",
                fluid="1",
                frames=[
                    _constant(100, z=3.0),
                    _constant(100, z=2.0),
                    _constant(50, z=1.0),
                    _constant(0, z=0.0),
                ],
            ),
            RaptorSeries(series="c", plane="Axial", fluid=1.0, frames=[_ramp()]),
        ],
    )
    # Over a 0-100 window, 50 and 100 truncate to 127 and 254; over 0-50, 50 is 254.
    levels = np.array([0, 127, 254, 0, 254, 0, 0], dtype=np.uint8)
    assert np.array_equal(bag.pixels, np.broadcast_to(levels[:, None, None], (7, 2, 2)))
    assert bag.mask.tolist() == [0, 1, 1, 0, 1, 0, 0]
    assert (bag.slot_sizes, bag.sagittal_slots) == ((3, 2, 2), (0, 1))


def test_raptor_window_spans_the_2nd_to_98th_percentile_of_selected_frames() -> None:
    prepare = RaptorPrepare.Config(image_size=2, slots=[("Axial", 1, 2)]).make()
    frames = [
        _axial(np.full((2, 3), 65_535, dtype=np.uint16), z=2.0),
        _axial(np.array([[0, 100], [200, 300]], dtype=np.uint16), z=0.0),
        _axial(np.array([[400, 500], [600, 700]], dtype=np.uint16), z=1.0),
    ]
    bag = prepare([RaptorSeries(series="s", plane="Axial", fluid=1, frames=frames)])
    # Two slots read the two lowest of three frames. Their eight levels 0-700 put
    # the 2nd and 98th percentiles at 14 and 686; over that 672 span, levels
    # 100-600 truncate to 32-222 and 700 clips to 255.
    assert bag.pixels.tolist() == [[[0, 32], [70, 108]], [[146, 184], [222, 255]]]


def test_zero_row_spacing_crops_with_the_median_header_spacing() -> None:
    image = np.zeros((5, 6), dtype=np.uint16)
    image[0:4, 1:5] = [
        [100, 100, 0, 0],
        [100, 0, 0, 0],
        [0, 0, 100, 0],
        [0, 0, 100, 100],
    ]
    frames = [
        _axial(image, z=0.0, spacing=(0.0, 0.5)),
        _axial(np.zeros((5, 6), dtype=np.uint16), z=1.0, spacing=(1.0, 0.5)),
        _axial(np.zeros((5, 6), dtype=np.uint16), z=2.0, spacing=(4.0, 0.5)),
    ]
    prepare = RaptorPrepare.Config(
        image_size=2,
        crop_mm=4.0,
        slots=[("Axial", 1, 1)],
    ).make()
    bag = prepare([RaptorSeries(series="s", plane="Axial", fluid=1, frames=frames)])
    # Spacings 0, 1 and 4 mm have median 1 mm, so the crop is the central 4x4. Area
    # halving averages its quadrants to 75, 0, 0 and 75; over the 0-100 window of
    # 24 zeros and 6 hundreds, 75 truncates to 191.
    assert bag.pixels.tolist() == [[[191, 0], [0, 191]]]


def test_empty_missing_header_and_missing_pixel_inputs_keep_all_slots() -> None:
    prepare = RaptorPrepare.Config(image_size=3, slots=[("Axial", 1, 5)]).make()
    assert np.count_nonzero(prepare([]).mask) == 0
    assert (
        np.count_nonzero(
            prepare(
                [RaptorSeries(series="empty", plane="Axial", fluid=1, frames=[])],
            ).mask,
        )
        == 0
    )
    valid, invalid_header = _ramp(), _ramp()
    invalid_header.values[0x00280030] = b"invalid"
    actual = prepare(
        [
            RaptorSeries(
                series="series",
                plane="Axial",
                fluid=1,
                frames=[None, valid, invalid_header],
            ),
        ],
    )
    assert actual.mask.tolist() == [0, 0, 0, 1, 1]
    assert (
        np.count_nonzero(
            prepare(
                [RaptorSeries(series="series", plane="Axial", fluid=1, frames=[None])],
            ).mask,
        )
        == 0
    )
    valid.values.pop(0x00200037)
    valid.values.pop(0x00200013)
    valid.values.pop(0x00280030)
    assert (
        np.count_nonzero(
            prepare(
                [RaptorSeries(series="series", plane="Axial", fluid=1, frames=[valid])],
            ).mask,
        )
        == 5
    )


def test_a_frame_whose_pixels_fail_to_decode_leaves_its_slot_empty() -> None:
    prepare = RaptorPrepare.Config(image_size=2, slots=[("Axial", 1, 2)]).make()
    ramp = 10 * np.arange(6, dtype=np.uint16).reshape(2, 3)
    broken = replace(_axial(ramp, z=1.0), pixel_data=b"")
    frames = [_axial(ramp, z=0.0), broken, _axial(ramp, z=2.0)]
    bag = prepare([RaptorSeries(series="s", plane="Axial", fluid=1, frames=frames)])
    # Two slots read the two lowest of three frames; the second has no pixels.
    assert bag.mask.tolist() == [1, 0]


@pytest.mark.parametrize(
    ("fluid", "failure"),
    [
        (None, None),
        ("0", None),
        (0.0, None),
        (float("nan"), ValueError),
        (float("inf"), OverflowError),
    ],
)
def test_csv_fluid_categories_convert_like_integers_and_reject_nonfinite_values(
    fluid: float | str | None,
    failure: type[Exception] | None,
) -> None:
    prepare = RaptorPrepare.Config(image_size=3, slots=[("Sagittal", 0, 2)]).make()
    inputs = [
        RaptorSeries(series="1", plane="Sagittal", fluid=1, frames=[]),
        RaptorSeries(series="2", plane="Sagittal", fluid=fluid, frames=[_ramp()]),
    ]
    if failure is None:
        assert prepare(inputs).mask.tolist() == [1, 1]
    else:
        with pytest.raises(failure):
            prepare(inputs)


def test_raptor_slice_crops_the_center_square_and_truncates_its_window() -> None:
    image = np.zeros((6, 8), dtype=np.float32)
    # The centered 4x4 crop holds four 2x2 blocks averaging 4.5, 9, 0 and 2.25.
    image[1:5, 2:6] = [[0, 9, 9, 9], [9, 0, 9, 9], [0, 0, 9, 0], [0, 0, 0, 0]]
    pixels = raptor_slice(
        image,
        spacing=0.5,
        window=np.array([0.0, 9.0]),
        image_size=2,
        crop_mm=2.0,
    )
    assert pixels.dtype == np.uint8
    # Over the 9 + 1e-6 span, 2.25, 4.5 and 9 truncate to 63, 127 and 254.
    assert pixels.tolist() == [[127, 254], [0, 63]]
    # A zero spacing counts as 1e-3 mm, so the crop spans the narrower side.
    assert np.array_equal(
        raptor_slice(
            np.full((5, 7), 9.0, dtype=np.float32),
            spacing=0.0,
            window=np.array([0.0, 9.0]),
            image_size=2,
            crop_mm=1.0,
        ),
        np.full((2, 2), 254, dtype=np.uint8),
    )


def _slice(
    image: np.ndarray,
    *,
    spacing: float = 0.5,
    crop_mm: float = 140.0,
    image_size: int = 3,
) -> np.ndarray:
    return raptor_slice(
        image,
        spacing=spacing,
        window=np.array([0.0, 9.0]),
        image_size=image_size,
        crop_mm=crop_mm,
    )


_PIXELS = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)


@pytest.mark.parametrize(
    ("image", "spacing", "crop_mm", "image_size"),
    [
        (_PIXELS, float("nan"), 140.0, 3),
        (_PIXELS, float("inf"), 140.0, 3),
        (_PIXELS, 0.5, 0.0, 3),
        (_PIXELS, 0.5, -1.0, 3),
        (_PIXELS, 0.5, float("nan"), 3),
        (_PIXELS, 0.5, float("inf"), 3),
        (_PIXELS, 0.5, 140.0, 0),
        (np.zeros((0, 5), dtype=np.float32), 0.5, 140.0, 3),
        (np.zeros((2, 0), dtype=np.float32), 0.5, 140.0, 3),
    ],
)
def test_invalid_crop_geometry_is_a_native_value_error(
    image: np.ndarray,
    spacing: float,
    crop_mm: float,
    image_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Raptor crop geometry must be finite and nonempty\.$",
    ):
        _slice(image, spacing=spacing, crop_mm=crop_mm, image_size=image_size)


def test_a_crop_smaller_than_one_pixel_is_a_native_value_error() -> None:
    with pytest.raises(ValueError, match=r"^Raptor crop contains no native pixels\.$"):
        _slice(_PIXELS, spacing=1_000.0)


def test_a_single_pixel_output_is_valid() -> None:
    assert _slice(_PIXELS, image_size=1).shape == (1, 1)


def test_the_crop_never_exceeds_the_narrower_image_side() -> None:
    image = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    crop = image[2:4, 0:2]
    expected = (np.clip(crop / (9.0 + 1e-6), 0, 1) * 255).astype(np.uint8)
    assert np.array_equal(
        _slice(image, spacing=1.0, crop_mm=100.0, image_size=2),
        expected,
    )


def test_csv_rows_keep_embedded_crlf_and_decode_utf8(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes('name,note\r\nx,"line1\r\nline2 \u00e9"\r\n'.encode())
    assert _csv_rows(path) == [{"name": "x", "note": "line1\r\nline2 \u00e9"}]


@pytest.mark.parametrize("payload", [b"a\n\xff\n", b"a\n\xc3"])
def test_csv_rows_reject_invalid_or_truncated_utf8(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "rows.csv"
    path.write_bytes(payload)
    with pytest.raises(UnicodeDecodeError):
        _csv_rows(path)


def _triplets(volume: NDArray[np.uint8], centers: Tensor) -> Tensor:
    return torch.from_numpy(volume)[torch.stack([centers - 1, centers, centers + 1], 1)]


def test_raptor_windows_gather_neighbor_triplets_and_normalize_each_channel() -> None:
    # RaptorWindows rejects non-square slices; 5x5 leaves room for the 3-pixel crop.
    volume = (torch.arange(96 * 5 * 5).reshape(96, 5, 5) % 256).byte().numpy()
    config = RaptorWindows.Config()
    config.crop_size = 3
    windows = config.make()
    # RaptorWindows normalizes by its three-channel mean and std viewed as (3, 1, 1).
    mean = torch.tensor(config.mean).view(3, 1, 1)
    std = torch.tensor(config.std).view(3, 1, 1)
    # Every slice is valid, so the 94 centers step through 1..94.
    centers = torch.arange(1, 95)
    triplets = _triplets(volume, centers)
    sagittal = (centers < 26 + 22).view(-1, 1, 1, 1)
    for reorient, expected in (
        (unchanged, triplets),
        (anatomical_mirror, torch.where(sagittal, triplets.flip(1), triplets.flip(-1))),
    ):
        assert torch.equal(
            windows(
                _bag(volume, np.ones(96, dtype=np.uint8)),
                output_size=3,
                reorient=reorient,
            ),
            (expected[..., 1:4, 1:4].float() / 255 - mean) / std,
        )
    first = _triplets(volume, torch.ones(94, dtype=torch.int64))
    assert torch.equal(
        windows(_bag(volume, np.zeros(96, dtype=np.uint8)), output_size=3),
        (first[..., 1:4, 1:4].float() / 255 - mean) / std,
    )


def test_raptor_windows_resize_crops_to_the_model_side_before_normalizing() -> None:
    # RaptorWindows checks slices are square and resizes them to a square output_size.
    levels = (torch.arange(96 * 2 * 2).reshape(96, 2, 2) % 256).byte()
    volume = torch.zeros(96, 5, 5, dtype=torch.uint8)
    # Constant 2x2 blocks make the halving bilinear resize exact.
    volume[:, :4, :4] = levels.repeat_interleave(2, 1).repeat_interleave(2, 2)
    config = RaptorWindows.Config()
    config.crop_size = 4
    mask = np.ones(96, dtype=np.uint8)
    # RaptorWindows normalizes by its three-channel mean and std viewed as (3, 1, 1).
    mean = torch.tensor(config.mean).view(3, 1, 1)
    std = torch.tensor(config.std).view(3, 1, 1)
    assert torch.equal(
        config.make()(_bag(volume.numpy(), mask), output_size=2),
        (_triplets(levels.numpy(), torch.arange(1, 95)).float() / 255 - mean) / std,
    )
    # 3-to-2 half-pixel weights are 3/4 and 1/4; nearest, corner and area reads differ.
    volume = torch.zeros(96, 5, 5, dtype=torch.uint8)
    volume[:, 1:4, 1:4] = 255 * torch.tensor(
        [[1, 1, 0], [0, 1, 0], [1, 0, 0]],
        dtype=torch.uint8,
    )
    config.crop_size = 3
    # RaptorWindows interpolates all 94 windows to a square output_size.
    weighted = (torch.tensor([[0.8125, 0.25], [0.625, 0.0625]]) - mean) / std
    assert torch.equal(
        config.make()(_bag(volume.numpy(), mask), output_size=2),
        weighted.expand(94, 3, 2, 2),
    )


def test_window_centers_cover_the_valid_interval_with_rounded_steps() -> None:
    assert window_centers(np.ones(96, dtype=np.uint8), 96, 94).tolist() == list(
        range(1, 95),
    )
    assert window_centers(np.zeros(96, dtype=np.uint8), 96, 94).tolist() == [1] * 94
    mask = np.zeros(96, dtype=np.uint8)
    mask[20:24] = 1
    assert window_centers(mask, 96, 94).tolist() == [21] * 47 + [22] * 47


def test_raptor_windows_validate_slot_geometry_and_normalization() -> None:
    config = RaptorWindows.Config()
    config.crop_size = 3
    windows = config.make()
    with pytest.raises(
        ValueError,
        match=r"^Raptor slots must sum to the complete volume depth\.$",
    ):
        # The default slots hold 96 square slices; 5x5 leaves room for the 3-pixel crop.
        windows(
            _bag(np.zeros((95, 5, 5), dtype=np.uint8), np.zeros(95, dtype=np.uint8)),
            output_size=3,
        )
    with pytest.raises(
        ValueError,
        match=r"^Raptor input pixels and slice mask must match their geometry\.$",
    ):
        # The default slots hold 96 square slices; 5x5 leaves room for the 3-pixel crop.
        windows(
            _bag(np.zeros((96, 5, 5), dtype=np.uint8), np.zeros(95, dtype=np.uint8)),
            output_size=3,
        )
    with pytest.raises(
        ValueError,
        match=r"^Raptor input pixels and slice mask must match their geometry\.$",
    ):
        windows(
            _bag(np.zeros((96, 4, 5), dtype=np.uint8), np.zeros(96, dtype=np.uint8)),
            output_size=3,
        )
    # The default slots hold 96 square slices; 5x5 leaves room for the 3-pixel crop.
    full = _bag(np.zeros((96, 5, 5), dtype=np.uint8), np.zeros(96, dtype=np.uint8))
    for output_size in (0, -1):
        with pytest.raises(ValueError, match="crop"):
            windows(full, output_size=output_size)
    config.crop_size = 6
    with pytest.raises(ValueError, match="crop"):
        config.make()(full, output_size=3)
    config.crop_size = 0
    with pytest.raises(ValueError, match="crop"):
        config.make()
    config.crop_size = 3
    config.std = (1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="normalization"):
        config.make()


def _bag(volume: NDArray[np.uint8], mask: NDArray[np.uint8]) -> RaptorBag:
    return RaptorBag(
        pixels=volume,
        mask=mask,
        slot_sizes=tuple(slot[2] for slot in RaptorPrepare.Config().slots),
        sagittal_slots=(0, 1),
    )


def _tensors(value: object) -> Iterator[Tensor]:
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for child in cast("list[object] | tuple[object, ...]", value):
            yield from _tensors(child)


class _DeviceGather(TorchDispatchMode):
    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        if func.name() == "aten::index.Tensor":
            assert all(tensor.device.type == "meta" for tensor in _tensors(args))
        return func(*args, **(kwargs or {}))


def test_requested_device_owns_gather_tables_indices_and_resize() -> None:
    config = RaptorWindows.Config()
    # The default slots hold 96 square slices; 5x5 leaves room for the 3-pixel crop.
    bag = _bag(np.zeros((96, 5, 5), dtype=np.uint8), np.ones(96, dtype=np.uint8))
    for crop_size in (3, 4):
        config.crop_size = crop_size
        with _DeviceGather():
            result = config.make()(
                bag,
                output_size=3,
                device="meta",
                reorient=anatomical_mirror,
            )
        assert result.device.type == "meta"
        assert result.shape == (94, 3, 3, 3)


def test_two_slice_volume_centers_on_its_only_interior_candidate() -> None:
    centers = window_centers(np.ones(2, dtype=np.uint8), 2, 3)
    assert centers.tolist() == [1, 1, 1]


@pytest.mark.parametrize(
    ("column", "row"),
    [(1.0, -1.0), (-1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)],
)
def test_anatomy_input_is_normalized_flipped_and_restored_per_orientation(
    column: float,
    row: float,
) -> None:
    # 0.3125 mm pixels already match the model, so 16x32 slices pass unresized.
    volume = (np.arange(3 * 16 * 32).reshape(3, 16, 32) % 7 * 2).astype(np.uint16)
    affines = np.zeros((3, 3, 3))
    affines[:, 1, 0], affines[:, 2, 1] = column, row
    affines[:, 0, 2] = 4.0 * np.arange(3)
    # AnatomySampler scores one slice per call and restores every output channel.
    logits = torch.full((1, 4, 16, 32), -math.inf)
    logits[..., 0, :] = logits[..., :, 0] = math.inf
    prediction = _Prediction(logits)
    result, reason = AnatomySampler.Config().make()(
        volume,
        np.full((3, 2), 0.3125),
        affines,
        prediction,
    )
    assert reason == "ok"
    assert result is not None
    flip_x, flip_y = column < 0, row > 0
    normalized = (volume.astype(np.float32) - float(volume.mean())) / float(
        volume.std(),
    )
    expected = normalized[:, :, ::-1] if flip_x else normalized
    expected = expected[:, ::-1] if flip_y else expected
    assert [index for index, _ in result] == [0, 1, 2]
    assert all(
        torch.equal(inputs, torch.from_numpy(expected[index].copy())[None, None])
        for index, inputs in enumerate(prediction.inputs)
    )
    restored = np.zeros((4, 16, 32), dtype=np.float32)
    restored[:, 15 if flip_y else 0], restored[:, :, 31 if flip_x else 0] = 1, 1
    assert all(np.array_equal(probability, restored) for _, probability in result)


def test_anatomy_input_pads_by_replication_and_restores_unpadded_output() -> None:
    # Mean 4 and std 2 normalize these slices to +-(a + b) / 2 at row a, column b.
    first = 4 + np.array([-2, 1, 1])[:, None] + np.arange(-2, 3)
    # AnatomySampler reads column, row and origin vectors in three LPS coordinates.
    affines = np.zeros((4, 3, 3))
    affines[:, 1, 0], affines[:, 2, 1] = 1.0, -1.0
    affines[:, 0, 2] = 4.0 * np.arange(4)
    # Block (i, j) of the cropped 6x10 output holds (i + j) % 5 ones; padding is 0.5.
    ones = torch.tensor(
        [
            [0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    # AnatomySampler scores one slice per call and restores every output channel.
    logits = torch.zeros(1, 4, 8, 12)
    logits[..., :6, :10] = torch.where(ones, math.inf, -math.inf)
    prediction = _Prediction(logits)
    config = AnatomySampler.Config()
    config.pad_multiple = 4
    result, reason = config.make()(
        np.stack([first, 8 - first] * 2).astype(np.uint16),
        np.full((4, 2), 0.625),
        affines,
        prediction,
    )
    assert reason == "ok"
    assert result is not None
    # 0.625 mm pixels double each axis at half-pixel centers, then replicate to 8x12.
    rows = np.array([-1, -0.625, 0.125, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    columns = np.array(
        [-1, -0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625, 0.875, 1, 1, 1],
        dtype=np.float32,
    )
    prepared = torch.from_numpy(rows[:, None] + columns)[None, None]
    assert len(prediction.inputs) == 4
    assert all(
        torch.equal(inputs, expected)
        for inputs, expected in zip(
            prediction.inputs,
            [prepared, -prepared] * 2,
            strict=True,
        )
    )
    # A half-pixel restore to 3x5 averages each 2x2 block of ones.
    restored = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1]]) / 4
    assert [index for index, _ in result] == [0, 1, 2, 3]
    assert all(
        np.array_equal(probability, np.broadcast_to(restored, (4, 3, 5)))
        for _, probability in result
    )


def test_oblique_sagittal_geometry_skips_the_anatomy_model() -> None:
    # AnatomySampler reads column, row and origin vectors in three LPS coordinates.
    affines = np.zeros((4, 3, 3))
    affines[0, 0, 0] = affines[0, 2, 1] = 1.0
    # AnatomySampler scores one slice per call, padded to its 16-pixel minimum side.
    prediction = _Prediction(torch.zeros(1, 4, 16, 16))
    result, reason = AnatomySampler.Config().make()(
        np.zeros((4, 3, 5), dtype=np.uint16),
        np.ones((4, 2)),
        affines,
        prediction,
    )
    assert (result, reason) == (None, "oblique")
    assert prediction.inputs == []


def test_all_seven_physical_slices_and_constant_volume_normalization() -> None:
    volume = np.full((35, 3, 5), 11, dtype=np.uint16)
    spacing = np.full((35, 2), 0.3125)
    # AnatomySampler consumes column, row, origin vectors in three LPS coordinates.
    affines = np.zeros((35, 3, 3), dtype=np.float64)
    affines[:, 1, 0] = 1
    affines[:, 2, 1] = -1
    affines[:, 0, 2] = np.arange(35)
    # AnatomySampler pads each one-slice, four-tissue model output to stride 16.
    prediction = _Prediction(torch.zeros((1, 4, 16, 16)))
    result, reason = AnatomySampler.Config().make()(
        volume,
        spacing,
        affines,
        prediction,
    )
    assert reason == "ok"
    assert result is not None
    assert [index for index, _ in result] == [5, 9, 13, 17, 21, 25, 29]
    assert len(prediction.inputs) == 7
    # AnatomySampler pads each single-channel MRI slice to its stride-16 minimum.
    assert all(
        torch.equal(inputs, torch.zeros((1, 1, 16, 16))) for inputs in prediction.inputs
    )
    assert all(
        np.array_equal(probability, np.full((4, 3, 5), 0.5, dtype=np.float32))
        for _, probability in result
    )


def test_prepared_model_input_reaches_the_requested_device() -> None:
    affines = np.array([[[0.0, 0.0, 13.0], [1.0, 0.0, -2.0], [0.0, -1.0, 7.0]]])
    # 3 rows of 3.125 mm and 5 columns of 0.625 mm resize to 30x16, padded to 32x16.
    with pytest.raises(_AtModelError, match=r"^requested device$"):
        AnatomySampler.Config().make()(
            np.arange(15, dtype=np.uint16).reshape(1, 3, 5),
            np.array([[3.125, 0.625]]),
            affines,
            _DeviceProbe(),
            device="meta",
        )


def test_resize_interpolates_half_pixel_centers_and_clamps_the_edges() -> None:
    image = np.array([[0.0, 4.0], [8.0, 12.0]], dtype=np.float32)
    resized = _resize_input(image, size=(4, 4))
    assert resized.dtype == np.float32
    assert np.array_equal(
        resized,
        [
            [0.0, 1.0, 3.0, 4.0],
            [2.0, 3.0, 5.0, 6.0],
            [6.0, 7.0, 9.0, 10.0],
            [8.0, 9.0, 11.0, 12.0],
        ],
    )
    assert np.array_equal(
        _resize_input(image, size=(2, 8)),
        [
            [0.0, 0.0, 0.5, 1.5, 2.5, 3.5, 4.0, 4.0],
            [8.0, 8.0, 8.5, 9.5, 10.5, 11.5, 12.0, 12.0],
        ],
    )


def test_resize_keeps_seventh_weights_in_float64_until_one_rounding() -> None:
    # Columns sample (3j - 2) / 7 and rows -1/6, 1/2, 7/6, so every output is a
    # whole or half value; float32 weights put 105 / 7 one ULP above 15. The
    # transposed input moves the sevenths onto the rows.
    image = np.array([[0.0, 105.0, 35.0], [70.0, 0.0, 140.0]], dtype=np.float32)
    expected = np.array(
        [
            [0.0, 15.0, 60.0, 105.0, 75.0, 45.0, 35.0],
            [35.0, 37.5, 45.0, 52.5, 67.5, 82.5, 87.5],
            [70.0, 60.0, 30.0, 0.0, 60.0, 120.0, 140.0],
        ],
    )
    assert np.array_equal(_resize_input(image, size=(3, 7)), expected)
    assert np.array_equal(_resize_input(image.T, size=(7, 3)), expected.T)


def test_resize_subtracts_pixels_in_float64_before_one_rounding() -> None:
    # 1 + 2**-23 and 2**-24 - 1 average to 3 * 2**-25, but their float32
    # difference rounds to -2. Columns 4 -> 2 and rows 2 -> 3 take midpoints.
    high, low = 1 + 2**-23, 2**-24 - 1
    image = np.array([[high, low, high, high], [0.0, 0.0, low, low]], dtype=np.float32)
    assert np.array_equal(
        _resize_input(image, size=(3, 2)),
        [[3 * 2**-25, high], [3 * 2**-26, 3 * 2**-25], [0.0, low]],
    )


class _AtModelError(RuntimeError):
    """Stop at the numerical model boundary after checking device placement."""


class _DeviceProbe:
    def __call__(self, inputs: Tensor) -> Tensor:
        assert inputs.device.type == "meta"
        assert inputs.shape == (1, 1, 32, 16)
        raise _AtModelError("requested device")


class _Prediction:
    def __init__(self, logits: Tensor) -> None:
        self.logits = logits
        self.inputs: list[Tensor] = []

    def __call__(self, inputs: Tensor) -> Tensor:
        self.inputs.append(inputs.detach().clone())
        return self.logits


def _sagittal_payloads() -> list[bytes]:
    """Return four 3x5 sagittal files named 0-3, enumerated out of physical order."""
    image = np.arange(15, dtype=np.uint16).reshape(3, 5)
    return [
        dicom_file(
            image + index,
            position=(x, 0.0, 0.0),
            spacing=(0.5, 0.25),
            instance=index + 1,
            laterality="R",
        )
        for index, x in enumerate((3.0, 0.0, 2.0, 1.0))
    ]


def _sagittal_frames() -> list[NamedFrame]:
    return [
        NamedFrame(name=f"{index}.dcm", frame=read_dicom(payload))
        for index, payload in enumerate(_sagittal_payloads())
    ]


def test_series_preparation_orders_canonicalizes_and_quantizes_the_acquisition() -> (
    None
):
    order = order_frames(_sagittal_frames(), "Sagittal")
    names = [1, 3, 2, 0]
    assert [item.name for item in order.frames] == [f"{name}.dcm" for name in names]
    assert order.positions.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert (order.sides, order.median_x) == (("R",), 1.5)
    result = prepare_series(order, series="1", side="R", fat=1, fluid=0)
    native = result.native
    # Canonical sagittal columns run anterior, reversing stored ones; 65535/17 = 3855.
    image = np.arange(15, dtype=np.uint16).reshape(3, 5)[:, ::-1]
    assert np.array_equal(native.pixels, 3_855 * np.stack([image + n for n in names]))
    assert native.positions.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert native.spacing.tolist() == [[0.5, 0.25]] * 4
    assert (native.slot, native.side, native.fat, native.fluid) == (1, "R", 1, 0)
    assert np.array_equal(
        result.affines,
        [[[0.0, 0.0, x], [-0.25, 0.0, 1.0], [0.0, -0.5, 0.0]] for x in range(4)],
    )
    quality = result.quality
    assert (quality.aligned, quality.drift_mm, quality.axis_difference) == (
        True,
        0.0,
        0.0,
    )
    assert quality.normal.tolist() == [1.0, 0.0, 0.0]
    assert quality.depth.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert result.intensity[:3] == (0.0, 17.0, 17 / 131_070)
    # The window samples rows 0 and columns 0 and 4: levels 0-7 times 3855.
    low, high = result.intensity[3:]
    assert 0 < low < 3_855 < 6 * 3_855 < high < 7 * 3_855
    assert native.window == (low, high)
    assert result.files == [f"{name}.dcm" for name in names]
    assert result.errors == []


@pytest.mark.parametrize(
    ("tag", "value", "message"),
    [
        (0x00200013, b"", r"^Every frame requires an integer acquisition instance\.$"),
        (0x00280010, b"\x04\0", r"^Pixel matrix changes within the acquisition\.$"),
        (
            0x00280030,
            b"0.4\\0.7",
            r"^Pixel spacing is invalid or changes within the acquisition\.$",
        ),
        (
            0x00280030,
            b"0\\0.7",
            r"^Pixel spacing is invalid or changes within the acquisition\.$",
        ),
        (
            0x00280004,
            b"MONOCHROME1",
            r"^Photometric interpretation changes within the acquisition\.$",
        ),
        (
            0x00200037,
            b"0\\0\\0\\0\\0\\-1",
            r"^DICOM orientation contains a zero or nonfinite axis\.$",
        ),
        (
            0x00200037,
            b"0\\-1\\0\\0\\1\\-1",
            r"^DICOM orientation axes are not orthogonal\.$",
        ),
    ],
)
def test_rejects_incomplete_or_changing_headers(
    tag: int,
    value: bytes,
    message: str,
) -> None:
    frames = _sagittal_frames()
    frames[0].frame.values[tag] = value
    with pytest.raises(ValueError, match=message):
        order_frames(frames, "Sagittal")


def test_order_guards_normals_axes_projection_and_plane() -> None:
    frames = _sagittal_frames()
    with pytest.raises(
        ValueError,
        match=r"^The acquisition contains no DICOM frames\.$",
    ):
        order_frames([], "Sagittal")
    with pytest.raises(
        ValueError,
        match=r"^Declared anatomical plane disagrees with its slice normal\.$",
    ):
        order_frames(frames, "Coronal")
    original = frames[1].frame.values[0x00200037]
    frames[1].frame.values[0x00200037] = b"0\\-1\\0\\0.2\\0\\-0.98"
    with pytest.raises(
        ValueError,
        match=r"^Slice normals vary within the acquisition\.$",
    ):
        order_frames(frames, "Sagittal")
    frames[1].frame.values[0x00200037] = b"0\\-0.98\\0.2\\0\\-0.2\\-0.98"
    with pytest.raises(
        ValueError,
        match=r"^In-plane orientation axes vary within the acquisition\.$",
    ):
        order_frames(frames, "Sagittal")
    frames[1].frame.values[0x00200037] = original
    frames[1].frame.values[0x00200032] = frames[0].frame.values[0x00200032]
    with pytest.raises(
        ValueError,
        match=r"^Physical slice projections are duplicate or non-increasing\.$",
    ):
        order_frames(frames, "Sagittal")


def test_axis_normalization_accepts_the_exact_minimum_nonzero_length() -> None:
    frames = _sagittal_frames()
    for item in frames:
        item.frame.values[0x00200037] = b"0\\-1e-8\\0\\0\\0\\-2"
    assert order_frames(frames, "Sagittal").positions.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_per_slice_decode_failures_and_matrix_changes_preserve_the_other_slices() -> (
    None
):
    order = order_frames(_sagittal_frames(), "Sagittal")
    order.frames[0].frame = replace(order.frames[0].frame, pixel_data=b"")
    order.frames[1].frame.values[0x00280010] = b"\x02\0"
    # The first successfully decoded frame establishes the retained matrix.
    result = prepare_series(order, series="series", side=None, fat=0, fluid=-1)
    assert len(result.files) == 1
    assert result.files == [order.frames[1].name]
    assert len(result.errors) == 3
    assert result.native.pixels.shape == (1, 2, 5)
    assert result.native.slot == 2
    for item in order.frames:
        item.frame = replace(item.frame, pixel_data=b"")
    with pytest.raises(
        ValueError,
        match=r"^The acquisition contains no usable native slices\.$",
    ):
        prepare_series(order, series="series", side=None, fat=0, fluid=-1)


def test_one_original_integer_sample_changes_the_complete_quantized_volume() -> None:
    payloads = _sagittal_payloads()
    reference = prepare_series(
        order_frames(_sagittal_frames(), "Sagittal"),
        series="series",
        side="R",
        fat=1,
        fluid=1,
    )
    changed = bytearray(payloads[0])
    changed[-2] += 1
    frames = _sagittal_frames()
    frames[0] = NamedFrame(name="0.dcm", frame=read_dicom(bytes(changed)))
    result = prepare_series(
        order_frames(frames, "Sagittal"),
        series="series",
        side="R",
        fat=1,
        fluid=1,
    )
    assert not np.array_equal(result.native.pixels, reference.native.pixels)


def _volume() -> NDArray[np.uint16]:
    # SideCheck's stride-4 window subsample reads four of these 5x6 pixels.
    return np.arange(11 * 5 * 6, dtype=np.uint16).reshape(11, 5, 6)


def _blocky(blocks: NDArray[np.int64]) -> NDArray[np.uint16]:
    """Upsample 4x4 levels to 8x12, varying mid-range blocks around their means."""
    middle = ((blocks >= 20) & (blocks <= 80)).astype(np.int64)
    pattern = np.kron(middle, [[10, -10, 0], [-10, 10, 0]])
    return (np.kron(blocks, np.ones((2, 3), dtype=np.int64)) + pattern).astype(
        np.uint16,
    )


def test_side_inputs_window_each_distinct_middle_slice_and_area_resize_it() -> None:
    # Each slice's six stride-4 samples include two 0s and two 100s: a 0-100 window.
    blocks = np.array(
        [[0, 20, 100, 40], [60, 150, 0, 80], [0, 40, 100, 20], [100, 80, 0, 150]],
    )
    volume = np.full((11, 8, 12), 7, dtype=np.uint16)
    order = [0, 3, 2, 1]
    for index, slice_blocks in zip(
        (4, 5, 6),
        (blocks, blocks.T, blocks[:, order]),
        strict=True,
    ):
        volume[index] = _blocky(slice_blocks)
    check = SideCheck.Config(image_size=4).make()
    images = check.inputs(volume)
    # Area averaging restores each 2x3 block's level; 150 clips to the window top.
    expected = np.array(
        [[0, 51, 255, 102], [153, 255, 0, 204], [0, 102, 255, 51], [255, 204, 0, 255]],
        dtype=np.uint8,
    )
    assert images.dtype == np.uint8
    assert np.array_equal(images, np.stack([expected, expected.T, expected[:, order]]))
    # A constant slice keeps the 1e-6 span floor and renders black.
    assert not check.inputs(np.full((3, 8, 12), 7, dtype=np.uint16)).any()
    empty = check.inputs(volume[:0])
    assert (empty.shape, empty.dtype) == ((0, 4, 4), np.uint8)


def test_side_window_reads_the_1st_and_99th_percentiles_of_the_subsample() -> None:
    levels = np.array([[0, 100, 300], [600, 1000, 1500], [2100, 2800, 3600]])
    volume = np.kron(levels, np.ones((4, 3), dtype=np.int64)).astype(np.uint16)
    images = SideCheck.Config(image_size=3).make().inputs(volume[None])
    # The stride-4 subsample holds the nine levels, whose 1st and 99th percentiles
    # are 8 and 3536; area averaging restores each 4x3 level block before rounding.
    assert images.tolist() == [[[0, 7, 21], [43, 72, 108], [151, 202, 255]]]


def test_area_resizes_weight_partial_pixels_at_a_three_to_two_scale() -> None:
    # Area weights 2/3 and 1/3 read a ramp at 1/3 and 5/3; bilinear reads 1/4 and 7/4.
    # Halving cannot tell the two apart.
    ramp = 27 * np.arange(3)[:, None] + 9 * np.arange(3)
    rendered, extent = _render_slice(
        (257 * ramp).astype(np.uint16),
        (1.0, 1.0),
        2,
        crop_mm=None,
        center=(0.5, 0.5),
        window=(0.0, 65_535.0),
        floor=0.0,
        knee=1.0,
    )
    assert (rendered.tolist(), extent) == ([[12, 24], [48, 60]], 3.0)
    # Over the 96 + 1e-6 span, 12, 24, 48 and 60 truncate to 31, 63, 127 and 159.
    raptor = raptor_slice(
        ramp.astype(np.float32),
        spacing=1.0,
        window=np.array([0.0, 96.0]),
        image_size=2,
        crop_mm=3.0,
    )
    assert raptor.tolist() == [[31, 63], [127, 159]]
    # Corners set a 0-100 window; 72 weighs 1/9 everywhere, each right 100 4/9.
    volume = np.array([[[0, 0, 100], [0, 72, 0], [0, 0, 100]]], dtype=np.uint16)
    side = SideCheck.Config(image_size=2, percentile_stride=2).make()
    assert side.inputs(volume).tolist() == [[[20, 134], [20, 134]]]


class _SideModel(nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.logits = torch.tensor(logits).view(-1, 1)
        self.inputs: list[Tensor] = []

    @override
    def forward(self, images: Tensor) -> Tensor:
        assert images.shape == (len(self.logits), 1, 4, 4)
        assert torch.is_inference_mode_enabled()
        self.inputs.append(images.clone())
        return self.logits.clone()


def _rerender(side: str, *, corrections: list[str]) -> set[str | None]:
    corrections.append(side)
    return {side}


@dataclass(frozen=True, kw_only=True, slots=True)
class _SideCase:
    """One study's side evidence and the note the check must report."""

    current: str | None
    status: str = "unknown_no_explicit_tags"
    x: float | None = None
    max_x: float = math.inf
    planes: tuple[str, ...] = STUDY_PLANES
    logits: list[float] | None = None
    score: float | None = None
    flipped: bool = False
    new_side: str | None = None
    contradicted: bool = False


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _SideCase(current="R", status="explicit_study_tags", x=-22.0),
            id="position-agrees",
        ),
        pytest.param(
            _SideCase(current="R", status="explicit_study_tags", x=19.9),
            id="position-central",
        ),
        pytest.param(
            _SideCase(
                current="R",
                status="explicit_study_tags",
                x=20.0,
                logits=[-3.0] * 6,
                score=-3.0,
                flipped=True,
                new_side="L",
                contradicted=True,
            ),
            id="right-tag-contradicted",
        ),
        pytest.param(
            _SideCase(
                current="L",
                status="explicit_study_tags",
                x=-20.0,
                logits=[-3.0] * 6,
                score=-3.0,
                flipped=True,
                new_side="R",
                contradicted=True,
            ),
            id="left-tag-contradicted",
        ),
        pytest.param(
            _SideCase(
                current="L",
                x=5.0,
                logits=[-3.0] * 6,
                score=-3.0,
                flipped=True,
                new_side="R",
            ),
            id="unknown-flips",
        ),
        pytest.param(
            _SideCase(
                current=None,
                status="conflicting_study_tags",
                logits=[-3.0] * 6,
                score=-3.0,
                flipped=True,
                new_side="R",
            ),
            id="conflicting-without-position",
        ),
        pytest.param(
            _SideCase(
                current=None,
                planes=STUDY_PLANES[1:],
                logits=[-3.0] * 6,
                score=-3.0,
                flipped=True,
                new_side="R",
            ),
            id="coronal-and-axial",
        ),
        pytest.param(
            _SideCase(current="L", x=20.0, max_x=20.0),
            id="far-position-bypass",
        ),
        pytest.param(
            # Half-precision logits with a float32 mean just below -2 that rounds to -2.
            _SideCase(
                current="L",
                x=19.999,
                max_x=20.0,
                logits=[-2.001953125, -1.9990234375] * 2 + [-2.0, -2.0],
                score=-2.0,
                flipped=True,
                new_side="R",
            ),
            id="mean-below-threshold",
        ),
        pytest.param(
            _SideCase(current="L", x=-10.0, logits=[-2.0] * 6, score=-2.0),
            id="mean-at-threshold",
        ),
        pytest.param(
            _SideCase(current=None, planes=STUDY_PLANES[:1]),
            id="sagittal-only",
        ),
    ],
)
def test_side_check_respects_explicit_evidence_and_flips_low_scores(
    case: _SideCase,
) -> None:
    series = [
        NativeSeries(
            series=str(index),
            pixels=_volume(),
            positions=np.array([] if case.x is None else [case.x]),
            spacing=np.ones((11, 2)),
            plane=plane,
            slot=1,
            side=case.current,
            fluid=0,
            fat=0,
            window=(0.0, 65_535.0),
        )
        for index, plane in enumerate(case.planes)
    ]
    model = _SideModel(case.logits or [])
    corrections: list[str] = []
    check = SideCheck.Config(image_size=4, max_x=case.max_x).make()
    note = check(
        series,
        {case.status},
        model,
        device="cpu",
        rerender=partial(_rerender, corrections=corrections),
    )
    assert note == SideNote(
        side=case.current,
        status=case.status,
        x=None if case.x is None else round(case.x, 1),
        score=case.score,
        flipped=case.flipped,
        new_side=case.new_side,
        tag_contradicted_by_position=case.contradicted,
    )
    assert corrections == ([] if case.new_side is None else [case.new_side])
    assert len(model.inputs) == int(case.logits is not None)
    if case.logits is not None:
        batch = np.concatenate(
            [check.inputs(item.pixels) for item in series if item.plane != "Sagittal"],
        )
        assert torch.equal(
            model.inputs[0],
            (torch.from_numpy(batch).float()[:, None] / 255 - 0.45) / 0.25,
        )


def test_side_check_is_disabled_without_a_model_and_skips_empty_acquisitions() -> None:
    series = [
        NativeSeries(
            series=plane,
            pixels=_volume()[:0],
            positions=np.array([10.0]),
            spacing=np.ones((11, 2)),
            plane=plane,
            slot=1,
            side="L",
            fluid=0,
            fat=0,
            window=(0.0, 65_535.0),
        )
        for plane in STUDY_PLANES
    ]
    check = SideCheck.Config(image_size=4).make()
    statuses = {"unknown_no_explicit_tags"}
    assert check(series, statuses, None, device="cpu", rerender=_never) is None
    model = _SideModel([])
    note = check(series, statuses, model, device="cpu", rerender=_never)
    assert note == SideNote(side="L", status="unknown_no_explicit_tags", x=10.0)
    assert model.inputs == []


def test_side_position_is_the_median_of_each_sagittal_series_median() -> None:
    series = [
        NativeSeries(
            series=str(index),
            pixels=_volume(),
            positions=np.array(positions),
            spacing=np.ones((11, 2)),
            plane="Sagittal",
            slot=1,
            side="R",
            fluid=0,
            fat=0,
            window=(0.0, 65_535.0),
        )
        for index, positions in enumerate(([-22.0], [-60.0, -21.0, 50.0], [30.0]))
    ]
    note = SideCheck.Config(image_size=4).make()(
        series,
        {"explicit_study_tags"},
        _SideModel([]),
        device="cpu",
        rerender=_never,
    )
    # Series medians -22, -21 and 30 mm place the study at -21 mm, agreeing with R.
    assert note == SideNote(side="R", status="explicit_study_tags", x=-21.0)


def _never(side: str) -> set[str | None]:
    raise AssertionError(f"Unexpected rebuild for side {side}.")


def test_failed_rebuild_is_rejected_before_reporting_a_flip() -> None:
    acquisition = NativeSeries(
        series="coronal",
        pixels=_volume(),
        positions=np.array([0.0]),
        spacing=np.ones((11, 2)),
        plane="Coronal",
        slot=3,
        side=None,
        fluid=1,
        fat=1,
        window=(0, 65_535),
    )

    def model(images: Tensor) -> Tensor:
        return images.new_full((len(images), 1), -3)

    def failed(side: str) -> set[str | None]:
        del side
        return set()

    check = SideCheck.Config(image_size=4).make()
    with pytest.raises(
        ValueError,
        match=r"^Forced study side was not applied to every rebuilt acquisition\.$",
    ):
        check(
            [acquisition],
            {"unknown_no_explicit_tags"},
            model,
            device="cpu",
            rerender=failed,
        )
    note = check([], set(), model, device="cpu", rerender=failed)
    assert note is not None
    assert note.side is None
    assert note.status == ""
    assert note.x is None
    assert note.score is None


class _Calls:
    def __init__(self, model: Callable[[Tensor], Tensor]) -> None:
        self.model = model
        self.inputs: list[Tensor] = []

    def __call__(self, images: Tensor) -> Tensor:
        self.inputs.append(images.clone())
        return self.model(images)


def _acquisitions() -> list[Acquisition]:
    return [
        Acquisition(
            series=str(index + 1),
            plane=plane,
            fat=(1, 0, -1)[index],
            fluid=(1, 1, 0)[index],
            frames=[
                NamedFrame(name=f"{number}.dcm", frame=read_dicom(payload))
                for number, payload in files
            ],
        )
        for index, (plane, files) in enumerate(
            zip(STUDY_PLANES, study_series(), strict=True),
        )
    ]


def _low_side(images: Tensor) -> Tensor:
    return images.new_full((len(images), 1), -3.0)


def _zero_anatomy(images: Tensor) -> Tensor:
    return images.new_zeros((len(images), 4, *images.shape[-2:]))


def _zero_side(images: Tensor) -> Tensor:
    return images.new_zeros((len(images), 1))


def test_study_preparation_flips_a_contradicted_side_and_assembles_every_window() -> (
    None
):
    anatomy, side = _Calls(_zero_anatomy), _Calls(_low_side)
    config = StudyPreparation.Config()
    config.render.image_size = config.side.image_size = 4
    actual = config.make()(_acquisitions(), anatomy=anatomy, side=side)
    assert actual.errors == []
    # The sagittal series sits at x 33.1 mm, contradicting its right tags.
    assert actual.side == SideNote(
        side="R",
        status="explicit_study_tags",
        x=33.1,
        score=-3.0,
        flipped=True,
        new_side="L",
        tag_contradicted_by_position=True,
    )
    assert {item.native.side for item in actual.acquisitions} == {"L"}
    # Two middle slices per coronal and axial series; both sagittal slices.
    assert [images.shape for images in side.inputs] == [(4, 1, 4, 4)]
    assert [images.shape for images in anatomy.inputs] == [(1, 1, 16, 16)] * 2
    assert actual.anchors.method == "slab_prior"
    assert actual.anchors.reasons == ["mass<250(0)"]
    rows = torch.from_numpy(evaluation_rows(actual.rendered, stride=1, slab_mm=3.5))
    pixels, metadata = actual.rendered.pixels, actual.rendered.metadata
    assert torch.equal(actual.images, pixels[rows[:, 2:]])
    assert torch.equal(
        actual.metadata,
        metadata[rows[:, 1]][
            :,
            [int(column) for column in (*MODEL_COLUMNS, *EXTENT_COLUMNS)],
        ],
    )
    fluid = actual.fluid
    assert (fluid.status, fluid.series, fluid.count, fluid.slices) == ("ok", "1", 1, 2)


def test_only_in_field_anchors_reach_the_rendered_series() -> None:
    config = StudyPreparation.Config()
    config.render.image_size = config.side.image_size = 4
    prepare = config.make()
    # Alone, the sagittal series' slab prior is its own middle-slice center.
    alone = prepare(_acquisitions()[:1], anatomy=_zero_anatomy, side=_zero_side)
    (native,) = [item.native for item in alone.acquisitions]
    assert (native.joint, native.patella) == ((0.5, 0.5), None)
    # Beside the coronal and axial stacks, the prior projects outside every series.
    study = prepare(_acquisitions(), anatomy=_zero_anatomy, side=_zero_side)
    assert not any(anchor.in_field for anchor in study.anchors.series.values())
    assert [item.native.joint for item in study.acquisitions] == [None] * 3


def test_empty_physical_study_fails_before_model_inference() -> None:
    model = _Calls(_low_side)
    with pytest.raises(ValueError, match=r"^The study contains no usable windows\.$"):
        StudyPreparation.Config().make()([], anatomy=model, side=model)
    assert model.inputs == []
    assert _PhysicalStudy().statuses() == set()


@pytest.mark.parametrize("mode", ["unknown", "conflicting"])
def test_anatomical_tag_trust_is_resolved_across_retained_series(mode: str) -> None:
    acquisitions = _acquisitions()
    for index, acquisition in enumerate(acquisitions):
        acquisition.frames = [
            replace(
                item,
                frame=replace(
                    item.frame,
                    values={
                        **item.frame.values,
                        0x00200060: b""
                        if mode == "unknown"
                        else b"LR"[index % 2 : index % 2 + 1],
                        0x00200062: b"",
                    },
                ),
            )
            for item in acquisition.frames
        ]
    config = StudyPreparation.Config()
    config.render.image_size = config.side.image_size = 4
    actual = config.make()(acquisitions, anatomy=_zero_anatomy, side=_zero_side)
    assert actual.side is not None
    assert actual.side.status == (
        "conflicting_study_tags"
        if mode == "conflicting"
        else "unknown_no_explicit_tags"
    )
    assert actual.side.side == (None if mode == "conflicting" else "L")
    assert not actual.side.flipped


def test_bad_acquisitions_preserve_every_other_series_and_its_side_correction() -> None:
    acquisitions = _acquisitions()
    missing_pixels = replace(
        acquisitions[1],
        series="missing-pixels",
        frames=[
            replace(item, frame=replace(item.frame, pixel_data=b""))
            for item in acquisitions[1].frames
        ],
    )
    invalid = [
        replace(acquisitions[0], series="unsupported", plane="unknown"),
        replace(acquisitions[0], series="unreadable", error="header decode failed"),
        replace(acquisitions[0], series="empty", frames=[]),
        missing_pixels,
    ]
    config = StudyPreparation.Config()
    config.render.image_size = config.side.image_size = 4
    prepare = config.make()
    clean = prepare(acquisitions, anatomy=_zero_anatomy, side=_low_side)
    actual = prepare([*invalid, *acquisitions], anatomy=_zero_anatomy, side=_low_side)
    assert torch.equal(actual.images, clean.images)
    assert actual.side == clean.side
    assert [item.native.series for item in actual.acquisitions] == ["1", "2", "3"]
    assert len(actual.errors) == 3
    assert {series for series, _ in actual.errors} == {
        "unreadable",
        "empty",
        "missing-pixels",
    }


def test_patient_conflicts_and_nonfinite_categories_fail_before_model_calls() -> None:
    acquisitions = _acquisitions()
    acquisitions[0].frames = [
        replace(
            item,
            frame=replace(
                item.frame,
                values={**item.frame.values, 0x00100020: b"other"},
            ),
        )
        for item in acquisitions[0].frames
    ]
    prepare = StudyPreparation.Config().make()
    with pytest.raises(
        ValueError,
        match=r"^Conflicting patients occur within one study\.$",
    ):
        prepare(acquisitions, anatomy=_zero_anatomy, side=None)
    acquisitions = _acquisitions()
    acquisitions[0].fluid = float("inf")
    with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
        prepare(acquisitions, anatomy=_zero_anatomy, side=None)


def test_unknown_categories_and_failed_anatomy_keep_slab_prior() -> None:
    acquisitions = _acquisitions()
    acquisitions[0].fat, acquisitions[0].fluid = None, float("nan")

    def failed(images: Tensor) -> Tensor:
        del images
        raise RuntimeError("Anatomy model failed.")

    config = StudyPreparation.Config()
    config.render.image_size = 4
    actual = config.make()(acquisitions, anatomy=failed, side=None)
    assert actual.side is None
    assert actual.acquisitions[0].native.fat == -1
    assert actual.acquisitions[0].native.fluid == -1
    assert actual.anchors.method == "slab_prior"
    assert actual.anchors.reasons == ["error:RuntimeError('Anatomy model failed.')"]
    assert actual.fluid.count == 0


def test_native_csv_and_every_original_file_are_read_in_enumeration_order(
    tmp_path: Path,
) -> None:
    config = study_inputs(tmp_path)
    dataset = config.make()
    assert dataset.ids == ["study", "empty"]
    assert dataset.sample_ids == ["empty", "study"]
    assert dataset.labels == list(reversed(FINDINGS))
    actual = dataset.load("study")
    assert len(actual.physical) == len(actual.raptor) == 3
    for acquisition, raptor in zip(actual.physical, actual.raptor, strict=True):
        directory = tmp_path / "test_series" / "study" / acquisition.series
        assert [frame.name for frame in acquisition.frames] == [
            path.name for path in directory.glob("*.dcm")
        ]
        assert [frame.frame for frame in acquisition.frames] == list(raptor.frames)
        assert acquisition.error is None
    empty = dataset.load("empty")
    assert empty.physical == empty.raptor == []
    with pytest.raises(KeyError, match="unlisted"):
        dataset.load("unlisted")


def test_unreadable_header_remains_a_raptor_slot_and_rejects_partial_r7_series(
    tmp_path: Path,
) -> None:
    dataset = study_inputs(tmp_path).make()
    directory = tmp_path / "test_series" / "study" / "2"
    (directory / "broken.dcm").write_bytes(b"invalid")
    actual = dataset.load("study")
    assert actual.physical[1].error is not None
    assert len(actual.physical[1].frames) == 4
    assert len(actual.raptor[1].frames) == 5
    assert list(actual.raptor[1].frames).count(None) == 1


@pytest.mark.parametrize("layout", ["test_series", "test_images"])
@pytest.mark.parametrize("suffix", [".dcm", ".DCM", ".dCm"])
def test_accepted_layouts_and_suffixes_preserve_complete_inference_enumeration(
    tmp_path: Path,
    layout: str,
    suffix: str,
) -> None:
    dataset = study_inputs(tmp_path).make()
    root = tmp_path / "test_series"
    if layout != root.name:
        root = root.rename(tmp_path / layout)
    for path in root.rglob("*.dcm"):
        path.rename(path.with_suffix(suffix))
    actual = dataset.load("study")
    assert [len(item.frames) for item in actual.raptor] == [2, 4, 4]
    assert [len(item.frames) for item in actual.physical] == [2, 4, 4]
    for physical, raptor in zip(actual.physical, actual.raptor, strict=True):
        assert [frame.name for frame in physical.frames] == [
            path.name
            for path in (root / "study" / physical.series).glob("*")
            if path.suffix.lower() == ".dcm"
        ]
        assert [frame.frame for frame in physical.frames] == list(raptor.frames)


def test_corrupt_deflate_retains_failure_slots_and_physical_diagnostics(
    tmp_path: Path,
) -> None:
    dataset = study_inputs(tmp_path).make()
    corrupt_deflate(tmp_path)
    actual = dataset.load("study")
    assert [len(item.frames) for item in actual.raptor] == [2, 4, 4]
    assert all(frame is None for item in actual.raptor for frame in item.frames)
    assert all(not item.frames and item.error for item in actual.physical)


@pytest.mark.parametrize("mode", ["duplicate", "different", "columns"])
def test_invalid_csv_identities_and_finding_headers_are_rejected(
    tmp_path: Path,
    mode: str,
) -> None:
    config = study_inputs(tmp_path)
    if mode == "duplicate":
        (tmp_path / "test.csv").write_text("StudyInstanceUID\nstudy\nstudy\n")
    elif mode == "different":
        path = tmp_path / "sample_submission.csv"
        path.write_text(path.read_text().replace("empty", "other"))
    else:
        path = tmp_path / "sample_submission.csv"
        path.write_text(path.read_text().replace("ACL", "unknown"))
    with pytest.raises(ValueError, match=r"identity|finding"):
        config.make()


def test_csv_inference_preserves_boolean_missing_and_object_columns() -> None:
    boolean = _categories(
        [{"fluid": value} for value in ("True", "false", "NA")],
        "fluid",
    )
    assert boolean[:2] == [1.0, 0.0]
    assert isinstance(boolean[2], float)
    assert math.isnan(boolean[2])
    objects = _categories(
        [{"fluid": value} for value in ("invalid", "1", None)],
        "fluid",
    )
    assert objects[:2] == ["invalid", "1"]
    assert isinstance(objects[2], float)
    assert math.isnan(objects[2])


@pytest.mark.parametrize("failure", [FileNotFoundError, EOFError])
def test_incomplete_acquisition_storage_fails_the_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
) -> None:
    dataset = study_inputs(tmp_path).make()

    def unavailable(path: Path) -> bytes:
        del path
        raise failure("Original acquisition is unavailable.")

    monkeypatch.setattr(Path, "read_bytes", unavailable)
    with pytest.raises(failure, match=r"^Original acquisition is unavailable\.$"):
        dataset.load("study")


def test_files_that_are_not_dicom_members_are_not_slices(tmp_path: Path) -> None:
    config = study_inputs(tmp_path)
    before = [frame.name for frame in config.make().load("study").physical[0].frames]
    (tmp_path / "test_series" / "study" / "1" / "notes.txt").write_text("not pixels")
    after = [frame.name for frame in config.make().load("study").physical[0].frames]
    assert after == before
    assert "notes.txt" not in after


def test_dicom_files_list_case_insensitive_members_in_lexical_order(
    tmp_path: Path,
) -> None:
    for name in ("b.dcm", "A.DCM", "notes.txt", "c.dCm"):
        (tmp_path / name).write_bytes(b"")
    assert dicom_files(tmp_path) == [
        tmp_path / name for name in ("A.DCM", "b.dcm", "c.dCm")
    ]
    assert dicom_files(tmp_path / "missing") == []


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
