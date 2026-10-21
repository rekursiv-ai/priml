"""Check physical DICOM preparation against its source captures."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import math
import re

from torch import Tensor, nn
from torch.utils._python_dispatch import TorchDispatchMode

import numpy as np
import pytest
import torch

from priml.baselines.rsna_knee.data import FINDINGS
from priml.baselines.rsna_knee.imaging import (
    Acquisition,
    AnatomicalSeries,
    AnatomySampler,
    AnchorConsensus,
    FluidMeasure,
    FrameGeometry,
    NamedFrame,
    NativeSeries,
    R7Render,
    RaptorBag,
    RaptorPrepare,
    RaptorSeries,
    RaptorWindows,
    RenderedStudy,
    SideCheck,
    SliceBlock,
    StudyPreparation,
    _categories,
    _csv_rows,
    _PhysicalStudy,
    _resize_input,
    anatomical_mirror,
    canonical_frame,
    evaluation_rows,
    order_frames,
    prepare_series,
    quantize_series,
    raptor_slice,
    read_dicom,
    study_side,
    window_centers,
)
from priml.baselines.rsna_knee.support_test import corrupt_deflate, study_inputs
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray
    from torch._ops import OpOverload

    import cv2
else:
    from wrapt import lazy_import

    cv2 = lazy_import("cv2")


_CWD: Final = Path(__file__).resolve().parent


def test_full_centroids_consensus_fallback_and_projections_match_source() -> None:
    expected = read_tensors(_CWD / "testdata" / "anchors.pt")
    for case in range(6):
        series: list[AnatomicalSeries] = []
        for index in range(3):
            quality = int(expected["input/quality"][case, index])
            series.append(
                AnatomicalSeries(
                    series=str(index),
                    plane="Coronal" if index == 2 else "Sagittal",
                    shape=(8, 8),
                    affines=np.asarray(
                        expected["input/affines"][case, index].numpy(),
                        dtype=np.float64,
                    )
                    if quality
                    else None,
                    aligned=quality == 2,
                    fluid=1,
                    probabilities=[
                        (z, np.asarray(probability.numpy(), dtype=np.float32))
                        for z, probability in enumerate(
                            expected["input/probabilities"][index],
                        )
                    ]
                    if index < 2
                    else [],
                ),
            )
        actual = AnchorConsensus.Config().make()(series)
        positions = torch.stack(
            [
                torch.from_numpy(point)
                if point is not None
                else torch.full((3,), float("nan"), dtype=torch.float64)
                for point in (actual.joint, actual.patella, actual.prior)
            ],
        )
        target = expected["output/positions"][case]
        assert torch.equal(positions.isnan(), target.isnan())
        assert torch.equal(positions.nan_to_num(), target.nan_to_num())
        projections: list[list[float]] = []
        for index in range(3):
            entry = actual.series.get(str(index))
            projections.append(
                [float("nan")] * 6
                if entry is None
                else [
                    *entry.joint,
                    float(entry.in_field),
                    entry.distance_mm,
                    *(entry.patella or [float("nan")] * 2),
                ],
            )
        projected = torch.tensor(projections, dtype=torch.float64)
        target = expected["output/projections"][case]
        assert torch.equal(projected.isnan(), target.isnan())
        assert torch.equal(projected.nan_to_num(), target.nan_to_num())
        assert {None: 0, "slab_prior": 1, "skm_tea_joint": 2, "skm_tea_joint_avg2": 3}[
            actual.method
        ] == int(expected["output/methods"][case])
        assert len(actual.reasons) == int(expected["output/reasons"][case])
        assert (
            actual.reasons
            == (
                [],
                ["disagree(29mm)"],
                ["far_from_prior(43mm)", "far_from_prior(43mm)"],
                ["not_aligned", "not_aligned"],
                ["no_geo", "no_geo", "no_geometry"],
                [],
            )[case]
        )
        if case == 0:
            probability = series[0].probabilities[0][1].copy()
            probability.view(np.int32)[3, -1, -1] += 1
            modified = replace(
                series[0],
                probabilities=[(0, probability), *series[0].probabilities[1:]],
            )
            changed = AnchorConsensus.Config().make()([modified, *series[1:]])
            assert actual.joint is not None
            assert changed.joint is not None
            assert not np.array_equal(actual.joint, changed.joint)


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


def test_every_d4_orientation_and_affine_replays_complete_source_arrays() -> None:
    expected = read_tensors(_CWD / "testdata" / "geometry.pt")
    for case, row in enumerate(expected["input/metadata"]):
        geometry = FrameGeometry(
            orientation=tuple(float(row[index]) for index in range(6)),
            origin=tuple(float(row[index]) for index in range(6, 9)),
            spacing=(float(row[9]), float(row[10])),
            slope=float(row[11]),
            intercept=float(row[12]),
            monochrome1=bool(row[13]),
            side="R" if bool(row[14]) else "L",
            plane=("Sagittal", "Coronal", "Axial")[int(row[15])],
        )
        image = expected["input/pixels"][case].reshape(3, 5).numpy()
        actual = canonical_frame(np.asarray(image, dtype=np.float32), geometry)
        assert torch.equal(
            torch.from_numpy(actual.image).flatten(),
            expected["canonical/pixels"][case],
        )
        assert actual.image.shape == tuple(
            int(value) for value in expected["canonical/shape"][case]
        )
        assert torch.equal(
            torch.tensor(actual.spacing, dtype=torch.float64),
            expected["canonical/spacing"][case],
        )
        assert torch.equal(
            torch.from_numpy(actual.affine),
            expected["canonical/affine"][case],
        )
        assert torch.equal(
            torch.from_numpy(actual.basis),
            expected["canonical/basis"][case],
        )
        if case == 0:
            perturbed = expected["input/pixels"][case].clone()
            perturbed.view(torch.int32)[-1].add_(1)
            changed = canonical_frame(
                np.asarray(perturbed.reshape(3, 5).numpy(), dtype=np.float32),
                geometry,
            )
            assert not torch.equal(
                torch.from_numpy(changed.image).flatten(),
                expected["canonical/pixels"][case],
            )


def test_d4_score_ties_choose_the_first_orientation() -> None:
    cosine = 2**-0.5
    geometry = FrameGeometry(orientation=(cosine, cosine, 0.0, -cosine, cosine, 0.0))
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert np.array_equal(canonical_frame(image, geometry).image, image)


def test_full_range_encoding_and_side_decisions_replay_source_numbers() -> None:
    expected = read_tensors(_CWD / "testdata" / "geometry.pt")
    encoded = quantize_series(
        [
            np.asarray(image.numpy(), dtype=np.float32)
            for image in expected["quantize/input"]
        ],
    )
    assert torch.equal(torch.from_numpy(encoded.volume), expected["quantize/volume"])
    assert torch.equal(
        torch.tensor(
            [encoded.minimum, encoded.maximum, encoded.error_bound, *encoded.window],
            dtype=torch.float64,
        ),
        expected["quantize/metadata"],
    )
    for index, (tags, positions) in enumerate(
        (
            (["L"], [-10.0]),
            (["R"], [10.0]),
            (["L", "R"], [10.0]),
            (list[str](), [-10.0]),
            (list[str](), [10.0]),
            (list[str](), list[float]()),
        ),
    ):
        side = study_side(tags, positions)
        assert (0 if side is None else 1 if side == "L" else -1) == int(
            expected["laterality"][index],
        )


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


def test_complete_encoded_files_replay_both_original_pixel_contracts() -> None:
    expected = read_tensors(_CWD / "testdata" / "dicom.pt")
    cases = sum(name.startswith("input/") for name in expected)
    for case in range(cases):
        frame = read_dicom(expected[f"input/{case}"].numpy().tobytes())
        geometry = frame.geometry("Axial", None)
        actual = canonical_frame(frame.pixel_array.astype(np.float32), geometry).image
        assert torch.equal(torch.from_numpy(actual), expected[f"r7/{case}"])
        pixels = frame.raptor_pixels()
        assert pixels.dtype == np.float32
        assert torch.equal(torch.from_numpy(pixels), expected[f"raptor/{case}"])
        instance = frame.instance_number
        assert instance is not None
        assert torch.equal(
            torch.tensor(
                [
                    instance,
                    *frame.shape,
                    *geometry.spacing,
                    *geometry.orientation,
                    *geometry.origin,
                    geometry.slope,
                    geometry.intercept,
                ],
                dtype=torch.float64,
            ),
            expected[f"header/{case}"],
        )
        assert frame.patient_id == "numeric-patient"
        assert frame.laterality == "R"
        assert frame.pixel_array is frame.pixel_array
    payload = bytearray(expected["input/0"].numpy().tobytes())
    payload[-2] += 1
    changed = read_dicom(bytes(payload))
    assert not torch.equal(
        torch.from_numpy(
            canonical_frame(
                changed.pixel_array.astype(np.float32),
                changed.geometry("Axial", None),
            ).image,
        ),
        expected["r7/0"],
    )


def test_modality_pixels_rescale_in_float32_and_invert_monochrome1() -> None:
    expected = read_tensors(_CWD / "testdata" / "dicom.pt")
    frame = read_dicom(expected["input/0"].numpy().tobytes())
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
    expected = read_tensors(_CWD / "testdata" / "dicom.pt")
    payload = expected["input/0"].numpy().tobytes()
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
    expected = read_tensors(_CWD / "testdata" / "dicom.pt")
    frame = read_dicom(expected["input/0"].numpy().tobytes())
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
    jpeg = read_dicom(expected["input/7"].numpy().tobytes())
    with pytest.raises(ValueError, match="payload"):
        _ = replace(jpeg, values={**jpeg.values, 0x00280010: b"\x01\0"}).pixel_array


def test_malformed_element_and_fragment_boundaries() -> None:
    expected = read_tensors(_CWD / "testdata" / "dicom.pt")
    payload = expected["input/0"].numpy().tobytes()
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
    expected = read_tensors(_CWD / "testdata" / "dicom.pt")
    frame = read_dicom(expected["input/6"].numpy().tobytes())
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


def test_all_native_pixels_metadata_blocks_and_window_rows_match_source() -> None:
    expected = read_tensors(_CWD / "testdata" / "render.pt")
    series: list[NativeSeries] = []
    for index in range(7):
        row = expected[f"input/metadata/{index}"]
        joint = (float(row[7]), float(row[8])) if bool(row[11]) else None
        patella = (float(row[9]), float(row[10])) if float(row[9]) >= 0 else None
        series.append(
            NativeSeries(
                series=str(index),
                pixels=np.asarray(
                    expected[f"input/pixels/{index}"].numpy(),
                    dtype=np.uint16,
                ),
                positions=np.asarray(
                    expected[f"input/positions/{index}"].numpy(),
                    dtype=np.float64,
                ),
                spacing=np.asarray(
                    expected[f"input/spacing/{index}"].numpy(),
                    dtype=np.float64,
                ),
                plane=("Sagittal", "Coronal", "Axial")[int(row[1])],
                slot=int(row[0]),
                side="R" if int(row[2]) < 0 else "L",
                fluid=None if int(row[3]) < 0 else int(row[3]),
                fat=None if int(row[4]) < 0 else int(row[4]),
                window=(float(row[5]), float(row[6])),
                joint=joint,
                patella=patella,
            ),
        )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        actual = R7Render.Config(image_size=8).make()(series)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert actual.metadata.dtype == torch.float32
    assert torch.equal(actual.pixels, expected["output/pixels"])
    assert torch.equal(actual.metadata, expected["output/metadata"])
    assert actual.removed == int(expected["output/removed"])
    assert torch.equal(
        torch.tensor(
            [
                [
                    int(block.series),
                    ("Sagittal", "Coronal", "Axial").index(block.plane),
                    block.n_slices,
                    view,
                    start,
                ]
                for block in actual.blocks
                for view, start in block.views
            ],
        ),
        expected["output/blocks"],
    )
    assert torch.equal(
        torch.from_numpy(
            evaluation_rows(actual, stride=3, slab_mm=R7Render.Config().slab_mm),
        ),
        expected["output/window_rows"],
    )


def test_empty_render_preserves_typed_zero_rows_and_blocks() -> None:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        study = R7Render.Config(image_size=3).make()([])
    finally:
        torch.set_default_dtype(previous_dtype)
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
    combined = render([first, second, replace(first, series="duplicate")])
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


def test_every_fluid_measurement_and_missing_path_replays_source() -> None:
    expected = read_tensors(_CWD / "testdata" / "render.pt")
    blocks: list[SliceBlock] = []
    for row in expected["output/blocks"]:
        series, plane, count, view, start = (int(value) for value in row)
        if not blocks or blocks[-1].series != str(series):
            blocks.append(
                SliceBlock(
                    series=str(series),
                    plane=("Sagittal", "Coronal", "Axial")[plane],
                    n_slices=count,
                    views=[],
                ),
            )
        blocks[-1].views.append((view, start))
    for index, excluded in enumerate((None, -1, 4, 0)):
        selected = (
            []
            if excluded == -1
            else [
                SliceBlock(
                    series=block.series,
                    plane=block.plane,
                    n_slices=block.n_slices,
                    views=[
                        (view, start) for view, start in block.views if view != excluded
                    ],
                )
                for block in blocks
            ]
        )
        result = FluidMeasure.Config().make()(
            RenderedStudy(
                pixels=expected["output/pixels"],
                metadata=expected["output/metadata"],
                blocks=selected,
                removed=int(expected["output/removed"]),
            ),
        )
        actual = torch.tensor(
            [
                float("nan") if value is None else float(value)
                for value in (
                    result.effusion,
                    result.baker,
                    result.count,
                    result.median,
                    result.slices,
                    result.spacing,
                )
            ],
            dtype=torch.float64,
        )
        target = expected["output/fluid"][index]
        assert torch.equal(actual.isnan(), target.isnan())
        assert torch.equal(actual.nan_to_num(), target.nan_to_num())
        assert result.status == (
            "ok"
            if index == 0
            else "no fluid-sensitive sagittal series"
            if index == 1
            else "no region view for eff_sag_p99_open"
            if index == 2
            else "error: KeyError(0)"
        )


def test_all96_raw_slice_slots_and_failures_replay_the_source() -> None:
    expected = read_tensors(_CWD / "testdata" / "raptor_prepare.pt")
    prepare = RaptorPrepare.Config(image_size=4).make()
    for case in range(6):
        acquisitions: list[RaptorSeries] = []
        indices = (
            torch.cat([torch.tensor([2]), expected["rows"][expected["rows"] != 2]])
            if case == 5
            else expected["rows"]
        )
        for row in range(len(indices)):
            index = int(indices[row])
            if case == 1 and index == 2:
                continue
            plane, fluid, _ = prepare.config.slots[index]
            if case == 5 and index == 2:
                plane, fluid = "Axial", 0
            count = (
                4 if case == 3 and index == 0 else 1 if case >= 4 and index == 0 else 3
            )
            frames = [
                read_dicom(
                    expected[
                        "input/bad"
                        if case == 2 and index == 0 and frame == 1
                        else f"input/mixed/{frame}"
                        if case == 3 and index == 0
                        else "input/near_constant"
                        if case >= 4 and index == 0
                        else f"input/{index}/{frame}"
                    ]
                    .numpy()
                    .tobytes(),
                )
                for frame in range(count)
            ]
            acquisitions.append(
                RaptorSeries(
                    series=str(index + 1),
                    plane=plane,
                    fluid=fluid,
                    frames=frames,
                ),
            )
        actual = prepare(acquisitions)
        assert torch.equal(torch.from_numpy(actual.pixels), expected[f"pixels/{case}"])
        assert torch.equal(torch.from_numpy(actual.mask), expected[f"mask/{case}"])
    assert torch.equal(
        torch.tensor(
            [
                (("Sagittal", "Coronal", "Axial").index(plane), fluid, count)
                for plane, fluid, count in prepare.config.slots
            ],
        ),
        expected["slots"],
    )


def test_empty_missing_header_and_missing_pixel_inputs_keep_all_slots() -> None:
    expected = read_tensors(_CWD / "testdata" / "raptor_prepare.pt")
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
    valid = read_dicom(expected["input/0/0"].numpy().tobytes())
    invalid_header = read_dicom(expected["input/0/0"].numpy().tobytes())
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


def test_slice_output_dtype_and_small_spacing_guard() -> None:
    expected = read_tensors(_CWD / "testdata" / "raptor_prepare.pt")
    parameters = expected["crop/parameters"]
    pixels = raptor_slice(
        expected["crop/input"].numpy(),
        spacing=float(parameters[0]),
        window=expected["crop/window"].numpy(),
        image_size=int(parameters[1]),
        crop_mm=float(parameters[2]),
    )
    assert pixels.dtype == np.uint8
    assert pixels.shape == (3, 3)
    assert torch.equal(torch.from_numpy(pixels), expected["crop/output"])
    percentile = np.percentile(expected["percentile/input"].numpy(), [2.0, 98.0])
    assert percentile.dtype == np.float64
    assert expected["percentile/output"].dtype == torch.float64
    assert torch.equal(torch.from_numpy(percentile), expected["percentile/output"])


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


def test_source_csv_fluid_conversion_and_nonfinite_failure_classes() -> None:
    expected = read_tensors(_CWD / "testdata" / "raptor_prepare.pt")
    prepare = RaptorPrepare.Config(image_size=3, slots=[("Sagittal", 0, 2)]).make()
    valid = read_dicom(expected["input/0/0"].numpy().tobytes())
    for index in range(len(expected["selection/results"])):
        outcome = int(expected["selection/results"][index])
        fluid = (
            None
            if index == 0
            else "0"
            if index == 1
            else float(expected["selection/values"][index])
        )
        inputs = [
            RaptorSeries(series="1", plane="Sagittal", fluid=1, frames=[]),
            RaptorSeries(series="2", plane="Sagittal", fluid=fluid, frames=[valid]),
        ]
        if outcome < 0:
            with pytest.raises(ValueError if outcome == -1 else OverflowError):
                prepare(inputs)
        else:
            assert outcome == 2
            assert prepare(inputs).mask.tolist() == [1, 1]


def test_all_raptor_windows_replay_source_tensors() -> None:
    expected = read_tensors(_CWD / "testdata" / "raptor_windows.pt")
    # Complete raptor_windows.pt fixes 96 slots and square RaptorWindows images.
    volume = (torch.arange(96 * 5 * 5).reshape(96, 5, 5) % 256).byte().numpy()
    mask = np.ones(96, dtype=np.uint8)
    config = RaptorWindows.Config()
    config.crop_size = 3
    with host_agnostic_numerics():
        windows = config.make()
        assert torch.equal(
            windows(_bag(volume, mask), output_size=3),
            expected["raptor_run"],
        )
        assert torch.equal(
            windows(_bag(volume, mask), output_size=3, reorient=anatomical_mirror),
            expected["mirror"],
        )
        assert torch.equal(
            windows(_bag(volume, np.zeros(96, dtype=np.uint8)), output_size=3),
            expected["raptor_run"][:1].expand(94, -1, -1, -1),
        )
        config.crop_size = 4
        assert torch.equal(
            config.make()(_bag(volume, mask), output_size=2),
            expected["raptor224_run"],
        )
    for index, selected_mask in enumerate(
        (
            mask,
            np.zeros(96, dtype=np.uint8),
            np.concatenate(
                [
                    np.zeros(20, dtype=np.uint8),
                    np.ones(4, dtype=np.uint8),
                    np.zeros(72, dtype=np.uint8),
                ],
            ),
        ),
    ):
        assert torch.equal(
            torch.from_numpy(window_centers(selected_mask, 96, 94)),
            expected[f"centers/{index}"],
        )


def test_raptor_windows_validate_slot_geometry_and_normalization() -> None:
    config = RaptorWindows.Config()
    config.crop_size = 3
    windows = config.make()
    with pytest.raises(
        ValueError,
        match=r"^Raptor slots must sum to the complete volume depth\.$",
    ):
        # Windows need 96 slices of the square 5x5 grid replayed from the source witness.
        windows(
            _bag(np.zeros((95, 5, 5), dtype=np.uint8), np.zeros(95, dtype=np.uint8)),
            output_size=3,
        )
    with pytest.raises(
        ValueError,
        match=r"^Raptor input pixels and slice mask must match their geometry\.$",
    ):
        # Windows need 96 slices of the square 5x5 grid replayed from the source witness.
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
    # Windows need 96 slices of the square 5x5 grid replayed from the source witness.
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
    # Windows need 96 slices of the square 5x5 grid replayed from the source witness.
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


def test_one_float32_scale_ulp_changes_the_frozen_outputs() -> None:
    expected = read_tensors(_CWD / "testdata" / "raptor_windows.pt")
    # Complete raptor_windows.pt fixes 96 slots and square RaptorWindows images.
    volume = (torch.arange(96 * 5 * 5).reshape(96, 5, 5) % 256).byte().numpy()
    config = RaptorWindows.Config()
    config.crop_size = 3
    scale = torch.tensor(config.std[0], dtype=torch.float32)
    scale.view(torch.int32).add_(1)
    config.std = (float(scale), *config.std[1:])
    with host_agnostic_numerics():
        assert not torch.equal(
            config.make()(_bag(volume, np.ones(96, dtype=np.uint8)), output_size=3),
            expected["raptor_run"],
        )


def test_two_slice_volume_centers_on_its_only_interior_candidate() -> None:
    centers = window_centers(np.ones(2, dtype=np.uint8), 2, 3)
    assert centers.tolist() == [1, 1, 1]


def test_all_orientation_inputs_and_four_tissue_outputs_replay_source() -> None:
    expected = read_tensors(_CWD / "testdata" / "segment.pt")
    with host_agnostic_numerics():
        for case in range(5):
            prediction = _Prediction(expected["input/logits"].clone().requires_grad_())
            actual, reason = AnatomySampler.Config().make()(
                np.asarray(expected["input/pixels"].numpy(), dtype=np.uint16),
                np.asarray(expected["input/spacing"].numpy(), dtype=np.float64),
                np.asarray(expected["input/affines"][case].numpy(), dtype=np.float64),
                prediction,
                device=torch.device("cpu"),
            )
            if case == 4:
                assert actual is None
                assert reason == "oblique"
                assert prediction.inputs == []
            else:
                assert reason == "ok"
                assert actual is not None
                assert len(prediction.inputs) == len(actual) == 1
                assert torch.equal(
                    prediction.inputs[0],
                    expected["output/prepared"][case],
                )
                assert torch.equal(
                    torch.tensor([index for index, _ in actual]),
                    expected["output/indices"][case],
                )
                assert torch.equal(
                    torch.stack(
                        [torch.from_numpy(probability) for _, probability in actual],
                    ),
                    expected["output/probabilities"][case],
                )
                if case == 0:
                    perturbed = np.asarray(
                        expected["input/pixels"].numpy(),
                        dtype=np.uint16,
                    ).copy()
                    perturbed[0, 0, 0] += 1
                    changed = _Prediction(expected["input/logits"])
                    AnatomySampler.Config().make()(
                        perturbed,
                        np.asarray(expected["input/spacing"].numpy(), dtype=np.float64),
                        np.asarray(
                            expected["input/affines"][case].numpy(),
                            dtype=np.float64,
                        ),
                        changed,
                    )
                    assert not torch.equal(
                        changed.inputs[0],
                        expected["output/prepared"][case],
                    )


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
    expected = read_tensors(_CWD / "testdata" / "segment.pt")
    with pytest.raises(_AtModelError, match=r"^requested device$"):
        AnatomySampler.Config().make()(
            np.asarray(expected["input/pixels"].numpy(), dtype=np.uint16),
            np.asarray(expected["input/spacing"].numpy(), dtype=np.float64),
            np.asarray(expected["input/affines"][0].numpy(), dtype=np.float64),
            _DeviceProbe(),
            device="meta",
        )


@pytest.mark.parametrize("canonical", [False, True])
def test_resize_preserves_nonbinary_horizontal_and_vertical_coefficients(
    canonical: bool,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "segment.pt")
    image = expected["interpolation/input"]
    if canonical:
        with host_agnostic_numerics():
            actual = _resize_input(image, size=(17, 19))
    else:
        actual = _resize_input(image, size=(17, 19))
    assert torch.equal(
        actual.view(torch.int32),
        expected["interpolation/output"].view(torch.int32),
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


def test_complete_physical_series_replay() -> None:
    expected = read_tensors(_CWD / "testdata" / "series.pt")
    for series, plane in enumerate(("Sagittal", "Coronal", "Axial")):
        frames = [
            NamedFrame(
                name=f"{index}.dcm",
                frame=read_dicom(expected[f"input/{series}/{index}"].numpy().tobytes()),
            )
            for index in expected[f"enumeration/{series}"].tolist()
        ]
        order = order_frames(frames, plane)
        metadata = expected[f"metadata/{series}"]
        result = prepare_series(
            order,
            series=str(series + 1),
            side="R",
            fat=int(metadata[1]),
            fluid=int(metadata[2]),
        )
        native, quality = result.native, result.quality
        assert torch.equal(
            torch.from_numpy(native.pixels),
            expected[f"volume/{series}"],
        )
        assert torch.equal(
            torch.from_numpy(native.positions),
            expected[f"positions/{series}"],
        )
        assert torch.equal(
            torch.from_numpy(native.spacing),
            expected[f"spacing/{series}"],
        )
        assert torch.equal(
            torch.from_numpy(result.affines),
            expected[f"affines/{series}"],
        )
        assert torch.equal(torch.from_numpy(quality.depth), expected[f"depth/{series}"])
        assert torch.equal(
            torch.tensor(
                [
                    float(quality.aligned),
                    quality.drift_mm,
                    quality.axis_difference,
                    *quality.normal,
                ],
                dtype=torch.float64,
            ),
            expected[f"quality/{series}"],
        )
        assert torch.equal(
            torch.tensor(result.intensity, dtype=torch.float64),
            expected[f"intensity/{series}"],
        )
        assert torch.equal(
            torch.tensor([native.slot, native.fat, native.fluid]),
            metadata,
        )
        assert torch.equal(
            torch.tensor([int(Path(name).stem) for name in result.files]),
            expected[f"order/{series}"],
        )
        assert order.sides == ("R",)
        assert native.side == "R"
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
    expected = read_tensors(_CWD / "testdata" / "series.pt")
    frames = [
        NamedFrame(
            name=f"{index}.dcm",
            frame=read_dicom(expected[f"input/0/{index}"].numpy().tobytes()),
        )
        for index in range(4)
    ]
    frames[0].frame.values[tag] = value
    with pytest.raises(ValueError, match=message):
        order_frames(frames, "Sagittal")


def test_order_guards_normals_axes_projection_and_plane() -> None:
    expected = read_tensors(_CWD / "testdata" / "series.pt")
    frames = [
        NamedFrame(
            name=f"{index}.dcm",
            frame=read_dicom(expected[f"input/0/{index}"].numpy().tobytes()),
        )
        for index in range(4)
    ]
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
    expected = read_tensors(_CWD / "testdata" / "series.pt")
    frames = [
        NamedFrame(
            name=f"{index}.dcm",
            frame=read_dicom(expected[f"input/0/{index}"].numpy().tobytes()),
        )
        for index in range(4)
    ]
    for item in frames:
        item.frame.values[0x00200037] = b"0\\-1e-8\\0\\0\\0\\-2"
    order = order_frames(frames, "Sagittal")
    assert torch.equal(torch.from_numpy(order.positions), expected["positions/0"])


def test_per_slice_decode_failures_and_matrix_changes_preserve_the_other_slices() -> (
    None
):
    expected = read_tensors(_CWD / "testdata" / "series.pt")
    frames = [
        NamedFrame(
            name=f"{index}.dcm",
            frame=read_dicom(expected[f"input/0/{index}"].numpy().tobytes()),
        )
        for index in range(4)
    ]
    order = order_frames(frames, "Sagittal")
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
    expected = read_tensors(_CWD / "testdata" / "series.pt")
    frames: list[NamedFrame] = []
    for index in range(4):
        payload = bytearray(expected[f"input/0/{index}"].numpy().tobytes())
        if index == 0:
            payload[-2] += 1
        frames.append(NamedFrame(name=f"{index}.dcm", frame=read_dicom(bytes(payload))))
    result = prepare_series(
        order_frames(frames, "Sagittal"),
        series="series",
        side="R",
        fat=1,
        fluid=1,
    )
    assert not torch.equal(torch.from_numpy(result.native.pixels), expected["volume/0"])


class _Resize(Protocol):
    def __call__(
        self,
        pixels: NDArray[np.float64],
        shape: tuple[int, int],
        *,
        interpolation: int,
    ) -> NDArray[np.float64]: ...


class _ReplayOperators:
    def __init__(self, samples: Tensor, normalized: Tensor) -> None:
        self.samples = samples
        self.normalized = normalized
        self.percentile_calls = self.resize_calls = 0
        self.original_percentile = cast(
            "Callable[[NDArray[np.float32], list[int]], NDArray[np.float64]]",
            np.percentile,
        )
        self.original_resize = cast(_Resize, cv2.resize)

    def percentile(
        self,
        pixels: NDArray[np.float32],
        quantiles: list[int],
    ) -> NDArray[np.float64]:
        expected = self.samples[self.percentile_calls]
        actual = torch.from_numpy(pixels)
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        self.percentile_calls += 1
        return self.original_percentile(pixels, quantiles)

    def resize(
        self,
        pixels: NDArray[np.float64],
        shape: tuple[int, int],
        *,
        interpolation: int,
    ) -> NDArray[np.float64]:
        expected = self.normalized[self.resize_calls]
        actual = torch.from_numpy(pixels)
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        self.resize_calls += 1
        return self.original_resize(pixels, shape, interpolation=interpolation)


class _FrozenSide(nn.Module):
    def __init__(self, inputs: Tensor, outputs: Tensor) -> None:
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.calls = 0

    @override
    def forward(self, images: Tensor) -> Tensor:
        assert images.dtype == self.inputs.dtype
        assert torch.equal(images, self.inputs)
        assert torch.is_inference_mode_enabled()
        self.calls += 1
        return self.outputs.clone()


def _rerender(side: str, *, corrections: list[str]) -> set[str | None]:
    corrections.append(side)
    return {side}


def test_every_source_side_decision_prepared_image_and_rebuild_request() -> None:
    expected = read_tensors(_CWD / "testdata" / "side_check.pt")
    statuses = (
        "unknown_no_explicit_tags",
        "explicit_study_tags",
        "conflicting_study_tags",
    )
    for index in range(2):
        pixels = (
            SideCheck.Config(image_size=4)
            .make()
            .inputs(expected["volumes"][index].numpy())
        )
        assert torch.equal(torch.from_numpy(pixels), expected[f"images/{index}"])
        assert pixels.dtype == np.uint8
    empty = (
        SideCheck.Config(image_size=4).make().inputs(expected["volumes"][0, :0].numpy())
    )
    assert torch.equal(torch.from_numpy(empty), expected["images/empty"])
    assert empty.dtype == np.uint8
    for case in range(sum(name.startswith("case/") for name in expected)):
        fields = expected[f"case/{case}"]
        current = None if int(fields[0]) == 0 else "L" if int(fields[0]) == 1 else "R"
        status = statuses[int(fields[1])]
        x = float(fields[3]) if bool(fields[2]) else None
        planes = (
            ("Sagittal", "Coronal", "Axial")
            if int(fields[5]) == 0
            else ("Coronal", "Axial")
            if int(fields[5]) == 1
            else ("Sagittal",)
        )
        acquisitions = [
            NativeSeries(
                series=str(index),
                pixels=expected["volumes"][index % 2, :0].numpy()
                if bool(fields[6])
                else expected["volumes"][index % 2].numpy(),
                positions=np.array([] if x is None else [x]),
                spacing=np.ones((11, 2)),
                plane=plane,
                slot=1,
                side=current,
                fluid=0,
                fat=0,
                window=(0, 65_535),
            )
            for index, plane in enumerate(planes)
        ]
        model = _FrozenSide(expected[f"prepared/{case}"], expected[f"logits/{case}"])
        corrections: list[str] = []

        # The source disabled the position bypass with a zero distance.
        max_x = float(fields[4]) or math.inf
        check = SideCheck.Config(image_size=4, max_x=max_x).make()
        note = check(
            acquisitions,
            {status},
            model if bool(fields[7]) else None,
            device="cpu",
            rerender=partial(_rerender, corrections=corrections),
        )
        if note is None:
            result = [0.0] * 11
        else:
            result = [
                1.0,
                0.0 if note.side is None else 1.0 if note.side == "L" else -1.0,
                float(statuses.index(note.status)),
                float(note.x is not None),
                note.x or 0.0,
                float(note.score is not None),
                note.score or 0.0,
                float(note.flipped),
                0.0 if note.new_side is None else 1.0 if note.new_side == "L" else -1.0,
                float(note.tag_contradicted_by_position),
                float(1 + len(corrections)),
            ]
        assert torch.equal(
            torch.tensor(result, dtype=torch.float64),
            expected[f"result/{case}"],
        )
        assert model.calls == int(bool(expected[f"logits/{case}"].numel()))


def test_full_intermediate_dtype_and_epsilon_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "side_check.pt")
    check = SideCheck.Config(image_size=4).make()
    for index in range(3):
        operators = _ReplayOperators(
            expected[f"percentile/{index}"],
            expected[f"resize/{index}"],
        )
        with monkeypatch.context() as replay:
            replay.setattr(np, "percentile", operators.percentile)
            replay.setattr(cv2, "resize", operators.resize)
            actual = check.inputs(
                expected["volumes"][index].numpy()
                if index < 2
                else expected["guarded"].numpy(),
            )
        assert actual.dtype == np.uint8
        assert torch.equal(torch.from_numpy(actual), expected[f"images/{index}"])
        assert operators.percentile_calls == len(expected[f"percentile/{index}"])
        assert operators.resize_calls == len(expected[f"resize/{index}"])


def test_failed_rebuild_is_rejected_before_reporting_a_flip() -> None:
    expected = read_tensors(_CWD / "testdata" / "side_check.pt")
    acquisition = NativeSeries(
        series="coronal",
        pixels=expected["volumes"][0].numpy(),
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


class _FrozenModel(nn.Module):
    def __init__(self, inputs: Tensor, outputs: Tensor) -> None:
        super().__init__()
        self.inputs, self.outputs = inputs, outputs
        self.calls = 0

    @override
    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dtype == self.inputs.dtype
        assert torch.equal(inputs, self.inputs[self.calls])
        assert not torch.is_grad_enabled()
        output = self.outputs[self.calls].clone()
        self.calls += 1
        return output


def _acquisitions(expected: dict[str, Tensor]) -> list[Acquisition]:
    return [
        Acquisition(
            series=str(index + 1),
            plane=plane,
            fat=(1, 0, -1)[index],
            fluid=(1, 1, 0)[index],
            frames=[
                NamedFrame(
                    name=f"{number}.dcm",
                    frame=read_dicom(
                        expected[f"input/{index}/{number}"].numpy().tobytes(),
                    ),
                )
                for number in expected[f"enumeration/{index}"].tolist()
            ],
        )
        for index, plane in enumerate(("Sagittal", "Coronal", "Axial"))
    ]


def test_source_complete_side_correction_rebuild_anchors_fluid_and_windows() -> None:
    expected = read_tensors(_CWD / "testdata" / "study.pt")
    side = _FrozenModel(expected["side/input"], expected["side/output"])
    anatomy = _FrozenModel(expected["anatomy/input"], expected["anatomy/output"])
    config = StudyPreparation.Config()
    config.render.image_size = config.side.image_size = 4
    actual = config.make()(_acquisitions(expected), anatomy=anatomy, side=side)
    assert actual.images.dtype == expected["images"].dtype
    assert torch.equal(actual.images, expected["images"])
    assert actual.metadata.dtype == expected["metadata"].dtype
    assert torch.equal(actual.metadata, expected["metadata"])
    assert side.calls == len(expected["side/input"])
    assert anatomy.calls == len(expected["anatomy/input"])
    assert actual.errors == []
    assert {item.native.side for item in actual.acquisitions} == {"L"}
    assert actual.anchors.method == "slab_prior"
    assert actual.anchors.reasons == ["mass<250(0)"]
    assert actual.side is not None
    assert actual.side.side == "R"
    assert actual.side.status == "explicit_study_tags"
    assert actual.side.x == 33.1
    assert actual.side.score == -3.0
    assert actual.side.flipped
    assert actual.side.new_side == "L"
    assert actual.side.tag_contradicted_by_position
    fluid = actual.fluid
    assert fluid.status == "ok"
    assert torch.equal(
        torch.tensor(
            [
                fluid.effusion,
                fluid.baker,
                fluid.count,
                fluid.median,
                fluid.slices,
                fluid.spacing,
                int(fluid.series or "0"),
            ],
            dtype=torch.float64,
        ),
        expected["fluid"],
    )


def test_empty_physical_study_fails_before_model_inference() -> None:
    expected = read_tensors(_CWD / "testdata" / "study.pt")
    model = _FrozenModel(expected["side/input"], expected["side/output"])
    with pytest.raises(ValueError, match=r"^The study contains no usable windows\.$"):
        StudyPreparation.Config().make()([], anatomy=model, side=model)
    assert model.calls == 0
    assert _PhysicalStudy().statuses() == set()


def _zero_anatomy(images: Tensor) -> Tensor:
    return images.new_zeros((len(images), 4, *images.shape[-2:]))


def _zero_side(images: Tensor) -> Tensor:
    return images.new_zeros((len(images), 1))


@pytest.mark.parametrize("mode", ["unknown", "conflicting"])
def test_anatomical_tag_trust_is_resolved_across_retained_series(mode: str) -> None:
    expected = read_tensors(_CWD / "testdata" / "study.pt")
    acquisitions = _acquisitions(expected)
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
    expected = read_tensors(_CWD / "testdata" / "study.pt")
    acquisitions = _acquisitions(expected)
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
    side = _FrozenModel(expected["side/input"], expected["side/output"])
    anatomy = _FrozenModel(expected["anatomy/input"], expected["anatomy/output"])
    actual = config.make()([*invalid, *acquisitions], anatomy=anatomy, side=side)
    assert torch.equal(actual.images, expected["images"])
    assert [item.native.series for item in actual.acquisitions] == ["1", "2", "3"]
    assert len(actual.errors) == 3
    assert {series for series, _ in actual.errors} == {
        "unreadable",
        "empty",
        "missing-pixels",
    }


def test_patient_conflicts_and_nonfinite_categories_fail_before_model_calls() -> None:
    expected = read_tensors(_CWD / "testdata" / "study.pt")
    acquisitions = _acquisitions(expected)
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
    acquisitions = _acquisitions(expected)
    acquisitions[0].fluid = float("inf")
    with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
        prepare(acquisitions, anatomy=_zero_anatomy, side=None)


def test_unknown_categories_and_failed_anatomy_keep_slab_prior() -> None:
    expected = read_tensors(_CWD / "testdata" / "study.pt")
    acquisitions = _acquisitions(expected)
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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
