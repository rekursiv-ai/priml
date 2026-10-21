"""Check well data construction, augmentation and loading."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

import csv
import math
import random

from configgle import InlineConfig
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.worker import WorkerInfo

import numpy as np
import pytest
import torch

from priml.baselines.wellbore_geology.data import (
    FORMATIONS,
    AugmentedFeatures,
    BootstrapTrend,
    BoxcarComponent,
    DerivativeBlock,
    ExpandPrediction,
    FeatureItem,
    FixedNanPattern,
    FoldObservations,
    GammaBias,
    ImageAugmentation,
    ImageFeatures,
    MismatchRange,
    ParticleFeatures,
    PathSimulation,
    ReferenceCalibration,
    ReferenceJitter,
    ReferenceTransform,
    SampledNanPattern,
    WellDataset,
    WellLoader,
    WellRecord,
    WorkerStreams,
    _boxcar_noise,
    _finish,
    _image_at,
    _quadratic_solve,
    _smooth,
    collate_features,
    discover_wells,
    initialize_worker,
    kahan_add,
    load_raw,
    load_split,
    normalize_probability,
    prepare_well,
    probability_image,
    read_centroids,
    read_numeric_csv,
    trajectory,
)
from priml.baselines.wellbore_geology.folds import GeographicPrior, RelativeSurface
from priml.baselines.wellbore_geology.loss import PosteriorLoss
from priml.baselines.wellbore_geology.metrics import pooled_rmse
from priml.baselines.wellbore_geology.particles import (
    ParticleCache,
    ParticlePrior,
    particle_well,
    prepare_particles,
    read_particle_cache,
    write_particle_cache,
)
from priml.baselines.wellbore_geology.scripts.prepare_data import (
    geographic_map,
    stage_data,
)
from priml.baselines.wellbore_geology.testing import (
    numerical_tensors,
    write_study_inputs,
)
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.wellbore_geology.custom_types import Batch


_CWD: Final = Path(__file__).resolve().parent


# Seeds whose union reaches every line and branch the sixteen original seeds did.
@pytest.mark.parametrize("seed", [31, 5, 6, 8, 9, 12, 15])
@pytest.mark.parametrize("mutation", ["none", "numpy_draw", "python_draw"])
def test_complete_original_augmented_inputs_and_rng(seed: int, mutation: str) -> None:
    record = augmentation_record(seed=seed, mutation=mutation)
    filename = "augmentation.pt" if seed == 31 else f"augmentation_{seed:02}.pt"
    differences = mismatches(read_tensors(_CWD / "testdata" / filename), record)
    assert bool(differences) == (mutation != "none"), differences


# Per family, the seeds whose union reaches every line and branch all three did.
@pytest.mark.parametrize(
    ("family", "seed"),
    [
        ("0719_v1", 31),
        ("0719_v1", 0),
        ("0719_v1", 1),
        ("0724_v1", 31),
        ("0724_v1", 1),
        ("0729_v3", 31),
        ("0729_v3", 0),
        ("0729_v3", 1),
        ("0801_v1", 0),
        ("0801_v1", 1),
        ("0801_v2", 0),
        ("0801_v2", 1),
    ],
)
def test_complete_earlier_family_simulation_defaults(family: str, seed: int) -> None:
    record = augmentation_record(seed=seed, family=family)
    expected = read_tensors(
        _CWD / "testdata" / f"augmentation_{family}_{seed:02}.pt",
    )
    assert mismatches(expected, record) == []


def augmentation_record(
    *,
    seed: int,
    mutation: str = "none",
    family: str = "0803_v2",
) -> dict[str, torch.Tensor]:
    """Capture all three complete simulated inputs and both complete used streams.

    Args:
      seed: Initial seed of the consumer-owned Python stream.
      mutation: Extra draw to prove the complete RNG comparison detects drift.
      family: Public family whose complete augmentation settings are replayed.

    Returns:
      record: Every numeric item leaf and both complete used generator states.

    """
    config = AugmentedFeatures.Config()
    paths = config.paths
    assert paths is not None
    config.features.prefix_rows = 32
    config.features.target_rows = 64
    config.features.reference_columns = 3
    if family in {"0719_v1", "0724_v1", "0729_v3"}:
        paths.shift_probability = 0.75
        paths.shift_nan_pattern = FixedNanPattern.Config(shift=False)
        paths.reference_jitter.probability = 0
        paths.reference_transform.gain_std = 0.03
        paths.reference_drift_probability = 0
        if family == "0719_v1":
            paths.gamma_noise_shift_probability = 0.1
        elif family == "0724_v1":
            nan_pattern = paths.shift_nan_pattern = SampledNanPattern.Config()
            nan_pattern.weights = {False: 1.0}
            config.features.channels += ["geo_tvt_diff"]
        else:
            nan_pattern = paths.shift_nan_pattern = SampledNanPattern.Config()
            nan_pattern.weights = {False: 1.0}
            paths.prefix_sample_rows = 256
            config.features.channels += ["pf_particle_density_prob", "pf_tvt_ffbsi"]
            config.features.particle = ParticleFeatures.Config(
                path="pf_tvt_rel_pred_ffbsi",
            )
    elif family == "0801_v2":
        config.features.channels = [*config.features.channels[:-1], "geo_tvt_diff"]
    elif family == "0803_v2":
        config.features.channels = [
            *config.features.channels[:-1],
            "geo_tvt_diff",
            "pf_particle_density_prob",
            "pf_prob_ffbsi",
            "geo_tvt_abs_diff",
            "geo_tvt_rel",
        ]
        config.features.particle = ParticleFeatures.Config(
            path="pf_tvt_rel_pred_ffbsi",
        )
    numpy = np.random.RandomState(29)
    python = InlineConfig(random.Random, seed).make()
    augmenter = config.make()
    query = augmentation_well(prefix=512, suffix=193, well_id="query")
    donor = augmentation_well(prefix=32, suffix=1001, well_id="donor")
    augmenter.prepare([query, donor])
    record: dict[str, torch.Tensor] = {}
    for step in range(3):
        item = augmenter(query, training=True, numpy=numpy, python=python)
        record.update(
            {
                f"{step}/{name}": torch.from_numpy(values)
                for name, values in item.arrays.items()
            },
        )
        record.update(
            {
                f"{step}/aux_targets/{name}": torch.from_numpy(values)
                for name, values in item.auxiliary.items()
            },
        )
        for name, value in item.metadata.items():
            if not isinstance(value, str):
                record[f"{step}/meta/{name}"] = torch.from_numpy(
                    np.asarray(value).copy(),
                )
    if mutation == "numpy_draw":
        numpy.random_sample()
    elif mutation == "python_draw":
        python.getrandbits(32)
    record.update(full_rng_record(numpy=numpy, python=python))
    return record


@pytest.mark.parametrize("condition", ["no_pf", "one_target", "constant", "zero_slope"])
def test_unusable_particle_rotation_preserves_every_plane_and_rng(
    condition: str,
) -> None:
    features = ImageFeatures.Config()
    features.channels = (
        ["gr", "gr_abs_diff"] if condition == "no_pf" else ["pf_tvt", "pf_tvt_ffbsi"]
    )
    features.reference_columns = 3
    config = ImageAugmentation.Config()
    config.rotate_probability = 1
    config.channel_probability = 0
    config.sequence_probability = 0
    target = np.asarray([0, 1, 2, 3], dtype=np.float32)
    if condition == "zero_slope":
        features.target_stats = (0, 1)
        target[:] = [1, 0, 0, 0]
    mask = np.ones(4, dtype=bool)
    if condition == "one_target":
        mask[1:] = False
    if condition == "constant":
        target[:] = 0
    planes = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    item = FeatureItem(
        arrays={
            "unet_static": planes.copy(),
            "target": target,
            "target_mask": mask,
            "suffix_mask": np.ones(4, dtype=bool),
        },
        auxiliary={},
        metadata={},
    )
    numpy = np.random.RandomState(7)
    before = numpy.get_state()
    result = config.make()(
        item,
        features=features,
        numpy=numpy,
        python=InlineConfig(random.Random, 11).make(),
    )
    assert result is item
    np.testing.assert_array_equal(result.arrays["unet_static"], planes)
    if condition == "zero_slope":
        reference = np.random.RandomState(7)
        reference.uniform(0.75, 1.25)
        before = reference.get_state()
    after = numpy.get_state()
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]


@pytest.mark.parametrize(
    ("target", "path", "maximum_shift", "std", "shifted"),
    [
        ((0, 1, 2), (0, 2, 3), 20, 1, (0, 3, 5)),
        ((0, 1, 2), (0, 2, 3), math.inf, 1, (0, 3, 5)),
        ((0, 2, -3), (1, 1, 1), 0.5, 1, (1, 1.5, 0.5)),
        ((0, 1, 2), (1e-6, 2e-6, 0), 20, 1, (0, 1.000002, 0)),
        ((0, 1, 0), (1, 1, 1), 1e-6, 1, (1, 1, 1)),
        (
            (0, 3.4028234663852886e38, -3.4028234663852886e38, 1, 0),
            (1, 1, 1, 1, 1),
            20,
            2,
            (1, 1, 1, 2, 1),
        ),
    ],
)
def test_path_only_particle_rotation_updates_active_rows_and_keeps_zero_rows(
    target: tuple[float, ...],
    path: tuple[float, ...],
    maximum_shift: float,
    std: float,
    shifted: tuple[float, ...],
) -> None:
    features = ImageFeatures.Config()
    features.channels = ["pf_tvt", "pf_tvt_ffbsi"]
    features.target_stats = (0, std)
    features.reference_columns = 4
    config = ImageAugmentation.Config()
    config.rotate_probability = 1
    config.ratio_jitter = 0
    config.maximum_shift = maximum_shift
    config.channel_probability = 0
    config.sequence_probability = 0
    paths = np.broadcast_to(
        np.asarray(path, dtype=np.float32)[None, :, None],
        (2, len(target), 4),
    ).copy()
    item = FeatureItem(
        arrays={
            "unet_static": paths,
            "target": np.asarray(target, dtype=np.float32),
            "target_mask": np.ones(len(target), dtype=bool),
            "suffix_mask": np.ones(len(target), dtype=bool),
        },
        auxiliary={},
        metadata={},
    )
    with np.errstate(over="ignore"):
        result = config.make()(
            item,
            features=features,
            numpy=np.random.RandomState(7),
            python=InlineConfig(random.Random, 11).make(),
        )
    expected = np.broadcast_to(
        np.asarray(shifted, dtype=np.float32)[None, :, None],
        (2, len(target), 4),
    )
    np.testing.assert_array_equal(result.arrays["unet_static"], expected)


@pytest.mark.parametrize("above", [False, True])
def test_fit_denominator_boundary_preserves_float64_decisions_and_rng(
    *,
    above: bool,
) -> None:
    numerator, denominator = (1e-6).as_integer_ratio()
    assert denominator == 2**72
    roots = [0]
    while numerator:
        root = min(math.isqrt(numerator), 2**24)
        roots.append(root)
        numerator -= root * root
    if above:
        roots.append(1)
    target = np.asarray(roots, dtype=np.float32) / np.float32(2**36)
    target64 = target.astype(np.float64)
    expected_denominator = math.nextafter(1e-6, math.inf) if above else 1e-6
    assert np.dot(target64, target64) == expected_denominator
    features = ImageFeatures.Config()
    features.channels = ["pf_tvt", "pf_tvt_ffbsi"]
    features.target_stats = (0, 1)
    features.reference_columns = 3
    config = ImageAugmentation.Config()
    config.rotate_probability = 1
    config.ratio_jitter = 0
    config.channel_probability = config.sequence_probability = 0
    planes = np.ones((2, len(target), 3), dtype=np.float32)
    item = FeatureItem(
        arrays={
            "unet_static": planes.copy(),
            "target": target,
            "target_mask": np.ones(len(target), dtype=bool),
            "suffix_mask": np.ones(len(target), dtype=bool),
        },
        auxiliary={},
        metadata={},
    )
    numpy = np.random.RandomState(7)
    python = InlineConfig(random.Random, 11).make()
    config.make()(item, features=features, numpy=numpy, python=python)
    expected = planes + target[None, :, None] if above else planes
    np.testing.assert_array_equal(item.arrays["unet_static"], expected)
    reference_numpy = np.random.RandomState(7)
    if above:
        reference_numpy.uniform(1, 1)
    reference_python = InlineConfig(random.Random, 11).make()
    for _ in range(5):
        reference_python.random()
    assert (
        mismatches(
            full_rng_record(numpy=reference_numpy, python=reference_python),
            full_rng_record(numpy=numpy, python=python),
        )
        == []
    )


@pytest.mark.parametrize("mutation", [False, True])
def test_complete_original_rotation_inputs_and_rng(*, mutation: bool) -> None:
    record = rotation_record()
    if mutation:
        values = record["8/unet_static"].reshape(-1)
        values[0] = torch.nextafter(values[0], torch.tensor(float("inf")))
    differences = mismatches(read_tensors(_CWD / "testdata/rotation.pt"), record)
    assert bool(differences) == mutation, differences


def rotation_record() -> dict[str, torch.Tensor]:
    """Build nine complete rotation items and retain both final RNG states.

    Returns:
        record: Every numeric item leaf and both complete used generator states.

    """
    channel_sets = (
        ["pf_prob"],
        ["pf_tvt"],
        ["pf_prob", "pf_tvt"],
        ["pf_prob_ffbsi"],
        ["pf_tvt_ffbsi"],
        ["pf_prob_ffbsi", "pf_tvt_ffbsi"],
        ["pf_prob", "pf_tvt", "pf_prob_ffbsi", "pf_tvt_ffbsi"],
        ["pf_particle_density_prob", "pf_prob_ffbsi", "pf_tvt_ffbsi"],
        ["pf_tvt"],
    )
    numpy = np.random.RandomState(29)
    python = InlineConfig(random.Random, 31).make()
    record: dict[str, torch.Tensor] = {}
    for index, channels in enumerate(channel_sets):
        features = ImageFeatures.Config()
        features.prefix_rows = 32
        features.target_rows = 64
        features.reference_columns = 11
        features.channels = channels
        features.particle = ParticleFeatures.Config(
            path="pf_tvt_rel_pred"
            if {"pf_prob", "pf_tvt"}.intersection(channels)
            else "pf_tvt_rel_pred_ffbsi",
        )
        well = augmentation_well(prefix=2, suffix=2, well_id="query")
        probability = np.arange(1, 45, dtype=np.float32).reshape(4, 11)
        probability /= probability.sum(axis=1, keepdims=True)
        assert well.particle is not None
        particle = dict(well.particle)
        for name in ("pf_prob", "pf_prob_ffbsi", "pf_particle_density_prob"):
            particle[name] = probability.copy()
        well = replace(well, particle=particle)
        item = features.make()(well, training=True)
        mean, std = features.target_stats
        item.arrays["target"] = (
            (np.asarray([0, 100, -100, 20], dtype=np.float32) - mean) / std
        ).astype(np.float32)
        item.arrays["target_mask"] = np.ones(4, dtype=bool)
        item.arrays["suffix_mask"] = np.ones(4, dtype=bool)
        image = np.zeros((len(channels), 4, 11), dtype=np.float32)
        probability = np.zeros((4, 11), dtype=np.float32)
        probability[[0, 1, 2], [5, -1, 0]] = 1
        for channel, name in enumerate(channels):
            image[channel] = (
                np.log1p(probability * np.float32(11))
                if "prob" in name
                else np.broadcast_to(
                    np.asarray([0.2, 0, -0.3, 0.4], dtype=np.float32)[:, None],
                    (4, 11),
                )
            )
        item.arrays["unet_static"] = image
        config = ImageAugmentation.Config()
        config.channel_probability = config.sequence_probability = 0
        if index == 8:
            config.maximum_shift = math.inf
        python.seed(31)
        config.make()(item, features=features, numpy=numpy, python=python)
        record.update(
            {
                f"{index}/{name}": torch.from_numpy(values)
                for name, values in item.arrays.items()
            },
        )
        record.update(
            {
                f"{index}/aux_targets/{name}": torch.from_numpy(values)
                for name, values in item.auxiliary.items()
            },
        )
        for name, value in item.metadata.items():
            if not isinstance(value, str):
                record[f"{index}/meta/{name}"] = torch.from_numpy(
                    np.asarray(value).copy(),
                )
    record.update(full_rng_record(numpy=numpy, python=python))
    return record


def full_rng_record(
    *,
    numpy: np.random.RandomState,
    python: random.Random,
) -> dict[str, torch.Tensor]:
    """Retain every numeric leaf of both complete used generator states.

    Args:
      numpy: Consumer-owned NumPy generator after all observed item draws.
      python: Consumer-owned Python generator after all observed item draws.

    Returns:
      record: Complete generator states under canonical numeric leaf names.

    """
    state = numpy.get_state()
    py_state = cast("tuple[int, tuple[int, ...], float | None]", python.getstate())
    assert py_state[2] is None
    return {
        "rng/numpy/keys": torch.from_numpy(state[1].copy()),
        "rng/numpy/position": torch.tensor(state[2]),
        "rng/numpy/has_gaussian": torch.tensor(state[3]),
        "rng/numpy/gaussian": torch.tensor(state[4], dtype=torch.float64),
        "rng/python/version": torch.tensor(py_state[0]),
        "rng/python/keys": torch.tensor(py_state[1], dtype=torch.int64),
    }


def augmentation_well(*, prefix: int, suffix: int, well_id: str) -> WellRecord:
    """Rebuild complete deterministic query/donor wells used by the source mint.

    Args:
      prefix: Number of visible rows preceding the hidden target suffix.
      suffix: Number of rows hidden from the observed depth input.
      well_id: Query or donor identity selecting the deterministic depth slope.

    Returns:
      well: Complete raw observations, paired curve and particle cache arrays.

    """
    row = np.arange(prefix + suffix, dtype=np.float32)
    target = (20 + row / (16 if well_id == "query" else 64)).astype(np.float32)
    visible = target.copy()
    visible[prefix:] = np.nan
    gamma = (75 + row / 64 + (row % 13) ** 2 / 8).astype(np.float32)
    gamma[5::17] = np.nan
    reference = np.arange(9, dtype=np.float32)
    probability = np.arange(1, 13, dtype=np.float32).reshape(4, 3)
    probability /= probability.sum(axis=1, keepdims=True)
    return WellRecord(
        well_id=well_id,
        horizontal={
            "MD": row,
            "X": (row / 2).astype(np.float32),
            "Y": (row / 4).astype(np.float32),
            "Z": (-10 + row / 32).astype(np.float32),
            "GR": gamma,
            "TVT_input": visible,
            "TVT": target,
            "orig_index": row,
            "geo_s_rel_prior_abs": (10 + row / 16).astype(np.float32),
        },
        reference_depth=(-180 + reference * 45).astype(np.float32),
        reference_gamma=(75 + reference**2 * 2).astype(np.float32),
        particle={
            "window_orig_index": np.asarray(
                [15.5, 527.5, 559.5, 703.5],
                dtype=np.float32,
            ),
            "window_tvt_rel": np.asarray([-0.5, 1, 3, 0], dtype=np.float32),
            "window_has_tvt": np.ones(4, dtype=np.bool_),
            "pf_prob": probability,
            "pf_prob_ffbsi": probability[:, ::-1].copy(),
            "pf_particle_density_prob": probability.copy(),
            "pf_tvt_rel_pred": np.asarray([-0.5, 1.25, 2.5, 0], dtype=np.float32),
            "pf_tvt_rel_pred_ffbsi": np.asarray([-0.5, 1.5, 3.5, 0], dtype=np.float32),
        },
    )


def _write_well(directory: Path, *, well_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{well_id}__horizontal_well.csv").write_text(
        "MD,X,Y,Z,GR,TVT_input,TVT,ANCC\n"
        "1,2,4,-1,10,100,100,200\n"
        "2,3,7,-2,,101,101,900\n"
        "3,10,13,-3,20,,102,-500\n",
    )
    (directory / f"{well_id}__typewell.csv").write_text(
        "TVT,GR,Geology\n100,10,ANCC\n101,20,EGFDL\n",
    )


def test_discovery_sorts_and_requires_paired_csvs(tmp_path: Path) -> None:
    _write_well(tmp_path, well_id="bbbbbbbb")
    _write_well(tmp_path, well_id="aaaaaaaa")
    assert discover_wells(tmp_path) == ("aaaaaaaa", "bbbbbbbb")
    (tmp_path / "cccccccc__typewell.csv").write_text("TVT,GR\n1,2\n")
    with pytest.raises(FileNotFoundError, match="cccccccc"):
        discover_wells(tmp_path)


def test_discovery_refuses_empty_or_horizontal_only_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_wells(tmp_path)
    (tmp_path / "aaaaaaaa__horizontal_well.csv").write_text("MD,X,Y\n1,2,3\n")
    with pytest.raises(FileNotFoundError, match="aaaaaaaa"):
        discover_wells(tmp_path)


def test_geographic_map_uses_all_rows_without_geological_labels(tmp_path: Path) -> None:
    _write_well(tmp_path, well_id="aaaaaaaa")
    expected = (
        b"well_id,horizontal_avg_X,horizontal_avg_Y\naaaaaaaa,5.000000,8.000000\n"
    )
    assert geographic_map(tmp_path) == expected
    horizontal = tmp_path / "aaaaaaaa__horizontal_well.csv"
    horizontal.write_text(horizontal.read_text().replace(",200\n", ",-1000000\n"))
    assert geographic_map(tmp_path) == expected


def test_numeric_csv_preserves_complete_rows_and_requested_column_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("Y,unused,X\n1,rock,3.125\n,soil,4.25\n-2,clay,NaN\n")
    result = read_numeric_csv(path, columns=("X", "Y"))
    assert list(result) == ["X", "Y"]
    assert all(values.dtype == np.float64 for values in result.values())
    np.testing.assert_array_equal(result["X"], [3.125, 4.25, np.nan])
    np.testing.assert_array_equal(result["Y"], [1, np.nan, -2])
    with pytest.raises(ValueError, match="missing columns"):
        read_numeric_csv(path, columns=("Z",))
    path.write_text("Y,X\n")
    assert all(
        values.size == 0
        for values in read_numeric_csv(path, columns=("X", "Y")).values()
    )


@pytest.mark.parametrize("mutation", [False, True])
def test_complete_original_csv_precision_and_missing_values(
    tmp_path: Path,
    *,
    mutation: bool,
) -> None:
    values = (
        "0.10000000000000001",
        "1.0000000000000002",
        "123456789.12345679",
        "-123456789.12345679",
        "0.33333333333333331",
        "2458.9999999999995",
        "0.95999999999999996",
        "0.31499999999999995",
        "1.4500000000000002",
        "2.2250738585072014e-308",
        "4.9406564584124654e-324",
        "1.7976931348623157e308",
        "-0.0",
        "1e-9",
        "-3.125e+7",
        "",
        "NaN",
        "NA",
        "null",
        "INF",
    )
    path = tmp_path / "observations.csv"
    path.write_text(
        "X,Y,note\n"
        + "".join(
            f"{first},{second},row\n"
            for first, second in zip(values, reversed(values), strict=True)
        ),
    )
    record = {
        name: torch.from_numpy(values)
        for name, values in read_numeric_csv(path, columns=("X", "Y")).items()
    }
    expected = read_tensors(_CWD / "testdata/numeric_csv.pt")
    if mutation:
        record["X"][0] = torch.nextafter(
            record["X"][0],
            torch.tensor(float("inf"), dtype=torch.float64),
        )
    differences = mismatches(expected, record)
    assert bool(differences) == mutation, differences


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("+.125e+2", 12.5),
        (" -2.5E1 ", -25.0),
        ("1e-1000", 0.0),
        ("-1e-1000", 0.0),
        ("+Infinity", np.inf),
        ("-INF", -np.inf),
    ],
)
def test_numeric_csv_signed_decimal_exponents(
    tmp_path: Path,
    token: str,
    expected: float,
) -> None:
    path = tmp_path / "observations.csv"
    path.write_text(f"X\n{token}\n")
    assert read_numeric_csv(path, columns=("X",))["X"].tolist() == [expected]


@pytest.mark.parametrize("token", ["rock", ".", "1.2.3", "1e309"])
def test_numeric_csv_rejects_invalid_numeric_observations(
    tmp_path: Path,
    token: str,
) -> None:
    path = tmp_path / "observations.csv"
    path.write_text(f"X\n{token}\n")
    with pytest.raises(ValueError, match=r"numeric CSV|float64 range"):
        read_numeric_csv(path, columns=("X",))


@pytest.mark.parametrize("values", ["", "1,,\n2,,\n"])
def test_geographic_map_rejects_empty_or_missing_coordinates(
    tmp_path: Path,
    values: str,
) -> None:
    _write_well(tmp_path, well_id="aaaaaaaa")
    (tmp_path / "aaaaaaaa__horizontal_well.csv").write_text("MD,X,Y\n" + values)
    with pytest.raises(ValueError, match=r"empty|missing"):
        geographic_map(tmp_path)


def test_staging_preserves_csvs_and_omits_train_pngs(tmp_path: Path) -> None:
    source = tmp_path / "release"
    _write_well(source / "train", well_id="aaaaaaaa")
    _write_well(source / "test", well_id="bbbbbbbb")
    sample = source / "sample_submission.csv"
    sample.write_text("id,tvt\nbbbbbbbb_2,0\n")
    (source / "train" / "aaaaaaaa.png").write_bytes(b"unused training image")
    staged = stage_data(source, destination=tmp_path / "prepared")
    assert staged.train_wells == ("aaaaaaaa",)
    assert staged.test_wells == ("bbbbbbbb",)
    assert (
        staged.directory / "sample_submission.csv"
    ).read_bytes() == sample.read_bytes()
    for split, ids in (("train", staged.train_wells), ("test", staged.test_wells)):
        for well_id in ids:
            for suffix in ("horizontal_well", "typewell"):
                name = f"{well_id}__{suffix}.csv"
                assert (staged.directory / split / name).read_bytes() == (
                    source / split / name
                ).read_bytes()
    assert (staged.directory / "train_png_typewell_map.csv").read_bytes() == (
        geographic_map(source / "train")
    )
    assert list(staged.directory.rglob("*.png")) == []
    with pytest.raises(FileExistsError):
        stage_data(source, destination=staged.directory)


def test_staging_rejects_source_overlap_and_missing_release(tmp_path: Path) -> None:
    source = tmp_path / "release"
    _write_well(source / "train", well_id="aaaaaaaa")
    _write_well(source / "test", well_id="bbbbbbbb")
    with pytest.raises(ValueError, match="overlap"):
        stage_data(source, destination=source / "prepared")
    with pytest.raises(FileNotFoundError, match="sample_submission"):
        stage_data(source, destination=tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_centroids_follow_requested_well_order_and_full_float64_columns(
    tmp_path: Path,
) -> None:
    filename = tmp_path / "centroids.csv"
    filename.write_text(
        "well_id,horizontal_avg_X,horizontal_avg_Y\nb,2.000001,3.125\na,-5.625,7.000003\nc,9.75,11.0625\n",
    )
    actual = read_centroids(filename, well_ids=("c", "a", "b"))
    np.testing.assert_array_equal(
        actual,
        np.asarray(
            [[9.75, 11.0625], [-5.625, 7.000003], [2.000001, 3.125]],
            dtype=np.float64,
        ),
        strict=True,
    )


@pytest.mark.parametrize(
    "body",
    ["a,2,3\na,4,5\n", "b,2,3\n", "a,nan,3\n", "a,2,inf\n"],
)
def test_centroid_errors_fail_before_native_data_or_model_construction(
    tmp_path: Path,
    body: str,
) -> None:
    filename = tmp_path / "centroids.csv"
    filename.write_text("well_id,horizontal_avg_X,horizontal_avg_Y\n" + body)
    with pytest.raises(ValueError, match="centroid"):
        read_centroids(filename, well_ids=("a",))


def _calibrate(
    depth: np.ndarray,
    *,
    gamma: np.ndarray,
    reference_depth: np.ndarray,
    reference_gamma: np.ndarray,
    blend: float,
    raw_weight: float,
) -> np.ndarray:
    config = ReferenceCalibration.Config()
    config.blend = blend
    config.raw_weight = raw_weight
    return config.make()(
        depth,
        gamma=gamma,
        reference_depth=reference_depth,
        reference_gamma=reference_gamma,
    )


@pytest.mark.parametrize(("blend", "raw_weight"), [(0.7, 0.3), (0.75, 0.25)])
def test_curve_uses_quarter_depth_groups_and_only_visible_gamma(
    blend: float,
    raw_weight: float,
) -> None:
    depth = np.asarray([10, 10.0625, 10.25, 11, np.nan], dtype=np.float64)
    gamma = np.asarray([70, 90, 84, 88, 1e9], dtype=np.float64)
    reference_depth = np.asarray([9, 10, 10.125, 10.25, 11, 12], dtype=np.float64)
    reference_gamma = np.asarray([20, 30, 40, 50, 60, 70], dtype=np.float64)
    result = _calibrate(
        depth,
        gamma=gamma,
        reference_depth=reference_depth,
        reference_gamma=reference_gamma,
        blend=blend,
        raw_weight=raw_weight,
    )
    np.testing.assert_array_equal(
        result,
        [
            20,
            blend * 80 + raw_weight * 30,
            blend * 82 + raw_weight * 40,
            blend * 84 + raw_weight * 50,
            blend * 88 + raw_weight * 60,
            70,
        ],
    )
    np.testing.assert_array_equal(reference_gamma, [20, 30, 40, 50, 60, 70])
    assert not np.shares_memory(reference_gamma, result)


def test_curve_requires_anchor_and_empty_prefix_returns_raw_curve() -> None:
    reference = np.asarray([30, np.nan, 50], dtype=np.float64)
    with pytest.raises(
        ValueError,
        match=r"^Reference calibration requires a visible depth anchor\.$",
    ):
        _calibrate(
            np.full(2, np.nan),
            gamma=np.ones(2),
            reference_depth=np.arange(3, dtype=np.float64),
            reference_gamma=reference,
            blend=0.7,
            raw_weight=0.3,
        )
    result = _calibrate(
        np.asarray([2, 3], dtype=np.float64),
        gamma=np.full(2, np.nan),
        reference_depth=np.arange(3, dtype=np.float64),
        reference_gamma=reference,
        blend=0.75,
        raw_weight=0.25,
    )
    np.testing.assert_array_equal(result, reference)


def test_curve_uses_last_anchor_and_includes_both_window_endpoints() -> None:
    result = _calibrate(
        np.asarray([-100.5, -100, -99.75, 100, 100.5, 0, np.nan], dtype=np.float64),
        gamma=np.asarray([9999, 20, 40, 80, 9999, np.nan, 1e9], dtype=np.float64),
        reference_depth=np.asarray(
            [-100.5, -100, -99.75, 100, 100.5],
            dtype=np.float64,
        ),
        reference_gamma=np.ones(5),
        blend=0.75,
        raw_weight=0.25,
    )
    np.testing.assert_array_equal(result, [1, 15.25, 30.25, 60.25, 1])


def test_window_and_resolution_are_declared_fields() -> None:
    config = ReferenceCalibration.Config()
    config.window = 50
    config.resolution = 2
    depth = np.asarray([-60, 0, 0.25, 0.5], dtype=np.float64)
    result = config.make()(
        depth,
        gamma=np.asarray([1000, 10, 20, 40], dtype=np.float64),
        reference_depth=np.asarray([-60, 0, 0.5], dtype=np.float64),
        reference_gamma=np.zeros(3),
    )
    np.testing.assert_array_equal(result, [0, 0.7 * 15, 0.7 * 40])


def test_curve_compensates_group_sums_and_preserves_public_recipe_weights() -> None:
    result = _calibrate(
        np.zeros(4),
        gamma=np.asarray([1e16, 1, 1, 1], dtype=np.float64),
        reference_depth=np.asarray([-1, 0, 1], dtype=np.float64),
        reference_gamma=np.zeros(3),
        blend=1,
        raw_weight=0,
    )
    np.testing.assert_array_equal(result, [0, (10_000_000_000_000_000 + 4) / 4, 0])
    calibrated = _calibrate(
        np.arange(3, dtype=np.float64),
        gamma=np.zeros(3),
        reference_depth=np.arange(3, dtype=np.float64),
        reference_gamma=np.ones(3),
        blend=0.7,
        raw_weight=0.3,
    )
    # The public model fixes raw weight 0.3; subtracting 0.7 changes its bits.
    np.testing.assert_array_equal(calibrated, np.full(3, 0.3))
    raw = np.asarray([np.inf, -np.inf, np.nan])
    preserved = _calibrate(
        np.arange(3, dtype=np.float64),
        gamma=np.zeros(3),
        reference_depth=np.arange(3, dtype=np.float64),
        reference_gamma=raw,
        blend=0.75,
        raw_weight=0.25,
    )
    np.testing.assert_array_equal(preserved, raw)


class WorkerProbe(Dataset[dict[str, Tensor]]):
    """Use the native worker protocol while retaining every source-observed leaf."""

    def __init__(self) -> None:
        self.streams = WorkerStreams.Config(seed=7).make()
        self.seed_offset = 0

    def seed_worker(self, seed: int) -> None:
        """Reset both consumer-owned streams exactly once per worker lifetime."""
        self.streams.reset(seed + self.seed_offset)

    def __len__(self) -> int:
        return 3

    @override
    def __getitem__(self, index: int) -> dict[str, Tensor]:
        result = {
            "index": torch.tensor(index),
            "initial_seed": torch.tensor(torch.initial_seed()),
            "normal": torch.from_numpy(self.streams.numpy.normal(size=7)),
            "python": torch.tensor(
                [self.streams.python.randint(0, 999) for _ in range(5)],
            ),
        }
        result.update(
            {
                name.removeprefix("rng/"): value
                for name, value in full_rng_record(
                    numpy=self.streams.numpy,
                    python=self.streams.python,
                ).items()
            },
        )
        return result


class SingleWorkerProbe(WorkerProbe):
    """Expose one complete item per epoch without losing persistent streams."""

    @override
    def __len__(self) -> int:
        return 1


def test_reseeding_restores_both_complete_independent_streams() -> None:
    streams = WorkerStreams.Config(seed=7).make()
    streams.numpy.normal(size=7)
    streams.python.getrandbits(32)
    streams.reset(19)
    expected = full_rng_record(
        numpy=np.random.RandomState(19),
        python=InlineConfig(random.Random, 19).make(),
    )
    actual = full_rng_record(numpy=streams.numpy, python=streams.python)
    assert mismatches(expected, actual) == []


def test_worker_seed_uses_modulo_of_the_complete_torch_seed() -> None:
    dataset = WorkerProbe()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2**32 + 19)
        initialize_worker(
            0,
            information=partial(WorkerInfo, dataset=dataset),
        )
    expected = full_rng_record(
        numpy=np.random.RandomState(19),
        python=InlineConfig(random.Random, 19).make(),
    )
    actual = full_rng_record(numpy=dataset.streams.numpy, python=dataset.streams.python)
    assert mismatches(expected, actual) == []


@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("mutation", [False, True])
@pytest.mark.parametrize("epochs", [1, 3])
def test_complete_original_worker_outputs_states_and_loader_generator(
    workers: int,
    mutation: bool,
    epochs: int,
) -> None:
    probe = WorkerProbe() if epochs == 1 else SingleWorkerProbe()
    probe.seed_offset = int(mutation)
    if workers == 0 and mutation:
        probe.streams.reset(8)
    generator = torch.Generator().manual_seed(8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        loader = DataLoader(
            probe,
            batch_size=None,
            num_workers=workers,
            worker_init_fn=initialize_worker,
            generator=generator,
            multiprocessing_context="fork" if workers else None,
            persistent_workers=bool(workers),
            prefetch_factor=1 if workers else None,
        )
        samples = [sample for _ in range(epochs) for sample in loader]
        record = {
            f"{step}/{name}": value
            for step, sample in enumerate(samples)
            for name, value in sample.items()
        }
        record["loader/generator"] = generator.get_state()
    filename = f"worker_{workers}.pt" if epochs == 1 else f"worker_epoch_{workers}.pt"
    differences = mismatches(read_tensors(_CWD / "testdata" / filename), record)
    assert bool(differences) == mutation, differences


@pytest.mark.parametrize("training", [False, True])
def test_complete_dataset_observations_without_simulation(training: bool) -> None:
    config = WellDataset.Config()
    config.streams.seed = 7
    config.observations = InlineConfig(tuple, [feature_well()])
    config.transform.paths = config.transform.images = None
    config.training = training
    config.transform.features.prefix_rows = 32
    config.transform.features.target_rows = 64
    config.transform.features.reference_columns = 7
    dataset = config.make()
    assert len(dataset) == 1
    item = dataset[0]
    expected = config.transform.features.make()(feature_well(), training=training)
    assert item.metadata["well_id"] == "well0"
    for name, value in expected.arrays.items():
        np.testing.assert_array_equal(item.arrays[name], value, strict=True)
    for name, value in expected.auxiliary.items():
        np.testing.assert_array_equal(item.auxiliary[name], value, strict=True)


def test_dataset_composition_retains_the_complete_original_augmented_record() -> None:
    config = WellDataset.Config()
    config.streams.seed = 7
    config.observations = InlineConfig(
        tuple,
        [
            augmentation_well(prefix=512, suffix=193, well_id="query"),
            augmentation_well(prefix=32, suffix=1001, well_id="donor"),
        ],
    )
    config.transform.features.prefix_rows = 32
    config.transform.features.target_rows = 64
    config.transform.features.reference_columns = 3
    config.transform.features.channels = [
        *config.transform.features.channels[:-1],
        "geo_tvt_diff",
        "pf_particle_density_prob",
        "pf_prob_ffbsi",
        "geo_tvt_abs_diff",
        "geo_tvt_rel",
    ]
    config.transform.features.particle = ParticleFeatures.Config(
        path="pf_tvt_rel_pred_ffbsi",
    )
    dataset = config.make()
    dataset.seed_worker(29)
    dataset.streams.python.seed(31)
    record: dict[str, Tensor] = {}
    samples: list[FeatureItem] = []
    for step in range(3):
        item = dataset[0]
        samples.append(item)
        record.update(
            {
                f"{step}/{name}": torch.from_numpy(values)
                for name, values in item.arrays.items()
            },
        )
        record.update(
            {
                f"{step}/aux_targets/{name}": torch.from_numpy(values)
                for name, values in item.auxiliary.items()
            },
        )
        for name, value in item.metadata.items():
            if not isinstance(value, str):
                record[f"{step}/meta/{name}"] = torch.from_numpy(
                    np.asarray(value).copy(),
                )
    record.update(
        full_rng_record(numpy=dataset.streams.numpy, python=dataset.streams.python),
    )
    assert mismatches(read_tensors(_CWD / "testdata" / "augmentation.pt"), record) == []
    batch = collate_features(samples[:2])
    source_names = {"media": "unet_static", "label": "target"}
    collated = {
        source_names.get(name, name): value
        for name, value in batch.items()
        if isinstance(value, Tensor)
    }
    typed = cast("Batch", batch)
    collated.update(
        {f"aux_targets/{name}": value for name, value in typed["aux_targets"].items()},
    )
    for index, metadata in enumerate(typed["meta"]):
        assert metadata["well_id"] == "query"
        for name, value in metadata.items():
            if not isinstance(value, str):
                collated[f"meta/{index}/{name}"] = torch.from_numpy(
                    np.asarray(value).copy(),
                )
    assert mismatches(read_tensors(_CWD / "testdata" / "collation.pt"), collated) == []


def test_collation_preserves_every_field_boolean_auxiliary_order_and_metadata() -> None:
    config = ImageFeatures.Config()
    config.prefix_rows = 32
    config.target_rows = 64
    config.reference_columns = 7
    builder = config.make()
    samples = [builder(feature_well(), training=True) for _ in range(2)]
    for item in samples:
        item.auxiliary["a"] = np.asarray([7], dtype=np.float32)
    batch = collate_features(samples)
    keys = {"unet_static": "media", "target": "label"}
    assert tuple(batch) == (
        *(keys.get(name, name) for name in samples[0].arrays),
        "aux_targets",
        "meta",
    )
    for name, values in samples[0].arrays.items():
        tensor = batch[keys.get(name, name)]
        assert isinstance(tensor, Tensor)
        expected = np.stack([item.arrays[name] for item in samples])
        if values.dtype != np.bool_:
            expected = expected.astype(np.float32)
        np.testing.assert_array_equal(tensor.numpy(), expected, strict=True)
    auxiliary = cast("Batch", batch)["aux_targets"]
    assert tuple(auxiliary) == ("GR_penalty_error", "a")
    assert batch["meta"] == [item.metadata for item in samples]


@pytest.mark.parametrize("persistent_workers", [False, True])
def test_cpu_loader_disables_worker_only_options_and_keeps_partial_batches(
    persistent_workers: bool,
) -> None:
    config = WellLoader.Config()
    config.seed = 8
    config.dataset.streams.seed = 7
    config.num_workers = 0
    config.drop_last = False
    config.shuffle = False
    config.persistent_workers = persistent_workers
    config.batch_size = 2
    config.dataset.transform.paths = None
    config.dataset.transform.images = None
    config.dataset.observations = InlineConfig(tuple, [feature_well()] * 3)
    config.dataset.transform.features.prefix_rows = 32
    config.dataset.transform.features.target_rows = 64
    config.dataset.transform.features.reference_columns = 7
    loader = config.make().loader
    assert len(loader) == 2
    assert loader.pin_memory is False
    assert loader.prefetch_factor is None
    batches = list(loader)
    for batch, count in zip(batches, (2, 1), strict=True):
        value = batch["media"]
        assert isinstance(value, Tensor)
        assert value.shape[0] == count


def test_worker_initialization_requires_an_actual_worker() -> None:
    with pytest.raises(ValueError, match="worker is not None"):
        initialize_worker(0)


def test_dataset_keeps_the_complete_empirical_donor_guard() -> None:
    config = WellDataset.Config()
    config.streams.seed = 7
    config.observations = InlineConfig(
        tuple,
        [augmentation_well(prefix=32, suffix=999, well_id="donor")],
    )
    with pytest.raises(ValueError, match="pool is empty"):
        config.make()


def test_loader_requires_a_resolved_generator_seed() -> None:
    with pytest.raises(ValueError, match="resolved generator seed"):
        WellLoader.Config().make()


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("perturb", [False, True])
def test_complete_active_features_match_external_record(
    training: bool,
    perturb: bool,
) -> None:
    config = ImageFeatures.Config(prefix_rows=32, target_rows=64, reference_columns=7)
    config.channels += [
        "geo_tvt_diff",
        "pf_particle_density_prob",
        "pf_tvt_ffbsi",
        "pf_prob_ffbsi",
        "geo_tvt_abs_diff",
        "geo_tvt_rel",
    ]
    config.particle = ParticleFeatures.Config(path="pf_tvt_rel_pred_ffbsi")
    if perturb:
        config.gamma_stats = (
            config.gamma_stats[0],
            float(
                cast(
                    "np.float32",
                    np.nextafter(np.float32(config.gamma_stats[1]), np.float32(np.inf)),
                ),
            ),
        )
    item = config.make()(feature_well(), training=training)
    prefix = str(training)
    record = {
        f"{prefix}/{name}": torch.from_numpy(values)
        for name, values in item.arrays.items()
    }
    record.update(
        {
            f"{prefix}/aux/{name}": torch.from_numpy(values)
            for name, values in item.auxiliary.items()
        },
    )
    for name, value in item.metadata.items():
        if isinstance(value, str):
            continue
        record[f"{prefix}/meta/{name}"] = torch.from_numpy(np.asarray(value).copy())
    expected = read_tensors(_CWD / "testdata" / "features.pt")
    differences = mismatches(
        {key: value for key, value in expected.items() if key.startswith(f"{prefix}/")},
        record,
    )
    if perturb:
        assert differences
        assert any("unet_static" in difference for difference in differences)
    else:
        assert differences == []


def test_finish_promotes_numeric_inputs_and_zeros_all_nonfinite_values() -> None:
    values = np.asarray([np.nan, np.inf, -np.inf, 1 + 2**-30], dtype=np.float64)
    result = _finish(values)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [0, 0, 0, 1])
    assert np.isnan(values[:1]).all()
    assert values[1] == np.inf
    assert values[2] == -np.inf


def test_feature_errors_and_absent_cache_support() -> None:
    config = ImageFeatures.Config(prefix_rows=32, target_rows=64, reference_columns=7)
    well = feature_well()
    absent_anchor = {
        **well.horizontal,
        "TVT_input": np.full(97, np.nan, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="visible depth anchor"):
        config.make()(replace(well, horizontal=absent_anchor), training=False)
    config.channels = ["pf_prob", "pf_tvt"]
    config.particle = ParticleFeatures.Config()
    with pytest.raises(ValueError, match="complete particle prior"):
        config.make()(replace(well, particle=None), training=False)
    assert well.particle is not None
    cache = {**well.particle, "window_orig_index": np.full(4, -1, dtype=np.float32)}
    item = config.make()(replace(well, particle=cache), training=False)
    assert np.count_nonzero(item.arrays["unet_static"]) == 0
    prediction = item.metadata["pf_tvt_rel_pred"]
    assert isinstance(prediction, np.ndarray)
    assert np.count_nonzero(prediction) == 0


def test_short_visible_context_retains_original_rows_and_owns_output_storage() -> None:
    well = feature_well()
    horizontal = {name: values.copy() for name, values in well.horizontal.items()}
    horizontal["TVT_input"][2:] = np.nan
    before = {name: values.copy() for name, values in horizontal.items()}
    config = ImageFeatures.Config(prefix_rows=32, target_rows=64, reference_columns=7)
    item = config.make()(replace(well, horizontal=horizontal), training=True)
    assert item.metadata["last_seen_idx"] == 1
    assert item.metadata["kept_suffix_len"] == 64
    assert item.metadata["suffix_len"] == 95
    np.testing.assert_array_equal(
        item.arrays["target_mask"],
        [False, True, True, False],
    )
    indices = item.metadata["submit_indices"]
    assert isinstance(indices, np.ndarray)
    assert len(indices) == 95
    for name, values in horizontal.items():
        np.testing.assert_array_equal(values, before[name])


def feature_well() -> WellRecord:
    """Rebuild complete deterministic paths used by the external feature witness."""
    row = np.arange(97, dtype=np.float32)
    target = (20 + row / 8 + row**2 / 512).astype(np.float32)
    visible = target.copy()
    visible[32:] = np.nan
    gamma = (70 + row / 16 + (row % 13) ** 2 / 8).astype(np.float32)
    gamma[[2, 41]] = np.nan
    axis = np.arange(7, dtype=np.float32)
    reference_gamma = (75 + axis**2 * 2).astype(np.float32)
    reference_gamma[3] = np.nan
    probability = np.arange(1, 29, dtype=np.float32).reshape(4, 7)
    probability /= probability.sum(axis=1, keepdims=True)
    return WellRecord(
        well_id="well0",
        horizontal={
            "MD": row,
            "X": (row / 2).astype(np.float32),
            "Y": (row / 4).astype(np.float32),
            "Z": (-10 + row / 16).astype(np.float32),
            "GR": gamma,
            "TVT_input": visible,
            "TVT": target,
            "orig_index": row,
            "geo_s_rel_prior_abs": (10 + row / 16).astype(np.float32),
        },
        reference_depth=(-180 + axis * 80).astype(np.float32),
        reference_gamma=reference_gamma,
        particle={
            "window_orig_index": np.asarray([15.5, 47.5, 79.5, -1], dtype=np.float32),
            "window_tvt_rel": np.asarray([-0.5, 1, 3, 0], dtype=np.float32),
            "window_has_tvt": np.asarray([1, 1, 1, 0], dtype=np.float32),
            "pf_prob": probability,
            "pf_prob_ffbsi": probability[:, ::-1].copy(),
            "pf_particle_density_prob": probability.copy(),
            "pf_tvt_rel_pred": np.asarray([-0.5, 1.25, 2.5, 0], dtype=np.float32),
            "pf_tvt_rel_pred_ffbsi": np.asarray([-0.5, 1.5, 3.5, 0], dtype=np.float32),
        },
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_probability_image_is_correctly_rounded_float32_log1p(transpose: bool) -> None:
    values = np.asarray(
        [[0, 0.01, 0.1, 0.2, 0.5], [1, 2, 3, 5, 10]],
        dtype=np.float32,
    )
    if transpose:
        values = values.T
    original = values.copy()
    expected = np.asarray(
        [
            [0, 1_008_928_438, 1_036_202_492, 1_044_034_177, 1_053_792_543],
            [1_060_205_080, 1_066_180_436, 1_068_593_688, 1_071_994_976, 1_075_410_718],
        ],
        dtype=np.uint32,
    ).view(np.float32)
    if transpose:
        expected = expected.T
    actual = probability_image(values, columns=1)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(values, original)
    assert actual.dtype == np.float32
    assert actual.shape == values.shape


def test_quadratic_original_binary64_solve_order() -> None:
    normal = np.asarray(
        [
            float.fromhex(value)
            for value in (
                "0x1.aab901d4493efp+2",
                "0x1.5247a66cab6aep-1",
                "0x1.531483103feeep+3",
                "0x1.5247a66cab6aep-1",
                "0x1.531483103feeep+3",
                "0x1.bdef7bdef7bd3p-1",
                "0x1.531483103feeep+3",
                "0x1.bdef7bdef7bd3p-1",
                "0x1.f000000000000p+4",
            )
        ],
        dtype=np.float64,
    ).reshape(3, 3)  # _quadratic_solve solves the three quadratic moments.
    rhs = np.asarray(
        [
            float.fromhex(value)
            for value in (
                "0x1.8b7bd6717b569p+9",
                "0x1.4b08421084210p+6",
                "0x1.2838000000000p+11",
            )
        ],
        dtype=np.float64,
    )
    expected = np.asarray(
        [
            float.fromhex(value)
            for value in (
                "-0x1.9da2ca6ccffb0p+2",
                "0x1.c09277b683a34p+0",
                "0x1.3a69f87a3d973p+6",
            )
        ],
        dtype=np.float64,
    )
    original_normal, original_rhs = normal.copy(), rhs.copy()
    actual = _quadratic_solve(normal, rhs)
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    np.testing.assert_array_equal(normal, original_normal)
    np.testing.assert_array_equal(rhs, original_rhs)
    assert actual.dtype == np.float64


def test_complete_probability_cleanup_is_owned_and_preserves_empty_rows() -> None:
    values = np.asarray([[np.nan, np.inf, -np.inf, -1], [1, 2, 0, 1]], dtype=np.float32)
    before = values.copy()
    np.testing.assert_array_equal(
        normalize_probability(values),
        np.asarray([[0, 0, 0, 0], [0.25, 0.5, 0, 0.25]], dtype=np.float32),
    )
    np.testing.assert_array_equal(values, before)


def test_particle_image_channels_require_the_particle_features() -> None:
    config = ImageFeatures.Config()
    config.channels = [*config.channels, "pf_prob_ffbsi"]
    config.particle = None
    with pytest.raises(ValueError, match="require the particle features"):
        config.make()


def write_well(directory: Path, *, with_target: bool = True) -> None:
    """Rebuild every complete deterministic CSV observation used externally."""
    row = np.arange(96, dtype=np.float64)
    depth = 20 + row / 128 + row**2 / 65_536
    known = depth.copy()
    known[32:] = np.nan
    gamma = 75 + (row % 17) / 4 + row / 32
    gamma[2::19] = np.nan
    columns = {
        "MD": row,
        "X": 4 + row / 2,
        "Y": 9 + row**2 / 128,
        "Z": -10 + row / 64,
        "GR": gamma,
        "TVT_input": known,
    }
    if with_target:
        columns["TVT"] = depth
    for index, name in enumerate(FORMATIONS):
        values = row / 16 + index
        values[index::7] = np.nan
        columns[name] = values
    with (directory / "tiny__horizontal_well.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*columns, "unused_text"])
        for values in zip(*(value.flat for value in columns.values()), strict=True):
            writer.writerow([*values, "ignored"])
    axis = np.arange(9, dtype=np.float64)
    with (directory / "tiny__typewell.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["TVT", "GR"])
        writer.writerows(
            zip((-140 + axis * 40).flat, (70 + axis**2 / 2).flat, strict=True),
        )


def calibration() -> ReferenceCalibration:
    return ReferenceCalibration.Config().make()


def surface() -> RelativeSurface:
    row = np.arange(96, dtype=np.float64)
    return RelativeSurface(
        values=(row**2 / 1024).astype(np.float32),
        diagnostics={
            "geo_nbr_std": (row / 16).astype(np.float32),
            "geo_nbr_dS_std": (row / 32).astype(np.float32),
            "geo_radial_extrap_score": (1 + row / 128).astype(np.float32),
        },
    )


@pytest.mark.parametrize("with_particle", [False, True])
def test_complete_original_loading_and_cache_outputs(
    tmp_path: Path,
    with_particle: bool,
) -> None:
    write_well(tmp_path)
    raw = load_raw(tmp_path, "tiny")
    expected_formation = (
        "BUDA"  # codespell:ignore buda -- Required ROGII horizontal CSV formation key.
    )
    assert expected_formation in raw.horizontal
    assert "unused_text" not in raw.horizontal
    original = read_tensors(_CWD / "testdata" / "particle.pt")
    producer = {
        key.removeprefix("64/True/"): value.numpy().copy()
        for key, value in original.items()
        if key.startswith("64/True/")
    }
    path = tmp_path / "tiny.npz"
    write_particle_cache(path, producer)
    cached = read_particle_cache(path) if with_particle else None
    prepared = prepare_well(
        raw,
        surface=surface(),
        calibration=calibration(),
        particle=cached,
    )
    actual: dict[str, torch.Tensor] = {
        f"well/horizontal/{name}": torch.from_numpy(value)
        for name, value in prepared.horizontal.items()
    }
    actual["well/typewell/TVT"] = torch.from_numpy(prepared.reference_depth)
    actual["well/typewell/GR"] = torch.from_numpy(prepared.reference_gamma)
    if cached is not None:
        assert prepared.particle is not None
        actual.update(
            {
                f"well/pf_cache/{name}": torch.from_numpy(value)
                for name, value in prepared.particle.items()
            },
        )
    filename = "particle_loading.pt" if with_particle else "loading.pt"
    assert mismatches(read_tensors(_CWD / "testdata" / filename), actual) == []
    if with_particle:
        assert prepared.particle is not None
        assert np.count_nonzero(prepared.particle["pf_prob"][0:1]) == 0
        assert np.isnan(prepared.particle["pf_tvt_pred"].flat[0])
    else:
        assert prepared.particle is None


def test_all_raw_tracking_inputs_keep_original_precision_and_row_count(
    tmp_path: Path,
) -> None:
    write_well(tmp_path, with_target=False)
    raw = load_raw(tmp_path, "tiny")
    geographic = trajectory(raw)
    particle = particle_well(raw)
    assert geographic.depth is None
    assert particle.depth is None
    assert geographic.suffix_start == 32
    assert len(geographic.x) == len(particle.md) == 96
    np.testing.assert_array_equal(
        geographic.md,
        raw.horizontal["MD"].astype(np.float32),
        strict=True,
    )
    np.testing.assert_array_equal(particle.md, raw.horizontal["MD"], strict=True)
    np.testing.assert_array_equal(
        particle.reference_gamma,
        raw.reference_gamma,
        strict=True,
    )
    prepared = prepare_well(raw, surface=surface(), calibration=calibration())
    assert prepared.particle is None
    assert "TVT" not in prepared.horizontal


def test_missing_geographic_rows_and_anchor_fail_before_feature_construction(
    tmp_path: Path,
) -> None:
    write_well(tmp_path)
    raw = load_raw(tmp_path, "tiny")
    prior = surface()
    with pytest.raises(ValueError, match="prior length"):
        prepare_well(
            raw,
            surface=replace(prior, values=prior.values[:-1]),
            calibration=calibration(),
        )
    with pytest.raises(ValueError, match="diagnostic length"):
        prepare_well(
            raw,
            surface=replace(prior, diagnostics={"geo_nbr_std": prior.values[:-1]}),
            calibration=calibration(),
        )
    hidden = {**raw.horizontal, "TVT_input": np.full(96, np.nan, dtype=np.float64)}
    with pytest.raises(ValueError, match="visible depth anchor"):
        trajectory(replace(raw, horizontal=hidden))


@pytest.mark.parametrize("anchor", [3, 2**24 + 3])
def test_cache_sanitization_includes_zero_and_the_rounded_anchor_boundary(
    tmp_path: Path,
    anchor: int,
) -> None:
    boundary = np.float32(anchor)
    step = np.float32(1 if anchor == 3 else 4)
    indices = np.asarray(
        [0, -1, np.nan, boundary, boundary + step, boundary - step],
        dtype=np.float32,
    )
    payload: dict[str, NDArray[np.generic]] = {
        name: np.ones((6, 7), dtype=np.float32)
        for name in ("pf_prob", "pf_particle_density_prob", "pf_prob_ffbsi")
    }
    payload.update(
        {
            name: np.ones(6, dtype=np.float32)
            for name in (
                "pf_tvt_pred",
                "pf_tvt_rel_pred",
                "pf_tvt_pred_ffbsi",
                "pf_tvt_rel_pred_ffbsi",
                "window_tvt",
                "window_tvt_rel",
            )
        },
    )
    payload.update(
        {
            name: np.ones(6, dtype=np.bool_)
            for name in ("window_has_tvt", "target_mask")
        },
    )
    payload.update(
        {
            name: np.asarray(value, dtype=np.int64)
            for name, value in (
                ("last_seen_idx", anchor),
                ("prefix_start", 0),
                ("suffix_len", 1),
            )
        },
    )
    payload["window_orig_index"] = indices
    payload["tvt0"] = np.asarray(1, dtype=np.float64)
    path = tmp_path / "edges.npz"
    write_particle_cache(path, payload)
    result = read_particle_cache(path)
    for name in ("pf_prob", "pf_particle_density_prob", "pf_prob_ffbsi"):
        expected = np.repeat(
            np.asarray([0, 1, 1, 0, 1, 0], dtype=np.float32)[:, None],
            7,
            axis=1,
        )
        np.testing.assert_array_equal(result[name], expected, strict=True)
    for name in (
        "pf_tvt_pred",
        "pf_tvt_rel_pred",
        "pf_tvt_pred_ffbsi",
        "pf_tvt_rel_pred_ffbsi",
    ):
        np.testing.assert_array_equal(
            result[name],
            np.asarray([np.nan, 1, 1, np.nan, 1, np.nan], dtype=np.float32),
            strict=True,
        )
    np.testing.assert_array_equal(result["window_orig_index"], indices, strict=True)


@pytest.mark.parametrize("training", [False, True])
def test_fold_observations_reconstruct_complete_rows_from_printable_inputs(
    tmp_path: Path,
    training: bool,
) -> None:
    write_study_inputs(tmp_path, rows=13)
    raw = load_split(tmp_path / "train")
    config = FoldObservations.Config()
    config.directory = tmp_path / "train"
    config.geography = GeographicPrior.Config()
    config.support_well_ids = ("well0", "well1")
    config.well_ids = config.support_well_ids if training else ("well3", "well2")
    config.leave_one_out = training
    observations = config.make()
    assert len(observations) == 2
    assert observations[:][0] is observations[0]
    prior = config.geography.make().build(
        [trajectory(raw[name]) for name in config.support_well_ids],
        queries=[trajectory(raw[name]) for name in config.well_ids],
        leave_one_out=training,
    )
    for name, actual in zip(config.well_ids, observations, strict=True):
        expected = prepare_well(
            raw[name],
            surface=prior[name],
            calibration=ReferenceCalibration.Config().make(),
        )
        records = [
            {
                "well_id": well.well_id,
                "horizontal": well.horizontal,
                "reference_depth": well.reference_depth,
                "reference_gamma": well.reference_gamma,
                "particle": well.particle,
            }
            for well in (expected, actual)
        ]
        assert (
            mismatches(
                numerical_tensors(records[0]),
                numerical_tensors(records[1]),
                ulps=True,
            )
            == []
        )


def test_queries_and_support_may_come_from_different_splits(tmp_path: Path) -> None:
    write_study_inputs(tmp_path, rows=13)
    training = load_split(tmp_path / "train")
    test = load_split(tmp_path / "test")
    config = FoldObservations.Config()
    config.directory = tmp_path / "test"
    config.geography = GeographicPrior.Config()
    config.support_directory = tmp_path / "train"
    config.support_well_ids = tuple(training)
    config.well_ids = ("well2", "well0")
    config.leave_one_out = False
    prior = config.geography.make().build(
        [trajectory(well) for well in training.values()],
        queries=[trajectory(test[name]) for name in config.well_ids],
        leave_one_out=False,
    )
    observations = config.make()
    assert [well.well_id for well in observations] == ["well2", "well0"]
    for name, actual in zip(config.well_ids, observations, strict=True):
        expected = prepare_well(
            test[name],
            surface=prior[name],
            calibration=ReferenceCalibration.Config().make(),
        )
        assert (
            mismatches(
                numerical_tensors(expected.horizontal),
                numerical_tensors(actual.horizontal),
                ulps=True,
            )
            == []
        )


def test_particle_slot_attaches_each_well_cache(tmp_path: Path) -> None:
    write_study_inputs(tmp_path, rows=13)
    raw = load_split(tmp_path / "train")
    cache = tmp_path / "cache"
    particle = ParticlePrior.Config()
    particle.particles = 3
    particle.seeds = 5
    particle.paths = 2
    particle.prefix_rows = 4
    particle.target_rows = 6
    particle.downsample = 2
    particle.reference_columns = 7
    particle.reference_window = 100.0
    records = prepare_particles(
        raw,
        config=particle,
        directory=cache,
        has_target=True,
    )
    config = FoldObservations.Config()
    config.directory = tmp_path / "train"
    config.geography = GeographicPrior.Config()
    config.support_well_ids = config.well_ids = ("well1", "well0")
    slot = config.particle = ParticleCache.Config()
    slot.directory = cache
    for name, actual in zip(config.well_ids, config.make(), strict=True):
        assert actual.particle is not None
        assert (
            mismatches(
                numerical_tensors(records[name]),
                numerical_tensors(actual.particle),
                ulps=True,
            )
            == []
        )


@pytest.mark.parametrize("missing", ["directory", "geography"])
def test_observations_require_their_directory_and_geography(
    tmp_path: Path,
    missing: str,
) -> None:
    config = FoldObservations.Config()
    if missing != "directory":
        config.directory = tmp_path
    if missing != "geography":
        config.geography = GeographicPrior.Config()
    with pytest.raises(ValueError, match="directory and geography"):
        config.make()


@pytest.mark.parametrize("include_target", [False, True])
def test_expand_all_original_rows_and_extend_only_the_truncated_tail(
    include_target: bool,
) -> None:
    config = ExpandPrediction.Config()
    config.prefix_rows = 2
    config.target_rows = 5
    config.downsample = 2
    config.target_stats = (1.5, 2.0)
    prediction = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    metadata: list[dict[str, object]] = [
        {
            "well_id": "a",
            "suffix_len": 3,
            "tvt0": 100.0,
            "submit_indices": np.arange(7, 10),
            "suffix_z": np.arange(3, dtype=np.float32),
            "TVT": np.arange(101, 104, dtype=np.float32),
        },
        {
            "well_id": "b",
            "suffix_len": 7,
            "tvt0": 200.0,
            "submit_indices": np.arange(8, 15),
            "suffix_z": np.arange(7, dtype=np.float32),
            "TVT": np.arange(201, 208, dtype=np.float32),
        },
    ]
    actual = config.make()(prediction, metadata=metadata, include_target=include_target)
    assert actual.well_ids == ("a",) * 3 + ("b",) * 7
    assert np.array_equal(
        actual.submit_indices,
        np.array([7, 8, 9, 8, 9, 10, 11, 12, 13, 14]),
    )
    assert actual.tvt.dtype == np.float32
    assert np.array_equal(
        actual.tvt,
        np.array(
            [103.5, 103.5, 105.5, 211.5, 211.5, 213.5, 213.5, 215.5, 215.5, 215.5],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(actual.z, np.concatenate((np.arange(3), np.arange(7))))
    if include_target:
        target = np.concatenate((np.arange(101, 104), np.arange(201, 208))).astype(
            np.float32,
        )
        assert actual.target is not None
        assert np.array_equal(actual.target, target)
        assert pooled_rmse(actual.target, prediction=actual.tvt) == (688.5 / 10) ** 0.5
    else:
        assert actual.target is None


def test_expansion_requires_its_inherited_geometry() -> None:
    with pytest.raises(ValueError, match="window geometry"):
        ExpandPrediction.Config().make()
    config = ExpandPrediction.Config(prefix_rows=1, target_rows=2, downsample=1)
    with pytest.raises(ValueError, match="target statistics"):
        config.make()


# Seeds whose union reaches every line and branch the sixteen original seeds did.
@pytest.mark.parametrize("seed", [31, 5, 6, 8, 9, 12, 15])
def test_complete_path_simulation_arrays_and_rng(seed: int) -> None:
    record = augmentation_record(seed=seed)
    filename = "augmentation.pt" if seed == 31 else f"augmentation_{seed:02}.pt"
    assert mismatches(read_tensors(_CWD / "testdata" / filename), record) == []


def test_whole_coherent_reference_curve_boundary_record() -> None:
    numpy = np.random.RandomState(29)
    python = InlineConfig(random.Random, 31).make()
    transform = ReferenceJitter.Config().make()
    record: dict[str, torch.Tensor] = {}
    for length in (1, 2, 3, 9):
        query = augmentation_well(prefix=4, suffix=5, well_id="query")
        horizontal = {name: values.copy() for name, values in query.horizontal.items()}
        horizontal["TVT"] = np.asarray(
            [-1, 0, 1 + 2**-25, 2 + 2**-24, 3, np.nan, -2, 4, 5],
            dtype=np.float64,
        )
        horizontal["GR"] = 77 + np.arange(9, dtype=np.float64) / 7 + 2**-22
        horizontal["GR"][[1, 4]] = [np.inf, np.nan]
        axis = np.arange(length, dtype=np.float32)
        query = replace(
            query,
            horizontal=horizontal,
            reference_depth=axis,
            reference_gamma=(75 + axis**2 * np.float32(2 / 7)).astype(np.float32),
        )
        for step in range(3):
            result = transform(query, numpy=numpy, python=python)
            prefix = f"{length}/{step}"
            record.update(
                {
                    f"{prefix}/horizontal/{name}": torch.from_numpy(values.copy())
                    for name, values in result.horizontal.items()
                },
            )
            record[f"{prefix}/typewell/TVT"] = torch.from_numpy(result.reference_depth)
            record[f"{prefix}/typewell/GR"] = torch.from_numpy(result.reference_gamma)
            assert result.particle is not None
            record.update(
                {
                    f"{prefix}/pf_cache/{name}": torch.from_numpy(values.copy())
                    for name, values in result.particle.items()
                },
            )
    record.update(full_rng_record(numpy=numpy, python=python))
    differences = mismatches(
        read_tensors(_CWD / "testdata" / "jitter.pt"),
        record,
    )
    assert differences == [], "\n".join(differences)


@pytest.mark.parametrize("mutate_join", [False, True])
def test_whole_original_multiple_block_joins_and_rng(mutate_join: bool) -> None:
    donor = augmentation_well(prefix=32, suffix=1025, well_id="donor")
    horizontal = {name: values.copy() for name, values in donor.horizontal.items()}
    row = np.arange(1057, dtype=np.float32)
    depth = (20 + row / 64 + (row % 17) ** 2 / 2048).astype(np.float32)
    visible = depth.copy()
    visible[32:] = np.nan
    horizontal.update(TVT=depth, TVT_input=visible)
    config = BootstrapTrend.Config()
    if mutate_join:
        config.join_rows = 0
    sampler = config.make()
    sampler.prepare([replace(donor, horizontal=horizontal)])
    numpy = np.random.RandomState(7)
    python = InlineConfig(random.Random, 11).make()
    record = {
        f"path/{step}": torch.from_numpy(sampler(1033, numpy=numpy, python=python))
        for step in range(3)
    }
    record.update(full_rng_record(numpy=numpy, python=python))
    differences = mismatches(
        read_tensors(_CWD / "testdata" / "bootstrap.pt"),
        record,
    )
    assert bool(differences) == mutate_join, differences


@pytest.mark.parametrize("suffix", [999, 1000, 1001])
def test_bootstrap_pool_keeps_the_public_minimum_suffix(suffix: int) -> None:
    donor = augmentation_well(prefix=32, suffix=suffix, well_id="donor")
    sampler = BootstrapTrend.Config().make()
    if suffix < 1000:
        with pytest.raises(ValueError, match="pool is empty"):
            sampler.prepare([donor])
        return
    sampler.prepare([donor])
    numpy = np.random.RandomState(7)
    python = InlineConfig(random.Random, 11).make()
    path = sampler(1033, numpy=numpy, python=python)
    np.testing.assert_array_equal(path, np.arange(1033, dtype=np.float32) / 64)


@pytest.mark.parametrize("length", [0, -2])
def test_empty_bootstrap_request_retains_rng_and_empty_shape(length: int) -> None:
    sampler = BootstrapTrend.Config().make()
    numpy = np.random.RandomState(7)
    python = InlineConfig(random.Random, 11).make()
    before_numpy = numpy.get_state()
    before_python = python.getstate()
    result = sampler(length, numpy=numpy, python=python)
    assert result.shape == (0,)
    assert result.dtype == np.float32
    assert python.getstate() == before_python
    after = numpy.get_state()
    np.testing.assert_array_equal(before_numpy[1], after[1])
    assert before_numpy[2:] == after[2:]


def test_bootstrap_pool_interpolates_gaps_and_rejects_absent_finite_paths() -> None:
    donor = augmentation_well(prefix=32, suffix=1001, well_id="donor")
    horizontal = {name: values.copy() for name, values in donor.horizontal.items()}
    horizontal["TVT"][500] = np.nan
    sampler = BootstrapTrend.Config().make()
    sampler.prepare([replace(donor, horizontal=horizontal)])
    np.testing.assert_array_equal(
        sampler.pool[0],
        np.full(1000, 1 / 64, dtype=np.float32),
    )
    horizontal["TVT"][33:] = np.nan
    with pytest.raises(ValueError, match="pool is empty"):
        sampler.prepare([replace(donor, horizontal=horizontal)])
    with pytest.raises(ValueError, match="pool is empty"):
        sampler(
            3,
            numpy=np.random.RandomState(7),
            python=InlineConfig(random.Random, 11).make(),
        )
    config = BootstrapTrend.Config()
    for block, weight in zip(config.blocks, (-1, 0, 0), strict=True):
        block.weight = weight
    with pytest.raises(ValueError, match="positive distribution"):
        config.make()


@pytest.mark.parametrize("width", [1, 4, 20])
def test_reference_smoothing_keeps_edges_and_float32_arithmetic(width: int) -> None:
    well = augmentation_well(prefix=32, suffix=33, well_id="query")
    config = ReferenceTransform.Config()
    config.gain_std = 0
    config.offset_std = 0
    config.smooth_probability = 1
    config.smooth_window = width
    result = config.make()(
        well,
        numpy=np.random.RandomState(7),
        python=InlineConfig(random.Random, 11).make(),
    )
    count = min(width, len(well.reference_gamma))
    if count % 2 == 0:
        count -= 1
    expected = well.reference_gamma.copy()
    if count > 1:
        expected = np.convolve(
            np.pad(expected, count // 2, mode="edge"),
            np.full(count, 1 / count, dtype=np.float32),
            mode="valid",
        ).astype(np.float32)
    np.testing.assert_array_equal(result.reference_gamma, expected)
    np.testing.assert_array_equal(well.reference_gamma, 75 + np.arange(9) ** 2 * 2)


@pytest.mark.parametrize("width", [1, 2])
def test_identity_smoothing_owns_storage_and_preserves_signed_zero(width: int) -> None:
    values = np.asarray([-0.0, -0.0, -0.0], dtype=np.float32)
    result = _smooth(values, width)
    np.testing.assert_array_equal(result.view(np.uint32), values.view(np.uint32))
    assert result is not values
    assert result.dtype == np.float32


@pytest.mark.parametrize("finite", [True, False])
def test_coherent_curve_shift_preserves_horizontal_residuals(finite: bool) -> None:
    well = augmentation_well(prefix=32, suffix=33, well_id="query")
    well = replace(
        well,
        reference_depth=np.zeros(9, dtype=np.float32),
        reference_gamma=np.full(9, 80 if finite else np.nan, dtype=np.float32),
    )
    config = ReferenceJitter.Config()
    config.gain_std = 0
    config.offset_std = 0
    config.drift_std = 0
    config.smooth_probability = 1
    result = config.make()(
        well,
        numpy=np.random.RandomState(7),
        python=InlineConfig(random.Random, 11).make(),
    )
    np.testing.assert_array_equal(result.reference_gamma, well.reference_gamma)
    for name, values in well.horizontal.items():
        np.testing.assert_array_equal(result.horizontal[name], values)
    if not finite:
        assert result is not well


def test_constant_gamma_bias_and_reversed_single_derivative_blocks() -> None:
    bias = GammaBias.Config()
    bias.components = [BoxcarComponent.Config(window=1, weight=1)]
    bias.depth_std = 0
    bias.ranges = [MismatchRange.Config(lower=0, upper=0, weight=1)]
    bias.offset_std = 0
    result = bias.make()(
        np.zeros(3, dtype=np.float32),
        np.full(3, 80, dtype=np.float64),
        reference_depth=np.asarray([-1, 1], dtype=np.float32),
        reference_gamma=np.asarray([80, 80], dtype=np.float32),
        numpy=np.random.RandomState(7),
    )
    np.testing.assert_array_equal(result, np.zeros(3, dtype=np.float32))
    sampler_config = BootstrapTrend.Config()
    sampler_config.reverse_probability = 1
    sampler = sampler_config.make()
    sampler.prepare([augmentation_well(prefix=32, suffix=1001, well_id="donor")])
    path = sampler(
        5,
        numpy=np.random.RandomState(7),
        python=InlineConfig(random.Random, 11).make(),
    )
    np.testing.assert_array_equal(path, -np.arange(5, dtype=np.float32) / 64)


@pytest.mark.parametrize("nan_pattern", [True, False])
def test_prefix_shift_and_cyclic_gamma_keep_whole_input_ownership(
    nan_pattern: bool,
) -> None:
    query = augmentation_well(prefix=512, suffix=193, well_id="query")
    horizontal = {name: values.copy() for name, values in query.horizontal.items()}
    horizontal["GR"][:] = np.nan
    query = replace(query, horizontal=horizontal)
    config = PathSimulation.Config()
    config.prefix_sample_rows = 256
    config.shift_probability = 1
    config.shift_range = 0
    config.shift_nan_pattern = FixedNanPattern.Config(shift=nan_pattern)
    config.gamma_noise_shift_probability = 1
    config.reference_jitter.probability = 0
    config.reverse_probability = 0
    config.reference_transform.probability = 0
    config.reference_drift_probability = 0
    config.stretch_probability = 0
    config.tail_cut_probability = 0
    pipeline = config.make()
    pipeline.prepare([augmentation_well(prefix=32, suffix=1001, well_id="donor")])
    result = pipeline(
        query,
        numpy=np.random.RandomState(7),
        python=InlineConfig(random.Random, 11).make(),
    )
    np.testing.assert_array_equal(
        result.horizontal["Z"] + result.horizontal["TVT"],
        query.horizontal["Z"] + query.horizontal["TVT"],
    )
    np.testing.assert_array_equal(
        result.horizontal["TVT_input"][:512],
        result.horizontal["TVT"][:512],
    )
    assert np.isnan(result.horizontal["GR"]).all()
    for name, values in query.horizontal.items():
        np.testing.assert_array_equal(values, horizontal[name])
    assert result.horizontal["TVT"] is not query.horizontal["TVT"]


@pytest.mark.parametrize("length", [0, -2])
def test_bootstrap_rejects_nonpositive_candidate_blocks(length: int) -> None:
    config = BootstrapTrend.Config()
    config.blocks = [DerivativeBlock.Config(length=length, weight=1)]
    with pytest.raises(ValueError, match=r"block lengths.*positive"):
        config.make()


def test_bootstrap_rejects_finite_weights_with_overflowing_total() -> None:
    config = BootstrapTrend.Config()
    for block in config.blocks:
        block.weight = 1e308
    with pytest.raises(ValueError, match="positive distribution"):
        config.make()


@pytest.mark.parametrize(
    ("component", "weight"),
    [(True, 3e38), (True, 1e39), (False, 1e308)],
)
def test_gamma_rejects_unusable_totals_at_construction(
    component: bool,
    weight: float,
) -> None:
    config = GammaBias.Config()
    parts = config.components if component else config.ranges
    for part in parts:
        part.weight = weight
    name = "component" if component else "range"
    with pytest.raises(ValueError, match=f"^Gamma {name} weights must form a positive"):
        config.make()


def test_image_at_interpolates_inside_and_zeros_outside_the_rows() -> None:
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    positions = np.asarray([-0.5, 0, 0.25, 2, 2.5, np.nan, np.inf], dtype=np.float32)
    expected = np.asarray(
        [[0, 0], [0, 1], [0.5, 1.5], [4, 5], [0, 0], [0, 0], [0, 0]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(_image_at(values, positions), expected, strict=True)


@pytest.mark.parametrize("component", [ReferenceJitter, ReferenceTransform])
@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_components_apply_their_own_probability_with_one_gate_draw(
    component: type[ReferenceJitter | ReferenceTransform],
    probability: float,
) -> None:
    well = augmentation_well(prefix=32, suffix=33, well_id="query")
    config = component.Config()
    config.probability = probability
    part = config.make()
    python = InlineConfig(random.Random, 11).make()
    gate = InlineConfig(random.Random, 11).make()
    gate.random()
    result = part.apply(well, numpy=np.random.RandomState(7), python=python)
    if probability == 0:
        assert result is well
        assert python.getstate() == gate.getstate()
        return
    direct = part(well, numpy=np.random.RandomState(7), python=gate)
    assert python.getstate() == gate.getstate()
    np.testing.assert_array_equal(result.reference_gamma, direct.reference_gamma)
    assert result is not well


@pytest.mark.parametrize("shift", [False, True])
def test_fixed_nan_pattern_never_draws_while_sampled_pattern_draws_once(
    shift: bool,
) -> None:
    fixed = FixedNanPattern.Config(shift=shift).make()
    python = InlineConfig(random.Random, 5).make()
    before = python.getstate()
    assert fixed(python) is shift
    assert python.getstate() == before
    sampled = SampledNanPattern.Config(weights={shift: 1.0}).make()
    assert sampled(python) is shift
    expected = InlineConfig(random.Random, 5).make()
    expected.random()
    assert python.getstate() == expected.getstate()


@pytest.mark.parametrize("prefix", [32, 512])
def test_reversal_defines_short_prefixes_and_preserves_empty_target_windows(
    prefix: int,
) -> None:
    config = _reversal_config()
    query = augmentation_well(prefix=prefix, suffix=1, well_id="query")
    result = config.make()(
        query,
        numpy=np.random.RandomState(7),
        python=InlineConfig(random.Random, 11).make(),
    )
    retained = prefix + 1 if prefix == 32 else 162
    assert len(result.horizontal["MD"]) == retained
    assert np.isfinite(result.horizontal["TVT_input"]).all()
    features = ImageFeatures.Config()
    features.prefix_rows = 32
    features.target_rows = 64
    features.reference_columns = 3
    item = features.make()(result, training=True)
    prediction = torch.from_numpy(item.arrays["target"]).requires_grad_()
    logits = torch.zeros(*prediction.shape, 3, requires_grad=True)
    loss, details = PosteriorLoss.Config().make()(
        prediction,
        logits=logits,
        target=torch.zeros_like(prediction),
        probabilities=torch.from_numpy(item.arrays["typewell_target_probs"]),
        gamma_error=torch.from_numpy(item.auxiliary["GR_penalty_error"]),
        mask=torch.from_numpy(item.arrays["target_mask"]),
    )
    assert loss == 0
    assert all(value == 0 for value in details.values())
    loss.backward()
    assert prediction.grad is not None
    assert logits.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("jitter", [False, True])
@pytest.mark.parametrize("finite", [False, True])
def test_simulation_results_own_every_input_array(jitter: bool, finite: bool) -> None:
    query = augmentation_well(prefix=32, suffix=33, well_id="query")
    if not finite:
        query = replace(query, reference_gamma=np.full(9, np.nan, dtype=np.float32))
    if jitter:
        transform = ReferenceJitter.Config().make()
    else:
        config = _reversal_config()
        config.reverse_probability = 0
        transform = config.make()
    result = transform(
        query,
        numpy=np.random.RandomState(7),
        python=InlineConfig(random.Random, 11).make(),
    )
    assert result is not query
    assert result.horizontal is not query.horizontal
    assert not np.shares_memory(result.reference_depth, query.reference_depth)
    assert not np.shares_memory(result.reference_gamma, query.reference_gamma)
    assert result.particle is not query.particle
    assert result.particle is not None
    assert query.particle is not None
    for name, values in query.particle.items():
        assert not np.shares_memory(result.particle[name], values)
    for name, values in query.horizontal.items():
        assert not np.shares_memory(result.horizontal[name], values)


@pytest.mark.parametrize("prefix_sample_rows", [0, 4])
def test_visible_path_shift_checks_selected_rows_after_prefix_inclusion(
    prefix_sample_rows: int,
) -> None:
    query = augmentation_well(prefix=32, suffix=0, well_id="query")
    config = _reversal_config()
    config.shift_probability = 1
    config.reverse_probability = 0
    config.prefix_sample_rows = prefix_sample_rows
    simulation = config.make()
    simulation.prepare([augmentation_well(prefix=32, suffix=1001, well_id="donor")])
    numpy = np.random.RandomState(7)
    python = InlineConfig(random.Random, 11).make()
    numpy_before = full_rng_record(numpy=numpy, python=python)
    actual = simulation(query, numpy=numpy, python=python)
    numpy_after = full_rng_record(numpy=numpy, python=python)
    same_numpy = all(
        torch.equal(values, numpy_after[name])
        for name, values in numpy_before.items()
        if name.startswith("rng/numpy/")
    )
    if prefix_sample_rows > 0:
        assert not np.array_equal(actual.horizontal["TVT"], query.horizontal["TVT"])
        assert not same_numpy
    else:
        for name, values in query.horizontal.items():
            np.testing.assert_array_equal(actual.horizontal[name], values)
        expected = InlineConfig(random.Random, 11).make()
        for _ in range(13):
            expected.random()
        assert python.getstate() == expected.getstate()
        assert same_numpy


def _reversal_config() -> PathSimulation.Config:
    config = PathSimulation.Config()
    config.shift_probability = 0
    config.reference_jitter.probability = 0
    config.reverse_probability = 1
    config.reference_transform.probability = 0
    config.reference_drift_probability = 0
    config.stretch_probability = 0
    config.tail_cut_probability = 0
    return config


@pytest.mark.parametrize(
    ("width", "expected_hex"),
    [
        (32, ("0x1.02e946p-5", "-0x1.15dd84p-4", "-0x1.f9614ep-6")),
        (63, ("0x1.37a3b2p-4", "0x1.6b526cp-4", "0x1.4d657ep-4")),
        (64, ("0x1.b3c346p-4", "0x1.0a69f0p-4", "0x1.8bf1aap-4")),
        (65, ("0x1.533b2ap-4", "0x1.490796p-4", "0x1.742a7ep-4")),
        (96, ("0x1.38ca60p-4", "0x1.c4cedep-5", "0x1.cd1d40p-5")),
        (256, ("0x1.5afd16p-6", "0x1.d2a580p-7", "0x1.9747f2p-6")),
        (1536, ("0x1.c8730cp-6", "0x1.c52f40p-6", "0x1.c90e12p-6")),
    ],
)
def test_boxcar_original_binary32_accumulation_order(
    width: int,
    expected_hex: tuple[str, str, str],
) -> None:
    values = np.random.RandomState(719).normal(size=width + 2).astype(np.float32)
    original = values.copy()
    expected = np.asarray(
        [float.fromhex(value) for value in expected_hex],
        dtype=np.float32,
    )
    actual = _boxcar_noise(values, width=width)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(values, original)
    assert actual.dtype == np.float32


def test_flat_drift_noise_leaves_the_paired_curve_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def flat(values: object) -> float:
        del values
        return 0.0

    monkeypatch.setattr(np, "std", flat)
    well = augmentation_well(prefix=32, suffix=33, well_id="query")
    config = ReferenceJitter.Config()
    config.gain_std = config.offset_std = config.smooth_probability = 0
    config.drift_std = 1
    result = config.make()(
        well,
        numpy=np.random.RandomState(7),
        python=InlineConfig(random.Random, 11).make(),
    )
    np.testing.assert_array_equal(result.reference_gamma, well.reference_gamma)


def test_shifted_nan_pattern_zero_fills_a_region_without_finite_gamma() -> None:
    query = augmentation_well(prefix=32, suffix=1001, well_id="query")
    query = replace(
        query,
        reference_gamma=np.full_like(query.reference_gamma, np.nan),
    )
    config = _reversal_config()
    config.shift_probability = 1
    config.reverse_probability = 0
    config.shift_nan_pattern = FixedNanPattern.Config(shift=True)
    simulation = config.make()
    simulation.prepare([augmentation_well(prefix=32, suffix=1001, well_id="donor")])
    result = simulation(
        query,
        numpy=np.random.RandomState(7),
        python=InlineConfig(random.Random, 11).make(),
    )
    suffix = np.isnan(query.horizontal["TVT_input"])
    assert np.isfinite(result.horizontal["GR"][suffix]).sum() < suffix.sum()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compensation_recovers_terms_a_naive_sum_loses(
    dtype: type[np.floating],
) -> None:
    total, correction = dtype(0), dtype(0)
    big = dtype(2.0**24 if dtype is np.float32 else 2.0**53)
    for value in (big, dtype(1), dtype(1), dtype(1), dtype(1)):
        total, correction = kahan_add(total, correction, value)
    naive = big
    for _ in range(4):
        naive += dtype(1)
    assert naive == big
    assert total - correction == big + dtype(4)
    assert type(total) is dtype


def test_step_returns_sum_and_the_exact_residual() -> None:
    total, correction = kahan_add(np.float64(1e16), np.float64(0), np.float64(1.0))
    assert total == np.float64(1e16)
    assert correction == np.float64(-1.0)


class _FixedDraw(random.Random):
    """Return one fixed uniform draw, so a gate sits exactly on its probability."""

    @override
    def random(self) -> float:
        return 0.5


def test_a_gate_draw_equal_to_the_probability_skips_the_component() -> None:
    well = augmentation_well(prefix=32, suffix=33, well_id="query")
    config = ReferenceJitter.Config()
    config.probability = 0.5
    result = config.make().apply(
        well,
        numpy=np.random.RandomState(7),
        python=_FixedDraw(),
    )
    assert result is well


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
