"""Trial alignment, normalization, grouping, and submission contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import csv

from treekle import from_plain

import numpy as np
import pytest
import torch

from priml.baselines.motor_imagery.data import (
    Preprocessing,
    Trial,
    WindowData,
    _butterworth_bandpass,
    _causal_filter,
    _complex_dot,
    _filter_initial_state,
    _hypot,
    _notch,
    _pole_polynomial,
    _product,
    _quotient,
    _square_root,
    subject_id,
    write_submission,
)
from priml.baselines.motor_imagery.testing import (
    source_preprocessing,
)
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

# exp000's filters as SciPy 1.16.3 designed them on x86_64 for filtering.pt.
# aarch64 SciPy rounds two denominator terms differently; do not recapture there.
_NOTCH_NUMERATOR: Final = [
    "0x1.f57ec3b2d27d1p-1",
    "-0x1.35f0e978b5cb3p-1",
    "0x1.f57ec3b2d27d1p-1",
]
_NOTCH_DENOMINATOR: Final = ["0x1p+0", "-0x1.35f0e978b5cb3p-1", "0x1.eafd8765a4fa2p-1"]
_BAND_NUMERATOR: Final = [
    "0x1.fca3a679bc166p-6",
    "0x0p+0",
    "-0x1.fca3a679bc166p-4",
    "0x0p+0",
    "0x1.7d7abcdb4d10cp-3",
    "0x0p+0",
    "-0x1.fca3a679bc166p-4",
    "0x0p+0",
    "0x1.fca3a679bc166p-6",
]
_BAND_DENOMINATOR: Final = [
    "0x1p+0",
    "-0x1.44f964874b0bcp+2",
    "0x1.673368b6f9f9dp+3",
    "-0x1.cc42acfd270acp+3",
    "0x1.7d7a6edb8de1ep+3",
    "-0x1.a61081d1061eap+2",
    "0x1.2c599bf9aab12p+1",
    "-0x1.ef794fc9b9221p-2",
    "0x1.7f2c89f4131c4p-5",
]
_SQRT3: Final = float.fromhex("0x1.bb67ae8584caap+0")
"""The correctly rounded root of 3; a root through pow need not round to it."""


def test_subject_id_handles_both_published_filename_formats() -> None:
    assert subject_id(Path("S002-001_eeg.csv")) == "S002"
    assert subject_id(Path("S008_test_003_eeg.csv")) == "S008"
    with pytest.raises(ValueError, match="subject"):
        subject_id(Path("unknown.csv"))


def test_standardization_is_trial_local_and_matches_reference_arithmetic() -> None:
    cfg = Preprocessing.Config()
    cfg.eps = 1e-8
    cfg.window_size = cfg.window_step = 5
    values = np.arange(15, dtype=np.float64).reshape(5, 3)
    expected = (values - np.mean(values, axis=0, keepdims=True)) / (
        np.std(values, axis=0, keepdims=True) + cfg.eps
    )
    assert np.array_equal(cfg.make().standardize(values), expected)


def test_window_alignment_retains_trial_ids_and_short_test_padding() -> None:
    cfg = Preprocessing.Config()
    cfg.window_size = 5
    cfg.window_step = 2
    processor = cfg.make()
    trial = np.arange(21, dtype=np.float64).reshape(7, 3)
    actual = processor.windows(trial, pad_short=False)
    assert np.array_equal(actual, np.stack([trial[:5].T, trial[2:].T]))
    short = processor.windows(trial[:3], pad_short=True)
    assert np.array_equal(short[0, :, :3], trial[:3].T)
    assert np.array_equal(short[0, :, 3:], np.zeros((3, 2)))
    assert len(processor.windows(trial[:3], pad_short=False)) == 0


def test_subject_split_keeps_every_window_of_a_person_together() -> None:
    trials = [
        Trial(subject="S002", trial_id=0, label=0, windows=torch.zeros(2, 3, 5)),
        Trial(subject="S004", trial_id=1, label=1, windows=torch.ones(4, 3, 5)),
    ]
    cfg = WindowData.Config()
    cfg.validation_subjects = ["S004"]
    cfg.batch_size = 2
    dataset = WindowData(cfg.finalized(), trials=trials)
    assert len(dataset.train_dataset) == 2
    assert len(dataset.eval_dataset) == 4
    assert dataset.train_dataset.tensors[2].tolist() == [0, 0]
    assert dataset.eval_dataset.tensors[2].tolist() == [1, 1, 1, 1]


def test_submission_uses_official_ids_labels_and_fails_on_count(tmp_path: Path) -> None:
    output = tmp_path / "submission.csv"
    write_submission(output, probabilities=[0.2, 0.5], expected_rows=2)
    assert output.read_text() == "ID,TARGET\n0,rest\n1,move\n"
    with pytest.raises(ValueError, match="rows"):
        write_submission(output, probabilities=[0.2], expected_rows=2)
    with pytest.raises(ValueError, match="finite"):
        write_submission(output, probabilities=[float("nan")], expected_rows=1)


@pytest.mark.parametrize("case", ["ramp", "offset"])
def test_filters_replay_complete_published_source_outputs(case: str) -> None:
    record = read_tensors(_CWD / "testdata" / "filtering.pt")
    actual = source_preprocessing().make().filter(record[f"{case}/input"].numpy())
    report = mismatches(
        {"filtered": record[f"{case}/output"]},
        actual={"filtered": torch.from_numpy(actual.copy())},
    )
    assert not report


@pytest.mark.parametrize("samples", [0, 9, 27])
def test_filter_requires_complete_zero_phase_padding(samples: int) -> None:
    with pytest.raises(ValueError, match="requires more than"):
        source_preprocessing().make().filter(np.zeros((samples, 3)))


@pytest.mark.parametrize(
    ("roots", "coefficients"),
    [([], [1 + 0j]), ([2 + 1j, -1 + 2j], [1 + 0j, -1 - 3j, -4 + 3j])],
)
def test_pole_polynomial_is_monic_and_preserves_complex_products(
    roots: list[complex],
    coefficients: list[complex],
) -> None:
    actual = _pole_polynomial(np.asarray(roots, dtype=np.complex128))
    assert actual.dtype == np.complex128
    assert np.array_equal(actual, np.asarray(coefficients))


def test_exp000_design_reproduces_the_x86_reference_coefficients() -> None:
    cfg = source_preprocessing()
    notch_numerator, notch_denominator = _notch(
        cfg.notch_hz,
        quality=cfg.notch_quality,
        sample_rate=cfg.sample_rate,
    )
    zeros, poles, gain = _butterworth_bandpass(
        cfg.band_hz,
        order=cfg.filter_order,
        sample_rate=cfg.sample_rate,
    )
    actual = {
        "notch/numerator": notch_numerator,
        "notch/denominator": notch_denominator,
        "band/numerator": gain * _pole_polynomial(zeros).real,
        "band/denominator": _pole_polynomial(poles).real,
    }
    expected = {
        "notch/numerator": _NOTCH_NUMERATOR,
        "notch/denominator": _NOTCH_DENOMINATOR,
        "band/numerator": _BAND_NUMERATOR,
        "band/denominator": _BAND_DENOMINATOR,
    }
    assert not mismatches(
        {key: _float64(values) for key, values in expected.items()},
        actual={key: torch.from_numpy(value.copy()) for key, value in actual.items()},
        ulps=True,
    )


# SciPy 1.16.3 iirnotch supplied these on aarch64 with NumPy 2.5.3.
@pytest.mark.parametrize(
    ("center_hz", "quality", "sample_rate", "numerator", "denominator"),
    [
        (
            60.0,
            20.0,
            500.0,
            ["0x1.f686c73312e02p-1", "-0x1.6e53771bf8be9p+0", "0x1.f686c73312e02p-1"],
            ["0x1p+0", "-0x1.6e53771bf8be9p+0", "0x1.ed0d8e6625c04p-1"],
        ),
        (
            16.0,
            5.0,
            64.0,
            ["0x1.b9feb0ecefaa2p-1", "-0x1.e78c6da74b4e9p-54", "0x1.b9feb0ecefaa2p-1"],
            ["0x1p+0", "-0x1.e78c6da74b4e9p-54", "0x1.73fd61d9df544p-1"],
        ),
        (
            0.0,
            30.0,
            250.0,
            ["0x1p+0", "-0x1p+1", "0x1p+0"],
            ["0x1p+0", "-0x1p+1", "0x1p+0"],
        ),
        (
            125.0,
            30.0,
            250.0,
            ["0x1.e680e1237202fp-1", "0x1.e680e1237202fp+0", "0x1.e680e1237202fp-1"],
            ["0x1p+0", "0x1.e680e1237202fp+0", "0x1.cd01c246e405ep-1"],
        ),
    ],
)
def test_notch_matches_reference_coefficients_from_zero_through_nyquist(
    center_hz: float,
    quality: float,
    sample_rate: float,
    numerator: list[str],
    denominator: list[str],
) -> None:
    actual = _notch(center_hz, quality=quality, sample_rate=sample_rate)
    assert not mismatches(
        {"numerator": _float64(numerator), "denominator": _float64(denominator)},
        actual={
            "numerator": torch.from_numpy(actual[0]),
            "denominator": torch.from_numpy(actual[1]),
        },
        ulps=True,
    )


@pytest.mark.parametrize("center_hz", [-1.0, 126.0])
def test_notch_rejects_a_center_outside_the_nyquist_band(center_hz: float) -> None:
    with pytest.raises(ValueError, match="Notch frequency"):
        _notch(center_hz, quality=30.0, sample_rate=250.0)


# SciPy 1.16.3 butter(output="zpk") supplied these on aarch64 with NumPy 2.5.3;
# both cases are insensitive to the per-ISA glibc hypot kernel. Order 1 pins the
# real prototype pole's signed zero, which orders its conjugate pair, and the
# np.sqrt center; order 6 pins the fused square and reciprocal angle division.
@pytest.mark.parametrize(
    ("order", "band_hz", "gain", "poles"),
    [
        (
            1,
            (8.0, 20.0),
            "0x1.227014a7a9e9cp-2",
            [
                ("0x1.f6e7ece10f34cp-2", "-0x1.c02843ccfac56p-2"),
                ("0x1.f6e7ece10f34cp-2", "0x1.c02843ccfac56p-2"),
            ],
        ),
        (
            6,
            (3.0, 9.0),
            "0x1.841bb1244b59dp-16",
            [
                ("0x1.ea5a60a568bb7p-1", "-0x1.77cc8f3f55a88p-3"),
                ("0x1.cf1d81c9bfc03p-1", "-0x1.728675df97ab0p-3"),
                ("0x1.abdd80a3e0e70p-1", "-0x1.8e05adbf141a1p-3"),
                ("0x1.abdd80a3e0e70p-1", "0x1.8e05adbf141a1p-3"),
                ("0x1.cf1d81c9bfc03p-1", "0x1.728675df97ab0p-3"),
                ("0x1.ea5a60a568bb7p-1", "0x1.77cc8f3f55a88p-3"),
                ("0x1.95ad3c7fa7c26p-1", "0x1.f6919d92ff177p-2"),
                ("0x1.7b0b947ba37a1p-1", "0x1.817424cf8f6c0p-2"),
                ("0x1.852a810e27129p-1", "0x1.0b22cce823adfp-2"),
                ("0x1.852a810e27129p-1", "-0x1.0b22cce823adfp-2"),
                ("0x1.7b0b947ba37a1p-1", "-0x1.817424cf8f6c0p-2"),
                ("0x1.95ad3c7fa7c26p-1", "-0x1.f6919d92ff177p-2"),
            ],
        ),
    ],
)
def test_butterworth_bandpass_matches_reference_zeros_poles_and_gain(
    order: int,
    band_hz: tuple[float, float],
    gain: str,
    poles: list[tuple[str, str]],
) -> None:
    zeros, actual_poles, actual_gain = _butterworth_bandpass(
        band_hz,
        order=order,
        sample_rate=100.0,
    )
    assert zeros.tolist() == [1] * order + [-1] * order
    assert actual_gain == float.fromhex(gain)
    assert not mismatches(
        {
            "poles": torch.tensor(
                [
                    complex(float.fromhex(real), float.fromhex(imag))
                    for real, imag in poles
                ],
                dtype=torch.complex128,
            ),
        },
        actual={"poles": torch.from_numpy(actual_poles)},
    )


@pytest.mark.parametrize(
    "band_hz",
    [(0.0, 45.0), (1.0, 125.0), (45.0, 45.0), (45.0, 1.0)],
)
def test_butterworth_bandpass_rejects_edges_outside_the_open_band(
    band_hz: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="Band edges"):
        _butterworth_bandpass(band_hz, order=4, sample_rate=250.0)


def test_product_rounds_each_part_once() -> None:
    assert _product(1 + 2j, 3 + 4j) == -5 + 10j
    # Two roundings would lose the 2**-60 term of (1 + 2**-30)**2 - 1.
    square = _product(complex(1 + 2**-30, 1), complex(1 + 2**-30, 1))
    assert square == complex(2**-29 + 2**-60, 2 + 2**-29)


@pytest.mark.parametrize(
    ("dividend", "divisor"),
    [(10 + 5j, 2 + 1j), (5 + 10j, 1 + 2j)],
)
def test_quotient_scales_by_the_larger_divisor_part(
    dividend: complex,
    divisor: complex,
) -> None:
    assert _quotient(dividend, divisor) == 5 + 0j


@pytest.mark.parametrize(
    ("value", "root"),
    [
        (3 + 4j, 2 + 1j),
        (-3 + 4j, 1 + 2j),
        (-3 - 4j, 1 - 2j),
        (complex(-4, -0.0), complex(0, -2)),
        (6j, complex(_SQRT3, _SQRT3)),
    ],
)
def test_square_root_follows_csqrt_in_every_half_plane(
    value: complex,
    root: complex,
) -> None:
    assert np.complex128(_square_root(value)).tobytes() == np.complex128(root).tobytes()


@pytest.mark.parametrize(
    ("x", "y", "norm"),
    [(0.34, -0.35, "0x1.f3aa7417b6f8fp-2"), (-0.46, 0.09, "0x1.dff8928653ed5p-2")],
)
def test_hypot_corrects_the_root_as_x86_glibc_does(
    x: float,
    y: float,
    norm: str,
) -> None:
    # The generic glibc 2.39 kernel rounds each one ULP from the exact norm;
    # aarch64's fused kernel and the uncorrected root round them differently.
    assert _hypot(x, y) == float.fromhex(norm)


def test_complex_dot_sums_four_real_partials_before_combining() -> None:
    # Summing each complex product first would keep the 2**-53 real term.
    assert _complex_dot([(1 + 1j, 1 + 1j), (complex(2**-53), 1 + 0j)]) == 2j


def test_csv_trials_follow_adjacent_nonempty_markers(tmp_path: Path) -> None:
    cfg = source_preprocessing()
    cfg.channels = ("Fz", "C3", "Cz")
    cfg.window_size = 5
    cfg.window_step = 2
    values = np.arange(0, 300, 3, dtype=np.float64)[:, None] + np.asarray(
        [277_867.16, 251_519.55, 287_807.6],
    )
    markers = {
        10: "rest",
        11: "NA",
        12: "null",
        20: "cue_stop",
        30: "move",
        31: "NaN",
        40: "cue_stop",
        99: "move",
    }
    directory = tmp_path / "Training set"
    directory.mkdir()
    path = directory / "S002-001_eeg.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([*cfg.channels, "Marker_val"])
        writer.writerows(
            [*row, markers.get(index, "")]
            for index, row in enumerate(
                from_plain(cast(object, values.tolist()), list[list[float]]),
            )
        )
    processor = cfg.make()
    trials = processor.read([path], labelled=True)
    assert [trial.label for trial in trials] == [0, 1]
    assert [trial.trial_id for trial in trials] == [0, 1]
    assert [trial.subject for trial in trials] == ["S002", "S002"]
    assert trials[0].windows.shape == (3, 3, 5)
    filtered = processor.filter(values)
    expected = processor.windows(
        processor.standardize(filtered[10:20]),
        pad_short=False,
    )
    assert torch.equal(trials[0].windows, torch.from_numpy(expected))
    record = read_tensors(_CWD / "testdata" / "ingestion.pt")
    data_cfg = WindowData.Config()
    data_cfg.base_dir = None
    data_cfg.working_dir = tmp_path
    data_cfg.preprocessing = cfg
    media, labels, identifiers = data_cfg.make().train_dataset.tensors
    assert torch.equal(media, record["media"])
    assert torch.equal(labels, record["label"])
    assert torch.equal(identifiers, record["trial"])
    assert torch.equal(torch.from_numpy(values), record["raw"])
    assert torch.equal(record["triggers"], torch.tensor([10, 20, 30, 40, 99]))


def test_csv_inference_pads_short_cues_and_rejects_missing_cues(tmp_path: Path) -> None:
    cfg = source_preprocessing()
    cfg.channels = ("Fz", "C3", "Cz")
    cfg.window_size = 5
    values = np.arange(300, dtype=np.float64).reshape(100, 3)
    markers = {10: "cue_start", 13: "cue_stop"}
    path = tmp_path / "S008_test_001_eeg.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([*cfg.channels, "Marker_val"])
        writer.writerows(
            [*row, markers.get(index, "")]
            for index, row in enumerate(
                from_plain(cast(object, values.tolist()), list[list[float]]),
            )
        )
    processor = cfg.make()
    trials = processor.read([path], labelled=False)
    assert len(trials) == 1
    assert trials[0].label == -1
    assert trials[0].windows.shape == (1, 3, 5)
    with pytest.raises(ValueError, match="No EEG cue"):
        processor.read([path], labelled=True)
    with pytest.raises(ValueError, match="No EEG CSV"):
        processor.read([], labelled=True)


def test_loaders_and_checkpoint_contract() -> None:
    cfg = WindowData.Config()
    trials = [
        Trial(subject="S002", trial_id=0, label=1, windows=torch.zeros(5, 3, 7)),
    ]
    cfg.batch_size = 2
    dataset = WindowData(cfg.finalized(), trials=trials)
    loader = dataset.train_dataloader()
    assert len(loader) == 2
    assert [batch["label"].tolist() for batch in loader] == [[1, 1], [1, 1]]
    assert len(dataset.eval_dataloader()) == 0
    assert list(dataset.eval_dataloader()) == []
    assert len(dataset.eval_dataset.tensors) == 3
    assert dataset.eval_dataset.tensors[0].shape == (0, 1, 3, 7)
    assert dataset.eval_dataset.tensors[0].dtype == torch.float32
    assert dataset.eval_dataset.tensors[1].shape == (0,)
    assert dataset.eval_dataset.tensors[1].dtype == torch.long
    assert dataset.eval_dataset.tensors[2].shape == (0,)
    assert dataset.eval_dataset.tensors[2].dtype == torch.long
    assert "timer_epoch" in dataset.state_dict()
    with pytest.raises(
        NotImplementedError,
        match=r"^The public reference does not support exact resume\.$",
    ):
        dataset.load_state_dict(dataset.state_dict())
    cfg.batch_size = 6
    with pytest.raises(ValueError, match="fill one"):
        WindowData(cfg.finalized(), trials=trials).train_dataloader()


def test_dataset_preserves_every_value_axis_label_id_and_eval_batch() -> None:
    first = torch.arange(4 * 3 * 7, dtype=torch.float64).reshape(4, 3, 7)
    second = torch.arange(5 * 3 * 7, dtype=torch.float64).reshape(5, 3, 7) + 100
    cfg = WindowData.Config()
    trials = [
        Trial(subject="S002", trial_id=6, label=0, windows=first),
        Trial(subject="S003", trial_id=8, label=1, windows=second),
    ]
    cfg.batch_size = 3
    dataset = WindowData(cfg.finalized(), trials=trials)
    media, label, identifiers = dataset.train_dataset.tensors
    assert len(dataset.train_dataset.tensors) == 3
    assert torch.equal(media, torch.cat([first, second])[:, None])
    assert media.dtype == torch.float64
    assert torch.equal(label, torch.tensor([0] * 4 + [1] * 5))
    assert label.dtype == torch.long
    assert torch.equal(identifiers, torch.tensor([6] * 4 + [8] * 5))
    assert identifiers.dtype == torch.long
    cfg.validation_subjects = ["S003"]
    validation = WindowData(cfg.finalized(), trials=trials).eval_dataloader()
    assert len(validation) == 2
    batches = list(validation)
    assert [len(batch["media"]) for batch in batches] == [3, 2]
    assert [int(batch["valid_count"]) for batch in batches] == [3, 2]
    assert torch.equal(
        torch.cat([batch["media"] for batch in batches]),
        second[:, None],
    )
    assert torch.equal(
        torch.cat([batch["label"] for batch in batches]),
        torch.ones(5, dtype=torch.long),
    )
    assert torch.equal(
        torch.cat([batch["trial"] for batch in batches]),
        torch.full((5,), 8, dtype=torch.long),
    )


@pytest.mark.parametrize("validation", [[], ["S004"]])
def test_cpu_dataset_storage_is_independent_of_the_ambient_device(
    validation: list[str],
) -> None:
    cfg = WindowData.Config()
    cfg.validation_subjects = validation
    trials = [
        Trial(
            subject="S002",
            trial_id=2,
            label=0,
            windows=torch.arange(42).reshape(2, 3, 7),
        ),
        Trial(
            subject="S004",
            trial_id=3,
            label=1,
            windows=torch.arange(84).reshape(4, 3, 7),
        ),
    ]
    with torch.device("meta"):
        dataset = WindowData(cfg.finalized(), trials=trials)
    for split in (dataset.train_dataset, dataset.eval_dataset):
        assert all(tensor.device.type == "cpu" for tensor in split.tensors)
    assert dataset.train_dataset.tensors[1].tolist() == [0, 0] + (
        [] if validation else [1, 1, 1, 1]
    )


def test_cpu_storage_rejects_a_trial_without_materialized_values() -> None:
    cfg = WindowData.Config()
    trials = [
        Trial(
            subject="S002",
            trial_id=2,
            label=0,
            windows=torch.empty(2, 3, 7, device="meta"),
        ),
    ]
    with pytest.raises(NotImplementedError, match="Cannot copy out of meta"):
        WindowData(cfg.finalized(), trials=trials)


@pytest.mark.parametrize("field", ["window_size", "window_step"])
def test_window_geometry_must_advance(field: str) -> None:
    cfg = Preprocessing.Config()
    setattr(cfg, field, 0)
    with pytest.raises(ValueError, match="positive"):
        cfg.make()


def test_causal_filter_preserves_row_major_reduction_order() -> None:
    values = np.arange(30, dtype=np.float64).reshape(10, 3) / 7
    actual = _causal_filter(
        values,
        b=np.array([0.2, 0.31, 0.47]),
        a=np.array([1.0, 0.23, -0.17]),
        initial=np.zeros((2, 3), dtype=np.float64),
    )
    assert actual.shape == values.shape
    assert actual.flags.c_contiguous


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (
            _NOTCH_NUMERATOR,
            _NOTCH_DENOMINATOR,
            ["0x1.502789a5b05e7p-6", "0x1.502789a5b05d4p-6"],
        ),
        (
            _BAND_NUMERATOR,
            _BAND_DENOMINATOR,
            [
                "-0x1.fca3a66af54f2p-6",
                "-0x1.fca3a6b5fd347p-6",
                "0x1.7d7abcf5b43b2p-4",
                "0x1.7d7abcc0922cbp-4",
                "-0x1.7d7abcc9fe255p-4",
                "-0x1.7d7abce25ac5bp-4",
                "0x1.fca3a680317dap-6",
                "0x1.fca3a6790b26bp-6",
            ],
        ),
    ],
)
def test_filter_initial_state_preserves_complete_captured_solve(
    numerator: list[str],
    denominator: list[str],
    expected: list[str],
) -> None:
    # These complete vectors retain the unchanged scientific source's float64 solve.
    actual = _filter_initial_state(
        np.asarray([float.fromhex(value) for value in numerator]),
        a=np.asarray([float.fromhex(value) for value in denominator]),
    )
    assert not mismatches(
        {
            "state": torch.tensor(
                [float.fromhex(value) for value in expected],
                dtype=torch.float64,
            ),
        },
        actual={"state": torch.from_numpy(actual)},
    )


def test_filter_initial_state_rejects_a_singular_steady_state() -> None:
    with pytest.raises(np.linalg.LinAlgError, match="Singular"):
        _filter_initial_state(
            np.asarray([0.2, 0.3, 0.4]),
            a=np.asarray([1.0, -1.0, 0.0]),
        )


def _float64(values: list[str]) -> torch.Tensor:
    return torch.tensor([float.fromhex(value) for value in values], dtype=torch.float64)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
