"""Trial alignment, normalization, grouping, and submission contracts."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, cast

import csv
import math

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
    _filter_initial_state,
    _notch,
    _zero_phase_filter,
    subject_id,
    write_submission,
)
from priml.baselines.motor_imagery.testing import (
    source_preprocessing,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray


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


def test_exp000_filter_composes_notch_then_bandpass_and_removes_dc() -> None:
    cfg = source_preprocessing()
    values = np.arange(0, 300, 3, dtype=np.float64)[:, None] + np.asarray(
        [277_867.16, 251_519.55, 287_807.6],
    )
    actual = cfg.make().filter(values)
    notch = _notch(cfg.notch_hz, quality=cfg.notch_quality, sample_rate=cfg.sample_rate)
    band = _butterworth_bandpass(
        cfg.band_hz,
        order=cfg.filter_order,
        sample_rate=cfg.sample_rate,
    )
    notched = _zero_phase_filter(values, b=notch[0], a=notch[1])
    assert np.array_equal(actual, _zero_phase_filter(notched, b=band[0], a=band[1]))
    assert actual.shape == values.shape
    # The bandpass zero at z = 1 removes the 2.8e5 offset; the ramp is in band.
    assert np.abs(actual).max() < 20
    constant = cfg.make().filter(np.full((100, 3), 2.8e5))
    assert np.abs(constant).max() < 2e-5


@pytest.mark.parametrize("samples", [0, 9, 27])
def test_filter_requires_complete_zero_phase_padding(samples: int) -> None:
    with pytest.raises(ValueError, match="requires more than"):
        source_preprocessing().make().filter(np.zeros((samples, 3)))


def test_exp000_bandpass_numerator_is_a_scaled_quartic() -> None:
    cfg = source_preprocessing()
    band = _butterworth_bandpass(
        cfg.band_hz,
        order=cfg.filter_order,
        sample_rate=cfg.sample_rate,
    )
    # Zeros at z = 1 and z = -1 make the numerator a scaled (z**2 - 1)**4.
    assert np.array_equal(
        band[0],
        band[0][0] * np.asarray([1, 0, -4, 0, 6, 0, -4, 0, 1]),
    )


@pytest.mark.parametrize(
    ("center_hz", "quality", "sample_rate"),
    [(50.0, 30.0, 250.0), (60.0, 20.0, 500.0), (16.0, 5.0, 64.0)],
)
def test_notch_zeroes_its_center_and_passes_dc_and_nyquist(
    center_hz: float,
    quality: float,
    sample_rate: float,
) -> None:
    numerator, denominator = _notch(
        center_hz,
        quality=quality,
        sample_rate=sample_rate,
    )
    assert numerator[0] == numerator[2]
    assert denominator[0] == 1
    assert denominator[1] == numerator[1]
    assert denominator[2] == 2 * numerator[0] - 1
    assert (
        abs(
            _response(
                numerator,
                denominator=denominator,
                hz=center_hz,
                rate=sample_rate,
            ),
        )
        < 5e-14
    )
    for hz in (0.0, sample_rate / 2):
        gain = abs(
            _response(numerator, denominator=denominator, hz=hz, rate=sample_rate),
        )
        assert abs(gain - 1) < 1e-15


def test_notch_at_zero_and_nyquist_is_a_double_real_zero() -> None:
    zero = _notch(0.0, quality=30.0, sample_rate=250.0)
    assert [values.tolist() for values in zero] == [[1, -2, 1], [1, -2, 1]]
    numerator, denominator = _notch(125.0, quality=30.0, sample_rate=250.0)
    assert numerator[1] == denominator[1] == 2 * numerator[0]
    assert denominator[2] == 2 * numerator[0] - 1


@pytest.mark.parametrize("center_hz", [-1.0, 126.0])
def test_notch_rejects_a_center_outside_the_nyquist_band(center_hz: float) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Notch frequency must lie between zero and Nyquist\.$",
    ):
        _notch(center_hz, quality=30.0, sample_rate=250.0)


@pytest.mark.parametrize(
    ("order", "band_hz", "sample_rate"),
    [(1, (8.0, 20.0), 100.0), (6, (10.0, 30.0), 100.0), (4, (1.0, 45.0), 250.0)],
)
def test_butterworth_bandpass_halves_power_at_both_edges(
    order: int,
    band_hz: tuple[float, float],
    sample_rate: float,
) -> None:
    numerator, denominator = _butterworth_bandpass(
        band_hz,
        order=order,
        sample_rate=sample_rate,
    )
    assert len(numerator) == len(denominator) == 2 * order + 1
    assert denominator[0] == 1
    for hz in band_hz:
        power = (
            abs(_response(numerator, denominator=denominator, hz=hz, rate=sample_rate))
            ** 2
        )
        assert abs(power - 0.5) < 1e-6
    # The band's geometric center in the prewarped domain has unit gain.
    low, high = (4 * math.tan(math.pi * hz / sample_rate) for hz in band_hz)
    center = 2 * math.atan((low * high) ** 0.5 / 4) * sample_rate / (2 * math.pi)
    peak = abs(
        _response(numerator, denominator=denominator, hz=center, rate=sample_rate),
    )
    assert abs(peak - 1) < 1e-6
    for hz in (0.0, sample_rate / 2):
        assert (
            abs(_response(numerator, denominator=denominator, hz=hz, rate=sample_rate))
            < 1e-9
        )


@pytest.mark.parametrize(
    "band_hz",
    [(0.0, 45.0), (1.0, 125.0), (45.0, 45.0), (45.0, 1.0)],
)
def test_butterworth_bandpass_rejects_edges_outside_the_open_band(
    band_hz: tuple[float, float],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Band edges must increase strictly between zero and Nyquist\.$",
    ):
        _butterworth_bandpass(band_hz, order=4, sample_rate=250.0)


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
    expected = [
        processor.windows(processor.standardize(filtered[start:stop]), pad_short=False)
        for start, stop in ((10, 20), (30, 40))
    ]
    assert torch.equal(trials[0].windows, torch.from_numpy(expected[0]))
    data_cfg = WindowData.Config()
    data_cfg.base_dir = None
    data_cfg.working_dir = tmp_path
    data_cfg.preprocessing = cfg
    media, labels, identifiers = data_cfg.make().train_dataset.tensors
    assert torch.equal(media, torch.from_numpy(np.concatenate(expected))[:, None])
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]
    assert identifiers.tolist() == [0, 0, 0, 1, 1, 1]


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


def test_causal_filter_runs_the_transposed_direct_form_recurrence() -> None:
    impulse = np.asarray([[1.0], [0.0], [0.0], [0.0]])
    # Normalized by a[0]: y[n] = x[n] + x[n - 1] + y[n - 1] / 2, so the tail halves.
    actual = _causal_filter(
        impulse,
        b=np.asarray([2.0, 2.0]),
        a=np.asarray([2.0, -1.0]),
        initial=np.zeros((1, 1)),
    )
    assert actual[:, 0].tolist() == [1.0, 1.5, 0.75, 0.375]
    held = _causal_filter(
        np.full((3, 2), 1.0),
        b=np.asarray([1.0, 1.0]),
        a=np.asarray([1.0, -0.5]),
        initial=np.asarray([[3.0, 3.0]]),
    )
    assert held.tolist() == [[4.0, 4.0]] * 3


def test_zero_phase_smoothing_returns_a_ramp_unchanged() -> None:
    # The reverse pass undoes the FIR's one-sample delay; odd padding extends the line.
    ramp = np.arange(40, dtype=np.float64).reshape(20, 2)
    actual = _zero_phase_filter(
        ramp,
        b=np.asarray([0.25, 0.5, 0.25]),
        a=np.asarray([1.0, 0.0, 0.0]),
    )
    assert np.array_equal(actual, ramp)


def test_zero_phase_filter_starts_both_passes_from_their_edge_samples() -> None:
    # Dyadic coefficients keep every step exact, so these bits are host-independent.
    actual = _zero_phase_filter(
        (np.arange(8, dtype=np.float64) ** 2)[:, None],
        b=np.asarray([0.25, 0.25]),
        a=np.asarray([1.0, -0.5]),
    )
    expected = [
        "0x1.1f44775ap-3",
        "0x1.a2288eeb4p+1",
        "0x1.d1b88eeb4p+2",
        "0x1.959c8eeb4p+3",
        "0x1.3a958eeb4p+4",
        "0x1.c3a79dd68p+4",
        "0x1.3146bdd68p+5",
        "0x1.875d0badp+5",
    ]
    assert actual[:, 0].tolist() == [float.fromhex(value) for value in expected]


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        ([1.0, 1.0], [1.0, -0.5], [3.0]),
        ([0.5, 0.5], [2.0, -1.0], [0.75]),
        ([0.25, 0.5, 0.25], [1.0, 0.0, 0.0], [0.75, 0.25]),
    ],
)
def test_filter_initial_state_is_the_unit_step_steady_state(
    numerator: list[float],
    denominator: list[float],
    expected: list[float],
) -> None:
    actual = _filter_initial_state(np.asarray(numerator), a=np.asarray(denominator))
    assert actual.tolist() == expected


@pytest.mark.parametrize(
    ("band_hz", "order", "sample_rate"),
    [((1.0, 45.0), 4, 250.0), ((0.5, 40.0), 6, 250.0), ((1.0, 400.0), 8, 1000.0)],
)
def test_filter_initial_state_rounds_to_the_exact_steady_state(
    band_hz: tuple[float, float],
    order: int,
    sample_rate: float,
) -> None:
    # Poles near z = 1 make a linear solve miss every float32 element of the last two.
    numerator, denominator = _butterworth_bandpass(
        band_hz,
        order=order,
        sample_rate=sample_rate,
    )
    b = [Fraction(float(value)) for value in numerator]
    a = [Fraction(float(value)) for value in denominator]
    steady = sum(b, start=Fraction(0)) / sum(a, start=Fraction(0))
    residual = [x - y * steady for x, y in zip(b[1:], a[1:], strict=True)]
    expected = [float(sum(residual[index:])) for index in range(len(residual))]
    actual = _filter_initial_state(numerator, a=denominator)
    assert np.array_equal(
        actual.astype(np.float32),
        np.asarray(expected, dtype=np.float32),
    )


@pytest.mark.parametrize(("design", "gain"), [("notch", 1.0), ("band", 0.0)])
def test_exp000_initial_state_holds_a_constant_offset(design: str, gain: float) -> None:
    cfg = source_preprocessing()
    numerator, denominator = (
        _notch(cfg.notch_hz, quality=cfg.notch_quality, sample_rate=cfg.sample_rate)
        if design == "notch"
        else _butterworth_bandpass(
            cfg.band_hz,
            order=cfg.filter_order,
            sample_rate=cfg.sample_rate,
        )
    )
    offset = np.full((50, 2), 2.8e5)
    held = _causal_filter(
        offset,
        b=numerator,
        a=denominator,
        initial=_filter_initial_state(numerator, a=denominator)[:, None] * offset[0],
    )
    assert np.abs(held - gain * offset).max() < 5e-4


def test_filter_initial_state_rejects_a_pole_at_dc() -> None:
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        _filter_initial_state(
            np.asarray([0.2, 0.3, 0.4]),
            a=np.asarray([1.0, -1.0, 0.0]),
        )


def _response(
    numerator: NDArray[np.float64],
    *,
    denominator: NDArray[np.float64],
    hz: float,
    rate: float,
) -> complex:
    z = np.exp(1j * np.pi * hz / (rate / 2))
    return complex(np.polyval(numerator, z) / np.polyval(denominator, z))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
