"""Reference peak selection and the nine-argument FPNet packing."""

from __future__ import annotations

import numpy as np
import torch

from priml.baselines.casmi2026.peaks import pack_inputs, prepare_peaks


def test_reference_peak_selection_and_degenerate_spectra() -> None:
    masses, values = prepare_peaks(
        np.array([20, 10, -1, 200, 30], dtype=np.float32),
        np.array([4, 1, 20, 40, 0], dtype=np.float32),
        precursor=50,
        max_peaks=2,
        floor=0.001,
    )
    assert masses.tolist() == [10, 20]
    assert values.tolist() == [0.5, 1.0]
    selected = prepare_peaks(
        np.arange(5, dtype=np.float32) + 10,
        np.arange(5, dtype=np.float32),
        precursor=50,
        max_peaks=2,
        floor=0.001,
    )
    assert selected[0].tolist() == [13, 14]
    for mz, intensity in [
        (list[int](), list[int]()),
        ([0, 200], [1, 2]),
        ([10, 20], [0, 0]),
    ]:
        result = prepare_peaks(
            np.array(mz, dtype=np.float32),
            np.array(intensity, dtype=np.float32),
            precursor=50,
            max_peaks=2,
            floor=0.001,
        )
        assert result[0].size == result[1].size == 0


def test_packing_pads_peaks_and_keeps_one_slot_for_an_empty_spectrum() -> None:
    inputs = pack_inputs(
        [np.array([10, 20], np.float32), np.zeros(0, np.float32)],
        [np.array([0.5, 1], np.float32), np.zeros(0, np.float32)],
        precursor=np.array([50, 60], np.float32),
        adduct=np.array([3, 4], np.int64),
        energy=np.array([10, 0], np.float32),
        energy_known=np.array([1, 0], np.float32),
        merged_count=np.array([8, 2], np.float32),
        mode=np.array([1, -1], np.float32),
    )
    assert list(inputs) == [
        "mz",
        "it",
        "pad",
        "prec",
        "ad",
        "ce",
        "ce_known",
        "merged",
        "mode",
    ]
    assert inputs["mz"].tolist() == [[10, 20], [0, 0]]
    assert inputs["pad"].tolist() == [[False, False], [False, True]]
    assert torch.equal(inputs["merged"], torch.tensor([2.0, 0.5]))
    assert inputs["ad"].dtype == torch.int64


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
