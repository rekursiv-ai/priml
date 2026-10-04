"""Tests for Lloyd-Max fitting and nearest-level coding."""

from __future__ import annotations

import math

import pytest
import torch

from priml.math.scalar_quantization import (
    dequantize,
    gaussian_levels,
    high_resolution_levels,
    lloyd_max,
    midpoints,
    quantize,
)


def _snr_db(values: torch.Tensor, levels: torch.Tensor) -> float:
    decoded = levels[torch.searchsorted(midpoints(levels), values)]
    noise = float((decoded - values).pow(2).mean())
    return 10 * math.log10(float(values.var()) / noise)


def test_gaussian_levels_reproduce_max_1960_table_for_four_levels() -> None:
    """Max (1960), Table I, N = 4: outputs 0.4528, 1.510; boundary 0.9816."""
    levels = gaussian_levels(4, num_points=1 << 16)
    assert levels.tolist() == pytest.approx([-1.510, -0.4528, 0.4528, 1.510], abs=5e-4)
    assert midpoints(levels)[2].item() == pytest.approx(0.9816, abs=5e-4)


def test_gaussian_levels_reproduce_max_1960_table_for_eight_levels() -> None:
    """Max (1960), Table I, N = 8: outputs 0.2451, 0.7560, 1.344, 2.152."""
    levels = gaussian_levels(8, num_points=1 << 16)
    assert levels[4:].tolist() == pytest.approx(
        [0.2451, 0.7560, 1.344, 2.152],
        abs=5e-4,
    )
    assert midpoints(levels)[4:].tolist() == pytest.approx(
        [0.5006, 1.050, 1.748],
        abs=5e-4,
    )


def test_gaussian_levels_approach_the_panter_dite_distortion() -> None:
    """At 256 levels the fit sits within 0.1 dB of ``(6 sqrt(3) pi / 12) / N**2``.

    Measured on a quantile grid OFFSET from the one fitted, so the check is not
    in-sample: scored on its own 2**16 fitting points the fit reads 0.14 dB
    better than the asymptote, which is overfitting, not accuracy.
    """
    count = 1 << 18
    points = torch.special.ndtri(
        (torch.arange(count, dtype=torch.float64) + 0.25) / count,
    )
    predicted = -10 * math.log10(6 * 3**0.5 * math.pi / 12 / 256**2)
    assert _snr_db(points, gaussian_levels(256)) == pytest.approx(predicted, abs=0.1)


def test_uniform_source_gets_uniform_levels() -> None:
    values = (torch.arange(4096, dtype=torch.float64) + 0.5) / 4096
    levels = lloyd_max(values, torch.tensor([0.1, 0.2, 0.3, 0.9]))
    assert levels.tolist() == pytest.approx([0.125, 0.375, 0.625, 0.875], abs=1e-3)


def test_fitting_never_raises_the_distortion() -> None:
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(4096, generator=generator, dtype=torch.float64)
    start = high_resolution_levels(values, 16)
    assert _snr_db(values, lloyd_max(values, start)) >= _snr_db(values, start)


def test_empty_cells_are_moved_into_the_data() -> None:
    """Levels starting outside the sample end up coding part of it."""
    values = (torch.arange(1000, dtype=torch.float64) + 0.5) / 1000
    levels = lloyd_max(values, torch.tensor([-10.0, -9.0, 0.5, 20.0]))
    counts = torch.bincount(
        torch.searchsorted(midpoints(levels), values),
        minlength=4,
    )
    assert bool((counts > 0).all())
    assert bool((levels[1:] >= levels[:-1]).all())


def test_fewer_distinct_values_than_levels_returns_the_values() -> None:
    values = torch.tensor([3.0, 1.0, 2.0, 1.0, 3.0])
    assert lloyd_max(values, torch.zeros(4)).tolist() == [1.0, 2.0, 3.0, 3.0]


def test_rejects_non_finite_samples() -> None:
    with pytest.raises(ValueError, match="finite"):
        _ = lloyd_max(torch.tensor([0.0, math.nan]), torch.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        _ = high_resolution_levels(torch.tensor([0.0, math.inf]), 2)


def test_high_resolution_start_puts_levels_inside_the_sample() -> None:
    generator = torch.Generator().manual_seed(1)
    values = torch.randn(10_000, generator=generator, dtype=torch.float64)
    levels = high_resolution_levels(values, 256)
    assert levels.min() >= values.min()
    assert levels.max() <= values.max()
    assert bool((levels[1:] >= levels[:-1]).all())


def test_high_resolution_start_follows_the_cube_root_density() -> None:
    """For N(0, 1), ``p ** (1 / 3)`` is N(0, 3): level k sits at ``sqrt(3) ndtri(t)``.

    Level 48 of 64 lands near 1.211; the equal-probability rule ``p`` would put
    it at 0.699, so a start that stopped taking the cube root fails here.
    """
    generator = torch.Generator().manual_seed(2)
    values = torch.randn(200_000, generator=generator, dtype=torch.float64)
    levels = high_resolution_levels(values, 64)
    expected = 3**0.5 * torch.special.ndtri(torch.tensor(48.5 / 64)).item()
    assert levels[48].item() == pytest.approx(expected, abs=0.03)


def test_constant_sample_places_every_level_on_the_constant() -> None:
    assert high_resolution_levels(torch.full((8,), 2.5), 3).tolist() == [2.5, 2.5, 2.5]


def test_ties_take_the_lower_cell_and_outliers_saturate() -> None:
    thresholds = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[-5.0, 0.0, 0.5, 1.0, 1.5, 9.0]])
    assert quantize(values, thresholds).tolist() == [[0, 0, 1, 1, 2, 2]]


def test_quantize_and_dequantize_apply_one_table_per_row() -> None:
    levels = torch.tensor([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]])
    values = torch.tensor([[0.2, 1.9, 1.4], [11.0, 26.0, 99.0]])
    indices = quantize(values, midpoints(levels))
    assert indices.tolist() == [[0, 2, 1], [0, 2, 2]]
    assert dequantize(indices, levels).tolist() == [[0.0, 2.0, 1.0], [10.0, 30.0, 30.0]]


def test_quantize_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        _ = quantize(torch.tensor([[math.nan]]), torch.tensor([[0.0]]))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
