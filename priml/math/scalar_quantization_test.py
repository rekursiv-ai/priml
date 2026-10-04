"""Tests for Lloyd-Max fitting and nearest-level coding."""

from __future__ import annotations

import math

from torch import Tensor

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("deterministic", [False, True])
def test_cuda_sample_can_be_fitted_and_coded_with_cpu_initial_levels(
    deterministic: bool,
) -> None:
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(deterministic)
        values = torch.randn(512, device="cuda")
        initial = high_resolution_levels(values, num_levels=8)
        assert initial.device == values.device
        levels = lloyd_max(values, init=initial.cpu())
        assert levels.device == values.device
        indices = quantize(values[None], thresholds=midpoints(levels.cpu())[None])
        assert dequantize(indices, levels=levels.cpu()[None]).device == values.device
        assert torch.are_deterministic_algorithms_enabled() == deterministic
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


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
    assert _snr_db(points, levels=gaussian_levels(256)) == pytest.approx(
        predicted,
        abs=0.1,
    )


def _snr_db(values: Tensor, levels: Tensor) -> float:
    decoded = levels[torch.searchsorted(midpoints(levels), values)]
    noise = float((decoded - values).pow(2).mean())
    return 10 * math.log10(float(values.var()) / noise)


def test_uniform_source_gets_uniform_levels() -> None:
    values = (torch.arange(4096, dtype=torch.float64) + 0.5) / 4096
    levels = lloyd_max(values, init=torch.tensor([0.1, 0.2, 0.3, 0.9]))
    assert levels.tolist() == pytest.approx([0.125, 0.375, 0.625, 0.875], abs=1e-3)


def test_fitting_never_raises_the_distortion() -> None:
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(4096, generator=generator, dtype=torch.float64)
    start = high_resolution_levels(values, num_levels=16)
    assert _snr_db(values, levels=lloyd_max(values, init=start)) >= _snr_db(
        values,
        levels=start,
    )


def test_empty_cells_are_moved_into_the_data() -> None:
    """Levels starting outside the sample end up coding part of it."""
    values = (torch.arange(1000, dtype=torch.float64) + 0.5) / 1000
    levels = lloyd_max(values, init=torch.tensor([-10.0, -9.0, 0.5, 20.0]))
    counts = torch.bincount(
        torch.searchsorted(midpoints(levels), values),
        minlength=4,
    )
    assert bool((counts > 0).all())
    assert bool((levels[1:] >= levels[:-1]).all())


def test_fewer_distinct_values_than_levels_returns_the_values() -> None:
    values = torch.tensor([3.0, 1.0, 2.0, 1.0, 3.0])
    assert lloyd_max(values, init=torch.zeros(4)).tolist() == [1.0, 2.0, 3.0, 3.0]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_samples_are_refused(bad: float) -> None:
    """Many distinct values, so the refusal is not the few-values early return."""
    values = torch.cat([torch.arange(100, dtype=torch.float64), torch.tensor([bad])])
    with pytest.raises(ValueError, match="finite"):
        _ = lloyd_max(values, init=torch.linspace(0, 99, 4))
    with pytest.raises(ValueError, match="finite"):
        _ = high_resolution_levels(values, num_levels=4)


def test_non_finite_initial_levels_are_refused() -> None:
    values = torch.arange(100, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite initial"):
        _ = lloyd_max(values, init=torch.tensor([0.0, math.nan, 50.0, 99.0]))


def test_midpoints_of_finite_levels_never_overflow() -> None:
    levels = torch.tensor([[40_000.0, 50_000.0]], dtype=torch.float16)
    thresholds = midpoints(levels)
    assert bool(thresholds.isfinite().all())
    values = torch.tensor([[41_000.0, 49_000.0]], dtype=torch.float16)
    assert quantize(values, thresholds=thresholds).tolist() == [[0, 1]]


def test_high_resolution_start_puts_levels_inside_the_sample() -> None:
    generator = torch.Generator().manual_seed(1)
    values = torch.randn(10_000, generator=generator, dtype=torch.float64)
    levels = high_resolution_levels(values, num_levels=256)
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
    levels = high_resolution_levels(values, num_levels=64)
    expected = 3**0.5 * torch.special.ndtri(torch.tensor(48.5 / 64)).item()
    assert levels[48].item() == pytest.approx(expected, abs=0.03)


def test_constant_sample_places_every_level_on_the_constant() -> None:
    assert high_resolution_levels(torch.full((8,), 2.5), num_levels=3).tolist() == [
        2.5,
        2.5,
        2.5,
    ]


def test_ties_take_the_lower_cell_and_outliers_saturate() -> None:
    thresholds = torch.tensor([[0.0, 1.0]])
    values = torch.tensor([[-5.0, 0.0, 0.5, 1.0, 1.5, 9.0]])
    assert quantize(values, thresholds=thresholds).tolist() == [[0, 0, 1, 1, 2, 2]]


def test_quantize_compiles_without_reading_a_python_scalar() -> None:
    values = torch.tensor([[-1.0, 0.0, 1.0]])
    thresholds = torch.tensor([[0.0]])
    compiled = torch.compile(quantize, backend="eager", fullgraph=True)
    assert torch.equal(
        compiled(values, thresholds),
        quantize(values, thresholds=thresholds),
    )


def test_quantize_and_dequantize_apply_one_table_per_row() -> None:
    levels = torch.tensor([[0.0, 1.0, 2.0], [10.0, 20.0, 30.0]])
    values = torch.tensor([[0.2, 1.9, 1.4], [11.0, 26.0, 99.0]])
    indices = quantize(values, thresholds=midpoints(levels))
    assert indices.tolist() == [[0, 2, 1], [0, 2, 2]]
    assert dequantize(indices, levels=levels).tolist() == [
        [0.0, 2.0, 1.0],
        [10.0, 30.0, 30.0],
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
