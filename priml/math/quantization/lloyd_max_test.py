"""Tests for Lloyd-Max fitting and its starting levels."""

from __future__ import annotations

import math

from torch import Tensor

import pytest
import torch

from priml.math.quantization.lloyd_max import (
    gaussian_levels,
    high_resolution_levels,
    lloyd_max,
)
from priml.math.quantization.nearest import dequantize, midpoints, quantize


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


def test_a_sample_far_from_zero_fits_as_its_shift_does() -> None:
    """Prefix sums of raw values cancel in float64 far from zero.

    Integers on both sides of the shift, so values and starting levels are exact
    at ``2**40``; only the fit's own arithmetic can differ.
    """
    generator = torch.Generator().manual_seed(3)
    near = torch.randint(-2_000, 2_000, (4_096,), generator=generator).double()
    init = torch.linspace(-1_920, 1_920, 16, dtype=torch.float64).round()
    offset = 2.0**40
    far = lloyd_max(near + offset, init=init + offset)
    assert (far - offset).tolist() == pytest.approx(
        lloyd_max(near, init=init).tolist(),
        abs=1e-3,
    )


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


def test_samples_and_levels_must_be_nonempty_and_one_dimensional() -> None:
    with pytest.raises(ValueError, match="1-D"):
        _ = lloyd_max(torch.zeros(2, 3), init=torch.zeros(2))
    with pytest.raises(ValueError, match="nonempty"):
        _ = lloyd_max(torch.zeros(0), init=torch.zeros(2))
    with pytest.raises(ValueError, match="nonempty 1-D"):
        _ = high_resolution_levels(torch.zeros(0), num_levels=2)


def test_an_empty_cell_stays_put_when_no_cell_holds_two_values() -> None:
    """Each of five levels already holds one distinct value; the sixth cannot split."""
    values = torch.arange(5, dtype=torch.float64).repeat_interleave(2)
    init = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 50.0], dtype=torch.float64)
    values = torch.cat([values, torch.tensor([4.0, 4.5])])
    levels = lloyd_max(values, init=init, max_iterations=1)
    assert levels[:4].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert levels[-1] < 50


def test_no_iterations_returns_the_sorted_start() -> None:
    values = torch.arange(10, dtype=torch.float64)
    init = torch.tensor([7.0, 1.0, 4.0], dtype=torch.float64)
    assert lloyd_max(values, init=init, max_iterations=0).tolist() == [1.0, 4.0, 7.0]


def test_a_budget_too_small_to_converge_stops_after_its_iterations() -> None:
    values = torch.randn(4096, generator=torch.Generator().manual_seed(4))
    init = torch.linspace(-3, 3, 32)
    one = lloyd_max(values, init=init, max_iterations=1, tolerance=0.0)
    two = lloyd_max(values, init=init, max_iterations=2, tolerance=0.0)
    assert one.tolist() != two.tolist()


def test_non_finite_initial_levels_are_refused() -> None:
    values = torch.arange(100, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite initial"):
        _ = lloyd_max(values, init=torch.tensor([0.0, math.nan, 50.0, 99.0]))


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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
