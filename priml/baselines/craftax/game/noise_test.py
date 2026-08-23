"""Tests for the batched Perlin terrain noise."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game.noise import (
    _smoothstep,
    fractal_noise,
    perlin_noise,
)


_DEVICE = torch.device("cpu")


def test_fractal_noise_spans_the_unit_interval_per_environment() -> None:
    # Terrain is thresholded against absolute values, so every environment must
    # be rescaled on its own rather than against the batch.
    noise = fractal_noise(
        num_envs=3,
        shape=(48, 48),
        resolution=(3, 3),
        generator=torch.Generator().manual_seed(0),
        device=_DEVICE,
    )
    assert noise.shape == (3, 48, 48)
    assert torch.allclose(noise.amin(dim=(-2, -1)), torch.zeros(3))
    assert torch.allclose(noise.amax(dim=(-2, -1)), torch.ones(3))


def test_each_environment_gets_a_different_world() -> None:
    noise = fractal_noise(
        num_envs=2,
        shape=(48, 48),
        resolution=(3, 3),
        generator=torch.Generator().manual_seed(0),
        device=_DEVICE,
    )
    assert not torch.equal(noise[0], noise[1])


def test_the_same_seed_reproduces_the_same_world() -> None:
    def generate() -> Tensor:
        return fractal_noise(
            num_envs=2,
            shape=(48, 48),
            resolution=(3, 3),
            generator=torch.Generator().manual_seed(7),
            device=_DEVICE,
        )

    assert torch.equal(generate(), generate())


def test_noise_is_spatially_correlated_rather_than_static() -> None:
    # The point of gradient noise: neighbouring tiles must resemble each other,
    # or the terrain is salt-and-pepper rather than landscape.
    noise = perlin_noise(
        num_envs=1,
        shape=(48, 48),
        resolution=(3, 3),
        generator=torch.Generator().manual_seed(0),
        device=_DEVICE,
    )[0]
    neighbour_gap = (noise[1:, :] - noise[:-1, :]).abs().mean()
    distant_gap = (noise[8:, :] - noise[:-8, :]).abs().mean()
    assert float(neighbour_gap) < 0.5 * float(distant_gap)


def test_noise_vanishes_on_the_lattice_corners() -> None:
    # A Perlin field is zero wherever it sits exactly on a lattice point, which
    # is the defining property of the construction.
    noise = perlin_noise(
        num_envs=1,
        shape=(48, 48),
        resolution=(3, 3),
        generator=torch.Generator().manual_seed(0),
        device=_DEVICE,
    )[0]
    assert float(noise[0, 0].abs()) < 1e-6
    assert float(noise[16, 32].abs()) < 1e-6


def test_more_octaves_add_finer_detail() -> None:
    def roughness(octaves: int) -> float:
        noise = fractal_noise(
            num_envs=1,
            shape=(48, 48),
            resolution=(3, 3),
            octaves=octaves,
            generator=torch.Generator().manual_seed(0),
            device=_DEVICE,
        )[0]
        return float((noise[1:, :] - noise[:-1, :]).abs().mean())

    assert roughness(3) > roughness(1)


@pytest.mark.parametrize("shape", [(48, 48), (16, 32)])
def test_shape_is_honored(shape: tuple[int, int]) -> None:
    noise = fractal_noise(
        num_envs=2,
        shape=shape,
        resolution=(2, 2),
        generator=torch.Generator().manual_seed(0),
        device=_DEVICE,
    )
    assert noise.shape == (2, *shape)


def test_smoothstep_is_flat_at_both_ends() -> None:
    # Zero slope at the cell boundaries is what removes the visible grid; a
    # linear blend would leave a crease at every lattice line.
    step = 1e-4
    at_zero = _smoothstep(torch.tensor([0.0, step]))
    at_one = _smoothstep(torch.tensor([1.0 - step, 1.0]))
    assert float(at_zero[1] - at_zero[0]) < 1e-8
    assert float(at_one[1] - at_one[0]) < 1e-8
    assert float(_smoothstep(torch.tensor(0.5))) == pytest.approx(0.5)


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
