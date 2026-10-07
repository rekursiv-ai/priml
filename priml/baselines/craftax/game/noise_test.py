"""Tests for the batched Perlin terrain noise."""

from __future__ import annotations

from unittest.mock import Mock

from torch import Tensor

import pytest
import torch

from priml.baselines.craftax.game import constants
from priml.baselines.craftax.game.noise import (
    _smoothstep,
    fractal_noise,
    perlin_noise,
)


def test_fractal_noise_spans_the_unit_interval_per_environment() -> None:
    # Terrain is thresholded against absolute values, so every environment must
    # be rescaled on its own rather than against the batch.
    noise = fractal_noise(
        num_envs=3,
        shape=(48, 64),
        resolution=(3, 4),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    assert noise.shape == (3, 48, 64)
    assert torch.allclose(noise.amin(dim=(-2, -1)), torch.zeros(3))
    assert torch.allclose(noise.amax(dim=(-2, -1)), torch.ones(3))


def test_each_environment_gets_a_different_world() -> None:
    noise = fractal_noise(
        num_envs=2,
        shape=(48, 64),
        resolution=(3, 4),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    assert not torch.equal(noise[0], noise[1])


def test_the_same_seed_reproduces_the_same_world() -> None:
    def generate() -> Tensor:
        return fractal_noise(
            num_envs=2,
            shape=(48, 64),
            resolution=(3, 4),
            generator=torch.Generator().manual_seed(7),
            device=torch.device("cpu"),
        )

    assert torch.equal(generate(), generate())


def test_noise_is_spatially_correlated_rather_than_static() -> None:
    # The point of gradient noise: neighbouring tiles must resemble each other,
    # or the terrain is salt-and-pepper rather than landscape.
    noise = perlin_noise(
        num_envs=1,
        shape=(48, 64),
        resolution=(3, 4),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )[0]
    neighbour_gap = (noise[1:, :] - noise[:-1, :]).abs().mean()
    distant_gap = (noise[8:, :] - noise[:-8, :]).abs().mean()
    assert float(neighbour_gap) < 0.5 * float(distant_gap)


def test_noise_vanishes_on_the_lattice_corners() -> None:
    # A Perlin field is zero wherever it sits exactly on a lattice point, which
    # is the defining property of the construction.
    noise = perlin_noise(
        num_envs=1,
        shape=(48, 64),
        resolution=(3, 4),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )[0]
    assert float(noise[0, 0].abs()) < 1e-6
    assert float(noise[16, 32].abs()) < 1e-6


def test_more_octaves_add_finer_detail() -> None:
    def roughness(octaves: int) -> float:
        noise = fractal_noise(
            num_envs=1,
            shape=(48, 64),
            resolution=(3, 4),
            octaves=octaves,
            generator=torch.Generator().manual_seed(0),
            device=torch.device("cpu"),
        )[0]
        return float((noise[1:, :] - noise[:-1, :]).abs().mean())

    assert roughness(3) > roughness(1)


@pytest.mark.parametrize("shape", [(48, 64), (16, 32)])
def test_shape_is_honored(shape: tuple[int, int]) -> None:
    noise = fractal_noise(
        num_envs=2,
        shape=shape,
        resolution=(2, 4),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
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


def test_perlin_noise_matches_seeded_non_square_reference() -> None:
    noise = perlin_noise(
        num_envs=2,
        shape=(6, 12),
        resolution=(3, 4),
        generator=torch.Generator().manual_seed(31),
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [
                [0.2762, -0.0198, 0.3893, -0.2286],
                [0.0873, -0.3045, -0.1550, 0.1724],
                [-0.0068, 0.2863, -0.1336, 0.1472],
            ],
            [
                [-0.0471, -0.2471, -0.1052, -0.4008],
                [-0.1379, -0.1361, -0.4608, 0.5279],
                [-0.3521, -0.1999, 0.4341, -0.2602],
            ],
        ],
    )
    torch.testing.assert_close(noise[:, ::2, 1::3], expected, atol=5e-5, rtol=0)


def test_noise_passes_device_to_torch_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factories = {
        "arange": Mock(wraps=torch.arange),
        "rand": Mock(wraps=torch.rand),
        "zeros": Mock(wraps=torch.zeros),
    }
    for name, factory in factories.items():
        monkeypatch.setattr(torch, name, factory)
    on_device = Mock(wraps=constants.on_device)
    monkeypatch.setattr(constants, "on_device", on_device)

    fractal_noise(
        num_envs=2,
        shape=(4, 6),
        resolution=(2, 3),
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )

    for factory in factories.values():
        assert factory.call_args_list
        assert all(
            call.kwargs["device"] == torch.device("cpu")
            for call in factory.call_args_list
        )
    assert on_device.call_args.args[1] == torch.device("cpu")


def test_fractal_noise_defaults_to_one_octave() -> None:
    default = fractal_noise(
        num_envs=2,
        shape=(12, 18),
        resolution=(2, 3),
        generator=torch.Generator().manual_seed(9),
        device=torch.device("cpu"),
    )
    explicit = fractal_noise(
        num_envs=2,
        shape=(12, 18),
        resolution=(2, 3),
        octaves=1,
        generator=torch.Generator().manual_seed(9),
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(default, explicit, atol=0, rtol=0)


def test_fractal_noise_defaults_to_half_persistence() -> None:
    default = fractal_noise(
        num_envs=2,
        shape=(12, 18),
        resolution=(2, 3),
        octaves=2,
        generator=torch.Generator().manual_seed(9),
        device=torch.device("cpu"),
    )
    explicit = fractal_noise(
        num_envs=2,
        shape=(12, 18),
        resolution=(2, 3),
        octaves=2,
        persistence=0.5,
        generator=torch.Generator().manual_seed(9),
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(default, explicit, atol=0, rtol=0)


def test_fractal_noise_uses_each_octave_and_rescales_per_environment() -> None:
    noise = fractal_noise(
        num_envs=2,
        shape=(24, 48),
        resolution=(3, 4),
        octaves=3,
        persistence=0.25,
        generator=torch.Generator().manual_seed(31),
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [0.5316, 0.4943, 0.4699, 0.5316, 0.6687, 0.3893],
            [0.8371, 0.6231, 0.3936, 0.7621, 0.3590, 0.3577],
            [0.5316, 0.8372, 0.2412, 0.5316, 0.1708, 0.6462],
            [0.0000, 0.4482, 0.5873, 0.5227, 0.0513, 0.4462],
            [0.5316, 0.4545, 0.7913, 0.5316, 0.3549, 0.7763],
            [0.9382, 0.2274, 0.4543, 0.0917, 0.4459, 0.8348],
        ],
    )
    torch.testing.assert_close(noise[0, ::4, ::8], expected, atol=5e-5, rtol=0)
    torch.testing.assert_close(noise.amin(dim=(-2, -1)), torch.zeros(2))
    torch.testing.assert_close(noise.amax(dim=(-2, -1)), torch.ones(2))


def test_fractal_noise_rescales_by_the_reciprocal_of_its_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The bits XLA's CPU build gives 71 in a field spanning 0 to 300; a true
    # division gives 1_047_681_215 and flips a terrain tile in some worlds.
    field = torch.zeros((2, 3, 4))
    field[:, 0, 0] = 300.0
    field[:, 1, 2] = 71.0
    monkeypatch.setattr(
        "priml.baselines.craftax.game.noise.perlin_noise",
        Mock(return_value=field),
    )
    rescaled = fractal_noise(
        num_envs=2,
        shape=(3, 4),
        resolution=(1, 1),
        device=torch.device("cpu"),
    )
    assert rescaled[:, 1, 2].view(torch.int32).tolist() == [1_047_681_216] * 2


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
