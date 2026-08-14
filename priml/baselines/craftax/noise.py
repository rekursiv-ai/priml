"""Batched Perlin noise, the source of the world's terrain.

Terrain is not drawn from independent per-tile randomness -- that would give
static, not landscape. Perlin noise assigns a random gradient to each point of
a coarse lattice and interpolates between them, so nearby tiles are correlated
and the result has continents, mountain ranges, and rivers at the lattice's
scale.

Every function is batched over a leading environment axis: one call generates
a distinct world per environment, since each is starting its own episode.

References:
    https://dl.acm.org/doi/10.1145/325165.325247
        Perlin 1985. An image synthesizer.
    https://github.com/pvigier/perlin-numpy
        Vigier's reference implementation, which the reference port follows.

"""

from __future__ import annotations

import math

from torch import Tensor

import torch


def fractal_noise(
    *,
    num_envs: int,
    shape: tuple[int, int],
    resolution: tuple[int, int],
    octaves: int = 1,
    persistence: float = 0.5,
    lacunarity: int = 2,
    generator: torch.Generator | None = None,
    device: torch.device,
) -> Tensor:
    """Sum several octaves of Perlin noise and rescale each field to ``[0, 1]``.

    Each octave doubles the lattice frequency and halves its amplitude, so the
    sum carries both the broad shape of the land and its fine detail. The
    rescaling is per environment: a world is thresholded against absolute
    values like "water above 0.35", so each must span the same range.

    Args:
      num_envs: Independent noise fields to generate.
      shape: Field height and width in tiles.
      resolution: Lattice cells along each axis; must divide ``shape``.
      octaves: Number of frequencies summed.
      persistence: Amplitude ratio between successive octaves.
      lacunarity: Frequency ratio between successive octaves.
      generator: Source of randomness; ``None`` uses the global stream.
      device: Device the field is built on.

    Returns:
      noise: Fields on ``[0, 1]``, ``[num_envs, shape[0], shape[1]]``.

    """
    noise = torch.zeros((num_envs, *shape), device=device)
    frequency, amplitude = 1, 1.0
    for _ in range(octaves):
        noise += amplitude * perlin_noise(
            num_envs=num_envs,
            shape=shape,
            resolution=(frequency * resolution[0], frequency * resolution[1]),
            generator=generator,
            device=device,
        )
        frequency *= lacunarity
        amplitude *= persistence
    lowest = noise.amin(dim=(-2, -1), keepdim=True)
    highest = noise.amax(dim=(-2, -1), keepdim=True)
    return (noise - lowest) / (highest - lowest)


def perlin_noise(
    *,
    num_envs: int,
    shape: tuple[int, int],
    resolution: tuple[int, int],
    generator: torch.Generator | None = None,
    device: torch.device,
) -> Tensor:
    """Generate one octave of two-dimensional gradient noise.

    Args:
      num_envs: Independent noise fields to generate.
      shape: Field height and width in tiles.
      resolution: Lattice cells along each axis; must divide ``shape``.
      generator: Source of randomness; ``None`` uses the global stream.
      device: Device the field is built on.

    Returns:
      noise: Fields roughly on ``[-1, 1]``, ``[num_envs, shape[0], shape[1]]``.

    """
    cells = (shape[0] // resolution[0], shape[1] // resolution[1])
    # Position within the lattice cell, which is what the gradients are
    # dotted against and what the interpolation runs over.
    rows = (torch.arange(shape[0], device=device) / cells[0]) % 1.0
    columns = (torch.arange(shape[1], device=device) / cells[1]) % 1.0
    grid = torch.stack(
        (
            rows[:, None].expand(shape),
            columns[None, :].expand(shape),
        ),
        dim=-1,
    )

    angles = (
        2
        * torch.pi
        * torch.rand(
            (num_envs, resolution[0] + 1, resolution[1] + 1),
            generator=generator,
            device=device,
        )
    )
    gradients = torch.stack((angles.cos(), angles.sin()), dim=-1)
    gradients = gradients.repeat_interleave(cells[0], dim=1).repeat_interleave(
        cells[1],
        dim=2,
    )

    # The four lattice corners surrounding every tile.
    top_left = gradients[:, : -cells[0], : -cells[1]]
    bottom_left = gradients[:, cells[0] :, : -cells[1]]
    top_right = gradients[:, : -cells[0], cells[1] :]
    bottom_right = gradients[:, cells[0] :, cells[1] :]

    offset = torch.tensor([1.0, 0.0], device=device)
    ramp_top_left = (grid * top_left).sum(-1)
    ramp_bottom_left = ((grid - offset) * bottom_left).sum(-1)
    ramp_top_right = ((grid - offset.flip(0)) * top_right).sum(-1)
    ramp_bottom_right = ((grid - 1.0) * bottom_right).sum(-1)

    weight = _smoothstep(grid)
    left = ramp_top_left * (1 - weight[..., 0]) + weight[..., 0] * ramp_bottom_left
    right = ramp_top_right * (1 - weight[..., 0]) + weight[..., 0] * ramp_bottom_right
    blended = (1 - weight[..., 1]) * left + weight[..., 1] * right
    # The scale restores unit variance: the ramps are dot products of unit
    # gradients against offsets of at most one, so the blend lands well
    # inside [-1, 1] without it.
    return blended * math.sqrt(2.0)


def _smoothstep(t: Tensor) -> Tensor:
    """Ease between lattice corners with zero first and second derivatives.

    A linear blend would leave a visible crease at every cell boundary, which
    reads as a grid in the terrain.
    """
    return t * t * t * (t * (t * 6 - 15) + 10)
