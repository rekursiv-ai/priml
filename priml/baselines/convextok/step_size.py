"""The initial PDLP step size: 0.998 over the scaled matrix's largest singular value.

cuOpt (Stable3) estimates the largest singular value by power iteration on ``A A^T``
and keeps the resulting step for the whole solve. The iteration starts from
libstdc++'s ``std::normal_distribution<double>`` over ``std::mt19937(1)``, reproduced
here: NumPy's legacy ``RandomState`` is the same Mersenne Twister, and the polar
transform runs through ``math.log``, which is the C library's ``log`` -- glibc's on
Linux, as in cuOpt's build.
"""

from collections.abc import Iterator
from typing import cast

import math

from torch import Tensor

import numpy as np
import torch

from priml.baselines.convextok.scaling import ScaledProgram


def singular_value_probe(size: int) -> Tensor:
    """Return libstdc++'s first ``size`` normal draws from ``std::mt19937(1)``.

    Args:
      size: Number of draws.

    Returns:
      probe: float64 draws, on the CPU.

    """
    out = np.empty(size)
    filled = 0
    for pairs in _polar_pairs():
        take = min(len(pairs), size - filled)
        out[filled : filled + take] = pairs[:take]
        filled += take
        if filled == size:
            break
    return torch.from_numpy(out)


def initial_step_size(
    scaled: ScaledProgram,
    *,
    tolerance: float = 1e-4,
    max_iterations: int = 5000,
) -> float:
    """Return cuOpt's initial step size for the scaled program.

    Args:
      scaled: The program in PDLP's scaled space.
      tolerance: Stop once ``||A A^T q - sigma^2 q||`` falls below this.
      max_iterations: Most power-iteration steps.

    Returns:
      step_size: ``0.998 / sigma_max``.

    """
    program = scaled.program
    z = singular_value_probe(program.num_rows).to(program.values.device)
    sigma_squared = torch.zeros((), dtype=torch.float64, device=z.device)
    for _ in range(max_iterations):
        q = z / torch.linalg.vector_norm(z)
        z = scaled.matrix @ (scaled.transpose @ q)
        sigma_squared = torch.dot(q, z)
        if torch.linalg.vector_norm(z - sigma_squared * q).item() < tolerance:
            break
    return 0.998 / sigma_squared.sqrt().item()


def _polar_pairs() -> Iterator[np.ndarray]:
    """Yield libstdc++'s normal draws in chunks, each the pairs of one batch of attempts."""
    state = np.random.RandomState()
    state.seed(1)
    while True:
        # Each attempt takes four 32-bit words: two per canonical double, low word first.
        words = state.randint(1 << 32, size=4 * 65_536)
        words = words.reshape(words.size // 4, 2, 2)
        canonical = (
            words[..., 0] + words[..., 1] * 4294967296.0
        ) / 18446744073709551616.0
        canonical = np.minimum(canonical, math.nextafter(1.0, 0.0))
        x = 2.0 * canonical[:, 0] - 1.0
        y = 2.0 * canonical[:, 1] - 1.0
        radius = x * x + y * y
        # A sum of squares is never negative, so ``> 0`` is libstdc++'s ``!= 0``.
        kept = (radius <= 1.0) & (radius > 0.0)
        x, y, radius = x[kept], y[kept], radius[kept]
        logs = np.array(
            [math.log(value) for value in cast("list[float]", radius.tolist())],
        )
        scale = np.sqrt(-2.0 * logs / radius)
        pairs = np.empty(2 * len(radius))
        pairs[0::2] = y * scale
        pairs[1::2] = x * scale
        yield pairs
