"""References independent of the port, which the game's kernels are checked against.

None of them comes from the port: a numpy transcription of glibc's ``rand_r``
(``stdlib/rand_r.c``), the live libc's ``rand_r`` through ``ctypes``, and the
light level composed from the platform libm's ``cosf``, ``powf`` and ``fmodf``.
The tests compare the kernels with them live, on every host. Each reference is
anchored in turn, in ``mint_goldens_test.py``: the transcription to the live libc
where it is glibc, and both references to PufferLib's own values on any host
(``SEED_73_DRAWS``, ``LIGHT_ANCHORS``), so an error a reference shares with the
kernel it checks still fails.

The references are numpy, not torch: the tests that read them drive Numba
kernels, which take numpy arrays, and the transcription needs uint32 arithmetic
that wraps, which numpy does and torch's CPU kernels do not. libc and libm are
called one value at a time, because their C functions take one value.
"""

from __future__ import annotations

from ctypes.util import find_library
from typing import TYPE_CHECKING, Final, Protocol, SupportsFloat, cast

import ctypes

import numpy as np

from priml.baselines.craftax.game.state import DAY_LENGTH, PI


if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from priml.baselines.craftax.lib.arrays import Shaped


SEED_73_DRAWS: Final = (
    415_296_222,
    1_677_370_356,
    237_865_384,
    2_132_668_485,
    1_774_877_301,
    1_422_619_285,
    417_103_133,
    1_000_715_045,
)
"""PufferLib's first eight ``rand_r`` draws from state 73, glibc's own.

Row 2 of its oracle's ``fixtures/env/rng_streams/first64_draws.i32``.
"""

LIGHT_ANCHORS: Final = {
    0: 0x3F4C034B,
    1: 0x3F4E3CA0,
    75: 0x3F7F051D,
    150: 0x3EF0E43E,
    225: 0x3D156E30,
    299: 0x3F49BDBD,
    54_321: 0x3F6FF76F,
    100_000: 0x3F6EC53D,
}
"""PufferLib's fp32 light level bits by timestep, where macOS's and glibc's libm agree.

From its oracle's ``fixtures/env/light_table/light_table.f32``. The two libms
differ at 401 of the 100,001 timesteps a default episode reaches.
"""


def glibc_rand_r(seeds: Shaped[np.uint32], count: int) -> NDArray[np.int32]:
    """Return the first ``count`` glibc ``rand_r`` draws of each seed.

    A transcription of glibc's ``rand_r``: three steps of the
    ``x <- 1103515245 * x + 12345`` LCG in uint32 per draw, taking 11, 10 and
    10 bits of the state's high half. The LCG is not stepped one state at a
    time: step ``m`` is its closed form ``a^m * x + c * (1 + a + ... +
    a^(m-1))``, with the powers and their sums as uint32 cumulative products
    and sums, which wrap modulo 2^32 exactly as the C does. That is the same
    state, and a single array expression instead of a Python loop per draw.

    Args:
      seeds: Initial states, ``uint32 [n]``.
      count: Draws per seed.

    Returns:
      draws: ``int32 [n, count]``, each in ``[0, 2^31)``.

    """
    multiplier = np.uint32(1_103_515_245)
    powers = np.cumprod(
        np.full(3 * count, multiplier, dtype=np.uint32),
        dtype=np.uint32,
    )
    sums = np.cumsum(np.concatenate(([np.uint32(1)], powers[:-1])), dtype=np.uint32)
    states = seeds.astype(np.uint32)[:, None] * powers + sums * np.uint32(12_345)
    high = (states >> np.uint32(16)).reshape(seeds.shape[0], count, 3)
    result = (high[..., 0] & np.uint32(0x7FF)) << np.uint32(10)
    result = (result ^ (high[..., 1] & np.uint32(0x3FF))) << np.uint32(10)
    return (result ^ (high[..., 2] & np.uint32(0x3FF))).astype(np.int32)


def libc_rand_r(seed: int, count: int) -> NDArray[np.int32]:
    """Return ``count`` draws from the live libc's ``rand_r`` for one seed.

        Only meaningful where the libc is glibc; other libcs implement ``rand_r``
        differently.

    Args:
      seed: The initial state.
      count: Draws to take.

    Returns:
      draws: ``int32 [count]``.

    """
    libc = ctypes.CDLL(find_library("c"))
    libc.rand_r.restype = ctypes.c_int
    libc.rand_r.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    state = ctypes.c_uint(seed)
    return np.fromiter(
        (libc.rand_r(ctypes.byref(state)) for _ in range(count)),
        np.int32,
        count,
    )


def light_levels_libm(timesteps: Iterable[int]) -> NDArray[np.float32]:
    """Return the light level at each of ``timesteps`` through the platform libm.

        ``1 - powf(fabsf(cosf(pi * (fmodf(t / 300, 1) + 0.3))), 3)`` with every
        operation in float32 and the transcendental ones from libm.

    Args:
      timesteps: Steps elapsed in the episode.

    Returns:
      values: ``float32``, one per timestep.

    """
    libm = _libm()
    values: list[np.float32] = []
    # One libm call per value (module docstring).
    for timestep in timesteps:
        day = libm.fmodf(np.float32(timestep) / np.float32(DAY_LENGTH), 1.0)
        progress = np.float32(day) + np.float32(0.3)
        cosine = np.float32(libm.cosf(PI * progress))
        values.append(np.float32(1.0) - np.float32(libm.powf(abs(cosine), 3.0)))
    return np.array(values, dtype=np.float32)


def _libm() -> _Libm:
    """Return the platform libm with float32 signatures declared."""
    # The process's own symbols: the libm the interpreter and Numba's kernels link.
    libm = ctypes.CDLL(None)
    libm.cosf.restype = ctypes.c_float
    libm.cosf.argtypes = [ctypes.c_float]
    for function in (libm.powf, libm.fmodf):
        function.restype = ctypes.c_float
        function.argtypes = [ctypes.c_float, ctypes.c_float]
    return cast("_Libm", libm)


class _Libm(Protocol):
    """The libm calls a light level is composed of, as :func:`_libm` declares them."""

    def cosf(self, x: SupportsFloat, /) -> float: ...
    def powf(self, x: SupportsFloat, y: SupportsFloat, /) -> float: ...
    def fmodf(self, x: SupportsFloat, y: SupportsFloat, /) -> float: ...
