"""The Craftax game as PufferLib plays it: rules, world, and what a player sees.

An open-ended 2D survival game. A player mines wood and stone, crafts tools,
fights creatures, eats, drinks, sleeps, and descends through nine floors. There
is no single goal -- 67 achievements form a tech tree, and the score rewards
breadth rather than depth.

This package is the game and nothing else, ported line for line from
PufferLib's ``craftax.h`` (pin ``6ffa5b10``) so that every draw, every float
and every byte of state matches the C. It steps one environment at a time in
Numba on the CPU; the learner code lives one directory up and depends on this
package, and nothing here depends on it.

Why numpy and scalar loops, not torch and batches. Numba compiles numpy
arrays, records and scalars, not tensors, and it matched the C's speed on
PufferLib's logic, while the same logic in compiled torch on the CPU ran 4-9x
slower than the C (measured 2026-09-25 on one thread over 512 environments,
every output bit-identical to the C's: observation packing 1,952 against 218
ns per environment, spawn collection 3,231 against 757). Batching the
environments does not recover it: a batch must keep ticking until its slowest
sleeper or rester wakes, 22x the real tick work, and the conditional draws and
the variable-length spawn-candidate lists turn into masked work over the whole
batch. So every kernel steps one environment in the C's order. The
``np.float32(...)`` casts keep that arithmetic in fp32, as the C's floats are:
a Python float is fp64, and one fp64 intermediate moves the bits.

Modules, each depending only on those before it:

- ``state``: the constants and the ``State`` record.
- ``jit``: the kernel decorator and C arithmetic.
- ``rng``: glibc's ``rand_r`` and the draws built on it.
- ``rules``: the shared mechanics and the player's phases of a tick.
- ``world_gen``: terrain noise, the floor recipes and world generation.
- ``mobs``: creatures moving, shooting and spawning.
- ``observation``: the observation and the action mask.
- ``step``: a whole step, a buffer's rows at a time.
- ``archive``: practice's saved worlds, a training option's kernels, here
  because the buffer loop that runs them may call only ``game/``'s.

``testing`` is test support, not the game: it reads a kernel's compiled code for
the tests of ``jit``'s rules.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``ocean/craftax/craftax.h``.
    https://arxiv.org/abs/2402.16801
        Matthews et al. 2024. Craftax: a lightning-fast benchmark for
        open-ended reinforcement learning.

"""

from __future__ import annotations
