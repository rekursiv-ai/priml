"""The actor: action sampling, and one CUDA graph per (slot, buffer) that runs it.

The rollout interleaves the GPU and the CPU per buffer: the policy scores one
buffer's observations while the other buffers step on their threads. Per step
``t`` of buffer ``b``, in one captured graph on the buffer's stream:

1. copy the env's pinned host buffers (fp32 observations, rewards and
   terminals; the uint8 action mask) to the device and round them into row
   ``t`` of the slot's storage: the rewards to the rollout's dtype, the rest
   to the policy's (bf16 for exp000);
2. zero the buffer's carry where the last step ended an episode; at
   ``t == 0`` snapshot the carry into the slot's initial states, so the
   learner starts its recurrence where the rollout did;
3. the policy forward, the sampler, and the row-``t`` stores of actions,
   log-probabilities and values;
4. the actions' copy back to the env's pinned buffer.

``t`` is a device tensor the graph reads and increments, so one graph per
``(slot, buffer)`` serves every step, rather than one per ``(slot, t,
buffer)``; a rollout zeroes it and replays exactly ``horizon`` times, so it
only ever indexes rows ``0..horizon-1``. Every tensor a graph addresses is
owned for the graph's lifetime (measured: a freed one made replay fail).

With practice (a :class:`PracticeEnv`), a restored row must start from the
carry its entry's donor had, not from zero, so its policy reads the branch as
the donor's episode continued. The rollout keeps a carry archive, one carry
per practice entry. A buffer holding rows that can save (the env's
``save_rows``) also uploads their save slots each step and, after step 2,
writes the (reset) carry of each that saved an entry on its last step into
that entry: the carry the next forward reads, which has consumed exactly the
observations before the saved one. The write is its own torch ops in the
graph, so the ingest kernel is the same with practice or without it, and the
other buffers' graphs are practice-off's. After the last step a tail does the
same for that step's saves. At the start of a rollout the rows the env
restored gather their entries' carries, before any buffer's first step, and
are marked in ``branch_starts``; the step-0 snapshot then stores those
carries as the learner's initial states.

A per-step feature (:class:`FeatureSource`) carries a history of its own,
which a restore replaces too, as the source's practice slot says: the
restored row's feature step begins a fresh window, or, where the source
keeps a history archive, the saving rows also write their feature histories
beside their carries, with the same timing, and a restored row resumes its
entry's, reading as its next previous action the one its donor took. A
feature step can change its shape between blocks of steps (its ``plan``):
each ``(slot, buffer)`` then captures one graph per plan, all sharing the
first one's memory pool, since a buffer replays one graph at a time.

A joint feature's weights change between rollouts, and its learner recomputes
the features: each step also stores its frame's tokens, the action that led to
it and its context (``RolloutStorage.frame_cells`` and the rest), and each
slot stores, before its first step and after the restores, each row's last
decisions its first contexts reach back to (``prefix_cells`` and the rest).
After a publication of new weights, :meth:`Rollout.rebuild_features` rebuilds
every row's history under them.

Each buffer's worker loop is design B when the env runs a buffer's steps in
one ``nogil`` loop (``CraftaxEnv.run_buffer``): the env launches the graph
and waits for it before every step, so no Python runs per step. Otherwise it
is design A: ``replay``, ``synchronize``, then the env's ``step_buffer`` --
three Python calls per buffer-step, each releasing the GIL for its wait.
Without CUDA the same steps run eagerly, which is what the CPU tests
exercise.

The sampler gives every agent of a buffer a curand Philox4x32-10 stream,
seeded ``curand_init(seed + buffer, agent, 0)``: key ``(seed + buffer)``
split into its low and high 32-bit words, counter ``(0, 0, agent, 0)``. Draw
``n`` of that stream is word ``n mod 4`` of the block at counter ``n // 4``
(curand caches a block of four and advances the counter's low word), and
``curand_uniform`` maps the word to ``(0, 1]`` as one fma: ``x * 2^-32 +
2^-33``. Keeping the draw count per agent on the device is what lets a
captured step graph advance the stream without the host.

The action is the masked inverse CDF: masked logits are ``-1e4``, the
logsumexp is the one-pass form the learner also uses, and the cumulative sum
adds ``exp(logit - logsumexp)`` in action order until it exceeds the draw. A
fall-through lands on the last action, which may be masked, and snaps to the
last legal one. The log-probability is the sampled logit minus the logsumexp.

:class:`PhiloxSampler` runs that kernel, on a CUDA device only, with Triton's
``exp`` and ``log``; a subclass replaces the logsumexp's last step and the
CDF's accumulation by name (:attr:`PhiloxSampler.helpers`), with its launch
options, for another implementation's rounding. :class:`TorchPhiloxSampler`
runs the same algorithm in torch on any device, on the same streams (the
uniforms are exact) but with torch's precise ``exp``: the reference for the
algorithm, not for the kernel's bits. The kernel refuses a CPU tensor rather
than falling back to it, so one config gives one set of bits on every host.

Two things here are numpy rather than torch. The Philox reference
(:func:`philox_uniform`) computes on uint64 words, and torch's CPU kernels
implement no ``+``, ``>>``, ``//`` or ``%`` on uint64 (torch 2.11). The env's
buffers are shared with its Numba workers, which compile numpy arrays and
scalars, not tensors; so the graph handles the env's ``nogil`` loop launches
are numpy scalars.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), the rollout and ``sample_logits`` in
        ``src/pufferl.cu``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Final,
    Protocol,
    cast,
    override,
    runtime_checkable,
)

import ctypes
import threading

from configgle import Fig, Makes
from torch import Tensor

import numpy as np
import torch

from priml.baselines.craftax.game.state import (
    INVENTORY_OBS_SIZE,
    OBS_COLS,
    OBS_ROWS,
    OBS_TILE_CHANNELS,
)
from priml.kernel import jit_kernel, require_power_of_two
from priml.loss.policy_gradient import TorchPPO
from priml.loss.policy_gradient_kernel import (
    add_log_triton,
    masked_logsumexp_triton,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from numpy.typing import NDArray
    from triton import language

    import numba
    import triton

    from priml.baselines.craftax.model import Policy
    from priml.baselines.craftax.world_model.feature import (
        RecentDecisions,
    )
else:
    from wrapt import lazy_import

    numba = lazy_import("numba")
    triton = lazy_import("triton")
    language = lazy_import("triton.language")


CAPTURE_LOCK: Final = threading.Lock()
"""One graph capture at a time in the process. ``torch.cuda.graph`` synchronizes
the whole device before it begins, and that fails while another thread's
capture is open; the buffers' workers reach their first capture together, and
the learner captures its epoch while a rollout runs."""


class EnvBuffers(Protocol):
    """What the rollout needs of ``CraftaxEnv``: its five buffers and a step."""

    observations: Tensor
    action_mask: Tensor
    rewards: Tensor
    terminals: Tensor
    actions: Tensor
    num_envs: int
    num_buffers: int

    def buffer_slice(self, buffer: int) -> slice:
        """Return the rows of every buffer attribute that belong to ``buffer``."""
        ...

    def step_buffer(self, buffer: int) -> None:
        """Step one buffer by its ``actions`` row, writing the other four."""
        ...


@runtime_checkable
class NogilEnv(EnvBuffers, Protocol):
    """An env that runs a buffer's steps in one ``nogil`` loop (design B)."""

    def run_buffer(
        self,
        buffer: int,
        steps: int,
        prepare: Callable[..., int],
        args: tuple[object, ...],
    ) -> None:
        """Step ``buffer`` ``steps`` times, calling ``prepare(*args)`` before each step.

        Args:
          buffer: Which buffer to step.
          steps: How many times.
          prepare: A ``nogil`` Numba function the env's loop calls before each
            step without returning to Python; a nonzero result ends the loop
            with an error.
          args: ``prepare``'s arguments.

        """
        ...


@runtime_checkable
class PracticeEnv(EnvBuffers, Protocol):
    """An env whose rows save and restore practice entries: the rollout keeps their carries.

    Attributes:
      save_slots: ``int32 [num_envs]`` host tensor: the entry each row saved
        into on its last step, else -1.
      restore_slots: ``int32 [num_envs]`` host tensor: the entry each row was
        restored from for the next rollout's first step, else -1.
      carry_slots: Entries: the rows of the rollout's carry archive; 0 keeps
        none.
      save_rows: The rows whose ``save_slots`` can be set; the rollout saves
        carries from these alone.

    """

    save_slots: Tensor
    restore_slots: Tensor
    carry_slots: int
    save_rows: slice


class Sampler(Protocol):
    """What the rollout needs of a sampler: one Philox stream per agent."""

    def draws(self, num_agents: int, *, device: torch.device | str) -> Tensor:
        """Return a fresh per-agent draw count."""
        ...

    def __call__(
        self,
        decoded: Tensor,
        action_mask: Tensor,
        draws: Tensor,
        *,
        buffer: int,
        dtype: torch.dtype | None = None,
    ) -> Sampled:
        """Draw one action per agent; log-probabilities and values in ``dtype``."""
        ...


class FeatureStep(Protocol):
    """One buffer's per-step feature: a graph-safe step and upkeep between blocks."""

    @property
    def plan(self) -> int:
        """The shape of the next block's steps: a step graph is captured per plan."""
        ...

    def __call__(
        self,
        observation: Tensor,
        terminals: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        """Take one env step for every row of the buffer; return ``[agents, width]``.

        Args:
          observation: The fp32 observations the step uploaded ``[agents, size]``.
          terminals: Nonzero where the observation begins an episode ``[agents]``.
          previous_action: The action that led to it, fp32 ``[agents]``.

        """
        ...

    def capture_state(self) -> list[Tensor]:
        """Return the tensors a graph warmup changes, for the capture to restore."""
        ...

    def ensure_room(self, steps: int) -> dict[str, float]:
        """Make room for ``steps`` more steps, between steps; return telemetry by name."""
        ...

    def reset(self) -> None:
        """Begin an episode in every row at its next step."""
        ...

    def begin_window(self, rows: Tensor) -> None:
        """Begin a mid-episode history at the next step of ``rows``, between steps.

        Args:
          rows: Bool ``[agents]``, on any device.

        """
        ...

    def context(self) -> tuple[Tensor, Tensor]:
        """Return each row's context after its last step: decisions, and its anchor."""
        ...

    def last_decisions(self, n: int) -> RecentDecisions:
        """Return each row's last ``n`` context decisions, with frames; graph-safe."""
        ...

    def rebuild(self) -> None:
        """Rebuild every row's history under the source's new weights, between steps."""
        ...

    def save_history(
        self,
        archive: Mapping[str, Tensor],
        entries: Tensor,
        rows: slice,
        *,
        previous_action: Tensor,
        fresh: Tensor,
    ) -> None:
        """Write ``rows``' histories into ``entries`` of the archive, in the graph.

        Args:
          archive: The source's :meth:`FeatureSource.history_archive`.
          entries: The archive row of each of ``rows`` ``[S]``, no two alike.
          rows: The saving rows of the buffer.
          previous_action: The action each took last ``[S]``.
          fresh: Nonzero where the next observation begins an episode ``[S]``.

        """
        ...

    def restore_history(
        self,
        archive: Mapping[str, Tensor] | None,
        entries: Tensor,
    ) -> Tensor | None:
        """Give restored rows their entries' histories, or windows without an archive.

        Args:
          archive: The source's :meth:`FeatureSource.history_archive`.
          entries: The entry each row was restored from, else -1 ``[agents]``.

        Returns:
          previous_action: The action each restored row's next step reads as
            its previous one ``[agents]``; None to read the env's.

        """
        ...

    def release(self) -> None:
        """Free what the steps hold; the step is not called again."""
        ...


class FeatureSource(Protocol):
    """Makes each buffer's :class:`FeatureStep`; one per rollout.

    Attributes:
      width: Floats per row of the feature.
      hook_interval: The most steps a buffer takes between ``ensure_room`` calls.
      joint: Whether its weights change between rollouts, so that the learner
        recomputes the features from the inputs the rollout stores.
      context_decisions: The most decisions a feature's context holds.

    """

    width: int
    hook_interval: int
    joint: bool
    context_decisions: int

    def make_engine(self, *, rows: int, device: torch.device) -> FeatureStep:
        """Return a fresh step over ``rows`` rows on ``device``."""
        ...

    def history_archive(
        self,
        entries: int,
        *,
        device: torch.device,
    ) -> dict[str, Tensor] | None:
        """Return storage for ``entries`` practice entries' histories; None keeps none.

        Args:
          entries: The carry archive's rows: practice's entries, then one dump
            row per saving row.
          device: Where the steps run.

        Returns:
          archive: Named tensors of ``entries`` rows, what the steps'
            ``save_history`` writes and ``restore_history`` reads.

        """
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class Sampled:
    """One buffer-step's draws.

    Attributes:
      actions: ``[agents]`` fp32, the action ids.
      logprobs: ``[agents]``, computed in fp32 and rounded to the sampler's
        ``dtype``.
      values: ``[agents]``, the decoder's value column in that dtype.

    """

    actions: Tensor
    logprobs: Tensor
    values: Tensor


class TorchPhiloxSampler:
    """Sample by masked inverse CDF from one Philox stream per agent, in torch.

    It runs on any device. The uniforms are the kernel's exactly; the
    probabilities use torch's precise ``exp`` and ``logsumexp``, so an action
    whose draw lies within an ulp of a CDF step can differ from
    :class:`PhiloxSampler`'s.
    """

    class Config(Fig["TorchPhiloxSampler"]):
        """The stream seed."""

        seed: int = 73
        """``curand_init(seed + buffer, agent, 0)`` seeds buffer ``buffer``;
        curand's seed is 64-bit, so ``seed + buffer`` wraps modulo 2^64."""

    def __init__(self, config: Config) -> None:
        """Keep the seed.

        Args:
          config: The seed.

        Raises:
          ValueError: The seed is not a 64-bit unsigned value.

        """
        if config.seed < 0 or config.seed >= 2**64:
            raise ValueError(f"seed must be in [0, 2**64), not {config.seed}")
        self.seed = config.seed

    def draws(self, num_agents: int, *, device: torch.device | str) -> Tensor:
        """Return a fresh per-agent draw count.

        Args:
          num_agents: Agents in the buffer.
          device: Where the counter lives; the sampler advances it in place.

        Returns:
          draws: ``[num_agents]`` int64 zeros.

        """
        return torch.zeros(num_agents, dtype=torch.int64, device=device)

    def __call__(
        self,
        decoded: Tensor,
        action_mask: Tensor,
        draws: Tensor,
        *,
        buffer: int,
        dtype: torch.dtype | None = None,
    ) -> Sampled:
        """Draw one action per agent and advance every agent's stream.

        Args:
          decoded: ``[agents, num_actions + 1]``, logits then value.
          action_mask: ``[agents, num_actions]``, nonzero where legal.
          draws: The buffer's draw counts from :meth:`draws`; incremented.
          buffer: Which buffer these agents are, for the stream seed.
          dtype: The log-probabilities' and values' dtype; ``decoded``'s if
            None.

        Returns:
          sampled: Actions, log-probabilities and values.

        """
        num_actions = decoded.shape[-1] - 1
        legal = action_mask != 0
        logits = torch.where(
            legal,
            decoded[:, :num_actions].float(),
            TorchPPO.Config.MASKED_LOGIT,
        )
        logsumexp = torch.logsumexp(logits, dim=-1)
        # The uniforms come from the uint64 numpy reference (module docstring).
        uniform = torch.from_numpy(
            philox_uniform(
                self.key(buffer),
                agents=np.arange(decoded.shape[0], dtype=np.uint64),
                draws=draws.cpu().numpy().astype(np.uint64),
            ),
        ).to(decoded.device)
        cdf = torch.exp(logits - logsumexp[:, None]).cumsum(dim=-1)
        hit = uniform[:, None] < cdf
        actions = torch.where(
            hit.any(dim=-1),
            hit.int().argmax(dim=-1),
            num_actions - 1,
        )
        # A fall-through lands on the last action; snap it to the last legal one.
        positions = torch.arange(num_actions, device=decoded.device)
        last_legal = torch.where(legal, positions, 0).amax(dim=-1)
        actions = torch.where(actions == num_actions - 1, last_legal, actions)
        chosen = logits.gather(-1, actions[:, None])[:, 0]
        draws.add_(1)
        dtype = dtype or decoded.dtype
        return Sampled(
            actions=actions.float(),
            logprobs=(chosen - logsumexp).to(dtype),
            values=decoded[:, num_actions].to(dtype, copy=True),
        )

    def key(self, buffer: int) -> int:
        """Return buffer ``buffer``'s Philox key: ``seed + buffer`` modulo 2^64.

        Args:
          buffer: Which buffer.

        Returns:
          key: The 64-bit key; its low word is the key's first word.

        """
        return (self.seed + buffer) & (2**64 - 1)


class PhiloxSampler(TorchPhiloxSampler):
    """Sample by masked inverse CDF from one Philox stream per agent, as one kernel.

    CUDA only. It takes :class:`TorchPhiloxSampler`'s seed, counts and keys,
    but not its sampling: that is the algorithm on other devices, with other
    bits.
    """

    class Config(Makes["PhiloxSampler"], TorchPhiloxSampler.Config):
        """The stream seed and the kernel's launch geometry."""

        block: int = 256
        """Agents per program."""

        num_warps: int = 8
        """Warps per program: one agent per thread at the default block."""

    helpers: ClassVar[dict[str, Callable[..., object]]] = {}
    """Device functions the kernel calls, by name, in place of its own
    (``add_log_triton``, ``_accumulate_exp_triton``); empty for Triton's
    standard arithmetic."""

    launch_options: ClassVar[dict[str, bool]] = {}
    """Keywords the launch adds to Triton's defaults."""

    def __init__(self, config: Config) -> None:
        """Keep the seed and the launch geometry.

        Args:
          config: The seed and the geometry.

        Raises:
          ValueError: The seed is not a 64-bit unsigned value, or the geometry
            is not a power of two.

        """
        super().__init__(config)
        require_power_of_two(block=config.block, num_warps=config.num_warps)
        self.block = config.block
        self.num_warps = config.num_warps

    @override
    def __call__(
        self,
        decoded: Tensor,
        action_mask: Tensor,
        draws: Tensor,
        *,
        buffer: int,
        dtype: torch.dtype | None = None,
    ) -> Sampled:
        """Draw one action per agent and advance every agent's stream, in one launch.

        Args:
          decoded: ``[agents, num_actions + 1]``, logits then value, on a
            CUDA device.
          action_mask: ``[agents, num_actions]``, nonzero where legal.
          draws: The buffer's draw counts from :meth:`draws`; incremented.
          buffer: Which buffer these agents are, for the stream seed.
          dtype: The log-probabilities' and values' dtype; ``decoded``'s if
            None.

        Returns:
          sampled: Actions, log-probabilities and values.

        Raises:
          ValueError: ``decoded`` is not on a CUDA device.

        """
        if not decoded.is_cuda:
            msg = (
                f"PhiloxSampler runs on a CUDA device, not {decoded.device}; "
                "select TorchPhiloxSampler to sample elsewhere"
            )
            raise ValueError(msg)
        agents, fused = decoded.shape
        actions = torch.empty(agents, dtype=torch.float32, device=decoded.device)
        logprobs = torch.empty(
            agents,
            dtype=dtype or decoded.dtype,
            device=decoded.device,
        )
        values = torch.empty_like(logprobs)
        key = self.key(buffer)
        _sample_kernel(**self.helpers)[(triton.cdiv(agents, self.block),)](
            (decoded.contiguous(), action_mask.contiguous(), draws),
            (actions, logprobs, values),
            agents,
            key & 0xFFFFFFFF,
            key >> 32,
            num_actions=fused - 1,
            block=self.block,
            num_warps=self.num_warps,
            **self.launch_options,
        )
        return Sampled(actions=actions, logprobs=logprobs, values=values)


def philox_uniforms(seed: int, subsequence: int, count: int) -> NDArray[np.float32]:
    """Return the first ``count`` ``curand_uniform`` draws of one stream, in numpy.

    The reference the Triton kernel is checked against (it matched curand at
    32,768 draws).

    Args:
      seed: ``curand_init``'s 64-bit seed.
      subsequence: ``curand_init``'s subsequence; the agent index.
      count: Draws.

    Returns:
      uniforms: ``float32 [count]``.

    """
    return philox_uniform(
        seed,
        agents=np.full(count, subsequence, dtype=np.uint64),
        draws=np.arange(count, dtype=np.uint64),
    )


def philox_uniform(
    seed: int,
    *,
    agents: NDArray[np.uint64],
    draws: NDArray[np.uint64],
) -> NDArray[np.float32]:
    """Return draw ``draws[i]`` of agent ``agents[i]``'s stream, for every ``i``.

    Args:
      seed: ``curand_init``'s 64-bit seed.
      agents: ``uint64``, each a ``curand_init`` subsequence.
      draws: ``uint64``, the same shape: which draw of each stream.

    Returns:
      uniforms: ``float32``, the same shape.

    """
    words = _philox(
        (
            (draws // 4) & 0xFFFFFFFF,
            np.zeros_like(draws),
            agents & 0xFFFFFFFF,
            np.zeros_like(draws),
        ),
        (np.full_like(draws, seed & 0xFFFFFFFF), np.full_like(draws, seed >> 32)),
    )
    word = np.choose((draws % 4).astype(np.intp), words)
    # As ``curand_uniform`` compiles (``sample_logits.sass``): the word rounded
    # to fp32, then one fma ``x * 2^-32 + 2^-33``, which the double product and
    # sum reproduce exactly before the single rounding.
    rounded = word.astype(np.float32).astype(np.float64)
    return (rounded * 2.0**-32 + 2.0**-33).astype(np.float32)


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutStorage:
    """One slot's rollout, time-major.

    Attributes:
      observations: ``[rows, agents, observation_size]``: ``rows`` is the
        horizon, plus one when the rollout stores its bootstrap row. The
        observations, terminals and mask are in the policy's dtype (bf16 for
        exp000); the log-probabilities, values and rewards in the rollout's.
      actions: ``[rows, agents]`` fp32.
      logprobs: ``[rows, agents]``.
      values: ``[rows, agents]``.
      rewards: ``[rows, agents]``, unclamped.
      terminals: ``[rows, agents]``.
      action_mask: ``[rows, agents, num_actions]``.
      initial_states: ``[layers, agents, width]`` in the carry's dtype, the
        carry at step 0.
      branch_starts: ``[agents]`` uint8, 1 where step 0 of the rollout is a
        practice restore; zeros without practice.
      features: ``[rows, agents, width]`` in the feature's dtype, the feature
        each step's policy read; None without a feature.
      frame_cells: With a joint feature, each step's frame cell tokens
        ``[rows, agents, 99, 8]`` uint8; else None, as every field below.
      frame_aux: Its aux tokens ``[rows, agents, 51]`` int16.
      previous_actions: The action that led to each step's observation, as
        the feature read it ``[rows, agents]`` uint8; 0 where the step begins
        the row's episode or window.
      context_decisions: The decisions each step's feature read, its own
        included ``[rows, agents]`` int16.
      context_anchored: Whether the episode's start leads them ``[rows,
        agents]`` bool.
      prefix_cells: Each row's decisions before step 0, oldest first and
        right-aligned ``[agents, P, 99, 8]``: ``P`` is one less than the
        feature's ``context_decisions``, and the first contexts reach no
        further back.
      prefix_aux: Their aux tokens ``[agents, P, 51]``.
      prefix_previous_actions: The actions that led to them ``[agents, P]``.
      prefix_decisions: How many of the ``P`` are the row's, the last ones,
        the rest zeros ``[agents]`` int16; 0 where step 0 begins the row's
        episode or window.

    """

    observations: Tensor
    actions: Tensor
    logprobs: Tensor
    values: Tensor
    rewards: Tensor
    terminals: Tensor
    action_mask: Tensor
    initial_states: Tensor
    branch_starts: Tensor
    features: Tensor | None = None
    frame_cells: Tensor | None = None
    frame_aux: Tensor | None = None
    previous_actions: Tensor | None = None
    context_decisions: Tensor | None = None
    context_anchored: Tensor | None = None
    prefix_cells: Tensor | None = None
    prefix_aux: Tensor | None = None
    prefix_previous_actions: Tensor | None = None
    prefix_decisions: Tensor | None = None

    @classmethod
    def allocate(
        cls,
        *,
        horizon: int,
        env: EnvBuffers,
        policy: Policy,
        device: torch.device,
        dtype: torch.dtype,
        feature_width: int = 0,
        context_decisions: int = 0,
    ) -> RolloutStorage:
        """Allocate one slot's buffers on ``device``.

        Args:
          horizon: Rows per slot: the steps, plus one for a bootstrap row.
          env: For the agent count and the observation and mask widths.
          policy: For the carry and the dtype of the observations, masks,
            terminals and features: the policy reads its feature in its own
            dtype, so storing it there keeps exactly what was read.
          device: Where the slot lives.
          dtype: The dtype of the log-probabilities, values and rewards.
          feature_width: Floats per row of the feature; 0 stores none.
          context_decisions: A joint feature's most decisions per context,
            for the learner's inputs; 0 stores none.

        Returns:
          storage: Zeroed buffers.

        """
        agents = env.num_envs

        def zeros(*shape: int, dtype: torch.dtype = policy.dtype) -> Tensor:
            return torch.zeros(*shape, dtype=dtype, device=device)

        inputs = (
            _learner_inputs(
                horizon=horizon,
                agents=agents,
                prefix=context_decisions - 1,
                device=device,
            )
            if context_decisions
            else {}
        )
        return cls(
            observations=zeros(horizon, agents, env.observations.shape[1]),
            actions=zeros(horizon, agents, dtype=torch.float32),
            logprobs=zeros(horizon, agents, dtype=dtype),
            values=zeros(horizon, agents, dtype=dtype),
            rewards=zeros(horizon, agents, dtype=dtype),
            terminals=zeros(horizon, agents),
            action_mask=zeros(horizon, agents, env.action_mask.shape[1]),
            initial_states=policy.initial_state(agents, device=device),
            branch_starts=zeros(agents, dtype=torch.uint8),
            features=zeros(horizon, agents, feature_width) if feature_width else None,
            **inputs,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CarrySaves:
    """Where one buffer's saving rows put their carries: practice's stored carry.

    Each step writes every saving row's carry, so the graph's shapes are
    static: a row that saved an entry into that entry, any other into its own
    dump row past the entries.

    Attributes:
      carries: ``[slots + dumps, layers, width]`` in the carry's dtype, the
        rollout's carry archive and then one dump row per row that can save.
      slots: The env's host ``save_slots`` of this buffer's saving rows.
      rows: Those rows, within the buffer.
      dumps: ``int64 [rows]`` on the device, each row's dump row.
      histories: The feature's history archive, rows as ``carries``'; None
        where the feature keeps none, or there is no feature.

    """

    carries: Tensor
    slots: Tensor
    rows: slice
    dumps: Tensor
    histories: dict[str, Tensor] | None = None


class StepGraph:
    """One buffer's step for one slot: captured once, replayed per step.

    Attributes:
      step: The device-side step index the graph reads and advances; the host
        zeroes it before a rollout.
      state: The buffer's carry, ``[layers, agents, width]``; advanced in
        place.
      draws: The buffer's sampler streams; advanced in place.
      saving: Where the buffer's saving rows put their carries; None when no
        row of it can save, and the graph is then practice's-off.
      feature: The buffer's feature, stepped between the ingest and the
        forward and stored per step; None reads none, and the graph is then
        the feature's-off.
      graphs: The captured graph of each feature plan; plan 0 without a
        feature.

    """

    def __init__(
        self,
        *,
        policy: Policy,
        sampler: Sampler,
        env: EnvBuffers,
        storage: RolloutStorage,
        buffer: int,
        stream: torch.cuda.Stream | None,
        saving: CarrySaves | None = None,
        feature: FeatureStep | None = None,
    ) -> None:
        self.policy = policy
        self.sampler = sampler
        self.env = env
        self.storage = storage
        self.buffer = buffer
        self.stream = stream
        self.saving = saving
        self.feature = feature
        self.rows = env.buffer_slice(buffer)
        device = storage.actions.device
        agents = len(range(*self.rows.indices(env.num_envs)))
        self.step = torch.zeros(1, dtype=torch.int64, device=device)
        self.state = policy.initial_state(agents, device=device)
        self.draws = sampler.draws(agents, device=device)
        # Device copies of the env's pinned rows: one upload per buffer, then
        # the casts read the device.
        self.uploaded_observations = torch.empty_like(
            env.observations[self.rows],
            device=device,
        )
        self.uploaded_rewards = torch.empty_like(env.rewards[self.rows], device=device)
        self.uploaded_terminals = torch.empty_like(
            env.terminals[self.rows],
            device=device,
        )
        self.uploaded_mask = torch.empty_like(env.action_mask[self.rows], device=device)
        self.uploaded_saves = torch.empty(
            0 if saving is None else len(range(*saving.rows.indices(agents))),
            dtype=torch.int32,
            device=device,
        )
        # The action each row last took, the history the feature reads with the
        # observation it led to: the sampler's last download, which the env read.
        self.uploaded_actions = torch.empty(
            0 if feature is None else agents,
            dtype=env.actions.dtype,
            device=device,
        )
        self.zero = torch.zeros((), dtype=self.state.dtype, device=device)
        # The rows the policy and the sampler read, in the storage's dtype,
        # written by the ingest.
        dtype = storage.observations.dtype
        self.observations = torch.empty(
            self.uploaded_observations.shape,
            dtype=dtype,
            device=device,
        )
        self.mask = torch.empty(self.uploaded_mask.shape, dtype=dtype, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        if stream is not None:
            # The counter, the carry and the draw counts above were zeroed on
            # this thread's stream. A replay stream that ran ahead read the
            # memory's stale values, and a stale counter stored rows past the
            # horizon: an illegal address under GPU load (measured).
            stream.wait_stream(torch.cuda.current_stream(device))

    def replay(self) -> None:
        """Run the step: the captured graph on its stream on CUDA, the ops eagerly elsewhere."""
        if self.stream is None:
            self._step()
            return
        # ``CUDAGraph.replay`` launches on the CURRENT stream, which is the
        # graph's only if the caller made it so.
        with torch.cuda.stream(self.stream):
            self._captured(self.stream).replay()

    def launch_handles(self) -> tuple[np.uint64, np.uint64]:
        """Return the captured graph's executable and its stream, for ``cuGraphLaunch``.

        Returns:
          graph_exec: The ``cudaGraphExec_t``; the graph owns it.
          stream: The ``cudaStream_t`` the graph replays on.

        Raises:
          ValueError: The step has no stream, so it runs eagerly and has no graph.

        """
        if self.stream is None:
            raise ValueError("an eager step graph has nothing to launch")
        graph = self._captured(self.stream)
        # ``np.uint64``: the env's Numba loop hands them to a uint64 cfunc.
        return (
            np.uint64(graph.raw_cuda_graph_exec()),
            np.uint64(self.stream.cuda_stream),
        )

    @torch.no_grad()
    def compute(self) -> Sampled:
        """Run the step's device work between its uploads and its download.

        The casts and stores of the uploaded rows, the carry's reset and
        snapshot, the forward, the sampler and its stores.

        Returns:
          sampled: The step's actions, log-probabilities and values.

        """
        storage, rows, step = self.storage, self.rows, self.step
        cuda = step.device.type == "cuda"
        if cuda:
            self._ingest_triton()
        else:
            self._ingest_torch()
        if self.saving is not None:
            _save_carries(self.saving, self.state, self.uploaded_saves)
            self._save_histories()
        features = None
        if self.feature is not None:
            if storage.features is None:
                raise ValueError("Expected storage.features is not None.")
            features = self.feature(
                self.uploaded_observations,
                self.uploaded_terminals,
                self.uploaded_actions,
            )
            # The store rounds to the policy's dtype, as its forward does: the
            # learner reads exactly what the actor read.
            storage.features[:, rows].index_copy_(
                0,
                step,
                features.to(storage.features.dtype)[None],
            )
            if storage.frame_cells is not None:
                self._store_inputs()
        # The carry is already reset for this step, so the policy skips its own,
        # and it advances in place.
        decoded, _ = self.policy.forward_fused(
            self.observations,
            self.state,
            None,
            carry=self.state,
            features=features,
        )
        sampled = self.sampler(
            decoded,
            self.mask,
            self.draws,
            buffer=self.buffer,
            dtype=storage.logprobs.dtype,
        )
        if cuda:
            self._emit_triton(sampled)
        else:
            storage.actions[:, rows].index_copy_(0, step, sampled.actions[None])
            storage.logprobs[:, rows].index_copy_(0, step, sampled.logprobs[None])
            storage.values[:, rows].index_copy_(0, step, sampled.values[None])
        step.add_(1)
        return sampled

    @torch.no_grad()
    def scatter_tail(self) -> None:
        """Keep the carries of the saves the buffer's last env step made.

        No replay of this rollout follows that step to take them, and the next
        rollout's first follows a prepare that clears the save slots. The
        carry is the one the next forward would read: zero where the step
        ended an episode; so is the feature's history. Call it after the
        rollout's last env step, on the buffer's stream; it does nothing where
        no row can save.
        """
        if self.saving is None:
            return
        self.uploaded_saves.copy_(self.saving.slots, non_blocking=True)
        self.uploaded_terminals.copy_(self.env.terminals[self.rows], non_blocking=True)
        if self.saving.histories is not None:
            self.uploaded_actions.copy_(
                self.env.actions[self.rows, 0],
                non_blocking=True,
            )
        carry = torch.where(
            self.uploaded_terminals[None, :, None] != 0,
            self.zero,
            self.state,
        )
        _save_carries(self.saving, carry, self.uploaded_saves)
        self._save_histories()

    # Every plan's graph shares the first one's pool: the buffer replays one graph at a
    # time, and no tensor a capture allocates outlives its replay -- each step writes
    # its results into tensors allocated before any capture.
    def _captured(self, stream: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
        """Return the feature plan's graph, capturing it on first use."""
        plan = 0 if self.feature is None else self.feature.plan
        graph = self.graphs.get(plan)
        if graph is None:
            with CAPTURE_LOCK:
                graph = self.graphs[plan] = self._capture(stream)
        return graph

    # Each warmup step is the step the next replay takes -- its counter, the carry it
    # reads -- so the warmup writes only the rows that replay rewrites (the initial
    # states only at step 0), never row ``horizon``, and any carry it saves into the
    # archive is the one that step saves. A feature's new plan is captured mid-rollout:
    # warming up at step 0 there overwrote the slot's first row and initial states.
    # Its effects on the counter, the carry, the draw counts, the feature's state and
    # the env's actions are undone. The env's actions are the previous action the next
    # step's feature reads: left as the warmup's draw, a graph captured mid-episode
    # (the second slot's) fed every row an action it never played. The archive, shared
    # with the other buffers' running steps, cannot be put back without racing them.
    def _capture(self, stream: torch.cuda.Stream) -> torch.cuda.CUDAGraph:
        """Warm up on the stream, then capture; the warmup's side effects are undone."""
        graph = torch.cuda.CUDAGraph()
        pool = next(iter(self.graphs.values())).pool() if self.graphs else None
        with torch.no_grad(), torch.cuda.stream(stream):
            state = [self.step, self.state, self.draws]
            if self.feature is not None:
                state += self.feature.capture_state()
            saved = [(value, value.clone()) for value in state]
            actions = self.env.actions[self.rows]
            played = actions.clone()
            for _ in range(2):
                for value, original in saved:
                    value.copy_(original)
                self._step()
            stream.synchronize()
            # Thread-local capture: the other buffers' streams keep working
            # while this one records.
            with torch.cuda.graph(
                graph,
                pool=pool,
                stream=stream,
                capture_error_mode="thread_local",
            ):
                self._step()
            for value, original in saved:
                value.copy_(original)
            # A host copy, so it runs now: after the synchronize above, which
            # landed the warmup's downloads, and before any replay uploads it.
            actions.copy_(played)
        stream.synchronize()
        return graph

    @torch.no_grad()
    def _step(self) -> None:
        """Step this buffer: upload, compute, download."""
        env, rows = self.env, self.rows
        self.uploaded_observations.copy_(env.observations[rows], non_blocking=True)
        self.uploaded_rewards.copy_(env.rewards[rows], non_blocking=True)
        self.uploaded_terminals.copy_(env.terminals[rows], non_blocking=True)
        self.uploaded_mask.copy_(env.action_mask[rows], non_blocking=True)
        if self.saving is not None:
            self.uploaded_saves.copy_(self.saving.slots, non_blocking=True)
        if self.feature is not None:
            self.uploaded_actions.copy_(env.actions[rows, 0], non_blocking=True)
        sampled = self.compute()
        env.actions[rows].copy_(sampled.actions[:, None], non_blocking=True)

    def _ingest_torch(self) -> None:
        """Round the uploads into row ``step``, reset the carry, snapshot at t == 0."""
        storage, rows, step = self.storage, self.rows, self.step
        self.observations.copy_(self.uploaded_observations)
        self.mask.copy_(self.uploaded_mask)
        storage.observations[:, rows].index_copy_(0, step, self.observations[None])
        storage.rewards[:, rows].index_copy_(
            0,
            step,
            self.uploaded_rewards.to(storage.rewards.dtype)[None],
        )
        storage.terminals[:, rows].index_copy_(
            0,
            step,
            self.uploaded_terminals.to(storage.terminals.dtype)[None],
        )
        storage.action_mask[:, rows].index_copy_(0, step, self.mask[None])
        # ``zero_term_state``, then the t == 0 snapshot as a masked copy.
        torch.where(
            self.uploaded_terminals[None, :, None] != 0,
            self.zero,
            self.state,
            out=self.state,
        )
        torch.where(
            step[:, None, None] == 0,
            self.state,
            storage.initial_states[:, rows],
            out=storage.initial_states[:, rows],
        )

    def _ingest_triton(self) -> None:
        """Do what :meth:`_ingest_torch` does, in one launch."""
        storage, rows = self.storage, self.rows
        agents, obs_size = self.uploaded_observations.shape
        num_actions = self.uploaded_mask.shape[1]
        layers, _, width = self.state.shape
        block = 1024
        counts = [
            -(-agents * obs_size // block),
            -(-agents * num_actions // block),
            -(-agents // block),
            -(-layers * agents * width // block),
        ]
        _kernels().ingest[(sum(counts),)](
            self.uploaded_observations,
            self.uploaded_mask,
            self.uploaded_rewards,
            self.uploaded_terminals,
            self.state,
            self.step,
            self.observations,
            self.mask,
            storage.observations[:, rows],
            storage.action_mask[:, rows],
            storage.rewards[:, rows],
            storage.terminals[:, rows],
            storage.initial_states[:, rows],
            agents,
            obs_size,
            num_actions,
            width,
            layers,
            counts[0],
            counts[1],
            counts[2],
            storage.observations.shape[1],
            storage.observations.shape[0],
            block=block,
            num_warps=4,
        )

    def _store_inputs(self) -> None:
        """Store the step's frame, previous action and context, for a joint learner."""
        if self.feature is None:
            raise ValueError("Expected self.feature is not None.")
        storage, rows, step = self.storage, self.rows, self.step
        recent = self.feature.last_decisions(1)
        decisions, anchored = self.feature.context()
        for store, value in (
            (storage.frame_cells, recent.cells[:, -1]),
            (storage.frame_aux, recent.aux[:, -1]),
            (storage.previous_actions, recent.previous_actions[:, -1]),
            (storage.context_decisions, decisions),
            (storage.context_anchored, anchored),
        ):
            if store is None:
                raise ValueError("Expected store is not None.")
            store[:, rows].index_copy_(0, step, value.to(store.dtype)[None])

    # Before the feature steps: the history saved is the one its next step extends,
    # as the carry saved is the one the next forward reads.
    def _save_histories(self) -> None:
        """Write the saving rows' feature histories beside their carries, if kept."""
        saving = self.saving
        if saving is None or saving.histories is None:
            return
        if self.feature is None:
            raise ValueError("Expected self.feature is not None.")
        self.feature.save_history(
            saving.histories,
            torch.where(self.uploaded_saves >= 0, self.uploaded_saves, saving.dumps),
            saving.rows,
            previous_action=self.uploaded_actions[saving.rows],
            fresh=self.uploaded_terminals[saving.rows],
        )

    def _emit_triton(self, sampled: Sampled) -> None:
        """Store the sampler's three rows at ``step`` in one launch."""
        storage, rows = self.storage, self.rows
        agents = sampled.actions.shape[0]
        block = 1024
        _kernels().emit[(-(-agents // block),)](
            sampled.actions,
            sampled.logprobs,
            sampled.values,
            self.step,
            storage.actions[:, rows],
            storage.logprobs[:, rows],
            storage.values[:, rows],
            agents,
            storage.actions.shape[1],
            storage.actions.shape[0],
            block=block,
            num_warps=4,
        )


class Rollout:
    """Slots of storage and a step graph per (slot, buffer)."""

    class Config(Fig["Rollout"]):
        """Slots and the per-buffer worker."""

        num_slots: int = 2
        """Rollout slots: with two, one collects while the learner reads the
        other; with one, the train step collects before it learns."""

        horizon: int = 256
        """Steps per rollout."""

        bootstrap: bool = False
        """Store one row past the horizon: the observation the last step leads
        to, the reward and terminal it arrived with, and the policy's value of
        it -- the bootstrap an advantage over the whole rollout starts from.
        The action sampled there is not played: the next rollout samples that
        observation again. A carry the row's forward advanced is put back, so
        the next rollout steps from the carry the last step left; the row's
        value reads it reset where the last step ended an episode."""

        dtype: torch.dtype = torch.float32
        """The stored log-probabilities', values' and rewards' dtype, and so
        the learning rule's advantages and returns. bf16 resolves a value near
        0.85 to 2^-8 and a log-probability near -3 to 2^-6, so a bf16 old
        log-probability alone moves an on-policy ratio up to 0.8% off 1."""

    def __init__(
        self,
        config: Config,
        *,
        policy: Policy,
        sampler: Sampler,
        env: EnvBuffers,
        device: torch.device,
        feature: FeatureSource | None = None,
    ) -> None:
        """Allocate the slots, the streams and a step graph per (slot, buffer).

        Args:
          config: Slots, horizon and the stored dtype.
          policy: The actor; the graphs address its parameters.
          sampler: The action streams.
          env: The environments' buffers.
          device: Where the slots and the graphs live.
          feature: A per-step feature each buffer's step computes from its
            observation and history, stores per step and hands the policy; it
            makes each buffer's step. None computes none.

        Raises:
          ValueError: ``num_slots`` or ``horizon`` is not positive, or a
            bootstrap row is asked of a rollout with a feature.

        """
        if config.num_slots <= 0 or config.horizon <= 0:
            raise ValueError("num_slots and horizon must be positive")
        if feature is not None and config.bootstrap:
            raise ValueError(
                "a feature's history advances once per observation: a bootstrap "
                "row would step it twice",
            )
        self._practice = (
            env if isinstance(env, PracticeEnv) and env.carry_slots else None
        )
        self.horizon = config.horizon
        self.bootstrap = config.bootstrap
        self.env = env
        cuda = device.type == "cuda"
        self.streams = [
            rollout_stream(device) if cuda else None for _ in range(env.num_buffers)
        ]
        self.feature = feature
        """The rollout's feature source; None without a feature."""
        self.engines = (
            []
            if self.feature is None
            else [
                self.feature.make_engine(
                    rows=env.num_envs // env.num_buffers,
                    device=device,
                )
                for _ in range(env.num_buffers)
            ]
        )
        """Each buffer's feature step, shared by its slots' graphs."""
        self._telemetry: list[list[dict[str, float]]] = [
            [] for _ in range(env.num_buffers)
        ]
        self.slots = [
            RolloutStorage.allocate(
                horizon=config.horizon + 1 if config.bootstrap else config.horizon,
                env=env,
                policy=policy,
                device=device,
                dtype=config.dtype,
                feature_width=0 if self.feature is None else self.feature.width,
                context_decisions=(
                    self.feature.context_decisions
                    if self.feature is not None and self.feature.joint
                    else 0
                ),
            )
            for _ in range(config.num_slots)
        ]
        self.archive: Tensor | None = None
        """The carry each practice entry saved, ``[carry_slots, layers, width]``
        in the carry's dtype: the one the donor's next forward read."""
        self.histories: dict[str, Tensor] | None = None
        """The feature's history of each practice entry, then of each saving
        row's dump row; None without practice, or where the feature keeps
        none. Not part of :meth:`state_dict`."""
        savings: list[CarrySaves | None] = [None] * env.num_buffers
        if self._practice is not None:
            initial = self.slots[0].initial_states
            save_rows = self._practice.save_rows
            saved = len(range(*save_rows.indices(env.num_envs)))
            entries = self._practice.carry_slots + saved
            carries = initial.new_zeros(entries, initial.shape[0], initial.shape[-1])
            self.archive = carries[: self._practice.carry_slots]
            if self.feature is not None:
                self.histories = self.feature.history_archive(entries, device=device)
            savings = [
                _carry_saves(
                    self._practice,
                    carries,
                    env.buffer_slice(buffer),
                    histories=self.histories,
                )
                for buffer in range(env.num_buffers)
            ]
        self.graphs = [
            [
                StepGraph(
                    policy=policy,
                    sampler=sampler,
                    env=env,
                    storage=storage,
                    buffer=buffer,
                    stream=self.streams[buffer],
                    saving=savings[buffer],
                    feature=self.engines[buffer] if self.engines else None,
                )
                for buffer in range(env.num_buffers)
            ]
            for storage in self.slots
        ]
        # The carry and the streams belong to the buffer, not the slot: the
        # second slot's graphs share them with the first's.
        for graphs in self.graphs[1:]:
            for graph, first in zip(graphs, self.graphs[0], strict=True):
                graph.state = first.state
                graph.draws = first.draws
        self.workers = ThreadPoolExecutor(max_workers=env.num_buffers)

    def collect(self, slot: int) -> RolloutStorage:
        """Run one horizon into ``slot``, every buffer on its own thread.

        With practice, the rows the env restored first take their entries'
        carries, before any buffer's first step: the donors' buffer saves into
        the archive during the rollout, and could otherwise overwrite an entry
        another buffer has yet to read. Each buffer then keeps the saves of its
        last step (:meth:`StepGraph.scatter_tail`) before this returns.

        With a feature, each buffer steps in blocks of at most the source's
        ``hook_interval`` and makes its feature room before each block
        (:meth:`feature_metrics` reports the blocks). A joint feature's rows
        store their prefixes after the restores.

        Args:
          slot: Which slot's storage to fill.

        Returns:
          storage: The slot, filled.

        """
        if self._practice is not None:
            assert isinstance(self.archive, Tensor)
            self._start_branches(slot, self._practice.restore_slots, self.archive)
        if self.slots[slot].prefix_cells is not None:
            self._store_prefixes(slot)
        for telemetry in self._telemetry:
            telemetry.clear()
        futures = [
            self.workers.submit(self._collect_buffer, slot, buffer)
            for buffer in range(self.env.num_buffers)
        ]
        for future in futures:
            future.result()
        return self.slots[slot]

    @torch.no_grad()
    def rebuild_features(self) -> None:
        """Rebuild every buffer's feature history under its source's new weights.

        Call it between rollouts, once the joint source's ``model`` holds the
        published weights, copied into its tensors in place: the captured
        graphs address them. Each buffer's rows re-encode their frames and
        re-prefill their contexts (``FeatureEngine.rebuild``).

        Raises:
          ValueError: The rollout has no joint feature.

        """
        if self.feature is None or not self.feature.joint:
            raise ValueError("Only a joint feature's history is rebuilt.")
        # Under the capture lock: a rebuild's shapes compile on first use, as a
        # re-prefill's do (``_make_room``).
        with self._between_steps(), CAPTURE_LOCK:
            for engine in self.engines:
                engine.rebuild()

    def feature_metrics(self) -> dict[str, float]:
        """Return the last collect's feature telemetry; empty without a feature.

        Call it between rollouts. Every block's ``ensure_room`` of every buffer
        reports; the re-prefilled rows are summed, every other value averaged.

        Returns:
          metrics: ``feature/*`` by name.

        """
        reports = [report for buffer in self._telemetry for report in buffer]
        names = sorted({name for report in reports for name in report})
        metrics: dict[str, float] = {}
        for name in names:
            values = [report[name] for report in reports if name in report]
            total = sum(values)
            metrics[name] = (
                total if name == "feature/reprefill_rows" else total / len(values)
            )
        return metrics

    def close(self) -> None:
        """Stop the worker threads and free the feature's state."""
        self.workers.shutdown()
        for engine in self.engines:
            engine.release()

    def reset(self) -> None:
        """Restart every buffer's carry and draw counts at zero, as construction does.

        Each zeroing runs on its buffer's stream, where the next replay reads it.
        A feature begins an episode in every row.
        """
        for graph in self.graphs[0]:
            with (
                nullcontext()
                if graph.stream is None
                else torch.cuda.stream(graph.stream)
            ):
                graph.state.zero_()
                graph.draws.zero_()
                if graph.feature is not None:
                    graph.feature.reset()

    def state_dict(self, *, slot: int) -> dict[str, Tensor]:
        """Return what a resumed rollout needs: the carries, the draw counts, a slot.

        Call it between rollouts; no graph is then in flight.

        Args:
          slot: The slot whose rows are saved: the one the learner reads next.

        Returns:
          state: ``carry``, ``[layers, num_envs, width]``, and ``draws``,
            ``[num_envs]``, the buffers' in order; the slot's
            :class:`RolloutStorage` tensors by field name, not copied, the
            ``features`` only with a feature; and with practice the carry
            ``archive``. A feature's history is not saved.

        """
        first = self.graphs[0]
        storage = self.slots[slot]
        state: dict[str, Tensor] = {
            "carry": torch.cat([graph.state for graph in first], dim=-2),
            "draws": torch.cat([graph.draws for graph in first]),
            **{name: getattr(storage, name) for name in self._slot_fields()},
        }
        if self.archive is not None:
            state["archive"] = self.archive
        return state

    def load_state_dict(self, state: Mapping[str, Tensor], *, slot: int) -> None:
        """Copy a :meth:`state_dict` into the live carries, draw counts and slot.

        The tensors are copied, not rebound: the captured graphs address them.
        A feature's history was not saved, so every row's next step begins a
        mid-episode window, as a training window that does not begin its
        episode reads.

        Args:
          state: What :meth:`state_dict` returned.
          slot: The slot to fill.

        """
        first = self.graphs[0]
        counts = [graph.draws.shape[0] for graph in first]
        for graph, carry, draws in zip(
            first,
            state["carry"].split(counts, dim=-2),
            state["draws"].split(counts),
            strict=True,
        ):
            graph.state.copy_(carry)
            graph.draws.copy_(draws)
        storage = self.slots[slot]
        for name in self._slot_fields():
            cast("Tensor", getattr(storage, name)).copy_(state[name])
        if self.archive is not None:
            self.archive.copy_(state["archive"])
        for engine in self.engines:
            engine.begin_window(
                torch.ones(self.env.num_envs // self.env.num_buffers, dtype=torch.bool),
            )
        # The copies ran on this thread's stream; the replays run on the
        # buffers', which would otherwise read stale values (the race
        # ``StepGraph.__init__`` guards against).
        for stream in self.streams:
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(stream.device))

    # On CUDA with an env that has ``run_buffer`` (design B) the loop is one ``nogil``
    # call per block: the env's loop launches the step graph and waits for it through
    # ``_graph_launcher`` before every step, so no Python runs per step. Otherwise
    # (design A) each step is three Python calls, each taking the GIL. Without a
    # feature the horizon is one block.
    def _collect_buffer(self, slot: int, buffer: int) -> None:
        """Replay, wait, step the env; ``horizon`` times, then the bootstrap row's."""
        graph = self.graphs[slot][buffer]
        stream = graph.stream
        if stream is None:
            graph.step.zero_()
            for steps in self._blocks():
                self._make_room(graph, buffer, steps)
                for _ in range(steps):
                    graph.replay()
                    self.env.step_buffer(buffer)
            if self.bootstrap:
                _replay_bootstrap(graph)
            graph.scatter_tail()
            return
        # The reset goes on the replay's stream too. Issued on the thread's
        # default stream it raced the first replay, which then read the last
        # rollout's count and stored past the horizon.
        with torch.cuda.stream(stream):
            graph.step.zero_()
            for steps in self._blocks():
                self._make_room(graph, buffer, steps)
                if isinstance(self.env, NogilEnv):
                    self.env.run_buffer(
                        buffer,
                        steps,
                        _graph_launcher(),
                        graph.launch_handles(),
                    )
                else:
                    for _ in range(steps):
                        graph.replay()
                        stream.synchronize()
                        self.env.step_buffer(buffer)
            if self.bootstrap:
                # The env is not stepped: the actions this replay writes back
                # are overwritten by the next rollout's first replay.
                _replay_bootstrap(graph)
                stream.synchronize()
            if graph.saving is not None:
                graph.scatter_tail()
                stream.synchronize()

    def _blocks(self) -> list[int]:
        """Return the horizon's steps in blocks of at most the feature's hook interval."""
        if self.feature is None:
            return [self.horizon]
        interval = self.feature.hook_interval
        blocks = [interval] * (self.horizon // interval)
        return [*blocks, self.horizon % interval] if self.horizon % interval else blocks

    # On the replay's stream, which the caller made current: ``ensure_room`` reads the
    # device once, and its re-prefill must land before the block's first step. Under
    # the capture lock: a feature's first re-prefill compiles, and Inductor's autotuning
    # synchronizes the whole device, which invalidated another buffer's capture in
    # flight (measured in a smoke run).
    def _make_room(self, graph: StepGraph, buffer: int, steps: int) -> None:
        """Make the buffer's feature room for the next ``steps`` steps; record its report."""
        if graph.feature is not None:
            with CAPTURE_LOCK:
                self._telemetry[buffer].append(graph.feature.ensure_room(steps))

    @torch.no_grad()
    def _start_branches(
        self,
        slot: int,
        restore_slots: Tensor,
        archive: Tensor,
    ) -> None:
        """Give restored rows their entries' carries and histories; mark them."""
        with self._between_steps():
            # A blocking copy: the env's first step clears the host rows.
            restores = restore_slots.to(archive.device, dtype=torch.int64)
            branches = restores >= 0
            self.slots[slot].branch_starts.copy_(branches)
            carries = archive[restores.clamp(min=0)].transpose(0, 1)
            for graph in self.graphs[slot]:
                rows = graph.rows
                torch.where(
                    branches[None, rows, None],
                    carries[:, rows],
                    graph.state,
                    out=graph.state,
                )
                if graph.feature is not None:
                    self._restore_history(graph.feature, restores[rows], rows=rows)

    @torch.no_grad()
    def _store_prefixes(self, slot: int) -> None:
        """Store each row's decisions before the slot's first step, for the learner."""
        storage = self.slots[slot]
        with self._between_steps():
            for graph in self.graphs[slot]:
                if graph.feature is None:
                    raise ValueError("Expected graph.feature is not None.")
                if storage.prefix_cells is None:
                    raise ValueError("Expected storage.prefix_cells is not None.")
                prefix = graph.feature.last_decisions(storage.prefix_cells.shape[1])
                for store, value in (
                    (storage.prefix_cells, prefix.cells),
                    (storage.prefix_aux, prefix.aux),
                    (storage.prefix_previous_actions, prefix.previous_actions),
                    (storage.prefix_decisions, prefix.count),
                ):
                    if store is None:
                        raise ValueError("Expected store is not None.")
                    store[graph.rows].copy_(value)

    # Every buffer's stream waits for the work, and the work for every buffer's last:
    # the buffers' steps write the state it reads and reads the state it writes.
    @contextmanager
    def _between_steps(self) -> Generator[None]:
        """Order the enclosed work, on the current stream, between the buffers'."""
        for stream in self.streams:
            if stream is not None:
                torch.cuda.current_stream(stream.device).wait_stream(stream)
        yield
        for stream in self.streams:
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(stream.device))

    # The env's actions were the last step's, which it has taken: the feature alone
    # reads them again, as the action that led to each row's next observation, which
    # for a restored row is the donor's. Under the capture lock: a re-prefill compiles
    # its shape on first use, as ``_make_room``'s does.
    def _restore_history(
        self,
        feature: FeatureStep,
        entries: Tensor,
        *,
        rows: slice,
    ) -> None:
        """Give a buffer's restored rows their entries' feature histories."""
        with CAPTURE_LOCK:
            previous = feature.restore_history(self.histories, entries)
        if previous is not None:
            actions = self.env.actions[rows, 0]
            actions.copy_(
                torch.where(entries >= 0, previous, actions.to(previous.device)),
            )

    # Without a carry archive no restore writes it, and leaving it out keeps the state,
    # and the goldens that digest it, as they were before practice; likewise the
    # features before the feature, and the learner's inputs before a joint one.
    def _slot_fields(self) -> list[str]:
        """Name the slot tensors a resume restores: those allocated, less one."""
        storage = self.slots[0]
        return [
            entry.name
            for entry in fields(RolloutStorage)
            if (entry.name != "branch_starts" or self.archive is not None)
            and getattr(storage, entry.name) is not None
        ]


def rollout_stream(device: torch.device) -> torch.cuda.Stream:
    """Return a stream at the device's highest priority, as every rollout's is.

    A rollout's short kernels then do not queue behind the learner's long
    ones.

    Args:
      device: A CUDA device.

    Returns:
      stream: A new stream on it.

    """
    greatest = torch.cuda.current_stream(device).priority_range()[1]
    return torch.cuda.Stream(device=device, priority=greatest)


def _learner_inputs(
    *,
    horizon: int,
    agents: int,
    prefix: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Return a joint learner's zeroed inputs by ``RolloutStorage`` field."""
    cells = (OBS_ROWS * OBS_COLS, OBS_TILE_CHANNELS)
    return {
        "frame_cells": torch.zeros(
            horizon,
            agents,
            *cells,
            dtype=torch.uint8,
            device=device,
        ),
        "frame_aux": torch.zeros(
            horizon,
            agents,
            INVENTORY_OBS_SIZE,
            dtype=torch.int16,
            device=device,
        ),
        "previous_actions": torch.zeros(
            horizon,
            agents,
            dtype=torch.uint8,
            device=device,
        ),
        "context_decisions": torch.zeros(
            horizon,
            agents,
            dtype=torch.int16,
            device=device,
        ),
        "context_anchored": torch.zeros(
            horizon,
            agents,
            dtype=torch.bool,
            device=device,
        ),
        "prefix_cells": torch.zeros(
            agents,
            prefix,
            *cells,
            dtype=torch.uint8,
            device=device,
        ),
        "prefix_aux": torch.zeros(
            agents,
            prefix,
            INVENTORY_OBS_SIZE,
            dtype=torch.int16,
            device=device,
        ),
        "prefix_previous_actions": torch.zeros(
            agents,
            prefix,
            dtype=torch.uint8,
            device=device,
        ),
        "prefix_decisions": torch.zeros(agents, dtype=torch.int16, device=device),
    }


def _carry_saves(
    env: PracticeEnv,
    carries: Tensor,
    rows: slice,
    *,
    histories: dict[str, Tensor] | None,
) -> CarrySaves | None:
    """Return where buffer ``rows``' saving rows put their carries; None if none can save."""
    row_start, row_stop, _ = rows.indices(env.num_envs)
    save_start, save_stop, _ = env.save_rows.indices(env.num_envs)
    start = max(row_start, save_start)
    stop = min(row_stop, save_stop)
    if start >= stop:
        return None
    dumps = env.carry_slots + start - save_start
    return CarrySaves(
        carries=carries,
        slots=env.save_slots[start:stop],
        rows=slice(start - row_start, stop - row_start),
        dumps=torch.arange(dumps, dumps + stop - start, device=carries.device),
        histories=histories,
    )


# Every saving row writes, so the shapes are static and the step graph captures it;
# the env never names one entry for two rows of a step, and each dump row is one row's,
# so no two writes of a call land on one row.
def _save_carries(saving: CarrySaves, carry: Tensor, saves: Tensor) -> None:
    """Write each saving row's carry into the entry it saved, else into its dump row."""
    saving.carries.index_copy_(
        0,
        torch.where(saves >= 0, saves, saving.dumps),
        carry[:, saving.rows].transpose(0, 1),
    )


# The bootstrap row's forward advances the carry in place, on the observation the next
# rollout's first step reads again. Put back, that step advances it once, from the carry
# the last step left, and its ingest resets the rows whose episodes ended, as the
# bootstrap row's did. On the caller's stream, the replay's: the three run in order.
def _replay_bootstrap(graph: StepGraph) -> None:
    """Replay the bootstrap row, then put back the carry its forward advanced."""
    carry = graph.state.clone()
    graph.replay()
    graph.state.copy_(carry)


# Every product is of two 32-bit words, so it fits a ``uint64`` exactly.
def _philox(
    counter: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    key: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Philox4x32-10, elementwise on ``uint64`` arrays of 32-bit words."""
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for round_index in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = (
            ((p1 >> 32) ^ c1 ^ k0) & 0xFFFFFFFF,
            p1 & 0xFFFFFFFF,
            ((p0 >> 32) ^ c3 ^ k1) & 0xFFFFFFFF,
            p0 & 0xFFFFFFFF,
        )
        if round_index < 9:
            k0, k1 = (k0 + 0x9E3779B9) & 0xFFFFFFFF, (k1 + 0xBB67AE85) & 0xFFFFFFFF
    return c0, c1, c2, c3


@dataclass(frozen=True, slots=True, kw_only=True)
class _StepKernels:
    ingest: triton.JITFunction[..., object]
    emit: triton.JITFunction[..., object]


# The driver's entry points, bound through ctypes, so a ``nogil`` Numba loop calls them
# without Python. Loaded on first use: only CUDA hosts have ``libcuda``. The calling
# thread's current context is the primary one, which the torch calls before the loop
# made current.
@lru_cache(maxsize=1)
def _graph_launcher() -> Callable[[np.uint64, np.uint64], int]:
    """Compile ``launch(graph_exec, stream)``: ``cuGraphLaunch``, then ``cuStreamSynchronize``."""
    driver = ctypes.CDLL("libcuda.so.1")
    cu_graph_launch = driver.cuGraphLaunch
    cu_graph_launch.argtypes = (ctypes.c_uint64, ctypes.c_uint64)
    cu_graph_launch.restype = ctypes.c_int
    cu_stream_synchronize = driver.cuStreamSynchronize
    cu_stream_synchronize.argtypes = (ctypes.c_uint64,)
    cu_stream_synchronize.restype = ctypes.c_int
    # Typed as the restype makes them: an int status.
    graph_launch = cast("Callable[[np.uint64, np.uint64], int]", cu_graph_launch)
    stream_synchronize = cast("Callable[[np.uint64], int]", cu_stream_synchronize)

    def launch(graph_exec: np.uint64, stream: np.uint64) -> int:
        status = int(graph_launch(graph_exec, stream))
        return status or int(stream_synchronize(stream))

    # A ``cfunc``: the env's cached loop takes it as a first-class function. A
    # jitted function holding these ctypes pointers would make that loop
    # uncacheable ("dynamic globals"), which the test suite raises as an error.
    return numba.cfunc("int32(uint64, uint64)")(launch)


@lru_cache(maxsize=1)
def _kernels() -> _StepKernels:
    """Jit the step's kernels once, on first use, so importing needs no Triton."""
    return _StepKernels(
        ingest=jit_kernel(_ingest_triton),
        emit=jit_kernel(_emit_triton),
    )


@lru_cache(maxsize=2)
def _sample_kernel(**helpers: Callable[..., object]) -> triton.JITFunction[..., object]:
    """Jit the sampler once per helper set, on first use; importing needs no Triton."""
    device = {
        "add_log_triton": add_log_triton,
        "_accumulate_exp_triton": _accumulate_exp_triton,
    } | helpers
    return jit_kernel(
        _sample_triton,
        _uniform_triton=jit_kernel(
            _uniform_triton,
            _philox_triton=jit_kernel(_philox_triton),
        ),
        masked_logsumexp_triton=jit_kernel(
            masked_logsumexp_triton,
            add_log_triton=jit_kernel(device["add_log_triton"]),
        ),
        _accumulate_exp_triton=jit_kernel(device["_accumulate_exp_triton"]),
    )


# Programs are sectioned: the observations, the mask, the two scalar rows, then the
# carry. No value is cast before its stores: a store rounds to its tensor's dtype, so the
# storage decides, bf16 for exp000 and fp32 for exp003, as the torch path rounds them.
# The carry rows of ended episodes are zeroed in place, and at ``step == 0`` the
# (zeroed) carry is also written to the initial states.
def _ingest_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    obs_ptr: language.tensor,
    mask_ptr: language.tensor,
    rewards_ptr: language.tensor,
    terminals_ptr: language.tensor,
    state_ptr: language.tensor,
    step_ptr: language.tensor,
    obs_out_ptr: language.tensor,
    mask_out_ptr: language.tensor,
    row_obs_ptr: language.tensor,
    row_mask_ptr: language.tensor,
    row_rewards_ptr: language.tensor,
    row_terminals_ptr: language.tensor,
    initial_ptr: language.tensor,
    agents: int,
    obs_size: int,
    num_actions: int,
    width: int,
    layers: int,
    obs_programs: int,
    mask_programs: int,
    scalar_programs: int,
    row_stride: int,
    horizon: int,
    block: language.constexpr,
) -> None:
    """One launch for the step's uploads: rounds, stores, resets, snapshots."""
    pid = language.program_id(0)
    step = language.load(step_ptr)
    # The storage has ``horizon`` rows; a store is bounded by them whatever the
    # counter holds.
    in_rows = step < horizon
    lane = language.arange(0, block)
    if pid < obs_programs:
        idx = pid * block + lane
        live = idx < agents * obs_size
        value = language.load(obs_ptr + idx, mask=live, other=0.0)
        language.store(obs_out_ptr + idx, value, mask=live)
        language.store(
            row_obs_ptr + step * (row_stride * obs_size) + idx,
            value,
            mask=live & in_rows,
        )
    elif pid < obs_programs + mask_programs:
        idx = (pid - obs_programs) * block + lane
        live = idx < agents * num_actions
        value = language.load(mask_ptr + idx, mask=live, other=0)
        language.store(mask_out_ptr + idx, value, mask=live)
        language.store(
            row_mask_ptr + step * (row_stride * num_actions) + idx,
            value,
            mask=live & in_rows,
        )
    elif pid < obs_programs + mask_programs + scalar_programs:
        idx = (pid - obs_programs - mask_programs) * block + lane
        live = idx < agents
        rewards = language.load(rewards_ptr + idx, mask=live, other=0.0)
        terminals = language.load(terminals_ptr + idx, mask=live, other=0.0)
        language.store(
            row_rewards_ptr + step * row_stride + idx,
            rewards,
            mask=live & in_rows,
        )
        language.store(
            row_terminals_ptr + step * row_stride + idx,
            terminals,
            mask=live & in_rows,
        )
    else:
        idx = (pid - obs_programs - mask_programs - scalar_programs) * block + lane
        per_layer = agents * width
        live = idx < layers * per_layer
        layer = idx // per_layer
        within = idx % per_layer
        agent = within // width
        terminal = language.load(terminals_ptr + agent, mask=live, other=0.0)
        carry = language.load(state_ptr + idx, mask=live, other=0.0)
        carry = language.where(terminal != 0.0, 0.0, carry)
        language.store(state_ptr + idx, carry, mask=live)
        language.store(
            initial_ptr + layer * (row_stride * width) + within,
            carry,
            mask=live & (step == 0),
        )


def _emit_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    actions_ptr: language.tensor,
    logprobs_ptr: language.tensor,
    values_ptr: language.tensor,
    step_ptr: language.tensor,
    row_actions_ptr: language.tensor,
    row_logprobs_ptr: language.tensor,
    row_values_ptr: language.tensor,
    agents: int,
    row_stride: int,
    horizon: int,
    block: language.constexpr,
) -> None:
    """Store the sampler's actions, log-probabilities and values at row ``step``."""
    idx = language.program_id(0) * block + language.arange(0, block)
    step = language.load(step_ptr)
    live = (idx < agents) & (step < horizon)
    at = step * row_stride + idx
    language.store(
        row_actions_ptr + at,
        language.load(actions_ptr + idx, mask=live),
        mask=live,
    )
    language.store(
        row_logprobs_ptr + at,
        language.load(logprobs_ptr + idx, mask=live),
        mask=live,
    )
    language.store(
        row_values_ptr + at,
        language.load(values_ptr + idx, mask=live),
        mask=live,
    )


def _sample_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    inputs: language.tuple,
    outputs: language.tuple,
    count: int,
    key0: int,
    key1: int,
    num_actions: language.constexpr,
    block: language.constexpr,
) -> None:
    """Sample one agent per lane: the masked inverse CDF at its next draw."""
    decoded_ptr, mask_ptr, draws_ptr = inputs
    actions_ptr, logprobs_ptr, values_ptr = outputs
    idx = language.program_id(0) * block + language.arange(0, block)
    lanes = idx < count
    fused = idx * (num_actions + 1)
    flat = idx * num_actions
    language.store(
        values_ptr + idx,
        language.load(decoded_ptr + fused + num_actions, mask=lanes),
        mask=lanes,
    )
    lse = masked_logsumexp_triton(
        decoded_ptr + fused,
        mask_ptr + flat,
        lanes,
        num_actions,
    )
    draw = language.load(draws_ptr + idx, mask=lanes, other=0)
    uniform = _uniform_triton(key0, key1, idx, draw)
    cumsum = language.zeros([block], dtype=language.float32)
    sampled = language.full([block], num_actions - 1, dtype=language.int32)
    chosen = language.zeros([block], dtype=language.float32)
    found = language.zeros([block], dtype=language.int32)
    last_legal = language.zeros([block], dtype=language.int32)
    last_logit = language.zeros([block], dtype=language.float32)
    for action in language.static_range(num_actions):
        logit = language.load(decoded_ptr + fused + action, mask=lanes, other=0.0)
        legal = language.load(mask_ptr + flat + action, mask=lanes, other=0.0)
        logit = language.where(
            legal.to(language.float32) == 0.0,
            -1e4,
            logit.to(language.float32),
        )
        cumsum = _accumulate_exp_triton(cumsum, logit - lse)
        hit = (uniform < cumsum) & (found == 0)
        sampled = language.where(hit, action, sampled)
        chosen = language.where(hit, logit, chosen)
        found = language.where(hit, 1, found)
        is_legal = legal.to(language.float32) != 0.0
        last_legal = language.where(is_legal, action, last_legal)
        last_logit = language.where(is_legal, logit, last_logit)
    # A fall-through lands on the last action, which may be masked; snap to
    # the last legal one. A legitimate last pick is legal, so this is exact.
    snap = sampled == num_actions - 1
    sampled = language.where(snap, last_legal, sampled)
    chosen = language.where(snap, last_logit, chosen)
    language.store(actions_ptr + idx, sampled.to(language.float32), mask=lanes)
    # The store rounds the log-probability to the decoder's dtype, as the value's is.
    language.store(logprobs_ptr + idx, chosen - lse, mask=lanes)
    language.store(draws_ptr + idx, draw + 1, mask=lanes)


def _uniform_triton(
    key0: int,
    key1: int,
    agent: language.tensor,
    draw: language.tensor,
) -> language.tensor:
    """Draw ``draw`` of agent ``agent``'s stream as ``curand_uniform`` maps it."""
    words = _philox_triton(
        (draw // 4).to(language.uint32),
        agent.to(language.uint32),
        (language.zeros(agent.shape, dtype=language.int64) + key0).to(language.uint32),
        (language.zeros(agent.shape, dtype=language.int64) + key1).to(language.uint32),
    )
    index = draw % 4
    word = language.where(index == 0, words[0], words[1])
    word = language.where(index == 2, words[2], word)
    word = language.where(index == 3, words[3], word)
    return language.fma(
        word.to(language.float32),
        2.3283064365386963e-10,
        1.1641532182693481e-10,
    )


def _philox_triton(
    counter0: language.tensor,
    counter2: language.tensor,
    key0: language.tensor,
    key1: language.tensor,
) -> tuple[language.tensor, language.tensor, language.tensor, language.tensor]:
    """Philox4x32-10 with counter ``(counter0, 0, counter2, 0)`` and key ``(key0, key1)``."""
    c0 = counter0
    c1 = language.zeros(counter0.shape, dtype=language.uint32)
    c2 = counter2
    c3 = language.zeros(counter0.shape, dtype=language.uint32)
    k0 = key0
    k1 = key1
    m0 = language.full(counter0.shape, 0xD2511F53, dtype=language.uint32)
    m1 = language.full(counter0.shape, 0xCD9E8D57, dtype=language.uint32)
    for round_index in language.static_range(10):
        hi0 = language.umulhi(m0, c0)
        lo0 = m0 * c0
        hi1 = language.umulhi(m1, c2)
        lo1 = m1 * c2
        c0 = hi1 ^ c1 ^ k0
        c1 = lo1
        c2 = hi0 ^ c3 ^ k1
        c3 = lo0
        if round_index < 9:
            k0 = k0 + 0x9E3779B9
            k1 = k1 + 0xBB67AE85
    return c0, c1, c2, c3


# ``PhiloxSampler.helpers`` replaces this by name to round the CDF's terms and their
# sum as another implementation does.
def _accumulate_exp_triton(
    total: language.tensor,
    a: language.tensor,
) -> language.tensor:
    """Return ``total + exp(a)``, the inverse CDF's next partial sum."""
    return total + language.exp(a)
