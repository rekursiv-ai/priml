"""World-model feature: a trained model's history state, step by step, for RL.

A ``FeatureEngine`` serves one actor buffer of ``B`` rows, each an environment
whose episode the world model reads as its training-time global
stream: ``start, obs_0, act_0, obs_1, ...`` from position 0, a ``start`` only
where the episode begins (``batch.py``). One call is one env step. It appends
``[start or act_{t-1}, obs_t]`` to every row's static KV cache in a single
two-query pass over the global stack and returns the final-normed hidden state
at ``obs_t``: the feature. The frames are the codec's tokens of the float32
observation (``codec.tokens_and_flags``), since the policy's bf16 copy cannot
hold the health grid above 10 HP. A call reads no device value and draws no
random number, so the rollout captures it inside its step graph.

Between blocks of steps, outside the graph, ``ensure_room`` reads the device
once and raises on a sticky counter (a step without room, an out-of-schema
frame, a non-finite feature). What it then does is the source's ``history``
slot: how a row's context is bounded once its episode outgrows the cache.
``Refill`` re-prefills each row that cannot take the next block from its last
``keep`` decisions at position 0: exactly a mid-episode training window.
``Sliding`` reads every decision as the last ``decisions`` decisions of its
history: a prefix of the episode (or of a mid-episode window) while it fits,
served by the cache, and once it slides, ``obs, act, ..., obs_t`` from
position 0 with no ``start``, recomputed every step from a ring of the last
``decisions`` decisions in a fallback batch of the rows that can slide in the
next block; ``ensure_room`` sizes that batch in whole chunks of
``fallback_rows``, and the rollout's step graph is captured once per size
(``FeatureEngine.plan``). ``begin_window`` restarts
rows as a mid-episode window from their next observation, for rows whose
history the engine no longer holds.

A practice restore puts a donor's world into a row; the ``practice`` slot
says what the row's history becomes. ``FreshWindow`` begins a mid-episode
window at its next step. ``DonorHistory`` resumes the donor's: each step
saves the donors' histories in-graph beside their carries
(``FeatureEngine.save_history``) into an archive the source allocates, and a
restore copies one into the row and re-prefills its context
(``FeatureEngine.restore_history``). Under ``Refill`` that context is what a
re-prefill at the save would have kept; under ``Sliding`` it is the donor's
own. The archive is not checkpointed: after a resume, a row restored from an
entry its donor has not saved again begins a fresh window.

A ``joint`` source's weights change between rollouts, published in place
into its ``model``: each row's rings also keep its longest context's frame
tokens, ``FeatureEngine.rebuild`` re-encodes them and re-prefills every
context under the new weights, a ``DonorHistory`` archive keeps frames rather
than their encodings, and ``FeatureEngine.last_decisions`` hands a learner
the frames and actions it recomputes the features from (``context_inputs``).

The math runs through the world model's own modules -- its table, frame
encoder, ``obs_proj``, ``action_embedding``, ``start``, global blocks and final
norm -- in the order the training forward runs them, with two departures that
change no value the training forward would read: the last encoder block
computes its attention output and MLP for the pooling position alone (the only
one the global stack reads), and the global attention reads the cache.
``WorldModelFeature`` loads the weights once per process and makes an engine
per buffer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, override, runtime_checkable

import copy
import dataclasses
import importlib

from configgle import Fig, Makeable
from torch import Tensor
from torch.nn import functional

import torch

from priml.baselines.craftax.game.state import (
    INVENTORY_OBS_SIZE,
    OBS_COLS,
    OBS_ROWS,
    OBS_SIZE,
    OBS_TILE_CHANNELS,
)
from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.checkpoint import (
    build_world_model,
    load_world_model,
)
from priml.baselines.craftax.world_model.codec import (
    encode,
    tokens_and_flags,
)
from priml.baselines.craftax.world_model.model import (
    FrameEncoder,
    GlobalTransformer,
    WorldModel,
)
from priml.lib.codec import from_plain
from priml.model.attention.attention import Attention, AttentionProjections
from priml.model.attention.flash4 import Flash4Varlen
from priml.model.attention.kernel import SdpaFused, SdpaVarlen
from priml.model.attention.rope import RoPE
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from torch import nn


class CacheAttention(Protocol):
    """Attention of each row's consecutive queries to its cache, up to its last query."""

    def __call__(
        self,
        q: Tensor,
        keys: Tensor,
        values: Tensor,
        seqused: Tensor,
    ) -> Tensor:
        """Attend ``[B, S, H, D]`` queries to ``[B, T, H_kv, D]`` cached keys and values.

        Query ``i`` of row ``r`` sits at position ``seqused[r] - S + i`` and sees
        keys ``0..seqused[r] - S + i``; the cache already holds the queries' own.
        """
        ...


class MaskedCacheAttention:
    """SDPA over each row's whole cache under its length mask: any device and dtype.

    The reference the gates run in float32; it reads every cache position, so
    its cost does not shrink with a row's length.
    """

    class Config(Fig["MaskedCacheAttention"]):
        """No options."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(
        self,
        q: Tensor,
        keys: Tensor,
        values: Tensor,
        seqused: Tensor,
    ) -> Tensor:
        """Attend as ``CacheAttention`` states, masking the keys past each query."""
        positions = torch.arange(keys.shape[1], device=keys.device)
        last = seqused[:, None] - q.shape[1] + torch.arange(q.shape[1], device=q.device)
        mask = positions <= last[..., None]
        return functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask=mask[:, None],
            enable_gqa=True,
        ).transpose(1, 2)


class Flash4CacheAttention:
    """FlashAttention 4 over the static cache, reading each row's first ``seqused`` keys.

    Bottom-right causal: the last query of a row sees ``seqused`` keys, so the
    ``obs`` query sees the ``act`` key written in the same step. bf16 or fp16 on
    Linux CUDA (``flash-attn-4``), imported when built.
    """

    class Config(Fig["Flash4CacheAttention"]):
        """No options: FA4 reads every shape from its inputs."""

    def __init__(self, config: Config) -> None:
        del config
        self._interface = _flash4()

    def __call__(
        self,
        q: Tensor,
        keys: Tensor,
        values: Tensor,
        seqused: Tensor,
    ) -> Tensor:
        """Attend as ``CacheAttention`` states."""
        result = self._interface.flash_attn_varlen_func(
            q.contiguous(),
            keys,
            values,
            seqused_k=seqused,
            causal=True,
        )
        return result[0] if isinstance(result, tuple) else result


class WorldModelWeights(Protocol):
    """Builds the feature's world model: float32 on the CPU, in eval mode."""

    def __call__(self) -> WorldModel:
        """Return a new model."""
        ...


class TrainedWeights:
    """A training checkpoint of a world-model experiment, in the port's layout."""

    class Config(Fig["TrainedWeights"]):
        """The experiment that trained the checkpoint, and the checkpoint."""

        experiment: str = "priml.baselines.craftax.world_model.experiments.exp001"
        """Dotted path of the experiment factory that trained the checkpoint."""

        checkpoint: Path = Path(
            "/opt/scratch/datasets/craftax/world-model-oracle/v1/checkpoints/"
            "exp001-s0/step_00001525.pt",
        )
        """A ``TrainLoop`` checkpoint of that experiment; the base model, seed 0."""

        overrides: list[str] = field(default_factory=list[str])
        """``PATH=VALUE`` overrides the run was launched with."""

    def __init__(self, config: Config) -> None:
        self.experiment = config.experiment
        self.checkpoint = config.checkpoint
        self.overrides = list(config.overrides)

    def __call__(self) -> WorldModel:
        """Load the checkpoint; the global generator is left where it was."""
        # Building the model draws its initialization from the global generator
        # before the checkpoint overwrites it; the policy's init must not move.
        with torch.random.fork_rng(devices=[]):
            model, _ = load_world_model(
                self.experiment,
                self.checkpoint,
                overrides=self.overrides,
            )
        return model


class InitialWeights:
    """A world-model experiment's freshly initialized weights, from a seed."""

    class Config(Fig["InitialWeights"]):
        """The experiment whose model is built, and the seed of its init."""

        experiment: str = "priml.baselines.craftax.world_model.experiments.exp001"
        """Dotted path of the experiment factory whose model is built."""

        overrides: list[str] = field(default_factory=list[str])
        """``PATH=VALUE`` overrides of that experiment."""

        seed: int = 0
        """Seed of the CPU generator the initialization draws from."""

    def __init__(self, config: Config) -> None:
        self.experiment = config.experiment
        self.overrides = list(config.overrides)
        self.seed = config.seed

    def __call__(self) -> WorldModel:
        """Build the model under its seed; the global generator is left where it was."""
        # Seeded through its state: ``torch.manual_seed`` would reseed CUDA too,
        # which ``fork_rng(devices=[])`` does not put back.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(self.seed).get_state())
            model, _ = build_world_model(self.experiment, overrides=self.overrides)
        return model.eval()


class History(Protocol):
    """How each row's context is bounded once its episode outgrows the cache.

    Attributes:
      context_decisions: The most decisions a context holds.
      hook_interval_bound: The most steps ``ensure_room`` can make room for.

    """

    context_decisions: int
    hook_interval_bound: int

    def ring(self, *, frames: bool) -> int:
        """Return the decisions each row's rings hold: a whole context with frames."""
        ...

    def engine(
        self,
        model: WorldModel,
        *,
        rows: int,
        frames: bool,
        layers: int,
        attention: CacheAttention,
        kernels: FeatureKernels,
        reprefill_rows: int,
    ) -> FeatureEngine:
        """Return an engine of ``rows`` rows that bounds each row's context this way.

        Args:
          model: The world model, on the engine's device and dtype.
          rows: Rows of the buffer.
          frames: Keep each row's raw frames for its longest context.
          layers: Global blocks up to the tap.
          attention: The step's attention over the cache.
          kernels: The step's per-block and encoder functions.
          reprefill_rows: Rows per re-prefill forward.

        Returns:
          engine: A fresh engine; every row begins an episode.

        """
        ...


class Refill:
    """A row that cannot take the next block restarts from its last ``keep`` decisions.

    Its context is then a mid-episode training window from position 0, which
    the following steps extend until the next re-prefill. A context therefore
    holds between ``keep`` and ``t_max / 2`` decisions once the episode is
    long, and the cache serves every step.

    Attributes:
      t_max: Global positions per row.
      keep: Decisions a re-prefill keeps.
      context_decisions: ``(t_max + 1) // 2``, a full row's.
      hook_interval_bound: The most steps a block after a re-prefill fits.

    """

    class Config(Fig["Refill"]):
        """The cache's positions and what a re-prefill keeps."""

        t_max: int = 1_024
        """Global positions per row; one more is scratch for a full row's writes."""

        keep: int = 256
        """Decisions a re-prefill keeps, from position 0."""

    def __init__(self, config: Config) -> None:
        """Check that a re-prefill leaves room for a step.

        Args:
          config: The positions and what a re-prefill keeps.

        Raises:
          ValueError: ``keep`` is not positive, or ``2 keep + 1 > t_max``.

        """
        if config.keep < 1 or 2 * config.keep + 1 > config.t_max:
            raise ValueError(
                f"keep={config.keep} needs 1 <= keep and 2 keep + 1 <= "
                f"t_max={config.t_max}.",
            )
        self.t_max = config.t_max
        self.keep = config.keep
        self.context_decisions = (config.t_max + 1) // 2
        self.hook_interval_bound = (config.t_max - (2 * config.keep - 1)) // 2

    def ring(self, *, frames: bool) -> int:
        """Return ``keep``, or with ``frames`` a full row's decisions, its rebuild's."""
        return self.context_decisions if frames else self.keep

    def engine(
        self,
        model: WorldModel,
        *,
        rows: int,
        frames: bool,
        layers: int,
        attention: CacheAttention,
        kernels: FeatureKernels,
        reprefill_rows: int,
    ) -> FeatureEngine:
        """Return an engine of ``rows`` rows that re-prefills them; see ``ring``.

        Args:
          model: The world model, on the engine's device and dtype.
          rows: Rows of the buffer.
          frames: Keep each row's raw frames.
          layers: Global blocks up to the tap.
          attention: The step's attention over the cache.
          kernels: The step's per-block and encoder functions.
          reprefill_rows: Rows per re-prefill forward.

        Returns:
          engine: A fresh engine; every row begins an episode.

        """
        return FeatureEngine(
            model,
            rows=rows,
            t_max=self.t_max,
            keep=self.keep,
            ring=self.ring(frames=frames),
            frames=frames,
            layers=layers,
            attention=attention,
            kernels=kernels,
            reprefill_rows=reprefill_rows,
        )


class Sliding:
    """Each decision reads exactly the last ``decisions`` decisions of its history.

    While they extend the episode's start (or a mid-episode window's), the
    cache serves the step; a row whose window slides is recomputed every
    step from its ring, as ``obs, act, ..., obs_t`` from position 0.

    Attributes:
      decisions: The window's decisions.
      context_decisions: The window's decisions.
      fallback_rows: Rows per forward of the sliding rows' recompute.
      hook_interval_bound: A block's most steps: one window.

    """

    class Config(Fig["Sliding"]):
        """The window and the recompute's chunk."""

        decisions: int = 512
        """Decisions per window, the current one included."""

        fallback_rows: int = 16
        """Rows per forward of the recompute: a step's fallback batch is whole
        chunks of exactly this many, so every forward has one shape."""

    def __init__(self, config: Config) -> None:
        """Check the window and the chunk.

        Args:
          config: The window and the recompute's chunk.

        Raises:
          ValueError: ``decisions`` or ``fallback_rows`` is not positive.

        """
        if config.decisions < 1 or config.fallback_rows < 1:
            raise ValueError(
                f"decisions={config.decisions} and fallback_rows="
                f"{config.fallback_rows} must be positive.",
            )
        self.decisions = self.context_decisions = config.decisions
        self.fallback_rows = config.fallback_rows
        self.hook_interval_bound = config.decisions

    def ring(self, *, frames: bool) -> int:
        """Return the window's decisions, with frames or without."""
        del frames
        return self.decisions

    def engine(
        self,
        model: WorldModel,
        *,
        rows: int,
        frames: bool,
        layers: int,
        attention: CacheAttention,
        kernels: FeatureKernels,
        reprefill_rows: int,
    ) -> SlidingEngine:
        """Return an engine of ``rows`` rows that slides their windows.

        Args:
          model: The world model, on the engine's device and dtype.
          rows: Rows of the buffer.
          frames: Keep each row's raw frames.
          layers: Global blocks up to the tap.
          attention: The step's attention over the cache.
          kernels: The step's per-block and encoder functions.
          reprefill_rows: Rows per re-prefill forward.

        Returns:
          engine: A fresh engine; every row begins an episode.

        """
        return SlidingEngine(
            model,
            rows=rows,
            decisions=self.decisions,
            fallback_rows=self.fallback_rows,
            frames=frames,
            layers=layers,
            attention=attention,
            kernels=kernels,
            reprefill_rows=reprefill_rows,
        )


class PracticeHistory(Protocol):
    """What a practice restore makes of a row's history, and the archive it needs."""

    def archive(
        self,
        *,
        entries: int,
        ring: int,
        width: int,
        frames: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Tensor] | None:
        """Return storage for ``entries`` saved histories; None saves none.

        Args:
          entries: Practice entries, then one dump row per saving row.
          ring: Decisions each history keeps.
          width: Width of an ``obs`` input.
          frames: Whether the engines keep raw frames: the weights change
            between rollouts, so a history keeps frames, not their inputs.
          dtype: The engine's dtype.
          device: The engine's device.

        Returns:
          archive: Named tensors with a leading ``entries`` axis, or None.

        """
        ...


class FreshWindow:
    """A restored row begins a mid-episode window at its next step; nothing is saved."""

    class Config(Fig["FreshWindow"]):
        """No options."""

    def __init__(self, config: Config) -> None:
        del config

    def archive(
        self,
        *,
        entries: int,
        ring: int,
        width: int,
        frames: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Return None: a restore begins a window, and no step saves.

        Args:
          entries: Practice entries, then one dump row per saving row.
          ring: Decisions each history keeps.
          width: Width of an ``obs`` input.
          frames: Whether the engines keep raw frames: the weights change
            between rollouts, so a history keeps frames, not their inputs.
          dtype: The engine's dtype.
          device: The engine's device.

        Returns:
          archive: None.

        """
        del entries, ring, width, frames, dtype, device


class DonorHistory:
    """A restored row resumes the history its donor had when it saved the entry."""

    class Config(Fig["DonorHistory"]):
        """No options."""

    def __init__(self, config: Config) -> None:
        del config

    def archive(
        self,
        *,
        entries: int,
        ring: int,
        width: int,
        frames: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, Tensor]:
        """Return zeroed histories; a zero ``length`` restores as a fresh window.

        Args:
          entries: Practice entries, then one dump row per saving row.
          ring: Decisions each history keeps.
          width: Width of an ``obs`` input.
          frames: Whether the engines keep raw frames: the weights change
            between rollouts, so a history keeps frames, not their inputs.
          dtype: The engine's dtype.
          device: The engine's device.

        Returns:
          archive: The rings: ``obs`` ``[entries, ring, width]``, or with
            ``frames`` the raw ``cells`` ``[entries, ring, 99, 8]`` and ``aux``
            ``[entries, ring, 51]`` a restore encodes afresh; ``actions``
            ``[entries, ring]``. The row counters ``count`` and ``length``, and
            ``previous_action``, the action that led to the donor's next frame.

        """
        rings = (
            {
                "cells": torch.zeros(
                    entries,
                    ring,
                    OBS_ROWS * OBS_COLS,
                    OBS_TILE_CHANNELS,
                    dtype=torch.uint8,
                    device=device,
                ),
                "aux": torch.zeros(
                    entries,
                    ring,
                    INVENTORY_OBS_SIZE,
                    dtype=torch.int16,
                    device=device,
                ),
            }
            if frames
            else {"obs": torch.zeros(entries, ring, width, dtype=dtype, device=device)}
        )
        return rings | {
            "actions": torch.zeros(entries, ring, dtype=torch.uint8, device=device),
            "count": torch.zeros(entries, dtype=torch.long, device=device),
            "length": torch.zeros(entries, dtype=torch.long, device=device),
            "previous_action": torch.zeros(entries, device=device),
        }


class WorldModelFeature:
    """Feature source: one world model per process, an engine per actor buffer.

    Attributes:
      model: The world model, float32 on the CPU until the first engine;
        frozen, unless a joint learner publishes its weights into it.
      width: Feature width, the global stack's.
      hook_interval: Steps the rollout takes between ``ensure_room`` calls.
      attention: The step's attention over the cache, shared by the engines.
      kernels: The step's per-block and encoder functions, shared by the engines.
      history: How each row's context is bounded.
      practice: What a practice restore makes of a row's history.
      joint: Whether the learner trains the weights between rollouts.
      context_decisions: The most decisions a context holds.
      ring: Decisions each engine row's rings hold, and each saved history.

    """

    class Config(Fig["WorldModelFeature"]):
        """The weights, the history, the tap and the kernels."""

        weights: Makeable[WorldModelWeights] = field(
            default_factory=TrainedWeights.Config,
        )
        """Where the weights come from."""

        history: Makeable[History] = field(default_factory=Refill.Config)
        """How each row's context is bounded once its episode outgrows the
        cache: ``Refill`` re-prefills, ``Sliding`` slides exactly."""

        practice: Makeable[PracticeHistory] = field(
            default_factory=FreshWindow.Config,
        )
        """What a practice restore makes of a row's history: ``FreshWindow``
        or ``DonorHistory``. Read only with practice."""

        joint: bool = False
        """The learner trains the weights, publishing them into ``model``
        between rollouts: each row keeps its context's raw frames, which
        ``FeatureEngine.rebuild`` re-encodes and re-prefills after a
        publication, a ``DonorHistory`` archive keeps raw frames, and the
        rollout stores each step's frame, previous action and context."""

        layers: int = 20
        """Global blocks up to the tap; the final norm reads the last one's output."""

        hook_interval: int = 128
        """Steps between ``ensure_room`` calls: the rollout splits its horizon into
        blocks of at most this many."""

        attention: Makeable[CacheAttention] = field(
            default_factory=MaskedCacheAttention.Config,
        )
        """The step's attention over the cache: masked SDPA anywhere, or
        ``Flash4CacheAttention`` on CUDA in bf16."""

        reprefill_rows: int = 128
        """Rows per re-prefill forward; a call's rows fill chunks of this many,
        the last padded, so the forward sees one shape."""

        dtype: torch.dtype = torch.bfloat16
        """Weight, activation and cache dtype; RoPE's tables stay float32."""

        compile: (
            Makeable[Callable[[Callable[..., object]], Callable[..., object]]] | None
        ) = None
        """Wraps the step's per-block and frame-encoder functions once per
        process, e.g. ``PartialConfig(torch.compile, fullgraph=True,
        dynamic=False, mode="max-autotune-no-cudagraphs")``; None runs them
        eagerly. ``FeatureKernels`` counts the shapes each compiles."""

    def __init__(self, config: Config) -> None:
        """Load the weights and check the history.

        Args:
          config: The weights, history, tap and kernels.

        Raises:
          ValueError: The history refuses its config or cannot make room for a
            block of ``hook_interval`` steps, ``layers`` exceeds the model's,
            or ``reprefill_rows`` is not positive.

        """
        history = config.history.make()
        bound = history.hook_interval_bound
        if config.hook_interval < 1 or config.hook_interval > bound:
            raise ValueError(
                f"hook_interval={config.hook_interval} must be in 1..{bound} for "
                f"{type(history).__name__}.",
            )
        if config.reprefill_rows < 1:
            raise ValueError(
                f"reprefill_rows={config.reprefill_rows} must be positive.",
            )
        model = config.weights.make()()
        blocks = len(model.transformer.blocks)
        if config.layers < 1 or config.layers > blocks:
            raise ValueError(f"layers={config.layers} must be in 1..{blocks}.")
        model.requires_grad_(requires_grad=False)
        self.model = model
        self._config = config
        self.history = history
        self.practice = config.practice.make()
        self.attention = config.attention.make()
        self.kernels = FeatureKernels.build(
            None if config.compile is None else config.compile.make(),
        )
        self.width: int = model.start.shape[-1]
        self.hook_interval = config.hook_interval
        self.joint = config.joint
        self.context_decisions = history.context_decisions
        self.ring = history.ring(frames=config.joint)

    def frozen(self) -> WorldModelFeature:
        """Return a source of the same live weights that keeps nothing for a learner.

        Its engines keep no frames and their rows' rings no more than their
        history needs, and a rollout stores no learner's inputs: an
        evaluation's source, which a joint learner's publications still reach.

        Returns:
          source: A view sharing ``model``, the kernels and the slots.

        """
        view = copy.copy(self)
        view.joint = False
        view.ring = self.history.ring(frames=False)
        return view

    def history_archive(
        self,
        entries: int,
        *,
        device: torch.device,
    ) -> dict[str, Tensor] | None:
        """Return the practice slot's storage for ``entries`` saved histories, or None.

        Args:
          entries: Practice entries, then one dump row per saving row.
          device: The engines' device.

        Returns:
          archive: What ``FeatureEngine.save_history`` writes and
            ``restore_history`` reads; None where a restore needs no history.

        """
        return self.practice.archive(
            entries=entries,
            ring=self.ring,
            width=self.width,
            frames=self.joint,
            dtype=self._config.dtype,
            device=device,
        )

    def make_engine(self, *, rows: int, device: torch.device) -> FeatureEngine:
        """Return an engine of ``rows`` rows on ``device``; every row begins an episode.

        Args:
          rows: Environments of the buffer.
          device: Where the engine runs; the model moves there at the first call.

        Returns:
          engine: A fresh engine sharing this source's weights.

        """
        config, model = self._config, self.model
        if model.start.device != device or model.start.dtype != config.dtype:
            model.to(device=device, dtype=config.dtype)
            for module in model.modules():
                if isinstance(module, RoPE):
                    # Training reads float32 tables under autocast, and the
                    # rotation rounds once from them.
                    module.to(torch.float32)
        return self.history.engine(
            model,
            rows=rows,
            frames=self.joint,
            layers=config.layers,
            attention=self.attention,
            kernels=self.kernels,
            reprefill_rows=config.reprefill_rows,
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FeatureKernels:
    """The functions a step runs, wrapped once per process and shared by its engines.

    Each takes its modules as arguments, so one compiled graph serves every
    block and every engine, one per input shape. A compiled function has one
    budget of shapes (dynamo's ``recompile_limit``, 8), which the actor's
    engines share with an evaluation's and a joint learner's
    (``context.ContextReplay``), and a full-graph compile past it raises. So
    each caller holds its shapes fixed: a joint ``Sliding`` source (512 rows
    a buffer, an evaluation of 256) compiles ``pre_attention`` and
    ``post_attention`` at six -- the actor's step ``[512, 2]``, the
    evaluation's ``[256, 2]``, the prefill's ``[128, 1024]`` (restores and
    rebuilds), the fallback chunk's ``[16, 1023]``, and the learner's packed
    passes with and without grad -- and ``encode_frames`` at four at most:
    the actor's ``[512]`` (its restores and rebuilds too), the evaluation's
    ``[256]``, and the learner's ``[512]`` with and without grad.
    """

    pre: Callable[
        [TransformerBlock, VarlenAttention, Tensor, Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor],
    ]
    post: Callable[[TransformerBlock, VarlenAttention, Tensor, Tensor], Tensor]
    encode: Callable[[WorldModel, FrameEncoder, Tensor, Tensor], Tensor]

    @classmethod
    def build(
        cls,
        wrap: Callable[[Callable[..., object]], Callable[..., object]] | None,
    ) -> FeatureKernels:
        """Return the functions, each wrapped by ``wrap``, or as they are without it."""
        return cls(
            pre=_wrapped(wrap, pre_attention),
            post=_wrapped(wrap, post_attention),
            encode=_wrapped(wrap, encode_frames),
        )


def pre_attention(
    block: TransformerBlock,
    attn: VarlenAttention,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return a global block's normed, rotated queries and keys, and its values.

    Args:
      block: The pre-norm block.
      attn: Its attention.
      x: Residual stream ``[..., S, C]``.
      cos: RoPE cosines of the positions ``[..., S, 1, D / 2]``.
      sin: RoPE sines, same shape.

    Returns:
      q: Queries ``[..., S, heads, D]``.
      k: Keys ``[..., S, heads_kv, D]``.
      v: Values ``[..., S, heads_kv, D]``.

    """
    q, k, v = attn.proj_qkv(block.norm1(x)).split(
        [attn.num_heads, attn.num_heads_kv, attn.num_heads_kv],
        dim=-2,
    )
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)
    q, k = RoPE.rotate(q, k, cos, sin)
    return q, k, v


def post_attention(
    block: TransformerBlock,
    attn: VarlenAttention,
    x: Tensor,
    out: Tensor,
) -> Tensor:
    """Return ``x`` after a block's attention residual and its MLP residual."""
    x = x + attn.proj_out(out.flatten(-2))
    return x + block.ffn(block.norm2(x))


def encode_frames(
    model: WorldModel,
    encoder: FrameEncoder,
    cells: Tensor,
    aux: Tensor,
) -> Tensor:
    """Return ``obs_proj(pooled)`` of frames, as the training forward encodes them.

    The last encoder block runs its attention output and MLP for the pooling
    position alone, the only one the global stack reads.

    Args:
      model: The world model, whose table embeds the frames.
      encoder: Its frame encoder.
      cells: In-range cell values ``[B, 99, 8]``.
      aux: In-range aux token values ``[B, 51]``.

    Returns:
      obs: The input of each frame's ``obs`` position ``[B, width]``.

    """
    slots = model.frame_slots(cells, aux)
    x = torch.cat(
        [encoder.pool.expand(len(slots), 1, -1), slots + encoder.slot_embedding],
        dim=-2,
    )
    blocks = list(encoder.stack.blocks)
    for index, block in enumerate(blocks):
        assert isinstance(block, TransformerBlock)
        attn = block.attn
        assert isinstance(attn, Attention)
        q, k, v = attn.proj_qkv(block.norm1(x)).split(
            [attn.num_heads, attn.num_heads_kv, attn.num_heads_kv],
            dim=-2,
        )
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        if index == len(blocks) - 1:
            q, x = q[:, :1], x[:, :1]
        out = functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        ).transpose(1, 2)
        x = x + attn.proj_out(out.flatten(-2))
        x = x + block.ffn(block.norm2(x))
    return model.obs_proj(encoder.stack.project_to_logits(x[:, 0]))


def context_inputs(
    obs: Tensor,
    actions: Tensor,
    start: Tensor,
    *,
    length: Tensor,
    anchored: Tensor,
    tokens: int,
) -> Tensor:
    """Lay out each row's context as the global stack's inputs from position 0.

    A context of the last ``L`` decisions is a training window, as ``batch.py``
    packs one: ``start, obs, act, ..., act, obs`` when it begins its episode,
    ``2 L`` positions, else ``obs, act, ..., act, obs``, ``2 L - 1``, the
    action before its first ``obs`` dropped. The current ``obs`` is the last.
    Positions past it hold zeros, which no context position reads under causal
    attention.

    Args:
      obs: The ``obs`` inputs of each row's last ``D`` decisions, oldest first
        ``[R, D, C]``; the last is the current decision's.
      actions: The embedded action taken at each of them ``[R, D, C]``; the
        last is unread.
      start: The ``start`` input ``[C]``.
      length: Decisions in each row's context, ``1..D`` ``[R]``.
      anchored: Whether each row's context begins its episode ``[R]``.
      tokens: Positions laid out, at least every row's ``2 L - 1 + anchored``.

    Returns:
      x: ``[R, tokens, C]``; row ``r``'s current ``obs`` is at
        ``2 L_r - 2 + anchored_r``.

    """
    decisions = obs.shape[-2]
    position = torch.arange(tokens, device=obs.device)
    first = anchored.long()[:, None]
    pair = position - first
    decision = (
        (decisions - length[:, None] + pair.clamp(min=0) // 2)
        .clamp(
            0,
            decisions - 1,
        )[..., None]
        .expand(-1, -1, obs.shape[-1])
    )
    x = torch.where(
        (pair % 2 == 0)[..., None],
        obs.gather(-2, decision),
        actions.gather(-2, decision),
    )
    x = torch.where((pair < 0)[..., None], start, x)
    real = position < 2 * length[:, None] - 1 + first
    return torch.where(real[..., None], x, 0)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RecentDecisions:
    """Each row's last decisions as the world model reads them, oldest first.

    Attributes:
      cells: Frame cell tokens ``[B, n, 99, 8]`` uint8.
      aux: Frame aux tokens ``[B, n, 51]`` int16.
      previous_actions: The action that led to each frame ``[B, n]`` uint8;
        0 at a decision that begins its episode or window, which reads none.
      count: How many of the ``n`` are the row's, the last ones; the rest are
        zeros ``[B]``.

    """

    cells: Tensor
    aux: Tensor
    previous_actions: Tensor
    count: Tensor


class FeatureEngine:
    """The world model's ``obs_t`` state for ``B`` actor rows over a static KV cache.

    The ``Refill`` history's engine. A row's context is what its cache holds:
    ``length`` positions, so ``(length + 1) // 2`` decisions, anchored at the
    episode's ``start`` when ``length`` is even (``context``).

    With ``frames``, each row's rings also keep the codec's tokens of its
    frames, for weights that change between rollouts: ``rebuild`` re-encodes
    them and re-prefills every context, and ``last_decisions`` hands them, with
    the actions that led to them, to a learner that recomputes the features.

    Attributes:
      feature: Static output ``[B, width]``, the last step's tap.
      keys: KV cache ``[layers, B, t_max + 1, heads_kv, head]``; position
        ``t_max`` is scratch.
      values: Same shape as ``keys``.
      length: Filled positions per row, the next RoPE position ``[B]``.
      count: Decisions per row since its episode or window began ``[B]``.
      needs_start: Rows whose next step begins an episode ``[B]``.
      window: Rows whose next step begins a mid-episode window ``[B]``.
      obs_ring: ``obs`` inputs of each row's last ``ring`` decisions;
        decision ``d`` of the window at ``d % ring``.
      action_ring: The action taken at each, written by the next step.
      cell_ring: With ``frames``, each decision's cell tokens ``[B, ring, 99,
        8]``, else None.
      aux_ring: With ``frames``, its aux tokens ``[B, ring, 51]``, else None.
      overflow: Steps that found a row without room, sticky ``[1]``.
      invalid: Frames with an out-of-schema value, sticky ``[1]``.
      nonfinite: Steps with a non-finite feature, sticky ``[1]``.
      t_max: Global positions per row.
      keep: Decisions a re-prefill keeps.
      ring: Decisions each row's rings hold.
      hook_interval_bound: The most steps ``ensure_room`` can make room for.
      heads: Query heads of the global attention.

    """

    def __init__(
        self,
        model: WorldModel,
        *,
        rows: int,
        t_max: int,
        keep: int,
        ring: int,
        frames: bool,
        layers: int,
        attention: CacheAttention,
        kernels: FeatureKernels,
        reprefill_rows: int,
    ) -> None:
        """Allocate every row's state on the model's device.

        Args:
          model: The world model, on the engine's device and dtype.
          rows: Rows of the buffer.
          t_max: Global positions per row, at least ``2 keep``.
          keep: Decisions a re-prefill keeps.
          ring: Decisions each row's rings hold, at least ``keep``.
          frames: Keep each decision's frame tokens.
          layers: Global blocks up to the tap.
          attention: The step's attention over the cache.
          kernels: The step's per-block and encoder functions.
          reprefill_rows: Rows per re-prefill forward.

        """
        self._model = model
        self._encoder = _frame_encoder(model)
        self._blocks = [_global_block(b) for b in model.transformer.blocks[:layers]]
        transformer = model.transformer
        assert isinstance(transformer, GlobalTransformer)
        self._transformer = transformer
        rope = self._blocks[0][1].rope
        assert isinstance(rope, RoPE)
        self._rope = rope
        self._attention = attention
        self._kernels = kernels
        self.t_max, self.keep, self.ring = t_max, keep, ring
        self.hook_interval_bound = (t_max - (2 * keep - 1)) // 2
        start = model.start
        device, dtype = start.device, start.dtype
        attn = self._blocks[0][1]
        self.heads = attn.num_heads
        shape = (layers, rows, t_max + 1, attn.num_heads_kv, attn.channels_head)
        self.keys = torch.zeros(shape, dtype=dtype, device=device)
        self.values = torch.zeros(shape, dtype=dtype, device=device)
        self.length = torch.zeros(rows, dtype=torch.long, device=device)
        self.count = torch.zeros(rows, dtype=torch.long, device=device)
        self.needs_start = torch.ones(rows, dtype=torch.bool, device=device)
        self.window = torch.zeros(rows, dtype=torch.bool, device=device)
        width = start.shape[-1]
        self.obs_ring = torch.zeros(rows, ring, width, dtype=dtype, device=device)
        self.action_ring = torch.zeros(rows, ring, dtype=torch.uint8, device=device)
        self.cell_ring, self.aux_ring = (
            (
                torch.zeros(
                    rows,
                    ring,
                    OBS_ROWS * OBS_COLS,
                    OBS_TILE_CHANNELS,
                    dtype=torch.uint8,
                    device=device,
                ),
                torch.zeros(
                    rows,
                    ring,
                    INVENTORY_OBS_SIZE,
                    dtype=torch.int16,
                    device=device,
                ),
            )
            if frames
            else (None, None)
        )
        self.feature = torch.zeros(rows, width, dtype=dtype, device=device)
        self.overflow = torch.zeros(1, dtype=torch.long, device=device)
        self.invalid = torch.zeros(1, dtype=torch.long, device=device)
        self.nonfinite = torch.zeros(1, dtype=torch.long, device=device)
        self._actions = model.action_embedding.weight.shape[0]
        self._rows = torch.arange(rows, device=device)
        self._pair = torch.arange(2, device=device)
        self._span = torch.arange(2 * keep - 1, device=device)
        self._chunk = min(reprefill_rows, rows)
        self._observation = torch.zeros(rows, OBS_SIZE, device=device)
        self._prefill_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None

    @torch.no_grad()
    def __call__(
        self,
        observation: Tensor,
        terminals: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        """Take one env step for every row; return ``feature``.

        Reads no device value and draws no random number, so a graph can hold it.

        Args:
          observation: Float32 packed observations ``[B, >= 843]``, ``o_t``.
          terminals: Nonzero where ``o_t`` begins an episode ``[B]``.
          previous_action: The action ids that led to ``o_t``, any number
            dtype ``[B]``; unread where an episode begins.

        Returns:
          feature: ``[B, width]``, the final-normed hidden state at ``obs_t``.

        """
        model, ring = self._model, self.ring
        self._observation = observation
        started = self.needs_start | (terminals != 0)
        window = self.window & started.logical_not()
        fresh = started | window
        self.length.masked_fill_(fresh, 0)
        self.count.masked_fill_(fresh, 0)
        action = previous_action.long().clamp(0, self._actions - 1)
        embedded = model.action_embedding(action)
        cells, aux = self._tokens(observation)
        second = self._kernels.encode(model, self._encoder, cells, aux)
        # A window's first query repeats its obs at position 0: in every layer
        # the two positions hold equal values, so the obs at position 1 reads
        # its own value, and position 0 keeps the obs as a training window does.
        first = torch.where(
            started[:, None],
            model.start,
            torch.where(window[:, None], second, embedded),
        )
        self.overflow.add_(self._unroomed().any())
        at = (self.length[:, None] + self._pair).clamp(max=self.t_max)
        hidden = self._global_step(torch.stack([first, second], dim=1), at)
        feature = self._transformer.project_to_logits(hidden[:, 1:])
        self.feature.copy_(feature[:, 0])
        previous = (self.count - 1) % ring
        kept = self.action_ring[self._rows, previous]
        self.action_ring[self._rows, previous] = torch.where(
            fresh,
            kept,
            action.to(torch.uint8),
        )
        slot = self.count % ring
        self.obs_ring[self._rows, slot] = second
        if self.cell_ring is not None and self.aux_ring is not None:
            self.cell_ring[self._rows, slot] = cells
            self.aux_ring[self._rows, slot] = aux
        self.count.add_(1)
        # A window holds one position; its repeat is overwritten next step.
        self.length.copy_(
            torch.where(window, 1, (self.length + 2).clamp(max=self.t_max)),
        )
        self._slide()
        self.nonfinite.add_(self.feature.isfinite().logical_not().any())
        self.needs_start.fill_(value=False)
        self.window.fill_(value=False)
        return self.feature

    def capture_state(self) -> list[Tensor]:
        """Return the tensors a graph warmup changes and must be restored (not KV).

        Cache positions at or past a row's length are never read, and a
        re-prefill rewrites the rest, so warmup writes there are harmless.

        Returns:
          state: The row counters and flags, the rings, the sticky counters and
            the feature.

        """
        return [
            self.length,
            self.count,
            self.needs_start,
            self.window,
            self.obs_ring,
            self.action_ring,
            *self._frame_rings().values(),
            self.overflow,
            self.invalid,
            self.nonfinite,
            self.feature,
        ]

    def reset(self) -> None:
        """Begin an episode in every row at its next step, as a fresh engine does."""
        self.needs_start.fill_(value=True)
        self.window.fill_(value=False)

    # A fresh engine's ``needs_start`` stands for an environment just reset; a window
    # begun on it is a resume or a restore into an episode already under way.
    def begin_window(self, rows: Tensor) -> None:
        """Begin a mid-episode window at ``rows``' next step, dropping their history.

        The step then reads the row's ``obs`` at position 0 with no ``start``, as
        ``batch.py`` lays out a segment that does not begin its episode. A row
        whose next observation begins an episode (its terminal) begins the
        episode instead.

        Args:
          rows: Bool ``[B]``, the rows to restart, on any device.

        """
        rows = rows.to(self.window.device)
        self.window.logical_or_(rows)
        self.needs_start.logical_and_(rows.logical_not())

    @property
    def plan(self) -> int:
        """The shape of the next block's steps, the key of the graph that replays them.

        One for every block here; ``SlidingEngine`` sets it per block.
        """
        return 0

    def context(self) -> tuple[Tensor, Tensor]:
        """Return each row's context after its last step.

        Returns:
          decisions: The decisions its last feature read, the current one
            included ``[B]``.
          anchored: Whether they begin with the episode's ``start`` ``[B]``.

        """
        return (self.length + 1) // 2, (self.length % 2 == 0) & (self.length > 0)

    def save_history(
        self,
        archive: Mapping[str, Tensor],
        entries: Tensor,
        rows: slice,
        *,
        previous_action: Tensor,
        fresh: Tensor,
    ) -> None:
        """Write ``rows``' histories into ``entries`` of a ``DonorHistory`` archive.

        Graph-safe. Call it between a step and the next: what is saved is the
        history that next step extends, and the action that led to its frame.

        Args:
          archive: The source's ``history_archive``.
          entries: The archive row of each saving row ``[S]``, no two alike.
          rows: The saving rows, ``S`` of them.
          previous_action: The action each took last ``[S]``.
          fresh: Nonzero where the next frame begins an episode ``[S]``; such
            a row, and one whose next step begins a window, saves an empty
            history.

        """
        empty = (self.needs_start | self.window)[rows] | (fresh != 0)
        for name, ring in self._rings().items():
            if name in archive:
                archive[name].index_copy_(0, entries, ring[rows])
        archive["count"].index_copy_(0, entries, self.count[rows])
        archive["length"].index_copy_(
            0,
            entries,
            self.length[rows].masked_fill(empty, 0),
        )
        archive["previous_action"].index_copy_(
            0,
            entries,
            previous_action.to(archive["previous_action"].dtype),
        )

    @torch.no_grad()
    def restore_history(
        self,
        archive: Mapping[str, Tensor] | None,
        entries: Tensor,
    ) -> Tensor | None:
        """Give restored rows a history, between steps: their entries', else a window.

        A saved history resumes as its donor's: the rings and counters are
        copied (saved frames encoded afresh) and the context re-prefilled, or
        left to the step where it slides. A context longer than ``keep``
        decisions (``Refill``'s) restarts from its last ``keep``, as a
        re-prefill at the save would. An empty history begins a window.

        Args:
          archive: The source's ``history_archive``; None begins a window in
            every restored row.
          entries: The archive row each row was restored from, -1 where none
            ``[B]``, on any device.

        Returns:
          previous_action: The action that led to each restored row's next
            frame ``[B]``, for its next step to read; None without an archive
            or a restored row.

        """
        entries = entries.to(self.length.device)
        restored = entries >= 0
        if archive is None:
            self.begin_window(restored)
            return None
        rows = restored.nonzero()[:, 0]
        if not len(rows):
            return None
        take = entries[rows]
        for name, ring in self._rings().items():
            if name in archive:
                ring[rows] = archive[name][take]
        if "obs" not in archive:
            self._encode_rings(rows)
        self.count[rows] = archive["count"][take]
        saved = archive["length"][take]
        decisions = (saved + 1) // 2
        anchored = (saved % 2 == 0) & (decisions <= self.keep)
        length = torch.where(
            saved > 0,
            2 * decisions.clamp(max=self.keep) - 1 + anchored.long(),
            0,
        )
        self.length[rows] = length
        self.needs_start[rows] = False
        self.window[rows] = length == 0
        cached = (length > 0) & self._cached(rows)
        self._prefill(rows[cached], length[cached])
        return archive["previous_action"][entries.clamp(min=0)]

    def last_decisions(self, n: int) -> RecentDecisions:
        """Return each row's last ``n`` decisions of its context, with frames.

        Graph-safe. A row whose next step begins an episode or a window holds
        none; another, the last ``min(n, decisions)`` of its context.

        Args:
          n: Decisions, at most ``ring``.

        Returns:
          recent: Their frames and previous actions, oldest first.

        Raises:
          ValueError: The engine keeps no frames, or ``n`` exceeds its ring.

        """
        cell_ring, aux_ring = self.cell_ring, self.aux_ring
        if cell_ring is None or aux_ring is None:
            raise ValueError("The engine keeps no frames: its source is not joint.")
        if n > self.ring:
            raise ValueError(f"n={n} exceeds the ring's {self.ring} decisions.")
        decisions, _ = self.context()
        count = decisions.clamp(max=n).masked_fill(self.needs_start | self.window, 0)
        span = torch.arange(n, device=self.count.device)
        index = self.count[:, None] - n + span
        real = span >= n - count[:, None]
        rows, ring = self._rows[:, None], self.ring
        return RecentDecisions(
            cells=cell_ring[rows, index % ring].masked_fill(
                real.logical_not()[..., None, None],
                0,
            ),
            aux=aux_ring[rows, index % ring].masked_fill(
                real.logical_not()[..., None],
                0,
            ),
            previous_actions=self.action_ring[rows, (index - 1) % ring].masked_fill(
                (real & (index > 0)).logical_not(),
                0,
            ),
            count=count,
        )

    @torch.no_grad()
    def rebuild(self) -> None:
        """Re-encode every row's frames and re-prefill its context, after new weights.

        Call it between steps on the stream that replays them, once the
        source's ``model`` holds the new weights; a row whose next step begins
        an episode or a window, or slides, needs no cache.

        Raises:
          ValueError: The engine keeps no frames.

        """
        if self.cell_ring is None:
            raise ValueError("The engine keeps no frames: its source is not joint.")
        self._encode_rings(self._rows)
        held = (self.length > 0) & (self.needs_start | self.window).logical_not()
        rows = held.nonzero()[:, 0]
        rows = rows[self._cached(rows)]
        self._prefill(rows, self.length[rows])

    @torch.no_grad()
    def ensure_room(self, steps: int) -> dict[str, float]:
        """Check the counters, then make room for ``steps`` more steps.

        Here that re-prefills the rows that cannot take them. Call it between
        steps on the stream that replays them. It reads the device once; what
        it enqueues is not waited for.

        Args:
          steps: Steps until the next call; at most ``hook_interval_bound``.

        Returns:
          telemetry: ``feature/*`` values: the context statistics are over the
            rows that continue, 0 with none; ``reprefill_ms`` is the previous
            call's re-prefill time (CUDA only), read now that it has finished.

        Raises:
          RuntimeError: A step found a row without room (a missed call), or
            produced a non-finite feature.
          ValueError: A frame held an out-of-schema value, or ``steps`` is
            outside ``1..hook_interval_bound``.

        """
        if steps < 1 or steps > self.hook_interval_bound:
            raise ValueError(f"steps={steps} must be in 1..{self.hook_interval_bound}.")
        need, counts = self._plan_room(steps)
        decisions, _ = self.context()
        # A row whose next step restarts holds no context: its count is a past one.
        context = decisions.float().masked_fill(
            self.needs_start | self.window,
            torch.nan,
        )
        feature = self.feature.float()
        quantiles = torch.nanquantile(
            context,
            torch.tensor([0.05, 0.95], device=context.device),
        )
        summary = torch.cat(
            [
                torch.cat([self.overflow, self.invalid, self.nonfinite]).float(),
                context.nanmean()[None].nan_to_num(0),
                quantiles.nan_to_num(0),
                feature.square().mean().sqrt()[None],
                feature.abs().amax()[None],
                counts.float(),
            ],
        )
        values = from_plain(summary.tolist(), list[float])
        overflow, invalid, nonfinite = values[:3]
        if overflow:
            raise RuntimeError(self._overflow_message(overflow))
        if invalid:
            self._raise_invalid(invalid)
        if nonfinite:
            raise RuntimeError(f"{nonfinite:.0f} steps produced a non-finite feature.")
        telemetry = {
            "feature/context_decisions_mean": values[3],
            "feature/context_decisions_p5": values[4],
            "feature/context_decisions_p95": values[5],
            "feature/rms": values[6],
            "feature/max_abs": values[7],
        }
        return telemetry | self._make_room(need, values[8:])

    def release(self) -> None:
        """Free the KV cache now, rather than when the last reference goes."""
        self.keys = self.values = torch.empty(0)

    def _unroomed(self) -> Tensor:
        """Return the rows the step's cache cannot hold two more positions of."""
        return self.length + 2 > self.t_max

    def _overflow_message(self, overflow: float) -> str:
        """Say what ``overflow`` steps found no room in."""
        return (
            f"Feature engine overflow: {overflow:.0f} steps found a row without "
            f"room in t_max={self.t_max}; ensure_room must run every "
            f"<= {self.hook_interval_bound} steps."
        )

    def _slide(self) -> None:
        """Finish the step for rows whose context slides; none slide here."""

    def _plan_room(self, steps: int) -> tuple[Tensor, Tensor]:
        """Return the rows that cannot take ``steps`` more steps, and their count."""
        need = (self.length + 2 * steps > self.t_max) & (
            self.needs_start | self.window
        ).logical_not()
        return need, need.sum()[None]

    def _make_room(self, need: Tensor, counts: list[float]) -> dict[str, float]:
        """Re-prefill the ``need`` rows; return the re-prefill telemetry."""
        telemetry = {"feature/reprefill_rows": counts[0]}
        if self._prefill_events is not None:
            begin, end = self._prefill_events
            telemetry["feature/reprefill_ms"] = begin.elapsed_time(end)
            self._prefill_events = None
        if counts[0]:
            self._reprefill(need.nonzero()[:, 0])
        return telemetry

    def _cached(self, rows: Tensor) -> Tensor:
        """Return whether each of ``rows``' next step reads its cache: all do here."""
        return torch.ones_like(rows, dtype=torch.bool)

    def _tokens(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        """Return each row's frame tokens: cells and aux; count invalid frames."""
        cells, aux, bad = tokens_and_flags(observation[..., :OBS_SIZE])
        self.invalid.add_(bad.any(-1).sum())
        return cells, aux

    def _frame_rings(self) -> dict[str, Tensor]:
        """Return the frame rings by archive name; empty without frames."""
        if self.cell_ring is None or self.aux_ring is None:
            return {}
        return {"cells": self.cell_ring, "aux": self.aux_ring}

    def _rings(self) -> dict[str, Tensor]:
        """Return every per-row ring by archive name."""
        return {
            "obs": self.obs_ring,
            "actions": self.action_ring,
            **self._frame_rings(),
        }

    # Batches of ``B`` frames, the step's encoder shape, so a compiled encoder sees no
    # other; the last is padded with its first frame, whose encodings are dropped.
    def _encode_rings(self, rows: Tensor) -> None:
        """Encode ``rows``' frame rings afresh into their ``obs`` rings."""
        cell_ring, aux_ring = self.cell_ring, self.aux_ring
        if cell_ring is None:
            raise ValueError("Expected cell_ring is not None.")
        if aux_ring is None:
            raise ValueError("Expected aux_ring is not None.")
        batch, ring = len(self._rows), self.ring
        frames = (
            rows[:, None] * ring + torch.arange(ring, device=rows.device)
        ).flatten()
        cells, aux = cell_ring.flatten(0, 1), aux_ring.flatten(0, 1)
        obs = self.obs_ring.view(-1, self.obs_ring.shape[-1])
        for first in range(0, len(frames), batch):
            real = frames[first : first + batch]
            index = torch.cat([real, real[:1].expand(batch - len(real))])
            encoded = self._kernels.encode(
                self._model,
                self._encoder,
                cells[index],
                aux[index],
            )
            obs.index_copy_(0, real, encoded[: len(real)])

    # The step keeps only a count; the latest frame, if it is one of them, names the
    # offending field.
    def _raise_invalid(self, invalid: float) -> None:
        """Raise the invalid-frame error, naming the latest frame's field if it is one."""
        msg = f"{invalid:.0f} frames held values outside the observation schema."
        try:
            encode(self._observation[..., :OBS_SIZE])
        except ValueError as error:
            raise ValueError(f"{msg} Latest frame: {error}") from error
        raise ValueError(msg)

    def _global_step(self, x: Tensor, at: Tensor) -> Tensor:
        """Write this step's two positions to the cache; return their hidden states."""
        cos, sin = self._rope(at.unsqueeze(-1))
        flat = (self._rows[:, None] * (self.t_max + 1) + at).flatten()
        seqused = (at[:, -1] + 1).to(torch.int32)
        attend = partial(self._attend_step, flat=flat, seqused=seqused)
        return self._stack(x, cos, sin, attend, finish=True)

    def _attend_step(
        self,
        layer: int,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        flat: Tensor,
        seqused: Tensor,
    ) -> Tensor:
        """Write one layer's new keys and values at ``flat``; attend over the cache."""
        self._write(layer, k, v, flat=flat)
        return self._attention(q, self.keys[layer], self.values[layer], seqused)

    def _write(self, layer: int, k: Tensor, v: Tensor, *, flat: Tensor) -> None:
        """Write keys and values ``[..., heads_kv, D]`` at a layer's flat positions."""
        self.keys[layer].flatten(0, 1).index_copy_(0, flat, k.flatten(0, -3))
        self.values[layer].flatten(0, 1).index_copy_(0, flat, v.flatten(0, -3))

    # Without ``finish`` the last block stops after writing its keys and values, and
    # the returned stream is the one entering that block.
    def _stack(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        attend: Callable[[int, Tensor, Tensor, Tensor], Tensor],
        *,
        finish: bool,
    ) -> Tensor:
        """Run the global blocks; ``attend(layer, q, k, v)`` writes and attends."""
        kernels = self._kernels
        for layer, (block, attn) in enumerate(self._blocks):
            q, k, v = kernels.pre(block, attn, x, cos, sin)
            out = attend(layer, q, k, v)
            if not finish and layer + 1 == len(self._blocks):
                return x
            x = kernels.post(block, attn, x, out)
        return x

    # Chunks have one shape: a chunk's padding rows repeat its first row and write
    # only into that row's scratch position, which no step reads.
    def _reprefill(self, rows: Tensor) -> None:
        """Rebuild ``rows``' caches from their last ``keep`` decisions, from position 0."""
        events = None
        if self.keys.is_cuda:
            events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            events[0].record()
        for first in range(0, len(rows), self._chunk):
            real = rows[first : first + self._chunk]
            chunk = torch.cat([real, real[:1].expand(self._chunk - len(real))])
            base = chunk[:, None] * (self.t_max + 1)
            padding = self._rows[: len(chunk), None] >= len(real)
            flat = torch.where(padding, base + self.t_max, base + self._span)
            self._reprefill_chunk(chunk, flat.flatten())
        if events is not None:
            events[1].record()
        self._prefill_events = events
        self.length[rows] = len(self._span)

    def _reprefill_chunk(self, rows: Tensor, flat: Tensor) -> None:
        """Re-prefill ``rows`` ``[R]`` from their rings, writing positions ``flat``."""
        model, keep = self._model, self.keep
        steps = self.count[rows, None] - keep + self._span[:keep]
        slots = steps % self.ring
        x = self.obs_ring.new_empty(len(rows), len(self._span), self.obs_ring.shape[-1])
        x[:, 0::2] = self.obs_ring[rows[:, None], slots]
        actions = self.action_ring[rows[:, None], slots[:, :-1]].long()
        x[:, 1::2] = model.action_embedding(actions)
        cos, sin = self._rope(self._span.expand(len(rows), -1)[..., None])
        self._stack(x, cos, sin, partial(self._attend_prefill, flat=flat), finish=False)

    # Chunks have one shape, ``2 ring`` positions, every context's room, for a restore
    # and a rebuild alike; positions past a row's context hold zeros the step overwrites
    # before it reads them. Padding rows write as ``_reprefill``'s do.
    def _prefill(self, rows: Tensor, length: Tensor) -> None:
        """Rebuild ``rows``' caches to hold ``length`` positions from their rings."""
        model = self._model
        span = torch.arange(2 * self.ring, device=rows.device)
        for first in range(0, len(rows), self._chunk):
            real = rows[first : first + self._chunk]
            pad = self._chunk - len(real)
            chunk = torch.cat([real, real[:1].expand(pad)])
            size = torch.cat(
                [length[first : first + len(real)], length[first:][:1].expand(pad)],
            )
            base = chunk[:, None] * (self.t_max + 1)
            padding = self._rows[: len(chunk), None] >= len(real)
            flat = torch.where(padding, base + self.t_max, base + span)
            obs, actions = self._unrolled(chunk, decisions=self.ring)
            x = context_inputs(
                obs,
                actions=model.action_embedding(actions.long()),
                start=model.start,
                length=(size + 1) // 2,
                anchored=size % 2 == 0,
                tokens=len(span),
            )
            cos, sin = self._rope(span.expand(len(chunk), -1)[..., None])
            attend = partial(self._attend_prefill, flat=flat.flatten())
            self._stack(x, cos, sin, attend, finish=False)

    def _unrolled(self, rows: Tensor, *, decisions: int) -> tuple[Tensor, Tensor]:
        """Return ``rows``' last ``decisions`` inputs and actions, oldest first."""
        span = torch.arange(decisions, device=rows.device)
        slots = (self.count[rows, None] - decisions + span) % self.ring
        return self.obs_ring[rows[:, None], slots], self.action_ring[
            rows[:, None],
            slots,
        ]

    def _attend_prefill(
        self,
        layer: int,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        flat: Tensor,
    ) -> Tensor:
        """Write one layer's keys and values at ``flat``; attend causally within rows."""
        self._write(layer, k, v, flat=flat)
        return self._attend_causal(layer, q, k, v)

    # A row attends causally to its own keys and values, unrounded by any cache, as the
    # training forward does.
    def _attend_causal(self, layer: int, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Attend causally within each row ``[R, S, heads, D]``."""
        del layer
        return functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            enable_gqa=True,
        ).transpose(1, 2)


class SlidingEngine(FeatureEngine):
    """The ``Sliding`` history's engine: a step reads at most ``decisions`` decisions.

    The cache holds a row's context while it extends the episode's ``start``
    or the window's first ``obs``: up to ``decisions`` decisions, ``2
    decisions`` positions. A step whose context would hold more slides it:
    the context becomes the last ``decisions`` decisions, ``obs, act, ...,
    obs_t`` from position 0, recomputed from the row's ring, and stays so
    until the row's next episode or window. Its ``length`` stays ``2
    decisions - 1``, so ``context`` reads that window.

    The recompute is a fallback batch of ``plan`` rows: ``ensure_room`` lists
    the rows whose context can slide within the next block and sizes the
    batch to whole chunks of ``fallback_rows``, the fewest that hold them, so
    each forward has one shape, padding wastes less than a chunk, and a
    rollout captures a graph per batch size. A planned row that does not
    slide yet computes a window the step discards, and so does a padding
    slot, whose result lands in a scratch row.

    Attributes:
      decisions: Decisions per window.

    """

    def __init__(
        self,
        model: WorldModel,
        *,
        rows: int,
        decisions: int,
        fallback_rows: int,
        frames: bool,
        layers: int,
        attention: CacheAttention,
        kernels: FeatureKernels,
        reprefill_rows: int,
    ) -> None:
        """Allocate every row's state and the fallback's plan on the model's device.

        Args:
          model: The world model, on the engine's device and dtype.
          rows: Rows of the buffer.
          decisions: Decisions per window.
          fallback_rows: Rows per forward of the recompute.
          frames: Keep each decision's frame tokens.
          layers: Global blocks up to the tap.
          attention: The step's attention over the cache.
          kernels: The step's per-block and encoder functions.
          reprefill_rows: Rows per forward of a restore's re-prefill.

        """
        super().__init__(
            model,
            rows=rows,
            t_max=2 * decisions,
            keep=decisions,
            ring=decisions,
            frames=frames,
            layers=layers,
            attention=attention,
            kernels=kernels,
            reprefill_rows=reprefill_rows,
        )
        self.decisions = decisions
        self.hook_interval_bound = decisions
        self._fallback_rows = fallback_rows
        device = self.length.device
        self._most = -(-rows // fallback_rows) * fallback_rows
        # The rows that can slide, then ``rows`` in each padding slot, then a dump slot
        # for the rows that cannot.
        self._plan = torch.full(
            (self._most + 1,),
            rows,
            dtype=torch.long,
            device=device,
        )
        self._planned = torch.zeros(rows, dtype=torch.bool, device=device)
        self._slid = self.feature.new_zeros(rows + 1, self.feature.shape[-1])
        self._capacity = 0

    @property
    @override
    def plan(self) -> int:
        """The fallback batch's capacity for the current block."""
        return self._capacity

    @override
    def _unroomed(self) -> Tensor:
        """Return the rows whose context slides without a slot in the fallback batch."""
        return (self.length + 2 > self.t_max) & self._planned.logical_not()

    @override
    def _overflow_message(self, overflow: float) -> str:
        """Say that ``overflow`` steps slid a row the block's fallback plan left out."""
        return (
            f"Feature engine overflow: {overflow:.0f} steps slid a row the "
            f"fallback plan of {self._capacity} rows left out; ensure_room must "
            f"plan every block of at most {self.hook_interval_bound} steps."
        )

    # A row slides when its window would hold one decision more than ``decisions``:
    # exactly the rows the cache has no room for.
    @override
    def _slide(self) -> None:
        """Recompute the sliding rows' windows and replace their features."""
        sliding = self.count > self.decisions
        if self._capacity:
            plan = self._plan[: self._capacity]
            rows = plan.clamp(max=len(self._rows) - 1)
            features = [
                self._window_feature(rows[first : first + self._fallback_rows])
                for first in range(0, self._capacity, self._fallback_rows)
            ]
            self._slid.index_copy_(0, plan, torch.cat(features))
            self.feature.copy_(
                torch.where(sliding[:, None], self._slid[:-1], self.feature),
            )
        self.length.masked_fill_(sliding, self.t_max - 1)

    def _window_feature(self, rows: Tensor) -> Tensor:
        """Return the final-normed ``obs`` state of ``rows``' last ``decisions``."""
        model = self._model
        obs, actions = self._unrolled(rows, decisions=self.decisions)
        x = context_inputs(
            obs,
            actions=model.action_embedding(actions.long()),
            start=model.start,
            length=self.count.new_full((len(rows),), self.decisions),
            anchored=self.window.new_zeros(len(rows)),
            tokens=len(self._span),
        )
        cos, sin = self._rope(self._span.expand(len(rows), -1)[..., None])
        hidden = self._stack(x, cos, sin, self._attend_causal, finish=True)
        return self._transformer.project_to_logits(hidden[:, -1])

    @override
    def _plan_room(self, steps: int) -> tuple[Tensor, Tensor]:
        """List the rows that can slide within ``steps`` steps first in the plan."""
        pending = self.needs_start | self.window
        need = self.count.masked_fill(pending, 0) + steps > self.decisions
        rows = len(self._rows)
        self._plan.fill_(rows)
        self._plan.scatter_(
            0,
            torch.where(need, need.cumsum(0) - 1, self._most),
            self._rows,
        )
        self._planned.copy_(need)
        sliding = (self.count > self.decisions) & pending.logical_not()
        return need, torch.stack([need.sum(), sliding.sum()])

    @override
    def _make_room(self, need: Tensor, counts: list[float]) -> dict[str, float]:
        """Size the fallback batch for the planned rows; return its telemetry."""
        del need
        rows, chunk = len(self._rows), self._fallback_rows
        # Whole chunks: one shape for the compiled per-block functions, whose budget
        # the step, the prefill and a joint learner share (``FeatureKernels``).
        self._capacity = -(-round(counts[0]) // chunk) * chunk
        return {
            "feature/sliding_fraction": counts[1] / rows,
            "feature/fallback_capacity": float(self._capacity),
        }

    @override
    def _cached(self, rows: Tensor) -> Tensor:
        """Return whether each of ``rows``' next step reads its cache, not its ring."""
        return self.count[rows] < self.decisions


# ---- The world model's modules, as the engine reads them. ----


def _frame_encoder(model: WorldModel) -> FrameEncoder:
    """Narrow the model's encoder to the slot encoder the engine runs block by block."""
    encoder = model.encoder
    if not isinstance(encoder, FrameEncoder) or encoder.stack.proj_in is not None:
        raise TypeError(
            f"The feature engine encodes with a FrameEncoder, not {type(encoder)}.",
        )
    for block in encoder.stack.blocks:
        _block_attention(block, attention=Attention, kernels=(SdpaFused,))
    return encoder


def _global_block(block: nn.Module) -> tuple[TransformerBlock, VarlenAttention]:
    """Narrow a global block to the pre-norm rotary block the engine decodes."""
    # FA4 is SdpaVarlen's function, rounded otherwise: ``scoring.load_trained`` keeps
    # it on CUDA, and the feature parity test holds the engine to it there.
    attn = _block_attention(
        block,
        attention=VarlenAttention,
        kernels=(SdpaVarlen, Flash4Varlen),
    )
    assert isinstance(block, TransformerBlock)
    assert isinstance(attn, VarlenAttention)
    if not isinstance(attn.rope, RoPE):
        raise TypeError("The engine's global blocks take a rotary embedding.")
    return block, attn


# The engine attends with its own SDPA calls, the function of ``kernels``, so a block
# whose kernel slot holds another kernel is refused rather than run differently.
def _block_attention(
    block: nn.Module,
    *,
    attention: type[Attention | VarlenAttention],
    kernels: tuple[type[nn.Module], ...],
) -> AttentionProjections:
    """Return a pre-norm block's attention, refusing a layout the engine does not run."""
    if not isinstance(block, TransformerBlock):
        raise TypeError(f"The engine runs TransformerBlocks, not {block}.")
    attn = block.attn
    if not isinstance(attn, attention):
        raise TypeError(f"The engine runs {attention.__name__}, not {attn}.")
    if not block.prenorm or attn.split_qkv_projection or attn.norm_out is not None:
        raise ValueError(
            f"The engine runs a fused-QKV {attention.__name__} without an output "
            f"norm, not {attn}.",
        )
    if type(attn.attn_kernel) not in kernels:
        raise ValueError(
            f"The engine attends as {' or '.join(k.__name__ for k in kernels)} "
            f"computes it, not as {type(attn.attn_kernel).__name__}.",
        )
    return attn


@runtime_checkable
class _Flash4Decode(Protocol):
    def flash_attn_varlen_func(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        seqused_k: Tensor,
        causal: bool,
    ) -> Tensor | tuple[Tensor, Tensor | None]: ...


def _flash4() -> _Flash4Decode:
    """Import FA4's CuTe interface (Linux-only ``flash-attn-4``)."""
    module = importlib.import_module("flash_attn.cute.interface")
    if not isinstance(module, _Flash4Decode):
        raise TypeError("FA4 must provide flash_attn_varlen_func.")
    return module


# The wrapper (``torch.compile``) returns a function of the same signature, which its
# loose type cannot say.
def _wrapped[**P, R](
    wrap: Callable[[Callable[..., object]], Callable[..., object]] | None,
    function: Callable[P, R],
) -> Callable[P, R]:
    """Return ``function`` wrapped by ``wrap``, or itself."""
    return function if wrap is None else cast("Callable[P, R]", wrap(function))
