"""Test support shared by the world model's inference tests: tiny models and frames.

The engine, dream, session and script tests all decode a seeded world model a
few positions wide over a cut-down schema, feed it random but schema-valid
decisions, and force its jobs to known tokens. The codec and rules tests read
frames of the game itself, and the models' cost tests run their attention as
explicit products. They import those pieces from here rather than from
another test module.
"""

from __future__ import annotations

from copy import deepcopy
from functools import cache, partial
from typing import (
    TYPE_CHECKING,
    cast,
    override,
)

import dataclasses
import math

from configgle import Makes
from torch import Tensor, nn

import numpy as np
import torch

from priml.baselines.craftax.game.state import (
    ATN_DIM,
    OBS_SIZE,
    env_state,
    env_stats,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules, observe_numba, play_numba
from priml.baselines.craftax.lib.arrays import ints
from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.batch import (
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.context import Contexts
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    FrameEncoder,
    WorldModel,
)
from priml.baselines.craftax.world_model.replay import (
    CELL_VALUES,
    reset_world,
    token_frame_numba,
)
from priml.baselines.craftax.world_model.schema import (
    FrameSchema,
    craftax_schema,
    done_id,
)
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive, SdpaVarlen
from priml.model.attention.window import segment_mask
from priml.model.embedding import Embedding
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from priml.baselines.craftax.world_model.engine import (
        Control,
        Engine,
    )
    from priml.model.transformer.transformer import Transformer


def small_schema() -> FrameSchema:
    """Return the Craftax schema cut to 3 cells and 4 auxiliary slots.

    Returns:
      schema: The cut schema, with the reward and done prefix.

    """
    full = craftax_schema()
    return dataclasses.replace(
        full,
        cell_slots=3,
        scalar_names=full.scalar_names[:4],
        scalar_ranges=full.scalar_ranges[:4],
    )


def tiny_model(
    schema: FrameSchema,
    *,
    global_layers: int = 1,
    local_layers: int = 1,
    untied_output: bool = False,
) -> WorldModel:
    """Return a seeded world model of width 16 locally and 36 globally, in eval mode.

    Args:
      schema: The slot layout.
      global_layers: Blocks of the global stack.
      local_layers: Blocks of the encoder and of the decoder.
      untied_output: Score local slots against their own output table.

    Returns:
      model: A copy of the model built after ``torch.manual_seed(0)``, the RNG
        left where building it leaves it.

    """
    model, rng = _seeded_tiny_model(
        schema,
        global_layers=global_layers,
        local_layers=local_layers,
        untied_output=untied_output,
    )
    torch.set_rng_state(rng)
    return deepcopy(model)


def random_segment(
    schema: FrameSchema,
    decisions: int,
    *,
    seed: int,
    starts: bool = True,
    terminal: bool = False,
) -> Segment:
    """Return schema-valid random decisions and their frames.

    Args:
      schema: The slot layout the values must fit.
      decisions: Decisions in the segment.
      seed: Seed of every draw.
      starts: Whether decision 0 begins its episode.
      terminal: Whether the last decision ends the episode, which then has no
        frame after it.

    Returns:
      segment: A frame per decision, plus the next unless ``terminal``.

    """
    generator = torch.Generator().manual_seed(seed)
    frames = decisions + (0 if terminal else 1)
    cells = torch.stack(
        [
            torch.randint(
                0,
                field.valid,
                (frames, schema.cell_slots),
                generator=generator,
            )
            for field in schema.cell_fields
        ],
        dim=-1,
    )
    aux = torch.stack(
        [
            torch.randint(low - 155, high - 154, (frames,), generator=generator)
            for low, high in schema.scalar_ranges
        ],
        dim=-1,
    )
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = terminal
    return Segment(
        cells=cells.to(torch.uint8),
        aux=aux.to(torch.int16),
        actions=torch.randint(0, 43, (decisions,), generator=generator).to(torch.uint8),
        reward=torch.randint(-1, 4, (decisions,), generator=generator).to(torch.int16),
        done=done,
        starts_episode=starts,
    )


def row_zero_done(make: Callable[[], Control], *, done: bool) -> Control:
    """Return ``make()`` with row 0's act-job ``done`` forced to ``done``."""
    control = make()
    control.job_forced[0, 1] = True
    control.job_tokens[0, 1, 0] = done_id(done=done)
    return control


def state_of(engine: Engine, *, row: int) -> dict[str, Tensor]:
    """Return a copy of one row's state, with only the valid part of its cache.

    Args:
      engine: The engine.
      row: The row.

    Returns:
      state: Each state tensor's row; the caches cut at the row's length.

    """
    state = engine.state
    length = int(state.length[row])
    copy: dict[str, Tensor] = {
        field.name: cast("Tensor", getattr(state, field.name))[row].clone()
        for field in dataclasses.fields(state)
        if field.name not in {"keys", "values"}
    }
    copy["keys"] = state.keys[:, row, :length].clone()
    copy["values"] = state.values[:, row, :length].clone()
    return copy


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PlayedFrames:
    """What the game showed before each decision of one randomly played episode.

    Attributes:
      observations: The game's observations, float32 ``[N, 843]``.
      masks: The game's action masks, bool ``[N, 43]``.
      cells: The States' token-frame cells, uint8 ``[N, 99, 8]``.
      aux: The States' token-frame aux values, int16 ``[N, 51]``.

    """

    observations: Tensor
    masks: Tensor
    cells: Tensor
    aux: Tensor


def played_frames(*, world_seed: int, decisions: int) -> PlayedFrames:
    """Play uniformly random actions in one world, up to its end or ``decisions``.

    Args:
      world_seed: The world, as ``replay.reset_world`` builds it; also seeds
        the actions.
      decisions: Decisions to play at most.

    Returns:
      frames: The observation, mask and token frame before each decision.

    """
    states, rng = reset_world(world_seed)
    stats = new_stats(1)
    state, rules = env_state(states, 0), Rules()
    actions = np.random.default_rng(world_seed).integers(ATN_DIM, size=decisions)
    observations = np.zeros((decisions, OBS_SIZE), dtype=np.float32)
    masks = np.zeros((decisions, ATN_DIM), dtype=np.uint8)
    cells = np.zeros((decisions, CELL_VALUES), dtype=np.uint8)
    aux = np.zeros((decisions, OBS_SIZE - CELL_VALUES), dtype=np.int16)
    played = 0
    for action in ints(actions):
        observe_numba(state, observations[played, :], masks[played, :], rules)
        if token_frame_numba(state, cells[played, :], aux[played, :]) != 0:
            raise ValueError(
                f"World {world_seed} shows a frame outside the schema.",
            )
        played += 1
        if play_numba(state, rng, env_stats(stats, 0), action, rules)[1]:
            break
    return PlayedFrames(
        observations=torch.from_numpy(observations[:played]),
        masks=torch.from_numpy(masks[:played]).bool(),
        cells=torch.from_numpy(cells[:played]).view(played, 99, 8),
        aux=torch.from_numpy(aux[:played]),
    )


def random_contexts(
    lengths: Tensor,
    anchored: Tensor,
    *,
    schema: FrameSchema,
    slots: int,
    counts: Tensor,
    seed: int,
) -> Contexts:
    """Return random schema-valid decisions of a window, read under the given contexts.

    Each row plays one random segment (:func:`random_segment`) of ``slots + T``
    decisions: its first ``slots`` fill the prefix, the rest are the steps, and
    each decision's previous action is the one the segment took before it.

    Args:
      lengths: Decisions in each step's context ``[R, T]``.
      anchored: Whether each context begins its episode ``[R, T]``.
      schema: The slot layout the frames fit.
      slots: Prefix slots ``P``.
      counts: Each row's filled prefix slots ``[R]``.
      seed: Row ``r``'s segment's seed is ``seed + r``.

    Returns:
      contexts: The window, on the CPU.

    """
    rows, steps = lengths.shape
    segments = [
        random_segment(schema, slots + steps, seed=seed + row) for row in range(rows)
    ]
    cells = torch.stack([s.cells[: slots + steps] for s in segments])
    aux = torch.stack([s.aux[: slots + steps] for s in segments])
    taken = torch.stack([s.actions for s in segments]).long()
    previous = torch.cat([torch.zeros_like(taken[:, :1]), taken[:, :-1]], dim=1)
    return Contexts(
        cells=cells[:, slots:],
        aux=aux[:, slots:],
        previous_actions=previous[:, slots:].float(),
        lengths=lengths,
        anchored=anchored,
        prefix_cells=cells[:, :slots],
        prefix_aux=aux[:, :slots],
        prefix_previous_actions=previous[:, :slots].float(),
        prefix_counts=counts,
    )


def window_segment(
    cells: Tensor,
    aux: Tensor,
    actions: Tensor,
    *,
    first: int,
    last: int,
    anchored: bool,
) -> Segment:
    """Return decisions ``first..last`` of one history as a training segment.

    Args:
      cells: Cell values per decision ``[T, 99, 8]``.
      aux: Aux values ``[T, 51]``.
      actions: The action taken at each decision ``[T]``.
      first: The segment's first decision.
      last: Its last, whose ``obs`` is the context's current one.
      anchored: Whether ``first`` begins the episode, so a ``start`` leads.

    Returns:
      segment: The decisions, ready to pack.

    """
    decisions = last + 1 - first
    return Segment(
        cells=cells[first : last + 1],
        aux=aux[first : last + 1],
        actions=actions[first : last + 1].to(torch.uint8),
        reward=torch.zeros(decisions, dtype=torch.int16),
        done=torch.zeros(decisions, dtype=torch.bool),
        starts_episode=anchored,
    )


def final_hidden(
    model: WorldModel,
    *,
    layers: int,
    windows: Sequence[Segment],
) -> Tensor:
    """Return the training forward's final-normed hiddens, one window per segment.

    The model's own ``logits`` over the packed windows, its global stack cut to
    ``layers`` blocks for the call and the forward stopped there: the local
    decoder that follows reads nothing the tap does. Under grad mode the result
    carries the graph back to the model's own parameters.

    Args:
      model: The world model, in the precision and on the device to run.
      layers: Global blocks up to the tap; the final norm reads the last one's
        output, as the engine's does.
      windows: One segment per window.

    Returns:
      hidden: ``[W, t_g, C]`` in the model's dtype.

    """
    t_g = max(2 * len(w.actions) + 1 for w in windows)
    batch = pack_windows([[w] for w in windows], t_g=t_g, s_max=1)
    transformer = model.transformer
    blocks = transformer.blocks
    output = _Output()
    hook = transformer.register_forward_hook(output)
    transformer.blocks = nn.ModuleList(list(blocks)[:layers])
    try:
        model.logits(batch.to(model.start.device))
    except _TapReachedError:
        pass
    finally:
        transformer.blocks = blocks
        hook.remove()
    return output.value


def context_features(
    model: WorldModel,
    *,
    layers: int,
    windows: Sequence[Segment],
) -> Tensor:
    """Return the training forward's final-normed hidden at each window's last ``obs``.

    One window per context: the reference for a feature that reads exactly
    that context, whichever history produced it.

    Args:
      model: The world model, in the precision and on the device to run.
      layers: Global blocks up to the tap.
      windows: Each context as a segment (:func:`window_segment`).

    Returns:
      hidden: Float32 ``[W, C]``; under grad mode, with its graph.

    """
    hidden = final_hidden(model, layers=layers, windows=windows)
    at = torch.tensor(
        [int(w.starts_episode) + 2 * (len(w.actions) - 1) for w in windows],
        device=hidden.device,
    )
    return hidden[torch.arange(len(windows), device=hidden.device), at].float()


def context_reference(model: WorldModel, contexts: Contexts, *, layers: int) -> Tensor:
    """Return each step's feature from the training forward of its own context alone.

    Every step is its own window (:func:`context_features`): no context shares
    a pass. Under grad mode the result carries the graph back to the model's
    own parameters.

    Args:
      model: The world model.
      contexts: The window.
      layers: Global blocks up to the tap.

    Returns:
      features: Float32 ``[R, T, C]``.

    """
    rows, steps = contexts.lengths.shape
    slots = contexts.prefix_cells.shape[1]
    cells = torch.cat([contexts.prefix_cells, contexts.cells], dim=1)
    aux = torch.cat([contexts.prefix_aux, contexts.aux], dim=1)
    previous = torch.cat(
        [contexts.prefix_previous_actions, contexts.previous_actions],
        dim=1,
    ).long()
    # The action taken at each decision is the one that led to the next.
    taken = torch.cat([previous[:, 1:], torch.zeros_like(previous[:, :1])], dim=1)
    windows = [
        window_segment(
            cells[row],
            aux[row],
            taken[row],
            first=slots + step + 1 - int(contexts.lengths[row, step]),
            last=slots + step,
            anchored=bool(contexts.anchored[row, step]),
        )
        for row in range(rows)
        for step in range(steps)
    ]
    return context_features(model, layers=layers, windows=windows).view(rows, steps, -1)


class NaiveVarlen(SdpaVarlen):
    """``SdpaVarlen``'s attention as explicit products, priced as ``SdpaVarlen`` is.

    A cost test measures what torch dispatches, and the CPU's fused attention
    dispatches no product torch's FLOP counter sees; these products it does.
    The max logit, a logging diagnostic ``SdpaVarlen`` forms by a second
    product that the cost, counting model work, leaves out, is skipped.
    """

    class Config(Makes["NaiveVarlen"], SdpaVarlen.Config):
        """``SdpaVarlen``'s cost; no options."""

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        cu_seqlens: Tensor,
        window: int = -1,
        is_causal: bool = True,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
        record_max_logit: bool = False,
        **kwargs: object,
    ) -> Tensor:
        """Attend causally within each segment, as ``SdpaVarlen`` does by default.

        Args:
          q: Queries ``[..., S, H, D]``.
          k: Keys ``[..., S, H_kv, D]``; ``H_kv`` divides ``H``.
          v: Values, shaped like ``k``.
          cu_seqlens: Int32 segment boundaries of the flattened leading and
            ``S`` axes.
          window: Unread; the default reach.
          is_causal: Unread; always causal.
          attn_mask: Unread; the segments are the mask.
          dropout_p: Unread; no dropout.
          scale: Unread; ``D**-0.5``.
          record_max_logit: Unread; ``max_logit`` stays None.
          **kwargs: The rest of the bus, unread.

        Returns:
          out: ``[..., S, H, D]``.

        """
        del window, is_causal, attn_mask, dropout_p, scale, record_max_logit, kwargs
        lead, length = q.shape[:-3], q.shape[-3]
        mask = segment_mask(cu_seqlens, rows=math.prod(lead), length=length)
        groups = q.shape[-2] // k.shape[-2]
        keys, values = (t.repeat_interleave(groups, dim=-2) for t in (k, v))
        q, keys, values = (t.movedim(-3, -2) for t in (q, keys, values))
        logits = q @ keys.transpose(-1, -2) * q.shape[-1] ** -0.5
        masked = logits.masked_fill(~mask.view(*lead, 1, length, length), -math.inf)
        return (masked.softmax(dim=-1) @ values).movedim(-2, -3)


def naive_attention(block: object) -> None:
    """Give a block's attentions explicit products: ``SdpaNaive``, or ``NaiveVarlen``.

    Args:
      block: A ``TransformerBlock`` or ``DecoderBlock`` config, edited in place.

    """
    assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
    attentions: list[object] = [block.attn]
    if isinstance(block, DecoderBlock.Config):
        attentions.append(block.cross_attn)
    for attn in attentions:
        if isinstance(attn, VarlenAttention.Config):
            attn.attn_kernel = NaiveVarlen.Config()
        else:
            assert isinstance(attn, Attention.Config)
            attn.attn_kernel = SdpaNaive.Config()


# Building a tiny model takes 15 ms and copying one 3 ms, and the inference tests
# build the same few dozens of times; each gets its own copy to change.
@cache
def _seeded_tiny_model(
    schema: FrameSchema,
    *,
    global_layers: int,
    local_layers: int,
    untied_output: bool,
) -> tuple[WorldModel, Tensor]:
    """Return the tiny model built after ``torch.manual_seed(0)``, and the RNG then."""
    config = WorldModel.Config(schema=schema)
    if untied_output:
        config.output_table = Embedding.Config(
            init_weight=partial(nn.init.normal_, std=0.02),
        )
    config.encoder.channels_in = 16
    config.decoder.channels_in = 16
    assert isinstance(config.encoder, FrameEncoder.Config)
    for stack in (config.encoder.stack, config.decoder.stack):
        _shrink_local(stack, layers=local_layers)
    config.transformer.channels_in = 36
    config.transformer.num_layers = global_layers
    block = config.transformer.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.attn.channels_head = 4
    block.ffn.channels_hidden = 48
    torch.manual_seed(0)
    return config.make().eval(), torch.get_rng_state()


def _shrink_local(stack: Transformer.Config, *, layers: int) -> None:
    """Shrink an encoder or decoder stack to ``layers`` layers with a narrow MLP."""
    stack.num_layers = layers
    block = stack.block
    assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.ffn.channels_hidden = 32


class _TapReachedError(Exception):
    """Raised by ``_Output`` once it holds the hiddens, to end the forward there."""


class _Output:
    """A forward hook keeping its module's output, the training forward's hiddens, then stopping it."""

    def __init__(self) -> None:
        self.value = torch.empty(0)

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: Tensor,
    ) -> None:
        """Keep ``output``, then end the forward.

        Raises:
          _TapReachedError: Always.

        """
        del module, inputs
        self.value = output
        raise _TapReachedError
