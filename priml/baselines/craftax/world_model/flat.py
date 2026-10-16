"""The flat Craftax model: every slot one position of the global stack.

The hierarchical model spends two global positions per decision; the flat model
spends 153, one per slot, on the same global transformer, so it measures what
the hierarchy costs in per-slot NLL. Each decision is the design's sequence
``o_t, a_t, r_{t+1}, d_{t+1}``: 99 board cells (a cell's eight field IDs summed
at one position), 51 auxiliary fields, the action, the reward, and done. An
episode opens with ``start``, which is followed, like each action, by a reward
of 0 and done false, as the hierarchical ``start`` job is fed. Each position
predicts the next under the same restricted softmaxes as the local decoder.

It reads the ``PackedBatch`` the hierarchical model reads and lays each window
out again, one block per local job: the job's three head positions (its action
or ``start``, reward, done), then the frame it generates, when that exists. A
window that opens mid-episode first holds its first frame, which no job
generates and nothing scores. So the flat model scores exactly the targets the
hierarchical model scores, in the same record layout, and
``metric.craftax_target_nll`` and ``CraftaxBitsPerByte`` serve both unchanged.

A window of ``t_g`` global positions needs at most ``flat_positions(t_g)`` flat
positions, provided only its last segment is cut short, as in the stream's
training and stratified validation windows. A ``data.EvalSpans`` window packs
several segments, each cut at its span's end, and can need more. One that needs
more has NaN logits, so the loss and the metric show it rather than dropping
targets.
"""

from dataclasses import field
from functools import partial
from typing import Final, override

import dataclasses

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.craftax.lib.costs import (
    broadcast_add_cost,
    concat_cost,
    gather_cost,
    logits_cost,
)
from priml.baselines.craftax.world_model.batch import Kind, PackedBatch
from priml.baselines.craftax.world_model.loss import (
    ModalityLoss,
    Nll,
    modality_loss,
)
from priml.baselines.craftax.world_model.model import (
    GlobalLanguageModel,
    WorldModelLogits,
    embed_frames,
    gathered_board,
    prefix_ids,
    schema_tables,
    slot_loss_cost,
    slot_terms,
)
from priml.baselines.craftax.world_model.schema import (
    FrameSchema,
    action_id,
    craftax_schema,
    number_id,
)
from priml.baselines.craftax.world_model.train_step import ForwardCost
from priml.cost import Cost, cost, elementwise_cost, traffic
from priml.model.embedding import Embedding
from priml.model.init import InitFn, call_init


HEAD: Final = 3
"""Positions before a job's frame: its action (or ``start``), reward, and done."""


def flat_positions(t_g: int, *, frame_slots: int = 150) -> int:
    """Return the most flat positions a stratified window of ``t_g`` can need.

    A job is ``HEAD + frame_slots`` positions for two global ones, so a window
    needs about ``t_g / 2`` blocks. On top of that, an even window can be one
    segment opened mid-episode, whose first frame precedes its first job, and
    an odd one can end on a job whose frame the loader appended. The bound
    assumes only the window's last segment is cut short, so an ``EvalSpans``
    window, several segments each cut at its span's end, can exceed it.

    Args:
      t_g: Global positions per window.
      frame_slots: Slots per frame.

    Returns:
      positions: The flat length every such window fits in.

    """
    block = HEAD + frame_slots
    return (block * t_g + (block if t_g % 2 else 2 * frame_slots)) // 2


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class FlatLayout:
    """Where each part of a packed batch sits once every slot is a position.

    Local offsets count from the start of the job's or frame's window; an
    offset at or past ``context`` is not laid out.

    Attributes:
      job_row: Window of each job, ``[J]``.
      job_start: Offset of each job's first head position, ``[J]``.
      frame_row: Window of each frame, ``[F]``.
      frame_start: Offset of each frame's first slot; ``context`` for a frame
        no job generates and no window opens with, ``[F]``.
      block_at: Flat index of each global position's first flat position,
        ``[B·t_g + 1]``, ending with ``B·context``.
      positions: RoPE position within the flat segment, int32 ``[B, context]``.
      cu_seqlens: Flat segment boundaries, int32, shaped like the batch's.
      fits: Whether each window fits ``context``, ``[B]``.

    """

    job_row: Tensor
    job_start: Tensor
    frame_row: Tensor
    frame_start: Tensor
    block_at: Tensor
    positions: Tensor
    cu_seqlens: Tensor
    fits: Tensor


def flat_layout(batch: PackedBatch, *, context: int) -> FlatLayout:
    """Lay out every window of ``batch`` in ``context`` flat positions.

    Each ``start`` or ``act`` position becomes its job's ``HEAD`` positions and,
    when the job has one, its next frame; an ``obs`` position that opens a
    segment becomes its frame; other ``obs`` positions and padding take none,
    because the job before them already placed their frame. A padding job (a
    start job with no next frame, which the stream adds at position 0 to round
    the job count) takes none either.

    Args:
      batch: The packed micro-batch.
      context: Flat positions per window.

    Returns:
      layout: Offsets, RoPE positions, and segment boundaries.

    """
    rows, t_g = batch.kind.shape
    frames, device = len(batch.aux), batch.kind.device
    frame_slots = batch.cells.shape[-2] + batch.aux.shape[-1]
    job_at, job_next = batch.job_at.long(), batch.job_next.long()
    # A real start job always generates its episode's first frame.
    padding = batch.job_is_start & (job_next < 0)
    # One spare entry absorbs the padding jobs' writes.
    has_next = (
        torch.full((rows * t_g + 1,), -1, device=device).index_put(
            (torch.where(padding, rows * t_g, job_at),),
            job_next,
        )[:-1]
        >= 0
    )
    kind = batch.kind
    opens = batch.segment != functional.pad(batch.segment[:, :-1], (1, 0), value=-1)
    lead = (kind == Kind.OBS) & opens
    size = (
        torch.where(
            (kind == Kind.START) | (kind == Kind.ACT),
            HEAD + frame_slots * has_next.view(rows, t_g),
            0,
        )
        + frame_slots * lead
    )
    end = size.cumsum(-1)
    start = (end - size).clamp(max=context)
    window = torch.arange(rows, device=device)
    block_at = functional.pad(
        (start + context * window[:, None]).flatten(),
        (0, 1),
        value=rows * context,
    )
    cu_seqlens = block_at[batch.cu_seqlens.long()]
    flat = torch.arange(rows * context, device=device)
    segment = torch.searchsorted(cu_seqlens, flat, right=True) - 1
    job_row = job_at // t_g
    # A padding job's head lands past the context, in the inputs' spare row.
    job_start = torch.where(padding, context, start.flatten()[job_at])
    # One spare entry absorbs the writes of jobs without a next frame and of
    # positions that open nothing.
    placed = torch.where(job_next >= 0, job_next, frames)
    opened = torch.where(lead.flatten(), batch.frame_of.flatten().long(), frames)
    frame_start = (
        torch.full((frames + 1,), context, device=device)
        .index_put((placed,), job_start + HEAD)
        .index_put((opened,), start.flatten())
    )
    frame_row = (
        torch.zeros(frames + 1, dtype=torch.long, device=device)
        .index_put((placed,), job_row)
        .index_put((opened,), window.repeat_interleave(t_g))
    )
    return FlatLayout(
        job_row=job_row,
        job_start=job_start,
        frame_row=frame_row[:-1],
        frame_start=frame_start[:-1],
        block_at=block_at,
        positions=(flat - cu_seqlens[segment]).view(rows, context).int(),
        cu_seqlens=cu_seqlens.int(),
        fits=end[:, -1] <= context,
    )


class FlatModel(nn.Module):
    """The global stack as a language model over every Craftax slot."""

    class Config(Fig["FlatModel"]):
        """The global stack's language model over the Craftax vocabulary and slot schema."""

        schema: FrameSchema = field(default_factory=craftax_schema)
        """Slot layout and allowed IDs; its local prefix must be reward and done."""

        num_actions: int = 43
        """The game's actions, the ``action.*`` IDs from ``action_id(0)``."""

        context: int = 8_192
        """Flat positions per window; a window must fit, see ``flat_positions``."""

        lm: GlobalLanguageModel.Config = field(
            default_factory=GlobalLanguageModel.Config,
        )
        """Tied token table and the design-size global stack."""

        init_weight: InitFn = partial(nn.init.normal_, std=0.02)
        """Init of the slot embeddings and the start vector."""

        z_loss: float = 1e-5
        """Weight of each modality's mean squared log-normalizer."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            frames: int,
            jobs: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one packed batch's forward: the flat inputs, the stack, heads, loss.

            As ``WorldModel.Config.cost``, ``frames`` and ``jobs`` are the batch's
            counts. The stack runs over every flat position of every window,
            padding included; the layout's integer bookkeeping is left out.

            Args:
              seq_len: Global positions per window, ``t_g``.
              batch_size: Windows in this invocation.
              frames: Frames laid out as slots.
              jobs: Local jobs laid out as heads.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to the stack.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            schema, vocab = self.schema, self.lm.vocab_size
            width = self.lm.transformer.channels_in
            flat = batch_size * self.context
            local = jobs * schema.local_slots
            hidden = cost(
                self.lm.transformer,
                seq_len=self.context,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            heads = gather_cost(
                rows=local,
                width=width,
                source=flat,
                dtype=dtype,
            ) + logits_cost(
                channels_in=width,
                channels_out=vocab,
                rows=local,
                weight=False,
                dtype=dtype,
            )
            heads += gather_cost(
                rows=seq_len * batch_size,
                width=width,
                source=flat,
                dtype=dtype,
            ) + logits_cost(
                channels_in=width,
                channels_out=self.num_actions,
                rows=seq_len * batch_size,
                weight=False,
                dtype=dtype,
            )
            # A window that does not fit selects NaN over all of its logits.
            logits = local * vocab + seq_len * batch_size * self.num_actions
            heads += traffic(
                "primal",
                "elementwise",
                elements=2 * logits,
                dtype=dtype,
            ) + traffic("adjoint", "elementwise", elements=2 * logits, dtype=dtype)
            return (
                _inputs_cost(
                    self,
                    batch_size=batch_size,
                    frames=frames,
                    jobs=jobs,
                    dtype=dtype,
                )
                + hidden
                + heads
                + slot_loss_cost(
                    schema,
                    positions=seq_len * batch_size,
                    jobs=jobs,
                    num_actions=self.num_actions,
                )
            )

    def __init__(self, config: Config) -> None:
        schema = config.schema
        if config.lm.vocab_size != schema.vocab_size:
            raise ValueError(
                f"The table has {config.lm.vocab_size} rows for a schema of "
                f"{schema.vocab_size} IDs.",
            )
        if schema.prefix_names != ("reward", "done"):
            raise ValueError(f"Unsupported local prefix {schema.prefix_names}.")
        super().__init__()
        self.schema = schema
        self.scalar_offset = number_id(0)
        self.num_actions = config.num_actions
        self.context = config.context
        self.z_loss = config.z_loss
        self.lm = config.lm.make()
        width = config.lm.transformer.channels_in
        self.slot_embedding = nn.Parameter(
            torch.empty(HEAD + schema.frame_slots, width),
        )
        self.start = nn.Parameter(torch.empty(width))
        self._init_weight = config.init_weight
        self.cell_offsets: Tensor
        self.cell_index: Tensor
        self.scalar_rows: Tensor
        self.scalar_allowed: Tensor
        for name, table in schema_tables(self.schema).items():
            self.register_buffer(name, table, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter, and refill the schema tables, in place."""
        self.lm.reset_parameters()
        call_init(self._init_weight, self.slot_embedding)
        call_init(self._init_weight, self.start)
        with torch.no_grad():
            for name, table in schema_tables(self.schema).items():
                self.get_buffer(name).copy_(table)

    @override
    def forward(self, batch: PackedBatch) -> ModalityLoss:
        """Return per-modality NLL sums and counts, the objective, and z-loss."""
        return modality_loss(
            self.target_terms(batch, self.logits(batch)),
            z_loss=self.z_loss,
        )

    def logits(self, batch: PackedBatch) -> WorldModelLogits:
        """Run the stack over the flat form of every window.

        Args:
          batch: The packed micro-batch.

        Returns:
          logits: Action logits per global position and local logits per job,
            as ``WorldModel.logits`` returns them; NaN in a window that does
            not fit the context.

        Raises:
          ValueError: The window length can need more than ``context``.

        """
        rows, t_g = batch.kind.shape
        if flat_positions(t_g, frame_slots=self.schema.frame_slots) > self.context:
            raise ValueError(f"A window of {t_g} does not fit {self.context}.")
        layout = flat_layout(batch, context=self.context)
        hidden = self.lm.transformer(
            self._inputs(batch, layout),
            positions=layout.positions,
            cu_seqlens=layout.cu_seqlens,
        ).flatten(0, 1)
        # Hidden state i predicts flat position i + 1: a job's reward from its
        # first head position, and position t's action from the flat position
        # just before the head of the job at t + 1.
        local = hidden[
            _spread(
                layout.job_row,
                layout.job_start,
                width=self.schema.local_slots,
                context=self.context,
                spare=0,
            )
        ]
        table = self.lm.embedding.weight
        first = action_id(0)
        action = functional.linear(
            hidden[layout.block_at[1:] - 1],
            table[first : first + self.num_actions],
        ).view(rows, t_g, -1)
        fits = layout.fits
        local = functional.linear(local, table)
        return WorldModelLogits(
            action=torch.where(fits[:, None, None], action, torch.nan).float(),
            local=torch.where(
                fits[layout.job_row, None, None],
                local,
                torch.nan,
            ).float(),
        )

    def target_terms(
        self,
        batch: PackedBatch,
        logits: WorldModelLogits,
    ) -> dict[str, tuple[Nll, Tensor]]:
        """Return each modality's per-target NLL and scored mask: ``slot_terms``.

        Args:
          batch: The packed micro-batch.
          logits: ``self.logits(batch)``.

        Returns:
          terms: Per modality, the NLL and a bool mask of the same shape.

        """
        return slot_terms(
            batch,
            logits,
            schema=self.schema,
            scalar_offset=self.scalar_offset,
            cell_index=self.cell_index,
            scalar_rows=self.scalar_rows,
            scalar_allowed=self.scalar_allowed,
        )

    def frame_slots(self, cells: Tensor, aux: Tensor) -> Tensor:
        """Embed frames as ``embed_frames`` does, in the stack's tied table.

        Args:
          cells: The game's cell field values ``[..., cell_slots, K]``.
          aux: The game's scalar values ``[..., scalars]``.

        Returns:
          slots: ``[..., frame_slots, C]``.

        """
        return embed_frames(
            self.lm.embedding,
            cells,
            aux,
            cell_offsets=self.cell_offsets,
            scalar_offset=self.scalar_offset,
            embed_board=gathered_board,
        )

    def _inputs(self, batch: PackedBatch, layout: FlatLayout) -> Tensor:
        """Embed every flat position: job heads and frames plus their slot rows."""
        rows = len(layout.fits)
        spare = rows * self.context
        heads = self._heads(batch) + self.slot_embedding[:HEAD]
        frames = self.frame_slots(batch.cells, batch.aux) + self.slot_embedding[HEAD:]
        head_index = _spread(
            layout.job_row,
            layout.job_start,
            width=HEAD,
            context=self.context,
            spare=spare,
        )
        frame_index = _spread(
            layout.frame_row,
            layout.frame_start,
            width=self.schema.frame_slots,
            context=self.context,
            spare=spare,
        )
        # One spare row takes every write past its window's context; padding
        # takes none, so it stays zero.
        inputs = heads.new_zeros(spare + 1, heads.shape[-1]).index_add(
            0,
            torch.cat([head_index.flatten(), frame_index.flatten()]),
            torch.cat([heads.flatten(0, 1), frames.flatten(0, 1)]),
        )
        return inputs[:-1].view(rows, self.context, -1)

    def _heads(self, batch: PackedBatch) -> Tensor:
        """Embed each job's head: its action or the start vector, reward, done."""
        action = batch.action.flatten()[batch.job_at.long()].long() + action_id(0)
        prefix = prefix_ids(batch, schema=self.schema, scalar_offset=self.scalar_offset)
        heads = self.lm.embedding(torch.cat([action[:, None], prefix], dim=-1))
        first = torch.where(batch.job_is_start[:, None], self.start, heads[:, 0])
        return torch.cat([first[:, None], heads[:, 1:]], dim=1)


def flat_cost(model: nn.Module, media: object) -> ForwardCost:
    """Count the flat model's forward matmul FLOPs: the stack per flat position, then heads.

    Args:
      model: A ``FlatModel``.
      media: The ``PackedBatch`` it runs on.

    Returns:
      cost: Forward FLOPs, attention scores excluded, and the flat positions,
        padding included.

    """
    assert isinstance(model, FlatModel)
    assert isinstance(media, PackedBatch)
    rows, t_g = media.kind.shape
    positions = rows * model.context
    matrices = sum(p.numel() for p in model.lm.transformer.parameters() if p.ndim >= 2)
    table = model.lm.embedding.weight
    local = table.numel() * len(media.job_at) * model.schema.local_slots
    action = table.shape[-1] * model.num_actions * rows * t_g
    return ForwardCost(
        flops=2.0 * (matrices * positions + local + action),
        positions=positions,
    )


# A head is its action's or the start vector, then reward and done; a frame is
# ``embed_frames``'s slots in the stack's table. Each adds its rows of the slot
# embeddings, and one scatter-add places every row in the zeroed flat inputs; its
# adjoint gathers them back.
def _inputs_cost(
    config: FlatModel.Config,
    *,
    batch_size: int,
    frames: int,
    jobs: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost embedding the job heads and the frames, then scattering them into place."""
    schema = config.schema
    width = config.lm.transformer.channels_in
    # The head lookup owns the table the frame lookups read.
    table = Embedding.Config(channels_in=config.lm.vocab_size, channels_out=width)
    heads = (
        table.cost(seq_len=HEAD, batch_size=jobs, dtype=dtype)
        + elementwise_cost(
            primal=0,
            adjoint=0,
            channels=width,
            params=width,
            rows=jobs,
            adjoint_inputs=1,
            dtype=dtype,
        )
        + concat_cost(elements=jobs * HEAD * width, dtype=dtype)
        + broadcast_add_cost(params=HEAD * width, rows=jobs, dtype=dtype)
    )
    slots = (
        cost(
            gathered_board,
            rows=frames * schema.cell_slots,
            fields=len(schema.cell_fields),
            vocab=config.lm.vocab_size,
            width=width,
            dtype=dtype,
        )
        + table.cost(
            seq_len=schema.frame_slots - schema.cell_slots,
            batch_size=frames,
            dtype=dtype,
        ).tile(1, copies=0)
        + concat_cost(elements=frames * schema.frame_slots * width, dtype=dtype)
        + broadcast_add_cost(
            params=schema.frame_slots * width,
            rows=frames,
            dtype=dtype,
        )
    )
    target = batch_size * config.context + 1
    placed = traffic(
        "primal",
        "elementwise",
        elements=target * width,
        dtype=dtype,
    ) + gather_cost(
        rows=jobs * HEAD + frames * schema.frame_slots,
        width=width,
        source=target,
        dtype=dtype,
    )
    return heads + slots + placed


def _spread(
    row: Tensor,
    start: Tensor,
    *,
    width: int,
    context: int,
    spare: int,
) -> Tensor:
    """Return ``[N, width]`` flat indices from each offset; ``spare`` past a window."""
    offset = start[:, None] + torch.arange(width, device=start.device)
    return torch.where(offset < context, row[:, None] * context + offset, spare)
