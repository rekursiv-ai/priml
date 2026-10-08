"""Sampling engine: batched per-row decoding of the hierarchical world model.

Each engine row holds one episode in progress: a static global KV cache
``[L, B, T_max + 1, H_kv, D]`` whose last position is scratch, a per-row
length, the current frame, a ring of the last ``T_max / 4`` decisions for
re-prefill, and a ``needs_start`` flag.
One decision runs the design's structured step for every row at once:

1. ``start``: rows that need one feed the ``start`` position and run a local
   job over the null memory with forced ``reward = 0, done = false``; its frame
   becomes ``o_0``. Other rows run the same kernels masked.
2. ``decide``: encode the current frame; the global ``obs`` step reads the
   action head and applies each row's action source; the ``act`` step gives the
   conditioning vector; the local job decodes reward, done, and the next frame
   slot by slot with Gumbel-max sampling under the schema's masks, where a
   per-row forced mask overrides any sample. ``done = true`` sets ``needs_start``.

Nothing inside ``start`` or ``decide`` reads a tensor value on the host, so
``GraphedStep`` captures each as a CUDA graph. ``step`` reads one flag before
the decision, whether any active row begins an episode, and when none does it
runs ``skip_start`` instead: masked for every row, the start step changes
nothing but the generator and costs as much as ``decide``, so ``skip_start``
only draws the start job's noise. Every step then draws the same noise, and
one row's reset never changes another row's samples.

New keys and values go into the static cache through ``index_copy_`` at each
row's length, and attention runs over the whole buffer under a per-row
key-length mask. A row's cache always begins at its segment's first position,
so its length is also its RoPE position. A row past its window, whose caller
skipped ``ensure_room``, writes only its own scratch position, never the next
row's cache; its own samples are then meaningless.

The engine runs its own attention in place of each attention's kernel slot: a
masked SDPA, the function of ``SdpaVarlen`` (and of ``Flash4Varlen``, rounding
apart) in the global stack and of ``SdpaFused`` in the decoder, so a model
whose slots hold any other kernel is refused rather than decoded differently.

Between decisions, ``ensure_room`` re-prefills every row that has no room for
another decision from its last ``T_max / 4`` decisions at position 0, exactly
the state of a mid-episode training window, and returns how many decisions
every row can then take without another check.
"""

from typing import Protocol, cast

import dataclasses

from torch import Tensor
from torch.nn import functional

import torch

from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    GlobalTransformer,
    WorldModel,
)
from priml.math.probability import gumbel_max, random_gumbel
from priml.model.attention.attention import Attention
from priml.model.attention.flash4 import Flash4Varlen
from priml.model.attention.kernel import SdpaFused, SdpaVarlen
from priml.model.attention.rope import RoPE
from priml.model.transformer.block import TransformerBlock


class Policy(Protocol):
    """An action source reading decoded observations."""

    def __call__(self, observation: Tensor, /) -> Tensor:
        """Map float32 observations ``[R, 843]`` to the game's actions ``[R]``."""
        ...


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Prefix:
    """Decisions to prefill without local decoding, the same count for every row.

    Attributes:
      cells: Cell values of each frame and of the frame after the last
        decision, integer ``[R, N + 1, cells, fields]``.
      aux: Auxiliary values of the same frames, integer ``[R, N + 1, scalars]``.
      actions: Executed actions, integer ``[R, N]``.
      starts_episode: Whether decision 0 is the episode's first decision.

    """

    cells: Tensor
    aux: Tensor
    actions: Tensor
    starts_episode: bool


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Control:
    """Per-row inputs of one decision; ``Engine.control`` builds a free one.

    Local tokens use the job layout of ``Engine.job_tokens``: ``[B, L, fields]``
    vocabulary IDs, where a single-ID slot holds its ID in field 0.

    Attributes:
      active: Rows that take the decision; the others keep their state, bool ``[B]``.
      start_forced: Start-job slots whose tokens are forced, bool ``[B, L]``.
      start_tokens: Forced start-job tokens, long ``[B, L, fields]``.
      action_forced: Rows whose action is forced, bool ``[B]``.
      action: Forced actions, long ``[B]``.
      job_forced: Act-job slots whose tokens are forced, bool ``[B, L]``.
      job_tokens: Forced act-job tokens, long ``[B, L, fields]``.

    """

    active: Tensor
    start_forced: Tensor
    start_tokens: Tensor
    action_forced: Tensor
    action: Tensor
    job_forced: Tensor
    job_tokens: Tensor


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class JobResult:
    """The tokens of one local job per row and their log-probabilities.

    Attributes:
      tokens: Chosen tokens in job layout, long ``[B, L, fields]``.
      logp: Log-probability of each slot's tokens, summed over a cell's fields;
        0 for slots never scored (a start job's prefix, a frame after
        ``done``), float32 ``[B, L]``.

    """

    tokens: Tensor
    logp: Tensor


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class StartResult:
    """Output of the start step.

    Attributes:
      started: Rows that began an episode in this step, bool ``[B]``.
      job: The start job of every row; meaningful only where ``started``.

    """

    started: Tensor
    job: JobResult


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    """Output of one decision.

    Attributes:
      action: Executed action, long ``[B]``.
      action_logp: Its log-probability under the action head, float32 ``[B]``.
      job: The act job: reward, done, and the next frame.

    """

    action: Tensor
    action_logp: Tensor
    job: JobResult


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Outcome:
    """A job's tokens as values; see ``Engine.outcome``.

    Attributes:
      reward: Reward value, long ``[...]``.
      done: Terminal flag, bool ``[...]``.
      cells: Cell values, long ``[..., cells, fields]``.
      aux: Auxiliary values, long ``[..., scalars]``.

    """

    reward: Tensor
    done: Tensor
    cells: Tensor
    aux: Tensor


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EngineState:
    """Per-row decoding state; every tensor is updated in place.

    Attributes:
      keys: Global key cache, ``[layers, B, t_max + 1, heads_kv, head_width]``;
        position ``t_max`` is scratch for masked writes of full rows.
      values: Global value cache, shaped like ``keys``.
      length: Valid cache positions, which is also the next RoPE position, long ``[B]``.
      cells: Current frame's cell values, long ``[B, cells, fields]``.
      aux: Current frame's auxiliary values, long ``[B, scalars]``.
      history_cells: Ring of recent decisions' frames, uint8 ``[B, t_max / 4, ...]``.
      history_aux: Ring of their auxiliary values, int16 ``[B, t_max / 4, scalars]``.
      history_actions: Ring of their actions, uint8 ``[B, t_max / 4]``.
      count: Decisions in the cache, long ``[B]``; the ring slot is ``count % (t_max / 4)``.
      needs_start: Rows whose next step begins an episode, bool ``[B]``.

    """

    keys: Tensor
    values: Tensor
    length: Tensor
    cells: Tensor
    aux: Tensor
    history_cells: Tensor
    history_aux: Tensor
    history_actions: Tensor
    count: Tensor
    needs_start: Tensor


class Engine:
    """Batched sampling of episodes from a ``WorldModel``, one episode per row."""

    def __init__(
        self,
        model: WorldModel,
        *,
        rows: int,
        t_max: int,
        generator: torch.Generator,
        prefill_rows: int = 4,
    ) -> None:
        """Allocate every row's caches.

        Args:
          model: The world model to decode; its schema has the reward and done
            prefix.
          rows: Episodes decoded at once.
          t_max: Global positions per row, the training window.
          generator: Source of every sample.
          prefill_rows: Full rows ``ensure_room`` re-prefills per global forward.
            Rows that start together fill together, and one forward over all of
            them holds each one's ``t_max / 4`` embedded frames at once: a
            256-row engine at ``t_max = 8192`` asked for 184 GiB at its first
            re-prefill.

        Raises:
          ValueError: If ``t_max`` is not a multiple of 4 of at least 8, the
            schema has no reward and done prefix, or a block or attention
            kernel is one the engine does not decode.

        """
        schema = model.schema
        if t_max < 8 or t_max % 4:
            raise ValueError(f"t_max={t_max} must be a multiple of 4, at least 8.")
        if not schema.prefix_ranges:
            raise ValueError("The engine decodes a reward and done prefix; got none.")
        self.model = model
        self.generator = generator
        self.rows = rows
        self.t_max = t_max
        self.prefill_rows = prefill_rows
        assert isinstance(model.transformer, GlobalTransformer)
        self._transformer: GlobalTransformer = model.transformer
        self._global = [_global_layer(block) for block in model.transformer.blocks]
        self._local = [_local_layer(block) for block in model.decoder.stack.blocks]
        if model.decoder.stack.proj_in is not None:
            raise ValueError(
                "The engine decodes a decoder stack without an input projection.",
            )
        parameter = model.start
        device, dtype = parameter.device, parameter.dtype
        self._cells = schema.cell_slots
        self._fields = len(schema.cell_fields)
        self._slots = schema.local_slots
        self._done_false, self._done_true = schema.prefix_ranges[1]
        allowed = schema.local_allowed().to(device)
        bias = torch.zeros(allowed.shape, device=device)
        self._bias = bias.masked_fill(~allowed, float("-inf"))
        self._local_mask = torch.ones(
            self._slots,
            self._slots,
            dtype=torch.bool,
            device=device,
        ).tril()
        # One scratch position past t_max: a full row's masked writes land there,
        # where in a t_max buffer they would overwrite the next row's position 0.
        self._positions = torch.arange(t_max + 1, device=device)
        self._row_index = torch.arange(rows, device=device)
        attn = self._global[0][1]
        cache = (
            len(self._global),
            rows,
            t_max + 1,
            attn.num_heads_kv,
            attn.channels_head,
        )
        memory = t_max // 4
        scalars = len(schema.scalar_ranges)
        self.state = EngineState(
            keys=torch.zeros(cache, device=device, dtype=dtype),
            values=torch.zeros(cache, device=device, dtype=dtype),
            length=torch.zeros(rows, dtype=torch.long, device=device),
            cells=torch.zeros(
                rows,
                self._cells,
                self._fields,
                dtype=torch.long,
                device=device,
            ),
            aux=torch.zeros(rows, scalars, dtype=torch.long, device=device),
            history_cells=torch.zeros(
                rows,
                memory,
                self._cells,
                self._fields,
                dtype=torch.uint8,
                device=device,
            ),
            history_aux=torch.zeros(
                rows,
                memory,
                scalars,
                dtype=torch.int16,
                device=device,
            ),
            history_actions=torch.zeros(rows, memory, dtype=torch.uint8, device=device),
            count=torch.zeros(rows, dtype=torch.long, device=device),
            needs_start=torch.ones(rows, dtype=torch.bool, device=device),
        )

    def control(self) -> Control:
        """Return a control in which every row is active and nothing is forced.

        Returns:
          control: Fresh tensors on the engine's device, safe to edit in place.

        """
        device = self.state.length.device
        rows, slots, fields = self.rows, self._slots, self._fields
        return Control(
            active=torch.ones(rows, dtype=torch.bool, device=device),
            start_forced=torch.zeros(rows, slots, dtype=torch.bool, device=device),
            start_tokens=torch.zeros(
                rows,
                slots,
                fields,
                dtype=torch.long,
                device=device,
            ),
            action_forced=torch.zeros(rows, dtype=torch.bool, device=device),
            action=torch.zeros(rows, dtype=torch.long, device=device),
            job_forced=torch.zeros(rows, slots, dtype=torch.bool, device=device),
            job_tokens=torch.zeros(
                rows,
                slots,
                fields,
                dtype=torch.long,
                device=device,
            ),
        )

    def job_tokens(
        self,
        *,
        reward: Tensor,
        done: Tensor,
        cells: Tensor,
        aux: Tensor,
    ) -> Tensor:
        """Return one job's tokens in job layout from values.

        Args:
          reward: Reward value, integer ``[...]``.
          done: Terminal flag, bool ``[...]``.
          cells: Cell values, integer ``[..., cells, fields]``.
          aux: Auxiliary values, integer ``[..., scalars]``.

        Returns:
          tokens: Vocabulary IDs, long ``[..., L, fields]``; a single-ID slot
            holds its ID in field 0 and zeros elsewhere.

        """
        offset = self.model.scalar_offset
        prefix = torch.stack(
            [reward.long() + offset, done.long() + self._done_false],
            dim=-1,
        ).to(aux.device)
        scalars = torch.cat([prefix, aux.long() + offset], dim=-1)
        single = functional.pad(scalars[..., None], (0, self._fields - 1))
        board = cells.long() + self.model.cell_offsets.to(cells.device)
        return torch.cat([single[..., :2, :], board, single[..., 2:, :]], dim=-2)

    def outcome(self, tokens: Tensor) -> Outcome:
        """Return the values of job-layout tokens.

        Args:
          tokens: Vocabulary IDs in job layout, ``[..., L, fields]``.

        Returns:
          outcome: Reward, done, and the frame as values; the frame is
            meaningless where ``done``.

        """
        offset, offsets = self.model.scalar_offset, self.model.cell_offsets
        frame = tokens[..., 2:, :]
        return Outcome(
            reward=tokens[..., 0, 0] - offset,
            done=tokens[..., 1, 0] == self._done_true,
            cells=frame[..., : self._cells, :] - offsets,
            aux=frame[..., self._cells :, 0] - offset,
        )

    def reset(self, rows: Tensor) -> None:
        """Mark rows, bool ``[B]``, to begin a new episode at their next step."""
        self.state.needs_start.logical_or_(rows.to(self.state.needs_start.device))

    def set_frame(self, row: int, *, cells: Tensor, aux: Tensor) -> None:
        """Replace one row's current frame before it is encoded."""
        self.state.cells[row] = cells
        self.state.aux[row] = aux

    def copy_row(self, source: int, target: int) -> None:
        """Copy every piece of one row's state onto another row.

        Args:
          source: Row to copy.
          target: Row to overwrite.

        """
        state = self.state
        state.keys[:, target] = state.keys[:, source]
        state.values[:, target] = state.values[:, source]
        for field in dataclasses.fields(state):
            if field.name not in {"keys", "values"}:
                tensor = cast("Tensor", getattr(state, field.name))
                tensor[target] = tensor[source]

    @torch.no_grad()
    def prefill(self, rows: Tensor, prefix: Prefix) -> None:
        """Load real decisions into ``rows`` with one global forward, no local jobs.

        A prefix longer than the cache allows keeps its last ``t_max / 4``
        decisions from position 0, as a mid-episode training window does.

        Args:
          rows: Row indices, long ``[R]``.
          prefix: The decisions and frames, ``R`` rows of them.

        """
        keep = self.t_max // 4
        head = int(prefix.starts_episode)
        decisions = prefix.actions.shape[-1]
        if head + 2 * decisions > self.t_max - 2:
            prefix = Prefix(
                cells=prefix.cells[:, -keep - 1 :],
                aux=prefix.aux[:, -keep - 1 :],
                actions=prefix.actions[:, -keep:],
                starts_episode=False,
            )
            head, decisions = 0, keep
        device = self.state.length.device
        rows = rows.to(device)
        cells, aux = prefix.cells.to(device).long(), prefix.aux.to(device).long()
        actions = prefix.actions.to(device).long()
        model = self.model
        inputs = model.start.expand(len(rows), head, -1)
        if decisions:
            pooled, _ = model.encoder(model.frame_slots(cells[:, :-1], aux[:, :-1]))
            steps = torch.stack(
                [model.obs_proj(pooled), model.action_embedding(actions)],
                dim=2,
            ).flatten(1, 2)
            inputs = torch.cat([inputs, steps], dim=1)
        length = inputs.shape[1]
        if length:
            at = self._positions[:length].expand(len(rows), length)
            self._global_forward(inputs, at=at, rows=rows)
        state = self.state
        state.length[rows] = length
        state.cells[rows] = cells[:, -1]
        state.aux[rows] = aux[:, -1]
        state.needs_start[rows] = False
        state.count[rows] = decisions
        steps = torch.arange(max(decisions - keep, 0), decisions, device=device)
        slots = (steps % keep)[None]
        state.history_cells[rows[:, None], slots] = cells[:, steps].to(torch.uint8)
        state.history_aux[rows[:, None], slots] = aux[:, steps].to(torch.int16)
        state.history_actions[rows[:, None], slots] = actions[:, steps].to(torch.uint8)

    @torch.no_grad()
    def ensure_room(self) -> int:
        """Re-prefill full rows, then return how many decisions fit in every row.

        Reads the device once, so call it between decisions, never inside one.

        Returns:
          decisions: Decisions every row can take, including restarts, before
            the next call.

        """
        state = self.state
        keep = self.t_max // 4
        full = ((state.length > self.t_max - 2) & ~state.needs_start).nonzero()[:, 0]
        for first in range(0, len(full), self.prefill_rows):
            rows = full[first : first + self.prefill_rows]
            steps = state.count[rows, None] - keep + self._positions[:keep]
            slots = steps % keep
            cells = state.history_cells[rows[:, None], slots].long()
            aux = state.history_aux[rows[:, None], slots].long()
            self.prefill(
                rows,
                Prefix(
                    cells=torch.cat([cells, state.cells[rows, None]], dim=1),
                    aux=torch.cat([aux, state.aux[rows, None]], dim=1),
                    actions=state.history_actions[rows[:, None], slots],
                    starts_episode=False,
                ),
            )
        # A restarted row holds at most 1 position before its first decision,
        # so the longest row, floored at 1, bounds every row's growth.
        longest = int(state.length.masked_fill(state.needs_start, 0).max().clamp(min=1))
        return (self.t_max - 2 - longest) // 2 + 1

    @torch.no_grad()
    def step(self, control: Control) -> tuple[StartResult, Decision]:
        """Run one full decision: the start step if a row begins, then ``decide``.

        Reads one flag from the device before the decision, never inside it.

        Args:
          control: The inputs of ``start`` and ``decide``.

        Returns:
          begun: The start step, or ``skip_start`` where no row begins.
          decision: The decision.

        """
        begun = (
            self.start(control) if self.starting(control) else self.skip_start(control)
        )
        return begun, self.decide(control)

    def starting(self, control: Control) -> bool:
        """Return whether an active row begins an episode at its next start step.

        Reads the device once, so call it between decisions, never inside one.

        Args:
          control: Its ``active`` rows are the ones that count.

        Returns:
          starting: Whether any active row has ``needs_start`` set.

        """
        return bool((self.state.needs_start & control.active).any())

    @torch.no_grad()
    def start(self, control: Control) -> StartResult:
        """Begin an episode in every active row that needs one.

        Args:
          control: ``active``, ``start_forced``, and ``start_tokens`` are read,
            the forced tokens only in starting rows; the start job's reward and
            done are always forced to 0 and false.

        Returns:
          result: The started rows and every row's start job.

        """
        state, model = self.state, self.model
        started = state.needs_start & control.active
        state.length.masked_fill_(started, 0)
        hidden = self._global_forward(
            model.start.expand(self.rows, 1, -1),
            at=state.length[:, None],
            rows=None,
        )
        state.length.add_(started.long())
        forced = control.start_forced & started[:, None]
        forced[:, :2] = True
        tokens = control.start_tokens.clone()
        tokens[:, 0, 0] = model.scalar_offset
        tokens[:, 1, 0] = self._done_false
        frame_slots = self._slots - 2
        null = model.decoder.memory_null.expand(self.rows, frame_slots, -1)
        job = self._local_job(
            model.cond_proj(hidden[:, 0]),
            memory=null,
            forced=forced,
            tokens=tokens,
        )
        logp = functional.pad(job.logp[:, 2:], (2, 0))
        frame = self.outcome(job.tokens)
        state.cells.copy_(torch.where(started[:, None, None], frame.cells, state.cells))
        state.aux.copy_(torch.where(started[:, None], frame.aux, state.aux))
        state.count.masked_fill_(started, 0)
        state.needs_start.logical_and_(~started)
        return StartResult(started=started, job=JobResult(tokens=job.tokens, logp=logp))

    @torch.no_grad()
    def skip_start(self, control: Control) -> StartResult:
        """Replace a start step that begins no row: draw its noise, change nothing.

        The start job's noise is drawn and discarded, so every step draws the
        same noise and a reset in one row never moves another row's samples.

        Args:
          control: Only its shapes and device are read.

        Returns:
          result: No row started, and a zero job.

        """
        model = self.model
        for slot in range(self._slots):
            # The shapes, dtype, and order of ``_choose``'s draws: any other draw moves
            # the generator differently from a start step, coupling the rows again.
            cell = 2 <= slot < 2 + self._cells
            shape = model.cell_index.shape if cell else model.table.weight.shape[:1]
            random_gumbel(
                self.rows,
                *shape,
                generator=self.generator,
                dtype=torch.float32,
                device=control.active.device,
            )
        return StartResult(
            started=torch.zeros_like(control.active),
            job=JobResult(
                tokens=torch.zeros_like(control.start_tokens),
                logp=torch.zeros_like(control.start_forced, dtype=torch.float32),
            ),
        )

    @torch.no_grad()
    def decide(self, control: Control) -> Decision:
        """Encode, choose actions, and decode reward, done, and the next frame.

        Every active row must have begun its episode (``needs_start`` false).

        Args:
          control: The action source and forced act-job tokens of every row.

        Returns:
          decision: Actions, their log-probabilities, and the act jobs.

        """
        state, model = self.state, self.model
        pooled, memory = model.encoder(model.frame_slots(state.cells, state.aux))
        hidden = self._global_forward(
            model.obs_proj(pooled)[:, None],
            at=state.length[:, None],
            rows=None,
        )
        state.length.add_(control.active.long())
        action, action_logp, hidden = self._act(control, obs=hidden[:, 0])
        job = self._local_job(
            model.cond_proj(hidden[:, 0]),
            memory=memory,
            forced=control.job_forced,
            tokens=control.job_tokens,
        )
        self._remember(control.active, action)
        frame = self.outcome(job.tokens)
        advance = control.active & ~frame.done
        state.cells.copy_(torch.where(advance[:, None, None], frame.cells, state.cells))
        state.aux.copy_(torch.where(advance[:, None], frame.aux, state.aux))
        state.needs_start.copy_(
            torch.where(control.active, frame.done, state.needs_start),
        )
        return Decision(action=action, action_logp=action_logp, job=job)

    @torch.no_grad()
    def action_logits(self) -> Tensor:
        """Return each row's action-head logits at its current frame, deciding nothing.

        The ``obs`` step's keys and values land at each row's next position,
        which the next decision writes over, so no row's state changes.

        Returns:
          logits: Float32 ``[B, actions]``.

        """
        state, model = self.state, self.model
        pooled, _ = model.encoder(model.frame_slots(state.cells, state.aux))
        hidden = self._global_forward(
            model.obs_proj(pooled)[:, None],
            at=state.length[:, None],
            rows=None,
        )
        return model.action_head(hidden[:, 0]).float()

    def _act(self, control: Control, *, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Choose each row's action from the ``obs`` step, then run the ``act`` step."""
        state, model = self.state, self.model
        logits = model.action_head(obs).float()
        sampled = gumbel_max(logits, generator=self.generator)
        action = torch.where(control.action_forced, control.action, sampled)
        action_logp = logits.log_softmax(-1).gather(-1, action[:, None])[:, 0]
        hidden = self._global_forward(
            model.action_embedding(action)[:, None],
            at=state.length[:, None],
            rows=None,
        )
        state.length.add_(control.active.long())
        return action, action_logp, hidden

    def _global_forward(self, x: Tensor, *, at: Tensor, rows: Tensor | None) -> Tensor:
        """Run the global stack on new positions, writing their keys and values."""
        for layer, (block, attn) in enumerate(self._global):
            h = block.norm1(x)
            q, k, v = attn.proj_qkv(h).split(
                [attn.num_heads, attn.num_heads_kv, attn.num_heads_kv],
                dim=-2,
            )
            if attn.norm_q is not None:
                q = attn.norm_q(q)
            if attn.norm_k is not None:
                k = attn.norm_k(k)
            if attn.rope is not None:
                cos, sin = attn.rope(at.unsqueeze(-1))
                q, k = RoPE.rotate(q, k, cos, sin)
            keys, values = self._write(layer, k, v, at=at, rows=rows)
            out = functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                keys.transpose(1, 2),
                values.transpose(1, 2),
                attn_mask=(self._positions <= at[..., None])[:, None],
                enable_gqa=True,
            )
            x = x + attn.proj_out(out.transpose(1, 2).flatten(-2))
            x = x + block.ffn(block.norm2(x))
        return self._transformer.project_to_logits(x)

    def _write(
        self,
        layer: int,
        k: Tensor,
        v: Tensor,
        *,
        at: Tensor,
        rows: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Write new keys and values at ``at``; return the rows' full caches."""
        index = self._row_index if rows is None else rows
        # A position past t_max is the next row's cache; clamped, a row run past
        # ``ensure_room`` overwrites only its own scratch, with no host read.
        flat = (index[:, None] * (self.t_max + 1) + at.clamp(max=self.t_max)).flatten()
        keys, values = self.state.keys[layer], self.state.values[layer]
        for cache, new in ((keys, k), (values, v)):
            cache.flatten(0, 1).index_copy_(0, flat, new.flatten(0, 1).to(cache.dtype))
        if rows is None:
            return keys, values
        return keys.index_select(0, rows), values.index_select(0, rows)

    def _local_job(
        self,
        cond: Tensor,
        *,
        memory: Tensor,
        forced: Tensor,
        tokens: Tensor,
    ) -> JobResult:
        """Decode one local job per row, slot by slot."""
        decoder = self.model.decoder
        cross = [_cross_memory(layer.cross, memory) for layer in self._local]
        caches = [
            (
                cond.new_zeros(
                    self.rows,
                    self._slots,
                    layer.attn.num_heads_kv,
                    layer.attn.channels_head,
                ),
                cond.new_zeros(
                    self.rows,
                    self._slots,
                    layer.attn.num_heads_kv,
                    layer.attn.channels_head,
                ),
            )
            for layer in self._local
        ]
        x = cond[:, None]
        done = torch.zeros(self.rows, dtype=torch.bool, device=cond.device)
        chosen: list[Tensor] = []
        logps: list[Tensor] = []
        for slot in range(self._slots):
            h = x + decoder.slot_embedding[slot]
            for layer, cache, memory_kv in zip(self._local, caches, cross, strict=True):
                h = self._local_block(
                    layer,
                    h,
                    slot=slot,
                    cache=cache,
                    memory_kv=memory_kv,
                )
            h = decoder.stack.project_to_logits(h)
            logits = functional.linear(h[:, 0], self.model.output_weight).float()
            ids, logp = self._choose(
                logits,
                slot=slot,
                forced=forced[:, slot],
                tokens=tokens[:, slot],
            )
            if slot == 1:
                done = ids[:, 0] == self._done_true
            chosen.append(ids)
            logps.append(torch.where(done, 0.0, logp) if slot >= 2 else logp)
            x = self._embed(ids, slot=slot)[:, None]
        return JobResult(
            tokens=torch.stack(chosen, dim=1),
            logp=torch.stack(logps, dim=1),
        )

    def _local_block(
        self,
        layer: "_LocalLayer",
        h: Tensor,
        *,
        slot: int,
        cache: tuple[Tensor, Tensor],
        memory_kv: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Run one decoder block on local position ``slot`` of every row."""
        block, attn = layer.block, layer.attn
        q, k, v = attn.proj_qkv(block.norm1(h)).split(
            [attn.num_heads, attn.num_heads_kv, attn.num_heads_kv],
            dim=-2,
        )
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        keys, values = cache
        keys[:, slot] = k[:, 0]
        values[:, slot] = v[:, 0]
        out = functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask=self._local_mask[slot : slot + 1],
            enable_gqa=True,
        )
        h = h + attn.proj_out(out.transpose(1, 2).flatten(-2))
        h = h + _cross_attend(layer.cross, block.norm2(h), memory_kv=memory_kv)
        return h + block.ffn(block.norm3(h))

    def _choose(
        self,
        logits: Tensor,
        *,
        slot: int,
        forced: Tensor,
        tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Sample or force one slot's tokens; return them and their log-probability."""
        if 2 <= slot < 2 + self._cells:
            offsets = self.model.cell_offsets
            padded = functional.pad(logits, (0, 1), value=float("-inf"))
            fields = padded[:, self.model.cell_index]
            sampled = gumbel_max(fields, generator=self.generator)
            value = torch.where(forced[:, None], tokens - offsets, sampled)
            logp = fields.log_softmax(-1).gather(-1, value[..., None])[..., 0].sum(-1)
            return value + offsets, logp
        biased = logits + self._bias[slot]
        sampled = gumbel_max(biased, generator=self.generator)
        value = torch.where(forced, tokens[:, 0], sampled)
        logp = biased.log_softmax(-1).gather(-1, value[:, None])[:, 0]
        return functional.pad(value[:, None], (0, self._fields - 1)), logp

    def _embed(self, ids: Tensor, *, slot: int) -> Tensor:
        """Embed one slot's tokens as the model's local inputs do."""
        if 2 <= slot < 2 + self._cells:
            return self.model.table(ids).sum(-2)
        return self.model.table(ids[:, 0])

    def _remember(self, active: Tensor, action: Tensor) -> None:
        """Write active rows' current frame and action into their history ring."""
        state = self.state
        keep = state.history_actions.shape[1]
        flat = self._row_index * keep + state.count % keep
        for ring, value in (
            (state.history_cells, state.cells.to(torch.uint8)),
            (state.history_aux, state.aux.to(torch.int16)),
            (state.history_actions, action.to(torch.uint8)),
        ):
            view = ring.flatten(0, 1)
            mask = active.view(-1, *[1] * (value.ndim - 1))
            view.index_copy_(
                0,
                flat,
                torch.where(mask, value, view.index_select(0, flat)),
            )
        state.count.add_(active.long())


class GraphedStep:
    """``Engine.step`` as CUDA graphs: ``start`` or ``skip_start``, then ``decide``.

    Each call copies the control into static buffers, reads whether an active
    row begins an episode, and replays. The returned results are static too:
    the next call overwrites them.
    """

    def __init__(self, engine: Engine, *, warmup: int = 3) -> None:
        """Warm up on a side stream, then capture start, skip_start, and decide.

        Args:
          engine: A CUDA engine sampling from its device's default generator,
            whose state CUDA graph capture tracks without registration; its
            device need not be the current one.
          warmup: Idle steps run before capture.

        Raises:
          ValueError: If the engine samples from another generator.

        """
        device = engine.state.length.device
        if engine.generator is not torch.cuda.default_generators[device.index or 0]:
            raise ValueError(
                "A graphed engine samples from its default CUDA generator.",
            )
        self._engine = engine
        self._control = engine.control()
        idle = engine.control()
        # Inactive rows keep their state, so warming up changes only the
        # generator and cache positions past every row's length.
        idle.active.zero_()
        # Streams and graphs belong to the current device, and ``torch.cuda.graph``
        # otherwise captures on one side stream shared by every device.
        with torch.cuda.device(device):
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    engine.start(idle)
                    engine.decide(idle)
            torch.cuda.current_stream().wait_stream(side)
            self._start_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._start_graph, stream=side):
                self._start = engine.start(self._control)
            self._skip_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._skip_graph, stream=side):
                self._skip = engine.skip_start(self._control)
            self._decide_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._decide_graph, stream=side):
                self._decision = engine.decide(self._control)

    def __call__(self, control: Control) -> tuple[StartResult, Decision]:
        """Run one decision with ``control``; see ``Engine.step``."""
        for field in dataclasses.fields(control):
            cast("Tensor", getattr(self._control, field.name)).copy_(
                cast("Tensor", getattr(control, field.name)),
            )
        starting = self._engine.starting(self._control)
        with torch.cuda.device(self._control.active.device):
            (self._start_graph if starting else self._skip_graph).replay()
            self._decide_graph.replay()
        return (self._start if starting else self._skip), self._decision


# ---- The attention modules, as the engine reads them. ----
#
# Everything below names the attention classes and the cross-attention's
# projections, and nothing above does: a change to those modules (a renamed
# class, another projection layout) is a change to these helpers alone. Each
# mirrors its module's forward, which ``engine_test`` holds the engine to.


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _LocalLayer:
    """One decoder block and its two attentions, narrowed to what the engine runs."""

    block: DecoderBlock
    attn: Attention
    cross: Attention


def _global_layer(block: object) -> tuple[TransformerBlock, VarlenAttention]:
    """Narrow a global block to the pre-norm Qwen3 block the engine decodes."""
    assert isinstance(block, TransformerBlock)
    attn = block.attn
    assert isinstance(attn, VarlenAttention)
    if not block.prenorm or attn.split_qkv_projection or attn.norm_out is not None:
        raise ValueError(
            "The engine decodes pre-norm global blocks with a fused QKV projection "
            "and no output norm.",
        )
    # FA4 is SdpaVarlen's function, rounded otherwise: ``scoring.load_trained`` keeps
    # it on CUDA, and fidelity dreams from that model (engine_checks measures the gap).
    if type(attn.attn_kernel) not in {SdpaVarlen, Flash4Varlen}:
        raise ValueError(
            "The engine decodes global attention as SdpaVarlen computes it; got "
            f"{type(attn.attn_kernel).__name__}.",
        )
    return block, attn


def _local_layer(block: object) -> _LocalLayer:
    """Narrow a decoder block to the one the engine decodes slot by slot."""
    assert isinstance(block, DecoderBlock)
    attn, cross = block.attn, block.cross_attn
    assert isinstance(attn, Attention)
    assert isinstance(cross, Attention)
    if attn.split_qkv_projection or attn.rope is not None or attn.norm_out is not None:
        raise ValueError(
            "The engine decodes decoder self-attention with a fused QKV projection, "
            "no rotary embedding and no output norm.",
        )
    if cross.kv_groups > 1 or cross.norm_out is not None:
        raise ValueError(
            "The engine decodes cross-attention with as many key heads as query "
            "heads and no output norm.",
        )
    kernels = (type(attn.attn_kernel), type(cross.attn_kernel))
    if kernels != (SdpaFused, SdpaFused):
        raise ValueError(
            "The engine decodes decoder attention as SdpaFused computes it; got "
            f"{kernels[0].__name__} and {kernels[1].__name__}.",
        )
    return _LocalLayer(block=block, attn=attn, cross=cross)


def _cross_memory(cross: Attention, memory: Tensor) -> tuple[Tensor, Tensor]:
    """Return a job's cross-attention keys and values, ``[B, heads, M, width]`` each."""
    k, v = cross.project_memory(memory)
    if cross.norm_k is not None:
        k = cross.norm_k(k)
    return k.movedim(-2, -3), v.movedim(-2, -3)


def _cross_attend(
    cross: Attention,
    x: Tensor,
    *,
    memory_kv: tuple[Tensor, Tensor],
) -> Tensor:
    """Attend ``[B, 1, C]`` queries to a job's memory keys and values."""
    q = cross.project_queries(x)
    if cross.norm_q is not None:
        q = cross.norm_q(q)
    out = functional.scaled_dot_product_attention(q.movedim(-2, -3), *memory_kv)
    return cross.proj_out(out.movedim(-3, -2).flatten(-2))
