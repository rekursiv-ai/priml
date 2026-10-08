"""World-model features recomputed from each step's stored context, with gradients.

The actor's feature at step ``t`` of a row (``feature.WorldModelFeature``) is the
final-normed hidden state at ``obs_t`` of the global sequence its history
holds: the last ``L_t`` decisions, from position 0, behind a ``start`` when
the first of them begins its episode. For the learner to train the world model
with the policy, it recomputes each feature from those decisions with its own
current weights. A context that starts at the same decision, under the same
anchor, as a later step's is a prefix of it: causal attention gives each
position exactly its prefix, so one pass over the later context computes both,
with the same function and the same gradient. :func:`plan_replay` groups a
window's steps by ``(row, first decision, anchor)``, one causal pass per group,
and packs the passes into bins of whole passes for the varlen kernels, a fixed
number of bins per micro-batch and a fixed number of frames per encoder
micro-batch, padded, so a compiled kernel sees few shapes; the
window's frames, each decision's tokens once, are encoded once.

Per window of ``R`` rows, ``T`` steps and ``P`` prefix slots (``feature.py``
computes the same function one step at a time; ``C`` is the global width)::

    o_k        = frame tokens of decision k               k < P: prefix slot k; k >= P: step k - P
    a_k        ∈ [43]                                     the action that led to decision k
    L_t        ∈ [1, 512],  A_t ∈ {0, 1}                  context length and anchor of step t
    s_t        = P + t - L_t + 1                          its first decision

    # frames, once per window: the F decisions some context reads
    E_k        = obs_proj(pool(enc_θ(o_k)))               ∈ ℝ^C

    # the context of step t, positions 0 .. 2 L_t - 2 + A_t
    x_t        = concat([start] if A_t, E_{s_t}, W_act[a_{s_t + 1}], E_{s_t + 1}, ..., W_act[a_{P+t}], E_{P+t})
    F_t        = norm(G_θ(x_t))[-1]                       ∈ ℝ^C   global blocks up to the tap, causal

    # sharing: (s_t, A_t) = (s_u, A_u), t < u  =>  x_t is a prefix of x_u
    F_t        = norm(G_θ(x_u))[A_u + 2 (P + t - s_u)]    one pass per group, read at each member's obs

The learner's gradient is two VJPs, so neither graph is held whole: the
policy's loss ``ℓ = L_φ(F)`` over ``F ∈ ℝ^{R×T×C}``, and the world model's
``F = glob_θ(enc_θ(U), c)`` over the window's frames ``U`` and contexts ``c``::

    F̂          = glob_θ(Ê, c),  Ê = enc_θ(U)             ∈ ℝ^{R×T×C}, ℝ^{F×C}   phase 1, no grad
    g_φ, G     = ∂ℓ/∂φ, ∂ℓ/∂F  at F = F̂                   ∈ ℝ^{R×T×C}           phase 2, the policy's backward
    g_θ^glob   = Σ_b ∂⟨G_b, glob_θ(Ê, c_b)⟩/∂θ                                    phase 3a, pass micro-batches b
    G_E        = Σ_b ∂⟨G_b, glob_θ(Ê, c_b)⟩/∂Ê           ∈ ℝ^{F×C}
    g_θ^enc    = Σ_m ∂⟨G_E[m], enc_θ(U[m])⟩/∂θ                                    phase 3b, frame micro-batches m
    ∂ℓ/∂θ      = g_θ^glob + g_θ^enc

Phase 3 recomputes phase 1's values in the same micro-batches, so the chain
rule holds at the point the policy's gradient was taken; the sum is the
gradient of one autograd graph through the whole computation, accumulated in
another order.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import copy

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.baselines.craftax.world_model.attention import (
    VarlenKernel,
)
from priml.baselines.craftax.world_model.batch import Kind
from priml.baselines.craftax.world_model.feature import (
    FeatureKernels,
    _frame_encoder,
    _global_block,
)
from priml.lib.codec import from_plain
from priml.model.attention.kernel import SdpaVarlen
from priml.model.attention.rope import RoPE
from priml.model.linear import EnsembleLinear
from priml.model.transformer.transformer import Transformer


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.baselines.craftax.world_model.model import WorldModel


@dataclass(frozen=True, slots=True, kw_only=True)
class Contexts:
    """A window of rows' decisions, and each step's context among them.

    A row's decisions are its ``P`` prefix slots, oldest first, then its ``T``
    steps: decision ``k`` is prefix slot ``k`` below ``P``, step ``k - P``
    from there. Step ``t``'s context is the ``lengths[r, t]`` decisions ending
    at its own, ``P + t``; the prefix slots it reaches must be among the row's
    last ``prefix_counts[r]``.

    Attributes:
      cells: Each step's frame, the codec's cell tokens, uint8 ``[R, T, 99, 8]``.
      aux: Its aux tokens, int16 ``[R, T, 51]``.
      previous_actions: The action that led to each step, any number dtype
        ``[R, T]``; unread where a context begins.
      lengths: Decisions in each step's context, its own included, integer
        ``[R, T]``; at least 1.
      anchored: Whether each context begins its episode, behind ``start``,
        bool ``[R, T]``.
      prefix_cells: The prefix slots' frames ``[R, P, 99, 8]``.
      prefix_aux: Their aux tokens ``[R, P, 51]``.
      prefix_previous_actions: The actions that led to them ``[R, P]``.
      prefix_counts: The filled prefix slots of each row, its last ones, ``[R]``.

    """

    cells: Tensor
    aux: Tensor
    previous_actions: Tensor
    lengths: Tensor
    anchored: Tensor
    prefix_cells: Tensor
    prefix_aux: Tensor
    prefix_previous_actions: Tensor
    prefix_counts: Tensor

    def rows(self, offset: int, count: int) -> Contexts:
        """Return the views of ``count`` rows from ``offset``.

        Args:
          offset: The first row.
          count: Rows to take.

        Returns:
          contexts: Views into these contexts, so nothing is copied.

        """
        rows = slice(offset, offset + count)
        return Contexts(
            cells=self.cells[rows],
            aux=self.aux[rows],
            previous_actions=self.previous_actions[rows],
            lengths=self.lengths[rows],
            anchored=self.anchored[rows],
            prefix_cells=self.prefix_cells[rows],
            prefix_aux=self.prefix_aux[rows],
            prefix_previous_actions=self.prefix_previous_actions[rows],
            prefix_counts=self.prefix_counts[rows],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PassBatch:
    """One micro-batch of causal passes, packed whole into ``bins`` rows of ``width`` tokens.

    A bin holds passes back to back and then padding, each its own segment.

    Attributes:
      kind: Each token's :class:`Kind` ``[bins, width]``; ``PAD`` past a bin's
        passes, and in every bin past the window's last.
      frame: The window frame each ``obs`` token reads, else 0, int64 ``[bins, width]``.
      action: The action each ``act`` token embeds, else 0, int64 ``[bins, width]``.
      positions: Each token's position within its pass, int64 ``[bins, width]``.
      cu_seqlens: The segments' boundaries over the flattened bins, int32.
      reads: The flat token at each computed step's ``obs``, int64 ``[n]``.
      steps: Those steps, ``row * T + t``, int64 ``[n]``.

    """

    kind: Tensor
    frame: Tensor
    action: Tensor
    positions: Tensor
    cu_seqlens: Tensor
    reads: Tensor
    steps: Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayPlan:
    """A window's frames, and its causal passes in micro-batches.

    Attributes:
      frame_cells: Every frame some context reads, row by row, uint8 ``[F, 99, 8]``.
      frame_aux: Their aux tokens ``[F, 51]``.
      batches: The packed passes, every step computed in exactly one.
      passes: Causal passes: one per (row, first decision, anchor).
      rows: ``R``.
      steps: ``T``.

    """

    frame_cells: Tensor
    frame_aux: Tensor
    batches: list[PassBatch]
    passes: int
    rows: int
    steps: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Replay:
    """A window's features under the weights of the moment, and the frames its backward reuses.

    Attributes:
      features: ``[R, T, C]`` in the model's dtype, without a graph.
      frames: Every frame's ``obs`` input ``[F, C]``, without a graph.
      plan: How the window ran.

    """

    features: Tensor
    frames: Tensor
    plan: ReplayPlan


class ContextReplay:
    """Recompute a window's world-model features, and backpropagate into the weights.

    :meth:`forward` encodes the window's frames and runs its passes without a
    graph; :meth:`backward` takes the features' gradient and recomputes both in
    the same micro-batches with one, so memory is bounded by a micro-batch, not
    the window (phases 1 and 3 of the module docstring).
    """

    class Config(Fig["ContextReplay"]):
        """The global attention and the micro-batches."""

        attention: Makeable[VarlenKernel] = field(default_factory=SdpaVarlen.Config)
        """Causal attention within the packed passes: masked SDPA anywhere, or
        ``Flash4Varlen`` on CUDA, which also runs its backward."""

        bin_tokens: int = 4_096
        """Tokens per bin, or the window's longest pass where that is longer:
        passes pack whole into bins, each bin a row of the varlen kernel. Wider
        bins waste fewer padding tokens on passes of mixed lengths; the kernel
        sizes its grid by a bin's width."""

        pass_tokens: int = 16_384
        """Tokens of packed passes per micro-batch, at least one bin's: what one
        backward through the global blocks holds."""

        frames_per_batch: int = 512
        """Frames per micro-batch of the frame encoder, the last padded: what one
        backward through it holds."""

        compile: (
            Makeable[Callable[[Callable[..., object]], Callable[..., object]]] | None
        ) = None
        """Wraps the per-block and frame-encoder functions once, e.g.
        ``PartialConfig(torch.compile, fullgraph=True, dynamic=False)``; None
        runs them eagerly. Each sees one micro-batch shape per window width, in
        each of grad mode and no-grad mode: with ``bin_tokens`` at least the
        longest pass, one width."""

    def __init__(self, config: Config) -> None:
        """Build the attention.

        Args:
          config: The attention and the micro-batches.

        Raises:
          ValueError: A micro-batch size is not positive.

        """
        sizes = (config.bin_tokens, config.pass_tokens, config.frames_per_batch)
        if min(sizes) < 1:
            raise ValueError(
                f"bin_tokens={config.bin_tokens}, pass_tokens={config.pass_tokens} "
                f"and frames_per_batch={config.frames_per_batch} must be positive.",
            )
        self.attention = config.attention.make()
        self.kernels = FeatureKernels.build(
            None if config.compile is None else config.compile.make(),
        )
        self.bin_tokens = config.bin_tokens
        self.pass_tokens = config.pass_tokens
        self.frames_per_batch = config.frames_per_batch

    @torch.no_grad()
    def forward(self, model: WorldModel, contexts: Contexts, *, layers: int) -> Replay:
        """Compute every step's feature from its context, without a graph.

        Reads the window's lengths, anchors and actions to the host once.

        Args:
          model: The world model, whose current weights compute.
          contexts: The window.
          layers: Global blocks up to the tap.

        Returns:
          replay: The features, and what :meth:`backward` reuses.

        """
        plan = plan_replay(
            contexts,
            bin_tokens=self.bin_tokens,
            pass_tokens=self.pass_tokens,
        )
        frames = torch.cat(
            [
                self.kernels.encode(model, _frame_encoder(model), cells, aux)[:count]
                for cells, aux, count in self._frame_batches(plan)
            ],
        )
        features = frames.new_empty(plan.rows * plan.steps, frames.shape[-1])
        for batch in plan.batches:
            hidden = self._hidden(model, frames, batch, layers=layers)
            features[batch.steps] = hidden[batch.reads]
        return Replay(
            features=features.view(plan.rows, plan.steps, -1),
            frames=frames,
            plan=plan,
        )

    def backward(
        self,
        model: WorldModel,
        replay: Replay,
        grad: Tensor,
        *,
        layers: int,
    ) -> None:
        """Add the gradient of ``<grad, features>`` to the weights' ``grad``.

        Args:
          model: The world model :meth:`forward` ran, with the same weights.
          replay: What it returned.
          grad: The loss's gradient with respect to the features ``[R, T, C]``.
          layers: Global blocks up to the tap, as :meth:`forward` had.

        """
        frames = replay.frames.detach().requires_grad_()
        flat = grad.reshape(-1, grad.shape[-1])
        for batch in replay.plan.batches:
            hidden = self._hidden(model, frames, batch, layers=layers)
            torch.autograd.backward(hidden[batch.reads], flat[batch.steps])
        frames_grad = frames.grad
        if frames_grad is None:
            raise ValueError("No pass read the window's frames.")
        first = 0
        for cells, aux, count in self._frame_batches(replay.plan):
            encoded = self.kernels.encode(model, _frame_encoder(model), cells, aux)
            torch.autograd.backward(
                encoded[:count],
                frames_grad[first : first + count],
            )
            first += count

    # Padding repeats the window's first frame; its encodings are dropped, so no
    # gradient reaches them.
    def _frame_batches(self, plan: ReplayPlan) -> list[tuple[Tensor, Tensor, int]]:
        """Split the window's frames into the encoder's micro-batches, the last padded."""
        size = self.frames_per_batch
        total = len(plan.frame_cells)
        padding = -total % size
        cells = torch.cat(
            [plan.frame_cells, plan.frame_cells[:1].expand(padding, -1, -1)],
        )
        aux = torch.cat([plan.frame_aux, plan.frame_aux[:1].expand(padding, -1)])
        return [
            (
                cells[first : first + size],
                aux[first : first + size],
                min(size, total - first),
            )
            for first in range(0, total, size)
        ]

    def _hidden(
        self,
        model: WorldModel,
        frames: Tensor,
        batch: PassBatch,
        *,
        layers: int,
    ) -> Tensor:
        """Return the final-normed states of one micro-batch's tokens, flat ``[bins * width, C]``."""
        blocks = [_global_block(block) for block in model.transformer.blocks[:layers]]
        rope = blocks[0][1].rope
        assert isinstance(rope, RoPE)
        cos, sin = rope(batch.positions.unsqueeze(-1))
        x = _inputs(model, frames, batch)
        for block, attn in blocks:
            q, k, v = self.kernels.pre(block, attn, x, cos, sin)
            out = self.attention(
                q,
                k,
                v.contiguous(),
                cu_seqlens=batch.cu_seqlens,
                record_max_logit=False,
            )
            x = self.kernels.post(block, attn, x, out)
        transformer = model.transformer
        assert isinstance(transformer, Transformer)
        return transformer.project_to_logits(x).flatten(0, 1)


class JointWorldModel:
    """A trainable copy of a feature's world model, and the replay of its features.

    The copy's feature parameters train: the frame table and encoder,
    ``obs_proj``, the action table, ``start``, the global blocks up to the tap
    and the final norm. Every other parameter stays the source's, unread by the
    feature. A weight of more than two dimensions, a fused QKV ensemble
    ``[heads, head width, width]``, trains as the matrix of its rows
    ``[heads * head width, width]``: an optimizer that orthogonalizes each
    parameter's ``[rows, -1]`` matrix (``FusedMuon``) then orthogonalizes the
    whole projection, as the reference's joint arms did, not one ``[heads,
    head width * width]`` matrix. The copy's ensemble projection computes
    from it as before, op for op.

    Attributes:
      model: The copy, on the source's device and in its dtype.
      source: The world model the actor and the evaluation read; :meth:`publish`
        writes it.
      layers: Global blocks up to the tap.
      replay: How each window's features are recomputed.
      guard: Held over each replay, forward and backward.

    """

    def __init__(
        self,
        source: WorldModel,
        *,
        layers: int,
        replay: ContextReplay,
        guard: AbstractContextManager[object] | None = None,
    ) -> None:
        """Copy the source and mark its feature parameters trainable.

        Args:
          source: The feature's world model, on the device and in the dtype the
            actor runs it.
          layers: Global blocks up to the tap.
          replay: How each window's features are recomputed.
          guard: Held over each replay, e.g. a lock another thread holds
            while it compiles: a compiled replay kernel called during
            another thread's compile raises, as torch flags an FX trace
            process-wide. None holds nothing.

        """
        self.guard = nullcontext() if guard is None else guard
        model = copy.deepcopy(source)
        model.requires_grad_(requires_grad=False)
        self.model = model
        self.source = source
        self.layers = layers
        self.replay = replay
        self._names = feature_parameter_names(source, layers=layers)
        self._targets = [source.get_parameter(name) for name in self._names]
        for name in self._names:
            path, _, _ = name.rpartition(".")
            owner = model.get_submodule(path)
            if model.get_parameter(name).ndim > 2:
                if not isinstance(owner, EnsembleLinear):
                    raise TypeError(
                        f"{name} has more than two dimensions outside an "
                        "EnsembleLinear.",
                    )
                parent, _, child = path.rpartition(".")
                model.get_submodule(parent).register_module(
                    child,
                    _RowEnsemble(owner),
                )
        self._leaves = [
            model.get_parameter(name).requires_grad_() for name in self._names
        ]

    def parameters(self) -> list[Tensor]:
        """Return the trained parameters, in the source's parameter order."""
        return list(self._leaves)

    def weights(self) -> dict[str, Tensor]:
        """Return the trained weights by the source's names, in its shapes.

        Returns:
          weights: Live views of the trained parameters, without a graph.

        """
        return {
            name: leaf.detach().view(target.shape)
            for name, leaf, target in zip(
                self._names,
                self._leaves,
                self._targets,
                strict=True,
            )
        }

    @torch.no_grad()
    def load_weights(self, weights: Mapping[str, Tensor]) -> None:
        """Copy :meth:`weights` of another copy into this one's parameters.

        Args:
          weights: The trained weights by the source's names, in its shapes.

        Raises:
          ValueError: ``weights`` names other parameters than this copy trains.

        """
        if sorted(weights) != sorted(self._names):
            missing = sorted(set(self._names) ^ set(weights))
            msg = f"the world model's trained weights differ by {missing}"
            raise ValueError(msg)
        for name, leaf in zip(self._names, self._leaves, strict=True):
            leaf.copy_(weights[name].reshape(leaf.shape))

    @torch.no_grad()
    def publish(self) -> None:
        """Copy the trained weights into the source, in place."""
        for target, weight in zip(self._targets, self.weights().values(), strict=True):
            target.copy_(weight)

    def forward(self, contexts: Contexts) -> Replay:
        """Recompute a window's features with the current weights, without a graph."""
        with self.guard:
            return self.replay.forward(self.model, contexts, layers=self.layers)

    def backward(self, replay: Replay, grad: Tensor) -> None:
        """Add the gradient of ``<grad, features>`` to the trained parameters' ``grad``."""
        with self.guard:
            self.replay.backward(self.model, replay, grad, layers=self.layers)


def feature_parameter_names(model: WorldModel, *, layers: int) -> list[str]:
    """Name the parameters a feature tapping ``layers`` global blocks reads, in model order.

    Args:
      model: The world model.
      layers: Global blocks up to the tap.

    Returns:
      names: The frame table and encoder, ``obs_proj``, the action table,
        ``start``, the first ``layers`` global blocks and the final norm.

    """
    transformer = model.transformer
    read: list[nn.Module] = [
        model.table,
        _frame_encoder(model),
        model.obs_proj,
        model.action_embedding,
        *transformer.blocks[:layers],
    ]
    if isinstance(transformer.proj_out, nn.Module):
        read.append(transformer.proj_out)
    ids = {id(model.start)} | {id(p) for module in read for p in module.parameters()}
    return [name for name, p in model.named_parameters() if id(p) in ids]


def plan_replay(contexts: Contexts, *, bin_tokens: int, pass_tokens: int) -> ReplayPlan:
    """Group a window's steps into causal passes, pack them into bins and list the frames.

    The steps of a row whose contexts share their first decision and anchor are
    one pass, as long as the longest. Passes go into bins of ``bin_tokens``, or
    of the longest pass where that is longer, longest first, each into the last
    bin while it fits (next fit decreasing); ``pass_tokens // width`` bins make
    a micro-batch, the last padded with empty bins. Reads the lengths, anchors,
    prefix counts and actions to the host once, and plans there.

    Args:
      contexts: The window.
      bin_tokens: Tokens per bin, at least the longest pass's.
      pass_tokens: Tokens per micro-batch; at least one bin's.

    Returns:
      plan: The frames and the micro-batches of packed passes, on the
        contexts' device.

    Raises:
      ValueError: A context is empty, or reaches a prefix slot its row never
        filled.

    """
    rows, steps = contexts.lengths.shape
    slots = contexts.prefix_cells.shape[1]
    actions = torch.cat(
        [contexts.prefix_previous_actions, contexts.previous_actions],
        dim=1,
    )
    host = torch.cat(
        [
            contexts.lengths.flatten().long(),
            contexts.anchored.flatten().long(),
            contexts.prefix_counts.long(),
            actions.flatten().long(),
        ],
    ).cpu()
    lengths, anchored, counts, stream = host.split(
        [rows * steps, rows * steps, rows, actions.numel()],
    )
    groups = _group(
        lengths.view(rows, steps),
        anchored.view(rows, steps),
        counts,
        slots=slots,
    )
    frames = _frames(groups, rows=rows)
    device = contexts.cells.device
    index = (frames.row.to(device), frames.decision.to(device))
    return ReplayPlan(
        frame_cells=torch.cat([contexts.prefix_cells, contexts.cells], dim=1)[index],
        frame_aux=torch.cat([contexts.prefix_aux, contexts.aux], dim=1)[index],
        batches=_batches(
            groups,
            frame_offset=frames.offset,
            stream=stream,
            bin_tokens=bin_tokens,
            pass_tokens=pass_tokens,
            device=device,
        ),
        passes=len(groups.first),
        rows=rows,
        steps=steps,
    )


# ---- Planning, on the host, in int64. ----


@dataclass(frozen=True, slots=True, kw_only=True)
class _Groups:
    """Each pass's row, first decision, anchor and tokens; each step's pass and decision.

    A row's decisions run ``0..span - 1``, its prefix slots and then its steps.
    """

    row: Tensor
    first: Tensor
    anchor: Tensor
    tokens: Tensor
    step_pass: Tensor
    step_decision: Tensor
    span: int


@dataclass(frozen=True, slots=True, kw_only=True)
class _Frames:
    """The window's frames, row by row: each one's row and decision, and each row's offset.

    Decision ``k`` of row ``r`` is frame ``offset[r] + k``.
    """

    row: Tensor
    decision: Tensor
    offset: Tensor


def _group(lengths: Tensor, anchored: Tensor, counts: Tensor, *, slots: int) -> _Groups:
    """Return one pass per (row, first decision, anchor), as long as its longest member."""
    rows, steps = lengths.shape
    span = slots + steps
    end = (slots + torch.arange(steps)).repeat(rows)
    first = end - lengths.flatten() + 1
    # A row's filled slots are its last ones, and none precedes slot 0.
    floor = (slots - counts).clamp(min=0).repeat_interleave(steps)
    # On the host already: counting them reads no device.
    unreadable = int(((first > end) | (first < floor)).sum())
    if unreadable:
        raise ValueError(
            f"{unreadable} contexts lack their own step or reach unfilled prefix slots.",
        )
    row = torch.arange(rows).repeat_interleave(steps)
    keys, inverse = ((row * span + first) * 2 + anchored.flatten()).unique(
        return_inverse=True,
    )
    last = torch.zeros_like(keys).scatter_reduce(0, inverse, end, reduce="amax")
    anchor = keys % 2
    pass_first = keys // 2 % span
    return _Groups(
        row=keys // (2 * span),
        first=pass_first,
        anchor=anchor,
        tokens=2 * (last - pass_first) + 1 + anchor,
        step_pass=inverse,
        step_decision=end,
        span=span,
    )


def _frames(groups: _Groups, *, rows: int) -> _Frames:
    """Return every decision from each row's earliest context to its last step."""
    earliest = torch.full((rows,), groups.span).scatter_reduce(
        0,
        groups.row,
        groups.first,
        reduce="amin",
    )
    count = groups.span - earliest
    offset = count.cumsum(0) - count - earliest
    return _Frames(
        row=torch.arange(rows).repeat_interleave(count),
        decision=torch.arange(int(count.sum())) - offset.repeat_interleave(count),
        offset=offset,
    )


# Every micro-batch holds ``pass_tokens // width`` bins, the last padded with empty
# ones, so every micro-batch of a window has one shape.
def _batches(
    groups: _Groups,
    *,
    frame_offset: Tensor,
    stream: Tensor,
    bin_tokens: int,
    pass_tokens: int,
    device: torch.device,
) -> list[PassBatch]:
    """Pack the passes into bins, the bins into micro-batches, and lay out their tokens."""
    width = max(bin_tokens, int(groups.tokens.max()))
    bin_of, offset = _next_fit_decreasing(groups.tokens, width=width)
    per = max(1, pass_tokens // width)
    batch_of = bin_of // per
    start = (bin_of - batch_of * per) * width + offset
    step_pass = groups.step_pass
    reads = (
        start[step_pass]
        + groups.anchor[step_pass]
        + 2 * (groups.step_decision - groups.first[step_pass])
    )
    count = int(batch_of.max()) + 1
    return [
        _layout(
            groups,
            passes=passes,
            start=start[passes],
            bins=per,
            width=width,
            frame_offset=frame_offset,
            stream=stream,
            reads=reads[steps],
            steps=steps,
            device=device,
        )
        for passes, steps in zip(
            _split(batch_of, count),
            _split(batch_of[step_pass], count),
            strict=True,
        )
    ]


def _next_fit_decreasing(tokens: Tensor, *, width: int) -> tuple[Tensor, Tensor]:
    """Return each pass's bin and its offset there, longest passes first."""
    bin_of = torch.empty_like(tokens)
    offset = torch.empty_like(tokens)
    current, used = -1, width
    sizes = from_plain(tokens.tolist(), list[int])
    order = torch.argsort(tokens, descending=True, stable=True)
    for index in from_plain(order.tolist(), list[int]):
        if used + sizes[index] > width:
            current, used = current + 1, 0
        bin_of[index], offset[index] = current, used
        used += sizes[index]
    return bin_of, offset


def _split(labels: Tensor, count: int) -> list[Tensor]:
    """Return the indices of each label ``0..count - 1``, ascending within each."""
    order = torch.argsort(labels, stable=True)
    sizes = from_plain(torch.bincount(labels, minlength=count).tolist(), list[int])
    return list(order.split(sizes))


def _layout(
    groups: _Groups,
    *,
    passes: Tensor,
    start: Tensor,
    bins: int,
    width: int,
    frame_offset: Tensor,
    stream: Tensor,
    reads: Tensor,
    steps: Tensor,
    device: torch.device,
) -> PassBatch:
    """Lay out one micro-batch's passes: each token's kind, frame, action and position."""
    tokens = groups.tokens[passes]
    owner = torch.arange(len(passes)).repeat_interleave(tokens)
    position = torch.arange(int(tokens.sum())) - (tokens.cumsum(0) - tokens)[owner]
    flat = start[owner] + position
    after = position - groups.anchor[passes][owner]
    decision = groups.first[passes][owner] + after.div(2, rounding_mode="floor")
    row = groups.row[passes][owner]
    obs, act = (after >= 0) & (after % 2 == 0), (after >= 0) & (after % 2 == 1)
    kind = torch.full((bins * width,), int(Kind.PAD), dtype=torch.uint8)
    kind[flat] = torch.where(
        obs,
        int(Kind.OBS),
        torch.where(act, int(Kind.ACT), int(Kind.START)),
    ).to(torch.uint8)
    frame = torch.zeros(bins * width, dtype=torch.int64)
    frame[flat[obs]] = (frame_offset[row] + decision)[obs]
    # The action that led to the next decision: the one taken at this one.
    action = torch.zeros(bins * width, dtype=torch.int64)
    action[flat[act]] = stream[(row * groups.span + decision + 1)[act]]
    positions = torch.zeros(bins * width, dtype=torch.int64)
    positions[flat] = position
    ends = torch.cat([start + tokens, width * torch.arange(1, bins + 1)]).unique()
    return PassBatch(
        kind=kind.view(bins, width).to(device),
        frame=frame.view(bins, width).to(device),
        action=action.view(bins, width).to(device),
        positions=positions.view(bins, width).to(device),
        cu_seqlens=torch.cat([ends.new_zeros(1), ends]).to(device, torch.int32),
        reads=reads.to(device),
        steps=steps.to(device),
    )


# ---- The world model's modules, as the replay reads them. ----
#
# Its blocks and encoder are narrowed by the engine's own validators: the replay
# computes the engine's function, so a layout or kernel the engine refuses, the
# replay refuses too, rather than recomputing a different one.


def _inputs(model: WorldModel, frames: Tensor, batch: PassBatch) -> Tensor:
    """Place ``start``, the frames' ``obs`` inputs and the action embeddings, as training does."""
    kind = batch.kind[..., None]
    obs = frames[batch.frame]
    # Clamped as the actor clamps the action it embeds.
    action = batch.action.clamp(0, model.action_embedding.weight.shape[0] - 1)
    act = model.action_embedding(action) * (kind == Kind.ACT)
    x = torch.where(kind == Kind.OBS, obs, act)
    return torch.where(kind == Kind.START, model.start, x)


class _RowEnsemble(nn.Module):
    """An ``EnsembleLinear`` without a bias, its weight held as the matrix of its rows.

    It computes as ``EnsembleLinear.forward`` does, op for op. One class for
    every block: a parametrization's per-instance class would make a compiled
    block function guard on each block's type and compile once per block.
    """

    def __init__(self, ensemble: EnsembleLinear) -> None:
        """Take over ``ensemble``'s weight as ``[members * channels_out, channels_in]``.

        Args:
          ensemble: The projection, which must have no bias.

        Raises:
          ValueError: ``ensemble`` has a bias.

        """
        if ensemble.bias is not None:
            raise ValueError("A trained ensemble projection must have no bias.")
        super().__init__()
        weight = ensemble.weight.detach()
        self.weight = nn.Parameter(weight.reshape(-1, weight.shape[-1]).clone())
        self.members = tuple(weight.shape[:-1])

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        """Project ``[..., channels_in]`` to ``[..., members, channels_out]``."""
        del kwargs
        out = torch.matmul(x, self.weight.to(x.dtype).T)
        return out.reshape(*x.shape[:-1], *self.members)


__all__ = [
    "ContextReplay",
    "Contexts",
    "JointWorldModel",
    "PassBatch",
    "Replay",
    "ReplayPlan",
    "feature_parameter_names",
    "plan_replay",
]
