"""Winner-take-all training over several fast-latent hypotheses.

Each puzzle runs several fast-latent hypotheses from one shared slow latent.
The step trains only the hypothesis with the lowest supervised loss, carries
that winner's slow latent, and carries every hypothesis's fast latent forward.
Evaluation is the plain single-chain path from :attr:`HypothesisRecursion.fast_init` row 0.

- :class:`HypothesisRecursion` stores the extra initial rows and draws per-puzzle
  hypotheses from a dedicated generator, never the ambient RNG.
- :class:`WinnerTakeAll` keeps every hypothesis in the gradient graph.
- :class:`WinnerRecompute` routes without gradient, then recomputes only the
  winner with it: the same winner at a fraction of the activation memory.

References:
  https://arxiv.org/abs/2601.19085
    Speed is Confidence.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, override

import math

from configgle import Fig, Makes
from torch import Tensor, nn

import torch

from priml.baselines.sudoku.latent_recursion import LatentRecursion
from priml.baselines.sudoku.routing import Routed, Routing
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.sudoku.act import AtomicPool


class Hypotheses(NamedTuple):
    """One forward over every hypothesis, not yet selected."""

    logits: Tensor
    """``[B, K, grid_len, V]`` grid logits."""

    q_halt: Tensor
    """``[B, K]`` halt logits."""

    z_slow: Tensor
    """``[B, K, S, C]`` slow latents."""

    z_fast: Tensor
    """``[B, K, S, C]`` fast latents."""


class HypothesisRecursion(LatentRecursion):
    """A TRM holding several fast-latent initializations; row 0 deploys."""

    class Config(Makes["HypothesisRecursion"], LatentRecursion.Config):
        """The TRM plus its hypothesis count and dedicated seed."""

        num_hypotheses: int = 2
        """Fast-latent hypotheses trained per puzzle."""

        hypothesis_seed: int = 0
        """Seeds the stored extra rows and every per-puzzle draw."""

    config: Config  # pyrefly: ignore[bad-override-mutable-attribute] -- Narrowed to this class's own config; nothing assigns the base's.
    """Narrowed from the parent's declaration for the hypothesis fields."""

    def __init__(self, config: Config) -> None:
        if config.num_hypotheses < 2:
            raise ValueError(
                f"num_hypotheses must be at least 2, got {config.num_hypotheses}.",
            )
        super().__init__(config)
        generator = torch.Generator(device=self.fast_init.device)
        generator.manual_seed(config.hypothesis_seed)
        rows = [self.fast_init[0]]
        rows.extend(
            truncated_normal_(torch.empty_like(self.fast_init[0]), generator=generator)
            for _ in range(config.num_hypotheses - 1)
        )
        self.fast_init = nn.Buffer(torch.stack(rows), persistent=True)
        with torch.no_grad():
            equalize_alignment(self.fast_init, self.ffn_subspace())

    @torch.no_grad()
    def fresh_fast_states(self, batch_size: int, *, seed: int) -> Tensor:
        """Draw ``[B, K, S, C]`` per-puzzle, per-position fast latents.

        Args:
          batch_size: Puzzle rows.
          seed: Seed of this draw's dedicated generator.

        Returns:
          z_fast: Every hypothesis's fast latent, aligned to the FFN subspace.

        """
        config = self.config
        states = torch.empty(
            (
                batch_size,
                config.num_hypotheses,
                config.total_seq_len,
                config.channels_in,
            ),
            device=self.fast_init.device,
            dtype=self.fast_init.dtype,
        )
        generator = torch.Generator(device=self.fast_init.device)
        generator.manual_seed(seed)
        truncated_normal_(states, generator=generator)
        equalize_alignment(states, self.ffn_subspace(), batched=True)
        return states

    def embed(
        self,
        input_ids: Tensor,
        puzzle_identifiers: Tensor | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Embed once for every hypothesis, consuming the feedback stash.

        Args:
          input_ids: ``[B, grid_len]`` puzzles.
          puzzle_identifiers: ``[B]`` ids when the model has a puzzle table.

        Returns:
          input_emb: ``[B, S, C]`` prefix-prepended embedding.
          cos_sin: Rotary factors.

        """
        feedback = self._feedback_ids
        self._feedback_ids = None
        return self._embed_and_prepare(input_ids, puzzle_identifiers, feedback)

    def hypotheses(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor],
    ) -> Hypotheses:
        """Run every hypothesis as one ``B * K`` batch from a shared embedding.

        Args:
          input_emb: ``[B, S, C]`` from :meth:`embed`.
          z_slow: ``[B, S, C]`` shared slow latent.
          z_fast: ``[B, K, S, C]`` per-hypothesis fast latents.
          cos_sin: Rotary factors from :meth:`embed`.

        Returns:
          hypotheses: Every hypothesis's outputs, under the caller's gradient.

        """
        batch, count, length, width = z_fast.shape
        result = self.run_slow_cycles(
            _tile(input_emb, count),
            _tile(z_slow, count),
            z_fast.reshape(batch * count, length, width),
            cos_sin,
        )
        logits = result.logits[:, self.config.num_prefix_tokens :]
        return Hypotheses(
            logits=logits.reshape(batch, count, *logits.shape[1:]),
            q_halt=result.q_halt.reshape(batch, count),
            z_slow=result.z_slow.reshape(batch, count, length, width),
            z_fast=result.z_fast.reshape(batch, count, length, width),
        )

    def ffn_subspace(self) -> Tensor:
        """Return the top-10 right singular vectors of block 0's FFN up-projection.

        Returns:
          basis: ``[C, r]`` orthonormal columns, ``r <= 10``.

        """
        block = self.reasoning[0]
        if not isinstance(block, TransformerBlock) or not isinstance(
            block.ffn,
            SwiGLU,
        ):
            raise TypeError(
                "HypothesisRecursion requires a TransformerBlock with a SwiGLU FFN.",
            )
        _, _, vectors = torch.linalg.svd(
            block.ffn.up_proj.weight.float(),
            full_matrices=False,
        )
        return vectors[:10].T.to(self.fast_init)


class WinnerTakeAll(Routing):
    """Train the lowest-loss hypothesis, gradient through every hypothesis.

    Every seated puzzle draws fresh hypotheses from
    ``hypothesis_seed + 1_024 * global_step``; each hypothesis carries its own
    fast latent across ACT steps, while the slow latent follows the winner.
    """

    class Config(Fig["WinnerTakeAll"]):
        """No parameters: the hypotheses live on the model."""

    def __init__(self, config: Config) -> None:
        del config
        self.fast: Tensor | None = None

    @override
    def seat(
        self,
        model: LatentRecursion,
        pool: AtomicPool,
        *,
        global_step: int,
    ) -> None:
        net = _hypothesis_recursion(model)
        fast = self.fast
        if fast is None:
            count = net.config.num_hypotheses
            fast = pool.z_fast.unsqueeze(1).expand(-1, count, -1, -1).clone()
        fresh = pool.active & (pool.steps == 0)
        seated = int(fresh.sum())
        if seated:
            seed = net.config.hypothesis_seed + global_step * 1_024
            fast[fresh] = net.fresh_fast_states(seated, seed=seed).to(fast.dtype)
        self.fast = fast

    @override
    def __call__(
        self,
        model: LatentRecursion,
        pool: AtomicPool,
        *,
        token_loss: Callable[[Tensor, Tensor], Tensor],
        ignore_index: int,
    ) -> Routed:
        net, fast = _hypothesis_recursion(model), self._carried()
        input_emb, cos_sin = net.embed(pool.inputs, _identifiers(net, pool))
        heads = net.hypotheses(input_emb, pool.z_slow, fast, cos_sin)
        losses = hypothesis_losses(
            heads.logits,
            pool.labels,
            token_loss,
            ignore_index=ignore_index,
        )
        winner = losses.argmin(dim=-1)
        rows = torch.arange(winner.shape[0], device=winner.device)
        self._carry(heads.z_fast.detach(), active=pool.active)
        return Routed(
            logits=heads.logits[rows, winner],
            q_halt=heads.q_halt[rows, winner],
            z_slow=heads.z_slow[rows, winner].detach(),
            z_fast=heads.z_fast[rows, winner].detach(),
            metrics=winner_metrics(losses.detach(), winner, active=pool.active),
        )

    def _carried(self) -> Tensor:
        """Every hypothesis's carried fast latent; ``seat`` runs first."""
        if self.fast is None:
            raise RuntimeError("seat() must run before the routed forward.")
        return self.fast

    def _carry(self, z_fast: Tensor, *, active: Tensor) -> None:
        """Carry every hypothesis's fast latent for the active slots."""
        carried = self._carried()
        self.fast = torch.where(
            active.view(-1, 1, 1, 1),
            z_fast.to(carried.dtype),
            carried,
        )


class WinnerRecompute(WinnerTakeAll):
    """Route every hypothesis without gradient, then recompute the winner with it."""

    class Config(Makes["WinnerRecompute"], WinnerTakeAll.Config):
        """No parameters: the hypotheses live on the model."""

    @override
    def __call__(
        self,
        model: LatentRecursion,
        pool: AtomicPool,
        *,
        token_loss: Callable[[Tensor, Tensor], Tensor],
        ignore_index: int,
    ) -> Routed:
        net, fast = _hypothesis_recursion(model), self._carried()
        input_emb, cos_sin = net.embed(pool.inputs, _identifiers(net, pool))
        with torch.no_grad():
            heads = net.hypotheses(input_emb, pool.z_slow, fast, cos_sin)
            losses = hypothesis_losses(
                heads.logits,
                pool.labels,
                token_loss,
                ignore_index=ignore_index,
            )
            winner = losses.argmin(dim=-1)
        rows = torch.arange(winner.shape[0], device=winner.device)
        out = net.run_slow_cycles(input_emb, pool.z_slow, fast[rows, winner], cos_sin)
        carried = heads.z_fast.clone()
        carried[rows, winner] = out.z_fast.detach().to(carried.dtype)
        self._carry(carried, active=pool.active)
        return Routed(
            logits=out.logits[:, net.config.num_prefix_tokens :].float(),
            q_halt=out.q_halt.float(),
            z_slow=out.z_slow.detach(),
            z_fast=carried[rows, winner],
            metrics=winner_metrics(losses, winner, active=pool.active),
        )


def hypothesis_losses(
    logits: Tensor,
    labels: Tensor,
    token_loss: Callable[[Tensor, Tensor], Tensor],
    *,
    ignore_index: int,
) -> Tensor:
    """Return ``[B, K]`` mean per-token loss over each puzzle's scored cells.

    Args:
      logits: ``[B, K, grid_len, V]`` per-hypothesis logits.
      labels: ``[B, grid_len]`` solutions.
      token_loss: ``[N, grid_len]`` per-token loss of ``[N, grid_len, V]``.
      ignore_index: Label of cells that neither score nor count.

    Returns:
      losses: Per-puzzle, per-hypothesis loss.

    """
    batch, count, length, vocab = logits.shape
    labels_k = labels.unsqueeze(1).expand(batch, count, length)
    per_token = token_loss(
        logits.reshape(batch * count, length, vocab),
        labels_k.reshape(batch * count, length),
    ).reshape(batch, count, length)
    counted = (labels_k != ignore_index).sum(dim=-1).clamp(min=1)
    return per_token.sum(dim=-1) / counted


def winner_metrics(
    losses: Tensor,
    winner: Tensor,
    *,
    active: Tensor,
) -> dict[str, float | Tensor]:
    """Return head utilization and the winner's loss margin over the runner-up.

    Args:
      losses: ``[B, K]`` per-hypothesis losses.
      winner: ``[B]`` selected hypothesis.
      active: ``[B]`` slots that trained this step.

    Returns:
      metrics: ``hypothesis_loss_gap``, ``hypothesis_effective_heads`` (the win distribution's
        exponentiated entropy), and ``hypothesis_max_win_fraction``.

    """
    with torch.autocast(device_type=losses.device.type, enabled=False):
        n_active = active.sum().clamp(min=1)
        wins = nn.functional.one_hot(winner, num_classes=losses.shape[1]).float()
        share = torch.where(active.unsqueeze(-1), wins, 0.0).sum(dim=0) / n_active
        entropy = -(share * share.clamp_min(1e-38).log()).sum()
        ordered = losses.sort(dim=-1).values
        gap = torch.where(active, ordered[:, 1] - ordered[:, 0], 0.0).sum() / n_active
    return {
        "hypothesis_loss_gap": gap.detach(),
        "hypothesis_effective_heads": entropy.exp().detach(),
        "hypothesis_max_win_fraction": share.max().detach(),
    }


def equalize_alignment(states: Tensor, top: Tensor, *, batched: bool = False) -> None:
    """Match every hypothesis's projection onto ``top`` to hypothesis 0's, in place.

    Args:
      states: ``[K, C]`` rows, or ``[B, K, S, C]`` when ``batched``.
      top: ``[C, r]`` orthonormal subspace basis.
      batched: Equalize per puzzle and position against that row's hypothesis 0.

    """
    eps = 1e-8
    top = top.to(states)
    reference = states[:, 0] if batched else states[0]
    unit = reference / reference.norm(dim=-1, keepdim=True)
    target = (unit @ top).norm(dim=-1, keepdim=True)
    for head in range(1, states.shape[1 if batched else 0]):
        vector = (states[:, head] if batched else states[head]).clone()
        original = vector.norm(dim=-1, keepdim=True)
        parallel = (vector @ top) @ top.T
        perpendicular = vector - parallel
        parallel_norm = (
            target.square()
            * perpendicular.norm(dim=-1, keepdim=True).square()
            / (1 - target.square() + eps)
        ).sqrt()
        current = parallel.norm(dim=-1, keepdim=True)
        if batched:
            parallel = torch.where(
                current > eps,
                parallel / (current + eps) * parallel_norm,
                parallel,
            )
            equalized = parallel + perpendicular
            states[:, head] = (
                equalized / (equalized.norm(dim=-1, keepdim=True) + eps) * original
            )
        else:
            # One row: a scalar guard, read once per hypothesis at construction.
            if bool(current.squeeze() > eps):
                parallel = parallel / current * parallel_norm
            equalized = parallel + perpendicular
            states[head] = equalized / equalized.norm() * original


def truncated_normal_(tensor: Tensor, *, generator: torch.Generator) -> Tensor:
    """Fill ``tensor`` with a unit-variance normal truncated to two std, in place.

    Args:
      tensor: Destination.
      generator: Dedicated RNG.

    Returns:
      tensor: The filled destination.

    Derivation:
      Inverse-CDF sampling: ``u ~ U(erf(-2/sqrt2), erf(2/sqrt2))`` maps through
      ``sqrt2 * erfinv(u)`` to N(0, 1) truncated to [-2, 2]. Truncation shrinks
      the variance, so samples scale by ``1 / std_trunc`` and clip at
      ``+-2 / std_trunc``, where ``std_trunc**2 = 1 - 2 * 2 * phi(2) / Z`` with
      ``Z = erf(2 / sqrt2)`` (the symmetric case of the truncated-normal
      variance, ``phi`` the unit normal density).

    """
    lower, upper = -2.0, 2.0
    sqrt2 = 2.0**0.5
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)
    normalizer = (b - a) / 2.0
    scale = (2.0 * math.pi) ** -0.5
    pdf_upper = scale * math.exp(-0.5 * upper * upper)
    pdf_lower = scale * math.exp(-0.5 * lower * lower)
    std = (
        1.0
        / (
            1.0
            - (upper * pdf_upper - lower * pdf_lower) / normalizer
            - ((pdf_upper - pdf_lower) / normalizer) ** 2
        )
        ** 0.5
    )
    with torch.no_grad():
        tensor.uniform_(a, b, generator=generator)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * std)
        tensor.clip_(lower * std, upper * std)
    return tensor


def _hypothesis_recursion(model: LatentRecursion) -> HypothesisRecursion:
    """Narrow the train step's model to the hypothesis-holding TRM."""
    if not isinstance(model, HypothesisRecursion):
        raise TypeError(
            f"winner-take-all routing requires a HypothesisRecursion; got {type(model).__name__}.",
        )
    return model


def _identifiers(net: HypothesisRecursion, pool: AtomicPool) -> Tensor | None:
    """Return the pool's puzzle ids when the model has a puzzle table."""
    return pool.puzzle_ids if net.puzzle_emb is not None else None


def _tile(x: Tensor, count: int) -> Tensor:
    """Repeat ``[B, S, C]`` once per hypothesis as a ``[B * K, S, C]`` batch."""
    batch, length, width = x.shape
    return x.unsqueeze(1).expand(batch, count, length, width).reshape(-1, length, width)
