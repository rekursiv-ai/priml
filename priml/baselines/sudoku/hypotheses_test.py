"""Tests for winner-take-all training over several fast-latent hypotheses.

The staircase golden replays a trained hypothesis generator through the sieve;
these tests pin each piece's contract on a tiny model.
"""

from __future__ import annotations

from typing import Final

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.sudoku.act import AtomicPool
from priml.baselines.sudoku.hypotheses import (
    HypothesisRecursion,
    WinnerRecompute,
    WinnerTakeAll,
    equalize_alignment,
    hypothesis_losses,
    truncated_normal_,
    winner_metrics,
)
from priml.baselines.sudoku.latent_recursion import LatentRecursion, recipe_block
from priml.baselines.sudoku.losses import token_cross_entropy
from priml.model.special import Identity
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


ROWS: Final = 3
"""Puzzles in a pool."""

CELLS: Final = 16
"""Grid cells: a 4x4 board."""

VOCAB: Final = 6
"""Pad, blank, and four digits."""


def tiny_config() -> HypothesisRecursion.Config:
    """Return a width-12 HypothesisRecursion over a 4x4 board.

    Width 12 with an FFN of hidden width 4: alignment projects on the FFN's
    top-10 subspace, which is the whole space at width 10 or less.
    """
    cfg = HypothesisRecursion.Config()
    cfg.vocab_size = VOCAB
    cfg.puzzle_grid_shape = (CELLS,)
    cfg.pos2d_grid_shape = (4, 4)
    cfg.pos2d_box_shape = (2, 2)
    cfg.channels_in = 12
    cfg.num_heads = 2
    cfg.num_layers = 1
    cfg.slow_cycles = 2
    cfg.fast_cycles = 1
    cfg.num_puzzle_identifiers = 0
    cfg.puzzle_emb_len = 2
    cfg.compile = False
    cfg.dtype = None
    cfg.block = recipe_block()
    assert isinstance(cfg.block.ffn, SwiGLU.Config)
    cfg.block.ffn.round_to = 4
    cfg.block.ffn.channels_hidden = 4
    return cfg


def seated(model: HypothesisRecursion) -> AtomicPool:
    """Return a pool with every slot freshly seated from ``model``."""
    config = AtomicPool.Config()
    config.batch_size = ROWS
    config.max_steps = 2
    config.halting = None
    config.grid_len = CELLS
    config.seq_len = model.config.total_seq_len
    config.channels_hidden = model.config.channels_in
    config.dtype = torch.float32
    pool = config.make()
    generator = torch.Generator().manual_seed(3)
    pool.refill(
        model.init_z,
        media=torch.randint(1, VOCAB, (ROWS, CELLS), generator=generator),
        labels=torch.randint(2, VOCAB, (ROWS, CELLS), generator=generator),
        valid_count=ROWS,
        puzzle_ids=None,
        ignore_label_id=-100,
    )
    return pool


def test_truncated_normal_is_seeded_bounded_and_unit_variance() -> None:
    before = rng_fingerprint()
    first = truncated_normal_(
        torch.empty(4_096),
        generator=torch.Generator().manual_seed(5),
    )
    again = truncated_normal_(
        torch.empty(4_096),
        generator=torch.Generator().manual_seed(5),
    )
    other = truncated_normal_(
        torch.empty(4_096),
        generator=torch.Generator().manual_seed(6),
    )
    assert torch.equal(rng_fingerprint(), before)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    # Rescaled to unit variance, the [-2, 2] truncation bound grows by the
    # same factor, so the largest magnitude exceeds 2 but stays below 2.5.
    assert 2.0 < float(first.abs().max()) < 2.5
    assert abs(float(first.var()) - 1.0) < 0.1


def test_equalize_alignment_matches_every_row_to_row_zero() -> None:
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(3, 12, generator=generator, dtype=torch.float64)
    # Four distinct coordinate axes: an orthonormal basis of a 4-D subspace.
    basis = torch.eye(12, dtype=torch.float64)[:, [1, 4, 7, 10]]
    norms = rows.norm(dim=-1)
    equalize_alignment(rows, basis)
    fraction = (rows @ basis).norm(dim=-1) / rows.norm(dim=-1)
    assert torch.allclose(rows.norm(dim=-1), norms)
    assert torch.allclose(fraction, fraction[:1].expand(3))


def test_hypothesis_recursion_keeps_the_deployed_row_and_the_ambient_stream() -> None:
    torch.manual_seed(0)
    plain = LatentRecursion.Config().update(tiny_config(), skip_missing=True).make()
    plain_rng = rng_fingerprint()
    torch.manual_seed(0)
    model = tiny_config().make()
    assert torch.equal(rng_fingerprint(), plain_rng)
    assert model.fast_init.shape == (2, 12)
    assert torch.equal(model.fast_init[0], plain.fast_init[0])
    plain_state = plain.state_dict()
    for name, value in model.state_dict().items():
        if name != "fast_init":
            assert torch.equal(value, plain_state[name]), name
    other = tiny_config()
    other.hypothesis_seed = 1
    torch.manual_seed(0)
    assert not torch.equal(other.make().fast_init[1], model.fast_init[1])
    single = tiny_config()
    single.num_hypotheses = 1
    with pytest.raises(ValueError, match="num_hypotheses"):
        single.make()


def test_hypothesis_recursion_requires_a_swiglu_ffn_to_align_on() -> None:
    config = tiny_config()
    assert config.block is not None
    config.block.ffn = Identity.Config()
    with pytest.raises(TypeError, match="TransformerBlock with a SwiGLU FFN"):
        config.make()


def test_fresh_fast_states_are_seeded_per_draw() -> None:
    torch.manual_seed(0)
    model = tiny_config().make()
    before = rng_fingerprint()
    states = model.fresh_fast_states(ROWS, seed=4)
    assert torch.equal(rng_fingerprint(), before)
    assert states.shape == (ROWS, 2, model.config.total_seq_len, 12)
    assert torch.equal(states, model.fresh_fast_states(ROWS, seed=4))
    assert not torch.equal(states, model.fresh_fast_states(ROWS, seed=5))


def test_hypothesis_losses_average_only_scored_cells() -> None:
    labels = torch.tensor([[2, 3, -100], [4, -100, -100]])
    logits = torch.randn(2, 4, 3, VOCAB, generator=torch.Generator().manual_seed(1))
    losses = hypothesis_losses(
        logits,
        labels,
        lambda x, y: token_cross_entropy(x, y, label_smoothing=0.0, ignore_index=-100),
        ignore_index=-100,
    )
    expected = torch.stack(
        [
            torch.stack(
                [
                    nn.functional.cross_entropy(
                        logits[row, head][labels[row] >= 0],
                        labels[row][labels[row] >= 0],
                    )
                    for head in range(4)
                ],
            )
            for row in range(2)
        ],
    )
    assert losses.shape == (2, 4)
    assert torch.allclose(losses, expected)


def test_winner_metrics_count_only_active_slots() -> None:
    losses = torch.tensor([[1.0, 3.0], [4.0, 2.5], [9.0, 0.0]])
    winner = losses.argmin(dim=-1)
    active = torch.tensor([True, True, False])
    metrics = winner_metrics(losses, winner, active=active)
    assert float(metrics["hypothesis_max_win_fraction"]) == 0.5
    assert float(metrics["hypothesis_effective_heads"]) == pytest.approx(2.0)
    assert float(metrics["hypothesis_loss_gap"]) == pytest.approx((2.0 + 1.5) / 2)


def _route(
    routing: WinnerTakeAll,
    model: HypothesisRecursion,
    pool: AtomicPool,
) -> Tensor:
    """Seat, route, and return the trained logits."""
    routing.seat(model, pool, global_step=7)
    routed = routing(
        model,
        pool,
        token_loss=lambda x, y: token_cross_entropy(
            x,
            y,
            label_smoothing=0.0,
            ignore_index=-100,
        ),
        ignore_index=-100,
    )
    return routed.logits


def test_winner_take_all_trains_the_lowest_loss_hypothesis() -> None:
    torch.manual_seed(0)
    model = tiny_config().make()
    pool = seated(model)
    routing = WinnerTakeAll(WinnerTakeAll.Config())
    routing.seat(model, pool, global_step=7)
    starts = routing._carried().clone()
    assert torch.equal(starts, model.fresh_fast_states(ROWS, seed=7 * 1_024))
    logits = _route(WinnerTakeAll(WinnerTakeAll.Config()), model, pool)
    every = torch.stack(
        [
            model(pool.inputs, pool.z_slow, starts[:, head])["logits"]
            for head in range(2)
        ],
        dim=1,
    )
    losses = hypothesis_losses(
        every,
        pool.labels,
        lambda x, y: token_cross_entropy(x, y, label_smoothing=0.0, ignore_index=-100),
        ignore_index=-100,
    )
    winner = losses.argmin(dim=-1)
    assert torch.equal(logits, every[torch.arange(ROWS), winner])
    logits.sum().backward()
    assert model.head.weight.grad is not None


def test_a_reseat_redraws_only_the_freshly_seated_slots() -> None:
    """Slots mid-rollout keep their carried hypotheses; none fresh redraws none."""
    torch.manual_seed(0)
    model = tiny_config().make()
    pool = seated(model)
    routing = WinnerTakeAll(WinnerTakeAll.Config())
    routing.seat(model, pool, global_step=7)
    carried = routing._carried().clone()
    pool.steps += 1
    routing.seat(model, pool, global_step=8)
    assert torch.equal(routing._carried(), carried)
    pool.steps[1] = 0
    routing.seat(model, pool, global_step=9)
    redrawn = model.fresh_fast_states(1, seed=9 * 1_024)
    expected = carried.clone()
    expected[1] = redrawn[0]
    assert torch.equal(routing._carried(), expected)


def test_winner_recompute_trains_the_same_winner() -> None:
    torch.manual_seed(0)
    model = tiny_config().make()
    with host_agnostic_numerics():
        full = _route(WinnerTakeAll(WinnerTakeAll.Config()), model, seated(model))
        recomputed = _route(
            WinnerRecompute(WinnerRecompute.Config()),
            model,
            seated(model),
        )
    assert torch.equal(full.float(), recomputed)


def test_routing_refuses_a_plain_recursion() -> None:
    torch.manual_seed(0)
    plain = LatentRecursion.Config().update(tiny_config(), skip_missing=True).make()
    with pytest.raises(TypeError, match="HypothesisRecursion"):
        WinnerTakeAll(WinnerTakeAll.Config()).seat(
            plain,
            seated(tiny_config().make()),
            global_step=0,
        )
    with pytest.raises(RuntimeError, match="seat"):
        WinnerTakeAll(WinnerTakeAll.Config())._carried()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
