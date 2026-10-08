"""Check the grouped cross-entropy against brute-force restricted softmaxes."""

import math

import pytest
import torch

from priml.baselines.craftax.world_model.loss import (
    Nll,
    cell_nll,
    cross_entropy,
    modality_loss,
    scalar_nll,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.lib.codec import from_plain


def _restricted_nll(logits: torch.Tensor, allowed: list[int], target: int) -> float:
    """Return ``-log softmax(logits[allowed])[target]`` computed the slow way."""
    kept = logits[allowed]
    return float(kept.logsumexp(-1) - logits[target])


def test_cross_entropy_matches_torch_and_reports_log_normalizer() -> None:
    logits = torch.randn(3, 5, 7)
    target = torch.randint(0, 7, (3, 5))
    out = cross_entropy(logits, target)
    expected = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target.flatten(),
        reduction="none",
    ).view(3, 5)
    torch.testing.assert_close(out.nll, expected)
    torch.testing.assert_close(out.logz_sq, logits.logsumexp(-1).square())


def test_cell_nll_sums_eight_restricted_field_softmaxes() -> None:
    schema = craftax_schema()
    table = schema.cell_index_table()
    logits = torch.randn(2, 3, schema.vocab_size)
    values = torch.stack(
        [torch.randint(0, field.valid, (2, 3)) for field in schema.cell_fields],
        dim=-1,
    )
    out = cell_nll(logits, index_table=table, target=values)
    assert out.nll.shape == (2, 3)
    for j in range(2):
        for c in range(3):
            expected = sum(
                _restricted_nll(
                    logits[j, c],
                    list(range(field.offset, field.offset + field.valid)),
                    field.offset + int(values[j, c, k]),
                )
                for k, field in enumerate(schema.cell_fields)
            )
            assert math.isclose(float(out.nll[j, c]), expected, rel_tol=1e-5)


def test_cell_nll_ignores_reserved_and_foreign_logits() -> None:
    schema = craftax_schema()
    table = schema.cell_index_table()
    logits = torch.randn(4, schema.vocab_size)
    values = torch.zeros(4, len(schema.cell_fields), dtype=torch.long)
    base = cell_nll(logits, index_table=table, target=values)
    reserved = torch.zeros(schema.vocab_size, dtype=torch.bool)
    reserved[37:64] = True
    reserved[154:] = True
    shifted = logits + 50.0 * reserved
    moved = cell_nll(shifted, index_table=table, target=values)
    torch.testing.assert_close(moved.nll, base.nll)
    torch.testing.assert_close(moved.logz_sq, base.logz_sq)


def test_scalar_nll_restricts_each_slot_to_its_allowed_ids() -> None:
    schema = craftax_schema()
    allowed = schema.local_allowed()[[0, 1, 2 + 99 + 48]]
    logits = torch.randn(2, 3, schema.vocab_size, requires_grad=True)
    target = torch.tensor([[156, 460, 157], [154, 459, 163]])
    out = scalar_nll(logits, allowed=allowed, target=target)
    nll = out.nll.detach()
    for j in range(2):
        for s in range(3):
            ids = from_plain(allowed[s].nonzero().flatten().tolist(), list[int])
            expected = _restricted_nll(logits[j, s].detach(), ids, int(target[j, s]))
            assert math.isclose(float(nll[j, s]), expected, rel_tol=1e-5)
    out.nll.sum().backward()
    assert logits.grad is not None
    assert bool(logits.grad.isfinite().all())
    assert float(logits.grad[:, 1, :459].abs().max()) == 0.0


def test_modality_loss_sums_modality_means_and_z_loss() -> None:
    action = Nll(
        nll=torch.tensor([1.0, 3.0, 100.0]),
        logz_sq=torch.tensor([4.0, 4.0, 9.0]),
    )
    # Two decisions of two cells each; ``logz_sq`` takes ``nll``'s shape.
    board = Nll(nll=torch.tensor([[2.0, 2.0], [7.0, 7.0]]), logz_sq=torch.ones(2, 2))
    loss = modality_loss(
        {
            "action": (action, torch.tensor([True, True, False])),
            "board": (board, torch.tensor([[True, True], [False, False]])),
        },
        z_loss=0.5,
    )
    assert float(loss.nll["action"]) == 4.0
    assert float(loss.count["action"]) == 2.0
    assert float(loss.count["board"]) == 2.0
    assert float(loss.objective) == 2.0 + 2.0
    assert float(loss.z_loss) == pytest.approx(0.5 * (4.0 + 1.0))
    assert float(loss.loss) == pytest.approx(4.0 + 2.5)


def test_modality_loss_with_nothing_scored_is_zero_not_nan() -> None:
    empty = Nll(nll=torch.ones(3), logz_sq=torch.ones(3))
    loss = modality_loss(
        {"reward": (empty, torch.zeros(3, dtype=torch.bool))},
        z_loss=1e-5,
    )
    assert float(loss.objective) == 0.0
    assert float(loss.count["reward"]) == 0.0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
