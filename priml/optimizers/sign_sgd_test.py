"""Tests for reference-style sparse puzzle embedding SignSGD."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

import pytest
import torch
import torch.distributed as dist

from priml.optimizers.sign_sgd import (
    SignSGD,
    _sparse_distributed_step,
    _sparse_embedding_parts,
)


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture
def single_rank_group(tmp_path: Path) -> Generator[None, None, None]:
    """Open a 1-rank gloo group over a file rendezvous for the collective paths."""
    dist.init_process_group(
        backend="gloo",
        init_method=(tmp_path / "gloo-rendezvous").resolve().as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


# The gathers really run over the single rank; only the world-size probe is faked, so
# the arithmetic under test is the real collective result.
def _world_of_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the 1-rank group read as 2 ranks so the aggregated paths are taken."""
    monkeypatch.setattr(dist, "get_world_size", _one_rank)
    monkeypatch.setattr(
        "priml.optimizers.sign_sgd._is_distributed",
        lambda: True,
    )


def _one_rank(group: object = None) -> int:
    del group
    return 1


def _sparse_group(
    weights: Tensor,
    local_weights: Tensor,
    local_ids: Tensor,
    *,
    lr: float,
    weight_decay: float,
) -> dict[str, object]:
    return {
        "params": [weights, local_weights, local_ids],
        "lr": lr,
        "weight_decay": weight_decay,
        "sparse_embedding": True,
    }


def test_sign_sgd_updates_sparse_embedding_buffers() -> None:
    weights = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    local_weights = torch.zeros(4, 4, requires_grad=True)
    local_ids = torch.tensor([2, 2, 4, 1], dtype=torch.int32)
    local_weights.grad = torch.tensor(
        [
            [0.1, -0.2, 0.0, 0.4],
            [0.3, 0.2, 0.0, -0.1],
            [-0.5, 0.0, 0.7, 0.2],
            [0.0, 0.9, -0.8, 0.0],
        ],
    )
    expected = weights.clone()
    opt = SignSGD(
        [_sparse_group(weights, local_weights, local_ids, lr=0.1, weight_decay=0.5)],
    )

    opt.step()

    grad_ids, inv = local_ids.unique(return_inverse=True)
    grad = torch.zeros(grad_ids.shape[0], 4)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, 4), local_weights.grad)
    rows = expected[grad_ids.to(torch.long)]
    rows.mul_(1.0 - 0.1 * 0.5).add_(torch.sign(grad), alpha=-0.1)
    expected[grad_ids.to(torch.long)] = rows
    torch.testing.assert_close(weights, expected)


def test_sign_sgd_keeps_untouched_sparse_rows_unchanged() -> None:
    weights = torch.ones(4, 3)
    local_weights = torch.zeros(2, 3, requires_grad=True)
    local_ids = torch.tensor([1, 3], dtype=torch.int32)
    local_weights.grad = torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.5, 0.0]])
    opt = SignSGD(
        [_sparse_group(weights, local_weights, local_ids, lr=0.1, weight_decay=0.5)],
    )

    opt.step()

    torch.testing.assert_close(weights[0], torch.ones(3))
    torch.testing.assert_close(weights[2], torch.ones(3))


def test_sign_sgd_steps_externally_accumulated_sparse_rows() -> None:
    weights = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    local_weights = torch.zeros(2, 4, requires_grad=True)
    local_ids = torch.zeros(2, dtype=torch.int32)
    grad_rows = torch.tensor(
        [
            [0.1, -0.2, 0.0, 0.4],
            [0.3, 0.2, 0.0, -0.1],
            [-0.5, 0.0, 0.7, 0.2],
        ],
    )
    grad_ids = torch.tensor([2, 2, 4], dtype=torch.int32)
    expected = weights.clone()
    opt = SignSGD(
        [weights, local_weights, local_ids],
        lr=0.1,
        weight_decay=0.5,
    )

    opt.step_sparse_embedding(grad_rows, grad_ids)

    unique_ids, inv = grad_ids.unique(return_inverse=True)
    grad = torch.zeros(unique_ids.shape[0], 4)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, 4), grad_rows)
    rows = expected[unique_ids.to(torch.long)]
    rows.mul_(1.0 - 0.1 * 0.5).add_(torch.sign(grad), alpha=-0.1)
    expected[unique_ids.to(torch.long)] = rows
    torch.testing.assert_close(weights, expected)


def test_aggregate_distributed_defaults_true_and_is_settable() -> None:
    """The cross-rank sparse-gather flag defaults True (DP) and can be disabled.

    Task-parallel use (per-task TTT) sets it False so the per-step sparse
    ``all_gather_into_tensor`` is skipped -- a per-step collective desyncs when
    ranks step an independent number of times and the NCCL watchdog aborts. With
    it False the sparse step is purely local; the actual update is identical to
    the single-process case (no distributed init here), which this exercises.
    """
    assert SignSGD([torch.zeros(2, 2)]).aggregate_distributed is True

    weights = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
    local_weights = torch.zeros(1, 4, requires_grad=True)
    local_ids = torch.zeros(1, dtype=torch.int32)
    grad_rows = torch.tensor([[0.1, -0.2, 0.0, 0.4]])
    grad_ids = torch.tensor([1], dtype=torch.int32)
    opt = SignSGD(
        [weights, local_weights, local_ids],
        lr=0.1,
        weight_decay=0.0,
        aggregate_distributed=False,
    )
    assert opt.aggregate_distributed is False
    before = weights[1].clone()

    opt.step_sparse_embedding(grad_rows, grad_ids)

    # Local SignSGD applied to row 1 only: p[1] -= lr * sign(grad).
    expected = before - 0.1 * torch.sign(grad_rows[0])
    torch.testing.assert_close(weights[1], expected)
    torch.testing.assert_close(weights[0], torch.arange(4, dtype=torch.float32) / 10)


def test_sign_sgd_preserves_grad_after_step() -> None:
    """SignSGD must not mutate ``p.grad`` in place.

    Downstream consumers (gradient logging, gradient clipping, a second
    optimizer on shared params) read ``.grad`` after ``opt.step()``.
    Replacing the gradient with its sign in-place silently corrupts those
    reads. The dense param path historically did ``grad.sign_()``; this
    test pins the contract that ``p.grad`` survives ``step()`` intact.
    """
    p = torch.randn(4, 3, requires_grad=True)
    original_grad = torch.randn(4, 3)
    p.grad = original_grad.clone()
    opt = SignSGD([p], lr=0.1, weight_decay=0.0)

    opt.step()

    torch.testing.assert_close(p.grad, original_grad)


def test_sparse_embedding_routing_requires_explicit_flag() -> None:
    """A 3-param group of plain dense tensors must NOT be sparse-routed.

    Currently ``_sparse_embedding_parts`` matches any group with exactly
    3 params, one requires_grad, one 1-D, one 2-D -- collidable by any
    user passing ``[2D_weight, 1D_bias, 2D_other]`` or similar. The fix
    is an explicit ``{"sparse_embedding": True}`` flag; without it, the
    group is treated as a dense group.
    """
    matrix_a = torch.randn(4, 3, requires_grad=True)  # 2D with grad.
    matrix_a.grad = torch.randn(4, 3)
    bias = torch.randn(3)  # 1D, no grad.
    matrix_b = torch.randn(4, 3)  # 2D, no grad.

    pre = matrix_a.clone()
    opt = SignSGD(
        [{"params": [matrix_a, bias, matrix_b], "lr": 0.1, "weight_decay": 0.0}],
    )
    opt.step()

    # matrix_a was the only grad-bearing param; it must have moved by
    # the dense sign update path, not been silently dropped by the
    # sparse path that ignores ``matrix_a`` entirely.
    diff = (matrix_a - pre).abs().max().item()
    assert diff > 0.0, (
        "matrix_a unchanged: sparse-embedding heuristic mis-routed a dense group"
    )


def test_sparse_embedding_routing_honors_explicit_flag() -> None:
    """A 3-param group flagged ``sparse_embedding=True`` IS routed to sparse path."""
    weights = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    local_weights = torch.zeros(2, 4, requires_grad=True)
    local_ids = torch.tensor([1, 3], dtype=torch.int32)
    local_weights.grad = torch.tensor([[1.0, 0.0, -1.0, 0.0], [0.0, 0.5, 0.0, -0.3]])
    expected = weights.clone()
    opt = SignSGD(
        [
            {
                "params": [weights, local_weights, local_ids],
                "lr": 0.1,
                "weight_decay": 0.5,
                "sparse_embedding": True,
            },
        ],
    )

    opt.step()

    # Sparse path: only the rows referenced by local_ids change.
    grad_ids, inv = local_ids.unique(return_inverse=True)
    grad = torch.zeros(grad_ids.shape[0], 4)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, 4), local_weights.grad)
    rows = expected[grad_ids.to(torch.long)]
    rows.mul_(1.0 - 0.1 * 0.5).add_(torch.sign(grad), alpha=-0.1)
    expected[grad_ids.to(torch.long)] = rows
    torch.testing.assert_close(weights, expected)


def test_dense_weight_decay_touches_only_rows_with_gradient() -> None:
    p = torch.nn.Parameter(torch.ones(3, 2))
    p.grad = torch.tensor([[1.0, -1.0], [0.0, 0.0], [0.0, 2.0]])
    SignSGD([p], lr=0.1, weight_decay=0.5).step()

    decayed = 1.0 - 0.1 * 0.5
    torch.testing.assert_close(p[0], torch.tensor([decayed - 0.1, decayed + 0.1]))
    torch.testing.assert_close(p[1], torch.ones(2))
    torch.testing.assert_close(p[2], torch.tensor([decayed, decayed - 0.1]))


def test_dense_weight_decay_on_a_vector_is_plain_decoupled_decay() -> None:
    p = torch.nn.Parameter(torch.ones(3))
    p.grad = torch.tensor([1.0, 0.0, -1.0])
    SignSGD([p], lr=0.1, weight_decay=0.5).step()

    decayed = 1.0 - 0.1 * 0.5
    torch.testing.assert_close(p, torch.tensor([decayed - 0.1, decayed, decayed + 0.1]))


def test_params_without_gradient_are_skipped() -> None:
    p = torch.nn.Parameter(torch.ones(2))
    SignSGD([p], lr=0.1, weight_decay=0.5).step()
    torch.testing.assert_close(p, torch.ones(2))


def test_step_evaluates_and_returns_the_closure() -> None:
    p = torch.nn.Parameter(torch.zeros(2))
    calls: list[int] = []

    def closure() -> float:
        calls.append(1)
        return 7.5

    assert SignSGD([p], lr=0.1).step(closure) == 7.5
    assert calls == [1]


def test_negative_hyperparameters_are_rejected() -> None:
    p = torch.zeros(2)
    with pytest.raises(ValueError, match="Invalid learning rate"):
        SignSGD([p], lr=-1.0)
    with pytest.raises(ValueError, match="Invalid weight_decay"):
        SignSGD([p], weight_decay=-1.0)


def test_config_builds_a_constructor_awaiting_parameters() -> None:
    build = SignSGD.Config(
        lr=0.5,
        weight_decay=0.25,
        aggregate_distributed=False,
    ).make()
    optimizer = build([torch.zeros(2)])
    assert optimizer.param_groups[0]["lr"] == 0.5
    assert optimizer.param_groups[0]["weight_decay"] == 0.25
    assert optimizer.aggregate_distributed is False


def test_finalized_config_is_not_finalized_twice() -> None:
    optimizer = SignSGD.Config(lr=0.5).finalize().make()([torch.zeros(2)])
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_sparse_embedding_flag_rejects_a_group_without_the_buffers() -> None:
    weights = torch.zeros(4, 3)
    local_weights = torch.zeros(2, 3, requires_grad=True)
    opt = SignSGD(
        [{"params": [weights, local_weights], "sparse_embedding": True}],
        lr=0.1,
    )
    with pytest.raises(ValueError, match="sparse_embedding=True requires"):
        opt.step()


def test_sparse_parts_need_both_ids_and_weights() -> None:
    grad_bearing = torch.zeros(2, 3, requires_grad=True)
    assert _sparse_embedding_parts([torch.zeros(3), torch.zeros(3)]) is None
    assert (
        _sparse_embedding_parts([grad_bearing, torch.zeros(3), torch.zeros(3)]) is None
    )
    assert (
        _sparse_embedding_parts([grad_bearing, torch.zeros(2, 3), torch.zeros(2, 3)])
        is None
    )


def test_step_sparse_embedding_ignores_a_plain_dense_group() -> None:
    p = torch.nn.Parameter(torch.ones(2, 3))
    opt = SignSGD([p], lr=0.1)
    opt.step_sparse_embedding(torch.ones(1, 3), torch.zeros(1, dtype=torch.int32))
    torch.testing.assert_close(p, torch.ones(2, 3))


def test_sparse_embedding_group_without_gradient_is_left_alone() -> None:
    weights = torch.ones(4, 3)
    local_weights = torch.zeros(2, 3, requires_grad=True)
    local_ids = torch.tensor([1, 3], dtype=torch.int32)
    opt = SignSGD(
        [_sparse_group(weights, local_weights, local_ids, lr=0.1, weight_decay=0.5)],
    )
    opt.step()
    torch.testing.assert_close(weights, torch.ones(4, 3))


def test_aggregated_sparse_step_matches_the_local_rule(
    single_rank_group: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gathering rows over the group leaves a one-rank world's update unchanged."""
    del single_rank_group
    _world_of_two(monkeypatch)
    weights = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
    local_weights = torch.zeros(3, 4, requires_grad=True)
    local_ids = torch.tensor([2, 2, 4], dtype=torch.int32)
    local_weights.grad = torch.tensor(
        [
            [0.1, -0.2, 0.0, 0.4],
            [0.3, 0.2, 0.0, -0.1],
            [-0.5, 0.0, 0.7, 0.2],
        ],
    )
    expected = weights.clone()
    opt = SignSGD(
        [_sparse_group(weights, local_weights, local_ids, lr=0.1, weight_decay=0.5)],
    )

    opt.step()

    grad_ids, inv = local_ids.unique(return_inverse=True)
    grad = torch.zeros(grad_ids.shape[0], 4)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, 4), local_weights.grad)
    rows = expected[grad_ids.to(torch.long)]
    rows.mul_(1.0 - 0.1 * 0.5).add_(torch.sign(grad), alpha=-0.1)
    expected[grad_ids.to(torch.long)] = rows
    torch.testing.assert_close(weights, expected)


def test_distributed_dense_step_decays_and_signs_touched_rows_only(
    single_rank_group: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cross-rank dense path reproduces the local touched-row rule."""
    del single_rank_group
    _world_of_two(monkeypatch)
    p = torch.nn.Parameter(torch.ones(3, 2))
    p.grad = torch.tensor([[1.0, -1.0], [0.0, 0.0], [0.0, 2.0]])
    SignSGD([p], lr=0.1, weight_decay=0.5).step()

    decayed = 1.0 - 0.1 * 0.5
    torch.testing.assert_close(p[0], torch.tensor([decayed - 0.1, decayed + 0.1]))
    torch.testing.assert_close(p[1], torch.ones(2))
    torch.testing.assert_close(p[2], torch.tensor([decayed, decayed - 0.1]))


def test_distributed_dense_step_without_decay_only_signs(
    single_rank_group: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del single_rank_group
    _world_of_two(monkeypatch)
    p = torch.nn.Parameter(torch.ones(2, 2))
    p.grad = torch.tensor([[1.0, -1.0], [0.0, 0.0]])
    SignSGD([p], lr=0.1, weight_decay=0.0).step()
    torch.testing.assert_close(p, torch.tensor([[0.9, 1.1], [1.0, 1.0]]))


def test_distributed_dense_step_with_no_touched_rows_is_a_noop(
    single_rank_group: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del single_rank_group
    _world_of_two(monkeypatch)
    p = torch.nn.Parameter(torch.ones(2, 2))
    p.grad = torch.zeros(2, 2)
    SignSGD([p], lr=0.1, weight_decay=0.5).step()
    torch.testing.assert_close(p, torch.ones(2, 2))


def test_sparse_distributed_step_rejects_noncontiguous_params() -> None:
    """A non-contiguous master weights tensor must raise, not silently no-op.

    ``_sparse_distributed_step`` does ``p.reshape(...)`` then writes into
    the result; for non-contiguous ``p`` that's a copy that's silently
    discarded. The guarded behaviour is to raise so the caller fixes the
    layout.
    """
    weights = torch.randn(4, 3, 2).transpose(0, 1)  # non-contiguous.
    assert not weights.is_contiguous()
    local_weights = torch.zeros(2, 6, requires_grad=True)
    local_weights.grad = torch.randn(2, 6)

    with pytest.raises(
        (AssertionError, RuntimeError),
        match="contig",
    ):
        _sparse_distributed_step(
            weights,
            local_weights.grad,
            lr=0.1,
            weight_decay=0.0,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
