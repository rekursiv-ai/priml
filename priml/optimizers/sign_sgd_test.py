"""Tests for reference-style sparse puzzle embedding SignSGD."""

from __future__ import annotations

import pytest
import torch

from priml.optimizers.sign_sgd import SignSGD, _sparse_distributed_step


def _sparse_group(
    weights: torch.Tensor,
    local_weights: torch.Tensor,
    local_ids: torch.Tensor,
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
    3 params, one requires_grad, one 1-D, one 2-D — collidable by any
    user passing ``[2D_weight, 1D_bias, 2D_other]`` or similar. The fix
    is an explicit ``{"sparse_embedding": True}`` flag; without it, the
    group is treated as a dense group.
    """
    matrix_a = torch.randn(4, 3, requires_grad=True)  # 2D with grad
    matrix_a.grad = torch.randn(4, 3)
    bias = torch.randn(3)  # 1D, no grad
    matrix_b = torch.randn(4, 3)  # 2D, no grad

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


def test_sparse_distributed_step_rejects_noncontiguous_params() -> None:
    """A non-contiguous master weights tensor must raise, not silently no-op.

    ``_sparse_distributed_step`` does ``p.reshape(...)`` then writes into
    the result; for non-contiguous ``p`` that's a copy that's silently
    discarded. The guarded behaviour is to raise so the caller fixes the
    layout.
    """
    weights = torch.randn(4, 3, 2).transpose(0, 1)  # non-contiguous
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
