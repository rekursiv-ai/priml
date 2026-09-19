"""Tests for AdversarialLoss."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

import pytest
import torch

from priml.cost import MEASURES, Cost, Kernel, Phase, cost
from priml.loss.gan import AdversarialLoss
from priml.loss.weighted_loss import WeightedSum
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Mapping


def _fp32(
    *,
    primal: Mapping[str, Mapping[Kernel, int]] | None = None,
    adjoint: Mapping[str, Mapping[Kernel, int]] | None = None,
    **fields: int,
) -> Cost:
    """Build a ``Cost`` from whole-invocation fp32 FLOPs and bytes."""
    cells: dict[tuple[object, ...], int] = {}
    phases: tuple[tuple[Phase, Mapping[str, Mapping[Kernel, int]] | None], ...] = (
        ("primal", primal),
        ("adjoint", adjoint),
    )
    for phase, silos in phases:
        for measure in MEASURES:
            for kernel, value in (silos or {}).get(measure, {}).items():
                cells[(measure, phase, kernel, torch.float32)] = value
    return Cost(cells=cells, **fields)


def test_adversarial_loss_is_pointwise_over_batch() -> None:
    loss = AdversarialLoss.Config(adversarial_weight=1.0, content_weight=2.0).make()
    fake_media = torch.zeros(2, 3, 4)
    result = loss(
        fake_media,
        fake_logits=torch.zeros(2, 1),
        fake_media=fake_media,
        real_media=torch.ones(2, 3, 4),
    )
    # BCE of a zero logit against one is log 2; L1 against ones is 1.
    torch.testing.assert_close(
        result["loss"],
        torch.full((2,), torch.tensor(2.0).log().item() + 2.0),
    )


def test_adversarial_loss_cost_spreads_per_sample_work_over_media() -> None:
    """One logit's BCE and the two weights are per sample; L1 and its mean are per element."""
    config = AdversarialLoss.Config()
    real_media = torch.randn(2, 3, 4)
    measured = assert_cost_matches_torch(
        WeightedSum.Config(fns=[config], weights=[1.0]),
        build_input=lambda: (
            torch.randn(2, 1, requires_grad=True),
            torch.randn(2, 3, 4, requires_grad=True),
        ),
        seq_len=12,
        batch_size=2,
        dtype=None,
        run=lambda module, inputs: _loss(
            module,
            inputs[1],
            fake_logits=inputs[0],
            fake_media=inputs[1],
            real_media=real_media,
        ),
    )
    expected = _fp32(
        primal={
            "flops": {"elementwise": 70, "reduction": 22},
            "bytes": {"elementwise": 744, "reduction": 120},
        },
        adjoint={
            "flops": {"elementwise": 86},
            "bytes": {"elementwise": 800, "reduction": 120},
        },
    )
    assert cost(config, seq_len=12, batch_size=2, dtype=None) == expected
    assert measured == expected + _fp32(
        primal={
            "flops": {"elementwise": 48},
            "bytes": {"elementwise": 384, "reduction": 192},
        },
        adjoint={
            "flops": {"elementwise": 48},
            "bytes": {"elementwise": 192, "reduction": 192},
        },
    )
    assert measured.params == 0
    assert measured["flops", "matmul"].sum() == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_adversarial_loss_operand_traffic(dtype: torch.dtype) -> None:
    costed = cost(AdversarialLoss.Config(), seq_len=12, batch_size=2, dtype=dtype)
    itemsize = dtype.itemsize
    assert costed["bytes", "primal", "elementwise"].sum() == 186 * itemsize
    assert costed["bytes", "primal", "reduction"].sum() == 30 * itemsize
    assert costed["bytes", "adjoint", "elementwise"].sum() == 200 * itemsize
    assert costed["bytes", "adjoint", "reduction"].sum() == 30 * itemsize


def _loss(module: nn.Module, model_output: Tensor, **batch: Tensor) -> Tensor:
    """Run the wrapped adversarial loss and return its ``loss`` tensor."""
    assert isinstance(module, WeightedSum)
    return module(model_output, **batch)["loss"]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
