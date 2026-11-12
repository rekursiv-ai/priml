"""Sparse programming writes exact addresses in order, all or nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import PartialConfig
from torch import nn

import pytest
import torch

from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.parameter_filter import complement, matching
from priml.optimizers.sparse_programming import SparseProgramming, TensorWrite
from priml.optimizers.weight_programming import UpdateOutput


if TYPE_CHECKING:
    from torch import Tensor


def _programmed(
    writes: list[TensorWrite],
) -> tuple[nn.Module, CompositeOptimizer, dict[str, Tensor]]:
    """Route ``0.*`` to the writes and ``1.*`` to SGD; snapshot the weights."""
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4))
    program = SparseProgramming.Config()
    program.writes = writes
    config = CompositeOptimizer.Config()
    config.optimizers = [program, PartialConfig(torch.optim.SGD, lr=0.1)]
    config.select = [matching("0."), complement(matching("0."))]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    return model, config.make()(model), before


def _step(model: nn.Module, optimizer: CompositeOptimizer) -> None:
    with optimizer.programming(model, next_batch=_no_input):
        optimizer.step()


def test_writes_apply_in_order_and_report_changed_scalars() -> None:
    model, optimizer, before = _programmed(
        [
            TensorWrite(parameter="0.weight", value=1.0),
            TensorWrite(parameter="0.weight", indices=(1,), value=2.0),
            TensorWrite(parameter="0.weight", indices=(1, 0), value=3.0),
            TensorWrite(parameter="0.bias", indices=(2,), value=-4.0),
        ],
    )

    _step(model, optimizer)

    state = model.state_dict()
    assert torch.equal(
        state["0.weight"],
        torch.tensor([[1.0, 1.0], [3.0, 2.0], [1.0, 1.0]]),
    )
    assert torch.equal(state["0.bias"][:2], before["0.bias"][:2])
    assert state["0.bias"][2] == -4.0
    assert all(
        torch.equal(state[name], before[name]) for name in ("1.weight", "1.bias")
    )
    assert optimizer.programming_output.updated
    assert optimizer.programming_output.metrics == {"programming/0/changed_scalars": 7}


def test_rewriting_the_same_values_changes_nothing() -> None:
    model, optimizer, _ = _programmed([TensorWrite(parameter="0.bias", value=0.5)])
    _step(model, optimizer)

    _step(model, optimizer)

    program = optimizer.optimizers[0]
    assert isinstance(program, SparseProgramming)
    assert program.output == UpdateOutput(metrics={"changed_scalars": 0})


@pytest.mark.parametrize(
    ("write", "error", "pattern"),
    [
        (
            TensorWrite(parameter="1.weight", value=5.0),
            ValueError,
            r"^Parameter '1\.weight' belongs to another optimizer\.$",
        ),
        (TensorWrite(parameter="2.weight", value=5.0), KeyError, r"^'2\.weight'$"),
        (
            TensorWrite(parameter="0.bias", indices=(3,), value=5.0),
            IndexError,
            r"^index 3 is out of bounds",
        ),
    ],
)
def test_a_bad_address_rejects_every_write(
    write: TensorWrite,
    error: type[Exception],
    pattern: str,
) -> None:
    model, optimizer, before = _programmed(
        [TensorWrite(parameter="0.weight", value=5.0), write],
    )

    with pytest.raises(error, match=pattern):
        _step(model, optimizer)

    state = model.state_dict()
    assert all(torch.equal(state[name], value) for name, value in before.items())


def _no_input() -> dict[str, object]:
    raise AssertionError("Fixed writes read no calibration input.")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
