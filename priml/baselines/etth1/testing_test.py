"""Tests for the ETTh1 test records."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

import pytest
import torch

from priml.baselines.etth1.testing import (
    golden_record,
    model_record,
    tiny_batches,
    tiny_config,
    training_record,
    update_record,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


class _Optimizer:
    def __init__(self, state: Mapping[str, object]) -> None:
        self.state = state

    def state_dict(self) -> Mapping[str, object]:
        return self.state


def test_tiny_config_shrinks_only_the_forecasting_geometry() -> None:
    cfg = tiny_config()
    assert (cfg.model.seq_len, cfg.model.pred_len, cfg.model.channels) == (5, 3, 4)
    assert cfg.seed is not None


def test_tiny_batches_are_three_distinct_non_broadcastable_batches() -> None:
    batches = list(tiny_batches())
    assert [b["media"].shape for b in batches] == [(2, 5, 4)] * 3
    assert [b["label"].shape for b in batches] == [(2, 3, 4)] * 3
    assert not torch.equal(batches[0]["media"], batches[1]["media"])
    assert not torch.equal(batches[1]["label"], batches[2]["label"])


def test_model_record_keeps_initial_weights_input_and_output() -> None:
    model = nn.Linear(4, 4)
    record = model_record(model)
    assert torch.equal(record["initial/weight"], model.weight)
    assert torch.equal(record["input"], next(tiny_batches())["media"])
    assert torch.equal(record["output"], model(record["input"]))
    assert "rng" in record


def test_update_record_flattens_optimizer_state_and_drops_initial_lr() -> None:
    model = nn.Linear(2, 3)
    model(torch.ones(4, 2)).sum().backward()
    state = {
        "state": {0: {"step": torch.tensor(1.0), "momentum_buffer": None}},
        "param_groups": [{"lr": 0.5, "nesterov": False, "initial_lr": 9.0}],
    }
    record = update_record(
        model,
        _Optimizer(state),
        output=torch.ones(2),
        loss=torch.tensor(0.25),
    )
    assert torch.equal(record["loss"], torch.tensor([0.25]))
    assert torch.equal(record["gradient/bias"], torch.full((3,), 4.0))
    assert torch.equal(record["optimizer/state/0/step"], torch.tensor(1.0))
    assert torch.equal(record["optimizer/param_groups/0/lr"], torch.tensor(0.5))
    assert torch.equal(
        record["optimizer/param_groups/0/nesterov"],
        torch.tensor(data=False),
    )
    assert not any("initial_lr" in key or "momentum_buffer" in key for key in record)


def test_update_record_refuses_state_it_cannot_flatten() -> None:
    with pytest.raises(TypeError, match="Unsupported record value: str"):
        update_record(
            nn.Linear(2, 3),
            _Optimizer({"note": "text"}),
            output=torch.ones(2),
            loss=torch.tensor(0.0),
        )


def test_training_record_captures_three_numbered_updates() -> None:
    record = training_record(tiny_config().make())
    assert {key.split("/")[0] for key in record if key.startswith("step")} == {
        "step1",
        "step2",
        "step3",
    }
    assert not torch.equal(record["step1/loss"], record["step3/loss"])


def test_golden_record_keeps_only_what_the_golden_stores() -> None:
    record = {
        "initial/w": torch.zeros(2),
        "output": torch.ones(2),
        "rng": torch.ones(3),
        "step1/loss": torch.ones(1),
        "step2/output": torch.ones(2),
        "step3/rng": torch.ones(3),
        "step3/parameter/w": torch.ones(2),
        "step3/optimizer/lr": torch.ones(()),
    }
    assert set(golden_record(record, training=False)) == {"initial/w", "output", "rng"}
    assert set(golden_record(record, training=True)) == {
        "initial/w",
        "step1/loss",
        "step3/rng",
        "step3/parameter/w",
        "step3/optimizer/lr",
    }


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
