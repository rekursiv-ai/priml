"""Tests for SpeedrunDiT's AdamW/Muon recipe."""

from __future__ import annotations

from typing import Literal, Protocol

from torch import Tensor, nn

import torch

from priml.baselines.speedrundit.optimizers import speedrundit_optimizer
from priml.optimizers.muon import Muon


class _ParameterGroup(Protocol):
    """Typed view of the optimizer group field used by this test."""

    def __getitem__(self, key: Literal["params"], /) -> list[Tensor]: ...


def _parameter_names(group: _ParameterGroup, names: dict[int, str]) -> set[str]:
    """Resolve optimizer parameter objects back to their module names."""
    return {names[id(parameter)] for parameter in group["params"]}


def test_speedrundit_optimizer_rates_and_parameter_partition() -> None:
    model = nn.Module()
    for name in (
        "x_embedder",
        "t_embedder",
        "y_embedder",
        "final_layer",
        "cls_projector",
        "adaLN_modulation",
        "blocks",
    ):
        setattr(model, name, nn.Linear(2, 3))

    config = speedrundit_optimizer()
    optimizer = config.make()(model)
    adamw, muon = optimizer.optimizers

    assert isinstance(adamw, torch.optim.AdamW)
    assert isinstance(muon, Muon)
    adam_group = adamw.param_groups[0]
    assert adam_group["lr"] == 1e-4
    assert adam_group["betas"] == (0.9, 0.999)
    assert adam_group["weight_decay"] == 0.0
    assert adam_group["eps"] == 1e-15
    assert muon.param_groups[0]["lr"] == 1e-3
    assert muon.param_groups[0]["momentum"] == 0.95
    assert muon.param_groups[0]["weight_decay"] == 0.0
    assert muon.param_groups[0]["nesterov"] is True
    assert muon.param_groups[0]["ns_steps"] == 5
    assert muon.param_groups[0]["reference_numerics"] is True

    names = {id(parameter): name for name, parameter in model.named_parameters()}
    adamw_names = _parameter_names(adam_group, names)
    muon_names = _parameter_names(muon.param_groups[0], names)
    expected_adamw = {
        f"{name}.{field}"
        for name in (
            "x_embedder",
            "t_embedder",
            "y_embedder",
            "final_layer",
            "cls_projector",
            "adaLN_modulation",
        )
        for field in ("weight", "bias")
    } | {"blocks.bias"}
    assert adamw_names == expected_adamw
    assert muon_names == {"blocks.weight"}


def test_speedrundit_optimizer_can_disable_reference_numerics() -> None:
    model = nn.Linear(2, 3)
    optimizer = speedrundit_optimizer(reference_numerics=False).make()(model)
    _, muon = optimizer.optimizers

    assert isinstance(muon, Muon)
    assert muon.param_groups[0]["reference_numerics"] is False


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
