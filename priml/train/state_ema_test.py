"""Pin eager snapshots, persistent-buffer arithmetic, and live-model ownership."""

from torch import nn

import pytest
import torch

from priml.train.state_ema import StateEMA


@pytest.mark.parametrize(("decay", "delta"), [(0.0, 6), (0.5, 3.875), (1.0, 0)])
def test_three_updates_average_all_float_state_and_copy_integer_state(
    decay: float,
    delta: float,
) -> None:
    model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
    model.register_buffer("derived", torch.arange(4).float(), persistent=False)
    model[0].requires_grad_(False)
    with torch.no_grad():
        for value in model.state_dict().values():
            if value.is_floating_point():
                value.copy_(torch.arange(value.numel()).reshape(value.shape) / 8)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    config = StateEMA.Config()
    config.decay = decay
    ema = config.make()(model)
    assert ema.module is not model
    assert not ema.module.training
    assert all(not parameter.requires_grad for parameter in ema.module.parameters())
    assert any(parameter.requires_grad for parameter in model.parameters())
    for step in range(1, 4):
        with torch.no_grad():
            for value in model.state_dict().values():
                value.add_(step)
            model.get_buffer("derived").add_(step)
        ema.update(model)
    for name, value in ema.module.state_dict().items():
        assert torch.equal(
            value,
            initial[name] + (delta if value.is_floating_point() else 6),
        )
    assert torch.equal(ema.module.get_buffer("derived"), torch.arange(4).float())
    assert torch.equal(model.get_buffer("derived"), torch.arange(4).float() + 6)
    assert model.training


@pytest.mark.parametrize("decay", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_decay_is_rejected_before_snapshot(decay: float) -> None:
    config = StateEMA.Config()
    config.decay = decay
    with pytest.raises(
        ValueError,
        match=r"^decay must be finite and between zero and one\.$",
    ):
        config.make()(nn.Linear(2, 3))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
