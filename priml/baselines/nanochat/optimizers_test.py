"""Check table update arithmetic, matrix routing, and schedule boundaries."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import cast

import math

import pytest
import torch

from priml.baselines.nanochat import optimizers
from priml.optimizers.normuon import NorMuon


@dataclass(slots=True, kw_only=True)
class _IndependentNorMuonConfig:
    lr: float = 0.04
    weight_decay: float = 0.2
    compile: bool = False

    def make(self) -> Callable[..., NorMuon]:
        return partial(
            NorMuon, lr=self.lr, weight_decay=self.weight_decay, compile=self.compile
        )


def test_ffn_factory_accepts_an_independent_config() -> None:
    """An injected factory needs the protocol, not NorMuon.Config inheritance."""
    config = optimizers.FFNScaledNorMuon.Config(channels_in=4, ffn_lr_multiplier=1.25)
    config.optimizer = _IndependentNorMuonConfig(lr=0.08)
    parameter = torch.nn.Parameter(torch.ones(8, 4))
    optimizer = config.make()([parameter])
    assert isinstance(optimizer, NorMuon)
    assert optimizer.param_groups[0]["lr"] == 0.08 * 2**0.5 * 1.25


def test_weight_decay_pulses_survive_copy_and_serialization() -> None:
    """Value records retain their order and schedule through Configgle roundtrips."""
    config = optimizers.ScheduledOptimizerUpdate.Config()
    config.weight_decay_pulses = (
        optimizers.WeightDecayPulse(center=0.5, half_width=0.25, multiplier=3),
        optimizers.WeightDecayPulse(center=0.75, multiplier=5, triangular=True),
    )
    copied = config.copy_tree()
    restored = optimizers.ScheduledOptimizerUpdate.Config.deserialize(
        config.serialize()
    )
    for candidate in (copied, restored):
        assert candidate.weight_decay_pulses == config.weight_decay_pulses
        assert candidate.make().weight_decay_multiplier(0.5) == 1.5
        assert candidate.make().weight_decay_multiplier(0.875) == 0.5


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sparse", [False, True])
def test_zero_beta_rmsprop_survives_checkpoint_and_positive_beta_update(
    dtype: torch.dtype, sparse: bool
) -> None:
    """Zero decay replaces moments and must not break the cumulative log state."""
    weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype))
    config = optimizers.BiasCorrectedRMSProp.Config(
        lr=0.25, beta2=0.0, eps=2.0, rowwise=True, sparse_rows=sparse
    )
    optimizer = config.make()([weight])
    sink = torch.tensor([[2.0, -2.0], [0.0, 0.0]])
    bitmap = torch.tensor([1, 0], dtype=torch.uint8)
    optimizer.gradient_sinks[weight] = sink
    optimizer.gradient_bitmaps[weight] = bitmap
    optimizer.step()
    assert torch.equal(weight, torch.tensor([[0.875, 2.125], [3.0, 4.0]], dtype=dtype))
    assert torch.equal(
        optimizer.state[weight]["second_moment"], torch.tensor([[4.0], [0.0]])
    )
    if sparse:
        assert optimizer.state[weight]["cum_log"] == -math.inf
        assert optimizer.state[weight]["sparse_scalars"]["cum_before"] == 0.0
        assert optimizer.state[weight]["sparse_scalars"]["cum_after"] == -math.inf

    restored_weight = torch.nn.Parameter(weight.detach().clone())
    restored = config.make()([restored_weight])
    restored.gradient_sinks[restored_weight] = sink
    restored.gradient_bitmaps[restored_weight] = bitmap
    restored.load_state_dict(optimizer.state_dict())
    sink.mul_(0.5)
    for member in (optimizer, restored):
        member.param_groups[0]["beta2"] = 0.5
        member.step()
    assert torch.isfinite(weight).all()
    assert torch.equal(restored_weight, weight)
    assert torch.equal(
        restored.state[restored_weight]["second_moment"],
        optimizer.state[weight]["second_moment"],
    )
    if sparse:
        for state in (optimizer.state[weight], restored.state[restored_weight]):
            assert state["cum_log"] == -math.inf
            assert state["sparse_scalars"]["cum_before"] == -math.inf
            assert state["sparse_scalars"]["cum_after"] == -math.inf


@pytest.mark.parametrize(
    "field", ["muon_warmdown", "adam_warmdown", "ngram_ramp_fraction"]
)
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_schedule_fractions_require_finite_positive_values(
    field: str, value: float
) -> None:
    """Reject invalid denominators while constructing the update policy."""
    config = optimizers.ScheduledOptimizerUpdate.Config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        config.make()


def test_schedule_fractions_above_one_remain_valid() -> None:
    """Positive denominator validation must not add an upper bound."""
    config = optimizers.ScheduledOptimizerUpdate.Config()
    config.muon_warmdown = 2.0
    config.adam_warmdown = 2.0
    config.ngram_ramp_fraction = 2.0
    config.make()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sparse", [False, True])
def test_rowwise_checkpoint_preserves_state_precision_and_next_update(
    dtype: torch.dtype, sparse: bool
) -> None:
    """Reloading narrow weights must retain FP32 moments and integer row indices."""
    weight = torch.nn.Parameter(torch.ones(4, 3, dtype=dtype))
    config = optimizers.BiasCorrectedRMSProp.Config(
        lr=0.13, beta2=0.91, rowwise=True, sparse_rows=sparse
    )
    optimizer = config.make()([weight])
    sink = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 7
    bitmap = torch.tensor([1, 0, 1, 1], dtype=torch.uint8)
    optimizer.gradient_sinks[weight] = sink
    optimizer.gradient_bitmaps[weight] = bitmap
    optimizer.step()
    restored_weight = torch.nn.Parameter(weight.detach().clone())
    restored = config.make()([restored_weight])
    restored.gradient_sinks[restored_weight] = sink
    restored.gradient_bitmaps[restored_weight] = bitmap
    restored.load_state_dict(optimizer.state_dict())
    before = optimizer.state[weight]
    after = restored.state[restored_weight]
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            assert after[name].dtype == value.dtype, name
            assert torch.equal(after[name], value), name
            assert after[name].data_ptr() != value.data_ptr(), name
        elif isinstance(value, dict):
            for key, scalar in cast(dict[str, object], value).items():
                assert isinstance(scalar, torch.Tensor)
                assert after[name][key].dtype == scalar.dtype, key
                assert torch.equal(after[name][key], scalar), key
    sink.mul_(0.37)
    bitmap.fill_(1)
    optimizer.step()
    restored.step()
    assert torch.equal(restored_weight, weight)
    assert torch.equal(after["second_moment"], before["second_moment"])


def test_rmsprop_bias_correction_follows_scheduled_beta() -> None:
    """Keep FP32 scalar rounding and the source's current-beta correction."""
    assert "BiasCorrectedRMSProp" in vars(optimizers)
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    config = optimizers.BiasCorrectedRMSProp.Config(lr=0.1, beta2=0.9)
    optimizer = config.make()([weight])
    expected = weight.detach().clone()
    moment = torch.zeros_like(weight)
    for index, beta in enumerate((0.9, 0.95, 0.99), start=1):
        gradient = torch.tensor([2.0, -0.5]) * index
        weight.grad = gradient
        optimizer.param_groups[0]["beta2"] = beta
        beta2 = torch.tensor(beta)
        lr = torch.tensor(config.lr)
        moment.lerp_(gradient.square(), 1 - beta2)
        denominator = (moment / (1 - beta2 ** torch.tensor(float(index)))).sqrt()
        expected.sub_(gradient / (denominator + torch.tensor(config.eps)) * lr)
        optimizer.step()
        assert torch.equal(weight, expected)
        assert torch.equal(optimizer.state[weight]["second_moment"], moment)


def test_sparse_rmsprop_keeps_idle_weights_and_advances_idle_moments() -> None:
    """Skipping an idle parameter update must not skip its moment evolution."""
    assert "BiasCorrectedRMSProp" in vars(optimizers)
    weight = torch.nn.Parameter(torch.ones(4, 3, dtype=torch.bfloat16))
    config = optimizers.BiasCorrectedRMSProp.Config(
        lr=0.1,
        beta2=0.9,
        rowwise=True,
        sparse_rows=True,
    )
    optimizer = config.make()([weight])
    sink = torch.ones(4, 3)
    bitmap = torch.tensor([0, 1, 0, 0], dtype=torch.uint8)
    optimizer.gradient_sinks[weight] = sink
    optimizer.gradient_bitmaps[weight] = bitmap
    optimizer.step()
    before = weight.detach().clone()
    moment = optimizer.state[weight]["second_moment"].clone()
    bitmap.zero_()
    optimizer.param_groups[0]["beta2"] = 0.95
    optimizer.step()
    expected = torch.lerp(
        moment, torch.zeros_like(moment), 1 - float(torch.tensor(0.95))
    )
    assert torch.equal(weight, before)
    assert torch.equal(optimizer.state[weight]["second_moment"], expected)
    assert optimizer.state[weight]["second_moment"].dtype == torch.float32


def test_sparse_rmsprop_refuses_missing_bitmap() -> None:
    """A miswired sparse run cannot silently fall back to dense updates."""
    assert "BiasCorrectedRMSProp" in vars(optimizers)
    weight = torch.nn.Parameter(torch.ones(2, 3))
    config = optimizers.BiasCorrectedRMSProp.Config(rowwise=True, sparse_rows=True)
    optimizer = config.make()([weight])
    weight.grad = torch.ones_like(weight)
    with pytest.raises(ValueError, match="no dirty bitmap"):
        optimizer.step()


def test_ffn_multiplier_changes_both_rectangular_projections() -> None:
    """The attention square and unrelated matrices keep their original rates."""
    assert "FFNScaledNorMuon" in vars(optimizers)
    parameters = [
        torch.nn.Parameter(torch.ones(shape))
        for shape in ((4, 4), (8, 4), (4, 8), (2, 3))
    ]
    config = optimizers.FFNScaledNorMuon.Config(channels_in=4, ffn_lr_multiplier=1.25)
    config.optimizer.compile = False
    optimizer = config.make()(parameters)
    rates = {
        tuple(group["params"][0].shape): group["lr"] for group in optimizer.param_groups
    }
    assert rates == pytest.approx(
        {(4, 4): 0.04, (8, 4): 0.05 * 2**0.5, (4, 8): 0.05, (2, 3): 0.04}
    )


@pytest.mark.parametrize("progress", [0.0, 0.25, 0.5, 1.0])
def test_weight_decay_pulse_has_open_endpoints(progress: float) -> None:
    """A rectangular pulse modifies only its open interval."""
    assert "ScheduledOptimizerUpdate" in vars(optimizers)
    config = optimizers.ScheduledOptimizerUpdate.Config()
    config.weight_decay_pulses = (
        optimizers.WeightDecayPulse(center=0.5, half_width=0.25, multiplier=3),
    )
    expected = (1 - progress) * (3 if 0.25 < progress < 0.75 else 1)
    assert config.make().weight_decay_multiplier(progress) == expected


def test_first_weight_decay_pulse_wins_and_triangle_reaches_peak() -> None:
    """Overlapping pulses are ordered policies, not multiplied together."""
    assert "ScheduledOptimizerUpdate" in vars(optimizers)
    config = optimizers.ScheduledOptimizerUpdate.Config()
    config.weight_decay_pulses = (
        optimizers.WeightDecayPulse(
            center=0.5, half_width=0.25, multiplier=3, triangular=True
        ),
        optimizers.WeightDecayPulse(center=0.5, half_width=0.5, multiplier=9),
    )
    update = config.make()
    assert update.weight_decay_multiplier(0.5) == 1.5
    assert update.weight_decay_multiplier(0.375) == 1.25


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
