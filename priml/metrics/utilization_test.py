"""Tests for priml.metrics.utilization."""

from __future__ import annotations

from types import SimpleNamespace

from configgle import Fig

import pytest
import torch

from priml.metrics.custom_types import MetricProtocol
from priml.metrics.utilization import Utilization
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.cost import Compute, Cost, Flops
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer


def _tiny_transformer() -> Transformer.Config:
    return Transformer.Config(
        proj_in=Embedding.Config(channels_in=128, shard="vocab"),
        channels_in=32,
        channels_out=128,
        num_layers=2,
        block=TransformerBlock.Config(
            attn=SelfAttention.Config(
                num_heads=4,
                channels_head=8,
                causal=True,
                rope=RoPE.Config(channels_head=8),
            ),
        ),
        proj_out=Sequential.Config(
            elements=[RMSNorm.Config(), Linear.Config(shard="vocab")],
        ),
    )


class _Counted:
    """A priced config that counts how often it was asked."""

    class Config(Fig["_Counted"]):
        calls: int = 0
        """Times ``cost`` ran."""

        def cost(self, **kwargs: object) -> Cost:
            del kwargs
            self.calls += 1
            return Cost(
                primal=Compute(flops=Flops(matmul=2, elementwise=3_000_000)),
                adjoint=Compute(flops=Flops(matmul=4, elementwise=5_000_000)),
            )

    def __init__(self, config: Config) -> None:
        del config


class _Unpriced:
    class Config(Fig["_Unpriced"]):
        pass

    def __init__(self, config: Config) -> None:
        del config


def _root(model_config: object) -> object:
    """Mimic the path ``bind`` reads off a built ``TrainLoop``: ``step.config.model``."""
    return SimpleNamespace(
        step=SimpleNamespace(config=SimpleNamespace(model=model_config)),
    )


def test_utilization_is_a_metric() -> None:
    assert isinstance(Utilization(Utilization.Config()), MetricProtocol)


def test_utilization_requires_completed_device_work() -> None:
    meter = Utilization(Utilization.Config())
    assert meter.requires_device_timing


@pytest.mark.parametrize("peak", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("vector", [False, True])
def test_utilization_rejects_nonpositive_or_nonfinite_peaks(
    peak: float,
    vector: bool,
) -> None:
    config = Utilization.Config()
    if vector:
        config.peak_vector_flops_per_sec = peak
    else:
        config.peak_flops_per_sec = peak
    with pytest.raises(ValueError, match="positive and finite"):
        Utilization(config)


def test_built_alone_it_is_unbound() -> None:
    """A bare ``make`` binds against the metric itself, which holds no model."""
    with pytest.raises(TypeError, match=r"step\.config\.model"):
        Utilization.Config().make()


def test_mfu_is_the_palm_flops_times_token_rate_over_peak() -> None:
    """``6N`` matrix FLOPs plus attention, times tokens/sec, over one device."""
    config = _tiny_transformer()
    meter_config = Utilization.Config()
    meter_config.peak_flops_per_sec = 1e9
    meter = Utilization(meter_config)
    meter.bind(_root(config.copy_tree().finalize()))

    meter.update(
        torch.empty(0),
        input_ids=torch.zeros(2, 16, dtype=torch.long),
        step_sec=0.5,
    )
    measured = meter.compute()

    params = sum(p.numel() for p in config.make().parameters())
    matrix = params - 128 * 32
    flops_per_token = 6 * matrix + 12 * 2 * (4 * 8) * 16
    assert measured["tokens_per_sec"] == 64.0
    assert measured["mfu"] == flops_per_token * 64.0 / 1e9


def test_compute_reports_every_silo_against_its_own_ceiling() -> None:
    meter_config = Utilization.Config()
    meter_config.tokens_key = "media"
    meter_config.peak_flops_per_sec = 1e3
    meter_config.peak_vector_flops_per_sec = 1e9
    meter = Utilization(meter_config)
    meter.bind(_root(_Counted.Config()))

    meter.update(torch.empty(0), media=torch.zeros(4, 8), step_sec=2.0)

    # 32 tokens over 2s: 6 matmul FLOPs each against 1e3; 8e6 elementwise
    # against 1e9; nothing in the other silos.
    assert meter.compute() == {
        "tokens_per_sec": 16.0,
        "mfu": 6 * 16.0 / 1e3,
        "utilization_elementwise": 8_000_000 * 16.0 / 1e9,
        "utilization_reduction": 0.0,
        "utilization_selection": 0.0,
        "utilization_sort": 0.0,
    }


def test_compute_averages_over_the_updates_since_reset() -> None:
    """Tokens and seconds are summed, so one slow step does not dominate."""
    meter_config = Utilization.Config()
    meter_config.tokens_key = "media"
    meter_config.peak_flops_per_sec = 1e3
    meter = Utilization(meter_config)
    meter.bind(_root(_Counted.Config()))

    meter.update(torch.empty(0), media=torch.zeros(4, 8), step_sec=1.0)
    meter.update(torch.empty(0), media=torch.zeros(4, 8), step_sec=3.0)
    assert meter.compute()["tokens_per_sec"] == 64 / 4.0
    meter.reset()
    meter.update(torch.empty(0), media=torch.zeros(1, 8), step_sec=1.0)
    assert meter.compute()["tokens_per_sec"] == 8.0


def test_compute_without_an_update_is_empty() -> None:
    """Nothing timed, nothing reported: a zero here would read as an idle run."""
    meter = Utilization(Utilization.Config())
    meter.bind(_root(_Counted.Config()))
    assert meter.compute() == {}


def test_cost_is_priced_once_per_sequence_length_and_token_count() -> None:
    priced = _Counted.Config()
    meter = Utilization(Utilization.Config())
    meter.bind(_root(priced))

    meter.update(torch.empty(0), input_ids=torch.zeros(1, 8), step_sec=1.0)
    meter.update(torch.empty(0), input_ids=torch.zeros(1, 8), step_sec=1.0)
    assert priced.calls == 1
    meter.update(torch.empty(0), input_ids=torch.zeros(3, 8), step_sec=1.0)
    assert priced.calls == 2
    meter.update(torch.empty(0), input_ids=torch.zeros(1, 16), step_sec=1.0)
    assert priced.calls == 3


def test_update_prices_the_batch_by_its_shape_not_by_rows() -> None:
    """The metric states ``seq_len`` and ``batch_size``; ``cost`` derives the rows."""
    seen: list[dict[str, object]] = []

    class _Spy:
        class Config(Fig["_Spy"]):
            def cost(self, **kwargs: object) -> Cost:
                seen.append(kwargs)
                return Cost()

        def __init__(self, config: Config) -> None:
            del config

    meter = Utilization(Utilization.Config())
    meter.bind(_root(_Spy.Config()))
    meter.update(torch.empty(0), input_ids=torch.zeros(3, 8), step_sec=1.0)
    assert seen == [{"seq_len": 8, "batch_size": 3, "rows": 24}]


def test_parameter_reductions_use_each_updates_actual_token_count() -> None:
    model = Linear.Config()
    model.channels_in = 2
    model.channels_out = 3
    model.bias = True
    config = Utilization.Config()
    config.peak_vector_flops_per_sec = 1_000
    meter = Utilization(config)
    meter.bind(_root(model.finalize()))

    meter.update(torch.empty(0), input_ids=torch.zeros(1, 8), step_sec=1.0)
    meter.update(torch.empty(0), input_ids=torch.zeros(3, 8), step_sec=1.0)

    assert meter.compute()["utilization_reduction"] == 3 * (7 + 23) / 2 / 1_000


def test_bind_rejects_evaluation_placement() -> None:
    meter = Utilization(Utilization.Config())
    root = SimpleNamespace(
        step=SimpleNamespace(config=SimpleNamespace(model=_Counted.Config())),
        metrics_eval={"utilization": meter},
    )
    with pytest.raises(ValueError, match=r"metrics_train.*metrics_eval"):
        meter.bind(root)


def test_bind_rejects_a_model_config_without_cost() -> None:
    meter = Utilization(Utilization.Config())
    with pytest.raises(TypeError, match=r"_Unpriced\.Config"):
        meter.bind(_root(_Unpriced.Config()))


def test_bind_rejects_a_root_without_a_model_config() -> None:
    meter = Utilization(Utilization.Config())
    with pytest.raises(TypeError, match=r"step\.config\.model"):
        meter.bind(SimpleNamespace(step=SimpleNamespace()))


def test_update_before_bind_fails_loudly() -> None:
    meter = Utilization(Utilization.Config())
    with pytest.raises(RuntimeError, match="bind"):
        meter.update(torch.empty(0), input_ids=torch.zeros(1, 8), step_sec=1.0)


def test_update_rejects_a_non_tensor_token_field() -> None:
    meter_config = Utilization.Config()
    meter_config.tokens_key = "valid_count"
    meter = Utilization(meter_config)
    meter.bind(_root(_Counted.Config()))
    with pytest.raises(TypeError, match="valid_count"):
        meter.update(torch.empty(0), valid_count=4, step_sec=1.0)


def test_update_requires_step_sec() -> None:
    """A bus without ``step_sec`` is a caller that cannot be timed."""
    meter = Utilization(Utilization.Config())
    meter.bind(_root(_Counted.Config()))
    with pytest.raises(TypeError, match="step_sec"):
        meter.update(torch.empty(0), input_ids=torch.zeros(1, 8))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
