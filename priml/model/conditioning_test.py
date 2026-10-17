"""Tests for diffusion timestep and label conditioning."""

from __future__ import annotations

import math

from configgle import InlineConfig

import pytest
import torch

from priml.math.diffusion.conditioning import timestep_embedding
from priml.model.conditioning import (
    CategorySlotFiLM,
    LabelEmbedder,
    SigmoidResidualGate,
    TimestepEmbedder,
)


def test_timestep_embedder_features_forward_and_cost() -> None:
    config = TimestepEmbedder.Config(4, channels_frequency=6, max_period=100.0)
    embedder = config.make()
    times = torch.tensor([0.25, 0.75])
    features = embedder.frequencies(times)
    assert features.shape == (2, 6)
    torch.testing.assert_close(features, timestep_embedding(times, 6, 100.0))
    assert embedder(times, ignored=True).shape == (2, 4)
    assert config.cost(batch_size=2, dtype=torch.float32).params > 0


def test_timestep_embedder_accepts_custom_activation() -> None:
    config = TimestepEmbedder.Config(
        4,
        channels_frequency=6,
        activation=InlineConfig(torch.nn.ReLU),
    )
    embedder = config.make()
    first = embedder.mlp[0]
    activation = embedder.mlp[1]
    second = embedder.mlp[2]
    assert isinstance(first, torch.nn.Linear)
    assert isinstance(activation, torch.nn.ReLU)
    assert isinstance(second, torch.nn.Linear)
    assert first.bias is not None
    assert second.bias is not None
    assert first.weight.shape == (4, 6)
    assert second.weight.shape == (4, 4)


def test_token_drop_uses_device_and_strict_probability_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fixed_random(size: int, *, device: torch.device) -> torch.Tensor:
        if device.type == "cpu":
            assert size == 3
            assert device.type == "cpu"
        return torch.full((size,), 0.5, device=device)

    monkeypatch.setattr(torch, "rand", fixed_random)
    embedder = LabelEmbedder.Config(5, 4, dropout=0.5).make()
    labels = torch.tensor([0, 2, 4])
    assert torch.equal(embedder.token_drop(labels), labels)
    assert embedder.token_drop(torch.empty((2,), device="meta")).device.type == "meta"


@pytest.mark.parametrize("dropout", [math.nan, -0.1, 1.0])
def test_label_embedder_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout"):
        LabelEmbedder.Config(5, 4, dropout=dropout).finalize()


def test_label_embedder_forced_and_training_drop_paths() -> None:
    config = LabelEmbedder.Config(5, 4, dropout=0.5)
    embedder = config.make()
    assert config.num_rows == 6
    labels = torch.tensor([0, 2, 4])
    forced = embedder(labels, force_drop=torch.tensor([True, False, True]))
    expected_labels = torch.tensor([5, 2, 5])
    assert torch.equal(forced, embedder.embedding_table(expected_labels))
    embedder.eval()
    assert torch.equal(embedder(labels), embedder.embedding_table(labels))
    embedder.train()
    torch.manual_seed(0)
    dropped = embedder.token_drop(labels)
    assert dropped.shape == labels.shape
    assert torch.all((dropped == labels) | (dropped == 5))
    torch.manual_seed(0)
    assert torch.equal(embedder(labels), embedder.embedding_table(dropped))
    estimate = config.cost(batch_size=3, dtype=torch.float32)
    assert estimate.params == 24


def test_label_embedder_without_dropout_has_no_null_row() -> None:
    config = LabelEmbedder.Config(5, 4, dropout=0.0)
    embedder = config.make()
    assert config.num_rows == 5
    assert embedder(torch.tensor([1, 3])).shape == (2, 4)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            CategorySlotFiLM.Config(num_categories=0, channels_out=4),
            "Category, augmentation, and slot counts",
        ),
        (
            CategorySlotFiLM.Config(channels_out=-2),
            "CategorySlotFiLM requires positive channels_out",
        ),
    ],
)
def test_category_slot_film_rejects_invalid_config_geometry(
    config: CategorySlotFiLM.Config,
    message: str,
) -> None:
    config.finalize()
    with pytest.raises(ValueError, match=message):
        config.make()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            CategorySlotFiLM.Config(channels_out=0),
            "CategorySlotFiLM requires positive channels_out",
        ),
        (
            CategorySlotFiLM.Config(num_slots=0, channels_out=4),
            "Category, augmentation, and slot counts",
        ),
        (
            CategorySlotFiLM.Config(num_slots=4, channels_out=4),
            "channels_out must provide a channel for each slot and the base",
        ),
    ],
)
def test_category_slot_film_constructor_checks_raw_geometry(
    config: CategorySlotFiLM.Config,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CategorySlotFiLM(config)


def test_category_slot_film_composes_slots_and_resets_semantic_tables() -> None:
    config = CategorySlotFiLM.Config(
        num_categories=3,
        num_augmentations=2,
        num_slots=2,
        channels_out=6,
    )
    module = config.make()
    category = torch.tensor([0, 2, 1])
    augmentation = torch.tensor([1, 0, 1])
    permutation = torch.tensor([[1, 0], [0, 1], [1, 0]])
    output = module(category, augmentation, permutation)
    assert output.shape == (3, 6)
    assert torch.isfinite(output).all()
    with torch.no_grad():
        module.category_embed.weight.fill_(1)
        module.augmentation_embed.weight.fill_(1)
        module.slot_base.fill_(1)
        module.slot_embeddings.fill_(1)
    module.reset_parameters()
    assert torch.equal(
        module.category_embed.weight,
        torch.zeros_like(module.category_embed.weight),
    )
    assert torch.equal(
        module.augmentation_embed.weight,
        torch.zeros_like(module.augmentation_embed.weight),
    )
    assert torch.equal(module.slot_base, torch.zeros_like(module.slot_base))
    assert torch.equal(module.slot_embeddings, torch.zeros_like(module.slot_embeddings))


def test_category_slot_film_cost_caps_active_rows_to_reachable_tables() -> None:
    """Active parameter ownership counts unique tasks, views, and color slots."""
    measured = CategorySlotFiLM.Config(
        num_categories=2,
        num_augmentations=8,
        num_slots=9,
        channels_out=12,
    ).cost(seq_len=1, batch_size=16, dtype=torch.bfloat16)

    assert measured.params == 732
    assert measured.params_active == 732


def test_sigmoid_residual_gate_resets_with_bias_two_and_open_probability() -> None:
    module = SigmoidResidualGate.Config(channels_out=4).make()
    with torch.no_grad():
        module.linear.weight.fill_(3)
        assert module.linear.bias is not None
        module.linear.bias.zero_()
    module.reset_parameters()
    output = module(torch.zeros(2, 4))
    assert torch.equal(output, torch.sigmoid(torch.full_like(output, 2.0)))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
