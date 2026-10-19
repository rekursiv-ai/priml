"""Tests for the VGG-11 network."""

from __future__ import annotations

from typing import Final

import torch

from priml.baselines.mlsbench.opt.gradient_compression.model import VGG
from priml.testing.cost import assert_cost_matches_torch


def tiny() -> VGG:
    """Return a two-stage VGG with three classes."""
    config = VGG.Config()
    config.channels_in = 3
    config.channels_out = 5
    config.channels_hidden = ((4,), (6, 7))
    config.channels_head = (8, 9)
    return config.make()


BATCH: Final = 2
"""Images of the cost run."""

SIDE: Final = 4
"""Side of the cost run's images: two stages halve it to 1."""


# A real image carries no gradient, so torch would skip the first layer's input
# gradient; the cost counts it, so the input requires grad, as ``conv_test.py``'s do.
def test_the_cost_counts_every_matmul_the_forward_issues() -> None:
    config = VGG.Config()
    config.channels_in = 3
    config.channels_out = 5
    config.channels_hidden = ((4,), (6, 7))
    config.channels_head = (8, 9)
    config.image_size = SIDE
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(BATCH, 3, SIDE, SIDE, requires_grad=True),
        batch_size=BATCH,
        dtype=None,
    )
    convs = [(3, 4, 4), (4, 6, 2), (6, 7, 2)]
    products = sum(c_in * c_out * 9 * side**2 for c_in, c_out, side in convs)
    assert analytical["flops", "primal", "matmul"].sum() == 2 * BATCH * (
        products + 7 * 8 + 8 * 9 + 9 * 5
    )


def test_the_default_is_vgg11_with_a_three_layer_classifier() -> None:
    config = VGG.Config()
    assert sum(map(len, config.channels_hidden)) == 8
    assert config.channels_head == (512, 512)
    assert config.dropout == 0.5


def test_layers_are_registered_in_the_reference_order() -> None:
    names = [name for name, _ in tiny().state_dict().items()]
    assert names[:2] == ["features.0.weight", "features.0.bias"]
    # Conv, norm, ReLU, then the stage's max pool at index 3: the next conv is 4.
    assert "features.4.weight" in names
    assert "features.3.weight" not in names
    assert [n for n in names if n.startswith("classifier")] == [
        "classifier.0.weight",
        "classifier.0.bias",
        "classifier.3.weight",
        "classifier.3.bias",
        "classifier.6.weight",
        "classifier.6.bias",
    ]


def test_forward_pools_globally_and_drops_only_in_training() -> None:
    model = tiny()
    media = torch.randn(2, 3, 4, 6, generator=torch.Generator().manual_seed(0))
    model.eval()
    first = model(media)
    assert first.shape == (2, 5)
    assert torch.equal(first, model(media))
    model.train()
    state = torch.get_rng_state()
    _ = model(media)
    assert not torch.equal(torch.get_rng_state(), state)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
