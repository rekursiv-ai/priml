"""Tests for parameter filters."""

from __future__ import annotations

from torch import nn

import torch

from priml.optimizers.parameter_filter import (
    complement,
    everything,
    excluding,
    matching,
    trainable,
)


def _weight() -> nn.Parameter:
    return nn.Parameter(torch.zeros(2, 3))


def test_everything_takes_frozen_parameters() -> None:
    frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
    assert everything("pos_embed", frozen)


def test_trainable_skips_frozen_parameters() -> None:
    frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
    assert trainable("weight", _weight())
    assert not trainable("pos_embed", frozen)


def test_matching_selects_by_name_fragment() -> None:
    select = matching("embed", "head")
    assert select("token_embed.weight", _weight())
    assert select("lm_head.weight", _weight())
    assert not select("block.0.attn.weight", _weight())


def test_excluding_rejects_a_named_fragment_and_defers_otherwise() -> None:
    select = excluding(trainable, "head")
    frozen = nn.Parameter(torch.zeros(2), requires_grad=False)
    assert select("block.weight", _weight())
    assert not select("lm_head.weight", _weight())
    assert not select("block.bias", frozen)


def test_complement_inverts_its_filter() -> None:
    select = complement(matching("head"))
    assert select("block.weight", _weight())
    assert not select("lm_head.weight", _weight())


def test_filters_compare_by_value_so_configs_can_be_diffed() -> None:
    """A closure never equals another; these do, or forks could not be diffed."""
    assert matching("a", "b") == matching("a", "b")
    assert matching("a") != matching("b")
    assert hash(matching("a")) == hash(matching("a"))
    assert excluding(everything, "x") == excluding(everything, "x")
    assert excluding(everything, "x") != excluding(everything, "y")
    assert hash(excluding(everything, "x")) == hash(excluding(everything, "x"))
    assert complement(everything) == complement(everything)
    assert complement(everything) != complement(matching("x"))
    assert hash(complement(everything)) == hash(complement(everything))
    assert matching("a") != "matching('a')"


def test_filter_reprs_name_functions_without_addresses() -> None:
    assert repr(matching("a", "b")) == "matching('a', 'b')"
    assert repr(excluding(everything, "head")) == "excluding(everything, 'head')"
    assert repr(complement(matching("x"))) == "complement(matching('x'))"
    assert repr(excluding(trainable, "head")) == "excluding(trainable, 'head')"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
