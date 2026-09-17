"""Tests for embedding module."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden

import torch

from priml.model.cost import Bytes, Compute, Cost, Flops
from priml.model.embedding import Embedding
from priml.model.init import normal
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_embedding_config_pprint() -> None:
    config = Embedding.Config(8, 4)
    assert_pprint_golden(
        test_file=__file__,
        name="embedding",
        config=config,
    )


def test_embedding_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="embedding",
        build_module=lambda: Embedding.Config(8, 4).make(),
        build_input=lambda: torch.tensor([[0, 3, 7]]),
        seed=0,
    )


def test_embedding():
    m = Embedding.Config(1000, 64).make()
    ids = torch.randint(0, 1000, (2, 8))
    assert m(ids).shape == (2, 8, 64)


def test_embedding_reset():
    m = Embedding.Config(1000, 64).make()
    m.reset_parameters()


def test_embedding_padding_idx():
    m = Embedding.Config(1000, 64, padding_idx=0).make()
    assert m(torch.zeros(1, dtype=torch.long)).abs().sum() == 0


def test_the_table_realizes_the_spread_it_was_asked_for():
    """A table must not be drawn narrower than its own initializer states.

    Every initializer here divides by ``sqrt(depth + 1)`` and DEFAULTS that
    depth to 1, so a ``reset_parameters`` that simply omits it draws at 0.707
    of the request -- a real change to the model, and one no shape, name, or
    dtype assertion can see. ``depth`` therefore has to be forwarded, exactly
    as ``Linear`` and ``Conv`` forward theirs.
    """
    torch.manual_seed(0)
    m = Embedding.Config(
        4096,
        256,
        init_weight=partial(normal, std=0.5),
    ).make()
    assert abs(float(m.weight.detach().std()) / 0.5 - 1.0) < 0.02


def test_a_depth_scales_the_table_down():
    """The field is not decorative: a stated depth still scales.

    Nothing in this repo asks a lookup table for depth scaling -- a table has
    no residual branch -- but the field exists so the default is a CHOICE
    rather than an omission, and a choice has to be honored to be one.
    """
    torch.manual_seed(0)
    flat = Embedding.Config(
        4096,
        256,
        init_weight=partial(normal, std=0.5),
    ).make()
    torch.manual_seed(0)
    scaled = Embedding.Config(
        4096,
        256,
        depth_index=((3, 4),),
        init_weight=partial(normal, std=0.5),
    ).make()
    assert torch.allclose(scaled.weight.detach(), flat.weight.detach() / 2.0)


def test_embedding_cost_is_a_gather() -> None:
    """A lookup gathers one row; the adjoint scatter-adds its four gradients."""
    config = Embedding.Config(8, 4)
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 8, (3,)),
        num_tokens=3,
    )
    assert analytical == Cost(
        primal=Compute(bytes=Bytes(selection=4 * (1 + 2 * 4))),
        adjoint=Compute(
            flops=Flops(selection=4),
            bytes=Bytes(selection=4 * (1 + 3 * 4 + 32 / 3)),
        ),
        params=32,
        params_active=4,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
