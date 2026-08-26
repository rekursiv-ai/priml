"""Tests for embedding module."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from priml.model.embedding import Embedding
from priml.model.init import normal
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


if TYPE_CHECKING:
    import pytest


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def test_embedding_config_pprint(request: pytest.FixtureRequest) -> None:
    config = Embedding.Config(4, num_embeddings=8)
    assert_text_golden(
        request,
        test_file=__file__,
        name="embedding",
        rendered=config.pformat(hide_default_values=False),
    )


def test_embedding_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="embedding",
        build_module=lambda: Embedding.Config(4, num_embeddings=8).make(),
        build_input=lambda: torch.tensor([[0, 3, 7]]),
        seed=0,
    )


def test_embedding():
    m = Embedding.Config(64, num_embeddings=1000).make()
    ids = torch.randint(0, 1000, (2, 8))
    assert m(ids).shape == (2, 8, 64)


def test_embedding_reset():
    m = Embedding.Config(64, num_embeddings=1000).make()
    m.reset_parameters()


def test_embedding_padding_idx():
    m = Embedding.Config(64, num_embeddings=1000, padding_idx=0).make()
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
        256,
        num_embeddings=4096,
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
        256,
        num_embeddings=4096,
        init_weight=partial(normal, std=0.5),
    ).make()
    torch.manual_seed(0)
    scaled = Embedding.Config(
        256,
        num_embeddings=4096,
        depth_index=((3, 4),),
        init_weight=partial(normal, std=0.5),
    ).make()
    assert torch.allclose(scaled.weight.detach(), flat.weight.detach() / 2.0)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
