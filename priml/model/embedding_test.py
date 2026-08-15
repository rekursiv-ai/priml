"""Tests for embedding module."""

from __future__ import annotations

from functools import partial

import torch

from priml.model.embedding import Embedding
from priml.model.init import normal
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
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
        depth=3,
        init_weight=partial(normal, std=0.5),
    ).make()
    assert torch.allclose(scaled.weight.detach(), flat.weight.detach() / 2.0)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
