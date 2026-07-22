"""Tests for embedding module."""

from __future__ import annotations

import torch

from priml.model.embedding import Embedding
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
