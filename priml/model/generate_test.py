"""Tests for the sampling helpers in :mod:`priml.model.generate`."""

from __future__ import annotations

from torch import Tensor

import torch

from priml.model.generate import _sample, _topp_filter


def test_sample_top_p_keeps_boundary_token():
    """Top-p must keep the smallest set whose cumulative prob reaches ``top_p``.

    Regression for GEN-TOPP (Issue#333): the nucleus mask used ``>=`` on
    the exclusive cumulative probability, dropping the token that brings
    the running mass exactly to ``top_p``. The HuggingFace convention uses
    strict ``>``. With a uniform 4-token distribution and ``top_p=0.5`` the
    exclusive cumsum is ``[0, .25, .5, .75]``; ``>=`` keeps only 2 tokens
    while the correct ``>`` keeps 3 (mass through the boundary token).
    """
    logits = torch.zeros(1, 4)  # softmax -> uniform 0.25 each.
    probs = _topp_probs(logits, top_p=0.5)
    kept = (probs > 0).sum(dim=-1).item()
    assert kept == 3


def test_sample_top_p_restores_vocab_order():
    """Filtered logits must map back to original vocab positions.

    Regression for GEN-TOPP (Issue#333): the surviving token's probability
    mass must land on its original vocab index, not a sorted position.
    """
    logits = torch.tensor([[1.0, 0.0, 9.0, 0.5, 0.2]])  # argmax at index 2.
    probs = _topp_probs(logits, top_p=0.5)
    assert probs.argmax(dim=-1).item() == 2


def test_sample_greedy_is_argmax():
    """Temperature 0 returns the argmax token id."""
    logits = torch.tensor([[1.0, 9.0, 0.5]])
    token = _sample(logits, 0.0, 0, 1.0)
    assert token.item() == 1


def _topp_probs(logits: Tensor, *, top_p: float) -> Tensor:
    """Recover the kept-token distribution under top-p, deterministically.

    The nucleus filter sets out-of-nucleus logits to ``-1e10``, which softmaxes
    to exactly 0, so the filtered softmax IS the kept distribution -- the same
    distribution ``_sample`` draws from, but without the 20k-iteration
    Monte-Carlo loop (or its sampling flakiness).
    """
    return _topp_filter(logits, top_p).softmax(dim=-1)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
