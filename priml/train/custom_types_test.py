"""Tests for train-layer structural types."""

from __future__ import annotations

import torch

from priml.train.custom_types import Closeable, ModelOutput


class _Resource:
    def close(self) -> None:
        pass


class _NotResource:
    pass


def test_model_output_accepts_indexable_predictions() -> None:
    assert isinstance(torch.tensor([1.0, 2.0]), ModelOutput)
    assert isinstance({"logits": torch.tensor([1.0, 2.0])}, ModelOutput)
    assert not isinstance(None, ModelOutput)
    assert not isinstance(3.0, ModelOutput)


def test_closeable_requires_a_close_method() -> None:
    assert isinstance(_Resource(), Closeable)
    assert not isinstance(_NotResource(), Closeable)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
