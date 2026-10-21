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
    assert isinstance(torch.tensor([1.0, 2.0]), ModelOutput)  # pyrefly: ignore[unsafe-overlap] -- pyrefly compares the full __getitem__ signature; runtime_checkable isinstance checks only that the member exists, which is the property under test.
    assert isinstance({"logits": torch.tensor([1.0, 2.0])}, ModelOutput)  # pyrefly: ignore[unsafe-overlap] -- pyrefly compares the full __getitem__ signature; runtime_checkable isinstance checks only that the member exists, which is the property under test.
    assert not isinstance(None, ModelOutput)
    assert not isinstance(3.0, ModelOutput)


def test_a_narrowed_model_output_indexes_like_the_model_output() -> None:
    output: object = torch.tensor([1.0, 2.0])
    assert isinstance(output, ModelOutput)
    assert output[0] == 1.0


def test_closeable_requires_a_close_method() -> None:
    assert isinstance(_Resource(), Closeable)
    assert not isinstance(_NotResource(), Closeable)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
