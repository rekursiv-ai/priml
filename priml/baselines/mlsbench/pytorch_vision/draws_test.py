"""Tests for the random-draw counter."""

from __future__ import annotations

import torch

from priml.baselines.mlsbench.pytorch_vision.draws import recorded_draws


def test_counts_a_sampler_by_api_and_calling_function() -> None:
    with recorded_draws() as counts:
        _ = torch.randint(0, 5, (2,))
        _ = torch.randint(0, 5, (2,))
        _ = torch.rand(1)
        _ = torch.randperm(3)
    me = (
        "pytorch_vision/draws_test.py:test_counts_a_sampler_by_api_and_calling_function"
    )
    assert counts == {
        f"randint@{me}": 2,
        f"rand@{me}": 1,
        f"randperm@{me}": 1,
    }


def test_counts_a_tensor_method_called_on_an_instance_or_through_the_class() -> None:
    tensor = torch.empty(3)
    with recorded_draws() as counts:
        assert tensor.uniform_() is tensor
        _ = torch.Tensor.normal_(tensor)
        _ = tensor.random_()
    assert sorted(counts.values()) == [1, 1, 1]
    assert sorted(key.split("@")[0] for key in counts) == [
        "normal_",
        "random_",
        "uniform_",
    ]


def test_counting_leaves_the_stream_untouched() -> None:
    generator = torch.Generator().manual_seed(0)
    expected = torch.randint(0, 100, (4,), generator=generator)
    with recorded_draws():
        generator = torch.Generator().manual_seed(0)
        assert torch.equal(torch.randint(0, 100, (4,), generator=generator), expected)


def test_the_samplers_are_restored_afterwards() -> None:
    before = torch.randint, torch.Tensor.uniform_
    with recorded_draws():
        pass
    assert (torch.randint, torch.Tensor.uniform_) == before


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
