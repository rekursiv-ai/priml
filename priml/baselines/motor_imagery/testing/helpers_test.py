"""Initializer-site adaptation and exact-mismatch sensitivity of EEG records."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.motor_imagery.testing.helpers_support_test import (
    _initializer_sites,
    compare_record,
    current_key,
    optimizer_record,
)
from priml.optimizers.composite import CompositeOptimizer


def test_initializer_location_adapter_retains_every_layer_site_and_count() -> None:
    source = {
        "random/aten.uniform_.default/conv.py:20": torch.tensor(6),
        "random/aten.uniform_.default/linear.py:30": torch.tensor(3),
        "random/aten.uniform_.default/linear.py:35": torch.tensor(1),
        "random/aten.uniform_.default/linear.py:36": torch.tensor(2),
        "random/aten.bernoulli_.float/dropout.py:70": torch.tensor(8),
        "initial/proj_out.weight": torch.arange(6).reshape(2, 3),
    }
    expected = {
        "random/aten.uniform_.default/conv.py:0": torch.tensor(6),
        "random/aten.uniform_.default/linear.py:0": torch.tensor(3),
        "random/aten.uniform_.default/linear.py:1": torch.tensor(1),
        "random/aten.uniform_.default/linear.py:2": torch.tensor(2),
        "random/aten.bernoulli_.float/dropout.py:70": torch.tensor(8),
        "initial/proj_out.weight": torch.arange(6).reshape(2, 3),
    }
    compare_record(expected, _initializer_sites(source))
    assert _initializer_sites(source).keys() == expected.keys()
    actual = {
        key.replace(":20", ":200").replace(":30", ":300").replace(":35", ":350"): value
        for key, value in source.items()
    }
    compare_record(source, actual)
    actual["random/aten.uniform_.default/linear.py:350"] = torch.tensor(2)
    with pytest.raises(ValueError, match=r"linear\.py:1"):
        compare_record(source, actual)


def test_initializer_location_adapter_rejects_an_omitted_or_added_site() -> None:
    expected = {
        "random/aten.uniform_.default/linear.py:30": torch.tensor(3),
        "random/aten.uniform_.default/linear.py:35": torch.tensor(1),
    }
    with pytest.raises(ValueError, match="missing"):
        compare_record(
            expected,
            {"random/aten.uniform_.default/linear.py:300": torch.tensor(3)},
        )
    with pytest.raises(ValueError, match="unexpected"):
        compare_record(
            expected,
            {**expected, "random/aten.uniform_.default/conv.py:20": torch.tensor(2)},
        )


@pytest.mark.parametrize(
    ("source", "current"),
    [
        ("initial/bn_t.weight", "initial/norm_t.weight"),
        ("final/bn_sep.running_var", "final/norm_sep.running_var"),
        ("bn.num_batches_tracked", "norm.num_batches_tracked"),
        ("gradient/classifier.bias", "gradient/proj_out.bias"),
        ("se1.fc1.weight", "se1.fc1.weight"),
        ("conv_t1.weight", "conv_t1.weight"),
    ],
)
def test_source_state_names_map_to_the_blessed_nouns(
    source: str,
    current: str,
) -> None:
    assert current_key(source) == current


def test_compare_record_reports_a_changed_bit() -> None:
    with pytest.raises(ValueError, match="bits: 1/2 differ"):
        compare_record(
            {"bits": torch.tensor([1.0, 2.0])},
            {"bits": torch.tensor([1.0, 3.0])},
        )


def test_optimizer_record_rejects_multiple_optimizer_members() -> None:
    optimizer = CompositeOptimizer(
        [
            torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]),
            torch.optim.SGD([torch.nn.Parameter(torch.ones(2))], lr=0.1),
        ],
    )
    with pytest.raises(ValueError, match=r"len\(optimizer.optimizers\) == 1"):
        optimizer_record(optimizer)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
