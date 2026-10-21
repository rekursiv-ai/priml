"""Checks for the boosting module."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026.boosting import (
    DecisionTree,
    LightGBMModel,
)
from priml.baselines.casmi2026.testing import float32_comparands
from priml.testing.golden import assert_tensor_golden, read_tensors


_CWD: Final = Path(__file__).resolve().parent
_BASELINE: Final = _CWD

GOLDEN: Final = _BASELINE / "testdata" / "lightgbm.pt"


def tiny_text(expected: dict[str, Tensor], case: int) -> str:
    """Express complete numeric model state in its ordinary external asset format."""
    header = [
        "tree",
        "version=v4",
        "num_class=1",
        "num_tree_per_iteration=1",
        "max_feature_idx=2",
        "feature_names=Column_0 Column_1 Column_2",
        "objective=binary sigmoid:1" if case == 3 else "objective=lambdarank",
    ]
    if case == 4:
        header.append("average_output")
    for tree in range(3):
        prefix = f"case{case}/tree{tree}/"
        if not any(name.startswith(prefix) for name in expected):
            continue
        header.append(f"Tree={tree}")
        for name, value in expected.items():
            if name.startswith(prefix):
                header.append(
                    name.removeprefix(prefix)
                    + "="
                    + " ".join(
                        str(value[index].item()) for index in range(value.numel())
                    ),
                )
    return "\n".join([*header, "end of trees"])


def test_all_trees_missing_directions_and_three_round_predictions() -> None:
    golden = GOLDEN
    expected = read_tensors(golden)
    actual: dict[str, Tensor] = {}
    for case in range(8):
        prefix = f"case{case}"
        model = LightGBMModel.parse(tiny_text(expected, case))
        query = expected[f"{prefix}/query"].numpy()
        actual[f"{prefix}/query"] = torch.from_numpy(query.copy())
        for index, tree in enumerate(model.trees):
            for name, values in tree.columns.items():
                actual[f"{prefix}/tree{index}/{name}"] = torch.from_numpy(values.copy())
        actual[f"{prefix}/prediction"] = torch.from_numpy(model.predict(query))
        actual[f"{prefix}/raw"] = torch.from_numpy(model.predict(query, raw=True))
        actual[f"{prefix}/leaves"] = torch.from_numpy(model.leaf_indices(query))
        for rounds in range(1, 4):
            actual[f"{prefix}/round{rounds}"] = torch.from_numpy(
                model.predict(query, rounds=rounds),
            )
    assert_tensor_golden(
        golden,
        float32_comparands(actual, inputs=("*/query", "*/tree*")),
    )


@pytest.mark.parametrize(
    ("before", "after", "match"),
    [
        ("num_class=1", "num_class=2", "one class"),
        ("num_tree_per_iteration=1", "num_tree_per_iteration=2", "one tree"),
        ("objective=lambdarank", "objective=multiclass", "objective"),
        ("is_linear=0", "is_linear=1", "linear"),
        ("num_leaves=3", "num_leaves=4", "leaf"),
        ("Tree=1", "Tree=9", "ordered"),
        ("version=v4", "version=v4\nversion=v4", "Duplicate"),
        ("max_feature_idx=2", "max_feature_idx=3", "features"),
        ("end of trees", "unrecognized=1\nend of trees", "fields"),
    ],
)
def test_unsupported_or_malformed_tree_assets_fail(
    before: str,
    after: str,
    match: str,
) -> None:
    text = tiny_text(read_tensors(GOLDEN), 0)
    assert before in text
    with pytest.raises(ValueError, match=match):
        _ = LightGBMModel.parse(text.replace(before, after))


def test_prediction_requires_the_complete_feature_schema() -> None:
    model = LightGBMModel.parse(tiny_text(read_tensors(GOLDEN), 0))
    with pytest.raises(ValueError, match="features"):
        # LightGBMModel.predict rejects two columns for its three-feature schema.
        _ = model.predict(np.zeros((2, 2), dtype=np.float64))


def test_split_references_must_use_declared_features() -> None:
    text = tiny_text(read_tensors(GOLDEN), 0)
    line = next(line for line in text.splitlines() if line.startswith("split_feature="))
    with pytest.raises(ValueError, match="undeclared feature"):
        _ = LightGBMModel.parse(text.replace(line, "split_feature=3 0"))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("num_cat", "1", "categorical"),
        ("left_child", "0 0", "repeated"),
        ("left_child", "-100 -2", "leaf reference"),
        ("left_child", "-1 -2", "Disconnected"),
        ("shrinkage", None, "fields"),
    ],
)
def test_tree_topology_and_complete_schema_are_checked(
    field: str,
    value: str | None,
    match: str,
) -> None:
    tree = LightGBMModel.parse(tiny_text(read_tensors(GOLDEN), 0)).trees[0]
    fields = {
        name: " ".join(str(values.item(index)) for index in range(len(values)))
        for name, values in tree.columns.items()
    }
    if value is None:
        del fields[field]
    else:
        fields[field] = value
    with pytest.raises(ValueError, match=match):
        _ = DecisionTree.parse(fields)


def test_binary_overflow_matches_limiting_probabilities() -> None:
    source = LightGBMModel.parse(tiny_text(read_tensors(GOLDEN), 3))
    source.trees[0].floats["leaf_value"][:] = -1000.0
    assert np.array_equal(source.predict(np.zeros((2, 3))), np.zeros(2))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
