"""Checks for the boosting module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import numpy as np
import pytest

from priml.baselines.casmi2026.boosting import (
    DecisionTree,
    LightGBMModel,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def numeric_tree() -> dict[str, str]:
    """Split feature 0 at 0.5 (NaN as zero), then feature 1 at 1.5 (NaN right)."""
    return {
        "num_leaves": "3",
        "num_cat": "0",
        "split_feature": "0 1",
        "split_gain": "1 1",
        "threshold": "0.5 1.5",
        "decision_type": "2 8",
        "left_child": "1 -1",
        "right_child": "-3 -2",
        "leaf_value": "0.25 -0.5 1",
        "leaf_weight": "1 1 1",
        "leaf_count": "1 1 1",
        "internal_value": "0 0",
        "internal_weight": "3 2",
        "internal_count": "3 2",
        "is_linear": "0",
        "shrinkage": "1",
    }


def zero_tree() -> dict[str, str]:
    """Send zero, subnormal and NaN feature-1 values left, every other value right."""
    return {
        "num_leaves": "2",
        "num_cat": "0",
        "split_feature": "1",
        "split_gain": "1",
        "threshold": "-1",
        "decision_type": "6",
        "left_child": "-1",
        "right_child": "-2",
        "leaf_value": "0.125 -0.125",
        "leaf_weight": "1 1",
        "leaf_count": "1 1",
        "internal_value": "0",
        "internal_weight": "2",
        "internal_count": "2",
        "is_linear": "0",
        "shrinkage": "1",
    }


def categorical_tree() -> dict[str, str]:
    """Send feature-2 categories 0 and 2 left through a one-word bitset."""
    return {
        "num_leaves": "2",
        "num_cat": "1",
        "split_feature": "2",
        "split_gain": "1",
        "threshold": "0",
        "decision_type": "1",
        "left_child": "-1",
        "right_child": "-2",
        "leaf_value": "2 -2",
        "leaf_weight": "1 1",
        "leaf_count": "1 1",
        "internal_value": "0",
        "internal_weight": "2",
        "internal_count": "2",
        "is_linear": "0",
        "shrinkage": "1",
        "cat_boundaries": "0 1",
        "cat_threshold": "5",
    }


def leaf_tree(value: str) -> dict[str, str]:
    """Return a single-leaf tree, whose internal columns are empty."""
    return {
        "num_leaves": "1",
        "num_cat": "0",
        "split_feature": "",
        "split_gain": "",
        "threshold": "",
        "decision_type": "",
        "left_child": "",
        "right_child": "",
        "leaf_value": value,
        "leaf_weight": "",
        "leaf_count": "1",
        "internal_value": "",
        "internal_weight": "",
        "internal_count": "",
        "is_linear": "0",
        "shrinkage": "1",
    }


def stump(feature: int, *, left: float, right: float) -> dict[str, str]:
    """Send a feature at or below zero to the left leaf, every other value right."""
    return {
        **zero_tree(),
        "split_feature": str(feature),
        "threshold": "0",
        "decision_type": "2",
        "leaf_value": f"{left} {right}",
    }


def model_text(
    *trees: dict[str, str],
    features: Sequence[str] = ("Column_0", "Column_1", "Column_2"),
    objective: str = "lambdarank",
    average: bool = False,
) -> str:
    """Serialize trees over the named features in the released text layout."""
    lines = [
        "tree",
        "version=v4",
        "num_class=1",
        "num_tree_per_iteration=1",
        f"max_feature_idx={len(features) - 1}",
        f"feature_names={' '.join(features)}",
        f"objective={objective}",
    ]
    if average:
        lines.append("average_output")
    for index, tree in enumerate(trees):
        lines.append(f"Tree={index}")
        lines.extend(f"{name}={value}" for name, value in tree.items())
    return "\n".join([*lines, "end of trees"])


def query() -> NDArray[np.float64]:
    """Reach every numeric, zero-missing and categorical direction at least once."""
    nan = math.nan
    return np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0],
            [nan, nan, nan],
            [1.0, 1e-36, -1.0],
            [0.25, 3.0, 40.0],
        ],
    )


def test_every_split_direction_reaches_its_hand_derived_leaf() -> None:
    model = LightGBMModel.parse(
        model_text(
            numeric_tree(),
            zero_tree(),
            categorical_tree(),
            leaf_tree("0.0625"),
        ),
    )
    assert model.features == ("Column_0", "Column_1", "Column_2")
    assert (model.objective, model.sigmoid, model.average) == ("lambdarank", 1.0, False)
    assert model.trees[0].columns.keys() == numeric_tree().keys()
    leaves = model.leaf_indices(query())
    assert leaves.dtype == np.int32
    assert leaves.tolist() == [
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [2, 0, 0, 0],
        [1, 0, 1, 0],
        [2, 0, 1, 0],
        [1, 1, 1, 0],
    ]
    expected = [
        0.25 - 0.125 + 2 + 0.0625,
        -0.5 - 0.125 - 2 + 0.0625,
        1 + 0.125 + 2 + 0.0625,
        -0.5 + 0.125 - 2 + 0.0625,
        1 + 0.125 - 2 + 0.0625,
        -0.5 - 0.125 - 2 + 0.0625,
    ]
    assert model.predict(query()).tolist() == expected
    assert model.predict(query(), raw=True).tolist() == expected
    assert model.predict(query(), rounds=0).tolist() == expected
    assert model.predict(query(), rounds=1).tolist() == [0.25, -0.5, 1, -0.5, 1, -0.5]
    assert model.predict(query(), rounds=2).tolist() == [
        0.125,
        -0.625,
        1.125,
        -0.375,
        1.125,
        -0.625,
    ]


def test_averaged_output_divides_the_selected_rounds_only_when_transformed() -> None:
    model = LightGBMModel.parse(model_text(numeric_tree(), zero_tree(), average=True))
    assert model.average
    assert model.predict(query()).tolist() == [
        0.0625,
        -0.3125,
        0.5625,
        -0.1875,
        0.5625,
        -0.3125,
    ]
    assert model.predict(query(), rounds=1).tolist() == [0.25, -0.5, 1, -0.5, 1, -0.5]
    assert model.predict(query(), raw=True).tolist() == [
        0.125,
        -0.625,
        1.125,
        -0.375,
        1.125,
        -0.625,
    ]


def test_binary_probabilities_use_the_declared_sigmoid_and_saturate() -> None:
    model = LightGBMModel.parse(
        model_text(leaf_tree("0.25"), objective="binary sigmoid:2"),
    )
    assert model.sigmoid == 2.0
    assert model.predict(np.zeros((2, 3))).tolist() == [1 / (1 + math.exp(-0.5))] * 2
    assert model.predict(np.zeros((2, 3)), raw=True).tolist() == [0.25, 0.25]
    for leaf, probability in (("-1000", 0.0), ("1000", 1.0), ("0", 0.5)):
        saturated = LightGBMModel.parse(
            model_text(leaf_tree(leaf), objective="binary sigmoid:2"),
        )
        assert saturated.predict(np.zeros((2, 3))).tolist() == [probability] * 2
    regression = LightGBMModel.parse(model_text(leaf_tree("3"), objective="regression"))
    assert regression.predict(np.zeros((2, 3))).tolist() == [3.0, 3.0]


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
    text = model_text(numeric_tree(), zero_tree())
    assert before in text
    with pytest.raises(ValueError, match=match):
        _ = LightGBMModel.parse(text.replace(before, after))


def test_an_asset_without_trees_is_incomplete() -> None:
    header = model_text(numeric_tree()).partition("Tree=0")[0]
    with pytest.raises(ValueError, match="features or trees"):
        _ = LightGBMModel.parse(header)


def test_prediction_requires_the_complete_feature_schema() -> None:
    model = LightGBMModel.parse(model_text(numeric_tree()))
    with pytest.raises(ValueError, match="features"):
        # LightGBMModel.predict rejects two columns for its three-feature schema.
        _ = model.predict(np.zeros((2, 2), dtype=np.float64))


def test_split_references_must_use_declared_features() -> None:
    tree = numeric_tree()
    tree["split_feature"] = "3 0"
    with pytest.raises(ValueError, match="undeclared feature"):
        _ = LightGBMModel.parse(model_text(tree))


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
    fields = numeric_tree()
    if value is None:
        del fields[field]
    else:
        fields[field] = value
    with pytest.raises(ValueError, match=match):
        _ = DecisionTree.parse(fields)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
