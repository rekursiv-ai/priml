"""Independent evaluation of the released CASMI LightGBM model data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import math

from numpy.typing import NDArray

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionTree:
    """One complete external tree record, retaining every serialized numeric field."""

    integers: dict[str, NDArray[np.int64]]
    floats: dict[str, NDArray[np.float64]]

    @property
    def columns(self) -> dict[str, NDArray[np.int64] | NDArray[np.float64]]:
        """Return all columns, including training statistics unused by inference."""
        return {**self.integers, **self.floats}

    @classmethod
    def parse(cls, fields: dict[str, str]) -> Self:
        """Parse a complete tree with exact numeric types and validate its topology.

        Args:
          fields: Serialized numeric fields from one tree.

        Returns:
          tree: Complete checked predictor state.

        """
        integer_names = {
            "num_leaves",
            "num_cat",
            "split_feature",
            "decision_type",
            "left_child",
            "right_child",
            "leaf_count",
            "internal_count",
            "is_linear",
            "cat_boundaries",
            "cat_threshold",
        }
        float_names = {
            "split_gain",
            "threshold",
            "leaf_value",
            "leaf_weight",
            "internal_value",
            "internal_weight",
            "shrinkage",
        }
        unknown = fields.keys() - integer_names - float_names
        required = (integer_names - {"cat_boundaries", "cat_threshold"}) | float_names
        if unknown or required - fields.keys():
            raise ValueError(
                f"Unsupported or missing tree fields: {unknown | (required - fields.keys())}",
            )
        integers = {
            name: np.array([int(value) for value in text.split()], dtype=np.int64)
            for name, text in fields.items()
            if name in integer_names
        }
        floats = {
            name: np.array([float(value) for value in text.split()], dtype=np.float64)
            for name, text in fields.items()
            if name in float_names
        }
        linear = integers["is_linear"].item()
        if linear != 0:
            raise ValueError("Unsupported linear leaf model.")
        leaves = integers["num_leaves"].item()
        if (
            leaves < 1
            or any(
                len(integers[name]) != leaves - 1
                for name in (
                    "split_feature",
                    "decision_type",
                    "left_child",
                    "right_child",
                    "internal_count",
                )
            )
            or any(
                len(floats[name]) != leaves - 1
                for name in (
                    "threshold",
                    "split_gain",
                    "internal_value",
                    "internal_weight",
                )
            )
            or len(floats["leaf_value"]) != leaves
            or len(floats["leaf_weight"]) not in ({0, 1} if leaves == 1 else {leaves})
            or len(integers["leaf_count"]) != leaves
        ):
            sizes = {
                name: len(values) for name, values in {**integers, **floats}.items()
            }
            raise ValueError(
                f"Inconsistent leaf ({leaves}) and internal-node column lengths: {sizes}",
            )
        tree = cls(integers=integers, floats=floats)
        tree.validate_topology()
        category_count = integers["num_cat"].item()
        if category_count and not {
            "cat_boundaries",
            "cat_threshold",
        }.issubset(integers):
            raise ValueError("Missing categorical bitset fields.")
        return tree

    def validate_topology(self) -> None:
        """Reject cycles, disconnected branches and invalid child references."""
        count = len(self.integers["split_feature"])
        pending = [0] if count else []
        seen: set[int] = set()
        while pending:
            node = pending.pop()
            if node in seen or node < 0 or node >= count:
                raise ValueError("Invalid or repeated internal node reference.")
            seen.add(node)
            for name in ("left_child", "right_child"):
                child = self.integers[name].item(node)
                if child >= 0:
                    pending.append(child)
                elif ~child < 0 or ~child > count:
                    raise ValueError("Invalid leaf reference.")
        if len(seen) != count:
            raise ValueError("Disconnected internal nodes.")

    def leaves(self, data: NDArray[np.float64]) -> NDArray[np.int32]:
        """Traverse numeric and categorical branches with their fitted missing rules.

        Args:
          data: Complete numerical feature rows.

        Returns:
          leaves: Selected zero-based leaf index per row.

        """
        if not len(self.integers["split_feature"]):
            return np.zeros(len(data), dtype=np.int32)
        positions = np.zeros(len(data), dtype=np.int64)
        while bool(np.any(positions >= 0)):
            rows = np.flatnonzero(positions >= 0)
            nodes = positions[rows]
            values = data[rows, self.integers["split_feature"][nodes]]
            decision = self.integers["decision_type"][nodes]
            categorical = np.not_equal(decision & 1, 0)
            missing_type = (decision >> 2) & 3
            numeric = np.where(
                np.isnan(values) & np.not_equal(missing_type, 2),
                0.0,
                values,
            )
            missing = (
                np.equal(missing_type, 1)
                & (np.abs(numeric) <= float(np.float32(1e-35)))
            ) | (np.equal(missing_type, 2) & np.isnan(numeric))
            left = np.where(
                missing,
                np.not_equal(decision & 2, 0),
                numeric <= self.floats["threshold"][nodes],
            )
            categories = np.flatnonzero(categorical)
            for index in range(len(categories)):
                row = categories.item(index)
                value = values.item(row)
                left[row] = False
                if math.isnan(value) or int(value) < 0:
                    continue
                node = nodes.item(row)
                category = int(value)
                group = int(self.floats["threshold"].item(node))
                start = self.integers["cat_boundaries"].item(group)
                stop = self.integers["cat_boundaries"].item(group + 1)
                word = category // 32
                if word < stop - start:
                    left[row] = bool(
                        (
                            self.integers["cat_threshold"].item(start + word)
                            >> (category % 32)
                        )
                        & 1,
                    )
            positions[rows] = np.where(
                left,
                self.integers["left_child"][nodes],
                self.integers["right_child"][nodes],
            )
        return (~positions).astype(np.int32)


@dataclass(frozen=True, slots=True, kw_only=True)
class LightGBMModel:
    """A released one-output model with ordered trees and its declared feature schema."""

    features: tuple[str, ...]
    trees: tuple[DecisionTree, ...]
    objective: str
    sigmoid: float
    average: bool

    @classmethod
    def parse(cls, text: str) -> Self:
        """Read ordinary model text as data and retain complete tree records.

        Args:
          text: Released LightGBM text serialization.

        Returns:
          model: Checked native predictor without any external booster runtime.

        """
        header: dict[str, str] = {}
        records: list[dict[str, str]] = []
        average = False
        for line in text.splitlines():
            if line == "end of trees":
                break
            if line == "average_output":
                average = True
            name, separator, value = line.partition("=")
            if not separator:
                continue
            if name == "Tree":
                if int(value) != len(records):
                    raise ValueError("Tree indices must be contiguous and ordered.")
                records.append({})
            else:
                target = records[-1] if records else header
                if name in target:
                    raise ValueError(f"Duplicate model field: {name}")
                target[name] = value
        if header.get("num_class") != "1":
            raise ValueError("Native CASMI inference requires one class output.")
        if header.get("num_tree_per_iteration") != "1":
            raise ValueError("Native CASMI inference requires one tree per update.")
        objective, *options = header.get("objective", "").split()
        if objective not in {"lambdarank", "binary", "regression"}:
            raise ValueError(f"Unsupported objective: {objective}")
        parameters = dict(option.split(":", 1) for option in options)
        features = tuple(header["feature_names"].split())
        if len(features) != int(header["max_feature_idx"]) + 1 or not records:
            raise ValueError("Incomplete features or trees in model asset.")
        trees = tuple(DecisionTree.parse(record) for record in records)
        if any(
            bool(
                np.any(tree.integers["split_feature"] < 0)
                or np.any(tree.integers["split_feature"] >= len(features)),
            )
            for tree in trees
        ):
            raise ValueError("Split references an undeclared feature.")
        return cls(
            features=features,
            trees=trees,
            objective=objective,
            sigmoid=float(parameters.get("sigmoid", "1")),
            average=average,
        )

    def selected_trees(
        self,
        data: NDArray[np.float64],
        rounds: int,
    ) -> tuple[DecisionTree, ...]:
        """Check feature width and retain the requested prefix of boosting updates."""
        if data.ndim != 2 or data.shape[1] != len(self.features):
            raise ValueError(f"Expected {len(self.features)} features.")
        return self.trees[:rounds] if rounds > 0 else self.trees

    def leaf_indices(self, data: NDArray[np.float64]) -> NDArray[np.int32]:
        """Return one leaf index per input row and fitted tree."""
        return np.column_stack(
            [tree.leaves(data) for tree in self.selected_trees(data, -1)],
        )

    def predict(
        self,
        data: NDArray[np.float64],
        *,
        rounds: int = -1,
        raw: bool = False,
    ) -> NDArray[np.float64]:
        """Accumulate leaves in model order, then apply the declared output transform.

        Args:
          data: Complete numerical feature rows.
          rounds: Positive prefix length, or all trees for a nonpositive value.
          raw: Return the untransformed ordered leaf sum.

        Returns:
          predictions: Ranking values or binary probabilities.

        """
        trees = self.selected_trees(data, rounds)
        output = np.zeros(len(data))
        for tree in trees:
            output += tree.floats["leaf_value"][tree.leaves(data)]
        if not raw:
            if self.average:
                output /= len(trees)
            if self.objective == "binary":
                for row in range(len(output)):
                    logit = -self.sigmoid * output.item(row)
                    try:
                        exponential = math.exp(logit)
                    except OverflowError:
                        exponential = math.inf
                    output[row] = 1.0 / (1.0 + exponential)
        return output
