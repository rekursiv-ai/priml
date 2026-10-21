"""Recover unseen prompt groups without touching prediction labels."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from priml.baselines.name_that_llm.topic import (
    TopicClusters,
    ValidationGroups,
    average_linkage,
    distance_clusters,
)


def test_topic_clusters_separate_disjoint_topics() -> None:
    clusters = TopicClusters.Config().make()
    actual = clusters(
        np.array(
            [
                "cats purr furry pets",
                "cats purr furry pets",
                "quasars orbit cosmic radiation",
                "quasars orbit cosmic radiation",
            ],
            dtype=object,
        ),
    )
    assert actual[0] == actual[1]
    assert actual[2] == actual[3]
    assert actual[0] != actual[2]


def test_single_passage_keeps_reference_singleton_cluster() -> None:
    # TopicClusters.__call__ explicitly handles prediction sets smaller than two.
    actual = TopicClusters.Config().make()(np.array(["single"], dtype=object))
    assert actual.dtype == np.int64
    assert actual.tolist() == [1]


def test_cosine_rounded_above_one_is_distance_zero() -> None:
    passage = "zebra quasar nebula nebula"
    actual = TopicClusters.Config().make()(
        np.array(["flour", "flour", passage, passage], dtype=object),
    )
    # The repeated passage's cosine rounds to 1 + 2**-52. Unfloored, its pair
    # would merge below the exact pair and take the first cluster number.
    assert actual.tolist() == [1, 1, 2, 2]


def test_two_unrelated_passages_form_two_topics() -> None:
    actual = TopicClusters.Config().make()(
        np.array(["cats purr", "quasars orbit"], dtype=object),
    )
    assert actual.dtype == np.int32
    assert actual.tolist() == [1, 2]


def test_validation_groups_chunk_each_topic_with_the_seeded_generator() -> None:
    texts = np.array(
        ["cats purr furry pets"] * 3 + ["quasars orbit cosmic radiation"] * 3,
        dtype=object,
    )
    config = ValidationGroups.Config()
    config.chunk_size = 2
    config.seed = 3
    groups = config.make()(texts)
    _, counts = np.unique(groups, return_counts=True)
    assert sorted(cast("list[int]", counts.tolist())) == [1, 1, 2, 2]
    assert (
        len({groups[0], groups[1], groups[2]} & {groups[3], groups[4], groups[5]}) == 0
    )
    assert groups.tolist() == config.make()(texts).tolist()
    config.seed = 4
    assert groups.tolist() != config.make()(texts).tolist()


@pytest.mark.parametrize(
    ("condensed", "expected"),
    argvalues=[
        # All ties: each chain restarts at the lowest live cluster.
        ([1] * 6, [[0, 1, 1, 2], [2, 4, 1, 3], [3, 5, 1, 4]]),
        # Points 0, 2, 10, 11: the far pair merges first, so sorting relabels.
        ([2, 10, 11, 8, 9, 1], [[2, 3, 1, 2], [0, 1, 2, 2], [4, 5, 9.5, 4]]),
        # Leaf 3 is as near leaf 1 as its previous link 2; the chain keeps 2.
        ([5, 3, 4, 2, 1, 1], [[2, 3, 1, 2], [1, 4, 1.5, 3], [0, 5, 4, 4]]),
    ],
)
def test_average_linkage_keeps_the_chain_tie_order(
    condensed: list[float],
    expected: list[list[float]],
) -> None:
    leaves = len(expected) + 1
    # Only the upper triangle is read, so the rest holds a value that would win.
    distances = np.full((leaves, leaves), -1.0)
    distances[np.triu_indices(leaves, k=1)] = condensed
    assert average_linkage(distances).tolist() == expected


def test_average_linkage_rounds_weighted_products_before_adding() -> None:
    a, b = 0.29, 0.66
    # average_linkage takes a square distance matrix.
    distances = np.full((7, 7), 0.25)
    distances[:3, :3] = distances[3:6, 3:6] = 0.125
    distances[:3, 6] = a
    distances[3:6, 6] = b
    tree = average_linkage(distances)
    # Fusing either product into the sum ends two units in the last place higher.
    height = (3 * ((2 * a + a) / 3) + 3 * ((2 * b + b) / 3)) / 6
    assert tree.tolist() == [
        [0, 1, 0.125, 2],
        [2, 7, 0.125, 3],
        [3, 4, 0.125, 2],
        [5, 9, 0.125, 3],
        [8, 10, 0.25, 6],
        [6, 11, height, 7],
    ]


@pytest.mark.parametrize(
    ("tree", "threshold", "expected"),
    argvalues=[
        # Leaf children are numbered after their sibling subtree.
        ([[0, 1, 1, 2], [2, 4, 1, 3], [3, 5, 1, 4]], 0.5, [1, 2, 3, 4]),
        ([[0, 1, 1, 2], [2, 4, 1, 3], [3, 5, 1, 4]], 1, [1, 1, 1, 1]),
        # A merge exactly at the threshold stays inside one cluster.
        ([[2, 3, 1, 2], [0, 1, 2, 2], [4, 5, 9.5, 4]], 1, [2, 3, 1, 1]),
        # A low root above a higher merge still splits at the higher one.
        ([[0, 1, 2, 2], [2, 3, 1, 3]], 1.5, [1, 2, 3]),
    ],
)
def test_distance_clusters_number_flat_clusters_depth_first(
    tree: list[list[float]],
    threshold: float,
    expected: list[int],
) -> None:
    labels = distance_clusters(np.array(tree, dtype=np.float64), threshold=threshold)
    assert labels.dtype == np.int32
    assert labels.tolist() == expected


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
