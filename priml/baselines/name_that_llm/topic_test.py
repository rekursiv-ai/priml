"""Recover unseen prompt groups without touching prediction labels."""

from __future__ import annotations

from typing import cast

import numpy as np

from priml.baselines.name_that_llm.topic import TopicClusters, ValidationGroups


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
    assert TopicClusters.Config().make()(
        np.array(["single"], dtype=object),
    ).tolist() == [1]


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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
