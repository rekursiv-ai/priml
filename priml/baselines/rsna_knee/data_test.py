"""The knee finding names shared by models and scores."""

from __future__ import annotations

from priml.baselines.rsna_knee.data import FINDINGS


def test_findings_are_the_twelve_ordered_distinct_names() -> None:
    assert FINDINGS == (
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    )
    assert len(set(FINDINGS)) == len(FINDINGS)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
