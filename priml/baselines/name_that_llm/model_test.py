"""Production numerical paths at small row and embedding dimensions."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from priml.baselines.name_that_llm.model import (
    QwenHeads,
    StyleCounts,
    center,
    random_chunks,
)
from priml.baselines.name_that_llm.testing import (
    centred_branch,
    miniature,
    miniature_split,
    plain_branch,
    recipe,
    svm_branch,
)


def test_style_features_include_case_unicode_and_empty_passages() -> None:
    actual = StyleCounts().transform(["A\N{EM DASH}b!\n\nC...", ""])
    assert actual.shape == (2, 25)
    assert actual.dtype == np.float64
    assert actual[0, 2] == 2 / len("A\N{EM DASH}b!\n\nC...") * 100
    assert actual[0, 3:6].tolist() == [1, 1, 0]
    assert actual[0, 18] == 2 / len("A\N{EM DASH}b!\n\nC...")
    assert actual[0, 21:].tolist() == [1, 0, 0, 1]
    assert np.count_nonzero(cast("np.ndarray", actual[1])) == 0


def test_style_features_count_every_column() -> None:
    actual = StyleCounts().transform(
        [
            "Abc DEF\n\n« • ;:,'’“\"*#( - !?…...\N{EM DASH}–",
            "one\tTWO  three\r\nFour",
            "",
        ],
    )
    # 34 and 20 characters over 7 and 4 words; "..." holds the only three periods
    # and three of the five sentence marks.
    first = [np.log(35), np.log(8), 2 / 34 * 100, 1, 1, 1, 1, 3, *[1] * 10]
    second = [np.log(21), np.log(5), 1 / 20 * 100, *[0] * 15]
    expected = np.array(
        [
            [*first, 4 / 34, 27 / 7, 5 / 7, *[1] * 4],
            [*second, 4 / 20, 15 / 4, *[0] * 5],
            [0] * 25,
        ],
    )
    assert np.array_equal(actual, a2=expected)


def test_center_preserves_input_and_uses_each_group_mean() -> None:
    features = np.arange(30, dtype=np.float32).reshape(6, 5)
    original = features.copy()
    actual = center(features, groups=np.array(["b", "a", "b", "a", "c", "c"]))
    assert np.array_equal(features, a2=original)
    assert np.array_equal(
        actual,
        a2=np.array(
            [
                [-5] * 5,
                [-5] * 5,
                [5] * 5,
                [5] * 5,
                [-2.5] * 5,
                [2.5] * 5,
            ],
            dtype=np.float32,
        ),
    )


def test_chunks_consume_one_continuing_generator() -> None:
    rng = np.random.default_rng(42)
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    first = random_chunks(groups, size=2, rng=rng)
    second = random_chunks(groups, size=3, rng=rng)
    assert first.tolist() == ["a_1", "a_0", "a_0", "b_0", "b_1", "b_0"]
    assert second.tolist() == ["a_0"] * 3 + ["b_0"] * 3


@pytest.mark.parametrize("size", argvalues=[0, -1, -5])
def test_invalid_chunk_sizes_leave_the_random_stream_unchanged(size: int) -> None:
    random = np.random.default_rng(42)
    before = random.bit_generator.state
    with pytest.raises(ValueError, match="positive chunk size"):
        random_chunks(
            np.array(["a", "a", "a", "b", "b", "b"]),
            size=size,
            rng=random,
        )
    assert random.bit_generator.state == before


def test_blend_uses_all_three_branches_and_clips_probabilities() -> None:
    blend = recipe().blend.make()
    svm = np.array([[0.0, 0.4, 0.6], [0.6, 0.3, 0.1]])
    plain = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    centred = np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]])
    assert np.array_equal(
        blend({"svm": svm, "plain": plain, "centred": centred}),
        a2=0.4 * np.log(np.clip(svm, a_min=1e-12, a_max=1))
        + 0.3 * np.log(np.clip(plain, a_min=1e-12, a_max=1))
        + 0.3 * np.log(np.clip(centred, a_min=1e-12, a_max=1)),
    )


@pytest.mark.parametrize("branches", argvalues=[["svm", "plain"], ["svm", "x", "y"]])
def test_blend_requires_one_weight_per_branch(branches: list[str]) -> None:
    blend = recipe().blend.make()
    probabilities = dict.fromkeys(branches, np.full((2, 3), 1 / 3))
    with pytest.raises(ValueError, match="one blend weight for each"):
        blend(probabilities)


def test_plain_heads_fit_every_layer() -> None:
    model = plain_branch().make()
    probabilities = model(miniature_split(), {})
    # The miniature predicts three rows over three classes.
    assert probabilities.shape == (3, 3)
    assert len(model.fitted) == 4


def test_heads_require_a_saved_layer() -> None:
    with pytest.raises(ValueError, match="at least one saved layer"):
        QwenHeads.Config().make()


def test_centered_heads_fit_all_chunks_and_keep_singletons() -> None:
    model = centred_branch().make()
    _, targets, groups, features = miniature()
    # Four rows over the miniature's three classes; cluster 3 is a singleton.
    plain = np.arange(12, dtype=np.float64).reshape(4, 3) / 48
    probabilities = model.predict(
        features,
        targets=targets,
        groups=groups,
        predict=features[:4],
        clusters=np.array([2, 2, 2, 3]),
        plain=plain,
    )
    assert probabilities.shape == (4, 3)
    assert len(model.heads.fitted) == 4
    assert np.array_equal(
        cast("np.ndarray", probabilities[3]),
        a2=cast("np.ndarray", plain[3]),
    )
    assert not np.array_equal(probabilities[:3], a2=plain[:3])


def test_centered_heads_read_their_fallback_from_the_earlier_branches() -> None:
    centred = centred_branch()
    centred.fallback = "other"
    with pytest.raises(KeyError, match="other"):
        centred.make()(miniature_split(), {"plain": np.zeros((2, 3))})


def test_svm_calibration_uses_prompt_groups() -> None:
    model = svm_branch().make()
    probabilities = model(miniature_split(), {})
    assert probabilities.shape == (3, 3)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
