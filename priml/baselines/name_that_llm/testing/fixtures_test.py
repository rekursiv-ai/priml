"""The shared fixtures expose the recipe and the captured miniature unchanged."""

from __future__ import annotations

import numpy as np

from priml.baselines.name_that_llm.testing.fixtures_support_test import (
    centred_branch,
    miniature,
    plain_branch,
    recipe,
)


def test_recipe_is_exp000_with_saved_layers_resolved() -> None:
    config = recipe()
    assert config.experiment_name == "exp000"
    assert plain_branch().layers == [0, 1, 2, 3]
    assert centred_branch().heads.layers == [0, 1, 2, 3]


def test_miniature_has_three_classes_in_six_prompt_groups() -> None:
    texts, targets, groups, embeddings = miniature()
    assert len(texts) == len(targets) == len(groups) == len(embeddings) == 18
    assert np.array_equal(np.unique(targets), [0, 1, 2])
    assert len(np.unique(groups)) == 6
    assert embeddings.shape == (18, 4, 2)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
