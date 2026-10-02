"""Tests for the blog-post launcher."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.arcagi1.scripts.reproduce_blog_post import overlay, recipe
from priml.baselines.arcagi1.train_step import TrmTrainStep
from priml.baselines.arcagi1.train_step_test import port_config
from priml.baselines.sudoku.act import AtomicPool
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.lib.custom_json import DictCodec


@pytest.mark.parametrize("single_gpu", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_recipe_finalizes(*, single_gpu: bool, checkpoint: bool) -> None:
    config = recipe(single_gpu=single_gpu, checkpoint=checkpoint).finalize()
    assert config.max_steps == 280_000
    assert config.eval_only == checkpoint
    pool = config.step.pool
    prefix = config.step.model.prefix
    assert isinstance(pool, AtomicPool.Config)
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    assert pool.batch_size == prefix.batch_size == config.dataset.batch_size
    assert pool.batch_size == (8 if single_gpu else 96)


def test_historical_names_land_on_the_recipes_state() -> None:
    """Every archive tensor, renamed, is one the exp008 model holds."""
    step = TrmTrainStep(port_config("exp008").finalize())
    names = [
        "embed_tokens.weight",
        "embed_feedback",
        "q_head.weight",
        "q_head.bias",
        "puzzle_emb.weights",
    ]
    tensors = {name: torch.zeros(1) for name in names}
    archive: dict[str, object] = {"step": {"model": tensors, "ema": tensors}}
    state = DictCodec.coerce(overlay(archive, {"step": {}})["step"])
    renamed = DictCodec.coerce(state["model"], torch.Tensor)
    assert set(renamed) <= set(step.model.state_dict())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
