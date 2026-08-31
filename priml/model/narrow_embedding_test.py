"""Canonical tests for ``NarrowEmbedding``.

Regenerate canonical artifacts through pytest so Priml's deterministic setup
applies::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/model/narrow_embedding_test.py
"""

from __future__ import annotations

from pathlib import Path

from configgle.testing import assert_pprint_golden

import torch

from priml.model.embedding import Embedding
from priml.model.narrow_embedding import NarrowEmbedding
from priml.testing.bfb import assert_bfb_against_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def test_narrow_embedding_config_pprint() -> None:
    config = NarrowEmbedding.Config(
        torch.bfloat16,
        channels_out=4,
        num_embeddings=8,
    )
    assert_pprint_golden(
        test_file=__file__,
        name="narrow_embedding",
        config=config,
    )


def test_narrow_embedding_forward_and_open_kwargs() -> None:
    config = NarrowEmbedding.Config(
        torch.bfloat16,
        channels_out=4,
        num_embeddings=8,
    )
    module = config.make()

    output = module(torch.tensor([[0, 3, 7]]), message=object())

    assert output.shape == (1, 3, 4)
    assert output.dtype == torch.bfloat16
    assert isinstance(config.inner, Embedding.Config)
    assert config.inner.channels_out == -1
    assert config.inner.num_embeddings == -1


def test_narrow_embedding_reset_draws_at_float32_then_narrows() -> None:
    module = NarrowEmbedding.Config(
        torch.bfloat16,
        channels_out=4,
        num_embeddings=8,
    ).make()
    expected = Embedding.Config(channels_out=4, num_embeddings=8).make()

    torch.manual_seed(17)
    module.reset_parameters()
    torch.manual_seed(17)
    expected.reset_parameters()

    assert module.inner.weight.dtype == torch.bfloat16
    assert torch.equal(module.inner.weight, expected.weight.bfloat16())


def test_narrow_embedding_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="narrow_embedding",
        build_module=lambda: NarrowEmbedding.Config(
            torch.bfloat16,
            channels_out=4,
            num_embeddings=8,
        ).make(),
        build_input=lambda: torch.tensor([[0, 3, 7]]),
        seed=0,
        run=lambda module, tokens: module(tokens, message=object()).float(),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
