"""Canonical tests for ``NarrowEmbedding``.

Regenerate canonical artifacts through pytest so Priml's deterministic setup
applies::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/model/narrow_embedding_test.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import torch

from priml.model.embedding import Embedding
from priml.model.narrow_embedding import NarrowEmbedding
from priml.testing.bfb import assert_bfb_against_golden


_CWD: Final = Path(__file__).resolve().parent


def test_narrow_embedding_config_pprint() -> None:
    config = NarrowEmbedding.Config(
        channels_in=8,
        channels_out=4,
        dtype=torch.bfloat16,
    )
    assert_pprint_golden(
        test_file=__file__,
        name="narrow_embedding",
        config=config,
    )


def test_narrow_embedding_forward_and_open_kwargs() -> None:
    config = NarrowEmbedding.Config(
        channels_in=8,
        channels_out=4,
        dtype=torch.bfloat16,
    )
    module = config.make()

    output = module(torch.tensor([[0, 3, 7]]), message=object())

    assert output.shape == (1, 3, 4)
    assert output.dtype == torch.bfloat16
    assert isinstance(config.inner, Embedding.Config)
    assert config.inner.channels_out == -1
    assert config.inner.channels_in == -1


def test_narrow_embedding_reset_draws_at_float32_then_narrows() -> None:
    module = NarrowEmbedding.Config(
        channels_in=8,
        channels_out=4,
        dtype=torch.bfloat16,
    ).make()
    expected = Embedding.Config(channels_out=4, channels_in=8).make()

    torch.manual_seed(17)
    module.reset_parameters()
    torch.manual_seed(17)
    expected.reset_parameters()

    assert module.inner.weight.dtype == torch.bfloat16
    assert torch.equal(module.inner.weight, expected.weight.bfloat16())


def _run_embedding(module: nn.Module, tokens: Tensor) -> Tensor:
    assert isinstance(module, NarrowEmbedding)
    return module(tokens, message=object()).float()


def test_narrow_embedding_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="narrow_embedding",
        build_module=lambda: NarrowEmbedding.Config(
            channels_in=8,
            channels_out=4,
            dtype=torch.bfloat16,
        ).make(),
        build_input=lambda: torch.tensor([[0, 3, 7]]),
        seed=0,
        run=_run_embedding,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
