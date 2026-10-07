"""Canonical tests for ``NarrowEmbedding``.

Regenerate canonical artifacts through pytest so Priml's deterministic setup
applies::

    uv --quiet run --frozen pytest priml/model/narrow_embedding_test.py --regenerate-b4b
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
from unittest.mock import patch

from torch import Tensor, nn

import pytest
import torch

from priml.cost import Cost, cost
from priml.model.embedding import Embedding
from priml.model.narrow_embedding import NarrowEmbedding
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


_CWD: Final = Path(__file__).resolve().parent


def test_narrow_embedding_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="narrow_embedding",
        config=NarrowEmbedding.Config(
            channels_in=2,
            channels_out=4,
            dtype=torch.bfloat16,
        ),
    )


def test_narrow_embedding_forward_and_open_kwargs() -> None:
    config = NarrowEmbedding.Config(channels_in=2, channels_out=4, dtype=torch.bfloat16)
    module = config.make()

    output = module(torch.tensor([[0, 1, 0], [1, 0, 1]]), message=object())

    assert output.shape == (2, 3, 4)
    assert output.dtype == torch.bfloat16
    assert isinstance(config.inner, Embedding.Config)
    assert config.inner.channels_out == -1
    assert config.inner.channels_in == -1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_narrow_embedding_reset_draws_at_float32_then_narrows(
    dtype: torch.dtype,
) -> None:
    module = NarrowEmbedding.Config(
        channels_in=2,
        channels_out=4,
        dtype=dtype,
    ).make()
    expected = Embedding.Config(channels_out=4, channels_in=2).make()

    torch.manual_seed(17)
    module.reset_parameters()
    torch.manual_seed(17)
    expected.reset_parameters()

    assert module.inner.weight.dtype == dtype
    assert torch.equal(module.inner.weight, expected.weight.to(dtype=dtype))


def test_narrow_embedding_reset_casts_before_and_after_drawing() -> None:
    module = NarrowEmbedding.Config(
        channels_in=2,
        channels_out=4,
        dtype=torch.bfloat16,
    ).make()
    with patch.object(module.inner, "to", wraps=module.inner.to) as to:
        module.reset_parameters()

    assert [call.kwargs for call in to.call_args_list] == [
        {"dtype": torch.float32},
        {"dtype": torch.bfloat16},
    ]


def _embed_float(module: nn.Module, tokens: Tensor) -> Tensor:
    """Look tokens up and widen, so the harness can reduce a narrowed table."""
    assert isinstance(module, NarrowEmbedding)
    return module(tokens).float()


def test_narrow_embedding_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="narrow_embedding",
        build_module=NarrowEmbedding.Config(
            channels_in=2,
            channels_out=4,
            dtype=torch.bfloat16,
        ).make,
        build_input=lambda: torch.tensor([[0, 1, 0], [1, 0, 1]]),
        seed=0,
        run=_embed_float,
    )


def test_narrow_embedding_cost_is_the_inner_gather() -> None:
    """Narrowing moves the table; it adds no parameters and no arithmetic."""
    config = NarrowEmbedding.Config(channels_in=2, channels_out=4, dtype=torch.bfloat16)
    seq_len, batch_size = 3, 2
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, config.channels_in, (batch_size, seq_len)),
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=None,
        run=_embed_float,
    )
    assert model_cost == cost(
        config.copy_tree().finalize().inner,
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=torch.bfloat16,
    )
    bf16, i64 = torch.bfloat16, torch.int64
    assert model_cost == Cost(
        cells={
            ("flops", "adjoint", "selection", bf16): batch_size
            * seq_len
            * config.channels_out,
            ("bytes", "primal", "selection", i64): batch_size * seq_len * 8,
            ("bytes", "primal", "selection", bf16): batch_size
            * seq_len
            * config.channels_out
            * 4,
            ("bytes", "adjoint", "selection", i64): batch_size * seq_len * 8,
            (
                "bytes",
                "adjoint",
                "selection",
                bf16,
            ): bf16.itemsize
            * (
                3 * batch_size * seq_len * config.channels_out
                + config.channels_in * config.channels_out
            ),
        },
        params=8,
        params_active=4,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
