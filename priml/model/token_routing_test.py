"""Tests for sparse token selection and fusion."""

from __future__ import annotations

import pytest
import torch

from priml.model.token_routing import SparseDenseFusion, select_tokens


def test_select_tokens_keeps_all_or_returns_original_ids() -> None:
    tokens = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    kept, ids = select_tokens(tokens, 0.0)
    assert ids is None
    assert torch.equal(kept, tokens)
    torch.manual_seed(0)
    kept, ids = select_tokens(tokens, 0.5)
    assert ids is not None
    assert kept.shape == (2, 2, 3)
    assert torch.equal(kept, tokens.gather(1, ids[..., None].expand(-1, -1, 3)))


@pytest.mark.parametrize("ratio", [-0.1, 1.0])
def test_select_tokens_rejects_invalid_ratio(ratio: float) -> None:
    with pytest.raises(ValueError, match="drop_ratio"):
        select_tokens(torch.zeros(2, 3, 4), ratio)


def test_sparse_dense_fusion_scatter_and_drop_path() -> None:
    config = SparseDenseFusion.Config(channels=4)
    assert config.cost(seq_len=2, batch_size=2, dtype=torch.float32).params == 40
    fusion = config.make()
    dense = torch.randn(3, 5, 4)
    sparse = torch.randn(3, 2, 4)
    ids = torch.tensor([[0, 3], [1, 4], [0, 2]])
    fused = fusion(dense, sparse, ids)
    expected_sparse = fusion.mask_token.expand_as(dense).clone()
    expected_sparse.scatter_(1, ids[..., None].expand_as(sparse), sparse)
    expected = fusion.proj(torch.cat((dense, expected_sparse), dim=-1))
    assert torch.equal(fused, expected)
    dropped = fusion(dense, sparse, ids, drop_path=True)
    mask = fusion.mask_token.expand_as(dense)
    assert torch.equal(dropped, fusion.proj(torch.cat((dense, mask), dim=-1)))
    assert fusion(dense, dense, None).shape == dense.shape


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
