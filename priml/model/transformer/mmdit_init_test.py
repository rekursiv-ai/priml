"""Constructor-state and RNG goldens for default multi-stream blocks."""

from pathlib import Path
from typing import Final

from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.transformer.mmdit import MMDiTBlock
from priml.testing import golden
from priml.testing.bfb import assert_bfb_against_golden


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize("cond_dim", [0, 4])
def test_mmdit_constructor_golden(cond_dim: int) -> None:
    config = MMDiTBlock.Config()
    config.channels_in = 4
    config.cond_dim = cond_dim
    config.attn = MultiStreamAttention.Config()
    config.attn.num_heads = 2
    config.attn.channels_head = 2

    def constructor_values(module: nn.Module, input: Tensor) -> Tensor:
        del module, input
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            module = config.make()
            state = module.state_dict()
            return torch.cat(
                [
                    golden.heads(state.values(), count=8),
                    golden.rng_fingerprint().float(),
                ],
            )

    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name=f"mmdit_constructor_cond_{cond_dim}",
        build_module=nn.Identity,
        build_input=lambda: torch.zeros(()),
        run=constructor_values,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
