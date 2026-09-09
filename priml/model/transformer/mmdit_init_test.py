"""Constructor-state and RNG goldens for default multi-stream blocks."""

from pathlib import Path

from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.transformer.mmdit import MMDiTBlock
from priml.testing.bfb import assert_bfb_against_golden


@pytest.mark.parametrize("cond_dim", [0, 4])
def test_mmdit_constructor_golden(cond_dim: int) -> None:
    config = MMDiTBlock.Config()
    config.channels_in = 8
    config.cond_dim = cond_dim
    config.attn = MultiStreamAttention.Config()
    config.attn.num_heads = 2
    config.attn.channels_head = 4
    assert_bfb_against_golden(
        golden_dir=Path(__file__).parent / "testdata",
        golden_name=f"mmdit_constructor_cond_{cond_dim}",
        build_module=nn.Identity,
        build_input=lambda: torch.zeros(1),
        run=lambda _module, _input: _constructor_state(config),
    )


def _constructor_state(config: MMDiTBlock.Config) -> Tensor:
    """Build inside the runner so golden replay cannot overwrite initialization."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        module = config.make()
        values = [value.reshape(-1).float() for value in module.state_dict().values()]
        values.append(torch.get_rng_state().float())
        return torch.cat(values)
