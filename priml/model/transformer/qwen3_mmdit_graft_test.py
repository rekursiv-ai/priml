"""Qwen-specific config defaults and strict checkpoint-loading behavior."""

from pathlib import Path

import json

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.attention.kernel import SdpaNaive
from priml.model.swiglu import SwiGLU
from priml.model.transformer.mmdit import MMDiTStream
from priml.model.transformer.mmdit_graft_test import (
    _assert_transferred,
    run_graft,
)
from priml.model.transformer.qwen3 import Qwen3
from priml.model.transformer.qwen3_mmdit_graft import Qwen3MMDiTGraft
from priml.model.transformer.qwen3_test import (
    _attn,
    _canonical_config,
    _hf_config,
    _synth_hf_state_dict,
)
from priml.testing.cost import assert_cost_matches_torch


def test_config_defaults_and_make() -> None:
    config = Qwen3MMDiTGraft.Config()
    assert isinstance(config.backbone, Qwen3.Config)
    config.backbone = _canonical_config()
    assert isinstance(config.make(), Qwen3MMDiTGraft)
    assert_pprint_golden(test_file=__file__, name="qwen3_mmdit_graft", config=config)


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_load_local_checkpoint(tmp_path: Path, tie: bool, dtype: torch.dtype) -> None:
    hf = _hf_config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        tie_word_embeddings=tie,
    )
    (tmp_path / "config.json").write_text(json.dumps(hf))
    backbone = Qwen3.Config.from_hf(hf)
    torch.save(_synth_hf_state_dict(backbone), tmp_path / "pytorch_model.bin")
    config = Qwen3MMDiTGraft.Config()
    config.streams = [MMDiTStream.Config(), MMDiTStream.Config()]
    config.streams[1].ffn = SwiGLU.Config(channels_hidden=24)
    before = config.pformat(finalize=False)
    graft = Qwen3MMDiTGraft.load(tmp_path, config=config, dtype=dtype, device="cpu")
    source = Qwen3.load(tmp_path, dtype=dtype)
    _assert_transferred(source, graft)
    assert graft.num_streams == 3
    assert isinstance(graft.blocks[0].ffns[2], SwiGLU)
    ffn = graft.blocks[0].ffns[2]
    assert isinstance(ffn, SwiGLU)
    assert ffn.up_proj.weight.shape[-2] == 48
    assert all(parameter.dtype == dtype for parameter in graft.parameters())
    assert config.pformat(finalize=False) == before


def test_load_rejects_other_qwen_families(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(_hf_config(model_type="qwen3_moe")),
    )
    torch.save({}, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="model_type"):
        Qwen3MMDiTGraft.load(tmp_path)


def test_cost_is_inherited_and_matches_torch() -> None:
    """The inherited graft cost prices this backbone's products exactly.

    One position per stream over three positions, so the joint key length is
    six; the CPU SDPA op is not counted by torch, so the naive kernel stands
    in (``mmdit_graft_test``).
    """
    config = Qwen3MMDiTGraft.Config()
    config.backbone = _canonical_config()
    _attn(config.backbone).attn_kernel = SdpaNaive.Config()
    config.streams[0].ffn = SwiGLU.Config(channels_hidden=24)
    assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randint(0, 32, (1, 3)),
            torch.randn(1, 3, 16, requires_grad=True),
        ),
        seq_len=6,
        batch_size=1,
        dtype=None,
        rows=3,
        run=run_graft,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
