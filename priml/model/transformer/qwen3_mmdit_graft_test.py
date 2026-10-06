"""Qwen-specific config defaults and strict checkpoint-loading behavior."""

from pathlib import Path

import json

import pytest
import torch

from priml.model.attention.kernel import SdpaNaive
from priml.model.swiglu import SwiGLU
from priml.model.transformer.mmdit import MMDiTStream
from priml.model.transformer.mmdit_graft_test import (
    _assert_transferred,
    _language_only_masks,
    run_graft,
)
from priml.model.transformer.qwen3 import Qwen3
from priml.model.transformer.qwen3_mmdit_graft import Qwen3MMDiTGraft
from priml.model.transformer.qwen3_test import (
    attn,
    canonical_config,
    ffn,
    hf_config,
    synth_hf_state_dict,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


def test_config_defaults_and_make() -> None:
    config = Qwen3MMDiTGraft.Config()
    assert isinstance(config.backbone, Qwen3.Config)
    config.backbone = canonical_config()
    assert isinstance(config.make(), Qwen3MMDiTGraft)
    assert_pprint_golden(test_file=__file__, name="qwen3_mmdit_graft", config=config)


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_load_local_checkpoint(tmp_path: Path, tie: bool, dtype: torch.dtype) -> None:
    _write_synthetic_checkpoint(tmp_path, tie=tie)
    config = Qwen3MMDiTGraft.Config()
    backbone = canonical_config()
    stream_width = backbone.channels_in
    config.streams = [MMDiTStream.Config(), MMDiTStream.Config()]
    config.streams[1].ffn = SwiGLU.Config(channels_hidden=stream_width)
    before = config.pformat(finalize=False)
    graft = Qwen3MMDiTGraft.load(tmp_path, config=config, dtype=dtype, device="cpu")
    source = Qwen3.load(tmp_path, dtype=dtype)
    _assert_transferred(source, graft)
    assert graft.num_streams == 3
    assert isinstance(graft.blocks[0].ffns[2], SwiGLU)
    built_ffn = graft.blocks[0].ffns[2]
    assert isinstance(built_ffn, SwiGLU)
    assert built_ffn.up_proj.weight.shape[-2] == 2 * stream_width
    assert all(parameter.dtype == dtype for parameter in graft.parameters())
    assert config.pformat(finalize=False) == before


def _write_synthetic_checkpoint(
    directory: Path,
    *,
    layers: int = 1,
    tie: bool = False,
) -> None:
    """Write a random tiny Qwen3 checkpoint in the HF on-disk layout."""
    config = canonical_config()
    attention = attn(config)
    hf = hf_config(
        vocab_size=config.channels_out,
        hidden_size=config.channels_in,
        intermediate_size=ffn(config).channels_hidden,
        num_hidden_layers=layers,
        num_attention_heads=attention.num_heads,
        num_key_value_heads=attention.num_heads_kv,
        head_dim=attention.channels_head,
        tie_word_embeddings=tie,
    )
    (directory / "config.json").write_text(json.dumps(hf))
    torch.save(
        synth_hf_state_dict(Qwen3.Config.from_hf(hf)),
        directory / "pytorch_model.bin",
    )


def test_load_freeze_and_step_recipe(tmp_path: Path) -> None:
    """Run the documented recipe: load, freeze, forward, backward, optimizer step."""
    _write_synthetic_checkpoint(tmp_path, layers=2)
    graft = Qwen3MMDiTGraft.load(tmp_path, dtype=torch.float32, device="cpu")
    source = Qwen3.load(tmp_path, dtype=torch.float32)
    graft.freeze_backbone()
    before = {name: parameter.clone() for name, parameter in graft.named_parameters()}
    frozen = {
        name
        for name, parameter in graft.named_parameters()
        if not parameter.requires_grad
    }
    assert frozen
    assert len(frozen) < len(before)
    config = canonical_config()
    tokens = torch.tensor([[1, 0], [0, 1]])
    modality = torch.randn(2, 3, config.channels_in)
    masks = _language_only_masks(
        tokens.shape[1],
        modality=modality.shape[1],
    )
    optimizer = torch.optim.SGD(graft.parameters(), lr=0.1)
    # Two steps, not one: the fresh modality FFN starts with a zero down
    # projection, so its up projection receives an exactly zero gradient until
    # the first step has moved the down projection.
    for _ in range(2):
        with host_agnostic_numerics():
            logits, streams = graft(tokens, [modality], attn_mask=masks)
            assert torch.equal(logits, source(tokens))
        streams[0].square().mean().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    _assert_transferred(source, graft=graft)
    for name, parameter in graft.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, before[name]), name
        else:
            assert not torch.equal(parameter, before[name]), name


def test_load_rejects_other_qwen_families(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(hf_config(model_type="qwen3_moe")),
    )
    torch.save({}, tmp_path / "pytorch_model.bin")
    with pytest.raises(ValueError, match="model_type"):
        Qwen3MMDiTGraft.load(tmp_path)


def test_cost_is_inherited_and_matches_torch() -> None:
    """The inherited graft cost costs this backbone's products exactly.

    One position per stream over three positions, so the joint key length is
    six; the CPU SDPA op is not counted by torch, so the naive kernel stands
    in (``mmdit_graft_test``).
    """
    config = Qwen3MMDiTGraft.Config()
    backbone = canonical_config()
    config.backbone = backbone
    attn(backbone).attn_kernel = SdpaNaive.Config()
    config.streams[0].ffn = SwiGLU.Config(channels_hidden=backbone.channels_in)
    assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randint(0, backbone.channels_out, (2, 3)),
            torch.randn(2, 3, backbone.channels_in, requires_grad=True),
        ),
        seq_len=3,
        batch_size=2,
        dtype=None,
        run=run_graft,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
