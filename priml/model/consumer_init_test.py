"""Freeze composite constructors, RNG consumption, and forward outputs."""

from functools import partial
from pathlib import Path
from typing import Final

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.nanochat.model import NanoChatLM
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.multi_stream import MultiStreamAttention
from priml.model.attention.self_attention import SelfAttention
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.model.embedding import Embedding
from priml.model.init import kaiming_uniform, unit_fan_in_uniform
from priml.model.linear import EnsembleLinear, Linear
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.moe import MoE, Router
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.special import TiedLinear
from priml.model.swiglu import SwiGLU
from priml.model.transformer import kimi_k2_test, qwen3_test
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.kimi_k2 import KimiK2
from priml.model.transformer.mmdit import MMDiTBlock, MMDiTStream
from priml.model.transformer.qwen3 import Qwen3
from priml.testing.bfb import assert_bfb_against_golden


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "config",
    [
        TransformerBlock.Config().ffn,
        MMDiTBlock.Config().ffn,
        MMDiTStream.Config().ffn,
        MLPMixerBlock.Config().token_mixer,
        MLPMixerBlock.Config().channel_mixer,
        MoE.Config().expert,
        MoE.Config().shared_expert,
    ],
)
def test_reusable_ffns_inherit_current_defaults(config: object) -> None:
    assert isinstance(config, SwiGLU.Config)
    assert config.init_weight is unit_fan_in_uniform
    assert config.init_weight_out is nn.init.zeros_


@pytest.mark.parametrize("kind", ["qwen", "kimi"])
def test_native_reference_norm_defaults(kind: str) -> None:
    config = Qwen3.Config() if kind == "qwen" else KimiK2.Config()
    final_norm = qwen3_test._final_norm(config)
    assert final_norm.elementwise_affine
    assert isinstance(config.block, TransformerBlock.Config)
    for norm in (config.block.norm1, config.block.norm2):
        assert isinstance(norm, RMSNorm.Config)
        assert norm.elementwise_affine


@pytest.mark.parametrize("kind", ["qwen", "kimi"])
@pytest.mark.parametrize("std", [0.02, 0.07])
def test_reference_initialization(kind: str, std: float) -> None:
    if kind == "qwen":
        config = Qwen3.Config.from_hf(qwen3_test._hf_config(initializer_range=std))
    else:
        config = KimiK2.Config.from_hf(
            kimi_k2_test._hf_config(initializer_range=std, q_lora_rank=16)
        )
    torch.manual_seed(13)
    model = config.make()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    initial_rng = torch.get_rng_state()
    torch.manual_seed(13)
    model.reset_parameters()
    assert torch.equal(torch.get_rng_state(), initial_rng)
    for name, value in model.state_dict().items():
        assert torch.equal(value, initial[name]), name
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            assert module.weight is not None, name
            assert torch.equal(module.weight, torch.ones_like(module.weight)), name
        if not isinstance(module, (Linear, EnsembleLinear, nn.Embedding)):
            continue
        torch.manual_seed(7)
        module.reset_parameters()
        actual = module.weight.detach()
        torch.manual_seed(7)
        expected = torch.empty_like(actual)
        if isinstance(module, EnsembleLinear):
            for weight in expected:
                nn.init.normal_(weight, std=std)
        else:
            nn.init.normal_(expected, std=std)
        assert torch.equal(actual, expected), name
    for module in model.modules():
        if isinstance(module, Router):
            torch.manual_seed(7)
            module.reset_parameters()
            torch.manual_seed(7)
            expected = torch.empty_like(module.gate.weight)
            nn.init.kaiming_uniform_(expected, a=5**0.5)
            assert torch.equal(module.gate.weight, expected)
            if module.e_score_correction_bias is not None:
                assert torch.count_nonzero(module.e_score_correction_bias) == 0


@pytest.mark.parametrize("kind", ["qwen", "kimi"])
@pytest.mark.parametrize("tie", [False, True])
def test_reference_leaf_overrides_survive_make_and_reset(kind: str, tie: bool) -> None:
    config = (
        qwen3_test._canonical_config()
        if kind == "qwen"
        else kimi_k2_test._canonical_config()
    )
    assert isinstance(config.in_proj, Embedding.Config)
    config.in_proj.init_weight = nn.init.ones_
    config.in_proj.padding_idx = 0
    assert isinstance(config.out_proj, Sequential.Config)
    head_elements = config.out_proj.elements
    assert isinstance(head_elements, list)
    head = head_elements[1]
    assert isinstance(head, Linear.Config)
    head.init_weight = nn.init.ones_
    if tie:
        head_elements[1] = TiedLinear.Config(tied="in_proj")
    assert isinstance(config.block, TransformerBlock.Config)
    ffns = (
        (config.block.ffn.expert, config.block.ffn.shared_expert)
        if isinstance(config.block.ffn, MoE.Config)
        else (config.block.ffn,)
    )
    for ffn in ffns:
        assert isinstance(ffn, SwiGLU.Config)
        ffn.init_weight = nn.init.ones_
        ffn.init_weight_out = nn.init.ones_
    model = config.make()
    assert isinstance(model.in_proj, Embedding)
    assert config.in_proj.channels_out == -1
    assert config.in_proj.num_embeddings == -1
    for reset in (False, True):
        if reset:
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            model.reset_parameters()
        assert torch.count_nonzero(model.in_proj.weight[0]) == 0
        assert torch.equal(
            model.in_proj.weight[1:], torch.ones_like(model.in_proj.weight[1:])
        )
        for module in model.modules():
            if isinstance(module, SwiGLU):
                for parameter in module.parameters():
                    assert torch.equal(parameter, torch.ones_like(parameter))
        assert isinstance(model.out_proj, Sequential)
        built_head = model.out_proj[1]
        if tie:
            assert isinstance(built_head, TiedLinear)
        else:
            assert isinstance(built_head, Linear)
            assert torch.equal(built_head.weight, torch.ones_like(built_head.weight))
        assert model(torch.tensor([[1, 2]])).shape == (1, 2, config.channels_out)
    assert "in_proj.weight" in model.state_dict()
    assert ("out_proj.1.weight" in model.state_dict()) is not tie


@pytest.mark.parametrize(
    "kind", ["transformer", "mmdit", "moe", "mixer", "nanochat", "qwen", "kimi"]
)
def test_legacy_composite_constructor_bfb(kind: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name=f"consumer_init_{kind}",
        build_module=nn.Identity,
        build_input=lambda: torch.empty(0),
        run=partial(_constructor_values, kind=kind),
    )


def _constructor_values(module: nn.Module, inp: Tensor, *, kind: str) -> Tensor:
    del module, inp
    legacy_ffn = SwiGLU.Config(
        init_weight=kaiming_uniform, init_weight_out=kaiming_uniform
    )
    if kind == "transformer":
        cfg = TransformerBlock.Config(channels_in=8, ffn=legacy_ffn)
        cfg.attn = SelfAttention.Config(num_heads=2, channels_head=4)
        model = cfg.make()
    elif kind == "mmdit":
        cfg_mmdit = MMDiTBlock.Config(channels_in=8, num_streams=2, ffn=legacy_ffn)
        cfg_mmdit.attn = MultiStreamAttention.Config(num_heads=2, channels_head=4)
        model = cfg_mmdit.make()
    elif kind == "moe":
        cfg_moe = MoE.Config(
            channels_in=8,
            num_shared_experts=1,
            expert=legacy_ffn,
            shared_expert=legacy_ffn.copy_tree(),
        )
        cfg_moe.router = Router.Config(num_experts=2, top_k=1)
        model = cfg_moe.make()
    elif kind == "mixer":
        model = MLPMixerBlock.Config(
            channels_in=8,
            seq_len=3,
            token_mixer=legacy_ffn,
            channel_mixer=legacy_ffn.copy_tree(),
        ).make()
    elif kind in ("qwen", "kimi"):
        config = (
            qwen3_test._canonical_config()
            if kind == "qwen"
            else kimi_k2_test._canonical_config()
        )
        config.in_proj = Embedding.Config(shard="vocab")
        config.out_proj = Sequential.Config(
            elements=[
                RMSNorm.Config(elementwise_affine=True),
                Linear.Config(shard="vocab"),
            ]
        )
        config = config.finalize()
        assert isinstance(config.block, list)
        for block in config.block:
            assert isinstance(block, TransformerBlock.Config)
            attn = block.attn
            assert isinstance(
                attn, (SelfAttention.Config, MultiHeadLatentAttention.Config)
            )
            attn.init_weight = kaiming_uniform
            if isinstance(attn, MultiHeadLatentAttention.Config):
                for projection in (
                    attn.proj_q,
                    attn.proj_q_a,
                    attn.proj_q_b,
                    attn.proj_kv_a,
                    attn.proj_kv_b,
                    attn.proj_out,
                ):
                    assert isinstance(projection, Linear.Config)
                    projection.init_weight = kaiming_uniform
            ffns = (
                (block.ffn.expert, block.ffn.shared_expert)
                if isinstance(block.ffn, MoE.Config)
                else (block.ffn,)
            )
            for ffn in ffns:
                assert isinstance(ffn, SwiGLU.Config)
                ffn.init_weight = kaiming_uniform
                ffn.init_weight_out = kaiming_uniform
        model = config.make()
    else:
        cfg_nano = NanoChatLM.Config(vocab_size=16, channels_in=8, num_layers=1)
        cfg_nano.max_seq_len = 3
        assert isinstance(cfg_nano.block, TransformerBlock.Config)
        assert isinstance(cfg_nano.block.attn, ValueGatedAttention.Config)
        cfg_nano.block.attn.channels_head = 4
        cfg_nano.block.attn.gate_channels = 4
        model = cfg_nano.make()
    values = [value.detach().flatten().float() for value in model.state_dict().values()]
    values.append(torch.get_rng_state().float())
    x = torch.randn(1, 3, 8)
    if isinstance(model, MMDiTBlock):
        outputs = model([x, x])
        values.extend(value.flatten().float() for value in outputs)
    elif kind in ("nanochat", "qwen", "kimi"):
        values.append(model(torch.tensor([[0, 1, 2]])).flatten().float())
    else:
        values.append(model(x).flatten().float())
    values.append(torch.get_rng_state().float())
    return torch.cat(values)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
