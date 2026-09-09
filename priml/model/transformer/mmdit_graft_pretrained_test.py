"""One retained layer of an actual Qwen3 checkpoint, compared on the GPU."""

from pathlib import Path

import logging

import pytest
import torch

from priml import hub
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.self_attention import SelfAttention
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.mmdit_graft_test import _assert_transferred
from priml.model.transformer.qwen3 import Qwen3, remap_hf_state_dict
from priml.model.transformer.qwen3_mmdit_graft import Qwen3MMDiTGraft
from priml.testing.bfb import host_agnostic_numerics


@pytest.mark.gpu_torch_cuda
@pytest.mark.network_huggingface
def test_one_layer_pretrained_qwen3_graft(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real-checkpoint memory test")
    transformers = pytest.importorskip("transformers")
    repo = "Qwen/Qwen3-0.6B"
    hf_config = transformers.AutoConfig.from_pretrained(repo, cache_dir=tmp_path / "hf")
    hf_config.num_hidden_layers = 1
    hf_config.layer_types = hf_config.layer_types[:1]
    config = Qwen3.Config.from_hf(hf_config.to_dict())
    assert isinstance(config.block, TransformerBlock.Config)
    assert isinstance(config.block.attn, SelfAttention.Config)
    config.block.attn.attn_kernel = SdpaNaive.Config()
    pretrained = hub.load_transformers_model(
        repo,
        "AutoModelForCausalLM",
        config=hf_config,
        cache_dir=tmp_path / "hf",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    source = config.make()
    source.load_state_dict(
        remap_hf_state_dict(pretrained.state_dict(), config.copy_tree().finalize()),
        strict=True,
    )
    checkpoint = tmp_path / "checkpoint"
    pretrained.save_pretrained(checkpoint)
    del pretrained
    source = source.cuda().eval()
    graft = Qwen3MMDiTGraft.load(checkpoint, device="cuda", dtype=torch.float32).eval()
    for block in graft.blocks:
        block.attn.attn_kernel = SdpaNaive.Config().make()
    _assert_transferred(source, graft)
    tokens = torch.tensor([[1, 2, 3]], device="cuda")
    other = torch.randn(1, 2, config.channels_in, device="cuda")
    mask = torch.cat(
        (
            torch.full((3, 3), float("-inf"), device="cuda").triu(1),
            torch.full((3, 2), float("-inf"), device="cuda"),
        ),
        -1,
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), host_agnostic_numerics():
        expected = source(tokens)
        actual, outputs = graft(tokens, [other], attn_mask=[mask, None])
        assert torch.equal(actual, expected)
        assert torch.count_nonzero(actual) > 0
        assert outputs[0].shape == other.shape
    peak = torch.cuda.max_memory_allocated()
    logging.getLogger(__name__).info(
        "Qwen3 checkpoint=%s, retained_layers=1, width=%s, peak_cuda_bytes=%s",
        repo,
        config.channels_in,
        peak,
    )
    assert peak < 8 * 1024**3
