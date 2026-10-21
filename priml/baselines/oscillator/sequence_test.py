"""Check temporal model geometry, gradients, and delegated cost counts."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch import Tensor

import pytest
import torch

from priml.baselines.oscillator.experiments import exp036, exp037
from priml.baselines.oscillator.nonlinear import HistoryNetwork, NeuralForecast
from priml.baselines.oscillator.sequence import (
    PatchSequence,
    RbfKernel,
    RecurrentSequence,
)
from priml.cost import cost
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_pprint_golden


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.compute_training
def test_temporal_blocks_keep_geometry_receive_gradients_and_price_parameters(
    recurrent: bool,
) -> None:
    cfg = _config(recurrent=recurrent).finalize()
    model = cfg.make()
    media = torch.arange(3 * 10 * 2).reshape(3, 10, 2).float() / 31
    prediction = model(media)
    assert prediction.shape == (3, 7, 2)
    prediction.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().max() > 0
        for parameter in model.parameters()
    )
    counted = cost(cfg, seq_len=10, batch_size=3, dtype=torch.float32)
    assert counted.params == sum(parameter.numel() for parameter in model.parameters())
    assert counted["primal", "flops", "matmul", torch.float32] > 0
    with pytest.raises(ValueError, match="geometry"):
        model.block(media.flatten(start_dim=1)[:, :-1])


@pytest.mark.parametrize("recurrent", [False, True])
def test_temporal_forward_reset_and_cost_are_independent_between_trajectories(
    recurrent: bool,
) -> None:
    with host_agnostic_numerics():
        cfg = _config(recurrent=recurrent).finalize()
        model = cfg.make()
        media = torch.arange(3 * 10 * 2).reshape(3, 10, 2).float() / 31
        with torch.no_grad():
            prediction = model(media)
            permutation = torch.tensor([2, 0, 1])
            assert torch.equal(model(media[permutation]), prediction[permutation])
        counted = cost(cfg, seq_len=10, batch_size=3, dtype=torch.float32)
        assert counted.params == sum(
            parameter.numel() for parameter in model.parameters()
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        assert isinstance(model.block, PatchSequence)
        model.block.reset_parameters()
        assert any(
            not torch.equal(left, right)
            for left, right in zip(before, model.parameters(), strict=True)
        )
        assert torch.isfinite(model(media)).all()
        with pytest.raises(ValueError, match="geometry"):
            model.block(media.flatten(start_dim=1)[:, :-1])


def test_recurrent_sequence_defaults_and_rejects_incompatible_widths() -> None:
    cfg = RecurrentSequence.Config(channels_in=5)
    assert cfg.finalize().channels_out == 5
    cfg.channels_out = 7
    cfg.block.channels_hidden = 6
    with pytest.raises(ValueError, match="width"):
        cfg.make()


def test_patch_sequence_matches_manual_module_composition() -> None:
    model = _config(recurrent=False).make()
    assert isinstance(model.block, PatchSequence)
    media = torch.arange(3 * 10 * 2).reshape(3, 10, 2).float() / 31
    patches = torch.stack(
        [media[:, start : start + 2].flatten(start_dim=1) for start in range(0, 10, 2)],
        dim=1,
    )
    projected = model.block.proj_in(patches)
    encoded = model.block.block(projected)
    ordered = torch.cat(encoded.unbind(dim=1), dim=-1)
    expected = model.block.proj_out(ordered).reshape(3, 7, 2)
    assert expected.abs().max() > 0
    assert torch.equal(model(media), expected)


def test_recurrent_wrapper_returns_fresh_owned_sequence() -> None:
    cfg = RecurrentSequence.Config(channels_in=5)
    model = cfg.make()
    values = torch.arange(2 * 3 * 5).reshape(2, 3, 5).float() / 17
    expected, _ = model.block(values)
    assert expected.abs().max() > 0
    assert torch.equal(model(values), expected)
    model(values.flip(dims=(1,)))
    assert torch.equal(model(values), expected)


@pytest.mark.parametrize("bad", ["patch", "partial", "output", "head"])
def test_bad_temporal_geometry_is_rejected(bad: str) -> None:
    cfg = _config(recurrent=False)
    assert isinstance(cfg.block, PatchSequence.Config)
    if bad == "patch":
        cfg.block.patch_size = 0
    elif bad == "partial":
        cfg.seq_len = 9
    elif bad == "output":
        cfg.pred_len = 0
    else:
        finalized = cfg.copy_tree().finalize()
        assert isinstance(finalized.block, PatchSequence.Config)
        finalized.block.proj_out.channels_in += 1
        with pytest.raises(ValueError, match="head"):
            finalized.make()
        return
    with pytest.raises(ValueError, match=r"geometry|patches|positive"):
        cfg.make()


def test_radial_kernel_is_symmetric_bounded_and_sensitive_to_scale() -> None:
    values = torch.arange(3 * 4).reshape(3, 4).double() / 7
    cfg = RbfKernel.Config()
    gram = cfg.make()(values, values)
    assert torch.equal(gram, gram.T)
    torch.testing.assert_close(gram.diag(), torch.ones(3, dtype=torch.float64))
    assert gram.min() >= 0
    assert gram.max() <= 1
    eigvalsh = cast(Callable[[Tensor], Tensor], torch.linalg.eigvalsh)
    assert eigvalsh(gram).min() >= -1e-12
    cfg.length_scale = 0.2
    assert not torch.equal(gram, cfg.make()(values, values))
    cfg.length_scale = float("nan")
    with pytest.raises(ValueError, match="scale"):
        cfg.make()


def test_patch_sequence_nexus_is_fully_printable() -> None:
    assert_pprint_golden(test_file=__file__, name="exp036", config=exp036())


def test_recurrent_fork_changes_only_the_finalized_mixing_stack() -> None:
    before = exp036().finalize()
    after = exp037().finalize()
    assert isinstance(before.method, NeuralForecast.Config)
    assert isinstance(after.method, NeuralForecast.Config)
    assert isinstance(before.method.step.model.block, PatchSequence.Config)
    assert isinstance(after.method.step.model.block, PatchSequence.Config)
    mixer = after.method.step.model.block.block
    assert isinstance(mixer, RecurrentSequence.Config)
    assert mixer.channels_in == mixer.channels_out == 32
    assert mixer.block.channels_in == mixer.block.channels_hidden == 32
    assert mixer.block.num_layers == 2
    after.method.step.model.block.block = before.method.step.model.block.block
    after.experiment_name = before.experiment_name
    assert after == before


def _config(*, recurrent: bool) -> HistoryNetwork.Config:
    cfg = HistoryNetwork.Config()
    cfg.seq_len = 10
    cfg.pred_len = 7
    block = cfg.block = PatchSequence.Config()
    block.patch_size = 2
    block.proj_in.channels_out = 12
    if recurrent:
        block.block = RecurrentSequence.Config()
    else:
        assert isinstance(block.block, Sequential.Config)
        assert isinstance(block.block.elements, TransformerBlock.Config)
        assert isinstance(block.block.elements.attn, Attention.Config)
        assert isinstance(block.block.elements.ffn, SwiGLU.Config)
        block.block.elements.attn.num_heads = 2
        assert isinstance(block.block.elements.attn.rope, RoPE.Config)
        block.block.elements.attn.rope.channels_head = (
            block.proj_in.channels_out // block.block.elements.attn.num_heads
        )
        block.block.elements.ffn.channels_hidden = 8
    return cfg


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
