"""Tests for ``MoE``, including bit-for-bit golden coverage.

Regenerate after an intentional numeric change::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/model/moe_test.py

Run through ``pytest``: the priml ``conftest.py`` sets ``MKL_CBWR`` and caps
math threads before torch imports. Minting from bare Python skips that setup.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.cost import cost
from priml.model.moe import MoE, Router, SigmoidRouter, SoftmaxRouter
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import (
    assert_bfb_against_golden,
    bfb_devices,
    first_tensor,
    move_to_device,
)
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_router_rejects_top_k_above_num_experts():
    """``top_k`` cannot exceed ``num_experts`` (topk would error).

    Regression for MODEL-009.
    """
    with pytest.raises(ValueError, match="top_k"):
        SoftmaxRouter.Config(channels_in=64, num_experts=4, top_k=5).make()


def test_router_rejects_non_positive_top_k():
    """``top_k`` must be at least 1. Regression for MODEL-009."""
    with pytest.raises(ValueError, match="top_k"):
        SoftmaxRouter.Config(channels_in=64, num_experts=4, top_k=0).make()


def test_router_rejects_top_k_above_grouped_eligible():
    """``top_k`` cannot exceed the experts kept by grouped routing.

    Regression for MODEL-009: with ``n_group`` groups and ``topk_group``
    kept, only ``topk_group * (num_experts // n_group)`` experts remain
    eligible; selecting more would draw from masked (-inf) experts.
    """
    with pytest.raises(ValueError, match="eligible"):
        SigmoidRouter.Config(
            channels_in=64,
            num_experts=8,
            top_k=5,
            n_group=4,
            topk_group=2,
        ).make()


def test_router():
    m = SoftmaxRouter.Config(channels_in=64, num_experts=4, top_k=2).make()
    x = torch.randn(2, 8, 64)
    weights, indices, logits = m(x)
    assert weights.shape == (2, 8, 2)
    assert indices.shape == (2, 8, 2)
    assert logits.shape == (2, 8, 4)


def test_router_jitter():
    m = SoftmaxRouter.Config(
        channels_in=64,
        num_experts=4,
        top_k=2,
        jitter_noise=0.1,
    ).make()
    m.train()
    x = torch.randn(2, 8, 64)
    weights, _, _ = m(x)
    assert weights.shape == (2, 8, 2)


def test_router_reset():
    m = SoftmaxRouter.Config(channels_in=64, num_experts=4).make()
    m.reset_parameters()


def test_router_forward_accepts_messages_and_rejects_positional_extras():
    m = SoftmaxRouter.Config(channels_in=64, num_experts=4).make()
    x = torch.randn(2, 8, 64)
    m(x, key="val")
    with pytest.raises(TypeError):
        cast(Callable[..., object], m)(x, "extra")


def test_base_router_is_not_buildable():
    """The activation is a subclass's decision; the base has none to make."""
    with pytest.raises(TypeError, match="scoring_func"):
        Router.Config(channels_in=8, num_experts=4, top_k=2).make()


def test_moe():
    m = MoE.Config(
        channels_in=64,
        router=SoftmaxRouter.Config(num_experts=4, top_k=2),
    ).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_moe_aux_loss():
    m = MoE.Config(
        channels_in=64,
        router=SoftmaxRouter.Config(num_experts=4, top_k=2),
    ).make()
    m.train()
    x = torch.randn(2, 8, 64)
    m(x)
    assert m._aux_loss.item() > 0


def test_moe_reset():
    m = MoE.Config(
        channels_in=64,
        router=SoftmaxRouter.Config(num_experts=4, top_k=2),
    ).make()
    m.reset_parameters()


def test_moe_forward_accepts_messages_and_rejects_positional_extras():
    m = MoE.Config(
        channels_in=64,
        router=SoftmaxRouter.Config(num_experts=4, top_k=2),
    ).make()
    x = torch.randn(2, 8, 64)
    assert m(x, key="val").shape == (2, 8, 64)
    with pytest.raises(TypeError):
        cast(Callable[..., object], m)(x, "extra")


def test_moe_channels_infer():
    cfg = MoE.Config(channels_in=64).finalize()
    assert cfg.channels_out == 64


def test_sort_dispatch_matches_mask_reference():
    """Sort-and-dispatch must be bit-identical to a mask-per-expert loop."""
    torch.manual_seed(0)
    m = MoE.Config(
        channels_in=32,
        router=SoftmaxRouter.Config(num_experts=6, top_k=3),
    ).make()
    m.eval()
    x = torch.randn(5, 3, 32)
    with torch.no_grad():
        fast = m(x)

    # Reference: explicit per-expert mask loop.
    x_flat = x.reshape(-1, 32)
    with torch.no_grad():
        weights, indices, _ = m.router(x_flat)
    ref = torch.zeros_like(x_flat)
    for e in range(m.num_experts):
        mask = indices == e
        using = mask.any(dim=-1)
        if not using.any():
            continue
        w_e = (weights * mask.float()).sum(dim=-1, keepdim=True)[using]
        with torch.no_grad():
            ref[using] += w_e * m.experts[e](x_flat[using])
    ref = ref.reshape(x.shape)
    assert torch.allclose(fast, ref, atol=1e-6, rtol=1e-5)


def test_sigmoid_routing_shared_experts_and_bias():
    """Sigmoid + shared experts + correction bias (DSV3/Kimi-K2 config)."""
    m = MoE.Config(
        channels_in=32,
        router=SigmoidRouter.Config(
            num_experts=8,
            top_k=2,
            routed_scaling_factor=2.5,
        ),
        num_shared_experts=1,
    ).make()
    # The slot is typed by what MoE needs of a router; this test built the
    # concrete one, so narrow back to inspect its own fields.
    router = m.router
    assert isinstance(router, SigmoidRouter)
    assert router.scoring_func == "sigmoid"
    assert router.norm_topk_prob is True  # auto-enabled for sigmoid.
    assert router.e_score_correction_bias is not None
    assert len(m.shared_experts) == 1
    x = torch.randn(2, 4, 32)
    assert m(x).shape == (2, 4, 32)


def test_moe_channels_out_differs_from_in():
    """MoE must support ``channels_out != channels_in``.

    Regression for MOE-CHANNELS-OUT (Issue#336): the dispatch buffer was
    allocated at ``channels_in`` width, so expert outputs of width
    ``channels_out`` could not be scattered back (shape mismatch crash).
    """
    m = MoE.Config(
        channels_in=8,
        channels_out=16,
        router=SoftmaxRouter.Config(num_experts=4, top_k=2),
    ).make()
    assert m(torch.randn(2, 3, 8)).shape == (2, 3, 16)


def test_moe_channels_out_with_shared_experts():
    """Shared experts also honor ``channels_out`` (DSV3-style projection)."""
    m = MoE.Config(
        channels_in=8,
        channels_out=16,
        router=SoftmaxRouter.Config(num_experts=4, top_k=2),
        num_shared_experts=1,
    ).make()
    assert m(torch.randn(2, 3, 8)).shape == (2, 3, 16)


def test_router_sigmoid_respects_explicit_norm_topk_prob():
    """Sigmoid routing must not clobber an explicit ``norm_topk_prob``.

    Regression for MOE-SIGMOID (Issue#336): ``finalize`` unconditionally
    set ``norm_topk_prob=True`` and ``use_correction_bias=True`` for
    sigmoid routing, silently overriding values the user supplied.
    """
    r = SigmoidRouter.Config(
        channels_in=8,
        num_experts=4,
        top_k=2,
        norm_topk_prob=False,
        use_correction_bias=False,
    ).make()
    assert r.norm_topk_prob is False
    assert r.e_score_correction_bias is None


def test_router_sigmoid_defaults_enable_norm_and_bias():
    """Sigmoid routing defaults ``norm_topk_prob`` and the bias on."""
    r = SigmoidRouter.Config(channels_in=8, num_experts=4, top_k=2).make()
    assert r.norm_topk_prob is True
    assert r.e_score_correction_bias is not None


def test_router_softmax_defaults_leave_weights_unnormalized():
    """Softmax already sums to 1, so its default skips the renormalization."""
    r = SoftmaxRouter.Config(channels_in=8, num_experts=4, top_k=2).make()
    assert r.norm_topk_prob is False
    assert "e_score_correction_bias" not in dict(r.named_buffers())


def test_moe_aux_loss_is_registered_buffer():
    """``_aux_loss`` must be a buffer so ``.to(device)`` moves it.

    Regression for MOE-AUX (Issue#336): a plain attribute tensor is not
    tracked by ``nn.Module``, so it stays on the original device after
    ``.to(...)`` and is invisible to ``state_dict``.
    """
    m = MoE.Config(
        channels_in=8,
        router=SoftmaxRouter.Config(num_experts=4, top_k=2),
    ).make()
    assert "_aux_loss" in dict(m.named_buffers())


def test_moe_reset_parameters_reinitializes_aux_loss():
    """``reset_parameters`` must reinitialize the ``_aux_loss`` buffer.

    Regression for MOE-AUX (Issue#336): the meta-init audit poisons every
    float buffer with NaN and requires ``reset_parameters`` to be the sole
    source of init. As a registered buffer, ``_aux_loss`` is in that audit
    set, so ``reset_parameters`` must restore it to a finite value.
    """
    m = MoE.Config(
        channels_in=8,
        router=SoftmaxRouter.Config(num_experts=4, top_k=2),
    ).make()
    with torch.no_grad():
        m._aux_loss.fill_(float("nan"))
    m.reset_parameters()
    assert torch.isfinite(m._aux_loss).all()


def test_group_topk_masks_inactive_groups():
    """8 experts / 2 groups / topk_group=1 keeps only one group alive."""
    m = MoE.Config(
        channels_in=32,
        router=SigmoidRouter.Config(
            num_experts=8,
            top_k=1,
            n_group=2,
            topk_group=1,
        ),
    ).make()
    assert isinstance(m.router, SigmoidRouter)
    bias = m.router.e_score_correction_bias
    assert bias is not None
    with torch.no_grad():
        bias.zero_()
        bias[0] = 1e3  # Pin group 0 (experts 0-3)
    x = torch.randn(8, 32)
    _, indices, _ = m.router(x)
    assert ((indices >= 0) & (indices < 4)).all()


def test_router_config_pprint() -> None:
    config = SoftmaxRouter.Config(channels_in=4, num_experts=2, top_k=1)
    assert_pprint_golden(
        test_file=__file__,
        name="router",
        config=config,
    )


def test_sigmoid_router_config_pprint() -> None:
    config = SigmoidRouter.Config(channels_in=4, num_experts=2, top_k=1)
    assert_pprint_golden(
        test_file=__file__,
        name="sigmoid_router",
        config=config,
    )


def test_moe_config_pprint() -> None:
    config = MoE.Config(
        channels_in=4,
        router=SoftmaxRouter.Config(num_experts=2, top_k=1),
    )
    assert_pprint_golden(
        test_file=__file__,
        name="moe",
        config=config,
    )


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_router_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="router",
        build_module=lambda: (
            SoftmaxRouter.Config(channels_in=4, num_experts=2, top_k=1)
            .make()
            .to(device)
        ),
        build_input=lambda: move_to_device(torch.randn(4, 4), device),
        seed=0,
        run=_first_tensor,
    )


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_moe_bfb(device: str) -> None:
    """Regenerate with ``BFB_REGENERATE=1`` against this canonical sidecar.

    Args:
      device: Device on which to run the golden comparison.

    """
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="moe",
        build_module=lambda: (
            MoE.Config(
                channels_in=4,
                router=SoftmaxRouter.Config(num_experts=2, top_k=1),
            )
            .make()
            .to(device)
        ),
        build_input=lambda: move_to_device(torch.randn(2, 2, 4), device),
        seed=0,
        run=_first_tensor,
    )


@pytest.mark.parametrize(
    "config",
    [
        SoftmaxRouter.Config(channels_in=8, num_experts=4, top_k=2),
        SigmoidRouter.Config(channels_in=8, num_experts=4, top_k=2),
    ],
    ids=["softmax", "sigmoid"],
)
def test_router_cost_is_the_gate_matmul(config: Router.Config) -> None:
    """The sigmoid correction bias is a buffer, so neither path counts it.

    The top-k pick sorts the expert row and gathers ``top_k`` weights, which
    the adjoint scatter-adds back.
    """
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(3, 8, requires_grad=True),
        num_tokens=3,
        run=_first_tensor,
    )
    assert model_cost.primal.flops.matmul == 2 * 8 * 4
    assert model_cost.adjoint.flops.matmul == 4 * 8 * 4
    assert model_cost.params == 8 * 4
    assert model_cost.primal.bytes.sort == 4 * (4 + 2 * 2)
    assert model_cost.primal.bytes.selection == 4 * 3 * 2
    assert model_cost.adjoint.flops.selection == 2


def test_softmax_router_counts_the_softmax_reductions() -> None:
    config = SoftmaxRouter.Config(channels_in=8, num_experts=4, top_k=2)
    model_cost = cost(config.copy_tree().finalize())
    # Max and sum over four experts in the primal; ``sum(g * p)`` in the adjoint.
    assert model_cost.primal.flops.reduction == 2 * 3
    assert model_cost.adjoint.flops.reduction == 3
    assert model_cost.primal.flops.elementwise == 3 * 4


def test_sigmoid_router_prices_nonlinearity_as_one_tensor_operator() -> None:
    config = SigmoidRouter.Config()
    config.channels_in = 8
    config.num_experts = 4
    config.norm_topk_prob = False
    config.use_correction_bias = False
    result = config.cost(itemsize=2)
    assert result.primal.bytes.elementwise == 2 * (2 * 4 + 2 * 2)
    assert result.adjoint.bytes.elementwise == 2 * (3 * 4 + 2 * 2)


def test_moe_cost_owns_every_expert_but_activates_top_k() -> None:
    """Torch's count agrees: every token runs exactly ``top_k`` experts.

    Sort-and-dispatch issues one matmul per active expert with as many rows
    as it received, so the total is ``top_k * rows`` expert rows however
    the router assigns them.
    """
    expert = SwiGLU.Config(channels_hidden=16, round_to=1)
    config = MoE.Config(
        channels_in=8,
        router=SoftmaxRouter.Config(num_experts=4, top_k=2),
        expert=expert,
        num_shared_experts=1,
        shared_expert=expert.copy_tree(),
    )
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 8, requires_grad=True),
        num_tokens=6,
    )
    finalized = config.copy_tree().finalize()
    router = cost(finalized.router, rows=6)
    one_expert = cost(finalized.expert, rows=6)
    assert router.params == 8 * 4
    assert one_expert.params == 8 * 32 + 16 * 8
    assert model_cost.params == router.params + (4 + 1) * one_expert.params
    assert model_cost.params_active == router.params + (2 + 1) * one_expert.params
    assert model_cost.primal.flops.matmul == (
        router.primal.flops.matmul + (2 + 1) * one_expert.primal.flops.matmul
    )
    assert model_cost.adjoint.flops.matmul == (
        router.adjoint.flops.matmul + (2 + 1) * one_expert.adjoint.flops.matmul
    )
    # Dispatch: gather each routed token's row in and scatter its output back.
    assert model_cost.primal.bytes.sort == router.primal.bytes.sort + 4 * 2 * 2
    assert model_cost.primal.flops.selection == (2 + 1) * 8
    assert (
        model_cost.adjoint.flops.selection
        == (2 + 1) * 8 + router.adjoint.flops.selection
    )


def test_moe_cost_does_not_price_an_unused_shared_expert() -> None:
    config = MoE.Config()
    config.channels_in = 4
    config.shared_expert = SwiGLU.Config()
    config.shared_expert.act = torch.nn.functional.gelu
    config.make()
    assert config.finalize().cost().params > 0


def test_moe_cost_shares_weights_over_balanced_expert_rows() -> None:
    config = MoE.Config()
    config.channels_in = 4
    config.router.num_experts = 4
    config.expert = SwiGLU.Config()
    config.expert.channels_hidden = 3
    config.expert.bias = True
    finalized = config.finalize()
    result = finalized.cost(rows=8, itemsize=2)
    router = cost(finalized.router, rows=8, itemsize=2)
    expert = cost(finalized.expert, rows=4, itemsize=2)
    assert (
        result.primal.bytes.matmul
        == router.primal.bytes.matmul + 2 * expert.primal.bytes.matmul
    )
    assert (
        result.adjoint.bytes.matmul
        == router.adjoint.bytes.matmul + 2 * expert.adjoint.bytes.matmul
    )
    assert (
        result.adjoint.flops.reduction
        == router.adjoint.flops.reduction
        + 2 * expert.adjoint.flops.reduction
        + 2 * (4 - 1)
    )


def _first_tensor(module: torch.nn.Module, value: object) -> torch.Tensor:
    """Call a module after narrowing the harness input to its tensor contract."""
    assert isinstance(value, torch.Tensor)
    assert isinstance(module, (MoE, Router))
    return first_tensor(module(value))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
