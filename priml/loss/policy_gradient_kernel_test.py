"""Tests for the rule's Triton kernels: their refusal off CUDA, the kernels on a GPU.

The kernels are checked against the torch reference at tolerance: Triton's
``exp`` is the fast ``ex2.approx`` and its contractions round once where torch
rounds twice. A golden freezes the kernels' outputs and gradients on one
128-row minibatch of 256 steps, per GPU model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import re

import pytest
import torch

from priml.loss.policy_gradient import LogProbs, TorchPPO
from priml.loss.policy_gradient_kernel import TritonPPO
from priml.testing.golden import assert_text_golden
from priml.testing.policy_gradient import (
    portable_minibatch,
    random_minibatch,
    rule_entries,
)


if TYPE_CHECKING:
    from torch import Tensor


def test_off_cuda_every_stage_is_refused() -> None:
    # The torch rule's precise ``exp`` lands a few ulp from the kernels', so a
    # fallback would give one config different bits on a CPU host.
    batch = random_minibatch(rows=2, horizon=TritonPPO.Config.ADVANTAGE_WIDTH)
    rule = TritonPPO.Config().make()
    logprobs = (
        TorchPPO.Config()
        .make()
        .log_probs(batch["decoded"], batch["actions"], batch["action_mask"])
    )
    with pytest.raises(ValueError, match="TorchPPO"):
        rule.log_probs(batch["decoded"], batch["actions"], batch["action_mask"])
    with pytest.raises(ValueError, match="TorchPPO"):
        rule.advantage(logprobs.values, batch["rewards"], batch["terminals"])
    with pytest.raises(ValueError, match="TorchPPO"):
        rule.loss(
            logprobs,
            decoded=batch["decoded"],
            actions=batch["actions"],
            old_logprobs=batch["old_logprobs"],
            advantages=batch["values"],
            values=batch["values"],
            returns=batch["values"],
        )


def test_the_kernels_take_the_reference_rules_coefficients() -> None:
    config = TritonPPO.Config()
    assert isinstance(config, TorchPPO.Config)
    config.entropy_coefficient = 0.25
    rule = config.make()
    assert isinstance(rule, TritonPPO)
    assert rule.entropy_coefficient == 0.25


@pytest.mark.parametrize(("block", "num_warps"), [(100, 8), (256, 3)])
def test_a_launch_size_triton_cannot_tile_is_refused(
    block: int,
    num_warps: int,
) -> None:
    config = TritonPPO.Config()
    config.block = block
    config.num_warps = num_warps
    with pytest.raises(ValueError, match="power of two"):
        config.make()


@pytest.mark.parametrize(
    "horizon",
    [0, TritonPPO.Config.ADVANTAGE_WIDTH // 2, TritonPPO.Config.ADVANTAGE_WIDTH + 1],
)
def test_a_horizon_the_advantage_kernel_cannot_tile_is_refused(horizon: int) -> None:
    rule = TritonPPO.Config().make()
    rule.check_horizon(2 * TritonPPO.Config.ADVANTAGE_WIDTH)
    with pytest.raises(
        ValueError,
        match=f"multiple of {TritonPPO.Config.ADVANTAGE_WIDTH}",
    ):
        rule.check_horizon(horizon)


def _cuda_batch(rows: int, horizon: int) -> dict[str, Tensor]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return random_minibatch(rows=rows, horizon=horizon, device="cuda", seed=3)


@pytest.mark.gpu_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=("bf16", "fp32"))
def test_the_triton_kernels_match_the_torch_reference(dtype: torch.dtype) -> None:
    """Over a bf16 policy's fused rows, or an fp32 decoder's."""
    batch = _cuda_batch(128, 256)
    reference = TorchPPO.Config().make()
    triton_ppo = TritonPPO.Config().make()
    batch["decoded"] = batch["decoded"].to(dtype)
    inputs = (batch["decoded"], batch["actions"], batch["action_mask"])
    expected = reference.log_probs(*inputs)
    actual = triton_ppo.log_probs(*inputs)
    assert torch.equal(actual.values, expected.values)
    torch.testing.assert_close(actual.logps, expected.logps, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(actual.new_lp, expected.new_lp, rtol=1e-5, atol=1e-4)

    expected_adv, expected_ret = reference.advantage(
        expected.values,
        batch["rewards"],
        batch["terminals"],
    )
    actual_adv, actual_ret = triton_ppo.advantage(
        expected.values,
        batch["rewards"],
        batch["terminals"],
    )
    # The kernel contracts ``decay * lastlam`` into its add; bf16 storage hides most
    # of that ulp, and torch's bf16 tolerance holds the rest.
    torch.testing.assert_close(actual_adv, expected_adv)
    torch.testing.assert_close(actual_ret, expected_ret)

    expected_loss = reference.loss(
        expected,
        decoded=batch["decoded"],
        actions=batch["actions"],
        old_logprobs=batch["old_logprobs"],
        advantages=expected_adv,
        values=batch["values"],
        returns=expected_ret,
    )
    actual_loss = triton_ppo.loss(
        LogProbs(values=actual.values, logps=expected.logps, new_lp=expected.new_lp),
        decoded=batch["decoded"],
        actions=batch["actions"],
        old_logprobs=batch["old_logprobs"],
        advantages=expected_adv,
        values=batch["values"],
        returns=expected_ret,
    )
    torch.testing.assert_close(
        actual_loss.grad_logits,
        expected_loss.grad_logits,
        rtol=1e-3,
        atol=1e-7,
    )
    torch.testing.assert_close(
        actual_loss.grad_values,
        expected_loss.grad_values,
        rtol=1e-5,
        atol=1e-7,
    )
    torch.testing.assert_close(
        actual_loss.losses,
        expected_loss.losses,
        rtol=1e-3,
        atol=1e-5,
    )


@pytest.mark.gpu_triton
def test_the_advantage_kernel_stores_the_dtype_its_inputs_promote_to() -> None:
    """fp32 rewards give fp32 advantages and returns: the reference's, to 1e-5.

    The tolerance leaves room for a contracted ``decay * lastlam``, one fp32
    ulp per step that the decay keeps from growing.
    """
    batch = _cuda_batch(128, 256)
    reference = TorchPPO.Config().make()
    triton_ppo = TritonPPO.Config().make()
    values, terminals = batch["values"], batch["terminals"]
    rewards = batch["rewards"].float() * 1.1
    expected = reference.advantage(values, rewards, terminals)
    actual = triton_ppo.advantage(values, rewards, terminals)
    rounded = triton_ppo.advantage(values, rewards.bfloat16(), terminals)
    for ours, theirs, low in zip(actual, expected, rounded, strict=True):
        assert ours.dtype == torch.float32
        assert low.dtype == torch.bfloat16
        torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-6)


# One name per GPU model with a golden, so each golden file has this test as its owner;
# minting for a new model starts by adding its name here.
@pytest.mark.gpu_triton
@pytest.mark.parametrize("name", ["triton_ppo_nvidia-h200"])
def test_the_triton_rule_matches_its_golden_on_the_gpu(
    request: pytest.FixtureRequest,
    name: str,
) -> None:
    """The kernels on one 128-row minibatch of 256 steps, frozen per GPU model.

    The inputs are drawn on the CPU with portable draws, so only the GPU and
    its kernels decide the bits; another model's golden skips.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    model = re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name().lower())
    if name != f"triton_ppo_{model.strip('-')}":
        pytest.skip(f"{name} is not this GPU's golden ({model})")
    batch = {
        key: value.cuda()
        for key, value in portable_minibatch(rows=128, horizon=256).items()
    }
    lines = rule_entries(TritonPPO.Config().make(), batch)
    assert_text_golden(
        request,
        test_file=__file__,
        name=name,
        rendered="\n".join(lines),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
