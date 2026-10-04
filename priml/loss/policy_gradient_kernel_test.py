"""Tests for the rule's Triton kernels: their refusal off CUDA, the kernels on a GPU.

The kernels are checked against the torch reference at tolerance: Triton's
``exp`` is the fast ``ex2.approx`` and its contractions round once where torch
rounds twice. The torch reference itself is pinned bit for bit by the portable
golden in ``policy_gradient_test.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from priml.loss.policy_gradient import LogProbs, TorchPPO
from priml.loss.policy_gradient_kernel import TritonPPO, _require_cuda
from priml.testing.policy_gradient import random_minibatch


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


def test_invalid_horizon_and_cpu_tensor_are_refused_without_cuda() -> None:
    rule = TritonPPO.Config().make()
    with pytest.raises(ValueError, match="positive multiple"):
        rule.check_horizon(0)
    with pytest.raises(ValueError, match="positive multiple"):
        rule.check_horizon(TritonPPO.Config.ADVANTAGE_WIDTH + 1)
    with pytest.raises(
        ValueError,
        match=r"^TritonPPO runs on a CUDA device, not cpu; select TorchPPO to run the rule elsewhere$",
    ):
        _require_cuda(torch.zeros(2))


def test_loss_sums_are_added_across_program_blocks() -> None:
    rule = TritonPPO.Config().make()
    partials = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [10, 20, 30, 40, 50, 60, 70, 80]],
        dtype=torch.float32,
    )
    assert torch.equal(
        rule.sum_blocks(partials),
        torch.tensor([11, 22, 33, 44, 55, 66, 77, 88], dtype=torch.float32),
    )


def test_the_kernels_take_the_reference_rules_coefficients() -> None:
    config = TritonPPO.Config()
    assert isinstance(config, TorchPPO.Config)
    config.entropy_coefficient = 0.25
    config.block = 128
    config.num_warps = 4
    rule = config.make()
    assert rule.entropy_coefficient == 0.25
    assert rule.block == 128
    assert rule.num_warps == 4


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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
