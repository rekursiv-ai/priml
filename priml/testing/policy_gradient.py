"""Minibatches and golden entries for the policy-gradient rule's tests.

The rule's coefficients are the caller's: priml's tests run its defaults, and a
study pins its own recipe beside its experiments. The portable minibatch draws
43 actions per step, as the random one does by default: a fixture size, and the
one the goldens' inputs were drawn at. A golden entry is a tensor's dtype,
shape and sha256, or a scalar's fp32 bits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib
import struct

from torch import Tensor

import torch

from priml.loss.policy_gradient import TorchPPO


if TYPE_CHECKING:
    from priml.loss.policy_gradient import PPO


def random_minibatch(
    *,
    rows: int,
    horizon: int,
    num_actions: int = 43,
    device: str = "cpu",
    seed: int = 0,
) -> dict[str, Tensor]:
    """Draw a minibatch with normal logits and values and a legal sampled action.

    Args:
      rows: Rows of the minibatch.
      horizon: Steps per row.
      num_actions: Logits per step.
      device: Where the draws happen; each device's generator has its own
        stream.
      seed: The draws' seed.

    Returns:
      batch: ``decoded``, ``actions``, ``old_logprobs``, ``action_mask``,
        ``rewards``, ``terminals`` and ``values``, in the rule's dtypes.

    """
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, device=device)

    def rand(*shape: int) -> Tensor:
        return torch.rand(*shape, generator=generator, device=device)

    decoded = (randn(rows, horizon, num_actions + 1) * 2).bfloat16()
    action_mask = (rand(rows, horizon, num_actions) > 0.2).bfloat16()
    action_mask[..., 0] = 1
    legal = action_mask.float()
    actions = (
        torch.multinomial(
            legal.reshape(-1, num_actions),
            1,
            generator=generator,
        )
        .reshape(rows, horizon)
        .float()
    )
    return {
        "decoded": decoded,
        "actions": actions,
        "old_logprobs": (-rand(rows, horizon) * 3).bfloat16(),
        "action_mask": action_mask,
        "rewards": (randn(rows, horizon) * 0.5).clamp(-1, 1).bfloat16(),
        "terminals": (rand(rows, horizon) < 0.02).bfloat16(),
        "values": randn(rows, horizon).bfloat16(),
    }


# The logits are small, so a legal action's log-probability is near ``-log(35)``, and
# the old ones scatter around it: ratios fall inside the clip and on both sides of it.
def portable_minibatch(*, rows: int, horizon: int) -> dict[str, Tensor]:
    """Draw a minibatch whose bits are the same on every host, and a legal action.

    Args:
      rows: Rows of the minibatch.
      horizon: Steps per row.

    Returns:
      batch: As :func:`random_minibatch`'s, on the CPU.

    """
    generator = torch.Generator().manual_seed(0)
    num_actions = 43

    def uniform(*shape: int, bound: float) -> Tensor:
        return _portable_uniform(*shape, bound=bound, generator=generator)

    def rand(*shape: int) -> Tensor:
        return torch.rand(shape, generator=generator)

    action_mask = rand(rows, horizon, num_actions) > 0.2
    action_mask[..., 0] = True
    # The legal action with the largest draw.
    scores = torch.where(action_mask, rand(rows, horizon, num_actions), -1.0)
    rewards = uniform(rows, horizon, bound=1.0)
    return {
        "decoded": uniform(rows, horizon, num_actions + 1, bound=0.25).bfloat16(),
        "actions": scores.argmax(dim=-1).float(),
        "old_logprobs": (uniform(rows, horizon, bound=0.25) - 3.55).bfloat16(),
        "action_mask": action_mask.bfloat16(),
        "rewards": torch.where(rand(rows, horizon) < 0.8, 0.0, rewards).bfloat16(),
        "terminals": (rand(rows, horizon) < 0.02).bfloat16(),
        "values": uniform(rows, horizon, bound=2.0).bfloat16(),
    }


def rule_entries(rule: PPO, batch: dict[str, Tensor]) -> list[str]:
    """Run the rule's three stages; digest every output.

    Args:
      rule: The rule under test, with its coefficients.
      batch: A minibatch, as :func:`portable_minibatch` draws it.

    Returns:
      lines: One golden entry per output, then one per loss term.

    """
    logprobs = rule.log_probs(
        batch["decoded"],
        batch["actions"],
        batch["action_mask"],
    )
    advantages, returns = rule.advantage(
        logprobs.values,
        batch["rewards"],
        batch["terminals"],
    )
    loss = rule.loss(
        logprobs,
        decoded=batch["decoded"],
        actions=batch["actions"],
        old_logprobs=batch["old_logprobs"],
        advantages=advantages,
        values=batch["values"],
        returns=returns,
    )
    outputs = {
        "values": logprobs.values,
        "logps": logprobs.logps,
        "new_lp": logprobs.new_lp,
        "advantages": advantages,
        "returns": returns,
        "grad_logits": loss.grad_logits,
        "grad_values": loss.grad_values,
        "losses": loss.losses,
    }
    lines = [f"{name} {_digest(value)}" for name, value in outputs.items()]
    return lines + [
        f"loss {name} {_fp32(value)}"
        for name, value in zip(TorchPPO.Config.LOSS_NAMES, loss.losses, strict=True)
    ]


# ``torch.rand`` fills multiples of 2^-24 from integer draws, and ``2u - 1`` is exact, so
# only the correctly rounded product with ``bound`` rounds. ``randn`` is not portable:
# its Box-Muller transform runs SLEEF's ``log``/``cos`` under AVX2 and libm's elsewhere
# (up to 6 ULP apart).
def _portable_uniform(
    *shape: int,
    bound: float,
    generator: torch.Generator,
) -> Tensor:
    """Draw fp32 ``U(-bound, bound)`` whose bits are the same on every host."""
    return (torch.rand(shape, generator=generator) * 2 - 1) * bound


def _digest(value: Tensor) -> str:
    """Return a tensor's golden entry: its dtype, shape and the sha256 of its bytes."""
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    sha = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
    return f"{value.dtype} {tuple(value.shape)} {sha}"


def _fp32(value: Tensor) -> str:
    """Return a one-element fp32 tensor's golden entry: its bits, then its decimal."""
    packed = struct.pack("<f", float(value))
    single = struct.unpack("<f", packed)[0]
    return f"0x{struct.unpack('<I', packed)[0]:08x} {single!r}"
