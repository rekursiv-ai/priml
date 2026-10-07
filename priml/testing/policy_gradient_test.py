"""Tests for deterministic policy-gradient minibatches and outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    import pytest

from priml.loss.policy_gradient import TorchPPO
from priml.testing.policy_gradient import (
    portable_minibatch,
    random_minibatch,
    rule_outputs,
)


def test_random_minibatch_has_legal_actions_and_expected_fields() -> None:
    batch = random_minibatch(rows=2, horizon=3, num_actions=5, seed=7)

    assert set(batch) == {
        "decoded",
        "actions",
        "old_logprobs",
        "action_mask",
        "rewards",
        "terminals",
        "values",
    }
    assert batch["decoded"].shape == (2, 3, 6)
    assert batch["decoded"].dtype == torch.bfloat16
    assert batch["actions"].shape == (2, 3)
    assert batch["actions"].dtype == torch.float32
    assert batch["action_mask"].shape == (2, 3, 5)
    assert torch.all(batch["action_mask"][..., 0] == 1)
    assert torch.all(
        batch["action_mask"].gather(-1, batch["actions"].long()[..., None]) != 0,
    )
    assert batch["old_logprobs"].dtype == torch.bfloat16
    assert batch["rewards"].dtype == torch.bfloat16
    assert batch["terminals"].dtype == torch.bfloat16
    assert batch["values"].dtype == torch.bfloat16


def test_random_minibatch_seed_replays_the_same_draws() -> None:
    default_seed = random_minibatch(rows=2, horizon=3)
    first = random_minibatch(rows=2, horizon=3, seed=0)
    second = random_minibatch(rows=2, horizon=3, seed=0)

    for name in first:
        assert torch.equal(first[name], second[name]), name
        assert torch.equal(default_seed[name], first[name]), name

    generator = torch.Generator().manual_seed(0)
    decoded = (torch.randn(2, 3, 44, generator=generator) * 2).bfloat16()
    action_mask = (torch.rand(2, 3, 43, generator=generator) > 0.2).bfloat16()
    action_mask[..., 0] = 1
    actions = (
        torch.multinomial(
            action_mask.float().reshape(-1, 43),
            1,
            generator=generator,
        )
        .reshape(2, 3)
        .float()
    )
    expected = {
        "decoded": decoded,
        "actions": actions,
        "old_logprobs": (-torch.rand(2, 3, generator=generator) * 3).bfloat16(),
        "action_mask": action_mask,
        "rewards": (torch.randn(2, 3, generator=generator) * 0.5)
        .clamp(-1, 1)
        .bfloat16(),
        "terminals": (torch.rand(2, 3, generator=generator) < 0.02).bfloat16(),
        "values": torch.randn(2, 3, generator=generator).bfloat16(),
    }
    for name, value in expected.items():
        assert torch.equal(first[name], value), name


def test_random_minibatch_uses_strict_probability_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    randn_factory = torch.randn
    rand_draws = 0
    randn_draws = 0

    def rand(
        *shape: int,
        generator: torch.Generator,
        device: str,
    ) -> torch.Tensor:
        nonlocal rand_draws
        del generator
        rand_draws += 1
        value = (0.2, 0.5, 0.02)[rand_draws - 1]
        return torch.full(shape, value, device=device)

    def randn(*shape: int, generator: torch.Generator, device: str) -> torch.Tensor:
        nonlocal randn_draws
        randn_draws += 1
        if randn_draws == 2:
            return torch.tensor([[-3.0, 3.0, 0.0], [0.0, 0.0, 0.0]], device=device)
        return randn_factory(*shape, generator=generator, device=device)

    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randn", randn)

    batch = random_minibatch(rows=2, horizon=3, num_actions=4)

    assert rand_draws == 3
    assert batch["action_mask"][0, 0, 1] == 0
    assert batch["rewards"][0, 0] == -1
    assert batch["rewards"][0, 1] == 1
    assert torch.count_nonzero(batch["terminals"]) == 0


def test_random_minibatch_passes_device_to_torch_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator_factory = torch.Generator
    rand_factory = torch.rand
    randn_factory = torch.randn
    generator_devices: list[str] = []
    draw_devices: list[str] = []

    def generator(*, device: str) -> torch.Generator:
        generator_devices.append(device)
        return generator_factory(device=device)

    def rand(*shape: int, generator: torch.Generator, device: str) -> torch.Tensor:
        draw_devices.append(device)
        return rand_factory(*shape, generator=generator, device=device)

    def randn(*shape: int, generator: torch.Generator, device: str) -> torch.Tensor:
        draw_devices.append(device)
        return randn_factory(*shape, generator=generator, device=device)

    monkeypatch.setattr(torch, "Generator", generator)
    monkeypatch.setattr(torch, "rand", rand)
    monkeypatch.setattr(torch, "randn", randn)

    random_minibatch(rows=2, horizon=3, device="cpu")

    assert generator_devices == ["cpu"]
    assert draw_devices == ["cpu"] * 6


def test_portable_minibatch_has_consistent_fields_and_legal_actions() -> None:
    batch = portable_minibatch(rows=2, horizon=3)

    assert batch["decoded"].shape == (2, 3, 44)
    assert batch["actions"].shape == (2, 3)
    assert batch["action_mask"].shape == (2, 3, 43)
    assert batch["actions"].dtype == torch.float32
    assert all(value.device.type == "cpu" for value in batch.values())
    assert all(
        value.dtype == torch.bfloat16
        for name, value in batch.items()
        if name != "actions"
    )
    assert torch.all(batch["action_mask"][..., 0] == 1)
    assert torch.all(
        batch["action_mask"].gather(-1, batch["actions"].long()[..., None]) != 0,
    )

    generator = torch.Generator().manual_seed(0)
    mask = torch.rand((2, 3, 43), generator=generator) > 0.2
    mask[..., 0] = True
    scores = torch.where(mask, torch.rand((2, 3, 43), generator=generator), -1.0)
    rewards = torch.rand((2, 3), generator=generator) * 2 - 1
    expected = {
        "decoded": (
            (torch.rand((2, 3, 44), generator=generator) * 2 - 1) * 0.25
        ).bfloat16(),
        "actions": scores.argmax(dim=-1).float(),
        "old_logprobs": (
            (torch.rand((2, 3), generator=generator) * 2 - 1) * 0.25 - 3.55
        ).bfloat16(),
        "action_mask": mask.bfloat16(),
        "rewards": torch.where(
            torch.rand((2, 3), generator=generator) < 0.8,
            0.0,
            rewards,
        ).bfloat16(),
        "terminals": (torch.rand((2, 3), generator=generator) < 0.02).bfloat16(),
        "values": ((torch.rand((2, 3), generator=generator) * 2 - 1) * 2.0).bfloat16(),
    }
    for name, value in expected.items():
        assert torch.equal(batch[name], value), name


def test_portable_minibatch_preserves_action_for_tied_masked_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def zeros(shape: tuple[int, ...], *, generator: torch.Generator) -> torch.Tensor:
        del generator
        return torch.zeros(shape)

    monkeypatch.setattr(torch, "rand", zeros)
    batch = portable_minibatch(rows=2, horizon=3)

    action_mask = torch.zeros((2, 3, 43), dtype=torch.bool)
    action_mask[..., 0] = True
    scores = torch.zeros((2, 3, 43))
    expected = scores.masked_fill(~action_mask, float("-inf")).argmax(-1).float()

    assert torch.equal(batch["actions"], expected)
    assert torch.equal(batch["action_mask"], action_mask.bfloat16())


def test_portable_minibatch_uses_strict_probability_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draws = 0

    def rand(shape: tuple[int, ...], *, generator: torch.Generator) -> torch.Tensor:
        nonlocal draws
        del generator
        draws += 1
        value = (0.2, 0.5, 0.8, 0.5, 0.5, 0.8, 0.02, 0.5)[draws - 1]
        result = torch.full(shape, value)
        if draws == 2:
            result[0, 0, 1] = 0.9
        return result

    monkeypatch.setattr(torch, "rand", rand)

    batch = portable_minibatch(rows=2, horizon=3)

    assert draws == 8
    assert batch["action_mask"][0, 0, 1] == 0
    assert batch["actions"][0, 0] == 0
    assert batch["rewards"][0, 0] == torch.tensor(0.6015625).bfloat16()
    assert torch.count_nonzero(batch["terminals"]) == 0


def test_rule_outputs_requests_owned_cpu_copies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    to_factory = torch.Tensor.to
    transfers: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def to(
        self: torch.Tensor,
        device: torch.device | str | None = None,
        *,
        copy: bool = False,
    ) -> torch.Tensor:
        args: tuple[object, ...] = () if device is None else (device,)
        kwargs: dict[str, object] = {"copy": copy}
        transfers.append((args, kwargs))
        return to_factory(self, device, copy=copy)

    monkeypatch.setattr(torch.Tensor, "to", to)

    rule_outputs(
        TorchPPO.Config().make(),
        portable_minibatch(rows=2, horizon=3),
    )

    cpu_transfers = [transfer for transfer in transfers if transfer[0] == ("cpu",)]
    assert cpu_transfers == [(("cpu",), {"copy": True})] * 8


def test_rule_outputs_returns_every_stage_detached_on_cpu() -> None:
    outputs = rule_outputs(
        TorchPPO.Config().make(),
        portable_minibatch(rows=2, horizon=3),
    )

    assert set(outputs) == {
        "values",
        "logps",
        "new_lp",
        "advantages",
        "returns",
        "grad_logits",
        "grad_values",
        "losses",
    }
    assert outputs["values"].shape == (2, 3)
    assert outputs["logps"].shape == (2, 3, 43)
    assert outputs["new_lp"].shape == (2, 3)
    assert outputs["advantages"].shape == (2, 3)
    assert outputs["returns"].shape == (2, 3)
    assert outputs["grad_logits"].shape == (2, 3, 43)
    assert outputs["grad_values"].shape == (2, 3)
    assert outputs["losses"].shape == (8,)
    assert all(value.device.type == "cpu" for value in outputs.values())
    assert all(not value.requires_grad for value in outputs.values())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
