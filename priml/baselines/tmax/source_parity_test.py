"""Offline exact comparisons against TMax commit ``6d3d606``."""

from __future__ import annotations

from pathlib import Path
from typing import Final
from typing_extensions import TypedDict

import torch

from priml.baselines.tmax.objective import (
    binary_divergence,
    dppo_mask,
    dppo_token_loss,
    importance_ratio,
)
from priml.baselines.tmax.rollouts import RolloutRecord, group_advantages, pack_rollouts
from priml.baselines.tmax.scoring import log_softmax_and_gather
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import read_tensors


ROOT = Path(__file__).parent
_PACKED_TENSOR_FIELDS: Final = (
    "query_responses",
    "attention_mask",
    "position_ids",
    "response_mask",
    "prompt_mask",
    "rollout_sample_ids",
    "model_steps",
    "vllm_logprobs",
    "dones",
)
_PACKED_SOURCE_FIELDS: Final = (
    *_PACKED_TENSOR_FIELDS,
    "num_actions",
    "packed_seq_lens",
)
_OBJECTIVE_KEYS: Final = frozenset(
    {
        "advantages",
        "divergence",
        "gathered_logprobs",
        "mask_divergence",
        "pg_losses",
        "policy_mask",
    }
    | {
        f"packed/row{row}/{field}"
        for row in range(2)
        for field in _PACKED_SOURCE_FIELDS
    },
)


class RolloutPayload(TypedDict):
    """Typed shape of the literal rollout records compared against the golden."""

    step: int
    sample_idx: int
    prompt_idx: int
    prompt_tokens: tuple[int, ...]
    response_tokens: tuple[int, ...]
    logprobs: tuple[float, ...]
    reward: float
    finish_reason: str
    tool_mask: tuple[int, ...]


def test_objective_matches_pinned_upstream_outputs() -> None:
    """Match every objective tensor recorded from the pinned upstream code."""
    expected = read_tensors(ROOT / "testdata" / "objective.pt")
    assert expected.keys() == _OBJECTIVE_KEYS
    # These tensors keep their leading batch dimension of 1 because that is
    # the shape the upstream reference recorded into the golden.
    behavior = torch.tensor([[-0.2, -0.4, -0.6, -0.3, -0.5]])
    policy = torch.tensor([[-0.1, -0.6, -0.3, -0.5, -0.7]])
    response_mask = torch.tensor([[True, True, False, True, True]])
    advantages = torch.tensor([[0.5, -0.5, 0.0, -0.25, 0.25]])
    with host_agnostic_numerics():
        ratio = importance_ratio(policy, behavior)
        divergence = binary_divergence(
            behavior_logprobs=behavior,
            policy_logprobs=policy,
            response_mask=response_mask,
        )
        policy_mask, mask_divergence = dppo_mask(
            new_logprobs=policy,
            behavior_logprobs=behavior,
            advantages=advantages,
            ratio=ratio,
            response_mask=response_mask,
            divergence_threshold=0.1,
        )
        actual = {
            "divergence": divergence,
            "mask_divergence": mask_divergence,
            "policy_mask": policy_mask,
            "pg_losses": dppo_token_loss(
                advantages=advantages,
                ratio=ratio,
                policy_mask=policy_mask,
            ),
            "gathered_logprobs": log_softmax_and_gather(
                torch.tensor([[[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]]),
                torch.tensor([[2, 0]]),
            ),
        }
    for name, value in actual.items():
        torch.testing.assert_close(value, expected[name], rtol=0.0, atol=0.0)


def test_packing_and_advantage_outputs_match_pinned_upstream() -> None:
    """Match every packed row and advantage recorded from upstream TMax."""
    records: list[RolloutPayload] = [
        {
            "step": 20,
            "sample_idx": 10,
            "prompt_idx": 0,
            "prompt_tokens": (1, 2),
            "response_tokens": (3, 4, 0),
            "logprobs": (-0.2, -0.4, -0.6),
            "reward": 0.0,
            "finish_reason": "stop",
            "tool_mask": (1, 0, 1),
        },
        {
            "step": 21,
            "sample_idx": 11,
            "prompt_idx": 0,
            "prompt_tokens": (5,),
            "response_tokens": (6, 10),
            "logprobs": (-0.3, -0.5),
            "reward": 1.0,
            "finish_reason": "stop",
            "tool_mask": (1, 1),
        },
        {
            "step": 22,
            "sample_idx": 12,
            "prompt_idx": 1,
            "prompt_tokens": (7, 8, 9),
            "response_tokens": (12, 0),
            "logprobs": (-0.7, -0.8),
            "reward": 0.5,
            "finish_reason": "stop",
            "tool_mask": (0, 1),
        },
        {
            "step": 23,
            "sample_idx": 13,
            "prompt_idx": 1,
            "prompt_tokens": (11,),
            "response_tokens": (13, 14),
            "logprobs": (-0.9, -1.0),
            "reward": 0.0,
            "finish_reason": "stop",
            "tool_mask": (1, 1),
        },
    ]
    typed = [RolloutRecord(**record) for record in records]
    with host_agnostic_numerics():
        rows = pack_rollouts(
            typed,
            advantages=group_advantages(
                [record["reward"] for record in records],
                num_samples_per_prompt=2,
            ),
            pack_length=7,
            pad_token_id=0,
            mask_tool_use=True,
        )
    expected = read_tensors(ROOT / "testdata" / "objective.pt")
    assert len(rows) == 2
    for index, row in enumerate(rows):
        for name in _PACKED_TENSOR_FIELDS:
            torch.testing.assert_close(
                getattr(row, name),
                expected[f"packed/row{index}/{name}"],
                rtol=0.0,
                atol=0.0,
                equal_nan=name == "vllm_logprobs",
            )
        source_actions = expected[f"packed/row{index}/num_actions"]
        source_lengths = expected[f"packed/row{index}/packed_seq_lens"]
        assert row.num_actions == int(source_actions.sum())
        assert row.packed_seq_lens == tuple(int(length) for length in source_lengths)
        starts: list[int] = []
        start = 0
        for length in source_lengths:
            starts.append(start)
            start += int(length)
        assert row.original_responses == tuple(
            tuple(
                expected[f"packed/row{index}/query_responses"][
                    start + int(source_lengths[offset] - source_actions[offset]) : start
                    + int(source_lengths[offset])
                ].tolist(),
            )
            for offset, start in enumerate(starts)
        )
        torch.testing.assert_close(
            row.advantages,
            torch.cat((torch.zeros(1), expected["advantages"]))[row.response_mask],
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(
        group_advantages([0.0, 1.0, 0.5, 0.0], num_samples_per_prompt=2),
        expected["advantages"],
        rtol=0.0,
        atol=0.0,
    )
