"""Tests for the openai/math rollout data layer."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from priml.baselines.openaimath.data import OpenMathRolloutData, record_from_payload
from priml.baselines.tmax.rollouts import (
    RolloutRecord,
    group_advantages,
    read_rollout_records,
)


FIXTURE = Path(__file__).resolve().parent / "testdata" / "smoke_rollout.jsonl"


def _payload(**overrides: object) -> dict[str, object]:
    """Build one collected record."""
    payload: dict[str, object] = {
        "step": 0,
        "sample_idx": 1,
        "prompt_idx": 0,
        "prompt_tokens": [10, 11, 12],
        "response_tokens": [20, 21, 22],
        "logprobs": [-0.5, -0.25, -1.0],
        "reward": 1.0,
        "finish_reason": "stop",
        "tool_mask": [1, 1, 0],
    }
    payload.update(overrides)
    return payload


def test_record_from_payload_reads_a_collected_transcript() -> None:
    """A collected record becomes the learner's record type."""
    record = record_from_payload(_payload())
    assert record == RolloutRecord(
        step=0,
        sample_idx=1,
        prompt_idx=0,
        prompt_tokens=(10, 11, 12),
        response_tokens=(20, 21, 22),
        logprobs=(-0.5, -0.25, -1.0),
        reward=1.0,
        finish_reason="stop",
        tool_mask=(1, 1, 0),
    )


def test_record_from_payload_tolerates_a_missing_mask() -> None:
    """A record without a tool mask keeps the mask absent rather than empty."""
    record = record_from_payload(_payload(tool_mask=None))
    assert record.tool_mask is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"logprobs": [-0.5, -0.25]}, "logprobs for"),
        ({"tool_mask": [1, 1]}, "tool-mask values"),
        ({"reward": float("inf")}, "reward"),
        ({"response_tokens": [20, 21.5, 22]}, "response_tokens"),
        ({"prompt_tokens": "not an array"}, "prompt_tokens"),
    ],
)
def test_record_from_payload_rejects_a_malformed_record(
    overrides: dict[str, object],
    match: str,
) -> None:
    """A transcript whose arrays disagree fails the update, not the objective."""
    with pytest.raises(ValueError, match=match):
        record_from_payload(_payload(**overrides))


def test_record_from_payload_rejects_a_missing_field() -> None:
    """A record missing a required field is refused by name."""
    payload = _payload()
    del payload["finish_reason"]
    with pytest.raises(ValueError, match="missing"):
        record_from_payload(payload)


def test_defaults_keep_the_tmax_group_filters() -> None:
    """Lemma groups are mixed, so TMax's zero-spread filter and refill stay on."""
    config = OpenMathRolloutData.Config()
    assert config.filter_zero_std_samples is True
    assert config.active_sampling is True
    assert config.pack_length == 81_920
    assert config.mask_tool_use is True


def test_smoke_fixture_is_one_group_with_a_nonzero_advantage() -> None:
    """The fixture's two rewards differ, so its centered advantage is nonzero."""
    records = read_rollout_records(FIXTURE)
    assert len(records) == 2
    assert {record.step for record in records} == {0}
    assert {record.prompt_idx for record in records} == {0}
    assert [record.sample_idx for record in records] == [0, 1]
    assert sorted(record.reward for record in records) == [0.0, 1.0]
    advantages = group_advantages(
        [record.reward for record in records],
        num_samples_per_prompt=2,
    )
    assert torch.allclose(advantages, torch.tensor([0.5, -0.5]))


def test_smoke_fixture_packs_through_the_data_source() -> None:
    """The fixture trains: one packed row, a nonzero gradient's worth of tokens."""
    config = OpenMathRolloutData.Config(
        base_dir=FIXTURE.parent,
        working_dir=FIXTURE.name,
        records_per_update=2,
        num_samples_per_prompt=2,
        mask_tool_use=False,
    )
    data = OpenMathRolloutData(config.finalize())
    batches = list(data.train_dataloader())
    assert len(batches) == 1
    rows = cast(tuple[dict[str, object], ...], batches[0]["rows"])
    assert len(rows) == 1
    assert cast(int, batches[0]["response_token_count"]) > 0
    response_mask = cast(torch.Tensor, rows[0]["response_mask"])
    assert int(response_mask[1:].bool().sum()) > 0
    # Both rollouts are in one row, so the row holds two attention segments.
    segments = cast(torch.Tensor, rows[0]["attention_mask"])
    assert int(segments.max().item()) == 2


def test_data_source_rejects_a_group_size_of_one() -> None:
    """A group of one has no reward spread, so the filter cannot be on."""
    config = OpenMathRolloutData.Config(
        base_dir=FIXTURE.parent,
        working_dir=FIXTURE.name,
        records_per_update=2,
        num_samples_per_prompt=1,
        filter_zero_std_samples=True,
        active_sampling=True,
        mask_tool_use=False,
    )
    with pytest.raises(ValueError, match="num_samples_per_prompt"):
        OpenMathRolloutData(config)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
