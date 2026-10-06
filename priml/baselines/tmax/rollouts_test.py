"""Tests for the TMax artifact and packing seam."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import json

import pytest
import torch

from priml.baselines.tmax.rollouts import (
    RolloutRecord,
    group_advantages,
    pack_rollouts,
    read_rollout_records,
)


if TYPE_CHECKING:
    from priml.baselines.tmax.rollouts import AdvantageNormalization


ROOT = Path(__file__).parent


def _published_payload() -> dict[str, object]:
    """Return the first released rollout record, decoded."""
    return cast(
        dict[str, object],
        json.loads(
            (ROOT / "testdata" / "recorded_rollout.jsonl").read_text().splitlines()[0],
        ),
    )


def _read_written(
    tmp_path: Path,
    payload: dict[str, object],
) -> list[RolloutRecord]:
    """Write and parse one temporary rollout record."""
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return read_rollout_records(path)


def _record(*, mask: tuple[int, ...] | None = (1, 0)) -> RolloutRecord:
    """Return a small rollout record for packing tests."""
    return RolloutRecord(
        step=1,
        sample_idx=0,
        prompt_idx=0,
        prompt_tokens=(1, 2),
        response_tokens=(3, 4),
        logprobs=(-0.2, -0.4),
        reward=1.0,
        finish_reason="stop",
        tool_mask=mask,
    )


def test_group_advantages_matches_the_recorded_advantage_fixture() -> None:
    """Centered advantages reproduce the published fixture exactly."""
    payload = cast(
        dict[str, object],
        json.loads(
            (ROOT / "testdata" / "recorded_advantages.json").read_text(),
        ),
    )
    groups = cast(list[dict[str, object]], payload["groups"])
    group = groups[0]
    actual = group_advantages(
        cast(list[float], group["reward"]),
        num_samples_per_prompt=cast(
            int,
            payload["num_samples_per_prompt_rollout"],
        ),
        normalization="centered",
    )
    torch.testing.assert_close(actual, torch.tensor(group["advantage"]))


def test_packing_keeps_tool_mask_and_shifts_nothing_early() -> None:
    """Packed segments keep their own positions and masks."""
    records = [
        _record(mask=(1, 0)),
        RolloutRecord(
            step=2,
            sample_idx=1,
            prompt_idx=0,
            prompt_tokens=(5,),
            response_tokens=(6, 7),
            logprobs=(-0.5, -0.6),
            reward=0.0,
            finish_reason="stop",
            tool_mask=(0, 1),
        ),
    ]
    rows = pack_rollouts(
        records,
        advantages=torch.tensor([0.5, -0.5]),
        pack_length=10,
        pad_token_id=0,
        mask_tool_use=True,
    )
    row = rows[0]
    assert row.query_responses.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert row.response_mask.tolist() == [0, 0, 1, 0, 0, 0, 2]
    assert row.attention_mask.tolist() == [1, 1, 1, 1, 2, 2, 2]
    assert row.position_ids.tolist() == [0, 1, 2, 3, 0, 1, 2]
    assert row.dones.tolist() == [0, 0, 0, 1, 0, 0, 2]
    assert row.advantages.tolist() == [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -0.5]


def test_packing_refuses_published_records_without_tool_masks() -> None:
    """Refuse tool masking when a published record has no tool mask."""
    record = read_rollout_records(ROOT / "testdata" / "recorded_rollout.jsonl")[0]
    assert record.tool_mask is None
    with pytest.raises(ValueError, match="tool_mask"):
        pack_rollouts(
            [record],
            advantages=torch.tensor([record.reward]),
            pack_length=100_000,
            pad_token_id=0,
            mask_tool_use=True,
        )


def test_published_record_is_read_without_rewriting_its_schema() -> None:
    """Parse the published shard without renaming or remapping fields."""
    record = read_rollout_records(ROOT / "testdata" / "recorded_rollout.jsonl")[0]
    assert len(record.prompt_tokens) == 1_484
    assert len(record.response_tokens) == len(record.logprobs) == 1_370
    assert record.step == 91
    assert record.sample_idx == 216


def test_records_without_logprobs_are_refused(tmp_path: Path) -> None:
    """Require saved generation log probabilities for the DPPO ratio."""
    payload = _published_payload()
    del payload["logprobs"]
    with pytest.raises(ValueError, match="logprobs"):
        _read_written(tmp_path, payload)


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    """Treat a shard containing only blank lines as empty."""
    path = tmp_path / "rollouts.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    assert read_rollout_records(path) == []


def test_a_logprob_count_mismatch_is_refused(tmp_path: Path) -> None:
    """Require one saved log probability for every response token."""
    payload = _published_payload()
    payload["logprobs"] = [0.0]
    with pytest.raises(ValueError, match="logprobs for"):
        _read_written(tmp_path, payload)


def test_a_tool_mask_length_mismatch_is_refused(tmp_path: Path) -> None:
    """Require one tool-mask value for every response token."""
    payload = _published_payload()
    payload["tool_mask"] = [1]
    with pytest.raises(ValueError, match="tool-mask values"):
        _read_written(tmp_path, payload)


def test_a_non_array_field_is_refused(tmp_path: Path) -> None:
    """Name the field when an expected JSON array is a scalar."""
    payload = _published_payload()
    payload["response_tokens"] = 5
    with pytest.raises(ValueError, match=r"response_tokens: expected a JSON array"):
        _read_written(tmp_path, payload)


def test_a_non_numeric_response_token_is_refused(tmp_path: Path) -> None:
    """Report the exact position of a non-integer response token."""
    payload = _published_payload()
    payload["response_tokens"] = ["x"] * len(
        cast(list[object], payload["response_tokens"]),
    )
    with pytest.raises(ValueError, match=r"response_tokens\[0\].*integer"):
        _read_written(tmp_path, payload)


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf")])
def test_integer_fields_reject_inexact_values(tmp_path: Path, value: object) -> None:
    """Reject values that cannot represent an exact integer field."""
    payload = _published_payload()
    payload["sample_idx"] = value
    with pytest.raises(ValueError, match=r"rollouts.jsonl:1: sample_idx.*integer"):
        _read_written(tmp_path, payload)


def test_integer_fields_accept_integral_floats_and_preserve_large_ints(
    tmp_path: Path,
) -> None:
    """Accept integral floats without narrowing large integer IDs."""
    payload = _published_payload()
    payload["step"] = 3.0
    payload["sample_idx"] = 2**63 + 1
    record = _read_written(tmp_path, payload)[0]
    assert record.step == 3
    assert record.sample_idx == 2**63 + 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reward", True),
        ("reward", float("nan")),
        ("advantage", float("inf")),
    ],
)
def test_floating_fields_reject_bool_and_nonfinite(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Reject booleans and non-finite rewards or advantages."""
    payload = _published_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=rf"rollouts.jsonl:1: {field}.*finite"):
        _read_written(tmp_path, payload)


def test_float_array_errors_include_the_index(tmp_path: Path) -> None:
    """Include the bad array index in numeric parsing errors."""
    payload = _published_payload()
    logprobs = cast(list[object], payload["logprobs"])
    logprobs[2] = float("-inf")
    with pytest.raises(ValueError, match=r"logprobs\[2\].*finite"):
        _read_written(tmp_path, payload)


def test_group_advantages_rejects_a_non_positive_group_size() -> None:
    """Require at least one sample in an advantage group."""
    with pytest.raises(ValueError, match="must be positive"):
        group_advantages([1.0], num_samples_per_prompt=0)


def test_group_advantages_rejects_a_partial_group() -> None:
    """Reject an incomplete prompt group."""
    with pytest.raises(ValueError, match="whole number of groups"):
        group_advantages([1.0, 2.0, 3.0], num_samples_per_prompt=2)


def test_group_advantages_standard_normalization_divides_by_the_spread() -> None:
    """Standard normalization centers rewards and divides by their spread."""
    advantages = group_advantages(
        [0.0, 2.0],
        num_samples_per_prompt=2,
        normalization="standard",
    )
    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0]))


def test_group_advantages_maxrl_normalizes_only_a_positive_group() -> None:
    """MaxRL normalizes only groups with a positive mean reward."""
    advantages = group_advantages(
        [0.0, 2.0, -1.0, -3.0],
        num_samples_per_prompt=2,
        normalization="maxrl",
    )
    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0, 0.0, 0.0]))


def test_group_advantages_rejects_an_unknown_normalization() -> None:
    """Reject advantage normalization modes not published by TMax."""

    def invalid_normalization() -> str:
        """Return a value outside the typed normalization choices."""
        return "not-a-normalization"

    with pytest.raises(ValueError, match="Invalid advantage normalization"):
        group_advantages(
            [1.0],
            num_samples_per_prompt=1,
            normalization=cast("AdvantageNormalization", invalid_normalization()),
        )


def test_packing_rejects_a_mismatched_advantage_count() -> None:
    """Require exactly one advantage for every rollout record."""
    with pytest.raises(ValueError, match="advantages for"):
        pack_rollouts(
            [_record()],
            advantages=torch.tensor([1.0, 2.0]),
            pack_length=10,
            pad_token_id=0,
            mask_tool_use=False,
        )


def test_packing_rejects_a_non_positive_min_num_batches() -> None:
    """Require a positive minimum number of packed rows."""
    with pytest.raises(ValueError, match="min_num_batches must be positive"):
        pack_rollouts(
            [_record()],
            advantages=torch.tensor([1.0]),
            pack_length=10,
            pad_token_id=0,
            mask_tool_use=False,
            min_num_batches=0,
        )


def test_min_num_batches_shrinks_the_effective_pack_length() -> None:
    """More rows than the token budget allows forces the rollouts apart."""
    rows = pack_rollouts(
        [_record(), _record()],
        advantages=torch.tensor([1.0, 1.0]),
        pack_length=100,
        pad_token_id=0,
        mask_tool_use=False,
        min_num_batches=4,
    )
    assert len(rows) == 2


def test_packing_refuses_a_padded_prompt() -> None:
    """Refuse a rollout artifact containing a padded prompt."""
    record = RolloutRecord(
        step=1,
        sample_idx=0,
        prompt_idx=0,
        prompt_tokens=(1, 0),
        response_tokens=(3,),
        logprobs=(-0.2,),
        reward=1.0,
        finish_reason="stop",
        tool_mask=(1,),
    )
    with pytest.raises(ValueError, match="prompt contains pad token"):
        pack_rollouts(
            [record],
            advantages=torch.tensor([1.0]),
            pack_length=10,
            pad_token_id=0,
            mask_tool_use=False,
        )
