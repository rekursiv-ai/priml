"""Tests for exactly-once rollout persistence and crash recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, override

import json

from treekle import json_loads

import numpy as np
import pytest
import torch

from priml.baselines.tmax.scripts import rollout_writer
from priml.baselines.tmax.scripts.rollout_writer import RolloutWriter


if TYPE_CHECKING:
    from pathlib import Path


def _records(step: int, count: int = 2) -> list[dict[str, object]]:
    return [{"step": step, "sample_idx": index} for index in range(count)]


def _shard(tmp_path: Path, index: int = 0) -> Path:
    return tmp_path / f"mock_rollouts_{index:06d}.jsonl"


def _marker(tmp_path: Path, step: int) -> Path:
    return tmp_path / f"mock_live_step_{step:08d}.json"


def _intent(tmp_path: Path, step: int) -> Path:
    return tmp_path / f"mock_live_step_{step:08d}.pending.json"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


# -- loading ------------------------------------------------------------------


def test_blank_lines_in_a_shard_are_record_boundaries(tmp_path: Path) -> None:
    _shard(tmp_path).write_text(
        '\n{"step": 0, "sample_idx": 0}\n\n{"step": 0, "sample_idx": 1}\n',
        encoding="utf-8",
    )
    assert RolloutWriter(tmp_path, "mock").completed_steps(2) == {0}


def test_invalid_json_before_the_tail_is_corruption(tmp_path: Path) -> None:
    _shard(tmp_path).write_text(
        '{"step": 0, "sample_idx": 0}\nnot json\n{"step": 0, "sample_idx": 1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="before the recoverable tail"):
        RolloutWriter(tmp_path, "mock")


def test_a_complete_final_record_without_a_newline_is_a_torn_tail(
    tmp_path: Path,
) -> None:
    _shard(tmp_path).write_text(
        '{"step": 0, "sample_idx": 0}\n{"step": 0, "sample_idx": 1}',
        encoding="utf-8",
    )
    writer = RolloutWriter(tmp_path, "mock")
    with pytest.raises(ValueError, match="interrupted final record"):
        writer.completed_steps(2)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("not json", "Invalid live rollout commit marker"),
        ({"version": 1}, "Invalid live rollout commit marker"),
        ({"version": 2, "step": 3}, "unsupported version"),
        ({"version": 1, "step": 4}, "filename/step mismatch"),
    ],
)
def test_malformed_commit_markers_are_refused(
    tmp_path: Path,
    payload: object,
    match: str,
) -> None:
    path = _marker(tmp_path, 3)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        _write_json(path, payload)
    with pytest.raises(ValueError, match=match):
        RolloutWriter(tmp_path, "mock")


def test_a_second_marker_for_one_step_is_refused(tmp_path: Path) -> None:
    _write_json(_marker(tmp_path, 3), {"version": 1, "step": 3})
    _write_json(tmp_path / "mock_live_step_3.json", {"version": 1, "step": 3})
    with pytest.raises(ValueError, match="filename/step mismatch"):
        RolloutWriter(tmp_path, "mock")


def test_marker_loading_checks_duplicates_by_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = RolloutWriter(tmp_path, "mock")
    _write_json(_marker(tmp_path, 3), {"version": 1, "step": 3})
    writer._markers[3] = {}
    monkeypatch.setattr(writer, "_reset_index", lambda: None)
    with pytest.raises(ValueError, match="Duplicate live rollout marker"):
        writer._load_markers()


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        {"version": 1, "step": 1, "shard": "mock_rollouts_000000.jsonl"},
        {"version": 1, "step": 1, "shard": "other.jsonl", "byte_offset": 0},
        {
            "version": 1,
            "step": 1,
            "shard": "mock_rollouts_000000.txt",
            "byte_offset": 0,
        },
        {
            "version": 1,
            "step": 1,
            "shard": "mock_rollouts_000000.jsonl",
            "byte_offset": -1,
        },
        {
            "version": 1,
            "step": 1,
            "shard": "mock_rollouts_000000.jsonl",
            "byte_offset": 5,
        },
        {
            "version": 2,
            "step": 1,
            "shard": "mock_rollouts_000000.jsonl",
            "byte_offset": 0,
        },
    ],
)
def test_malformed_write_intents_are_refused(tmp_path: Path, payload: object) -> None:
    path = _intent(tmp_path, 1)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        _write_json(path, payload)
    with pytest.raises(ValueError, match="Invalid live rollout write intent"):
        RolloutWriter(tmp_path, "mock")


def test_an_intent_offset_past_the_shard_end_is_refused(tmp_path: Path) -> None:
    RolloutWriter(tmp_path, "mock").write(0, _records(0, 1))
    size = _shard(tmp_path).stat().st_size
    _write_json(
        _intent(tmp_path, 1),
        {
            "version": 1,
            "step": 1,
            "shard": _shard(tmp_path).name,
            "byte_offset": size + 1,
        },
    )
    with pytest.raises(ValueError, match="Invalid live rollout write intent"):
        RolloutWriter(tmp_path, "mock")


def test_intent_loading_checks_duplicates_by_step(
    tmp_path: Path,
) -> None:
    writer = RolloutWriter(tmp_path, "mock")
    _write_json(
        _intent(tmp_path, 1),
        {"version": 1, "step": 1, "shard": _shard(tmp_path).name, "byte_offset": 0},
    )
    writer._pending[1] = {}
    with pytest.raises(ValueError, match="Duplicate live rollout write intent"):
        writer._load_intents()


# -- commit validation --------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["records_per_update", "sample_indices", "prompt_loader_state_after"],
)
def test_a_marker_with_a_broken_contract_is_refused(
    tmp_path: Path,
    field: str,
) -> None:
    writer = RolloutWriter(tmp_path, "mock")
    writer.write_live(0, _records(0), prompt_loader_state_after={"position": 1})
    marker = json_loads(_marker(tmp_path, 0).read_bytes(), dict[str, object])
    marker[field] = "broken"
    _write_json(_marker(tmp_path, 0), marker)
    with pytest.raises(ValueError, match="invalid record or prompt-cursor contract"):
        RolloutWriter(tmp_path, "mock").committed_live_steps(2)


def test_the_cursor_of_an_uncommitted_step_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not committed"):
        RolloutWriter(tmp_path, "mock").live_cursor_after(0, 2)


def test_live_records_must_hold_every_sample_index(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly once"):
        RolloutWriter(tmp_path, "mock").write_live(
            0,
            [{"step": 0, "sample_idx": 1}],
            prompt_loader_state_after={},
        )


# -- writing ------------------------------------------------------------------


def test_records_of_another_step_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="belonging to another step"):
        RolloutWriter(tmp_path, "mock").write(0, _records(1))


def test_duplicate_records_within_one_write_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        RolloutWriter(tmp_path, "mock").write(
            0,
            [{"step": 0, "sample_idx": 0}, {"step": 0, "sample_idx": 0}],
        )


def test_a_full_shard_rolls_over_to_the_next_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_writer, "ROLLOUT_SHARD_SIZE", 2)
    writer = RolloutWriter(tmp_path, "mock")
    assert writer.write(0, _records(0)) == _shard(tmp_path, 0)
    assert writer.write(1, _records(1)) == _shard(tmp_path, 1)


# -- live recovery ------------------------------------------------------------


def test_live_recovery_refuses_uncommitted_intents_before_the_frontier(
    tmp_path: Path,
) -> None:
    writer = RolloutWriter(tmp_path, "mock")
    writer.write_live(0, _records(0), prompt_loader_state_after={})
    _write_json(
        _intent(tmp_path, 1),
        {
            "version": 1,
            "step": 1,
            "shard": _shard(tmp_path).name,
            "byte_offset": _shard(tmp_path).stat().st_size,
        },
    )
    with pytest.raises(ValueError, match="write intents before the durable frontier"):
        RolloutWriter(tmp_path, "mock").recover_uncommitted_live_tail(2, 2)


def test_live_recovery_refuses_a_tail_followed_by_committed_data(
    tmp_path: Path,
) -> None:
    writer = RolloutWriter(tmp_path, "mock")
    writer.write(1, _records(1, 1))
    writer.write_live(2, _records(2), prompt_loader_state_after={})
    with pytest.raises(ValueError, match="committed steps follow it"):
        RolloutWriter(tmp_path, "mock").recover_uncommitted_live_tail(1, 2)


def test_live_recovery_deletes_later_shards_and_stale_temporaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_writer, "ROLLOUT_SHARD_SIZE", 1)
    writer = RolloutWriter(tmp_path, "mock")
    writer.write_live(0, _records(0, 1), prompt_loader_state_after={"position": 1})
    writer.write(1, _records(1, 1))
    writer.write(2, _records(2, 1))
    boundary, later = _shard(tmp_path, 1), _shard(tmp_path, 2)
    assert later.exists()
    stale = tmp_path / "mock_live_step_00000001.json.tmp"
    stale.write_text("{}", encoding="utf-8")

    recovered = RolloutWriter(tmp_path, "mock").recover_uncommitted_live_tail(1, 1)

    assert recovered == {1, 2}
    assert boundary.read_bytes() == b""
    assert not later.exists()
    assert not stale.exists()
    assert RolloutWriter(tmp_path, "mock").committed_live_steps(1) == {0}


def test_live_recovery_truncates_a_torn_final_record(tmp_path: Path) -> None:
    writer = RolloutWriter(tmp_path, "mock")
    writer.write_live(0, _records(0), prompt_loader_state_after={})
    size = _shard(tmp_path).stat().st_size
    with _shard(tmp_path).open("ab") as handle:
        handle.write(b'{"step": 1')
    RolloutWriter(tmp_path, "mock").recover_uncommitted_live_tail(1, 2)
    assert _shard(tmp_path).stat().st_size == size


# -- fixed-corpus recovery ----------------------------------------------------


def test_fixed_recovery_truncates_a_torn_tail_and_later_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rollout_writer, "ROLLOUT_SHARD_SIZE", 2)
    writer = RolloutWriter(tmp_path, "mock")
    writer.write(0, _records(0))
    writer.write(1, _records(1, 1))
    writer.write(2, _records(2, 1))
    with _shard(tmp_path, 2).open("ab") as handle:
        handle.write(b'{"step": 3')
    first_size = _shard(tmp_path, 0).stat().st_size

    recovered = RolloutWriter(tmp_path, "mock").recover_incomplete_fixed_tail(2)

    assert recovered == {1, 2}
    assert _shard(tmp_path, 0).stat().st_size == first_size
    assert _shard(tmp_path, 1).read_bytes() == b""
    assert not _shard(tmp_path, 2).exists()


def test_fixed_recovery_without_a_partial_update_changes_nothing(
    tmp_path: Path,
) -> None:
    RolloutWriter(tmp_path, "mock").write(0, _records(0))
    before = _shard(tmp_path).read_bytes()
    assert RolloutWriter(tmp_path, "mock").recover_incomplete_fixed_tail(2) == set()
    assert _shard(tmp_path).read_bytes() == before


# -- serialization ------------------------------------------------------------


class _Color(Enum):
    RED = "red"


@dataclass(frozen=True, kw_only=True, slots=True)
class _Point:
    x: int


class _Model:
    def model_dump(self) -> dict[str, int]:
        return {"m": 1}


class _Opaque:
    @override
    def __str__(self) -> str:
        return "opaque"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_Point(x=1), {"x": 1}),
        (_Color.RED, "red"),
        (torch.tensor([1, 2]), [1, 2]),
        (np.array([3, 4]), [3, 4]),
        (np.int64(5), 5),
        (_Model(), {"m": 1}),
        (_Opaque(), "opaque"),
    ],
)
def test_json_conversion_matches_the_released_writer(
    value: object,
    expected: object,
) -> None:
    assert (
        json_loads(rollout_writer._encode({"v": value}), dict[str, object])["v"]
        == expected
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
