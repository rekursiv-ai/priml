"""Check global ARC2 ballots, stable ties, and empty-rank participation."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Final, Literal, override

from torch import Tensor

import pytest
import torch
import torch.distributed as dist

from priml.baselines.arcagi.parity.metric import PassK
from priml.baselines.arcagi.parity.record_test import load, reduce
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from pathlib import Path


type Mode = Literal["majority", "tie", "empty", "all_empty", "empty_corpus"]
"""Which ballots each rank casts; see :func:`_worker`."""

MODES: Final[tuple[Mode, ...]] = (
    "majority",
    "tie",
    "empty",
    "all_empty",
    "empty_corpus",
)
"""Every :data:`Mode`, in parametrize order."""


def test_reject_missing_gather(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject rank-local scoring with the same global-ballot comparator."""
    _manifest(tmp_path)
    ports = [_local_metric(tmp_path), _local_metric(tmp_path)]
    with pytest.raises(AssertionError, match="global pass@1"):
        _score_both_ranks(ports, "majority", monkeypatch)


def test_frozen_ballot_scores_bite() -> None:
    """A changed recorded score is reported against the frozen record."""
    frozen = load("metric_distributed")["majority/rank0"]
    nudged = {key: value.clone() for key, value in frozen.items()}
    nudged["pass@1"] = nudged["pass@1"] - 1
    assert mismatches(frozen, nudged) == ["pass@1: 1/1 differ"]


def test_checkpoint_rejects_malformed_ballot() -> None:
    metric = PassK.Config().make()
    with pytest.raises(TypeError):
        metric.load_state_dict(
            {"votes": {"puzzle": {"input": [["prediction", "bad"]]}}},
        )


def test_checkpoint_roundtrip() -> None:
    metric = PassK.Config().make()
    metric.votes = {"puzzle": {"input": [("prediction", 0.125)]}}
    restored = PassK.Config().make()
    restored.load_state_dict(metric.state_dict())
    assert restored.votes == metric.votes


class _LocalVotes(PassK):
    @override
    def _global_votes(self) -> dict[str, dict[str, list[tuple[str, float]]]]:
        return self.votes


def _local_metric(root: Path) -> PassK:
    return _LocalVotes(PassK.Config(working_dir=root).finalize())


def _manifest(root: Path) -> None:
    (root / "identifiers.json").write_text('["<blank>", "puzzle"]')
    (root / "test_puzzles.json").write_text(
        '{"puzzle": {"test": [{"input": [[0]], "output": [[0]]}, '
        '{"input": [[0]], "output": [[0]]}]}}',
    )


def _ballots(mode: Mode, rank: int) -> tuple[Tensor, dict[str, object]] | None:
    """Return the prediction and batch ``rank`` casts under ``mode``, or None."""
    count = (
        0
        if mode in ("all_empty", "empty_corpus") or (mode == "empty" and rank == 1)
        else 2
        if mode == "majority" and rank == 1
        else 1
    )
    if not count:
        return None
    prediction = 2 if rank == 1 or mode == "empty" else 3
    packed = torch.tensor([[0.0, float(prediction)]]).expand(count, 2)
    return packed, {
        "media": torch.full((count, 1), 2, dtype=torch.int32),
        "puzzle_identifiers": torch.ones(count, dtype=torch.int64),
        "valid_count": count,
    }


# The metric's one collective is ``all_gather_object`` of the local votes, answered here
# with both ranks' votes in rank order.
def _score_both_ranks(
    ports: list[PassK],
    mode: Mode,
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Tensor]]:
    """Cast each rank's ballots, then compute each rank's scores over the gather."""
    for rank, port in enumerate(ports):
        cast_ = _ballots(mode, rank)
        if cast_ is not None:
            port.update(cast_[0], **cast_[1])
    snapshots = [deepcopy(port.state_dict()) for port in ports]

    def gather(
        out: list[dict[str, dict[str, list[tuple[str, float]]]]],
        votes: object,
    ) -> None:
        del votes
        out[:] = [port.votes for port in ports]

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_gather_object", gather)
    scores: list[dict[str, Tensor]] = []
    for port, snapshot in zip(ports, snapshots, strict=True):
        actual = port.compute()
        assert actual["pass@1"] == (
            0.0 if mode in ("tie", "all_empty", "empty_corpus") else 1.0
        ), "global pass@1"
        assert actual["pass@2"] == (
            0.0 if mode in ("all_empty", "empty_corpus") else 1.0
        ), "global pass@2"
        assert port.compute() == actual, "repeated compute"
        assert port.state_dict() == snapshot, "local ballots mutated"
        scores.append(reduce(actual))
    return scores


@pytest.mark.parametrize("mode", MODES)
def test_source_global_ballots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: Mode,
) -> None:
    """Gather all ranks once per compute, preserving local state and tie order.

    Every rank's scores also equal the ones the source computed on two CPU ranks.
    """
    _manifest(tmp_path)
    if mode == "empty_corpus":
        (tmp_path / "test_puzzles.json").write_text("{}")
    ports = [PassK.Config(working_dir=tmp_path).make() for _ in range(2)]
    frozen = load("metric_distributed")
    for rank, actual in enumerate(_score_both_ranks(ports, mode, monkeypatch)):
        report = mismatches(frozen[f"{mode}/rank{rank}"], actual)
        assert not report, "\n".join(report)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
