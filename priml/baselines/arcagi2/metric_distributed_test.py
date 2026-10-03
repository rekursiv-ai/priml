"""Check global ARC2 ballots, stable ties, and empty-rank participation."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, cast, override

import traceback

from torch import Tensor

import pytest
import torch

from priml.baselines.arcagi2.metric import PassK
from priml.baselines.arcagi2.record_test import load, reduce
from priml.lib.custom_json import DictCodec
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter
    from priml.metrics.custom_types import MetricProtocol


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


@pytest.mark.compute_distributed
def test_reject_missing_gather(tmp_path: Path, warm_pools: WarmPoolGetter) -> None:
    """Reject rank-local scoring with the same global-ballot comparator."""
    _manifest(tmp_path)
    warm_pools({"dp": 2})(
        partial(_worker, tmp_path, "majority", candidate=_local_metric),
    )
    result = (tmp_path / "metric_0").read_text()
    assert "AssertionError: global pass@1" in result


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
    return _LocalVotes(PassK.Config(working_dir=root))


def _manifest(root: Path) -> None:
    (root / "identifiers.json").write_text('["<blank>", "puzzle"]')
    (root / "test_puzzles.json").write_text(
        '{"puzzle": {"test": [{"input": [[0]], "output": [[0]]}, '
        '{"input": [[0]], "output": [[0]]}]}}',
    )


def _worker(
    root: Path,
    mode: Mode,
    mesh: DeviceMesh,
    *,
    candidate: Callable[[Path], MetricProtocol] | None = None,
) -> None:
    rank = mesh.get_rank()
    try:
        port = candidate(root) if candidate else PassK.Config(working_dir=root).make()
        count = (
            0
            if mode in ("all_empty", "empty_corpus") or (mode == "empty" and rank == 1)
            else 2
            if mode == "majority" and rank == 1
            else 1
        )
        prediction = 2 if rank == 1 or mode == "empty" else 3
        packed = torch.tensor([[0.0, float(prediction)]]).expand(count, 2)
        batch = {
            "media": torch.full((count, 1), 2, dtype=torch.int32),
            "puzzle_identifiers": torch.ones(count, dtype=torch.int64),
            "valid_count": count,
        }
        if count:
            port.update(packed, **batch)
        snapshot = deepcopy(port.state_dict())
        actual = port.compute()
        torch.save(reduce(actual), root / f"scores_{rank}.pt")
        assert actual["pass@1"] == (
            0.0 if mode in ("tie", "all_empty", "empty_corpus") else 1.0
        ), "global pass@1"
        assert actual["pass@2"] == (
            0.0 if mode in ("all_empty", "empty_corpus") else 1.0
        ), "global pass@2"
        assert port.compute() == actual, "repeated compute"
        assert port.state_dict() == snapshot, "local ballots mutated"
        result = "ok"
    except (AssertionError, RuntimeError, ValueError, TypeError, KeyError):
        result = traceback.format_exc()
    (root / f"metric_{rank}").write_text(result)


@pytest.mark.compute_distributed
@pytest.mark.parametrize(
    "mode",
    MODES,
)
def test_source_global_ballots(
    tmp_path: Path,
    warm_pools: WarmPoolGetter,
    mode: Mode,
) -> None:
    """Gather all ranks once per compute, preserving local state and tie order.

    Every rank's scores also equal the ones the source computed on two CPU ranks.
    """
    _manifest(tmp_path)
    if mode == "empty_corpus":
        (tmp_path / "test_puzzles.json").write_text("{}")
    warm_pools({"dp": 2})(partial(_worker, tmp_path, mode))
    frozen = load("metric_distributed")
    for rank in range(2):
        assert (tmp_path / f"metric_{rank}").read_text() == "ok"
        actual = DictCodec.coerce(
            cast(
                object,
                torch.load(tmp_path / f"scores_{rank}.pt", weights_only=True),
            ),
            Tensor,
        )
        report = mismatches(frozen[f"{mode}/rank{rank}"], actual)
        assert not report, "\n".join(report)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
