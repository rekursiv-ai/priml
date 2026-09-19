"""Check global ARC2 ballots, stable ties, and empty-rank participation."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Literal, override

import traceback

import pytest
import torch

from priml.baselines.arcagi2.metric import PassK


# isort: split
if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter
    from priml.metrics.custom_types import MetricProtocol


@pytest.mark.compute_distributed
@pytest.mark.parametrize(
    "mode",
    ["majority", "tie", "empty", "all_empty", "empty_corpus"],
)
def test_global_ballots(
    tmp_path: Path,
    warm_pools: WarmPoolGetter,
    mode: Literal["majority", "tie", "empty", "all_empty", "empty_corpus"],
) -> None:
    """Gather all ranks once per compute, preserving local state and tie order."""
    _manifest(tmp_path)
    if mode == "empty_corpus":
        (tmp_path / "test_puzzles.json").write_text("{}")
    warm_pools({"dp": 2})(partial(_worker, tmp_path, mode))
    for rank in range(2):
        assert (tmp_path / f"metric_{rank}").read_text() == "ok"


@pytest.mark.compute_distributed
def test_reject_missing_gather(tmp_path: Path, warm_pools: WarmPoolGetter) -> None:
    """Reject rank-local scoring with the same global-ballot comparator."""
    _manifest(tmp_path)
    warm_pools({"dp": 2})(
        partial(_worker, tmp_path, "majority", candidate=_local_metric),
    )
    result = (tmp_path / "metric_0").read_text()
    assert "AssertionError: global pass@1" in result


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
        '{"puzzle": {"test": [{"input": [[0]], "output": [[0]]}]}}',
    )


def _worker(
    root: Path,
    mode: Literal["majority", "tie", "empty", "all_empty", "empty_corpus"],
    mesh: DeviceMesh,
    *,
    candidate: Callable[[Path], PassK] | None = None,
    reference: Callable[[Path], MetricProtocol] | None = None,
) -> None:
    rank = mesh.get_rank()
    try:
        port = candidate(root) if candidate else PassK.Config(working_dir=root).make()
        source = reference(root) if reference else None
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
            if source is not None:
                source.update(packed, **batch)
        snapshot = deepcopy(port.state_dict())
        expected = source.compute() if source is not None else None
        actual = port.compute()
        assert actual["pass@1"] == (
            0.0 if mode in ("tie", "all_empty", "empty_corpus") else 1.0
        ), "global pass@1"
        assert actual["pass@2"] == (
            0.0 if mode in ("all_empty", "empty_corpus") else 1.0
        ), "global pass@2"
        if expected is not None:
            assert actual == expected, "source metrics"
        assert port.compute() == actual, "repeated compute"
        assert port.state_dict() == snapshot, "local ballots mutated"
        result = "ok"
    except (AssertionError, RuntimeError, ValueError, TypeError, KeyError):
        result = traceback.format_exc()
    (root / f"metric_{rank}").write_text(result)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
