from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from priml.custom_types import (
    CheckpointableProtocol,
    HasNormalizedWorkingDirPattern,
    JobProtocol,
    LaunchableExperiment,
    MetricObjective,
)


def test_metric_objective_minimize_is_better():
    obj = MetricObjective(metric_key="eval/total_loss", direction="minimize")
    assert obj.is_better(1.0, 2.0)
    assert not obj.is_better(2.0, 1.0)
    assert not obj.is_better(1.0, 1.0)


def test_metric_objective_maximize_is_better():
    obj = MetricObjective(metric_key="eval/roc_auc", direction="maximize")
    assert obj.is_better(2.0, 1.0)
    assert not obj.is_better(1.0, 2.0)


@dataclass(slots=True, kw_only=True)
class _Launchable:
    study_name: str = ""
    experiment_name: str = ""
    doc: str = ""
    base_dir: Path | str | None = None
    working_dir: Path | str = "runs"


def test_runtime_protocols_match_on_declared_fields_alone() -> None:
    launchable = _Launchable()
    assert isinstance(launchable, LaunchableExperiment)
    assert isinstance(launchable, HasNormalizedWorkingDirPattern)
    assert not isinstance(object(), LaunchableExperiment)
    assert not isinstance(object(), HasNormalizedWorkingDirPattern)


class _DefaultBodies:
    """Borrows every protocol stub so its default body can be exercised."""

    run = JobProtocol.run
    state_dict = CheckpointableProtocol.state_dict
    load_state_dict = CheckpointableProtocol.load_state_dict


def test_protocol_default_bodies_are_inert() -> None:
    """A stub hides no behavior: every default body is a no-op returning None."""
    bodies = _DefaultBodies()
    assert isinstance(bodies, JobProtocol)
    assert isinstance(bodies, CheckpointableProtocol)
    assert bodies.run("--flag") is None
    assert bodies.load_state_dict({}) is None
    assert cast(object, bodies.state_dict()) is None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
