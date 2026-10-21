"""Freeze the full experiment tree, its fitted branches and the forks' deltas."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import math

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm.experiments import (
    exp000,
    exp001,
    exp002,
    exp_smoke,
)
from priml.baselines.name_that_llm.model import (
    CenteredHeads,
    QwenHeads,
    SvmBranch,
)
from priml.baselines.name_that_llm.scripts.prepare_data import prepare_data
from priml.baselines.name_that_llm.testing import (
    NumericalTrace,
    completed_record,
    float32_comparands,
    miniature_split,
    recipe,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_pprint_golden, assert_tensor_golden_part


_CWD: Final = Path(__file__).resolve().parent


def test_exp000_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


@pytest.mark.parametrize("branch", argvalues=["plain", "centred", "svm"])
def test_exp000_branch_replays_every_fitted_state_and_output(branch: str) -> None:
    config = recipe()
    plain, centred, svm = (
        config.branches[name] for name in ("plain", "centred", "svm")
    )
    assert isinstance(plain, QwenHeads.Config)
    assert isinstance(centred, CenteredHeads.Config)
    assert isinstance(svm, SvmBranch.Config)
    # Three updates per solver pin every fit; the solver tests cover convergence.
    for solver in (
        plain.classifier.optimizer,
        centred.heads.classifier.optimizer,
        svm.classifier.optimizer,
        svm.calibration.optimizer,
    ):
        solver.max_iter = 3
    # Distinct rows, so the centred branch's singleton fallback names its source row.
    earlier = {"plain": np.arange(9, dtype=np.float64).reshape(3, 3) / 36}
    with host_agnostic_numerics(), NumericalTrace() as trace:
        output = config.branches[branch].make()(miniature_split(), earlier)
    record = completed_record(trace, outputs={"output": torch.from_numpy(output)})
    assert_tensor_golden_part(
        _CWD / "testdata" / "exp000.pt",
        prefix=f"{branch}/",
        record={
            f"{branch}/{name}": value
            for name, value in float32_comparands(record).items()
        },
    )


def test_exp000_names_its_own_output() -> None:
    resolved = exp000().finalized()
    assert resolved.working_dir == Path("/opt/scratch/artifacts/name-that-llm/exp000")


def test_five_minute_candidate_changes_only_the_validation_budget() -> None:
    reference, candidate = exp000(), exp001()
    assert math.isinf(reference.budget.seconds)
    assert candidate.budget.seconds == 300
    assert candidate.outer_folds == reference.outer_folds
    assert candidate.experiment_name == "exp001"
    candidate.experiment_name = reference.experiment_name
    candidate.budget.seconds = reference.budget.seconds
    assert candidate.serialize() == reference.serialize()


def test_two_fold_candidate_changes_only_the_validation_fold_count() -> None:
    parent, candidate = exp001(), exp002()
    assert candidate.outer_folds == 2
    assert candidate.experiment_name == "exp002"
    candidate.experiment_name = parent.experiment_name
    candidate.outer_folds = parent.outer_folds
    assert candidate.serialize() == parent.serialize()


def test_smoke_keeps_all_algorithm_and_ensemble_choices() -> None:
    reference, smoke = exp000(), exp_smoke()
    assert smoke.experiment_name == "exp_smoke"
    assert smoke.encoder.batch_size == 2
    assert smoke.encoder.max_length == 64
    assert smoke.dataset.max_prompts == 6
    assert smoke.outer_folds == 2
    smoke.experiment_name = reference.experiment_name
    smoke.dataset.max_prompts = reference.dataset.max_prompts
    smoke.outer_folds = reference.outer_folds
    smoke.encoder.batch_size = reference.encoder.batch_size
    smoke.encoder.max_length = reference.encoder.max_length
    assert smoke.serialize() == reference.serialize()


def test_preparation_matches_training_input_location() -> None:
    assert prepare_data().directory == exp000().dataset.directory


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
