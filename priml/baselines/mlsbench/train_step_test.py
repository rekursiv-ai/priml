"""Tests for the step that owns nothing."""

from __future__ import annotations

import pytest

from priml.baselines.mlsbench.train_step import NullStep


def test_null_step_refuses_every_loss() -> None:
    step = NullStep(NullStep.Config())
    assert step.global_step == 0
    for f in (step.eval_loss, step.train_loss, step.train_step, step.call_eval):
        with pytest.raises(NotImplementedError, match="eval_only"):
            f()


def test_null_step_passes_batches_through_and_keeps_no_state() -> None:
    step = NullStep(NullStep.Config())
    batch: dict[str, object] = {"setting": "a"}
    assert step.preprocess_batch(batch) is batch
    assert step.state_dict() == {}
    step.load_state_dict({"x": 1})
    assert step.state_dict() == {}
