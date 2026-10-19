"""Tests for the evaluation reports' schemas, as ``report.py`` reads them."""

from __future__ import annotations

import json

from treekle import ReadError, json_loads

import pytest

from priml.baselines.craftax.world_model.scripts.reports import (
    BaselinesReport,
    DreamReport,
    EngineChecks,
    Settings,
    TilesReport,
    WandbReport,
)


def test_a_report_keeps_its_declared_fields_and_drops_the_rest() -> None:
    text = json.dumps({"all_passed": True, "checks": [{"name": "x"}]})
    assert json_loads(text, EngineChecks) == {"all_passed": True}


def test_a_report_missing_a_required_field_is_refused() -> None:
    with pytest.raises(ReadError):
        json_loads(json.dumps({"beats": {}, "model_nll": {}}), BaselinesReport)


def test_a_mistyped_field_is_refused() -> None:
    with pytest.raises(ReadError):
        json_loads(json.dumps({"all_passed": "yes"}), EngineChecks)


def test_the_tiles_run_is_optional_and_read_whole_when_present() -> None:
    bare = {"metric": {"bpb": 1.5}, "decisions": 7}
    assert json_loads(json.dumps(bare), TilesReport) == bare
    run = {
        "checkpoint": "c.pt",
        "experiment": "e",
        "overrides": ["a=1"],
        "corpus": "k.json",
        "t_g": 16,
        "seed": 3,
        "tiles_available": 9,
        "seconds": 0.5,
    }
    assert json_loads(json.dumps(bare | {"run": run}), TilesReport).get("run") == run


def test_a_training_run_reads_its_trends_and_partial_timing() -> None:
    run = {
        "run": "abc",
        "last_step": None,
        "eval_steps": [1, 2],
        "loss_finite": True,
        "grad_norm_finite": False,
        "unrecovered_spikes": 0,
        "spikes": [],
        "logged": {"loss": 2},
        "grad_norm_first_last_max": None,
        "clip_fraction_last": 0.25,
        "max_attention_logit_max": None,
        "modalities": {"cells": {"decreased": True, "monotone": False}},
        "timing": {"runtime_seconds": 4.0},
    }
    read = json_loads(json.dumps({"runs": [run]}), WandbReport)["runs"][0]
    assert read["modalities"]["cells"] == {"decreased": True, "monotone": False}
    assert read.get("timing") == {"runtime_seconds": 4.0}


def test_dream_and_settings_documents_read_as_written() -> None:
    dream = {"distance": {"tv": 0.1}, "run": {"rows": 2}}
    assert json_loads(json.dumps(dream), DreamReport) == dream
    settings = {"names": ["a"], "sampler_seeds": "0,1", "windows": [16]}
    assert json_loads(json.dumps(settings), Settings) == settings


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
