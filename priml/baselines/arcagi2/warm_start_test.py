"""Tests for initializing a model body from another run's checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import logging

import pytest
import torch

from priml.baselines.arcagi1.model import from_reference_name
from priml.baselines.arcagi1.train_step import TrmTrainStep
from priml.baselines.arcagi1.train_step_test import canonical_name, port_config
from priml.baselines.arcagi2.warm_start import WarmStart
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding


if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn


def _step(num_puzzles: int) -> TrmTrainStep:
    config = port_config("exp007")
    assert isinstance(config.model.prefix, SparsePuzzleEmbedding.Config)
    config.model.prefix.num_puzzles = num_puzzles
    return TrmTrainStep(config.finalize())


def _save(model: nn.Module, path: Path, *, ema_scale: float) -> None:
    """Write a checkpoint whose EMA shadow scales two of the model's tensors."""
    state = model.state_dict()
    ema = {
        name: state[name] * ema_scale
        for name in ("embedding.embed_tokens.weight", "head.weight")
    }
    torch.save({"step": {"model": state, "ema": {"shadow_params": ema}}}, path)


def test_body_loads_and_the_task_table_stays_fresh(tmp_path: Path) -> None:
    """The EMA shadow wins; a table sized for other tasks keeps its init."""
    torch.manual_seed(0)
    source = _step(num_puzzles=8).model
    path = tmp_path / "step_1.pt"
    _save(source, path, ema_scale=2.0)
    torch.manual_seed(1)
    target = _step(num_puzzles=16).model
    fresh_table = target.state_dict()["prefix.weights"].clone()

    report = WarmStart.Config(path=path).make()(target)

    loaded = target.state_dict()
    expected = source.state_dict()
    assert torch.equal(
        loaded["embedding.embed_tokens.weight"],
        expected["embedding.embed_tokens.weight"] * 2.0,
    )
    assert torch.equal(loaded["head.weight"], expected["head.weight"] * 2.0)
    assert torch.equal(loaded["halt_head.weight"], expected["halt_head.weight"])
    assert torch.equal(loaded["prefix.weights"], fresh_table)
    assert report.loaded == tuple(
        sorted(name for name in source.state_dict() if name != "prefix.weights"),
    )
    assert report.skipped_shape == ("prefix.weights",)
    assert report.skipped_missing == ()
    assert report.fresh == ("prefix.weights",)


def test_wrapper_prefixes_are_matched_through(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A checkpoint saved through DDP or compile still reaches bare names."""
    torch.manual_seed(0)
    source = _step(num_puzzles=8).model
    wrapped = {f"module._orig_mod.{k}": v for k, v in source.state_dict().items()}
    path = tmp_path / "step_1.pt"
    torch.save({"step": {"model": wrapped}}, path)
    caplog.set_level(logging.INFO)
    torch.manual_seed(1)
    target = _step(num_puzzles=8).model
    report = WarmStart.Config(path=path).make()(target)
    assert report.loaded == tuple(sorted(source.state_dict()))
    assert report.skipped_shape == ()
    assert report.skipped_missing == ()
    assert report.fresh == ()
    assert caplog.records[-1].getMessage() == (
        f"warm start {path}: loaded {len(source.state_dict())} tensor(s); "
        "shape-skipped none; fresh none"
    )
    for name, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[name], value), name


def test_reference_names_are_renamed_so_only_the_table_stays_fresh(
    tmp_path: Path,
) -> None:
    """A reference-era checkpoint loads everything but the per-task table."""
    torch.manual_seed(0)
    source = _step(num_puzzles=8).model
    legacy = {canonical_name(k): v for k, v in source.state_dict().items()}
    path = tmp_path / "step_1.pt"
    torch.save({"step": {"model": legacy}}, path)
    torch.manual_seed(1)
    target = _step(num_puzzles=16).model
    report = WarmStart.Config(path=path, rename=from_reference_name).make()(target)
    assert report.fresh == ("prefix.weights",)
    # Skips name the checkpoint's own tensor, like ``skipped_missing``.
    assert report.skipped_shape == ("puzzle_emb.weights",)
    assert torch.equal(
        target.state_dict()["halt_head.weight"],
        source.state_dict()["halt_head.weight"],
    )


def test_small_checkpoint_reports_each_outcome_and_logs_them(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "small.pt"
    torch.save(
        {
            "step": {
                "model": {
                    "weight": torch.ones(3, 2),
                    "bias": torch.ones(2),
                    "missing.weight": torch.ones(2, 3),
                    "metadata": "not a tensor",
                },
                "ema": {
                    "shadow_params": {
                        "weight": torch.zeros(3, 2),
                        "metadata": "not a tensor",
                    },
                },
            },
        },
        path,
    )
    model = torch.nn.Linear(2, 3)
    assert model.bias is not None
    fresh_bias = model.bias.detach().clone()
    load_state_dict = model.load_state_dict
    strict_values: list[bool] = []

    def load_state_dict_spy(state: dict[str, torch.Tensor], *, strict: bool) -> object:
        strict_values.append(strict)
        return load_state_dict(state, strict=strict)

    monkeypatch.setattr(model, "load_state_dict", load_state_dict_spy)
    caplog.set_level(logging.INFO)

    with patch.object(torch, "load", wraps=torch.load) as load:
        report = WarmStart.Config(path=path).make()(model)

    load.assert_called_once_with(path, map_location="cpu", weights_only=False)
    assert strict_values == [False]
    assert report.loaded == ("weight",)
    assert report.skipped_shape == ("bias",)
    assert report.skipped_missing == ("missing.weight",)
    assert report.fresh == ("bias",)
    assert model.bias is not None
    assert torch.equal(model.weight, torch.zeros(3, 2))
    assert torch.equal(model.bias, fresh_bias)
    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == (
        f"warm start {path}: loaded 1 tensor(s); shape-skipped ['bias']; fresh ['bias']"
    )


def test_ema_mapping_without_shadow_params_overlays_live_weights(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 3)
    path = tmp_path / "ema.pt"
    torch.save(
        {
            "step": {
                "model": {
                    name: torch.ones_like(value)
                    for name, value in model.state_dict().items()
                },
                "ema": {"weight": torch.zeros(3, 2)},
            },
        },
        path,
    )

    report = WarmStart.Config(path=path).make()(model)

    assert report.loaded == ("bias", "weight")
    assert model.bias is not None
    assert torch.equal(model.weight, torch.zeros(3, 2))
    assert torch.equal(model.bias, torch.ones(3))


def test_a_checkpoint_matching_nothing_raises(tmp_path: Path) -> None:
    path = tmp_path / "step_1.pt"
    torch.save({"step": {"model": {"absent.weight": torch.zeros(1)}}}, path)
    with pytest.raises(ValueError, match="matched no tensors") as error:
        WarmStart.Config(path=path).make()(_step(num_puzzles=8).model)
    assert str(error.value) == (
        f"warm start from {path} matched no tensors; the checkpoint "
        "layout does not fit this model."
    )


def test_the_step_warm_starts_its_model(tmp_path: Path) -> None:
    """The step's ``warm_start`` slot loads before training begins."""
    torch.manual_seed(0)
    source = _step(num_puzzles=8).model
    path = tmp_path / "step_1.pt"
    _save(source, path, ema_scale=3.0)
    config = port_config("exp007")
    config.warm_start = WarmStart.Config(path=path)
    torch.manual_seed(1)
    step = TrmTrainStep(config.finalize())
    assert torch.equal(
        step.model.state_dict()["head.weight"],
        source.state_dict()["head.weight"] * 3.0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
