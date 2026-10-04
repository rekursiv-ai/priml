"""Tests for the blog-post launcher."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import argparse
import hashlib
import logging
import sys

import pytest
import torch

from priml.baselines.arcagi1.scripts import reproduce_blog_post
from priml.baselines.arcagi1.scripts.reproduce_blog_post import overlay, recipe
from priml.baselines.arcagi1.train_step import TrmTrainStep
from priml.baselines.arcagi1.train_step_test import port_config
from priml.baselines.sudoku.act import AtomicPool
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.lib.custom_json import DictCodec
from priml.runtime import SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import NoParallel
from priml.train.tracker import FileTracker


@pytest.mark.parametrize("single_gpu", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_recipe_finalizes(*, single_gpu: bool, checkpoint: bool) -> None:
    config = recipe(
        single_gpu=single_gpu,
        checkpoint=checkpoint,
        steps=17,
    ).finalize()
    assert config.experiment_name == (
        "exp008_blog_gx10" if single_gpu else "exp008_blog_8gpu"
    ) + ("_historical_eval" if checkpoint else "")
    assert config.max_steps == 17
    assert config.max_time == float("inf")
    assert config.num_steps_eval == -1
    assert config.eval_warmup_batches == 0
    assert config.eval_only == checkpoint
    assert isinstance(config.tracker, FileTracker.Config)
    assert config.dataset.eval_batch_size == (32 if single_gpu else 256)
    if checkpoint:
        assert config.checkpointer is None
    else:
        assert isinstance(config.checkpointer, Checkpointer.Config)
        assert config.checkpointer.resume is False
    pool = config.step.pool
    prefix = config.step.model.prefix
    assert isinstance(pool, AtomicPool.Config)
    assert isinstance(prefix, SparsePuzzleEmbedding.Config)
    assert pool.batch_size == prefix.batch_size == config.dataset.batch_size
    assert pool.batch_size == (8 if single_gpu else 96)
    if single_gpu:
        assert isinstance(config.runtime, SingleProcess.Config)
        assert config.runtime.device == "cuda"
        assert isinstance(config.step.parallelism, NoParallel.Config)
        assert config.step.parallelism.device == "cuda"


def test_recipe_uses_the_default_training_horizon() -> None:
    assert recipe(single_gpu=False, checkpoint=False).max_steps == 280_000


def test_recipe_disables_time_limit_and_configures_single_gpu_device() -> None:
    config = recipe(single_gpu=True, checkpoint=False, steps=2)

    assert config.max_time == float("inf")
    assert isinstance(config.runtime, SingleProcess.Config)
    assert config.runtime.device == "cuda"
    assert isinstance(config.step.parallelism, NoParallel.Config)
    assert config.step.parallelism.device == "cuda"


@runtime_checkable
class Flags(Protocol):
    checkpoint: Path | None


def test_add_arguments_accepts_only_a_path_checkpoint() -> None:
    parser = argparse.ArgumentParser()
    reproduce_blog_post._add_arguments(parser)
    flags = parser.parse_args([])
    assert isinstance(flags, Flags)
    assert flags.checkpoint is None
    flags = parser.parse_args(["--checkpoint", "archive.pt"])
    assert isinstance(flags, Flags)
    assert flags.checkpoint == Path("archive.pt")
    assert "SHA-pinned historical archive" in parser.format_help()


@pytest.mark.parametrize(
    ("world_size", "device_count", "single_gpu"),
    [(None, 1, True), ("8", 8, False)],
)
def test_main_runs_training_for_supported_world_sizes(
    monkeypatch: pytest.MonkeyPatch,
    world_size: str | None,
    device_count: int,
    single_gpu: bool,
) -> None:
    runs: list[str] = []
    recipes: list[tuple[bool, bool]] = []
    log_levels: list[int] = []

    class TrainingLoop:
        def run(self) -> None:
            runs.append("run")

    class Config:
        def make(self) -> TrainingLoop:
            return TrainingLoop()

    def make_recipe(*, single_gpu: bool, checkpoint: bool) -> Config:
        recipes.append((single_gpu, checkpoint))
        return Config()

    def configure_logging(*, level: int) -> None:
        log_levels.append(level)

    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: device_count,
    )
    monkeypatch.setattr(sys, "argv", ["reproduce_blog_post.py"])
    if world_size is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    monkeypatch.setattr(reproduce_blog_post, "recipe", make_recipe)
    monkeypatch.setattr(logging, "basicConfig", configure_logging)
    assert reproduce_blog_post.main() == 0
    assert recipes == [(single_gpu, False)]
    assert runs == ["run"]
    assert log_levels == [logging.INFO]


def test_main_rejects_unsupported_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(sys, "argv", ["reproduce_blog_post.py"])
    monkeypatch.setenv("WORLD_SIZE", "4")

    with pytest.raises(SystemExit) as error:
        reproduce_blog_post.main()

    assert error.value.code == "Use one CUDA GPU or torchrun with eight CUDA GPUs"


def test_main_rejects_missing_docstring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reproduce_blog_post, "__doc__", None)

    with pytest.raises(ValueError, match=r"Expected __doc__ is not None\.") as error:
        reproduce_blog_post.main()

    assert str(error.value) == "Expected __doc__ is not None."


def test_main_help_shows_launcher_documentation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptions: list[str] = []
    original_init = argparse.ArgumentParser.__init__

    def capture_init(
        parser: argparse.ArgumentParser,
        *,
        description: str,
        formatter_class: type[argparse.HelpFormatter],
    ) -> None:
        descriptions.append(description)
        original_init(
            parser,
            description=description,
            formatter_class=formatter_class,
        )

    monkeypatch.setattr(argparse.ArgumentParser, "__init__", capture_init)
    monkeypatch.setattr(sys, "argv", ["reproduce_blog_post.py", "--help"])
    with pytest.raises(SystemExit) as error:
        reproduce_blog_post.main()
    assert error.value.code == 0
    assert descriptions == [(reproduce_blog_post.__doc__ or "").split("\n", 2)[2]]
    help_text = capsys.readouterr().out
    assert "Reproduce the ARC-AGI-1 blog-post experiment" in help_text
    assert "Train from scratch on eight visible H200 GPUs" in help_text
    assert "To train on one GX10/GB10 (128 GB shared memory)" in help_text
    with pytest.raises(SystemExit):
        reproduce_blog_post.main()


def test_main_checks_and_loads_the_pinned_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "historical.pt"
    checkpoint.write_bytes(b"pinned")
    sha256 = "64bd795f8dc984d7e647be155afc098bb41c7a8c22daeac36f57c4ad76fc1cff"
    runs: list[str] = []
    loaded_states: list[dict[str, object]] = []
    recipes: list[tuple[bool, bool]] = []
    load_calls: list[tuple[Path, str, bool]] = []

    class CheckpointDigest:
        def hexdigest(self) -> str:
            return sha256

    class TrainingLoop:
        def state_dict(self) -> dict[str, object]:
            return {"step": {"model": {}, "ema": {}}}

        def load_state_dict(self, state: dict[str, object]) -> None:
            loaded_states.append(state)

        def run(self) -> None:
            runs.append("run")

    class Config:
        def make(self) -> TrainingLoop:
            return TrainingLoop()

    def make_recipe(*, single_gpu: bool, checkpoint: bool) -> Config:
        recipes.append((single_gpu, checkpoint))
        return Config()

    def digest(file: object, algorithm: str) -> CheckpointDigest:
        assert file is not None
        assert algorithm == "sha256"
        return CheckpointDigest()

    def load_archive(
        path: Path,
        map_location: str,
        *,
        weights_only: bool,
    ) -> dict[str, object]:
        load_calls.append((path, map_location, weights_only))
        return {"step": {"model": {}, "ema": {}}}

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(hashlib, "file_digest", digest)
    monkeypatch.setattr(torch, "load", load_archive)
    monkeypatch.setattr(reproduce_blog_post, "recipe", make_recipe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["reproduce_blog_post.py", "--checkpoint", str(checkpoint)],
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert reproduce_blog_post.main() == 0
    assert recipes == [(True, True)]
    assert load_calls == [(checkpoint, "cpu", True)]
    assert len(loaded_states) == 1
    state = DictCodec.coerce(loaded_states[0]["step"])
    assert state["model"] == {}
    assert state["ema"] == {"shadow_params": {}, "global_step": 280_000}
    assert state["timer_step"] == {"global_count": 280_000, "global_sec": 0.0}
    assert runs == ["run"]


def test_main_rejects_a_checkpoint_with_the_wrong_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "historical.pt"
    checkpoint.write_bytes(b"not the pinned archive")
    sha256 = "64bd795f8dc984d7e647be155afc098bb41c7a8c22daeac36f57c4ad76fc1cff"
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        sys,
        "argv",
        ["reproduce_blog_post.py", "--checkpoint", str(checkpoint)],
    )
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    with pytest.raises(SystemExit) as error:
        reproduce_blog_post.main()

    assert error.value.code == f"Historical checkpoint SHA256 must be {sha256}"


def test_historical_names_land_on_the_recipes_state() -> None:
    """Every archive tensor, renamed, is one the exp008 model holds."""
    step = TrmTrainStep(port_config("exp008").finalize())
    names = [
        "embed_tokens.weight",
        "embed_feedback",
        "q_head.weight",
        "q_head.bias",
        "puzzle_emb.weights",
    ]
    tensors = {name: torch.zeros(1) for name in names}
    archive: dict[str, object] = {"step": {"model": tensors, "ema": tensors}}
    state = DictCodec.coerce(overlay(archive, {"step": {}})["step"])
    renamed = DictCodec.coerce(state["model"], torch.Tensor)
    assert set(renamed) <= set(step.model.state_dict())


def test_overlay_replaces_exact_state_and_preserves_other_fields() -> None:
    model = {
        "embed_tokens.weight": torch.tensor([2.0, 3.0]),
        "embed_feedback": torch.tensor([5.0, 7.0]),
        "q_head.bias": torch.tensor([11.0, 13.0]),
        "puzzle_emb.weights": torch.tensor([17.0, 19.0]),
        "layers.0.weight": torch.tensor([23.0, 29.0]),
    }
    ema = {name: value + 1 for name, value in model.items()}
    archive: dict[str, object] = {"step": {"model": model, "ema": ema}}
    expected_model = {
        "embedding.embed_tokens.weight": model["embed_tokens.weight"],
        "embedding.channels.0.embed_feedback": model["embed_feedback"],
        "halt_head.bias": model["q_head.bias"],
        "prefix.weights": model["puzzle_emb.weights"],
        "layers.0.weight": model["layers.0.weight"],
    }
    expected_ema = {
        "embedding.embed_tokens.weight": ema["embed_tokens.weight"],
        "embedding.channels.0.embed_feedback": ema["embed_feedback"],
        "halt_head.bias": ema["q_head.bias"],
        "prefix.weights": ema["puzzle_emb.weights"],
        "layers.0.weight": ema["layers.0.weight"],
    }
    for requested_steps, expected_steps in ((None, 280_000), (17, 17)):
        step = {
            "model": {"stale": torch.tensor([31.0])},
            "ema": {"stale": torch.tensor([37.0])},
            "timer_step": {"global_count": 0, "global_sec": 4.5},
            "unrelated": "preserved",
        }
        into: dict[str, object] = {"step": step, "unrelated": "preserved"}
        result = (
            overlay(archive, into)
            if requested_steps is None
            else overlay(archive, into, steps=requested_steps)
        )
        state = DictCodec.coerce(result["step"])
        actual_model = DictCodec.coerce(state["model"], torch.Tensor)
        actual_ema = DictCodec.coerce(state["ema"])
        actual_shadow = DictCodec.coerce(actual_ema["shadow_params"], torch.Tensor)
        actual_timer = DictCodec.coerce(state["timer_step"])
        assert result is into
        assert set(actual_model) == set(expected_model)
        assert all(
            torch.equal(actual_model[name], expected_model[name])
            for name in expected_model
        )
        assert set(actual_shadow) == set(expected_ema)
        assert all(
            torch.equal(actual_shadow[name], expected_ema[name])
            for name in expected_ema
        )
        assert actual_ema["global_step"] == expected_steps
        assert actual_timer == {"global_count": expected_steps, "global_sec": 0.0}
        assert state["unrelated"] == "preserved"
        assert into["unrelated"] == "preserved"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
