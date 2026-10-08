"""Check the report: the in-loop metric, the summary of every step, and the driver."""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import argparse
import json
import math
import sys

import pytest
import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    ManifestLine,
    Receipt,
    write_corpus,
    write_shard,
)
from priml.baselines.craftax.world_model.experiments import exp_smoke
from priml.baselines.craftax.world_model.metric import MODALITIES
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.scripts import (
    dream,
    engine_checks,
    report,
)
from priml.baselines.craftax.world_model.train_step import (
    WorldModelTrainStep,
)
from priml.lib.codec import from_plain, loads


SMOKE: str = "priml.baselines.craftax.world_model.experiments.exp_smoke"


@pytest.mark.compute_large_fixture
def test_validation_metric_is_the_loops_own_evaluation(tmp_path: Path) -> None:
    _smoke_corpus(tmp_path)
    config = exp_smoke()
    config.base_dir = tmp_path
    torch.manual_seed(0)
    loop = config.make()
    expected = {
        f"val/{key.removeprefix('val_')}": value
        for key, value in loop.eval().items()
        if key.startswith("val_")
    }
    step = loop.step
    assert isinstance(step, WorldModelTrainStep)
    model = step.model
    assert isinstance(model, WorldModel)
    finalized = config.copy_tree().finalize()
    metrics = report.validation_metric(
        model,
        finalized,
        device=torch.device("cpu"),
        sampler_seed=0,
    )
    assert metrics == expected
    assert "val/zstd19_bpb" in metrics
    # Another seed's windows are other decisions.
    other = report.validation_metric(
        model,
        finalized,
        device=torch.device("cpu"),
        sampler_seed=1,
    )
    assert other["val/bpb"] != metrics["val/bpb"]


def test_summarize_reads_every_step_and_spreads_over_checkpoints(
    tmp_path: Path,
) -> None:
    for name, nats, own in (("a", 4.0, 0), ("b", 6.0, 1)):
        _write(tmp_path / f"baselines-{name}.json", _baselines(beats=name == "a"))
        _write(tmp_path / f"engine-checks-{name}.json", {"all_passed": True})
        _write(tmp_path / f"data-eval-{name}.json", _tiles(nats))
        for windows in (0, 1):
            # Each checkpoint scores 1 nat more on the other seed's windows.
            shift = float(windows != own)
            _write(
                tmp_path / f"evalonly-{name}-on-v{windows}.json",
                _in_loop(nats + shift),
            )
    dreams = tmp_path / "dreams-a"
    dreams.mkdir()
    _write(dreams / "report.json", {"distance": {"action_tv": 0.25}, "run": {}})
    summary = report.summarize(tmp_path, names=["a", "b"], own=[0, 1], windows=[0, 1])
    assert summary["criterion_1_stable"] is None
    criterion3 = from_plain(summary["criterion_3_beats_baselines"], dict[str, object])
    assert from_plain(
        from_plain(criterion3["a"], dict[str, object])["beats"],
        dict[str, object],
    )["cells"]
    assert summary["criterion_4_engine_checks"] == {"a": True, "b": True}
    assert from_plain(summary["dreams"], dict[str, object])["b"] is None
    sigma = from_plain(summary["sigma"], dict[str, object])
    own = _spread(sigma, "own_windows")
    assert own["values"] == [4.0, 6.0]
    assert from_plain(own["sd"], float) == pytest.approx(2**0.5)
    v0 = _spread(from_plain(sigma["same_windows"], dict[str, object]), "v0")
    assert v0["values"] == [4.0, 7.0]
    # Variances 4.5 on v0 (4 and 7) and 0.5 on v1 (5 and 6) pool to 2.5.
    assert sigma["same_windows_pooled_seed_sd"] == pytest.approx(2.5**0.5)
    assert sigma["window_set_sd_of_means"] == pytest.approx(0.0)
    tiles = from_plain(sigma["common_natural_tiles"], dict[str, object])
    board = from_plain(
        from_plain(tiles["nll"], dict[str, object])["board"],
        dict[str, object],
    )
    # 0.01 bits per canonical byte of a cell's 8 fields.
    assert from_plain(board["mean"], float) == pytest.approx(0.01 * 8 * math.log(2))
    text = report.render(summary, names=["a", "b"])
    assert "| 3. a beats the trivial baselines | pass |" in text
    assert "| 3. b beats the trivial baselines | FAIL |" in text
    assert "| 1. Training stable | not measured |" in text
    assert "| a | action_tv | 0.25 |" in text
    # 0.01 bits per byte of a cell's 8 bytes, the same for every checkpoint.
    assert "| board | 0.055452 +- 0 | 0.055452 +- 0 |" in text


def test_summarize_reads_the_training_criteria_from_wandb(tmp_path: Path) -> None:
    trend = {m: {"decreased": True, "monotone": m != "done"} for m in MODALITIES}
    run = {
        "run": "abc",
        "last_step": 9,
        "eval_steps": [5, 9],
        "loss_finite": True,
        "grad_norm_finite": True,
        "unrecovered_spikes": 0,
        "spikes": [],
        "logged": {},
        "grad_norm_first_last_max": [2.0, 1.0, 2.0],
        "clip_fraction_last": 0.5,
        "max_attention_logit_max": 9.0,
        "modalities": trend,
        "timing": {
            "dt_median_seconds": 1.5,
            "eval_total_seconds": 30.0,
            "runtime_seconds": 99.0,
        },
    }
    _write(tmp_path / "wandb-report.json", {"runs": [run]})
    summary = report.summarize(tmp_path, names=["a"], own=[0], windows=[0])
    assert summary["criterion_1_stable"] is True
    assert summary["criterion_2_every_modality_decreased"] is True
    text = report.render(summary, names=["a"])
    assert "| abc | 9 | True | 0 | 5/5 | 1.5 | 30 | 99 |" in text
    _write(tmp_path / "wandb-report.json", {"runs": [run | {"unrecovered_spikes": 1}]})
    unstable = report.summarize(tmp_path, names=["a"], own=[0], windows=[0])
    assert unstable["criterion_1_stable"] is False


def test_summarize_reads_the_names_and_seeds_of_the_runs_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings: dict[str, object] = {
        "names": ["a", "b"],
        "sampler_seeds": "0,1",
        "windows": [0, 1],
    }
    _write(tmp_path / "settings.json", settings)
    for name, own in (("a", 0), ("b", 1)):
        for windows in (0, 1):
            # Each checkpoint scores 4 nats on its own windows, 9 on the other's.
            nats = 4.0 if windows == own else 9.0
            _write(tmp_path / f"evalonly-{name}-on-v{windows}.json", _in_loop(nats))
    monkeypatch.setattr(
        sys,
        "argv",
        ["report.py", "--summarize", "--output", str(tmp_path)],
    )
    assert report.main() == 0
    summary = from_plain(
        loads((tmp_path / "summary.json").read_text()),
        dict[str, object],
    )
    assert set(from_plain(summary["criterion_4_engine_checks"], dict[str, object])) == {
        "a",
        "b",
    }
    own = _spread(from_plain(summary["sigma"], dict[str, object]), "own_windows")
    assert own["values"] == [4.0, 4.0]
    # Seeds on the command line would contradict the run's own.
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--sampler-seeds", "0,0"])
    with pytest.raises(SystemExit):
        report.main()


@pytest.mark.parametrize(
    "flags",
    [
        ("--names", "a,b"),
        ("--sampler-seeds", "0,1"),
        ("--wandb-runs", "x,y"),
        ("absent.pt",),
    ],
)
def test_main_refuses_flags_that_do_not_fit_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flags: tuple[str, ...],
) -> None:
    checkpoint = tmp_path / "step.pt"
    checkpoint.write_bytes(b"")
    output = tmp_path / "report"
    monkeypatch.chdir(tmp_path)
    argv = ["report.py", str(checkpoint), *flags, "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        report.main()
    assert not output.exists()


def test_children_pass_flags_their_scripts_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(report, "_child", _recording(calls))
    parser = argparse.ArgumentParser()
    report._add_arguments(parser)
    flags = cast(
        "report.Flags",
        parser.parse_args(["c.pt", "--override", "a=1", "--output", str(tmp_path)]),
    )
    checkpoint = report.Checkpoint(name="s1", path=Path("c.pt"), sampler_seed=1)
    failed = report._children(
        checkpoint,
        flags=flags,
        corpus=Path("corpus.json"),
        output=tmp_path,
    )
    assert failed == []
    assert [script for script, _ in calls] == ["engine_checks.py", "dream.py"]
    checks_parser, dream_parser = argparse.ArgumentParser(), argparse.ArgumentParser()
    engine_checks._add_arguments(checks_parser)
    dream._add_arguments(dream_parser)
    checks = cast("engine_checks.Flags", checks_parser.parse_args(calls[0][1]))
    dreams = cast("dream.Flags", dream_parser.parse_args(calls[1][1]))
    for child in (checks, dreams):
        assert child.checkpoint == Path("c.pt")
        assert child.corpus == Path("corpus.json")
        assert child.override == ["a=1", "dataset.sampler_seed=1"]
        assert (child.experiment, child.device) == (flags.experiment, flags.device)
    assert checks.output == tmp_path / "engine-checks-s1.json"
    assert dreams.output == tmp_path / "dreams-s1"
    assert (dreams.rows, dreams.decisions, dreams.seed) == (256, 4_000, 0)


@pytest.mark.compute_large_fixture
def test_main_scores_a_checkpoint_and_writes_the_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _smoke_checkpoint(tmp_path)
    output = tmp_path / "report"
    _argv(
        monkeypatch,
        checkpoint,
        "--skip",
        "engine",
        "--skip",
        "dreams",
        "--window-seeds",
        "0,1",
        "--output",
        str(output),
        root=tmp_path,
    )
    assert report.main() == 0
    names = {path.name for path in output.iterdir()}
    assert names == {
        "settings.json",
        "baselines-s0.json",
        "data-eval-s0.json",
        "evalonly-s0-on-v0.json",
        "evalonly-s0-on-v1.json",
        "summary.json",
        "report.md",
    }
    summary = from_plain(
        loads((output / "summary.json").read_text()),
        dict[str, object],
    )
    sigma = from_plain(summary["sigma"], dict[str, object])
    assert set(sigma) >= {"own_windows", "same_windows", "common_natural_tiles"}


@pytest.mark.cli_python_subprocess
def test_a_failed_child_is_named_and_the_summary_still_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = _smoke_checkpoint(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "engine_checks.py").write_text("raise SystemExit(3)\n")
    monkeypatch.setattr(report, "_THIS", scripts / "report.py")
    output = tmp_path / "report"
    _argv(
        monkeypatch,
        checkpoint,
        *("--skip", "baselines", "--skip", "tiles", "--skip", "matrix"),
        *("--skip", "dreams", "--output", str(output)),
        root=tmp_path,
    )
    assert report.main() == 1
    printed = capsys.readouterr().out
    assert "Failed: s0 engine_checks.py." in printed
    summary = from_plain(
        loads((output / "summary.json").read_text()),
        dict[str, object],
    )
    # The checks crashed: criterion 4 failed, it was not left unmeasured.
    assert summary["criterion_4_engine_checks"] == {"s0": False}
    report_md = (output / "report.md").read_text()
    assert "| 4. s0 passes the engine checks | FAIL |" in report_md


def _recording(calls: list[tuple[str, list[str]]]) -> Callable[..., int]:
    """Return a stand-in for ``report._child`` that records each script and its flags."""

    def child(script: str, arguments: Sequence[str], *, name: str, clock: float) -> int:
        del name, clock
        calls.append((script, list(arguments)))
        return 0

    return child


def _argv(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
    *flags: str,
    root: Path,
) -> None:
    """Set ``sys.argv`` to score ``checkpoint`` as ``exp_smoke`` on the CPU."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "report.py",
            str(checkpoint),
            *("--experiment", SMOKE, "--override", f"base_dir={root}"),
            *("--device", "cpu", "--tiles", "4"),
            *flags,
        ],
    )


def _smoke_checkpoint(root: Path) -> Path:
    """Publish ``exp_smoke``'s corpus and save a fresh model; return its checkpoint."""
    _smoke_corpus(root)
    torch.manual_seed(0)
    path = root / "step.pt"
    torch.save({"step": {"model": exp_smoke().step.model.make().state_dict()}}, path)
    return path


def _smoke_corpus(root: Path) -> None:
    """Publish one training and one validation shard where ``exp_smoke`` reads them."""
    archive = root / str(exp_smoke().dataset.working_dir).lstrip("/")
    entries: list[tuple[Path, ManifestLine]] = []
    for split, name in enumerate(("train", "val")):
        directory = archive / name / "arm0" / "w0"
        directory.mkdir(parents=True)
        episodes = [
            _episode(12 + 5 * i, seed=10 * split + i, split=split) for i in range(3)
        ]
        line = write_shard(directory, index=0, episodes=episodes, provenance={})
        entries.append((directory, line))
    write_corpus(archive / "corpora" / "smoke.json", entries=entries)


def _episode(decisions: int, *, seed: int, split: int) -> Episode:
    """Return an episode of random valid frames that ends in a death."""
    schema = craftax_schema()
    generator = torch.Generator().manual_seed(seed)
    cells = torch.stack(
        [
            torch.randint(0, field.valid, (decisions, 99), generator=generator)
            for field in schema.cell_fields
        ],
        dim=-1,
    )
    aux = torch.stack(
        [
            torch.randint(low - 155, high - 154, (decisions,), generator=generator)
            for low, high in schema.scalar_ranges
        ],
        dim=-1,
    )
    reward = torch.randint(0, 3, (decisions,), generator=generator).short()
    reward[-1] = -1
    return Episode(
        receipt=Receipt(
            world_seed=seed,
            sampling_seed=0,
            initial_state_hash=0,
            arm=0,
            split=split,
        ),
        actions=torch.randint(0, 43, (decisions,), generator=generator).byte(),
        hashes=torch.zeros(decisions // 256 + 1, dtype=torch.int64),
        cells=cells.byte(),
        aux=aux.short(),
        reward=reward,
        done=torch.arange(decisions) == decisions - 1,
        summary={"death": 1, "timeout": 0},
    )


def _baselines(*, beats: bool) -> dict[str, object]:
    """Return a baselines report the model wins or loses throughout."""
    return {
        "beats": {"action": beats, "reward": beats, "done": beats, "cells": beats},
        "model_nll": {"action": 0.5, "reward": 0.01, "done": 0.0},
        "empirical_nll": {"action": 2.0, "reward": 0.1, "done": 0.002},
        "cell_accuracy": {"model": 0.99, "copy": 0.8},
    }


def _in_loop(nats: float) -> dict[str, object]:
    """Return an in-loop evaluation scoring ``nats`` per decision in the natural mix."""
    return {
        "val/nats_per_decision_natural": nats,
        "val/bpb": nats / 1000,
        "val/zstd19_bpb": 0.1,
        **{f"val/bpb/{m}": 0.01 for m in MODALITIES},
    }


def _tiles(nats: float) -> dict[str, object]:
    """Return a data_eval report of ``nats`` per decision over 100 decisions."""
    metric = {"nats_per_decision": nats, "bpb": nats / 1000}
    return {"metric": metric | {f"bpb/{m}": 0.01 for m in MODALITIES}, "decisions": 100}


def _spread(parent: dict[str, object], key: str) -> dict[str, object]:
    """Return the primary metric's spread under ``parent[key]``."""
    return from_plain(
        from_plain(parent[key], dict[str, object])[report.PRIMARY],
        dict[str, object],
    )


def _write(path: Path, value: dict[str, object]) -> None:
    """Write one step's JSON output."""
    path.write_text(json.dumps(value))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
