from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast
from urllib.parse import quote, unquote

import functools
import json
import logging
import random
import re
import sys

from configgle import Fig
from torch import Tensor
from torch._inductor import config as inductor_config
from treekle import ReadError, json_loads

import numpy as np
import pytest
import torch

from priml.baselines.sudoku import experiments, latent_recursion, puzzle_data
from priml.baselines.sudoku.agreement_lock import (
    AgreementLockEval,
    nine_view_members,
)
from priml.baselines.sudoku.harvest import Harvest
from priml.baselines.sudoku.puzzle_data import PuzzleDataset
from priml.baselines.sudoku.reproduction import (
    Reproduction,
    _outer_run,
    _OuterRun,
    _run_training,
    _screen_pass,
)
from priml.baselines.sudoku.runner import load_eval_weights
from priml.baselines.sudoku.search import HpsSearch
from priml.baselines.sudoku.sieve import SieveEval
from priml.baselines.sudoku.testing import (
    FIXTURES,
    Harness,
    metric_tensors,
    replay,
    shrink_newton_schulz,
    weight_tensors,
    write_dataset,
)
from priml.baselines.sudoku.train_loop import SudokuTrainLoop
from priml.baselines.sudoku.verifier import VerifierAcceptor
from priml.baselines.sudoku.views import NINE_VIEWS
from priml.testing import regenerate
from priml.testing.golden import (
    assert_tensor_golden_part,
    locked_directory,
    mismatches,
    read_tensors,
    write_tensors,
)
from priml.train.tracker import WandbTracker


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.lib.npyio import NpzFile


_CWD: Final = Path(__file__).resolve().parent


class _ProtocolConfig(Protocol):
    model: latent_recursion.LatentRecursion.Config
    dataset: PuzzleDataset.Config
    search: HpsSearch.Config
    evaluation_count: int


_PipelineConfig = Reproduction.Config


def _pipeline(h: Harness, name: str) -> _PipelineConfig:
    """Return a 2-step pipeline of 1-step segments: the second resumes the first."""
    cfg = Reproduction.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = name
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    generator = experiments.exp006()
    generator.step.model = h.model()
    generator.step.dtype_autocast = None
    generator.step.pool.max_steps = 2
    shrink_newton_schulz(generator)
    generator.dataset.working_dir = h.data
    generator.dataset.batch_size = 2
    generator.dataset.eval_num_instances = None
    # The recipe augments; the segments draw from the dedicated seeded stream.
    generator.dataset.augment_digits_only = False
    generator.max_steps = 2
    generator.eval_warmup_batches = 0
    cfg.generator = generator
    cfg.generator_name = f"{name}_generator"
    cfg.eval_every_steps = 1
    cfg.dev_screen_count = 2
    # A two-step model never reaches the recipe's 0.98 gate; opening it makes
    # every boundary run its dev screen.
    cfg.screen_gate_det_accuracy = 0.0
    return cfg


def _full_eval_defaults(h: Harness, full_eval: _ProtocolConfig) -> None:
    """Configure a pipeline's full-set protocol for the shrunk fixture."""
    full_eval.model = h.model()
    full_eval.dataset.working_dir = h.data
    full_eval.dataset.batch_size = 2
    full_eval.search = h.search()
    full_eval.evaluation_count = 2


def _record_repro_lock(h: Harness) -> _PipelineConfig:
    cfg = _pipeline(h, "repro_lock")
    lock = AgreementLockEval.Config()
    _full_eval_defaults(h, lock)
    lock.members = nine_view_members("template.pt")[:2]
    cfg.full_eval = lock
    return cfg


def _record_repro_sieve(h: Harness) -> _PipelineConfig:
    # The frozen source is an input, not the run under test: the stored ``gen_a``
    # fixture, whose shard the harvest stage must reproduce. The generator this
    # pipeline trains is its own, at the pipeline's shorter size.
    source = h.generator("gen_a", seed=0)
    cfg = _pipeline(h, "repro_sieve")
    cfg.screen = "committee"
    cfg.harvest_source_checkpoint = source
    cfg.harvest = Harvest.Config()
    cfg.harvest.model = h.model()
    cfg.harvest.working_dir = "/data"
    cfg.harvest.group_count = 3
    cfg.harvest.views_per_group = 1
    cfg.harvest.max_act_steps = 2
    cfg.harvest.checkpoints = (1,)
    cfg.harvest.search_depth = 1
    cfg.harvest.search_candidates = 2
    cfg.harvest.search_cell_attempts = 1
    cfg.harvest.search_budget = 2
    cfg.harvest.random_corruption_strengths = (1,)
    cfg.harvest.random_starts_per_strength = 1
    cfg.harvest.dtype_autocast = None
    cfg.verifier_names = tuple(f"repro_verifier_s{s}" for s in range(3))
    cfg.verifier.model = h.verifier_model()
    cfg.verifier.dataset.working_dir = "/data"
    cfg.verifier.dataset.batch_size = 2
    cfg.verifier.dataset.steps_per_epoch = 1
    cfg.verifier.dataset.train_group_end = 1
    cfg.verifier.dtype_autocast = None
    cfg.verifier.max_steps = 3
    sieve = SieveEval.Config()
    _full_eval_defaults(h, sieve)
    sieve.views = NINE_VIEWS[:1]
    sieve.tail_search = (1, 1, 1, 1)
    acceptor = cast("VerifierAcceptor.Config", sieve.acceptor)
    acceptor.model = h.verifier_model()
    # The recipe's 0 locks nothing on a three-step committee (see
    # _record_sieve), which _record_sieve already covers; locking everything
    # instead drives the pipeline's committee screens and the sieve's lock.
    acceptor.threshold = -1e9
    cfg.full_eval = sieve
    return cfg


def _reproduction_config(tmp_path: Path) -> Reproduction.Config:
    """Return a constructible pipeline config that never trains."""
    cfg = Reproduction.Config()
    cfg.experiment_name = "repro"
    cfg.base_dir = tmp_path
    cfg.runtime.device = "cpu"
    cfg.generator.max_steps = 4
    cfg.generator.seed = 44
    cfg.generator_name = "gen"
    return cfg


def _rejects(
    cfg: Reproduction.Config,
    message: str,
    error: type[Exception] = ValueError,
) -> None:
    """Assert ``cfg`` fails to build with exactly ``message``."""
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        cfg.make()


def test_reproduction_rejects_an_invalid_config(tmp_path: Path) -> None:
    cfg = _reproduction_config(tmp_path)
    cfg.experiment_name = ""
    _rejects(
        cfg,
        "Reproduction.Config.experiment_name is required: it keys the aggregate "
        "run directory.",
    )
    cfg = _reproduction_config(tmp_path)
    cfg.eval_every_steps = 0
    _rejects(cfg, "eval_every_steps must be positive.")
    cfg = _reproduction_config(tmp_path)
    cfg.generator.max_steps = float("inf")
    _rejects(
        cfg,
        "the generator recipe needs a finite max_steps horizon (the staged "
        "boundaries derive from it).",
    )
    cfg = _reproduction_config(tmp_path)
    cfg.generator_name = ""
    _rejects(cfg, "generator_name must name a fresh checkpoint identity.")
    cfg = _reproduction_config(tmp_path)
    cfg.generator.seed = None
    _rejects(
        cfg,
        "the generator recipe needs a seed: it also seeds the dedicated "
        "augmentation stream that makes resumed segments replay exactly.",
    )
    cfg = _reproduction_config(tmp_path)
    cfg.generator.checkpointer = None
    _rejects(
        cfg,
        "the generator recipe needs a checkpointer: each segment resumes from the "
        "last, and every screen and full evaluation loads one.",
        TypeError,
    )
    cfg = _reproduction_config(tmp_path)
    cfg.full_eval = SieveEval.Config()
    _rejects(
        cfg,
        "nine_view screens run the agreement-lock protocol; full_eval must be an "
        "AgreementLockEval config.",
    )
    cfg = _reproduction_config(tmp_path)
    cfg.screen = "committee"
    _rejects(
        cfg,
        "harvest_source_checkpoint must name a frozen generator checkpoint (train "
        "the recipe generator first).",
    )
    cfg.harvest_source_checkpoint = "/runs/source/step.pt"
    cfg.verifier_names = ("a", "b")
    _rejects(cfg, "the committee is exactly the three verifier seed recipes.")
    cfg.verifier_names = ("a", "b", "c")
    _rejects(
        cfg,
        "committee screens share the sieve's search policy and acceptor; "
        "full_eval must be a SieveEval config.",
    )


@pytest.mark.parametrize("emulate", [True, False])
def test_reproduction_accepts_a_one_step_cadence_and_sets_the_cast_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    emulate: bool,
) -> None:
    """The smallest legal cadence builds; Inductor's flag follows the recipe step."""
    monkeypatch.setattr(inductor_config, "emulate_precision_casts", not emulate)
    cfg = _reproduction_config(tmp_path)
    cfg.eval_every_steps = 1
    cfg.generator.step.emulate_precision_casts = emulate
    assert cfg.make().config.eval_every_steps == 1
    assert inductor_config.emulate_precision_casts is emulate


def test_reproduction_restores_stage_seconds_from_progress(tmp_path: Path) -> None:
    repro = _reproduction_config(tmp_path).make()
    assert repro._load_stage_seconds() == {}
    metrics = repro._metrics_path()
    metrics.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"stage": "train_to_2", "seconds": 2.5}]
    metrics.write_text(json.dumps({"eval/stages": rows}))
    # A finished run's metrics are not a resume record.
    assert repro._load_stage_seconds() == {}
    restored = repro._record_stage_seconds(
        "train_to_2",
        9.0,
        completed=False,
        stage_seconds={"train_to_2": 2.5},
    )
    assert restored == 2.5
    recorded = repro._record_stage_seconds(
        "train_to_4",
        4.0,
        completed=True,
        stage_seconds={},
    )
    assert recorded == 4.0
    assert repro._load_stage_seconds() == {"train_to_4": 4.0}
    # A stage re-run to completion overwrites the restored duration.
    rerun = repro._record_stage_seconds(
        "train_to_4",
        6.0,
        completed=True,
        stage_seconds={"train_to_4": 4.0},
    )
    assert rerun == 6.0
    assert repro._load_stage_seconds() == {"train_to_4": 6.0}
    progress = repro._progress_path()
    for payload in (
        "[]",
        '{"schema_version": 2}',
        '{"schema_version": 1}',
        '{"schema_version": 1, "stage_seconds": {"a": "x"}}',
    ):
        progress.write_text(payload)
        with pytest.raises(ValueError, match="invalid reproduction progress"):
            repro._load_stage_seconds()


def test_reproduction_reuses_only_a_harvest_from_its_own_source(
    tmp_path: Path,
) -> None:
    cfg = _reproduction_config(tmp_path)
    source = tmp_path / "source.pt"
    cfg.harvest_source_checkpoint = source
    repro = cfg.make()
    out_dir = tmp_path / "runs" / cfg.harvest.experiment_name / "harvest"
    out_dir.mkdir(parents=True)
    manifest = out_dir / "manifest.json"
    manifest.write_text(json.dumps({"source_checkpoint": str(source)}))
    assert repro._run_harvest() == (out_dir, 0.0)
    manifest.write_text(json.dumps({"source_checkpoint": "/elsewhere.pt"}))
    with pytest.raises(ValueError, match="rolled out from"):
        repro._run_harvest()
    for text in ("[]", "{}"):
        manifest.write_text(text)
        with pytest.raises(ReadError):
            repro._run_harvest()


def test_reproduction_skips_a_trained_verifier(tmp_path: Path) -> None:
    cfg = _reproduction_config(tmp_path)
    cfg.verifier_names = ("member_a", "member_b", "member_c")
    repro = cfg.make()
    checkpoint = repro._verifier_checkpoint_path(1)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    assert repro._verifier_fit(1, tmp_path) == 0.0
    fit = repro._verifier_fit_config(2, tmp_path)
    assert (fit.experiment_name, fit.seed) == ("member_c", 2)
    suffix = f"/checkpoints/step_{cfg.verifier.max_steps:08d}.pt"
    assert repro._verifier_checkpoints()[0].endswith(suffix)


class _RecordingTracker:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int, str]] = []
        self.closed = False

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        self.logged.append((dict(metrics), step, prefix))

    def close(self) -> None:
        self.closed = True


def test_outer_run_clamps_steps_and_forwards_segments(tmp_path: Path) -> None:
    tracker = _RecordingTracker()
    outer = _OuterRun(tracker)
    outer.log({"a": 1}, 5, prefix="x/")
    outer.log({"b": 2}, 3)
    segment = tmp_path / "metrics.json"
    outer.forward_segment(segment, 7)
    segment.write_text('{"eval/loss": 0.5}')
    outer.forward_segment(segment, 7)
    outer.close()
    assert tracker.logged == [
        ({"a": 1}, 5, "x/"),
        ({"b": 2}, 5, ""),
        ({"eval/loss": 0.5}, 7, ""),
    ]
    assert tracker.closed
    segment.write_text("[]")
    with pytest.raises(ReadError):
        _OuterRun(_RecordingTracker()).forward_segment(segment, 8)
    disabled = _outer_run(None, name="n", notes="")
    disabled.log({"a": 1}, 2)
    disabled.forward_segment(segment, 2)
    disabled.close()


def test_staged_run_gates_screens_and_latches_full_evals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The gate opens at 4, whose perfect screen latches full evals at 4 and 6."""
    cfg = _reproduction_config(tmp_path)
    cfg.generator.max_steps = 6
    cfg.eval_every_steps = 2
    cfg.dev_screen_count = 2
    cfg.full_eval.evaluation_count = 2
    repro = cfg.make()
    calls: list[tuple[str, int]] = []

    def train_segment(boundary: int) -> tuple[float, bool]:
        calls.append(("train", boundary))
        return 1.0, True

    def det_gate(step: int) -> tuple[int, int, float]:
        calls.append(("gate", step))
        return (0 if step == 2 else 2), 2, 1.0

    def dev_screen(step: int) -> tuple[int, int, None, float]:
        calls.append(("screen", step))
        return 2, 2, None, 1.0

    def full_eval(step: int) -> tuple[dict[str, object], float]:
        calls.append(("full", step))
        return {"eval/n_correct": 1 if step == 4 else 2}, 1.0

    monkeypatch.setattr(repro, "_train_segment", train_segment)
    monkeypatch.setattr(repro, "_det_gate", det_gate)
    monkeypatch.setattr(repro, "_dev_screen", dev_screen)
    monkeypatch.setattr(repro, "_full_eval", full_eval)
    with caplog.at_level(logging.INFO):
        repro.run("--extra")
    assert "--extra" in caplog.text
    assert calls == [
        ("train", 2),
        ("gate", 2),
        ("train", 4),
        ("gate", 4),
        ("screen", 4),
        ("full", 4),
        ("train", 6),
        ("screen", 6),
        ("full", 6),
    ]
    results = json_loads(repro._metrics_path().read_text(), dict[str, object])
    assert results["eval/first_dev_perfect_step"] == 4
    assert results["eval/first_fullset_perfect_step"] == 6
    assert results["eval/total_seconds"] == 3 + 2 + 2 + 2


def test_checkpoint_logical_needs_a_generator_root(tmp_path: Path) -> None:
    cfg = _reproduction_config(tmp_path)
    cfg.base_dir = None
    cfg.generator.base_dir = None
    repro = cfg.make()
    with pytest.raises(ValueError, match="Expected base is not None"):
        repro._checkpoint_logical(2)


class _AcceptAll:
    """A committee lock that accepts every grid."""

    class Config(Fig["_AcceptAll"]):
        """No settings."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, preds: Tensor, media: Tensor) -> Tensor:
        del media
        return torch.ones(preds.shape[0], dtype=torch.bool)


def _committee_config(tmp_path: Path) -> Reproduction.Config:
    cfg = _reproduction_config(tmp_path)
    cfg.screen = "committee"
    cfg.harvest_source_checkpoint = "/runs/source/step.pt"
    cfg.verifier_names = ("a", "b", "c")
    cfg.full_eval = SieveEval.Config()
    return cfg


def test_screen_acceptor_repoints_only_a_verifier_committee(tmp_path: Path) -> None:
    """The default committee lock is re-pointed and keeps the finalized placement."""
    repro = _committee_config(tmp_path).make()
    acceptor = repro._screen_acceptor_config()
    assert isinstance(acceptor, VerifierAcceptor.Config)
    assert acceptor.checkpoint_paths == repro._verifier_checkpoints()
    # Reproduction.finalize -> SieveEval.finalize fill both before any build.
    assert (acceptor.device, acceptor.base_dir) == ("cpu", tmp_path)
    cfg = _committee_config(tmp_path)
    sieve = cast("SieveEval.Config", cfg.full_eval)
    sieve.acceptor = _AcceptAll.Config()
    custom = cfg.make()
    assert isinstance(custom.config.full_eval, SieveEval.Config)
    assert custom._screen_acceptor_config() is custom.config.full_eval.acceptor


def test_screen_pass_rejects_a_split_of_only_padding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = PuzzleDataset.Config(
        working_dir=write_dataset(tmp_path / "data"),
        batch_size=2,
        device="cpu",
    ).make()
    padded = [{**batch, "valid_count": 0} for batch in dataset.eval_dataloader()]
    monkeypatch.setattr(dataset, "eval_dataloader", lambda: padded)

    def run_search(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("an all-padding batch must not search")

    monkeypatch.setattr(
        "priml.baselines.sudoku.reproduction.run_search",
        run_search,
    )
    with pytest.raises(ValueError, match="dev screen found no valid eval rows"):
        _screen_pass(
            model=cast("latent_recursion.LatentRecursion", object()),
            dataset=dataset,
            search=HpsSearch.Config(),
            device=torch.device("cpu"),
        )


def test_outer_run_opens_one_resumable_wandb_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run id is a stable hash of project and name, so a rerun resumes it."""
    opened: list[WandbTracker.Config] = []

    def init(tracker: WandbTracker, config: WandbTracker.Config) -> None:
        del tracker
        opened.append(config)

    monkeypatch.setattr(WandbTracker, "__init__", init)
    _outer_run("proj", name="exp", notes="why")
    _outer_run("proj", name="exp", notes="why")
    _outer_run("proj", name="other", notes="")
    first, again, other = opened
    assert (first.project, first.name, first.notes) == ("proj", "exp", "why")
    assert len(first.run_id) == 16
    assert first.run_id == again.run_id != other.run_id


def test_run_training_skips_or_refuses_existing_segment_checkpoints(
    tmp_path: Path,
) -> None:
    cfg = SudokuTrainLoop()
    cfg.experiment_name = "segment"
    cfg.base_dir = tmp_path
    cfg.max_steps = 4
    checkpoints = tmp_path / "runs" / "segment" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step_00000006.pt").write_bytes(b"")
    with pytest.raises(RuntimeError, match="cannot safely resume"):
        _run_training(cfg)
    (checkpoints / "step_00000004.pt").write_bytes(b"")
    assert _run_training(cfg) == (0.0, False)


_RECORDERS: Final[dict[str, Callable[[Harness], Reproduction.Config]]] = {
    "exp008": _record_repro_sieve,
    "exp012": _record_repro_lock,
}


_STAGES: Final[dict[str, tuple[str, ...]]] = {
    "exp008": ("harvest", "verifier", "screen_1", "screen_2", "full_2", "run"),
    "exp012": ("train_1", "train_2", "gate_1", "screen_1", "screen_2", "full_2", "run"),
}
"""Each pipeline's stages, each replayed alone on what the stage before it froze.

``train_1`` freezes its whole checkpoint, which ``train_2`` resumes. Both
pipelines train the same generator, so only exp012 replays it, and every
evaluation reads its boundary eval weights from ``exp012.pt``. ``run``
drives the staged body over all of them, each evaluation's result stubbed with
what its own case froze, so it checks the orchestration alone.
"""

_TRAINED: Final = _CWD / "testdata" / "exp012.pt"
"""The golden holding the pipelines' trained boundary checkpoints."""


# Every stage reads only files: the stored generator and verifier fixtures, the
# stored shard, or the boundary checkpoints the golden froze, written where the
# stage before it would have left them.
def _stage_record(
    h: Harness,
    case: str,
    stage: str,
    golden: Mapping[str, Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Tensor]:
    """Run one stage of ``case``'s pipeline on stored inputs; return its record."""
    # Inputs, not the stage under test: read from the store even under
    # ``--regenerate-b4b``, so a re-mint changes only this stage's part.
    inputs = Harness(h.scratch, train=True, known=read_tensors(FIXTURES))
    repro = _RECORDERS[case](inputs).make()
    if stage == "harvest":
        return _harvest_stage(inputs, repro)
    harvest = repro.config.harvest.copy_tree()
    harvest.harvest_source_checkpoint = repro.config.harvest_source_checkpoint
    harvest_dir = inputs._stored_harvest(harvest)
    if stage == "verifier":
        return _verifier_stage(inputs, repro, harvest_dir)
    _write_committee(inputs, repro)
    if stage.startswith("train_"):
        return _train_stage(repro, stage, golden)
    _write_boundaries(repro, golden if case == "exp012" else read_tensors(_TRAINED))
    if stage == "run":
        # Stage 0 reuses the stored corpus and committee, and every segment
        # finds its boundary checkpoint, so each runs for real and skips.
        stub = _FrozenEvaluations(golden)
        monkeypatch.setattr(repro, "_det_gate", stub.det_gate)
        monkeypatch.setattr(repro, "_dev_screen", stub.dev_screen)
        monkeypatch.setattr(repro, "_full_eval", stub.full_eval)
        repro.run()
        metrics = json_loads(repro._metrics_path().read_text(), dict[str, object])
        return {f"{stage}/{k}": v for k, v in metric_tensors(metrics).items()}
    # The evaluation writes the dumps, so they are read only after it runs.
    record = _evaluation_stage(repro, stage)
    dumps = inputs.dumps(repro.config.experiment_name)
    return record | {f"{stage}/{k}": v for k, v in dumps.items()}


def _harvest_stage(h: Harness, repro: Reproduction) -> dict[str, Tensor]:
    """Harvest, require the shard the fixtures froze, and record its rows."""
    harvested, _ = repro._run_harvest()
    shard = _shard(harvested, prefix="harvest/")
    stored = {k: v for k, v in h.fixtures.items() if k.startswith("harvest/")}
    report = mismatches(stored, shard)
    assert not report, report
    return {"harvest/rows": torch.tensor(len(shard["harvest/original"]))}


# Each member's fit is the stored fixture's recipe at its seed, which ``testing_test``
# pins bit for bit; this pins the pipeline's recipe to it.
def _verifier_stage(
    h: Harness,
    repro: Reproduction,
    harvest_dir: Path,
) -> dict[str, Tensor]:
    """Fit committee member 0 and require the stored member's weights."""
    repro._verifier_fit(0, harvest_dir)
    weights = load_eval_weights(repro._verifier_checkpoint_path(0))
    report = mismatches(h._stored("verifier_s0/"), weights)
    assert not report, report
    return {"verifier/members": torch.tensor(len(repro.config.verifier_names))}


def _write_committee(h: Harness, repro: Reproduction) -> None:
    """Write the stored committee where a committee screen reads it."""
    if repro.config.screen != "committee":
        return
    for index in range(len(repro.config.verifier_names)):
        path = repro._verifier_checkpoint_path(index)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": h._stored(f"verifier_s{index}/")}, path)


def _boundaries(repro: Reproduction) -> tuple[int, int]:
    """Return the pipeline's two segment boundaries."""
    return repro.config.eval_every_steps, int(repro.config.generator.max_steps)


def _train_stage(
    repro: Reproduction,
    stage: str,
    golden: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Train one segment, resuming the first's frozen state; record what it wrote."""
    first, _ = _boundaries(repro)
    boundary = int(stage.removeprefix("train_"))
    if boundary != first:
        prefix = f"train_{first}/state/"
        if not any(key.startswith(prefix) for key in golden):
            raise ValueError(
                f"{stage} resumes {prefix}, which the golden does not hold; "
                f"mint train_{first} first.",
            )
        state = cast("dict[str, dict[str, object]]", _state(golden, prefix=prefix))
        _refill_generators(state, seed=0)
        path = repro._checkpoint_path(first)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
    repro._train_segment(boundary)
    record = _segment_record(repro, stage)
    if boundary == first:
        # ``train_2`` resumes from this checkpoint rebuilt from its record, so
        # the rebuild must hold every value the real file did.
        rebuilt = _tensors(_state(record, prefix=f"{stage}/state/"), prefix="")
        written = _tensors(_segment_state(repro, first), prefix="")
        assert mismatches(written, rebuilt) == []
    return record


def _segment_record(repro: Reproduction, stage: str) -> dict[str, Tensor]:
    """Record the eval weights a segment wrote, and its whole state if one follows."""
    boundary = int(stage.removeprefix("train_"))
    path = repro._checkpoint_path(boundary)
    out = weight_tensors(f"{stage}/eval", load_eval_weights(path))
    if boundary != _boundaries(repro)[-1]:
        out |= _tensors(_segment_state(repro, boundary), prefix=f"{stage}/state/")
    return out


def _segment_state(repro: Reproduction, boundary: int) -> dict[str, dict[str, object]]:
    """Load the checkpoint a segment wrote, without the generators it never reads."""
    state = cast(
        "dict[str, dict[str, object]]",
        torch.load(repro._checkpoint_path(boundary), weights_only=True),
    )
    _drop_generators(state)
    return state


def _write_boundaries(repro: Reproduction, trained: Mapping[str, Tensor]) -> None:
    """Write each boundary's frozen eval weights where an evaluation loads them."""
    for boundary in _boundaries(repro):
        path = repro._checkpoint_path(boundary)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = _golden_part(trained, f"train_{boundary}/eval/")
        torch.save({"step": {"model": model}}, path)


def _evaluation_stage(repro: Reproduction, stage: str) -> dict[str, Tensor]:
    """Run one gate, dev screen or full evaluation; record its result."""
    kind, _, at = stage.partition("_")
    step = int(at)
    if kind == "gate":
        exact, of, _ = repro._det_gate(step)
        return {f"{stage}/exact": torch.tensor(exact), f"{stage}/of": torch.tensor(of)}
    if kind == "screen":
        exact, of, false_accepts, _ = repro._dev_screen(step)
        out = {f"{stage}/exact": torch.tensor(exact), f"{stage}/of": torch.tensor(of)}
        if false_accepts is not None:
            out[f"{stage}/false_accepts"] = torch.tensor(false_accepts)
        return out
    metrics, _ = repro._full_eval(step)
    return {f"{stage}/{k}": v for k, v in metric_tensors(metrics).items()}


class _FrozenEvaluations:
    """Each evaluation the staged body calls, answered with what its case froze."""

    def __init__(self, golden: Mapping[str, Tensor]) -> None:
        self.golden = golden

    def det_gate(self, step: int) -> tuple[int, int, float]:
        """Return the frozen gate result at ``step``."""
        return self._int(f"gate_{step}/exact"), self._int(f"gate_{step}/of"), 1.0

    def dev_screen(self, step: int) -> tuple[int, int, int | None, float]:
        """Return the frozen dev screen at ``step``."""
        prefix = f"screen_{step}/"
        accepts = f"{prefix}false_accepts"
        false_accepts = self._int(accepts) if accepts in self.golden else None
        return self._int(f"{prefix}exact"), self._int(f"{prefix}of"), false_accepts, 1.0

    def full_eval(self, step: int) -> tuple[dict[str, object], float]:
        """Return the frozen full-set metrics at ``step``."""
        prefix = f"full_{step}/metrics/"
        metrics: dict[str, object] = {
            key.removeprefix(prefix): value.item()
            for key, value in self.golden.items()
            if key.startswith(prefix)
        }
        if f"{prefix}eval/locked_total" in self.golden:
            metrics["eval/rounds"] = []
        return metrics, 1.0

    def _int(self, key: str) -> int:
        return int(self.golden[key].item())


_GENERATORS: Final = (
    ("rng", "python"),
    ("rng", "torch"),
    ("step", "halt_rng"),
    ("step", "corruption_rng"),
)
"""The checkpoint's generator states, 21 KB of Mersenne words between them.

None of them is read by this recipe's second segment -- the segment reseeds its
draws -- so the stage stores the training state without them and refills them
from a fixed seed. A fixed segment schedule replays exactly, but is not one
uninterrupted run: a resume starts the next epoch's shuffle. So
``train_2``'s golden is what two segments write: ``train_1`` checks its real
checkpoint survives the round trip ``train_2`` resumes from.
``test_the_refilled_generators_are_never_read`` fails the day a segment starts
reading one of them.
"""


def _drop_generators(state: dict[str, dict[str, object]]) -> None:
    """Remove each of :data:`_GENERATORS` from a checkpoint, in place."""
    for group, name in _GENERATORS:
        del state[group][name]
    # A process that initialized CUDA also saves its device generators; the
    # recipe runs on the CPU, so they are host state, not the run's.
    state["rng"].pop("cuda", None)
    state["rng"].pop("cuda_uuids", None)


def _refill_generators(state: dict[str, dict[str, object]], *, seed: int) -> None:
    """Put a fresh generator state seeded ``seed`` at each of :data:`_GENERATORS`."""
    torch_state = torch.Generator().manual_seed(seed).get_state()
    stand_in = random.Random(seed)  # noqa: S311 -- A fixed state, not a secret.
    python_state = stand_in.getstate()
    for group, name in _GENERATORS:
        state[group][name] = python_state if name == "python" else torch_state


# A checkpoint as tensors keyed by path. Each segment names one step down: ``k``
# and a quoted string key, ``i`` and an int key, ``l`` or ``t`` and a list or tuple
# index. The last segment names the leaf: ``=tensor``, or the Python value it was
# (``=int`` and ``=str`` as decimal or UTF-8 bytes, since numpy's PCG64 words are
# 128-bit; ``=float``, ``=bool``, ``=none``; ``=dict``, ``=list``, ``=tuple`` for an
# empty one). A float under a key ending ``_sec`` is wall time, so it is zeroed: a
# resumed segment only adds to it.
def _tensors(node: object, *, prefix: str) -> dict[str, Tensor]:
    """Flatten a checkpoint into tensors keyed by their path under ``prefix``."""
    segments = _segments(node)
    if segments:
        return {
            key: value
            for segment, child in segments
            for key, value in _tensors(child, prefix=f"{prefix}{segment}/").items()
        }
    return {
        f"{prefix}={_kind(node)}": _leaf(node, zeroed=prefix.endswith("_sec/")),
    }


def _segments(node: object) -> list[tuple[str, object]]:
    """Name each child of a container by its path segment; a leaf has none."""
    if isinstance(node, dict):
        children = cast("dict[object, object]", node).items()
        return [(_key_segment(name), child) for name, child in children]
    if isinstance(node, list | tuple):
        tag = "t" if isinstance(node, tuple) else "l"
        items = cast("list[object]", node)
        return [(f"{tag}{index}", child) for index, child in enumerate(items)]
    return []


def _key_segment(name: object) -> str:
    """Name one dict key as a path segment: ``i`` and an int, or ``k`` and a string."""
    if isinstance(name, bool) or not isinstance(name, int | str):
        raise TypeError(
            f"A checkpoint key of type {type(name).__name__} has no encoding.",
        )
    return f"i{name}" if isinstance(name, int) else f"k{quote(name, safe='')}"


def _kind(node: object) -> str:
    """Name a leaf's type as its last path segment does."""
    if isinstance(node, Tensor):
        return "tensor"
    if node is None:
        return "none"
    return type(node).__name__


def _leaf(node: object, *, zeroed: bool) -> Tensor:
    """Encode one leaf as a tensor :func:`_value` decodes."""
    if isinstance(node, Tensor):
        return node.detach().clone()
    if isinstance(node, bool):
        return torch.tensor(node)
    if isinstance(node, str | int):
        return torch.tensor(list(str(node).encode()), dtype=torch.uint8)
    if isinstance(node, float):
        return torch.tensor(0.0 if zeroed else node, dtype=torch.float64)
    if node is None or node in ({}, [], ()):
        return torch.zeros(0)
    raise TypeError(f"A checkpoint leaf of type {type(node).__name__} has no encoding.")


def _state(golden: Mapping[str, Tensor], *, prefix: str) -> object:
    """Invert :func:`_tensors` for the keys under ``prefix``."""
    root: dict[str, object] = {}
    for key, value in golden.items():
        if key.startswith(prefix):
            *parents, leaf = key.removeprefix(prefix).split("/")
            node = root
            for segment in parents:
                node = cast("dict[str, object]", node.setdefault(segment, {}))
            node[leaf] = value
    return _built(root)


def _built(node: dict[str, object]) -> object:
    """Rebuild the value one path node of :func:`_state` holds."""
    [first, *_] = node
    if first.startswith("="):
        return _value(first.removeprefix("="), cast("Tensor", node[first]))
    children = {k: _built(cast("dict[str, object]", v)) for k, v in node.items()}
    if first[0] in "lt":
        items = [children[k] for k in sorted(children, key=lambda k: int(k[1:]))]
        return tuple(items) if first[0] == "t" else items
    return {
        int(k[1:]) if k[0] == "i" else unquote(k[1:]): value
        for k, value in children.items()
    }


def _value(kind: str, value: Tensor) -> object:
    """Decode one leaf :func:`_leaf` encoded as ``kind``."""
    if kind in {"str", "int"}:
        text = value.numpy().tobytes().decode()
        return int(text) if kind == "int" else text
    if kind in {"bool", "float"}:
        return bool(value.item()) if kind == "bool" else float(value.item())
    empty: dict[str, object] = {"none": None, "dict": {}, "list": [], "tuple": ()}
    return value if kind == "tensor" else empty[kind]


def _golden_part(golden: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Return the golden's tensors under ``prefix``, the prefix removed."""
    return {
        k.removeprefix(prefix): v for k, v in golden.items() if k.startswith(prefix)
    }


def _shard(out: Path, *, prefix: str) -> dict[str, Tensor]:
    """Every array of a harvest's one shard, keyed under ``prefix``."""
    with cast("NpzFile", np.load(out / "shard-00000.npz")) as shard:
        return {
            f"{prefix}{key}": torch.from_numpy(np.array(shard[key]))
            for key in shard.files
        }


@pytest.mark.parametrize(
    "node",
    [
        {"a": [], "b": ()},
        {"7": 1, 7: 2},
        {"a#b": 1, "c:d": "x", "e/f": None},
        {"seq": (1, [2.5, True]), "big": 2**127, "neg": -1, "empty": {}},
    ],
)
def test_a_checkpoint_round_trips_through_tensors(node: dict[object, object]) -> None:
    assert _state(_tensors(node, prefix="p/"), prefix="p/") == node


def test_every_seconds_leaf_is_zeroed_whatever_timer_holds_it() -> None:
    node = {"timer_step": {"global_sec": 2.5}, "timer_eval": {"epoch_sec": 1.5, "n": 3}}
    record = _tensors(node, prefix="p/")
    seconds = [key for key in record if "_sec/" in key]
    assert len(seconds) == 2
    assert all(record[key].item() == 0.0 for key in seconds)
    assert record["p/ktimer_eval/kn/=int"].tolist() == list(b"3")


def test_a_key_that_is_neither_text_nor_an_int_has_no_encoding() -> None:
    with pytest.raises(TypeError, match="float"):
        _tensors({1.5: 0}, prefix="p/")


def test_a_segment_state_holds_no_host_cuda_generators() -> None:
    state: dict[str, dict[str, object]] = {
        "rng": {"python": 0, "torch": 1, "cuda": [2], "cuda_uuids": ["GPU-0"]},
        "step": {"halt_rng": 3, "corruption_rng": 4, "model": 5},
    }
    _drop_generators(state)
    assert state == {"rng": {}, "step": {"model": 5}}


def test_a_resumed_segment_needs_the_first_frozen(tmp_path: Path) -> None:
    write_dataset(tmp_path / "data")
    h = Harness(tmp_path, train=True, known=read_tensors(FIXTURES))
    repro = _RECORDERS["exp012"](h).make()
    with pytest.raises(ValueError, match=r"^train_2 resumes train_1/state/, "):
        _train_stage(repro, "train_2", {})


@pytest.mark.parametrize("seed", [1, 7])
def test_the_refilled_generators_are_never_read(
    seed: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    puzzle_data.dihedral_indices.cache_clear()
    golden = read_tensors(_TRAINED)
    refill = functools.partial(_refill_generators, seed=seed)
    monkeypatch.setattr(sys.modules[__name__], "_refill_generators", refill)
    record = replay(
        lambda h: _stage_record(h, "exp012", "train_2", golden, monkeypatch),
        tmp_path,
        train=False,
    )
    part = {k: v for k, v in golden.items() if k.startswith("train_2/")}
    assert mismatches(part, record) == []


@pytest.mark.parametrize(
    ("case", "stage"),
    [(case, stage) for case, stages in _STAGES.items() for stage in stages],
)
def test_a_pipeline_stage_replays_bit_for_bit(
    case: str,
    stage: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_goldens: dict[str, dict[str, Tensor]],
) -> None:
    """Each stage reproduces its part of the frozen pipeline with zero mismatches."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "torchinductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    golden = pipeline_goldens[case]
    record = _replay_stage(case, stage, golden, tmp_path / "replay", monkeypatch)
    assert_tensor_golden_part(_CWD / "testdata" / f"{case}.pt", f"{stage}/", record)


@pytest.mark.parametrize("b4b", [False, True])
def test_the_pipeline_goldens_regenerate_only_under_the_flag(
    b4b: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    regenerated: list[Path] = []
    monkeypatch.setattr(
        sys.modules[__name__],
        "_regenerate_pipeline",
        regenerated.append,
    )
    regenerate.override(monkeypatch, b4b=b4b)
    goldens = _pipeline_goldens(tmp_path)
    assert regenerated == ([tmp_path] if b4b else [])
    assert set(goldens) == set(_STAGES)


@pytest.fixture(scope="module")
def pipeline_goldens(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, Tensor]]:
    """Return each pipeline's golden, regenerated once per module under the flag."""
    return _pipeline_goldens(tmp_path_factory.mktemp("regenerate"))


def _pipeline_goldens(scratch: Path) -> dict[str, dict[str, Tensor]]:
    """Regenerate both pipelines under ``--regenerate-b4b``, then read each golden."""
    if regenerate.b4b():
        _regenerate_pipeline(scratch)
    return {case: read_tensors(_CWD / "testdata" / f"{case}.pt") for case in _STAGES}


@pytest.mark.parametrize("case", list(_STAGES))
def test_a_pipeline_golden_holds_exactly_its_stages(case: str) -> None:
    golden = read_tensors(_CWD / "testdata" / f"{case}.pt")
    assert {key.split("/", 1)[0] for key in golden} == set(_STAGES[case])


def _replay_stage(
    case: str,
    stage: str,
    golden: Mapping[str, Tensor],
    scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Tensor]:
    """Replay one stage of ``case`` in ``scratch``; return its record."""
    puzzle_data.dihedral_indices.cache_clear()
    return replay(
        lambda h: _stage_record(h, case, stage, golden, monkeypatch),
        scratch,
        train=False,
    )


# Every stage reads what the stages before it froze, and exp008's evaluations
# read exp012's trained boundaries, so a regeneration re-mints both pipelines
# whole and in order, then rewrites each file: no stage is minted from a stale
# input, and a stage no longer in ``_STAGES`` is dropped. The lock serializes
# xdist workers, and every part writer of these goldens; each re-mints the same bits.
def _regenerate_pipeline(scratch: Path) -> None:
    """Re-mint every stage of both pipelines in order and rewrite their goldens."""
    with (
        locked_directory(_CWD / "testdata"),
        regenerate.suppressed(),
        pytest.MonkeyPatch.context() as monkeypatch,
    ):
        for case in ("exp012", "exp008"):
            golden: dict[str, Tensor] = {}
            for stage in _STAGES[case]:
                where = scratch / case / stage
                golden |= _replay_stage(case, stage, golden, where, monkeypatch)
            write_tensors(_CWD / "testdata" / f"{case}.pt", golden)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
