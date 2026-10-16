from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

import json
import logging
import re

from configgle import Fig
from torch import Tensor
from torch._inductor import config as inductor_config
from treekle import ReadError, json_loads

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
from priml.baselines.sudoku.search import HpsSearch
from priml.baselines.sudoku.sieve import SieveEval
from priml.baselines.sudoku.testing import (
    Harness,
    load_golden,
    metric_tensors,
    replay,
    weight_tensors,
    write_dataset,
)
from priml.baselines.sudoku.train_loop import SudokuTrainLoop
from priml.baselines.sudoku.verifier import VerifierAcceptor
from priml.baselines.sudoku.views import NINE_VIEWS
from priml.testing.golden import mismatches
from priml.train.tracker import WandbTracker


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


class _ProtocolConfig(Protocol):
    model: latent_recursion.LatentRecursion.Config
    dataset: PuzzleDataset.Config
    search: HpsSearch.Config
    evaluation_count: int


_PipelineConfig = Reproduction.Config


def _pipeline(h: Harness, name: str) -> _PipelineConfig:
    """Return a 3-step pipeline with 2-step evaluation segments."""
    cfg = Reproduction.Config()
    cfg.study_name = "sudoku"
    cfg.experiment_name = name
    cfg.base_dir = h.scratch
    cfg.runtime.device = "cpu"
    generator = experiments.exp006()
    generator.step.model = h.model()
    generator.step.dtype_autocast = None
    generator.step.pool.max_steps = 2
    generator.dataset.working_dir = h.data
    generator.dataset.batch_size = 2
    generator.dataset.eval_num_instances = None
    # The recipe augments; the segments draw from the dedicated seeded stream.
    generator.dataset.augment_digits_only = False
    generator.max_steps = 3
    generator.eval_warmup_batches = 0
    cfg.generator = generator
    cfg.generator_name = f"{name}_generator"
    cfg.eval_every_steps = 2
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


def _run_pipeline(h: Harness, config: _PipelineConfig) -> dict[str, Tensor]:
    """Run a pipeline; record its dumps, metrics, and final generator weights."""
    config.make().run()
    out = h.dumps(config.experiment_name)
    metrics = cast(
        "dict[str, object]",
        json.loads(
            (h.scratch / "runs" / config.experiment_name / "metrics.json").read_text(),
        ),
    )
    out |= metric_tensors(metrics)
    generator = config.generator_name
    state = cast(
        "dict[str, dict[str, dict[str, Tensor]]]",
        torch.load(
            h.scratch
            / "runs"
            / config.generator.study_name
            / generator
            / "checkpoints"
            / f"step_{int(config.generator.max_steps):08d}.pt",
            weights_only=True,
        ),
    )
    return out | weight_tensors(generator, state["step"]["model"])


def _record_repro_lock(h: Harness) -> dict[str, Tensor]:
    cfg = _pipeline(h, "repro_lock")
    lock = AgreementLockEval.Config()
    _full_eval_defaults(h, lock)
    lock.members = nine_view_members("template.pt")[:2]
    cfg.full_eval = lock
    return _run_pipeline(h, cfg)


def _record_repro_sieve(h: Harness) -> dict[str, Tensor]:
    # The frozen source is an input, not the run under test: the generator this
    # pipeline trains is its own. ``gen_a`` is the same recipe at the same seed.
    source = Harness(h.scratch, train=False).generator("gen_a", seed=0)
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
    out = _run_pipeline(h, cfg)
    for name in cfg.verifier_names:
        state = cast(
            "dict[str, dict[str, dict[str, Tensor]]]",
            torch.load(
                h.scratch / "runs" / name / "checkpoints" / "step_00000003.pt",
                weights_only=True,
            ),
        )
        out |= weight_tensors(name, state["step"]["model"])
    return out


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


_RECORDERS: Final[dict[str, Callable[[Harness], dict[str, Tensor]]]] = {
    "exp008": _record_repro_sieve,
    "exp012": _record_repro_lock,
}


@pytest.mark.parametrize("case", _RECORDERS)
def test_golden_replays_bit_for_bit(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner reproduces the frozen evaluation with zero mismatches."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "torchinductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    puzzle_data.dihedral_indices.cache_clear()
    report = mismatches(load_golden(case), replay(_RECORDERS[case], tmp_path))
    assert not report, f"{len(report)} mismatches:\n" + "\n".join(report)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
