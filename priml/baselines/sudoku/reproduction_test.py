from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, cast

import json

from torch import Tensor
from treekle import ReadError

import pytest
import torch

from priml.baselines.sudoku import latent_recursion, puzzle_data
from priml.baselines.sudoku.agreement_lock import (
    AgreementLockEval,
    nine_view_members,
)
from priml.baselines.sudoku.harvest import Harvest
from priml.baselines.sudoku.reproduction import (
    Reproduction,
    _outer_run,
    _OuterRun,
    _run_training,
)
from priml.baselines.sudoku.sieve import SieveEval
from priml.baselines.sudoku.testing import (
    Harness,
    load_golden,
    metric_tensors,
    replay,
    weight_tensors,
)
from priml.baselines.sudoku.trainer import Trainer
from priml.baselines.sudoku.views import NINE_VIEWS
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from priml.baselines.sudoku.puzzle_data import PuzzleDataset
    from priml.baselines.sudoku.search import HpsSearch
    from priml.baselines.sudoku.verifier import VerifierAcceptor


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
    generator = Trainer.Config()
    generator.model = h.model()
    generator.dataset.working_dir = h.data
    generator.dataset.batch_size = 2
    generator.dataset.augment = True  # The recipe's; draws from the segment seed.
    generator.max_steps = 3
    generator.max_act_steps = 2
    generator.dtype_autocast = None
    generator.eval_warmup_batches = 0
    generator.seed = 44
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
    source = h.generator("source", seed=0)
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


def test_reproduction_rejects_an_invalid_config(tmp_path: Path) -> None:
    cfg = _reproduction_config(tmp_path)
    cfg.experiment_name = ""
    with pytest.raises(ValueError, match="experiment_name is required"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.eval_every_steps = 0
    with pytest.raises(ValueError, match="eval_every_steps"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.generator.max_steps = float("inf")
    with pytest.raises(ValueError, match="finite max_steps"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.generator_name = ""
    with pytest.raises(ValueError, match="generator_name"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.generator.seed = None
    with pytest.raises(ValueError, match="needs a seed"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.full_eval = SieveEval.Config()
    with pytest.raises(ValueError, match="nine_view"):
        cfg.make()
    cfg = _reproduction_config(tmp_path)
    cfg.screen = "committee"
    with pytest.raises(ValueError, match="harvest_source_checkpoint"):
        cfg.make()
    cfg.harvest_source_checkpoint = "/runs/source/step.pt"
    cfg.verifier_names = ("a", "b")
    with pytest.raises(ValueError, match="three verifier"):
        cfg.make()
    cfg.verifier_names = ("a", "b", "c")
    with pytest.raises(ValueError, match="SieveEval"):
        cfg.make()


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


def test_run_training_skips_or_refuses_existing_segment_checkpoints(
    tmp_path: Path,
) -> None:
    cfg = Trainer.Config()
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
    from priml.lib.testing.main import test_main

    test_main(__file__)
