"""Tests for experiment trackers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import Event, Thread, get_ident
from typing import TYPE_CHECKING, Any, cast

import json
import tempfile

from configgle import Fig
from wandb.sdk.wandb_run import Run

import torch

from priml.train import tracker as tracker_mod
from priml.train.tracker import (
    AsyncTracker,
    FileTracker,
    TensorBoardTracker,
    TrackerList,
    WandbIngestion,
    WandbTracker,
    default_metrics_tracker,
)


if TYPE_CHECKING:
    import pytest


class _FakeWriter:
    """Records add_scalar / close calls in place of a real SummaryWriter."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, Any, int]] = []
        self.close_count = 0

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def close(self) -> None:
        self.close_count += 1


def _tracker_with_fake_writer() -> tuple[TensorBoardTracker, _FakeWriter]:
    """Build a TensorBoardTracker whose writer is a recording fake."""
    tracker = TensorBoardTracker.__new__(TensorBoardTracker)
    writer = _FakeWriter()
    tracker.writer = writer
    return tracker, writer


def test_tracker_logs_scalar_tensor() -> None:
    tracker, writer = _tracker_with_fake_writer()
    tracker.log_metrics({"loss": torch.tensor(2.0)}, step=3)
    assert writer.scalars == [("loss", 2.0, 3)]


def test_del_safe_when_writer_missing() -> None:
    """T-029: __del__ must not raise when __init__ never set self.writer."""
    tracker = TensorBoardTracker.__new__(TensorBoardTracker)
    # Simulate a partially-constructed tracker (writer never assigned).
    tracker.__del__()  # must not raise AttributeError


def test_del_does_not_double_close_after_explicit_close() -> None:
    """T-029: explicit close followed by GC __del__ must not double-close."""
    tracker, writer = _tracker_with_fake_writer()
    tracker.close()
    tracker.__del__()
    assert writer.close_count == 1


def test_log_metrics_skips_non_scalar_value() -> None:
    """A dict-valued metric is skipped, not logged (no flatten, no raise)."""
    tracker, writer = _tracker_with_fake_writer()
    tracker.log_metrics(
        {"score": 1.0, "bundle": {"loss": torch.tensor(1.0), "aux": 5.0}}, step=0
    )
    assert writer.scalars == [("score", 1.0, 0)]


def test_tensorboard_log_metrics_prepends_prefix() -> None:
    """The prefix is prepended to each scalar key."""
    tracker, writer = _tracker_with_fake_writer()
    tracker.log_metrics({"loss": 0.5}, step=2, prefix="eval/")
    assert writer.scalars == [("eval/loss", 0.5, 2)]


def test_tensorboard_default_working_dir_is_opinionated() -> None:
    assert TensorBoardTracker.Config().working_dir == "/tensorboard"


def test_tensorboard_working_dir_is_scoped_to_the_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: list[str] = []

    def writer_factory(log_dir: str) -> _FakeWriter:
        observed.append(log_dir)
        return _FakeWriter()

    monkeypatch.setattr(tracker_mod, "_summary_writer_cls", writer_factory)
    config = TensorBoardTracker.Config()
    config.base_dir = tmp_path

    config.make()

    assert observed == [str(tmp_path / "tensorboard")]


# -- WandbTracker ------------------------------------------------------------


class _FakeRun:
    """Records log / finish calls in place of a real wandb run."""

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.defined_metrics: list[tuple[str, str | None]] = []
        self.finish_count = 0
        self.notes: str | None = None

    def define_metric(self, name: str, *, step_metric: str | None = None) -> None:
        self.defined_metrics.append((name, step_metric))

    def log(
        self,
        data: dict[str, Any],
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        del commit
        self.logged.append((dict(data), step if step is not None else -1))

    def finish(self, exit_code: int | None = None, quiet: bool | None = None) -> None:
        del exit_code, quiet
        self.finish_count += 1


def _wandb_tracker_with_fake_run() -> tuple[WandbTracker, _FakeRun]:
    """Build a WandbTracker on a recording fake run (rank-0 path, no wandb.init)."""
    tracker = WandbTracker.__new__(WandbTracker)
    run = _FakeRun()
    tracker._run = cast(Run, run)
    return tracker, run


def test_wandb_logs_scalar_tensor_coerced() -> None:
    tracker, run = _wandb_tracker_with_fake_run()
    tracker.log_metrics({"loss": torch.tensor(2.0), "lr": 0.1}, step=5)
    assert run.logged == [({"loss": 2.0, "lr": 0.1}, 5)]


def test_wandb_skips_non_scalar_value() -> None:
    """A non-scalar metric (dict payload) is skipped, not raised on."""
    tracker, run = _wandb_tracker_with_fake_run()
    tracker.log_metrics({"score": 1.0, "extras": {"payload": object()}}, step=0)
    assert run.logged == [({"score": 1.0}, 0)]


def test_wandb_log_metrics_prepends_prefix() -> None:
    """The prefix is prepended to each logged scalar key."""
    tracker, run = _wandb_tracker_with_fake_run()
    tracker.log_metrics({"loss": 0.5}, step=2, prefix="eval/")
    assert run.logged == [({"eval/loss": 0.5}, 2)]


def test_wandb_logs_images(monkeypatch: pytest.MonkeyPatch) -> None:
    tracker, run = _wandb_tracker_with_fake_run()

    class _FakeImage:
        def __init__(self, image: Any) -> None:
            self.image = image

    class _FakeWandbImage:
        Image = _FakeImage

    monkeypatch.setattr(tracker_mod, "wandb", _FakeWandbImage)
    tracker.log_images("eval/samples", ["samples.png"], step=5)
    logged, step = run.logged[0]
    assert step == 5
    image = logged["eval/samples"][0]
    assert isinstance(image, _FakeImage)
    assert image.image == "samples.png"


def test_wandb_close_idempotent_and_del_safe() -> None:
    tracker, run = _wandb_tracker_with_fake_run()
    tracker.close()
    tracker.__del__()
    assert run.finish_count == 1


def test_wandb_del_safe_when_run_missing() -> None:
    """__del__ must not raise when __init__ never set self._run."""
    tracker = WandbTracker.__new__(WandbTracker)
    tracker.__del__()  # must not raise AttributeError


def test_wandb_log_notes_sets_run_notes() -> None:
    """log_notes sets the run overview when no note is present."""
    tracker, run = _wandb_tracker_with_fake_run()
    tracker.log_notes("Hypothesis: X.")
    assert run.notes == "Hypothesis: X."


def test_wandb_log_notes_does_not_overwrite_explicit_note() -> None:
    """An explicitly-configured note wins over the launcher's docstring."""
    tracker, run = _wandb_tracker_with_fake_run()
    run.notes = "explicit note"
    tracker.log_notes("docstring note")
    assert run.notes == "explicit note"


def test_wandb_log_notes_noop_without_run() -> None:
    """Off rank 0 (no run) log_notes is a no-op, not a crash."""
    tracker = WandbTracker.__new__(WandbTracker)
    tracker._run = None
    tracker.log_notes("anything")  # must not raise


def test_wandb_non_rank_zero_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off rank 0, __init__ opens no run and log/close are no-ops.

    Guards the single-run invariant: an N-rank job must not spawn N runs.
    ``wandb.init`` is never imported on a non-rank-0 process, so a missing
    wandb install would not even matter there.
    """
    monkeypatch.setattr(tracker_mod, "is_rank_zero", lambda: False)
    tracker = WandbTracker(WandbTracker.Config(project="trm"))
    assert tracker._run is None
    tracker.log_metrics({"loss": torch.tensor(1.0)}, step=0)  # no-op, no raise
    tracker.close()  # no-op, no raise


def _init_tracker(
    monkeypatch: pytest.MonkeyPatch, config: WandbTracker.Config
) -> tuple[dict[str, Any], _FakeRun]:
    """Build a rank-0 WandbTracker against a fake W&B run."""
    captured: dict[str, Any] = {}
    run = _FakeRun()

    class _FakeWandb:
        class Settings:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        @staticmethod
        def init(**kwargs: Any) -> _FakeRun:
            captured.update(kwargs)
            return run

    monkeypatch.setattr(tracker_mod, "is_rank_zero", lambda: True)
    monkeypatch.setattr(tracker_mod, "wandb", _FakeWandb)
    if config.working_dir == "/wandb":
        config.working_dir = Path(tempfile.mkdtemp())
    WandbTracker(config)
    return captured, run


def _init_kwargs(
    monkeypatch: pytest.MonkeyPatch, config: WandbTracker.Config
) -> dict[str, Any]:
    """Build a rank-0 WandbTracker against a fake wandb; return init kwargs."""
    captured, _ = _init_tracker(monkeypatch, config)
    return captured


def test_wandb_defines_configured_metric_step_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WandbTracker.Config(project="trm")
    config.metric_step_metrics["train/loss_raw_time"] = "train/charged_seconds"

    _, run = _init_tracker(monkeypatch, config)

    assert run.defined_metrics == [
        ("train/charged_seconds", None),
        ("train/loss_raw_time", "train/charged_seconds"),
    ]


def test_wandb_run_name_ignores_slurm_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_NAME", "hostile_ambient_name")

    kwargs = _init_kwargs(monkeypatch, WandbTracker.Config(project="trm"))

    assert kwargs["name"] is None


def test_wandb_explicit_name_overrides_slurm_job_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit config name wins over the SLURM_JOB_NAME fallback."""
    monkeypatch.setenv("SLURM_JOB_NAME", "job_xyz")
    kwargs = _init_kwargs(
        monkeypatch, WandbTracker.Config(project="trm", name="my_run")
    )
    assert kwargs["name"] == "my_run"


def test_wandb_run_name_none_without_job_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No config name and no SLURM_JOB_NAME -> None (W&B auto-names)."""
    monkeypatch.delenv("SLURM_JOB_NAME", raising=False)
    kwargs = _init_kwargs(monkeypatch, WandbTracker.Config(project="trm"))
    assert kwargs["name"] is None


def test_wandb_run_id_resumes_existing_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured run_id resumes that exact run (id + resume='allow').

    Lets a standalone eval job append eval/* to the training run that produced
    the checkpoint instead of opening a separate run.
    """
    kwargs = _init_kwargs(
        monkeypatch, WandbTracker.Config(project="trm", run_id="9vralbfd")
    )
    assert kwargs["id"] == "9vralbfd"
    assert kwargs["resume"] == "allow"


def test_wandb_no_run_id_opens_fresh_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty run_id -> fresh run (id=None, resume=None)."""
    kwargs = _init_kwargs(monkeypatch, WandbTracker.Config(project="trm"))
    assert kwargs["id"] is None
    assert kwargs["resume"] is None


def test_wandb_defaults_bound_startup_and_capture_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default W&B settings keep tracker startup bounded and console on."""
    kwargs = _init_kwargs(monkeypatch, WandbTracker.Config(project="trm"))
    settings = kwargs["settings"]
    assert settings is not None
    assert "console" not in settings.kwargs
    assert settings.kwargs["init_timeout"] == 30.0
    assert settings.kwargs["x_service_wait"] == 30.0


def test_wandb_capture_console_false_disables_console(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """capture_console=False maps to W&B console='off'."""
    kwargs = _init_kwargs(
        monkeypatch,
        WandbTracker.Config(project="trm", capture_console=False),
    )
    settings = kwargs["settings"]
    assert settings is not None
    assert settings.kwargs["console"] == "off"


def test_wandb_capture_console_rebinds_logging_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank-0 W&B console capture retargets Loop's logging stream."""
    called = False

    def _bind_logging_to_current_stdout() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        tracker_mod,
        "bind_logging_to_current_stdout",
        _bind_logging_to_current_stdout,
    )

    _init_kwargs(monkeypatch, WandbTracker.Config(project="trm"))

    assert called is True


def test_wandb_capture_console_disabled_does_not_rebind_logging_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When W&B console is off, leave the existing job-log stream untouched."""
    called = False

    def _bind_logging_to_current_stdout() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        tracker_mod,
        "bind_logging_to_current_stdout",
        _bind_logging_to_current_stdout,
    )

    _init_kwargs(
        monkeypatch,
        WandbTracker.Config(project="trm", capture_console=False),
    )

    assert called is False


def test_wandb_flush_interval_bounds_history_buffering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """flush_interval_sec maps to Settings.x_file_stream_transmit_interval.

    Regression: a long run's history stream buffered indefinitely client-side,
    so dashboards showed "no data" while the summary kept updating. A bounded
    transmit interval keeps dashboard data flowing.
    """
    kwargs = _init_kwargs(
        monkeypatch,
        WandbTracker.Config(
            project="trm",
            ingestion=WandbIngestion(flush_interval_sec=15.0),
        ),
    )
    settings = kwargs["settings"]
    assert settings is not None
    assert settings.kwargs["x_file_stream_transmit_interval"] == 15.0


def test_wandb_system_metrics_interval_throttles_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """system_metrics_interval_sec maps to Settings.x_stats_sampling_interval.

    The built-in system metrics dominate history cardinality; sampling them
    less often cuts the logged-point volume that backs up the dashboard.
    """
    kwargs = _init_kwargs(
        monkeypatch,
        WandbTracker.Config(
            project="trm",
            ingestion=WandbIngestion(system_metrics_interval_sec=60.0),
        ),
    )
    settings = kwargs["settings"]
    assert settings is not None
    assert settings.kwargs["x_stats_sampling_interval"] == 60.0
    # System metrics stay ON (sampled, not disabled).
    assert "x_disable_stats" not in settings.kwargs


def test_wandb_system_metrics_off_disables_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """system_metrics=False disables the built-in stats outright."""
    kwargs = _init_kwargs(
        monkeypatch,
        WandbTracker.Config(
            project="trm",
            ingestion=WandbIngestion(system_metrics=False),
        ),
    )
    settings = kwargs["settings"]
    assert settings is not None
    assert settings.kwargs["x_disable_stats"] is True
    # Disabling supersedes the sampling-interval knob.
    assert "x_stats_sampling_interval" not in settings.kwargs


def test_wandb_no_tuning_keeps_default_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All tuning knobs neutral -> settings=None (W&B defaults)."""
    kwargs = _init_kwargs(
        monkeypatch,
        WandbTracker.Config(
            project="trm",
            capture_console=True,
            ingestion=WandbIngestion(
                init_timeout_sec=0.0,
                service_wait_sec=0.0,
                flush_interval_sec=0.0,
                system_metrics=True,
                system_metrics_interval_sec=0.0,
            ),
        ),
    )
    assert kwargs["settings"] is None


def test_wandb_startup_failure_becomes_noop_tracker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A W&B init failure should not strand a distributed training run."""

    class _FailingWandb:
        class Settings:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

        @staticmethod
        def init(**kwargs: Any) -> _FakeRun:
            del kwargs
            raise RuntimeError("wandb unavailable")

    monkeypatch.setattr(tracker_mod, "is_rank_zero", lambda: True)
    monkeypatch.setattr(tracker_mod, "wandb", _FailingWandb)

    config = WandbTracker.Config(project="trm")
    config.working_dir = Path(tempfile.mkdtemp())
    tracker = WandbTracker(config)

    assert tracker._run is None


# -- FileTracker -------------------------------------------------------------


def test_file_tracker_writes_prefixed_scalar_json(tmp_path: Path) -> None:
    """FileTracker expands explicit context and writes prefixed scalar JSON.

    Non-scalar values (an ``extras`` payload) are dropped; the file holds only
    numeric metrics, keyed by ``prefix + name`` (the format consumers read).
    """
    tracker = FileTracker.Config(
        working_dir=tmp_path / "metrics.json",
    ).make()

    tracker.log_metrics(
        {"exact_accuracy": 0.5, "loss": torch.tensor(0.25), "extras": {"x": object()}},
        7,
        prefix="eval/",
    )

    out = tmp_path / "metrics.json"
    assert json.loads(out.read_text()) == {
        "eval/exact_accuracy": 0.5,
        "eval/loss": 0.25,
    }
    assert not list(tmp_path.glob("*.tmp.*"))


def test_file_tracker_last_write_wins(tmp_path: Path) -> None:
    """Each call rewrites the file; it converges to the final value."""
    target = tmp_path / "metrics.json"
    tracker = FileTracker.Config(working_dir=str(target)).make()
    tracker.log_metrics({"score": 0.1}, 1, prefix="eval/")
    tracker.log_metrics({"score": 0.9}, 2, prefix="eval/")
    assert json.loads(target.read_text()) == {"eval/score": 0.9}


def test_file_tracker_ignores_non_capture_prefix(tmp_path: Path) -> None:
    """Train-step calls (prefix != capture_prefix) do not touch the eval file.

    The same tracker receives both train/ and eval/ metrics; only the eval ones
    belong in metrics.json. A train/ call must not write or clobber it.
    """
    target = tmp_path / "metrics.json"
    tracker = FileTracker.Config(working_dir=str(target)).make()
    tracker.log_metrics({"batch_time": 1.25}, 1, prefix="train/")
    assert not target.exists()
    tracker.log_metrics({"score": 0.9}, 2, prefix="eval/")
    tracker.log_metrics({"batch_time": 1.30}, 3, prefix="train/")  # must not clobber
    assert json.loads(target.read_text()) == {"eval/score": 0.9}


def test_file_tracker_empty_path_is_noop(tmp_path: Path) -> None:
    """An empty path disables the file."""
    FileTracker.Config(working_dir="").make().log_metrics(
        {"score": 1.0},
        0,
        prefix="eval/",
    )
    assert not list(tmp_path.iterdir())


def test_file_tracker_non_rank_zero_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off rank 0, FileTracker writes nothing (one file per job, not per rank)."""
    monkeypatch.setattr(tracker_mod, "is_rank_zero", lambda: False)
    target = tmp_path / "metrics.json"
    FileTracker.Config(working_dir=str(target)).make().log_metrics(
        {"score": 1.0}, 0, prefix="eval/"
    )
    assert not target.exists()


# -- TrackerList -------------------------------------------------------------


class _RecordingChild:
    """Child tracker recording each fan-out call for TrackerList tests."""

    class Config(Fig["_RecordingChild"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.metrics: list[tuple[dict[str, Any], int, str]] = []
        self.images: list[tuple[str, list[Any], int]] = []
        self.notes: list[str] = []
        self.closed = 0

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        self.metrics.append((dict(metrics), step, prefix))

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        self.images.append((key, images, step))

    def log_notes(self, notes: str) -> None:
        self.notes.append(notes)

    def close(self) -> None:
        self.closed += 1


def _tracker_list_with_children(n: int) -> tuple[TrackerList, list[_RecordingChild]]:
    """Build a TrackerList over ``n`` recording children, returning both."""
    tracker_list = TrackerList(
        TrackerList.Config(
            trackers={str(i): _RecordingChild.Config() for i in range(n)},
        ),
    )
    children = [cast(_RecordingChild, c) for c in tracker_list.trackers.values()]
    return tracker_list, children


def test_tracker_list_fans_metrics_to_all_children() -> None:
    """log_metrics forwards to every child, prefix included."""
    tracker_list, children = _tracker_list_with_children(2)
    tracker_list.log_metrics({"score": 1.0}, 3, prefix="eval/")
    for child in children:
        assert child.metrics == [({"score": 1.0}, 3, "eval/")]


def test_tracker_list_fans_images_and_close_to_all_children() -> None:
    """log_images and close fan out to every child."""
    tracker_list, children = _tracker_list_with_children(2)
    tracker_list.log_images("samples", ["a.png"], 4)
    tracker_list.close()
    for child in children:
        assert child.images == [("samples", ["a.png"], 4)]
        assert child.closed == 1


def test_tracker_list_fans_notes_to_all_children() -> None:
    """log_notes forwards to every child (notes-less children ignore it)."""
    tracker_list, children = _tracker_list_with_children(2)
    tracker_list.log_notes("Hypothesis: X.")
    for child in children:
        assert child.notes == ["Hypothesis: X."]


def test_default_metrics_tracker_wraps_file_tracker() -> None:
    config = default_metrics_tracker()
    assert isinstance(config, TrackerList.Config)
    metrics = config.trackers["metrics"]
    assert isinstance(metrics, FileTracker.Config)
    assert metrics.working_dir == "/metrics.json"


def test_default_metrics_tracker_custom_path(tmp_path: Path) -> None:
    path = str(tmp_path / "metrics.json")
    config = default_metrics_tracker(working_dir=path)
    metrics = config.trackers["metrics"]
    assert isinstance(metrics, FileTracker.Config)
    assert metrics.working_dir == path


def test_tracker_list_scopes_child_working_directories(tmp_path: Path) -> None:
    config = default_metrics_tracker()
    config.base_dir = tmp_path

    finalized = config.finalize()

    metrics = finalized.trackers["metrics"]
    assert isinstance(metrics, FileTracker.Config)
    assert metrics.working_dir == tmp_path / "metrics.json"


def test_file_tracker_working_dir_is_scoped_by_owner() -> None:
    config = FileTracker.Config()
    config.base_dir = "/scratch/runs/study/run-1"

    assert config.finalize().working_dir == Path(
        "/scratch/runs/study/run-1/metrics.json"
    )


class _BlockingAsyncChild:
    """Recording child used to prove ordered background delivery."""

    class Config(Fig["_BlockingAsyncChild"]):
        pass

    def __init__(self, config: Config) -> None:
        del config
        self.entered = Event()
        self.release = Event()
        self.metrics: list[dict[str, Any]] = []
        self.thread_ids: list[int] = []
        self.closed = False

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        del step, prefix
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("blocking tracker was not released")
        self.metrics.append(dict(metrics))
        self.thread_ids.append(get_ident())

    def log_images(self, key: str, images: list[Any], step: int) -> None:
        del key, images, step

    def log_notes(self, notes: str) -> None:
        del notes

    def close(self) -> None:
        self.closed = True


def test_async_tracker_is_enabled_ordered_and_nonblocking_by_default() -> None:
    tracker = AsyncTracker.Config(tracker=_BlockingAsyncChild.Config()).make()
    child = cast(_BlockingAsyncChild, tracker.tracker)
    caller_thread = get_ident()

    first = {"index": 1}
    tracker.log_metrics(first, 1)
    assert child.entered.wait(timeout=1.0)
    first["index"] = 99

    second_returned = Event()
    second = Thread(
        target=lambda: (
            tracker.log_metrics({"index": 2}, 2),
            second_returned.set(),
        )
    )
    second.start()
    assert second_returned.wait(timeout=1.0)

    child.release.set()
    second.join(timeout=1.0)
    tracker.close()

    assert AsyncTracker.Config().enabled is True
    assert child.metrics == [{"index": 1}, {"index": 2}]
    assert child.thread_ids[0] != caller_thread
    assert child.closed


def test_async_tracker_can_be_disabled() -> None:
    tracker = AsyncTracker.Config(
        tracker=_RecordingChild.Config(),
        enabled=False,
    ).make()
    child = cast(_RecordingChild, tracker.tracker)

    tracker.log_metrics({"loss": 1.0}, 1)
    tracker.flush()
    tracker.close()

    assert child.metrics == [({"loss": 1.0}, 1, "")]
    assert child.closed == 1


def test_tracker_list_scopes_wrapped_child_working_directory(
    tmp_path: Path,
) -> None:
    config = TrackerList.Config(
        base_dir=tmp_path,
        working_dir="run",
        trackers={
            "wandb": AsyncTracker.Config(
                tracker=FileTracker.Config(working_dir="metrics.json"),
            ),
        },
    )

    finalized = config.finalize()
    wrapper = finalized.trackers["wandb"]
    assert isinstance(wrapper, AsyncTracker.Config)
    child = wrapper.tracker
    assert isinstance(child, FileTracker.Config)
    assert child.working_dir == tmp_path / "run/metrics.json"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
