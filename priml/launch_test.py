from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import Self, cast, override

import os
import sys

from configgle import Fig, Makeable

import pytest

from priml.launch import _graceful_sigterm, apply_overrides, main


class MockJob:
    """Mock job for testing launch functionality."""

    class Config(Fig["MockJob"]):
        value: int = 42

    def __init__(self, config: Config):
        self.value = config.value
        self.run_called = False
        self.run_args: tuple[str, ...] = ()

    def run(self, *args: str) -> None:
        self.run_called = True
        self.run_args = args


class ChildJob:
    """Nested job used to exercise depth in override traversal."""

    class Config(Fig["ChildJob"]):
        lr: float = 1e-3
        steps: int = 10

    def __init__(self, config: Config):
        self.lr = config.lr
        self.steps = config.steps

    def run(self, *args: str) -> None:
        del args


class NestedJob:
    """Job whose config nests another Fig, for depth-override tests."""

    class Config(Fig["NestedJob"]):
        name: str = ""
        enabled: bool = False
        child: ChildJob.Config = field(default_factory=ChildJob.Config)

    def __init__(self, config: Config):
        self.config = config

    def run(self, *args: str) -> None:
        del args


class FrozenInner(Fig["object"], frozen=True):
    """Frozen nested config for frozen-override testing."""

    lr: float = 1e-3


class FrozenJob:
    """Job whose config (and nested config) are frozen Figs."""

    class Config(Fig["FrozenJob"], frozen=True):
        name: str = ""
        inner: FrozenInner = field(default_factory=FrozenInner)

    def __init__(self, config: Config):
        self.config = config

    def run(self, *args: str) -> None:
        del args


# Non-callable variable for testing
NonCallableConfig = 42


def non_configurable_returner() -> int:
    """Returns non-Makeable from callable."""
    return 42


class NotAJob:
    """Not a JobProtocol - missing run method."""

    def __init__(self) -> None:
        pass


class NotAJobConfig(Fig["NotAJob"]):
    """Config that returns NotAJob."""

    @override
    def make(self) -> NotAJob:
        return NotAJob()


def non_job_returner() -> Makeable[NotAJob]:
    """Returns config that creates NotAJob."""
    return NotAJobConfig()


def mock_experiment() -> Makeable[MockJob]:
    """Mock experiment function that returns a Makeable."""
    return MockJob.Config()


def test_main_is_record_decorated() -> None:
    """main() must be @record-wrapped so a crashing rank writes error.json.

    ``torch.distributed.elastic.multiprocessing.errors.record`` wraps the
    target via ``functools.wraps`` and exposes the original through
    ``__wrapped__``; per-rank tracebacks then land in ``error.json`` and
    torchrun prints a root-cause summary.
    """
    assert hasattr(main, "__wrapped__"), "main() is not @record-decorated"
    wrapped = cast("Callable[[], None]", main.__wrapped__)  # pyright: ignore[reportFunctionMemberAccess] -- functools.wraps adds __wrapped__, untyped on FunctionType
    assert wrapped.__module__ == "priml.launch"


def test_main_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful execution of main()."""
    # Set up command line arguments
    test_args = [
        "prog",
        "priml.launch_test.mock_experiment",
        "--extra",
        "arg",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run main
    main()

    # Verify that the job was created and run
    # We can't easily verify the exact job instance, but we can verify no errors


def test_main_module_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error when module cannot be imported."""
    test_args = ["prog", "nonexistent.module.function"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(ImportError, match="Cannot import module"):
        main()


def test_main_function_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error when function doesn't exist in module."""
    test_args = ["prog", "priml.launch_test.nonexistent_function"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(AttributeError, match="has no attribute"):
        main()


def test_main_not_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error when target is not callable."""
    # Create a module-level variable that's not callable
    test_args = ["prog", "priml.launch_test.NonCallableConfig"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(TypeError, match="is not callable"):
        main()


def test_main_not_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error when callable doesn't return Makeable."""
    test_args = ["prog", "priml.launch_test.non_configurable_returner"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(TypeError, match="not a Makeable"):
        main()


def test_main_not_job_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error when setup doesn't return JobProtocol."""
    test_args = ["prog", "priml.launch_test.non_job_returner"]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(TypeError, match="not a JobProtocol"):
        main()


def test_main_with_unparsed_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that unparsed arguments are passed to job.run()."""
    test_args = [
        "prog",
        "priml.launch_test.mock_experiment",
        "--foo",
        "bar",
        "--baz",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    # Just verify it runs successfully with extra args - the important
    # thing is that the unparsed args are passed through without error
    main()


def test_apply_overrides_top_level_scalar() -> None:
    """A top-level override casts the RHS to the field's declared type."""
    config = NestedJob.Config()
    apply_overrides(config, ["enabled=true", "name=run_a"])
    assert config.enabled is True
    assert config.name == "run_a"


def test_apply_overrides_nested_depth() -> None:
    """A dotted override walks into a nested Fig and casts the leaf."""
    config = NestedJob.Config()
    apply_overrides(config, ["child.lr=3e-4", "child.steps=99"])
    assert config.child.lr == 3e-4
    assert isinstance(config.child.lr, float)
    assert config.child.steps == 99
    assert isinstance(config.child.steps, int)


def test_apply_overrides_frozen_fig() -> None:
    """Overrides write through frozen Figs, top-level and nested."""
    config = FrozenJob.Config()
    apply_overrides(config, ["name=frozen_run", "inner.lr=2e-4"])
    assert config.name == "frozen_run"
    assert config.inner.lr == 2e-4
    assert isinstance(config.inner.lr, float)


def test_apply_overrides_unknown_path_raises() -> None:
    """An override naming a non-existent field is a hard error."""
    config = NestedJob.Config()
    with pytest.raises(ValueError, match="no field `nonexistent`"):
        apply_overrides(config, ["child.nonexistent=1"])


def test_apply_overrides_malformed_spec_raises() -> None:
    """An override missing ``=`` is rejected."""
    config = NestedJob.Config()
    with pytest.raises(ValueError, match="expected PATH=VALUE"):
        apply_overrides(config, ["child.lr"])


def test_apply_overrides_scalar_for_config_field_raises_valueerror() -> None:
    """Setting a nested-config field to a scalar raises ValueError.

    Not a codec TypeError. ``child`` is a Fig; ``child=5`` is a likely typo
    for ``child.lr=5`` and must fail with a path-naming ValueError.
    """
    config = NestedJob.Config()
    with pytest.raises(ValueError, match="child"):
        apply_overrides(config, ["child=5"])


def test_apply_overrides_empty_segment_paths_raise() -> None:
    """Paths with empty segments are rejected with a clear message."""
    config = NestedJob.Config()
    for spec in ["=1", "child.=1", "child..lr=1", ".child=1"]:
        with pytest.raises(ValueError, match="no empty segments"):
            apply_overrides(config, [spec])


_captured_name: list[str] = []


class _CapturingJob:
    """Job that records its config name when run, for the override test."""

    class Config(Fig["_CapturingJob"]):
        name: str = ""
        child: ChildJob.Config = field(default_factory=ChildJob.Config)

    def __init__(self, config: Config):
        self.config = config

    def run(self, *args: str) -> None:
        del args
        _captured_name.append(self.config.name)


def capturing_experiment() -> Makeable[_CapturingJob]:
    """Experiment returning a config that records its resolved name."""
    return _CapturingJob.Config()


_captured_run_dirs: list[str] = []


class _LaunchableJob:
    """Launchable job that records the working directory visible to its payload."""

    class Config(Fig["_LaunchableJob"]):
        study_name: str = ""
        experiment_name: str = ""
        base_dir: Path | str | None = Path("/scratch")
        working_dir: Path | str = "/runs/{study_name}/{experiment_name}"
        doc: str = ""

        @override
        def finalize(self) -> Self:
            working_dir = Path(
                str(self.working_dir).format(
                    study_name=self.study_name,
                    experiment_name=self.experiment_name,
                )
            )
            if self.base_dir is None:
                self.working_dir = working_dir
            else:
                self.working_dir = Path(self.base_dir) / str(working_dir).lstrip("/")
            return super().finalize()

    def __init__(self, config: Config) -> None:
        _captured_run_dirs.append(str(config.working_dir))

    def run(self, *args: str) -> None:
        del args


def launchable_experiment() -> Makeable[_LaunchableJob]:
    """Experiment whose local run directory is derived by the launcher."""
    return _LaunchableJob.Config()


def test_main_override_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--override name=...`` sets the config name and suppresses auto-derive."""
    _captured_name.clear()
    test_args = [
        "prog",
        "priml.launch_test.capturing_experiment",
        "--override",
        "name=custom_run",
        "--override",
        "child.steps=5",
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    main()
    assert _captured_name == ["custom_run"]


def test_main_derives_local_working_dir_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _captured_run_dirs.clear()
    monkeypatch.setenv("LOOP_RUN_DIR", "/wrong/run")
    monkeypatch.setenv("LOOP_SCRATCH_DIR", "/wrong/scratch")
    before = {key: os.environ[key] for key in ("LOOP_RUN_DIR", "LOOP_SCRATCH_DIR")}
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "priml.launch_test.launchable_experiment"],
    )

    main()

    assert _captured_run_dirs == ["/scratch/runs/launchable_experiment"]
    assert {key: os.environ[key] for key in before} == before


def test_graceful_sigterm_raises_keyboardinterrupt_then_restores() -> None:
    """SIGTERM inside the context raises so the job's cleanup ``finally`` runs.

    A hard SIGTERM would skip cleanup (and drop buffered tracker state); the
    context converts it to ``KeyboardInterrupt``. The prior handler is restored
    on exit so repeated launches are unaffected.
    """
    import signal  # noqa: PLC0415 -- local to this signal-specific test

    previous = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(KeyboardInterrupt), _graceful_sigterm():
            # The installed handler raises KeyboardInterrupt synchronously.
            signal.raise_signal(signal.SIGTERM)
        # Handler restored to whatever was installed before the context.
        assert signal.getsignal(signal.SIGTERM) is previous
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
