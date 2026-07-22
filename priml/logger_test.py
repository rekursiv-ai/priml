from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import ast
import io
import logging
import sys
import time

import pytest

from priml import logger as logger_module
from priml.logger import (
    CustomFormatter,
    Timer,
    bind_logging_to_current_stdout,
    replay_buffered_logs,
    setup_logging,
)


def test_module_and_public_defs_have_docstrings() -> None:
    """Module, Timer, and CustomFormatter must have docstrings (CORE-007)."""
    tree = ast.parse(Path(logger_module.__file__).read_text())
    assert ast.get_docstring(tree) is not None
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    for name in ("Timer", "CustomFormatter"):
        assert ast.get_docstring(classes[name]) is not None, name


class TestTimer:
    def test_timer_basic(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test basic timer functionality."""
        caplog.set_level(logging.INFO)
        with Timer("test operation"):
            time.sleep(0.01)

        assert len(caplog.records) == 1
        assert "test operation" in caplog.text
        assert "seconds" in caplog.text

    def test_timer_with_logger_object(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test timer with a logger object instead of string."""
        logger = logging.getLogger("test_logger")
        caplog.set_level(logging.INFO)

        with Timer("custom operation", logger=logger):
            time.sleep(0.01)

        assert len(caplog.records) == 1
        assert "custom operation" in caplog.text

    def test_timer_with_logger_string(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test timer with logger name as string (line 32)."""
        caplog.set_level(logging.DEBUG)

        with Timer("string logger operation", logger="my_logger", level=logging.DEBUG):
            pass

        assert "string logger operation" in caplog.text

    def test_timer_elapsed_time(self) -> None:
        """Test that elapsed time is recorded correctly."""
        timer = Timer("test")
        with timer:
            time.sleep(0.01)

        assert timer.elapsed > 0
        assert timer.elapsed >= 0.01
        assert timer.stop > timer.start

    def test_timer_exception_propagation(self) -> None:
        """Test that exceptions are re-raised (line 53)."""
        with (
            pytest.raises(ValueError, match="test error"),
            Timer("failing operation"),
        ):
            raise ValueError("test error")

    def test_timer_custom_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test timer with custom log level."""
        caplog.set_level(logging.DEBUG)

        with Timer("debug operation", level=logging.DEBUG):
            pass

        assert "debug operation" in caplog.text
        assert caplog.records[0].levelno == logging.DEBUG


class TestCustomFormatter:
    def test_formatter_initialization_with_config(self) -> None:
        """Test CustomFormatter initialization with Config (lines 87-88)."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        assert formatter.datefmt == config.datefmt
        # Verify the formatter was initialized with the config
        assert hasattr(formatter, "_style")

    def test_formatter_with_distributed_initialized(self) -> None:
        """Test formatter when distributed is initialized (lines 93-96)."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        with (
            patch.object(logger_module.dist, "is_initialized", return_value=True),
            patch.object(logger_module.dist, "get_rank", return_value=2),
            patch.object(logger_module.dist, "get_world_size", return_value=8),
        ):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="Test message",
                args=(),
                exc_info=None,
                func="test_func",
            )

            formatted = formatter.format(record)
            assert "rank 2/8" in formatted

    def test_formatter_without_distributed(self) -> None:
        """Test formatter when distributed is not initialized (lines 97-98)."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        with patch.object(logger_module.dist, "is_initialized", return_value=False):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=42,
                msg="Test message",
                args=(),
                exc_info=None,
                func="test_func",
            )

            formatted = formatter.format(record)
            assert "rank" not in formatted

    def test_formatter_levelname_coloring_debug(self) -> None:
        """Test DEBUG level coloring (lines 100-106)."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        with patch.object(logger_module.dist, "is_initialized", return_value=False):
            record = logging.LogRecord(
                name="test",
                level=logging.DEBUG,
                pathname="test.py",
                lineno=1,
                msg="Debug msg",
                args=(),
                exc_info=None,
                func="func",
            )

            formatted = formatter.format(record)
            assert "DEBUG" in formatted

    def test_formatter_levelname_coloring_info(self) -> None:
        """Test INFO level coloring (lines 100-106)."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        with patch.object(logger_module.dist, "is_initialized", return_value=False):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="test.py",
                lineno=1,
                msg="Info msg",
                args=(),
                exc_info=None,
                func="func",
            )

            formatted = formatter.format(record)
            assert "INFO" in formatted

    def test_formatter_levelname_coloring_warning(self) -> None:
        """Test WARNING level coloring (lines 100-106)."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        with patch.object(logger_module.dist, "is_initialized", return_value=False):
            record = logging.LogRecord(
                name="test",
                level=logging.WARNING,
                pathname="test.py",
                lineno=1,
                msg="Warning msg",
                args=(),
                exc_info=None,
                func="func",
            )

            formatted = formatter.format(record)
            assert "WARNING" in formatted

    def test_formatter_levelname_coloring_error(self) -> None:
        """Test ERROR level coloring (lines 100-106)."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        with patch.object(logger_module.dist, "is_initialized", return_value=False):
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error msg",
                args=(),
                exc_info=None,
                func="func",
            )

            formatted = formatter.format(record)
            assert "ERROR" in formatted

    def test_formatter_levelname_coloring_critical(self) -> None:
        """Test CRITICAL level coloring (lines 100-106)."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        with patch.object(logger_module.dist, "is_initialized", return_value=False):
            record = logging.LogRecord(
                name="test",
                level=logging.CRITICAL,
                pathname="test.py",
                lineno=1,
                msg="Critical msg",
                args=(),
                exc_info=None,
                func="func",
            )

            formatted = formatter.format(record)
            assert "CRITICAL" in formatted

    def test_formatter_format_time(self) -> None:
        """Test custom formatTime method (lines 110-111)."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
            func="func",
        )
        record.created = time.time()

        formatted_time = formatter.formatTime(record)
        assert formatted_time is not None
        # Should contain date components based on datefmt
        assert len(formatted_time) > 0

    def test_formatter_format_time_custom_datefmt(self) -> None:
        """Test formatTime with custom datefmt."""
        config = CustomFormatter.Config()
        formatter = CustomFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
            func="func",
        )
        record.created = time.time()

        formatted_time = formatter.formatTime(record, datefmt="%Y-%m-%d")
        assert formatted_time is not None
        assert "-" in formatted_time


class TestSetupLogging:
    def test_setup_logging_basic(self) -> None:
        """Test basic setup_logging functionality (lines 115-122)."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging()

        assert len(root_logger.handlers) > 0
        assert root_logger.level == logging.INFO
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)
        assert isinstance(root_logger.handlers[0].formatter, CustomFormatter)

    def test_setup_logging_clears_existing_handlers(self) -> None:
        """Test that setup_logging clears existing handlers (line 116-117)."""
        root_logger = logging.getLogger()

        # Add a dummy handler
        dummy_handler = logging.StreamHandler(sys.stdout)
        root_logger.addHandler(dummy_handler)
        initial_count = len(root_logger.handlers)
        assert initial_count > 0

        # Setup should clear and add new handlers
        setup_logging()

        # Stream handler + replay-buffer handler after setup; the dummy is gone.
        assert len(root_logger.handlers) == 2
        assert dummy_handler not in root_logger.handlers
        assert isinstance(root_logger.handlers[0], logging.StreamHandler)

    def test_setup_logging_handler_configuration(self) -> None:
        """Test handler is properly configured (lines 119-121)."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        setup_logging()

        handler = root_logger.handlers[0]
        assert handler.level == logging.INFO
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream == sys.stdout  # pyright: ignore[reportUnknownMemberType]
        assert isinstance(handler.formatter, CustomFormatter)

    def test_bind_logging_to_current_stdout_retargets_loop_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After W&B wraps stdout, Loop logs must use the wrapped stream."""
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        setup_logging()
        wrapped_stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", wrapped_stdout)

        bind_logging_to_current_stdout()
        logging.getLogger("test.rebind").info("line-after-wandb-wrap")

        assert "line-after-wandb-wrap" in wrapped_stdout.getvalue()


class TestReplayBufferedLogs:
    """Replay re-emits pre-init records to the current stdout, then detaches."""

    def test_replay_re_emits_buffered_records(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Records logged before replay are re-emitted to stdout on replay."""
        logging.getLogger().handlers.clear()
        setup_logging()
        logging.getLogger("test.replay").info("buffered-line-xyz")
        # The line is already on stdout once (stream handler); capture+clear it
        # so the assertion isolates the replay emission.
        capsys.readouterr()

        replay_buffered_logs()

        assert "buffered-line-xyz" in capsys.readouterr().out

    def test_replay_detaches_buffer_handler(self) -> None:
        """After replay, the buffer handler is removed (idempotent second call)."""
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging()
        assert len(root.handlers) == 2

        replay_buffered_logs()
        assert len(root.handlers) == 1  # buffer detached, stream remains

        # Second call is a no-op (no buffer left).
        replay_buffered_logs()
        assert len(root.handlers) == 1

    def test_replay_without_setup_is_noop(self) -> None:
        """Replay with no buffer installed does nothing and does not raise."""
        root = logging.getLogger()
        root.handlers.clear()
        replay_buffered_logs()  # no buffer -> no-op


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
