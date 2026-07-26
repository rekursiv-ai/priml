from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import (
    MagicMock,
    patch,
)

import logging
import os
import sys

import pytest
import torch

from priml.hub import (
    get_cache_dir,
    load_transformers_model,
)


@contextmanager
def _mock_transformers(mock_auto_model: Any) -> Generator[MagicMock]:
    """Inject a fake transformers module to avoid the ~3s real import."""
    fake = MagicMock()
    fake.AutoModel = mock_auto_model
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = fake
    try:
        yield fake
    finally:
        if saved is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved


def test_get_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_cache_dir returns the XDG per-user models cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", "/xdg-cache")
    with patch("pathlib.Path.mkdir"):
        cache_dir = get_cache_dir()
    assert cache_dir == Path("/xdg-cache/loop/models")


def test_get_cache_dir_creates_directory():
    """Test get_cache_dir creates directory if it doesn't exist."""
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        get_cache_dir()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_load_transformers_model_with_class(tmp_path: Path):
    """Test load_transformers_model with string class name."""
    mock_model = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        patch("priml.hub.get_cache_dir", return_value=tmp_path),
        patch("pathlib.Path.mkdir"),
        _mock_transformers(mock_auto_model),
    ):
        model = load_transformers_model(
            "test/model",
            "AutoModel",
            device="cpu",
        )

        assert model == mock_model
        mock_auto_model.from_pretrained.assert_called_once()
        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert str(tmp_path / "huggingface") in call_kwargs["cache_dir"]
        assert call_kwargs["revision"] is None
        assert call_kwargs["trust_remote_code"] is False


def test_load_transformers_model_with_string_class(tmp_path: Path):
    """Test load_transformers_model with string class name."""
    mock_model = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        patch("priml.hub.get_cache_dir", return_value=tmp_path),
        patch("pathlib.Path.mkdir"),
        _mock_transformers(mock_auto_model),
    ):
        model = load_transformers_model(
            "test/model",
            "AutoModel",
            device="cpu",
        )

        assert model == mock_model


def test_load_transformers_model_with_revision(tmp_path: Path):
    """Test load_transformers_model with specific revision."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        patch("priml.hub.get_cache_dir", return_value=tmp_path),
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model(
            "test/model",
            "AutoModel",
            revision="v1.0",
        )

        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["revision"] == "v1.0"


def test_load_transformers_model_with_trust_remote_code(tmp_path: Path):
    """Test load_transformers_model with trust_remote_code."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        patch("priml.hub.get_cache_dir", return_value=tmp_path),
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model(
            "test/model",
            "AutoModel",
            trust_remote_code=True,
        )

        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["trust_remote_code"] is True


def test_load_transformers_model_with_device(tmp_path: Path):
    """Test load_transformers_model moves to device."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        patch("priml.hub.get_cache_dir", return_value=tmp_path),
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model(
            "test/model",
            "AutoModel",
            device="cpu",
        )

        mock_model.to.assert_called_once_with("cpu")


def test_load_transformers_model_extra_kwargs(tmp_path: Path):
    """Test load_transformers_model passes extra kwargs."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        patch("priml.hub.get_cache_dir", return_value=tmp_path),
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model(
            "test/model",
            "AutoModel",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["torch_dtype"] == torch.float16
        assert call_kwargs["low_cpu_mem_usage"] is True


def test_load_transformers_model_no_global_env_mutation(tmp_path: Path) -> None:
    """load_transformers_model must not touch process-global offline state.

    The HF_HUB_OFFLINE env var and the transformers logger level are
    process-global; concurrent loads would interleave/clobber them. Offline
    behavior is controlled per-call via local_files_only instead.
    """
    sentinel = "w7-sentinel-value"
    transformers_logger = logging.getLogger("transformers")
    original_logger_level = transformers_logger.level

    # Capture process-global state *during* from_pretrained -- the old code
    # toggled it around the call and restored in finally, so the leak is only
    # observable mid-call.
    observed: dict[str, Any] = {}

    def _capture(*_args: Any, **_kwargs: Any) -> MagicMock:
        observed["env"] = os.environ.get("HF_HUB_OFFLINE")
        observed["level"] = transformers_logger.level
        return MagicMock()

    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.side_effect = _capture

    with (
        patch.dict(os.environ, {"HF_HUB_OFFLINE": sentinel}, clear=False),
        patch("priml.hub.get_cache_dir", return_value=tmp_path),
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model("test/model", "AutoModel")

    # Env var and logger level untouched even mid-call.
    assert observed["env"] == sentinel
    assert observed["level"] == original_logger_level
    # Cache-first path still requests offline via local_files_only.
    call_kwargs = mock_auto_model.from_pretrained.call_args[1]
    assert call_kwargs["local_files_only"] is True


def test_load_transformers_model_cache_miss_falls_back_online(tmp_path: Path) -> None:
    """On cache miss, retries with local_files_only=False (no env toggling)."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.side_effect = [
        OSError("cache miss"),
        mock_model,
    ]

    with (
        patch("priml.hub.get_cache_dir", return_value=tmp_path),
        _mock_transformers(mock_auto_model),
    ):
        model = load_transformers_model("test/model", "AutoModel")

    assert model == mock_model
    assert mock_auto_model.from_pretrained.call_count == 2
    first_kwargs = mock_auto_model.from_pretrained.call_args_list[0][1]
    second_kwargs = mock_auto_model.from_pretrained.call_args_list[1][1]
    assert first_kwargs["local_files_only"] is True
    assert second_kwargs["local_files_only"] is False


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
