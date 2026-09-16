from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast
from unittest.mock import (
    MagicMock,
    Mock,
    patch,
)

import json
import logging
import os
import sys

import pytest
import torch

from priml import hub
from priml.hub import (
    get_cache_dir,
    load_hf_checkpoint,
    load_transformers_model,
)
from priml.lib.userdirs import cache_dir


if TYPE_CHECKING:
    from safetensors.torch import save_file
else:
    from wrapt import lazy_import

    save_file = lazy_import("safetensors.torch", "save_file")


@contextmanager
def _mock_transformers(mock_auto_model: object) -> Generator[MagicMock]:
    """Inject a fake transformers module to avoid the ~3s real import."""
    fake = MagicMock()
    fake.AutoModel = mock_auto_model
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = cast(ModuleType, fake)
    try:
        yield fake
    finally:
        if saved is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = saved


def test_get_cache_dir_follows_torch_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """Weights with no library env var of their own share TORCH_HOME."""
    monkeypatch.setenv("TORCH_HOME", "/opt/scratch/models/torch")
    with patch("pathlib.Path.mkdir"):
        models_dir = get_cache_dir()
    assert models_dir == Path("/opt/scratch/models/torch")


def test_get_cache_dir_defers_to_userdirs(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no TORCH_HOME the location is per-user and userdirs owns it.

    Re-deriving the XDG layout here returned ``~/.cache`` on macOS, where
    ``cache_dir()`` resolves ``~/Library/Caches`` -- so hub wrote to a
    directory no other caller read.
    """
    monkeypatch.delenv("TORCH_HOME", raising=False)
    with patch("pathlib.Path.mkdir"):
        models_dir = get_cache_dir()
    assert models_dir == cache_dir() / "rekursiv-ai" / "models"


def test_get_cache_dir_creates_directory():
    """Test get_cache_dir creates directory if it doesn't exist."""
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        get_cache_dir()
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_load_transformers_model_with_class():
    """Test load_transformers_model with string class name."""
    mock_model = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
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
        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        # No cache_dir kwarg: passing one overrides HF_HOME, which is how a
        # provisioned shared cache became inert for every priml processor.
        assert "cache_dir" not in call_kwargs
        assert call_kwargs["revision"] is None
        assert call_kwargs["trust_remote_code"] is False


def test_load_transformers_model_with_revision():
    """Test load_transformers_model with specific revision."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model(
            "test/model",
            "AutoModel",
            revision="v1.0",
        )

        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        assert call_kwargs["revision"] == "v1.0"


def test_load_transformers_model_with_trust_remote_code():
    """Test load_transformers_model with trust_remote_code."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model(
            "test/model",
            "AutoModel",
            trust_remote_code=True,
        )

        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        assert call_kwargs["trust_remote_code"] is True


def test_load_transformers_model_with_device():
    """Test load_transformers_model moves to device."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model(
            "test/model",
            "AutoModel",
            device="cpu",
        )

        mock_model.to.assert_called_once_with("cpu")


def test_load_transformers_model_extra_kwargs():
    """Test load_transformers_model passes extra kwargs."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model(
            "test/model",
            "AutoModel",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
        assert call_kwargs["torch_dtype"] == torch.float16
        assert call_kwargs["low_cpu_mem_usage"] is True


def test_dtype_uses_the_current_transformers_spelling() -> None:
    """``dtype=`` reaches from_pretrained as ``dtype``, not ``torch_dtype``.

    transformers deprecated ``torch_dtype`` and warns on every call that uses
    it; the declared floor already accepts the new name.
    """
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = MagicMock()

    with (
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model("test/model", "AutoModel", dtype=torch.float16)

    call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
    assert call_kwargs["dtype"] == torch.float16
    assert "torch_dtype" not in call_kwargs


def test_load_transformers_model_no_global_env_mutation() -> None:
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
    observed: dict[str, object] = {}

    def _capture(*_args: object, **_kwargs: object) -> MagicMock:
        observed["env"] = os.environ.get("HF_HUB_OFFLINE")
        observed["level"] = transformers_logger.level
        return MagicMock()

    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.side_effect = _capture

    with (
        patch.dict(os.environ, {"HF_HUB_OFFLINE": sentinel}, clear=False),
        _mock_transformers(mock_auto_model),
    ):
        load_transformers_model("test/model", "AutoModel")

    # Env var and logger level untouched even mid-call.
    assert observed["env"] == sentinel
    assert observed["level"] == original_logger_level
    # Cache-first path still requests offline via local_files_only.
    call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
    assert call_kwargs["local_files_only"] is True


def test_load_transformers_model_cache_miss_falls_back_online() -> None:
    """On cache miss, retries with local_files_only=False (no env toggling)."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.side_effect = [
        OSError("cache miss"),
        mock_model,
    ]

    with (
        _mock_transformers(mock_auto_model),
    ):
        model = load_transformers_model("test/model", "AutoModel")

    assert model == mock_model
    assert mock_auto_model.from_pretrained.call_count == 2
    first_kwargs = mock_auto_model.from_pretrained.call_args_list[0].kwargs
    second_kwargs = mock_auto_model.from_pretrained.call_args_list[1].kwargs
    assert first_kwargs["local_files_only"] is True
    assert second_kwargs["local_files_only"] is False


def test_load_hf_checkpoint_reads_a_local_safetensors_directory(
    tmp_path: Path,
) -> None:
    """A directory with ``config.json`` never touches transformers."""
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "tiny", "hidden_size": 4}),
    )
    weights = {"a.weight": torch.arange(4.0), "b.bias": torch.zeros(2)}
    save_file(weights, str(tmp_path / "model.safetensors"))

    with patch("priml.hub.load_transformers_model") as mock_load:
        hf_config, hf_sd = load_hf_checkpoint(tmp_path, dtype=None)

    mock_load.assert_not_called()
    assert hf_config == {"model_type": "tiny", "hidden_size": 4}
    assert hf_sd.keys() == weights.keys()
    for key, value in weights.items():
        torch.testing.assert_close(hf_sd[key], value)


def test_load_hf_checkpoint_reads_a_local_pytorch_bin(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "tiny"}))
    torch.save({"w": torch.ones(3)}, tmp_path / "pytorch_model.bin")

    hf_config, hf_sd = load_hf_checkpoint(str(tmp_path), dtype=torch.float16)

    assert hf_config == {"model_type": "tiny"}
    torch.testing.assert_close(hf_sd["w"], torch.ones(3))


def test_load_hf_checkpoint_rejects_a_non_object_config(tmp_path: Path) -> None:
    """A ``config.json`` that is not a JSON object is caller input, not a KeyError."""
    (tmp_path / "config.json").write_text(json.dumps([1, 2]))
    torch.save({}, tmp_path / "pytorch_model.bin")

    with pytest.raises(TypeError, match="dict"):
        load_hf_checkpoint(tmp_path, dtype=None)


def test_load_hf_checkpoint_downloads_a_repo_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-directory goes through transformers and lands on CPU, detached."""
    hf_model = MagicMock()
    hf_model.config.to_dict.return_value = {"model_type": "tiny"}
    hf_model.state_dict.return_value = {"w": torch.ones(2, requires_grad=True)}
    load = Mock(return_value=hf_model)
    monkeypatch.setattr(hub, "load_transformers_model", load)

    hf_config, hf_sd = load_hf_checkpoint(
        "org/tiny",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    load.assert_called_once_with(
        "org/tiny",
        "AutoModelForCausalLM",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    assert hf_config == {"model_type": "tiny"}
    assert hf_sd["w"].requires_grad is False
    assert hf_sd["w"].device.type == "cpu"


def test_load_hf_checkpoint_defaults_to_no_remote_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_model = MagicMock()
    hf_model.config.to_dict.return_value = {}
    hf_model.state_dict.return_value = {}
    load = Mock(return_value=hf_model)
    monkeypatch.setattr(hub, "load_transformers_model", load)

    load_hf_checkpoint("org/tiny", dtype=None)

    assert load.call_args.kwargs["trust_remote_code"] is False


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
