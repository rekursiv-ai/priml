from __future__ import annotations

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

from torch import nn

import pytest
import torch

from priml import hub
from priml.hub import (
    get_cache_dir,
    load_hf_checkpoint,
    load_local_state_dict,
    load_transformers_model,
    resolve_hf_dtype,
)
from priml.lib.codec import ReadError
from priml.lib.userdirs import cache_dir


if TYPE_CHECKING:
    from collections.abc import Generator

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


def test_load_transformers_model_with_class(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test load_transformers_model with string class name."""
    mock_model = MagicMock()
    mock_model.to = MagicMock(return_value=mock_model)
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        patch("pathlib.Path.mkdir"),
        _mock_transformers(mock_auto_model),
        caplog.at_level(logging.DEBUG, logger="priml.hub"),
    ):
        model = load_transformers_model(
            "test/model",
            "AutoModel",
            device="cpu",
        )

        assert model == mock_model
        assert [record.getMessage() for record in caplog.records] == [
            "Attempting to load test/model from cache (offline)",
            "Loaded test/model from cache",
        ]
        assert {record.name for record in caplog.records} == {"priml.hub"}
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


def test_load_transformers_model_cache_miss_falls_back_online(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On cache miss, retries with local_files_only=False (no env toggling)."""
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.side_effect = [
        OSError("cache miss"),
        mock_model,
    ]

    with (
        _mock_transformers(mock_auto_model),
        caplog.at_level(logging.DEBUG, logger="priml.hub"),
    ):
        model = load_transformers_model("test/model", "AutoModel")

    assert [record.getMessage() for record in caplog.records] == [
        "Attempting to load test/model from cache (offline)",
        "Cache miss for test/model, downloading from HuggingFace",
        "Cache miss reason: cache miss",
    ]
    assert model == mock_model
    assert mock_auto_model.from_pretrained.call_count == 2
    assert [call.args for call in mock_auto_model.from_pretrained.call_args_list] == [
        ("test/model",),
        ("test/model",),
    ]
    first_kwargs = mock_auto_model.from_pretrained.call_args_list[0].kwargs
    second_kwargs = mock_auto_model.from_pretrained.call_args_list[1].kwargs
    assert first_kwargs["local_files_only"] is True
    assert second_kwargs["local_files_only"] is False


def test_load_transformers_model_force_redownload_skips_the_cache(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with (
        _mock_transformers(mock_auto_model),
        caplog.at_level(logging.DEBUG, logger="priml.hub"),
    ):
        model = load_transformers_model(
            "test/model",
            "AutoModel",
            force_redownload=True,
        )

    assert [record.getMessage() for record in caplog.records] == [
        "Force redownloading test/model",
    ]
    assert model == mock_model
    mock_auto_model.from_pretrained.assert_called_once_with(
        "test/model",
        revision=None,
        trust_remote_code=False,
        local_files_only=False,
        force_download=True,
    )
    call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
    assert call_kwargs["local_files_only"] is False
    assert call_kwargs["force_download"] is True


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("int8", torch.float32),
    ],
)
def test_resolve_hf_dtype_maps_names_and_defaults_to_float32(
    name: str,
    expected: torch.dtype,
) -> None:
    assert resolve_hf_dtype(name) == expected


def test_load_local_state_dict_merges_pytorch_shards(tmp_path: Path) -> None:
    torch.save({"a": torch.ones(3)}, tmp_path / "pytorch_model-00001-of-00002.bin")
    torch.save({"b": torch.zeros(2)}, tmp_path / "pytorch_model-00002-of-00002.bin")

    state_dict = load_local_state_dict(tmp_path)

    assert set(state_dict) == {"a", "b"}
    torch.testing.assert_close(state_dict["b"], torch.zeros(2))


def test_load_local_state_dict_loads_single_pytorch_file_on_cpu(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.touch()
    expected = {"weight": torch.tensor([2.0, 3.0])}

    with patch("torch.load", return_value=expected) as load:
        result = load_local_state_dict(tmp_path)

    assert result == expected
    load.assert_called_once_with(
        str(checkpoint),
        map_location="cpu",
        weights_only=True,
    )


def test_load_local_state_dict_rejects_a_directory_without_weights(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="No safetensors or pytorch_model"):
        load_local_state_dict(tmp_path)


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


def test_load_hf_checkpoint_does_not_treat_config_file_as_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("not a directory")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "tiny"}))
    hf_model = MagicMock()
    hf_model.config.to_dict.return_value = {"model_type": "remote"}
    hf_model.state_dict.return_value = {}
    load = Mock(return_value=hf_model)
    monkeypatch.setattr(hub, "load_transformers_model", load)

    config, state_dict = load_hf_checkpoint(checkpoint, dtype=None)

    load.assert_called_once_with(
        str(checkpoint),
        "AutoModelForCausalLM",
        dtype=None,
        trust_remote_code=False,
    )
    assert config == {"model_type": "remote"}
    assert state_dict == {}


def test_load_hf_checkpoint_rejects_a_non_object_config(tmp_path: Path) -> None:
    """A ``config.json`` that is not a JSON object is caller input, not a KeyError."""
    (tmp_path / "config.json").write_text(json.dumps([1, 2]))
    torch.save({}, tmp_path / "pytorch_model.bin")

    with pytest.raises(ReadError):
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


def test_load_hf_checkpoint_rejects_a_non_object_remote_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_model = MagicMock()
    hf_model.config.to_dict.return_value = ["not", "an", "object"]
    monkeypatch.setattr(hub, "load_transformers_model", Mock(return_value=hf_model))

    with pytest.raises(ReadError):
        load_hf_checkpoint("org/tiny", dtype=None)


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


def test_load_local_state_dict_merges_safetensors_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "a.safetensors"
    second = tmp_path / "b.safetensors"
    first.touch()
    second.touch()
    seen: list[str] = []

    def load_file(path: str) -> dict[str, torch.Tensor]:
        seen.append(path)
        if path == str(first):
            return {"shared": torch.tensor([2.0]), "first": torch.tensor([3.0])}
        return {"shared": torch.tensor([5.0]), "second": torch.tensor([7.0])}

    monkeypatch.setattr(hub, "load_file", load_file)

    state_dict = load_local_state_dict(tmp_path)

    assert seen == [str(first), str(second)]
    assert state_dict == {
        "shared": torch.tensor([5.0]),
        "first": torch.tensor([3.0]),
        "second": torch.tensor([7.0]),
    }


def test_load_local_state_dict_reads_pytorch_shards_with_safe_loading(
    tmp_path: Path,
) -> None:
    first = tmp_path / "pytorch_model-00001-of-00002.bin"
    second = tmp_path / "pytorch_model-00002-of-00002.bin"
    first.touch()
    second.touch()

    with patch(
        "torch.load",
        side_effect=[{"a": torch.tensor([2.0])}, {"b": torch.tensor([3.0])}],
    ) as load:
        state_dict = load_local_state_dict(tmp_path)

    assert state_dict == {"a": torch.tensor([2.0]), "b": torch.tensor([3.0])}
    assert [call.args[0] for call in load.call_args_list] == [str(first), str(second)]
    assert all(
        call.kwargs == {"map_location": "cpu", "weights_only": True}
        for call in load.call_args_list
    )


def test_load_torch_hub_distributed_rank_zero_broadcasts_success() -> None:
    model = nn.Linear(2, 3)

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list") as broadcast,
        patch("torch.hub.load", return_value=model) as load,
    ):
        result = hub.load_torch_hub_distributed("owner/repo", "entry")

    assert result is model
    load.assert_called_once_with("owner/repo", "entry")
    broadcast.assert_called_once_with([None], src=0)


def test_load_torch_hub_distributed_nonzero_rank_loads_after_broadcast() -> None:
    model = nn.Linear(2, 3)

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
        patch("torch.distributed.broadcast_object_list") as broadcast,
        patch("torch.hub.load", return_value=model) as load,
    ):
        result = hub.load_torch_hub_distributed("owner/repo", "entry")

    assert result is model
    load.assert_called_once_with("owner/repo", "entry")
    broadcast.assert_called_once_with([None], src=0)


@pytest.mark.parametrize(
    ("available", "initialized"),
    [(False, True), (True, False)],
)
def test_torch_hub_distributed_loads_directly_without_process_group(
    available: bool,
    initialized: bool,
) -> None:
    model = nn.Linear(2, 3)
    with (
        patch("torch.distributed.is_available", return_value=available),
        patch("torch.distributed.is_initialized", return_value=initialized),
        patch("torch.hub.load", return_value=model) as load,
        patch("torch.distributed.broadcast_object_list") as broadcast,
    ):
        result = hub.load_torch_hub_distributed("owner/repo", "entry")

    assert result is model
    load.assert_called_once_with("owner/repo", "entry")
    broadcast.assert_not_called()


def test_load_torch_hub_distributed_broadcasts_remote_failure() -> None:
    def broadcast_failure(status: list[object], *, src: int) -> None:
        del src
        status[0] = "download failed"

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
        patch(
            "torch.distributed.broadcast_object_list",
            side_effect=broadcast_failure,
        ) as broadcast,
        patch("torch.hub.load") as load,
        pytest.raises(
            RuntimeError,
            match=r"^rank 0 could not load owner/repo/entry: download failed$",
        ) as error,
    ):
        hub.load_torch_hub_distributed("owner/repo", "entry")

    assert str(error.value) == "rank 0 could not load owner/repo/entry: download failed"
    broadcast.assert_called_once_with(["download failed"], src=0)
    load.assert_not_called()


def test_load_torch_hub_distributed_broadcasts_rank_zero_failure() -> None:
    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.broadcast_object_list") as broadcast,
        patch("torch.hub.load", side_effect=OSError("download failed")),
        pytest.raises(OSError, match="download failed"),
    ):
        hub.load_torch_hub_distributed("owner/repo", "entry")

    broadcast.assert_called_once_with(["download failed"], src=0)


def test_load_transformers_model_forwards_options_on_forced_download() -> None:
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with _mock_transformers(mock_auto_model):
        result = load_transformers_model(
            "org/model",
            "AutoModel",
            device="cpu",
            dtype=torch.float16,
            revision="rev-7",
            trust_remote_code=True,
            force_redownload=True,
            low_cpu_mem_usage=True,
        )

    assert result == mock_model
    mock_auto_model.from_pretrained.assert_called_once_with(
        "org/model",
        revision="rev-7",
        trust_remote_code=True,
        local_files_only=False,
        force_download=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    mock_model.to.assert_called_once_with("cpu")


def test_load_transformers_model_fallback_keeps_caller_options() -> None:
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.side_effect = [OSError("offline miss"), mock_model]

    with _mock_transformers(mock_auto_model):
        result = load_transformers_model(
            "org/model",
            "AutoModel",
            revision="rev-3",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    assert result == mock_model
    assert [call.kwargs for call in mock_auto_model.from_pretrained.call_args_list] == [
        {
            "revision": "rev-3",
            "trust_remote_code": True,
            "local_files_only": True,
            "force_download": False,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        },
        {
            "revision": "rev-3",
            "trust_remote_code": True,
            "local_files_only": False,
            "force_download": False,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        },
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
