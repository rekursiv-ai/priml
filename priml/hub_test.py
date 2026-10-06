from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType, ModuleType
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
    load_local_state_dict,
    load_transformers_model,
    resolve_hf_dtype,
    save_hf_checkpoint,
)
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


def test_load_transformers_model_force_redownload_skips_the_cache() -> None:
    mock_model = MagicMock()
    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.return_value = mock_model

    with _mock_transformers(mock_auto_model):
        model = load_transformers_model(
            "test/model",
            "AutoModel",
            force_redownload=True,
        )

    assert model == mock_model
    mock_auto_model.from_pretrained.assert_called_once()
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


def test_save_hf_checkpoint_round_trips_locally(tmp_path: Path) -> None:
    """Load back the exact config and tensors written by a local export."""
    config = {"model_type": "toy", "hidden_size": 4}
    state = {
        "model.embed_tokens.weight": torch.arange(12.0).reshape(3, 4),
        "lm_head.weight": torch.arange(12.0).reshape(4, 3),
        # Non-contiguous on arrival: safetensors refuses those, so the save
        # must make them contiguous without touching the values.
        "model.norm.weight": torch.arange(6.0).reshape(2, 3).t(),
    }
    directory = save_hf_checkpoint(tmp_path / "nested" / "out", config, state)

    hf_config, hf_sd = load_hf_checkpoint(directory, dtype=None)

    assert hf_config == config
    for name, value in state.items():
        torch.testing.assert_close(hf_sd[name], value, rtol=0, atol=0)


def test_save_hf_checkpoint_accepts_a_non_dict_mapping(tmp_path: Path) -> None:
    """Accept every mapping type promised by the public signature."""
    config = MappingProxyType({"model_type": "toy"})
    directory = save_hf_checkpoint(
        tmp_path / "out",
        config,
        {"weight": torch.ones(1)},
    )

    saved = cast(dict[str, object], json.loads((directory / "config.json").read_text()))
    assert saved == dict(config)


def test_save_hf_checkpoint_copies_only_explicit_auxiliary_files(
    tmp_path: Path,
) -> None:
    """Copy only the auxiliary files selected by the caller."""
    source = tmp_path / "source"
    source.mkdir()
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "old.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
    ):
        (source / name).write_bytes(b"stale")

    out = save_hf_checkpoint(
        tmp_path / "out",
        {"model_type": "toy"},
        {"w": torch.zeros(2, 3)},
        auxiliary_files=[
            source / "tokenizer.json",
            source / "tokenizer_config.json",
            source / "special_tokens_map.json",
            source / "generation_config.json",
        ],
    )

    assert {entry.name for entry in out.iterdir()} == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    }


@pytest.mark.parametrize(
    "name",
    ["config.json", "model.safetensors", "weights.pt", "model.safetensors.index.json"],
)
def test_save_hf_checkpoint_rejects_conflicting_auxiliary_files(
    tmp_path: Path,
    name: str,
) -> None:
    """Reject auxiliary files that could replace owned checkpoint files."""
    source = tmp_path / name
    source.write_bytes(b"stale")
    with pytest.raises(ValueError, match="Auxiliary file"):
        save_hf_checkpoint(
            tmp_path / "out",
            {},
            {"w": torch.ones(1)},
            auxiliary_files=[source],
        )


def test_save_hf_checkpoint_refuses_foreign_shards(tmp_path: Path) -> None:
    """A stale shard would merge into every later load of the directory."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "model-00001-of-00002.safetensors").write_bytes(b"stale")

    with pytest.raises(ValueError, match="foreign"):
        save_hf_checkpoint(out, {"model_type": "toy"}, {"w": torch.zeros(2, 3)})


@pytest.mark.parametrize(
    "name",
    [
        "pytorch_model.bin",
        "weights.pt",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ],
)
def test_save_hf_checkpoint_refuses_all_conflicting_weight_artifacts(
    tmp_path: Path,
    name: str,
) -> None:
    """Refuse every recognized foreign weight artifact beside an export."""
    out = tmp_path / "out"
    out.mkdir()
    (out / name).write_bytes(b"stale")
    with pytest.raises(ValueError, match="weight artifacts"):
        save_hf_checkpoint(out, {}, {"w": torch.ones(1)})


def test_save_hf_checkpoint_failure_never_publishes_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave no destination or staging directory after an initial save fails."""
    out = tmp_path / "out"

    def fail(*_args: object, **_kwargs: object) -> None:
        """Simulate a safetensors write failure."""
        raise RuntimeError("write failed")

    monkeypatch.setattr(hub, "save_file", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        save_hf_checkpoint(out, {}, {"w": torch.ones(1)})

    assert not out.exists()
    assert not list(tmp_path.glob(".out.staging-*"))


def test_save_hf_checkpoint_atomically_replaces_owned_weights(tmp_path: Path) -> None:
    """Replace only the weights when updating an owned export."""
    out = save_hf_checkpoint(
        tmp_path / "out", {"model_type": "toy"}, {"w": torch.ones(1)}
    )
    save_hf_checkpoint(out, {"model_type": "toy"}, {"w": torch.zeros(1)})
    _, state = load_hf_checkpoint(out, dtype=None)
    torch.testing.assert_close(state["w"], torch.zeros(1))
    assert {entry.name for entry in out.iterdir()} == {
        "config.json",
        "model.safetensors",
    }


def test_save_hf_checkpoint_rejects_changed_immutable_metadata(
    tmp_path: Path,
) -> None:
    """Reject a re-export whose immutable metadata changed."""
    out = save_hf_checkpoint(
        tmp_path / "out", {"model_type": "toy"}, {"w": torch.ones(1)}
    )
    before = (out / "model.safetensors").read_bytes()
    with pytest.raises(ValueError, match="metadata does not match"):
        save_hf_checkpoint(out, {"model_type": "other"}, {"w": torch.zeros(1)})
    assert (out / "model.safetensors").read_bytes() == before


def test_failed_owned_overwrite_preserves_old_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the published weights when their replacement fails."""
    out = save_hf_checkpoint(
        tmp_path / "out", {"model_type": "toy"}, {"w": torch.ones(1)}
    )
    before = (out / "model.safetensors").read_bytes()

    def fail(_state: object, filename: str) -> None:
        """Write a partial temporary file and then fail."""
        Path(filename).write_bytes(b"partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(hub, "save_file", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        save_hf_checkpoint(out, {"model_type": "toy"}, {"w": torch.zeros(1)})

    assert (out / "model.safetensors").read_bytes() == before
    assert not list(tmp_path.glob(".out.weights-*"))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
