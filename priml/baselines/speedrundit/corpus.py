"""A latent corpus's receipt, its stored files, and its codec table.

A corpus directory holds one ``.npy`` per image, ``dataset.json`` (labels),
``codec.pt`` when the codec is fitted, and ``corpus.json``: the receipt stating
which autoencoder, checkpoint, latent geometry, codec, and table produced it.
The loader checks the receipt against its own config, so a corpus encoded by
one autoencoder or codec is never silently read as another's.

The check compares IDENTITIES -- class, checkpoint digests, latent shape,
stored dtype, table digest -- not the full printed config, which also records
defaults that can change without changing a single stored byte. The full
``pformat`` is kept in the receipt for a reader, not compared.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Final, cast

import hashlib
import json

from configgle.pprinting import pformat

import numpy as np
import torch

from priml.lib.custom_json import DictCodec, loads
from priml.model.vision_ae.custom_types import CheckpointFile


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from configgle import Makeable
    from numpy.typing import NDArray
    from torch import Tensor

    from priml.baselines.speedrundit.latent_codec import FittedCodec, LatentCodec
    from priml.model.vision_ae.custom_types import VisionAutoencoderConfig


RECEIPT: Final = "corpus.json"
"""The receipt's filename inside a latent directory."""

CODEC_TABLE: Final = "codec.pt"
"""A fitted codec's table, beside the receipt."""

LABELS: Final = "dataset.json"
"""The REG label manifest, beside the receipt."""

FORMAT_VERSION: Final = 1
"""Bumped when the receipt's schema changes meaning."""


class CorpusMismatchError(ValueError):
    """A corpus was produced by a different autoencoder or codec than configured."""


def save_stored(path: Path, stored: Tensor) -> None:
    """Write one stored latent as ``.npy``.

    bfloat16 has no NumPy dtype, so it is written as its int16 bit pattern; the
    receipt's stored dtype says how to read it back.

    Args:
      path: Destination.
      stored: Tensor in a codec's stored dtype.

    """
    tensor = stored.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    np.save(path, tensor.numpy())


def load_stored(path: Path, dtype: torch.dtype) -> Tensor:
    """Read one stored latent written by :func:`save_stored`.

    Args:
      path: ``.npy`` file.
      dtype: The codec's stored dtype.

    Returns:
      stored: Tensor in ``dtype``.

    Raises:
      ValueError: The file's dtype is not ``dtype``'s storage dtype.

    """
    array = cast("NDArray[np.generic]", np.load(path))
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if dtype == torch.bfloat16 and tensor.dtype == torch.int16:
        return tensor.view(torch.bfloat16)
    if tensor.dtype != dtype:
        raise ValueError(f"{path} holds {tensor.dtype}; the codec stores {dtype}.")
    return tensor


def autoencoder_identity(config: VisionAutoencoderConfig) -> dict[str, object]:
    """Return what distinguishes one autoencoder's corpus from another's.

    Args:
      config: The experiment's autoencoder config.

    Returns:
      identity: Class, latent shape, and every checkpoint file's identity.

    """
    return {
        "class": _qualified(config),
        "latent_shape": list(config.latent_shape()),
        "checkpoints": [file.identity() for file in _checkpoint_files(config)],
    }


def codec_identity(
    config: Makeable[LatentCodec],
    codec: LatentCodec,
    table_sha256: str | None,
) -> dict[str, object]:
    """Return what distinguishes one codec's stored bytes from another's.

    Args:
      config: The codec's config.
      codec: The built codec.
      table_sha256: Digest of the table file, for a fitted codec.

    Returns:
      identity: Class, stored dtype, and table digest.

    """
    return {
        "class": _qualified(config),
        "stored_dtype": str(codec.stored_dtype),
        "table_sha256": table_sha256,
    }


def table_path(directory: Path) -> Path:
    """Return where a fitted codec's table lives.

    Args:
      directory: Latent directory.

    Returns:
      path: ``directory / codec.pt``.

    """
    return directory / CODEC_TABLE


def save_table(directory: Path, codec: FittedCodec) -> str:
    """Write a fitted codec's table atomically and return its digest.

    Args:
      directory: Latent directory.
      codec: A fitted codec.

    Returns:
      sha256: Hex digest of the written file.

    """
    path = table_path(directory)
    staging = path.with_suffix(".pt.partial")
    torch.save(codec.table(), staging)
    staging.replace(path)
    return _sha256(path)


def load_table(directory: Path, codec: FittedCodec) -> str:
    """Load a fitted codec's table and return its digest.

    Args:
      directory: Latent directory.
      codec: The codec to restore.

    Returns:
      sha256: Hex digest of the table file.

    Raises:
      FileNotFoundError: The corpus has no table.

    """
    path = table_path(directory)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: a fitted codec needs its table beside the corpus.",
        )
    table = cast(
        "dict[str, Tensor]",
        torch.load(path, map_location="cpu", weights_only=True),
    )
    codec.load_table(table)
    return _sha256(path)


def write_receipt(
    directory: Path,
    *,
    autoencoder: VisionAutoencoderConfig,
    codec_config: Makeable[LatentCodec],
    codec: LatentCodec,
    table_sha256: str | None,
    details: Mapping[str, object],
) -> Path:
    """Write the corpus receipt atomically.

    Args:
      directory: Latent directory.
      autoencoder: The autoencoder config that encoded the corpus.
      codec_config: The codec config that stored it.
      codec: The built (and, if fitted, fitted) codec.
      table_sha256: Digest of the table, for a fitted codec.
      details: Provenance for a reader: image source, counts, seed, device,
        batch size, error statistics.

    Returns:
      path: The receipt.

    """
    receipt = {
        "format": FORMAT_VERSION,
        "autoencoder": {
            **autoencoder_identity(autoencoder),
            "config": pformat(autoencoder, hide_default_values=False),
        },
        "codec": {
            **codec_identity(codec_config, codec, table_sha256),
            "config": pformat(codec_config, hide_default_values=False),
        },
        "suggested_latent_norm": pformat(
            autoencoder.latent_norm,
            hide_default_values=False,
        ),
        "details": dict(details),
    }
    path = directory / RECEIPT
    staging = path.with_suffix(".json.partial")
    _ = staging.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    staging.replace(path)
    return path


def verify_receipt(
    directory: Path,
    *,
    autoencoder: VisionAutoencoderConfig,
    codec_config: Makeable[LatentCodec],
    codec: LatentCodec,
    table_sha256: str | None,
) -> None:
    """Raise unless the corpus receipt matches the configured producers.

    Args:
      directory: Latent directory.
      autoencoder: The autoencoder the experiment declares.
      codec_config: The codec the experiment declares.
      codec: The built codec (table loaded, if fitted).
      table_sha256: Digest of the loaded table, for a fitted codec.

    Raises:
      FileNotFoundError: The corpus has no receipt.
      CorpusMismatchError: A recorded identity differs, naming every field.

    """
    path = directory / RECEIPT
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Encode the corpus with scripts/prepare_data.py, or "
            "record an existing REG corpus with its --receipt-only mode.",
        )
    receipt = DictCodec.coerce(loads(path.read_text()), default=None)
    recorded_autoencoder = DictCodec.coerce(receipt["autoencoder"], default=None)
    recorded_codec = DictCodec.coerce(receipt["codec"], default=None)
    expected = {
        **{f"autoencoder.{k}": v for k, v in autoencoder_identity(autoencoder).items()},
        **{
            f"codec.{k}": v
            for k, v in codec_identity(codec_config, codec, table_sha256).items()
        },
    }
    recorded = {
        **{f"autoencoder.{k}": v for k, v in recorded_autoencoder.items()},
        **{f"codec.{k}": v for k, v in recorded_codec.items()},
    }
    problems = [
        f"{key}: corpus {recorded.get(key)!r}, config {value!r}"
        for key, value in expected.items()
        if _jsonable(recorded.get(key)) != _jsonable(value)
    ]
    if problems:
        raise CorpusMismatchError(
            f"{directory} was not produced by the configured autoencoder and codec:\n  "
            + "\n  ".join(problems),
        )


def _qualified(config: object) -> str:
    """Return the dotted name of the class a config makes."""
    made = cast("type", getattr(type(config), "parent_class", type(config)))  # pyright: ignore[reportAny] -- configgle stores the made class on the Config type.
    return f"{made.__module__}.{made.__qualname__}"


# A checkpoint is found by what it makes, not by name, so an autoencoder wrapping
# several (RAE's encoder, decoder, and statistics) reports all of them.
def _checkpoint_files(config: object) -> list[CheckpointFile]:
    """Return every checkpoint file reachable in a config tree, in field order."""
    found: list[CheckpointFile] = []
    made = getattr(type(config), "parent_class", None)
    if isinstance(made, type) and issubclass(made, CheckpointFile):
        built = cast("Makeable[CheckpointFile]", config).make()
        return [built]
    if is_dataclass(config) and not isinstance(config, type):
        for entry in fields(config):
            found += _checkpoint_files(getattr(config, entry.name))  # pyright: ignore[reportAny] -- Dataclass fields are read by name.
    elif isinstance(config, list | tuple):
        for item in cast("list[object]", config):
            found += _checkpoint_files(item)
    return found


def _jsonable(value: object) -> object:
    """Return ``value`` as a JSON round trip would, so tuples compare as lists."""
    return cast(object, json.loads(json.dumps(value, default=str)))


def _sha256(path: Path) -> str:
    """Return a file's hex SHA-256."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
