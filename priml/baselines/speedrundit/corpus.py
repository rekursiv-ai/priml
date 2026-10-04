"""A latent corpus's receipt, its stored files, and its codec table.

A corpus directory holds one ``.npy`` per image, ``dataset.json`` (labels),
``codec.pt`` when the codec is fitted, and ``corpus.json``: the receipt stating
which autoencoder, checkpoint, latent geometry, codec, and table produced it.
The loader checks the receipt against its own config, so a corpus encoded by
one autoencoder or codec is never silently read as another's.

The check compares what can change a stored byte: the autoencoder's ENCODING
config tree, checkpoints by digest rather than by location, and the codec's
class, stored dtype, and table digest. Decode-only fields (the decoder, the
pixel decoder, and the latent normalizer) are left out, so a corpus survives
re-normalization or a new decoder. Every encoding field counts, defaults
included: a new field changes the identity even when it changes no byte, and
``scripts/prepare_data.py --receipt-only`` re-records such a corpus. The full
``pformat`` is kept in the receipt for a reader, not compared.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import hashlib
import json
import os
import tempfile
import types

from configgle.pprinting import pformat

import numpy as np
import torch

from priml.lib.custom_json import DictCodec, loads
from priml.model.vision_ae.custom_types import CheckpointFile


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import BinaryIO

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

FORMAT_VERSION: Final = 2
"""Bumped when the receipt's schema changes meaning."""

PREPARING: Final = "preparing"
"""The provenance of a receipt pinned before preparation finished."""

# Decode-only modules and diffusion normalization never produce raw latent bytes.
_DECODE_FIELDS: Final = frozenset({"decoder", "pixel_decoder", "latent_norm"})


class CorpusMismatchError(ValueError):
    """A corpus on disk does not match what the experiment or preparer declares."""


def write_atomically(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """Write a file through a private sibling renamed into place.

    An interrupted write leaves the old file or none, never a partial one, and
    two processes writing one path never share a staging file.

    Args:
      path: Destination; its directory must exist.
      write: Writes the content to the open binary stream it is given.

    """
    descriptor, staging = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".partial",
    )
    staged = Path(staging)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            _ = write(stream)
        _ = staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def save_stored(path: Path, stored: Tensor) -> None:
    """Write one stored latent as ``.npy``.

    bfloat16 has no NumPy dtype, so it is written as its int16 bit pattern; the
    receipt's stored dtype says how to read it back.

    Args:
      path: Destination.
      stored: Tensor in a codec's stored dtype.

    Raises:
      ValueError: A floating ``stored`` holds a non-finite value.

    """
    tensor = stored.detach().cpu().contiguous()
    if (
        tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
    ):  # house-ignore[tensor-value-guard] -- Host tensor checked once before publication; no device sync.
        raise ValueError(
            f"{path}: stored latents must be finite; the codec overflowed.",
        )
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    array = tensor.numpy()
    write_atomically(path, write=lambda stream: np.save(stream, array))


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
      identity: Class, latent shape, and the encoding config tree.

    Raises:
      ValueError: An encoding field holds a lambda or a closure, which print
        alike whatever they compute.

    """
    return {
        "class": _qualified(config),
        "latent_shape": list(config.latent_shape()),
        "encoding": _encoding_identity(config),
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
    table = {name: value.detach().cpu() for name, value in codec.table().items()}
    write_atomically(path, write=lambda stream: torch.save(table, stream))
    return _sha256(path)


def load_table(directory: Path, codec: FittedCodec) -> str:
    """Load a fitted codec's table and return its digest.

    Args:
      directory: Latent directory.
      codec: The codec to restore; it validates the table.

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
        batch size, error statistics. A ``provenance`` of :data:`PREPARING`
        marks a corpus the loader must refuse.

    Returns:
      path: The receipt.

    """
    receipt = {
        "format": FORMAT_VERSION,
        "identity": {
            "autoencoder": autoencoder_identity(autoencoder),
            "codec": codec_identity(
                codec_config,
                codec=codec,
                table_sha256=table_sha256,
            ),
        },
        "config": {
            "autoencoder": pformat(autoencoder, hide_default_values=False),
            "codec": pformat(codec_config, hide_default_values=False),
        },
        "suggested_latent_norm": pformat(
            autoencoder.latent_norm,
            hide_default_values=False,
        ),
        "details": dict(details),
    }
    text = json.dumps(receipt, indent=2, sort_keys=True, default=str)
    path = directory / RECEIPT
    write_atomically(path, write=lambda stream: stream.write(text.encode()))
    return path


def verify_receipt(
    directory: Path,
    *,
    autoencoder: VisionAutoencoderConfig,
    codec_config: Makeable[LatentCodec],
    codec: LatentCodec,
    table_sha256: str | None,
) -> dict[str, object]:
    """Raise unless the corpus receipt matches the configured producers.

    Args:
      directory: Latent directory.
      autoencoder: The autoencoder the experiment declares.
      codec_config: The codec the experiment declares.
      codec: The built codec (table loaded, if fitted).
      table_sha256: Digest of the loaded table, for a fitted codec.

    Returns:
      details: The receipt's provenance, as :func:`write_receipt` recorded it.

    Raises:
      FileNotFoundError: The corpus has no receipt.
      CorpusMismatchError: The receipt has another format, or a recorded
        identity differs, naming every field.

    """
    path = directory / RECEIPT
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. Encode the corpus with scripts/prepare_data.py, or "
            "record an existing REG corpus with its --receipt-only mode.",
        )
    receipt = DictCodec.coerce(loads(path.read_text()), default=None)
    if receipt.get("format") != FORMAT_VERSION:
        raise CorpusMismatchError(
            f"{path} has receipt format {receipt.get('format')!r}, not "
            f"{FORMAT_VERSION}. Re-encode the corpus with scripts/prepare_data.py, or "
            "delete the receipt and re-record the corpus with --receipt-only.",
        )
    expected = _flatten(
        {
            "autoencoder": autoencoder_identity(autoencoder),
            "codec": codec_identity(
                codec_config,
                codec=codec,
                table_sha256=table_sha256,
            ),
        },
    )
    problems = mismatches(_flatten(receipt["identity"]), expected=expected)
    if problems:
        raise CorpusMismatchError(
            f"{directory} was not produced by the configured autoencoder and codec:\n  "
            + "\n  ".join(problems),
        )
    return DictCodec.coerce(receipt["details"], default=None)


def mismatches(
    recorded: Mapping[str, object],
    expected: Mapping[str, object],
) -> list[str]:
    """Name every field whose recorded and expected values differ.

    Args:
      recorded: Values read from disk.
      expected: Values the configuration declares.

    Returns:
      problems: ``"key: corpus <recorded>, config <expected>"`` per difference,
        a field one side lacks shown as ``<absent>``.

    """
    return [
        f"{key}: corpus {recorded.get(key, '<absent>')!r}, "
        f"config {expected.get(key, '<absent>')!r}"
        for key in sorted(recorded.keys() | expected.keys())
        if key not in recorded
        or key not in expected
        or _jsonable(recorded[key]) != _jsonable(expected[key])
    ]


def _qualified(config: object) -> str:
    """Return the dotted name of the class a config makes."""
    made = getattr(type(config), "parent_class", None)
    cls = made if isinstance(made, type) else type(config)
    return f"{cls.__module__}.{cls.__qualname__}"


def _encoding_identity(config: object) -> object:
    """Describe encoding inputs, naming checkpoints by digest rather than location."""
    made = getattr(type(config), "parent_class", None)
    if isinstance(made, type) and issubclass(made, CheckpointFile):
        return cast("Makeable[CheckpointFile]", config).make().identity()
    if is_dataclass(config) and not isinstance(config, type):
        return {
            "class": _qualified(config),
            **{
                entry.name: _encoding_identity(
                    cast("object", getattr(config, entry.name)),
                )
                for entry in fields(config)
                if entry.name not in _DECODE_FIELDS
            },
        }
    if isinstance(config, list | tuple):
        return [_encoding_identity(item) for item in cast("list[object]", config)]
    # ``pformat`` masks a function's address, so two lambdas, or two closures
    # over different captured state, would share one identity.
    if isinstance(config, types.FunctionType) and (
        "<lambda>" in config.__qualname__ or "<locals>" in config.__qualname__
    ):
        raise ValueError(
            f"{config.__qualname__} cannot identify a corpus; use a module-level "
            "function so the receipt names what it computes.",
        )
    return pformat(config, hide_default_values=False)


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    """Key each leaf of nested mappings by dotted path, so a diff names the leaf."""
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, object] = {}
    for key, child in cast("dict[str, object]", value).items():
        flat |= _flatten(child, prefix=f"{prefix}.{key}" if prefix else key)
    return flat


def _jsonable(value: object) -> object:
    """Return ``value`` as a JSON round trip would, so tuples compare as lists."""
    return cast(object, json.loads(json.dumps(value, default=str)))


def _sha256(path: Path) -> str:
    """Return a file's hex SHA-256."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
