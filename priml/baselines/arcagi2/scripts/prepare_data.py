#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Stage an admitted, already-built ARC2 source-format dataset.

Raw Kaggle ingestion and augmentation remain outside this preparer. Supply a
complete ARC2 tree with its original _build_params.json. The command verifies
the ARC2 subset identity, copies only expected files, checks their hashes, and
publishes into a new destination. Existing directories are never overwritten.
'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import argparse
import hashlib
import shutil
import tempfile

from priml.lib.custom_json import DictCodec, ListCodec, StrCodec, loads
from priml.paths import validated_output_path


def prepare(source: Path, *, destination: Path) -> None:
    """Verify and stage an already-built ARC2 tree without overwriting data.

    Args:
      source: Original source-format tree, including its ARC2 build sentinel.
      destination: New destination directory; it must not already exist.

    """
    destination = validated_output_path(destination, protected=(source,))
    if destination.exists():
        raise FileExistsError(destination)
    names = _files(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="arc2-prepared-",
        dir=destination.parent,
    ) as temporary:
        staging = Path(temporary)
        for name in names:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            expected = _digest(source / name)
            shutil.copyfile(source / name, target)
            if _digest(target) != expected:
                raise ValueError(f"ARC2 source changed while staging {name}")
        _files(staging)
        destination.mkdir()
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / name).rename(target)


def main() -> int:
    """Stage the explicitly supplied prepared source tree.

    Returns:
      exit_code: Zero after successful staging.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 2)[2])
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    prepare(flags.source, destination=flags.destination)
    return 0


def _files(source: Path) -> list[Path]:
    """Admit the source subset identity and complete prepared-file boundary."""
    params = DictCodec.coerce(loads((source / "_build_params.json").read_text()))
    subsets = ListCodec.coerce(params.get("subsets"), str)
    if (
        subsets != ["training2", "evaluation2", "concept"]
        or StrCodec.coerce(params.get("test_set_name")) != "evaluation2"
    ):
        raise ValueError("Expected an ARC2 training2/evaluation2/concept build")
    names = [Path("identifiers.json"), Path("test_puzzles.json")]
    for split in ("train", "test"):
        names.append(Path(split) / "dataset.json")
        names.extend(
            Path(split) / f"all__{name}.npy"
            for name in (
                "inputs",
                "labels",
                "puzzle_indices",
                "group_indices",
                "puzzle_identifiers",
            )
        )
    names.append(Path("_build_params.json"))
    for name in names:
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or linked prepared ARC2 file: {path}")
    return names


def _digest(path: Path) -> bytes:
    """Hash a payload without loading the dataset into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


class _Flags(Protocol):
    source: Path
    destination: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/opt/scratch/datasets/arcagi2/arc2concept-aug-1000"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
