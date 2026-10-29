"""Checks for the manifest module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib

from priml.baselines.casmi2026.scripts.manifest import (
    input_digests,
    sha256_file,
    source_digests,
    write_manifest,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_file_digest_matches_the_whole_file_digest(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc" * 100_000)
    assert sha256_file(path) == hashlib.sha256(b"abc" * 100_000).hexdigest()


def test_inputs_are_keyed_by_resolved_path_and_sources_by_name(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    first, second = tmp_path / "sub" / "b.txt", tmp_path / "a.txt"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    relative = tmp_path / "sub" / ".." / "a.txt"
    assert list(input_digests([first, relative])) == [
        str(first.resolve()),
        str(second.resolve()),
    ]
    assert (
        input_digests([first])[str(first.resolve())]
        == hashlib.sha256(b"one").hexdigest()
    )
    assert source_digests([first, second]) == {
        "b.txt": hashlib.sha256(b"one").hexdigest(),
        "a.txt": hashlib.sha256(b"two").hexdigest(),
    }


def test_manifest_is_indented_json_with_a_trailing_newline(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"format_version": 1, "inputs": {"x": "y"}})
    assert (tmp_path / "manifest.json").read_text() == (
        '{\n  "format_version": 1,\n  "inputs": {\n    "x": "y"\n  }\n}\n'
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
