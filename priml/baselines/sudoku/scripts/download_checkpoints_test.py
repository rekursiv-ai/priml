"""Tests for scripts.download_checkpoints (Hub calls monkeypatched; no network)."""

from __future__ import annotations

from pathlib import Path

import hashlib
import logging

import pytest

from priml.baselines.sudoku.scripts import download_checkpoints
from priml.baselines.sudoku.scripts.download_checkpoints import (
    available_steps,
    destination_path,
    main,
)


_FILES = [
    "README.md",
    "generator_s44/step_10000/model.pt",
    "generator_s44/step_19500/full_state.pt",
    "generator_s44/step_19500/model.pt",
    "verifier_s0/step_4000/full_state.pt",
    "verifier_s0/step_4000/model.pt",
]


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Replace the Hub calls with a local fake; returns the download log."""
    downloads: list[str] = []

    def fake_list_repo_files(repo_id: str) -> list[str]:
        del repo_id
        return list(_FILES)

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        del repo_id
        downloads.append(filename)
        cached = tmp_path / "hub-cache" / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(f"blob:{filename}".encode())
        return str(cached)

    monkeypatch.setattr(download_checkpoints, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(download_checkpoints, "hf_hub_download", fake_hf_hub_download)
    return downloads


def _blob_digest(filename: str) -> str:
    """SHA-256 of the fake hub's deterministic file content."""
    return hashlib.sha256(f"blob:{filename}".encode()).hexdigest()


@pytest.fixture
def fake_hub_with_sums(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, str]:
    """Fake hub publishing SHA256SUMS; returns the mutable digest map."""
    sums: dict[str, str] = {f: _blob_digest(f) for f in _FILES if f.endswith(".pt")}

    def fake_list_repo_files(repo_id: str) -> list[str]:
        del repo_id
        return [*_FILES, "SHA256SUMS"]

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        del repo_id
        cached = tmp_path / "hub-cache" / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        if filename == "SHA256SUMS":
            cached.write_text(
                "".join(f"{digest}  {path}\n" for path, digest in sums.items()),
            )
        else:
            cached.write_bytes(f"blob:{filename}".encode())
        return str(cached)

    monkeypatch.setattr(download_checkpoints, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(download_checkpoints, "hf_hub_download", fake_hf_hub_download)
    return sums


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------


def test_available_steps_orders_and_filters() -> None:
    assert available_steps(_FILES, name="generator_s44", filename="model.pt") == [
        10_000,
        19_500,
    ]
    assert available_steps(_FILES, name="generator_s44", filename="full_state.pt") == [
        19_500,
    ]
    assert available_steps(_FILES, name="absent", filename="model.pt") == []


def test_destination_path_run_dir_convention(tmp_path: Path) -> None:
    assert destination_path(tmp_path, name="verifier_s1", step=4_000) == (
        tmp_path / "runs" / "verifier_s1" / "checkpoints" / "step_00004000.pt"
    )


# ---------------------------------------------------------------------------
# CLI flows.
# ---------------------------------------------------------------------------


def test_download_places_files_at_eval_defaults(
    fake_hub: list[str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dest = tmp_path / "scratch"
    argv = ["--names", "generator_s44", "verifier_s0", "--dest", str(dest)]
    with caplog.at_level(logging.INFO):
        assert main(argv) == 0
    # Latest step per name, model-only flavor, at the eval-config defaults.
    generator = dest / "runs" / "generator_s44" / "checkpoints" / "step_00019500.pt"
    verifier = dest / "runs" / "verifier_s0" / "checkpoints" / "step_00004000.pt"
    assert generator.read_bytes() == b"blob:generator_s44/step_19500/model.pt"
    assert verifier.read_bytes() == b"blob:verifier_s0/step_4000/model.pt"
    assert fake_hub == [
        "generator_s44/step_19500/model.pt",
        "verifier_s0/step_4000/model.pt",
    ]
    assert f"generator_s44 step 19500: {generator} [downloaded" in caplog.text
    assert f"verifier_s0 step 4000: {verifier} [downloaded" in caplog.text


@pytest.mark.usefixtures("fake_hub")
def test_download_renames_hub_checkpoint_to_run_name(tmp_path: Path) -> None:
    dest = tmp_path / "scratch"
    assert main(["--names", "generator_s44:exp010", "--dest", str(dest)]) == 0
    placed = dest / "runs" / "exp010" / "checkpoints" / "step_00019500.pt"
    assert placed.read_bytes() == b"blob:generator_s44/step_19500/model.pt"
    assert not (dest / "runs" / "generator_s44").exists()


def test_download_default_dest_is_opt_scratch() -> None:
    assert destination_path(Path("/opt/scratch"), name="verifier_s0", step=4_000) == (
        Path("/opt/scratch/runs/verifier_s0/checkpoints/step_00004000.pt")
    )


def test_download_is_idempotent(
    fake_hub: list[str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dest = tmp_path / "scratch"
    argv = ["--names", "generator_s44", "--dest", str(dest)]
    assert main(argv) == 0
    with caplog.at_level(logging.INFO):
        assert main(argv) == 0  # Rerun: skip-if-exists, no second fetch.
    assert fake_hub == ["generator_s44/step_19500/model.pt"]
    assert "[skipped, exists]" in caplog.text


def test_download_replaces_existing_checkpoint_when_content_changes(
    fake_hub: list[str],
    tmp_path: Path,
) -> None:
    dest = tmp_path / "scratch"
    argv = ["--names", "generator_s44", "--dest", str(dest)]
    assert main(argv) == 0
    target = dest / "runs/generator_s44/checkpoints/step_00019500.pt"
    target.write_bytes(b"corrupt but source sidecar still matches")

    assert main(argv) == 0

    assert target.read_bytes() == b"blob:generator_s44/step_19500/model.pt"
    assert fake_hub == [
        "generator_s44/step_19500/model.pt",
        "generator_s44/step_19500/model.pt",
    ]


def test_download_replaces_destination_symlink_without_following_it(
    fake_hub: list[str],
    tmp_path: Path,
) -> None:
    dest = tmp_path / "scratch"
    target = dest / "runs/generator_s44/checkpoints/step_00019500.pt"
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"must not be overwritten")
    target.symlink_to(outside)

    assert main(["--names", "generator_s44", "--dest", str(dest)]) == 0

    assert not target.is_symlink()
    assert target.read_bytes() == b"blob:generator_s44/step_19500/model.pt"
    assert outside.read_bytes() == b"must not be overwritten"
    assert fake_hub == ["generator_s44/step_19500/model.pt"]


@pytest.mark.usefixtures("fake_hub")
def test_download_step_selection_and_full_flavor(tmp_path: Path) -> None:
    dest = tmp_path / "scratch"
    main(["--names", "generator_s44", "--dest", str(dest), "--step", "10000"])
    picked = dest / "runs" / "generator_s44" / "checkpoints" / "step_00010000.pt"
    assert picked.read_bytes() == b"blob:generator_s44/step_10000/model.pt"

    main(["--names", "verifier_s0", "--dest", str(dest), "--flavor", "full"])
    full = dest / "runs" / "verifier_s0" / "checkpoints" / "step_00004000.pt"
    assert full.read_bytes() == b"blob:verifier_s0/step_4000/full_state.pt"


def test_download_replaces_existing_checkpoint_when_flavor_changes(
    fake_hub: list[str],
    tmp_path: Path,
) -> None:
    dest = tmp_path / "scratch"
    base = ["--names", "verifier_s0", "--dest", str(dest)]
    main(base)
    target = dest / "runs" / "verifier_s0" / "checkpoints" / "step_00004000.pt"
    assert target.read_bytes() == b"blob:verifier_s0/step_4000/model.pt"

    main([*base, "--flavor", "full"])
    assert target.read_bytes() == b"blob:verifier_s0/step_4000/full_state.pt"
    assert fake_hub == [
        "verifier_s0/step_4000/model.pt",
        "verifier_s0/step_4000/full_state.pt",
    ]


def test_download_unknown_name_and_step_fail_clearly(
    fake_hub: list[str],
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="generator_s44, verifier_s0"):
        main(["--names", "absent", "--dest", str(tmp_path)])
    # step_10000 exists for the model flavor only.
    with pytest.raises(SystemExit, match="Step 10000 not available"):
        main(
            [
                "--names",
                "generator_s44",
                "--dest",
                str(tmp_path),
                "--step",
                "10000",
                "--flavor",
                "full",
            ],
        )
    assert fake_hub == []


# ---------------------------------------------------------------------------
# SHA256SUMS verification.
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_hub_with_sums")
def test_download_verifies_against_sha256sums(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dest = tmp_path / "scratch"
    with caplog.at_level(logging.INFO):
        assert main(["--names", "generator_s44", "--dest", str(dest)]) == 0
    assert "sha256 ok" in caplog.text
    target = dest / "runs" / "generator_s44" / "checkpoints" / "step_00019500.pt"
    assert target.is_file()


def test_download_refreshes_when_published_checksum_changes(
    fake_hub_with_sums: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dest = tmp_path / "scratch"
    argv = ["--names", "generator_s44", "--dest", str(dest)]
    assert main(argv) == 0
    repo_path = "generator_s44/step_19500/model.pt"
    replacement = b"new published checkpoint"
    fake_hub_with_sums[repo_path] = hashlib.sha256(replacement).hexdigest()

    def changed_hub_download(repo_id: str, filename: str) -> str:
        del repo_id
        cached = tmp_path / "changed-hub-cache" / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        if filename == "SHA256SUMS":
            cached.write_text(
                "".join(
                    f"{digest}  {path}\n" for path, digest in fake_hub_with_sums.items()
                ),
            )
        else:
            assert filename == repo_path
            cached.write_bytes(replacement)
        return str(cached)

    monkeypatch.setattr(
        download_checkpoints,
        "hf_hub_download",
        changed_hub_download,
    )

    assert main(argv) == 0

    target = dest / "runs/generator_s44/checkpoints/step_00019500.pt"
    assert target.read_bytes() == replacement


def test_download_rejects_corrupt_file(
    fake_hub_with_sums: dict[str, str],
    tmp_path: Path,
) -> None:
    fake_hub_with_sums["verifier_s0/step_4000/model.pt"] = "0" * 64
    dest = tmp_path / "scratch"
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        main(["--names", "verifier_s0", "--dest", str(dest)])
    # The corrupt file was deleted, so a rerun re-downloads it.
    bad = dest / "runs" / "verifier_s0" / "checkpoints" / "step_00004000.pt"
    assert not bad.exists()


def test_download_notes_missing_checksum_entry(
    fake_hub_with_sums: dict[str, str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del fake_hub_with_sums["verifier_s0/step_4000/model.pt"]
    dest = tmp_path / "scratch"
    with caplog.at_level(logging.INFO):
        assert main(["--names", "verifier_s0", "--dest", str(dest)]) == 0
    assert "no entry for verifier_s0/step_4000/model.pt" in caplog.text
    assert "no checksum entry" in caplog.text
    target = dest / "runs" / "verifier_s0" / "checkpoints" / "step_00004000.pt"
    assert target.is_file()  # Kept: unverifiable, not proven bad.


@pytest.mark.usefixtures("fake_hub")
def test_download_notes_absent_sha256sums(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert main(["--names", "verifier_s0", "--dest", str(tmp_path)]) == 0
    assert "publishes no SHA256SUMS" in caplog.text


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
