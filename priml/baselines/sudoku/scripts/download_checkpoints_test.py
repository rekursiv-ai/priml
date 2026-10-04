"""Tests for scripts.download_checkpoints (Hub calls monkeypatched; no network)."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Final, Literal, Self

import hashlib
import logging
import sys
import tempfile

import pytest

from priml.baselines.sudoku.scripts import download_checkpoints
from priml.baselines.sudoku.scripts.download_checkpoints import (
    _existing_matches_source,
    _file_sha256,
    _load_checksums,
    _parse_args,
    _read_provenance,
    _verify_download,
    _write_provenance,
    available_steps,
    destination_path,
    main,
)


_FILES: Final = [
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
        assert repo_id == "rekursiv-ai/sudoku-trm"
        return list(_FILES)

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "rekursiv-ai/sudoku-trm"
        downloads.append(filename)
        cached = tmp_path / "hub-cache" / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(f"blob:{filename}".encode())
        return str(cached)

    monkeypatch.setattr(download_checkpoints, "list_repo_files", fake_list_repo_files)
    monkeypatch.setattr(download_checkpoints, "hf_hub_download", fake_hf_hub_download)
    return downloads


@pytest.fixture
def fake_hub_with_sums(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, str]:
    """Fake hub publishing SHA256SUMS; returns the mutable digest map."""
    sums: dict[str, str] = {
        f: hashlib.sha256(f"blob:{f}".encode()).hexdigest()
        for f in _FILES
        if f.endswith(".pt")
    }

    def fake_list_repo_files(repo_id: str) -> list[str]:
        assert repo_id == "rekursiv-ai/sudoku-trm"
        return [*_FILES, "SHA256SUMS"]

    def fake_hf_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "rekursiv-ai/sudoku-trm"
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


def test_parse_args_defaults_and_types(tmp_path: Path) -> None:
    args = _parse_args(["--names", "exp006"])
    assert args.names == ["exp006"]
    assert args.repo_id == "rekursiv-ai/sudoku-trm"
    assert args.dest is None
    assert args.step is None
    assert args.flavor == "model"
    assert _parse_args(["--names", "exp006", "--flavor", "model"]).flavor == "model"

    args = _parse_args(
        [
            "--names",
            "hub:run",
            "--repo-id",
            "owner/repo",
            "--dest",
            str(tmp_path),
            "--step",
            "17",
            "--flavor",
            "full",
        ],
    )
    assert args.names == ["hub:run"]
    assert args.repo_id == "owner/repo"
    assert args.dest == tmp_path
    assert args.step == 17
    assert args.flavor == "full"


def test_parse_args_requires_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(download_checkpoints, "__doc__", None)

    with pytest.raises(ValueError, match=r"^Expected __doc__ is not None\.$"):
        _parse_args(["--names", "exp006"])


def test_parse_args_rejects_missing_names_and_bad_flavor() -> None:
    with pytest.raises(SystemExit, match="2"):
        _parse_args([])
    with pytest.raises(SystemExit, match="2"):
        _parse_args(["--names", "exp006", "--flavor", "other"])


def test_help_includes_usage_description(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        _parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert (
        "Download released checkpoints from the Hub into the local run-dir layout."
        in help_text
    )
    assert (
        "Fetches the released checkpoints and places each one EXACTLY where"
        in help_text
    )


def test_provenance_round_trip_and_rejects_malformed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "step.pt.source.json"
    value = {
        "sha256": "a" * 64,
        "repo_path": "exp/step_2/model.pt",
        "repo_id": "owner/repo",
    }
    temporary_paths: list[Path] = []
    temporary_kwargs: list[dict[str, str | bool | Path | None]] = []
    real_named_temporary_file = tempfile.NamedTemporaryFile

    def record_tempfile(
        *,
        dir: Path,
        prefix: str,
        suffix: str,
        delete: bool,
        mode: Literal["w"] | None = None,
        encoding: str | None = None,
    ):
        kwargs: dict[str, str | bool | Path | None] = {
            "dir": dir,
            "prefix": prefix,
            "suffix": suffix,
            "delete": delete,
        }
        if mode is not None:
            kwargs.update(mode=mode, encoding=encoding)
            temporary = real_named_temporary_file(
                mode=mode,
                encoding=encoding,
                dir=dir,
                prefix=prefix,
                suffix=suffix,
                delete=delete,
            )
        else:
            temporary = real_named_temporary_file(
                dir=dir,
                prefix=prefix,
                suffix=suffix,
                delete=delete,
            )
        temporary_kwargs.append(kwargs)
        temporary_paths.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", record_tempfile)
    _write_provenance(receipt, value)
    assert len(temporary_paths) == 1
    temporary_path = temporary_paths[0]
    assert temporary_path.parent == tmp_path
    assert temporary_path.name.startswith(".step.pt.source.json.")
    assert temporary_path.name.endswith(".tmp")
    assert not temporary_path.exists()
    assert temporary_kwargs == [
        {
            "mode": "w",
            "encoding": "utf-8",
            "dir": receipt.parent,
            "prefix": ".step.pt.source.json.",
            "suffix": ".tmp",
            "delete": False,
        },
    ]
    assert receipt.read_text() == (
        '{"repo_id": "owner/repo", "repo_path": "exp/step_2/model.pt", '
        f'"sha256": "{"a" * 64}"' + "}\n"
    )
    assert _read_provenance(receipt) == value
    assert list(tmp_path.iterdir()) == [receipt]

    for text in (
        "not json",
        "[]",
        '{"repo_id": "owner/repo"}',
        '{"repo_id": 3, "repo_path": "x", "sha256": "y"}',
    ):
        receipt.write_text(text)
        assert _read_provenance(receipt) is None
    assert _read_provenance(tmp_path / "missing.json") is None


def test_existing_match_returns_false_when_hashing_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"payload")
    source = {"repo_id": "owner/repo", "repo_path": "x/model.pt", "sha256": "x"}

    def unreadable(path: Path) -> str:
        del path
        raise PermissionError("denied")

    monkeypatch.setattr(download_checkpoints, "_file_sha256", unreadable)
    assert not _existing_matches_source(
        target,
        source,
        repo_id="owner/repo",
        repo_path="x/model.pt",
        expected_sha256=None,
    )


def test_existing_match_requires_file_source_identity_and_content(
    tmp_path: Path,
) -> None:
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    source = {"repo_id": "owner/repo", "repo_path": "x/model.pt", "sha256": digest}
    assert _existing_matches_source(
        target,
        source,
        repo_id="owner/repo",
        repo_path="x/model.pt",
        expected_sha256=digest,
    )
    assert not _existing_matches_source(
        target,
        source,
        repo_id="other/repo",
        repo_path="x/model.pt",
        expected_sha256=digest,
    )
    assert not _existing_matches_source(
        target,
        source,
        repo_id="owner/repo",
        repo_path="other/model.pt",
        expected_sha256=digest,
    )
    assert not _existing_matches_source(
        target,
        source,
        repo_id="owner/repo",
        repo_path="x/model.pt",
        expected_sha256="0" * 64,
    )
    assert not _existing_matches_source(
        target,
        {**source, "sha256": "0" * 64},
        repo_id="owner/repo",
        repo_path="x/model.pt",
        expected_sha256=None,
    )
    assert not _existing_matches_source(
        target,
        None,
        repo_id="owner/repo",
        repo_path="x/model.pt",
        expected_sha256=None,
    )
    target.unlink()
    assert not _existing_matches_source(
        target,
        source,
        repo_id="owner/repo",
        repo_path="x/model.pt",
        expected_sha256=None,
    )
    target.write_bytes(b"payload")
    link = tmp_path / "checkpoint-link.pt"
    link.symlink_to(target)
    assert not _existing_matches_source(
        link,
        source,
        repo_id="owner/repo",
        repo_path="x/model.pt",
        expected_sha256=digest,
    )


def test_file_sha256_reads_complete_file_in_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "large.bin"
    payload = b"a" * (1 << 20) + b"tail"
    path.write_bytes(payload)
    read_sizes: list[int] = []
    real_path_open = Path.open

    class RecordingFile:
        file: BinaryIO | None = None

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *args: object) -> None:
            del args
            assert self.file is not None
            self.file.close()

        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            assert self.file is not None
            return self.file.read(size)

    def recording_open(
        file_path: Path,
        mode: Literal["rb"] = "rb",
    ) -> RecordingFile | BinaryIO:
        if file_path != path:
            return real_path_open(file_path, mode)
        result = RecordingFile()
        result.file = real_path_open(file_path, mode)
        return result

    monkeypatch.setattr(Path, "open", recording_open)
    assert _file_sha256(path) == hashlib.sha256(payload).hexdigest()
    assert read_sizes == [1 << 20, 1 << 20, 1 << 20]


def test_load_checksums_skips_invalid_and_normalizes_sha256sum_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sums_path = tmp_path / "SHA256SUMS"
    digest = "A" * 64
    sums_path.write_text(
        f"  # comment\n\n{digest}  file.pt\n{digest} *star.pt\n"
        f"{digest}  *XXstar.pt\n{digest}  file with spaces.pt\n"
        f"{'#' + 'a' * 63}  comment.pt\n"
        "short  ignored.pt\n" + "b" * 63 + "  wrong.pt extra\n"
        f"{digest}  after-malformed.pt\n",
    )

    def fake_hub_download(repo_id: str, filename: str) -> str:
        assert repo_id == "owner/repo"
        assert filename == "SHA256SUMS"
        return str(sums_path)

    monkeypatch.setattr(download_checkpoints, "hf_hub_download", fake_hub_download)
    assert _load_checksums("owner/repo", ["SHA256SUMS"]) == {
        "file.pt": "a" * 64,
        "star.pt": "a" * 64,
        "XXstar.pt": "a" * 64,
        "file with spaces.pt": "a" * 64,
        "after-malformed.pt": "a" * 64,
    }


def test_load_checksums_absent_file_logs_and_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert _load_checksums("owner/repo", []) is None
    assert caplog.messages == [
        "Note: owner/repo publishes no SHA256SUMS; downloads are not checksum-verified.",
    ]


def test_verify_download_exact_status_missing_entry_and_corruption(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = tmp_path / "checkpoint.pt"
    target.write_bytes(b"payload")
    digest = hashlib.sha256(b"payload").hexdigest()
    assert _verify_download(target, "x/model.pt", None) == ""
    assert (
        _verify_download(target, "x/model.pt", {"x/model.pt": digest}) == ", sha256 ok"
    )
    with caplog.at_level(logging.WARNING):
        assert _verify_download(target, "x/model.pt", {}) == ", no checksum entry"
    assert caplog.messages == [
        "Note: SHA256SUMS has no entry for x/model.pt; file not verified.",
    ]

    with pytest.raises(SystemExit) as error:
        _verify_download(target, "x/model.pt", {"x/model.pt": "0" * 64})
    assert str(error.value) == (
        f"SHA-256 mismatch for x/model.pt:\n  expected {'0' * 64} (from SHA256SUMS)\n"
        f"  got      {digest}\nThe corrupt file was deleted ({target}); rerun to re-download."
    )
    assert not target.exists()


@pytest.mark.usefixtures("fake_hub")
def test_download_places_files_at_eval_defaults(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dest = tmp_path / "scratch"
    argv = ["--names", "generator_s44", "verifier_s0", "--dest", str(dest)]
    temp_options: list[dict[str, str | bool | Path | None]] = []
    temporary_paths: list[Path] = []
    real_named_temporary_file = tempfile.NamedTemporaryFile

    def record_tempfile(
        *,
        dir: Path,
        prefix: str,
        suffix: str,
        delete: bool,
        mode: Literal["w"] | None = None,
        encoding: str | None = None,
    ):
        kwargs: dict[str, str | bool | Path | None] = {
            "dir": dir,
            "prefix": prefix,
            "suffix": suffix,
            "delete": delete,
        }
        if mode is not None:
            kwargs.update(mode=mode, encoding=encoding)
            temporary = real_named_temporary_file(
                mode=mode,
                encoding=encoding,
                dir=dir,
                prefix=prefix,
                suffix=suffix,
                delete=delete,
            )
        else:
            temporary = real_named_temporary_file(
                dir=dir,
                prefix=prefix,
                suffix=suffix,
                delete=delete,
            )
        temp_options.append(kwargs)
        temporary_paths.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", record_tempfile)
    with caplog.at_level(logging.INFO):
        assert main(argv) == 0
    generator = dest / "runs" / "generator_s44" / "checkpoints" / "step_00019500.pt"
    verifier = dest / "runs" / "verifier_s0" / "checkpoints" / "step_00004000.pt"
    assert generator.read_bytes() == b"blob:generator_s44/step_19500/model.pt"
    assert verifier.read_bytes() == b"blob:verifier_s0/step_4000/model.pt"
    assert temp_options == [
        {
            "dir": generator.parent,
            "prefix": f".{generator.name}.",
            "suffix": ".tmp",
            "delete": False,
        },
        {
            "mode": "w",
            "encoding": "utf-8",
            "dir": generator.parent,
            "prefix": f".{generator.with_suffix('.pt.source.json').name}.",
            "suffix": ".tmp",
            "delete": False,
        },
        {
            "dir": verifier.parent,
            "prefix": f".{verifier.name}.",
            "suffix": ".tmp",
            "delete": False,
        },
        {
            "mode": "w",
            "encoding": "utf-8",
            "dir": verifier.parent,
            "prefix": f".{verifier.with_suffix('.pt.source.json').name}.",
            "suffix": ".tmp",
            "delete": False,
        },
    ]
    assert len(temporary_paths) == 4
    assert all(not path.exists() for path in temporary_paths)
    assert caplog.messages == [
        "Note: rekursiv-ai/sudoku-trm publishes no SHA256SUMS; downloads are not checksum-verified.",
        "Checkpoints from https://huggingface.co/rekursiv-ai/sudoku-trm (model.pt):",
        f"  generator_s44 step 19500: {generator} [downloaded, 0.0 MB]",
        f"  verifier_s0 step 4000: {verifier} [downloaded, 0.0 MB]",
    ]
    assert [(record.msg, record.args) for record in caplog.records] == [
        (
            "Note: %s publishes no %s; downloads are not checksum-verified.",
            ("rekursiv-ai/sudoku-trm", "SHA256SUMS"),
        ),
        (
            "Checkpoints from https://huggingface.co/%s (%s):",
            ("rekursiv-ai/sudoku-trm", "model.pt"),
        ),
        ("%s", (caplog.messages[2],)),
        ("%s", (caplog.messages[3],)),
    ]


def test_download_size_is_reported_in_binary_megabytes(
    fake_hub: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def large_checkpoint(repo_id: str, filename: str) -> str:
        assert repo_id == "rekursiv-ai/sudoku-trm"
        fake_hub.append(filename)
        cached = tmp_path / "hub-cache" / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"x" * 577_300)
        return str(cached)

    monkeypatch.setattr(download_checkpoints, "hf_hub_download", large_checkpoint)
    dest = tmp_path / "scratch"
    with caplog.at_level(logging.INFO):
        assert main(["--names", "verifier_s0", "--dest", str(dest)]) == 0
    target = dest / "runs/verifier_s0/checkpoints/step_00004000.pt"
    assert target.stat().st_size == 577_300
    assert caplog.messages[-1] == (
        f"  verifier_s0 step 4000: {target} [downloaded, 0.6 MB]"
    )
    assert fake_hub == ["verifier_s0/step_4000/model.pt"]


def test_download_renames_hub_checkpoint_to_run_name(
    fake_hub: list[str],
    tmp_path: Path,
) -> None:
    dest = tmp_path / "scratch"
    assert main(["--names", "generator_s44:exp010", "--dest", str(dest)]) == 0
    placed = dest / "runs" / "exp010" / "checkpoints" / "step_00019500.pt"
    assert placed.read_bytes() == b"blob:generator_s44/step_19500/model.pt"
    assert fake_hub == ["generator_s44/step_19500/model.pt"]
    assert not (dest / "runs" / "generator_s44").exists()


def test_download_default_dest_is_opt_scratch() -> None:
    assert destination_path(Path("/opt/scratch"), name="verifier_s0", step=4_000) == (
        Path("/opt/scratch/runs/verifier_s0/checkpoints/step_00004000.pt")
    )


def test_download_configures_logging_when_no_root_handlers(
    fake_hub: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_logger = logging.getLogger()
    monkeypatch.setattr(root_logger, "handlers", [])
    basic_config_calls: list[dict[str, object]] = []

    def record_basic_config(**kwargs: object) -> None:
        basic_config_calls.append(kwargs)

    monkeypatch.setattr(logging, "basicConfig", record_basic_config)
    assert main(["--names", "verifier_s0", "--dest", str(tmp_path)]) == 0
    assert basic_config_calls == [
        {
            "level": logging.INFO,
            "format": "%(message)s",
            "stream": sys.stderr,
        },
    ]
    assert fake_hub == ["verifier_s0/step_4000/model.pt"]


def test_download_uses_default_scratch_root(
    fake_hub: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destinations: list[Path] = []

    def local_destination(root: Path, *, name: str, step: int) -> Path:
        destinations.append(root)
        return destination_path(tmp_path, name=name, step=step)

    monkeypatch.setattr(download_checkpoints, "destination_path", local_destination)
    assert main(["--names", "verifier_s0"]) == 0
    assert destinations == [Path("/opt/scratch")]
    assert (tmp_path / "runs/verifier_s0/checkpoints/step_00004000.pt").is_file()
    assert fake_hub == ["verifier_s0/step_4000/model.pt"]


def test_download_is_idempotent(
    fake_hub: list[str],
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    dest = tmp_path / "scratch"
    argv = ["--names", "generator_s44", "--dest", str(dest)]
    assert main(argv) == 0
    with caplog.at_level(logging.INFO):
        assert main(argv) == 0
    assert fake_hub == ["generator_s44/step_19500/model.pt"]
    assert "[skipped, exists]" in caplog.text


def test_download_replaces_corrupt_checkpoint(
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
    assert fake_hub == ["generator_s44/step_19500/model.pt"] * 2


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


def test_download_step_selection_and_full_flavor(
    fake_hub: list[str],
    tmp_path: Path,
) -> None:
    dest = tmp_path / "scratch"
    assert (
        main(["--names", "generator_s44", "--dest", str(dest), "--step", "10000"]) == 0
    )
    picked = dest / "runs" / "generator_s44" / "checkpoints" / "step_00010000.pt"
    assert picked.read_bytes() == b"blob:generator_s44/step_10000/model.pt"
    assert (
        main(["--names", "verifier_s0", "--dest", str(dest), "--flavor", "full"]) == 0
    )
    full = dest / "runs" / "verifier_s0" / "checkpoints" / "step_00004000.pt"
    assert full.read_bytes() == b"blob:verifier_s0/step_4000/full_state.pt"
    assert fake_hub == [
        "generator_s44/step_10000/model.pt",
        "verifier_s0/step_4000/full_state.pt",
    ]


def test_download_replaces_existing_checkpoint_when_flavor_changes(
    fake_hub: list[str],
    tmp_path: Path,
) -> None:
    dest = tmp_path / "scratch"
    base = ["--names", "verifier_s0", "--dest", str(dest)]
    assert main(base) == 0
    target = dest / "runs" / "verifier_s0" / "checkpoints" / "step_00004000.pt"
    assert target.read_bytes() == b"blob:verifier_s0/step_4000/model.pt"
    assert main([*base, "--flavor", "full"]) == 0
    assert target.read_bytes() == b"blob:verifier_s0/step_4000/full_state.pt"
    assert fake_hub == [
        "verifier_s0/step_4000/model.pt",
        "verifier_s0/step_4000/full_state.pt",
    ]


def test_download_unknown_name_with_no_available_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def empty_repository(repo_id: str) -> list[str]:
        del repo_id
        return []

    monkeypatch.setattr(download_checkpoints, "list_repo_files", empty_repository)
    with pytest.raises(SystemExit) as error:
        main(["--names", "absent", "--dest", str(tmp_path)])
    assert str(error.value) == (
        "No 'model.pt' checkpoints for 'absent' in rekursiv-ai/sudoku-trm; "
        "available names: none."
    )


def test_download_unknown_name_and_step_fail_clearly(
    fake_hub: list[str],
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--names", "absent", "--dest", str(tmp_path)])
    assert str(error.value) == (
        "No 'model.pt' checkpoints for 'absent' in rekursiv-ai/sudoku-trm; "
        "available names: generator_s44, verifier_s0."
    )
    with pytest.raises(SystemExit) as error:
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
    assert str(error.value) == (
        "Step 10000 not available for 'generator_s44' in rekursiv-ai/sudoku-trm "
        "(available steps: [19500])."
    )
    assert fake_hub == []


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

    monkeypatch.setattr(download_checkpoints, "hf_hub_download", changed_hub_download)
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
    assert target.is_file()


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
