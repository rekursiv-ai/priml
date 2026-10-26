"""Acquire complete selected DICOM series through the authorized Kaggle archive."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self, override
from zipfile import ZipInfo

import csv
import hashlib
import logging
import shutil
import subprocess

from configgle import Fig
from treekle import json_dumps, to_plain

from priml.baselines.rsna_knee.imaging import (
    dicom_files,
    is_dicom_member,
)
from priml.baselines.rsna_knee.scripts.archive import (
    VerifiedFile,
    fetch_member,
    index_archive,
    verify_member,
)
from priml.baselines.rsna_knee.train_loop import (
    RECEIPT,
    SELECTED_FILES,
    SERIES_FILES,
    VERIFIED_FILES,
)
from priml.data.ensure import DataSpec, FileSpec, ensure_data
from priml.paths import resolve_working_dir, validated_output_path


if TYPE_CHECKING:
    import httpx

else:
    from wrapt import lazy_import

    httpx = lazy_import("httpx")


class Acquisition:
    """Resume complete verified files and publish only a fully acquired inventory."""

    class Config(Fig["Acquisition"]):
        base_dir: Path | None = Path("/opt/scratch")
        """Resource root beneath which every logical directory below resolves."""

        input_dir: Path = Path("/datasets/rsna-knee/raw")
        """Official tables and final complete DICOM series."""

        selected_series: Path = Path("/datasets/rsna-knee/selected-series.csv")
        """Acquisition-only inventory of selected native training series."""

        staging_dir: Path = Path("/datasets/rsna-knee/staging")
        """Resumable pixels, hidden from model inputs until every selected file verifies."""

        workers: int = 4
        """Bounded simultaneous range requests."""

        working_dir: Path = Path("/artifacts/rsna-knee/acquisition")
        """Durable complete archive catalog and per-file verification manifests."""

        @override
        def finalize(self) -> Self:
            self.input_dir = resolve_working_dir(self.base_dir, self.input_dir)
            self.selected_series = resolve_working_dir(
                self.base_dir,
                self.selected_series,
            )
            self.staging_dir = resolve_working_dir(self.base_dir, self.staging_dir)
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.workers < 1 or config.workers > 8:
            raise ValueError("Acquisition workers must lie between one and eight.")
        source, staging = config.input_dir.resolve(), config.staging_dir.resolve()
        if source.is_relative_to(staging) or staging.is_relative_to(source):
            raise ValueError(
                "Acquisition input and staging directories must be separate.",
            )
        self.config = config

    def run(self, *args: str) -> None:
        """Acquire selected raw pixels, retaining a false receipt until publication.

        Args:
          *args: Passthrough arguments are rejected.

        """
        if args:
            raise ValueError("Acquisition accepts configuration fields only.")
        for name in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
        config = self.config
        protected = [config.selected_series, config.input_dir / "test_series.csv"]
        working_dir = validated_output_path(config.working_dir, protected=protected)
        if any(
            path.resolve().is_relative_to(working_dir.resolve()) for path in protected
        ):
            raise ValueError("Acquisition output directory contains protected inputs.")
        manifests = {
            name: validated_output_path(working_dir / name, protected=protected)
            for name in (
                RECEIPT,
                SELECTED_FILES,
                VERIFIED_FILES,
                SERIES_FILES,
            )
        }
        working_dir.mkdir(parents=True, exist_ok=True)
        receipt = manifests[RECEIPT]
        receipt.write_text(json_dumps({"complete": False}) + "\n")
        url = _archive_url()
        members, archive_size = index_archive(url, working_dir=working_dir)
        selected = select_members(
            members,
            training_csv=config.selected_series,
            test_csv=config.input_dir / "test_series.csv",
        )
        del members
        series_counts = Counter(
            str(Path(member.filename).parent) for member in selected
        )
        if {
            str(path.relative_to(config.input_dir))
            for series in series_counts
            for path in dicom_files(config.input_dir / series)
        } - {member.filename for member in selected}:
            raise ValueError("Obsolete published DICOM members in selected series.")
        with manifests[SELECTED_FILES].open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "bytes", "crc32"])
            writer.writerows(
                (member.filename, member.file_size, member.CRC) for member in selected
            )
        config.staging_dir.mkdir(parents=True, exist_ok=True)
        for member in selected:
            staged = config.staging_dir
            for part in Path(member.filename).parts:
                staged /= part
                if staged.is_symlink():
                    staged.unlink()
            partial_file = staged.with_suffix(staged.suffix + ".part")
            if partial_file.is_symlink():
                partial_file.unlink()
        verified: dict[str, VerifiedFile] = {}
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            fetch = _ArchiveFetch(
                config=config,
                url=url,
                archive_size=archive_size,
                members=selected,
                pool=pool,
                verified=verified,
            )
            ensure_data(
                DataSpec(
                    target_dir=config.staging_dir,
                    manifest=[
                        FileSpec(rel_path=member.filename, size=member.file_size)
                        for member in selected
                    ],
                    fetch=fetch,
                ),
            )
        for member in selected:
            if member.filename not in verified:
                verified[member.filename] = verify_member(
                    member,
                    path=config.staging_dir / member.filename,
                )
        count = len(verified)
        total_bytes = sum(result.size for result in verified.values())
        with manifests[VERIFIED_FILES].open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "bytes", "crc32", "sha256"])
            for member in selected:
                result = verified[member.filename]
                writer.writerow([result.name, result.size, result.crc32, result.sha256])
        for member in selected:
            published = config.input_dir / member.filename
            published.parent.mkdir(parents=True, exist_ok=True)
            (config.staging_dir / member.filename).replace(published)
        with manifests[SERIES_FILES].open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["series", "files"])
            writer.writerows(sorted(series_counts.items()))
        receipt.write_text(
            json_dumps(
                {
                    "complete": True,
                    "input_dir": str(config.input_dir.resolve()),
                    "inventory_sha256": hashlib.sha256(
                        manifests[SELECTED_FILES].read_bytes(),
                    ).hexdigest(),
                    "verified_sha256": hashlib.sha256(
                        manifests[VERIFIED_FILES].read_bytes(),
                    ).hexdigest(),
                    "files": count,
                    "bytes": total_bytes,
                    "series": len(series_counts),
                    "archive_bytes": archive_size,
                    "verification": "Complete central directory; every selected file length and CRC32 verified; SHA256 recorded for subsequent inventory verification.",
                },
                indent=2,
            )
            + "\n",
        )


def select_members(
    members: list[ZipInfo],
    *,
    training_csv: Path,
    test_csv: Path,
) -> list[ZipInfo]:
    """Require every selected training and test series in the full archive inventory.

    Args:
      members: Complete central-directory entries.
      training_csv: Native training series inventory.
      test_csv: Official complete test-series table.

    Returns:
      selected: Every DICOM member of each requested series.

    """
    expected: set[tuple[str, str, str]] = set()
    for split, path in (("train", training_csv), ("test", test_csv)):
        with path.open() as handle:
            expected.update(
                (split, row["StudyInstanceUID"], row["SeriesInstanceUID"])
                for row in csv.DictReader(handle)
            )
    selected: list[ZipInfo] = []
    found: set[tuple[str, str, str]] = set()
    names: set[str] = set()
    for member in members:
        path = Path(member.filename)
        parts = path.parts
        if len(parts) != 4 or not is_dicom_member(path):
            continue
        split = {
            "train_images": "train",
            "train_series": "train",
            "test_images": "test",
            "test_series": "test",
        }.get(parts[0])
        if split is None:
            continue
        key = split, parts[1], parts[2]
        if key not in expected:
            continue
        if (
            member.filename in names
            or ".." in parts
            or member.filename != path.as_posix()
        ):
            raise ValueError("Duplicate or unsafe selected ZIP filename.")
        names.add(member.filename)
        found.add(key)
        selected.append(member)
    if found != expected or not selected:
        raise ValueError("The complete archive is missing selected DICOM series.")
    return sorted(selected, key=lambda member: member.filename)


def acquire() -> Acquisition.Config:
    """Build the acquisition-only priml CLI configuration."""
    return Acquisition.Config()


def _archive_url() -> str:
    error = RuntimeError(
        "Authenticated archive request failed. Use an installed, authenticated Kaggle CLI.",
    )
    try:
        token = subprocess.run(
            ["/usr/bin/env", "kaggle", "auth", "print-access-token"],
            capture_output=True,
            check=True,
            text=True,
            timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise error from None
    if not token or any(character.isspace() for character in token):
        raise error
    target: httpx.URL | None = None
    try:
        with (
            httpx.Client(follow_redirects=False, timeout=60) as client,
            client.stream(
                "POST",
                url="https://api.kaggle.com/v1/competitions.CompetitionApiService/DownloadDataFiles",
                headers={"Authorization": f"Bearer {token}"},
                json=to_plain({"competitionName": "rsna-knee-abnormality-detection"}),
            ) as response,
        ):
            redirect = response.next_request
            if redirect is not None:
                target = redirect.url
            else:
                response.raise_for_status()
    except httpx.HTTPStatusError as failure:
        raise RuntimeError(f"{error} HTTP {failure.response.status_code}.") from None
    except (ValueError, httpx.HTTPError) as failure:
        raise RuntimeError(f"{error} {type(failure).__name__}.") from None
    if target is None or target.scheme != "https" or target.host == "api.kaggle.com":
        raise error
    return str(target)


@dataclass(slots=True, kw_only=True)
class _ArchiveFetch:
    """Materialize one member per call, fetching the next few ahead of the request."""

    config: Acquisition.Config
    url: str
    archive_size: int
    members: list[ZipInfo]
    pool: ThreadPoolExecutor
    verified: dict[str, VerifiedFile]
    futures: dict[str, Future[VerifiedFile]] = field(default_factory=dict)

    def __call__(self, *, rel_path: str, dest: Path) -> None:
        """Fetch and verify one member, with the following members in flight.

        Args:
          rel_path: Requested member, relative to the staging directory.
          dest: Its staged location, `config.staging_dir / rel_path`.

        """
        del dest
        names = [member.filename for member in self.members]
        start = names.index(rel_path)
        for member in self.members[start : start + self.config.workers]:
            if member.filename not in self.futures:
                self.futures[member.filename] = self.pool.submit(self._fetch, member)
        self.verified[rel_path] = self.futures.pop(rel_path).result()

    def _fetch(self, member: ZipInfo) -> VerifiedFile:
        published = self.config.input_dir / member.filename
        staged = self.config.staging_dir / member.filename
        if published.is_file() and not staged.is_file():
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(published, staged)
        return fetch_member(
            member,
            url=self.url,
            archive_size=self.archive_size,
            staging_dir=self.config.staging_dir,
        )
