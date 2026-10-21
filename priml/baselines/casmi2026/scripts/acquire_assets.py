"""Verify the manually placed public archives against the pinned digests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self, override

from configgle import Fig

from priml.data.ensure import DataSpec, FileSpec, ensure_data
from priml.paths import resolve_working_dir


_CWD: Final = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True, kw_only=True)
class Asset:
    """One public archive version and the digest of its complete file."""

    owner: str
    slug: str
    version: int
    sha256: str

    @property
    def archive(self) -> str:
        """Return the archive's file name, as the public host names a download."""
        return f"{self.slug}.zip"


def read_assets(path: Path) -> list[Asset]:
    """Read the tab-separated ``owner/slug``, version and digest of every archive.

    Args:
      path: Pin table with one archive per line.

    Returns:
      assets: Every pinned archive in table order.

    """
    assets: list[Asset] = []
    for line in path.read_text().splitlines():
        name, version, digest = line.split("\t")
        owner, slug = name.split("/")
        assets.append(
            Asset(owner=owner, slug=slug, version=int(version), sha256=digest),
        )
    return assets


class AssetAcquisition:
    """Check each placed archive and stamp its directory as verified."""

    class Config(Fig["AssetAcquisition"]):
        base_dir: Path | str | None = "/opt/scratch"
        """Resource root beneath which the archive directory resolves."""

        working_dir: Path | str = "/datasets/casmi2026/archives"
        """Directory holding ``<owner>/<slug>/<slug>.zip`` for every pinned archive."""

        assets: Path = _CWD.parent / "assets.tsv"
        """Pin table of archive versions and digests."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Verify every pinned archive, naming the first one that is absent.

        Args:
          *args: No passthrough arguments are accepted.

        Raises:
          ValueError: Arguments were passed.
          FileNotFoundError: An archive is absent and must be downloaded.

        """
        if args:
            raise ValueError("AssetAcquisition accepts configuration fields only.")
        for asset in read_assets(self.config.assets):
            _ = ensure_data(
                DataSpec(
                    target_dir=Path(self.config.working_dir) / asset.owner / asset.slug,
                    manifest=[FileSpec(rel_path=asset.archive, sha256=asset.sha256)],
                    fetch=_Missing(asset=asset),
                ),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Missing:
    """Name the archive a person must download, since none can be fetched here."""

    asset: Asset

    def __call__(self, *, rel_path: str, dest: Path) -> None:
        raise FileNotFoundError(
            f"Download version {self.asset.version} of "
            f"{self.asset.owner}/{self.asset.slug} to {dest} ({rel_path}).",
        )


def acquire_assets() -> AssetAcquisition.Config:
    """Declare the verification of every pinned public archive."""
    return AssetAcquisition.Config()
