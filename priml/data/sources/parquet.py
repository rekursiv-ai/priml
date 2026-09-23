"""Parquet-based data source for WebDataset-style datasets.

Generic source for reading metadata from parquet files.

Uses pyarrow directly -- not pandas. pandas adds ~200ms import overhead
which compounds across test and multiprocessing startup.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

import logging
import weakref


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
else:
    from wrapt import lazy_import

    pa = lazy_import("pyarrow")  # ~150 ms; only ParquetAndTarSource needs it.
    pc = lazy_import("pyarrow.compute")  # ~150 ms; only ParquetAndTarSource needs it.
    pq = lazy_import("pyarrow.parquet")  # ~150 ms; only ParquetAndTarSource needs it.

from collections.abc import Callable

from configgle import Fig

from priml.data.sources.sharding import shard_and_shuffle
from priml.data.sources.tarhandle import TarFileHandle
from priml.paths import resolve_working_dir


logger = logging.getLogger(__name__)


__all__ = ["ParquetAndTarSource"]


class ParquetAndTarSource:
    """Load samples from parquet+tar webdataset shards with automatic tar file management.

    This source:
    - Reads metadata from parquet files
    - Injects refcounted tar file handles for lazy image loading
    - Automatically manages tar file lifecycle via Python GC and weak references

    Supports:
    - Directory scanning with glob patterns
    - Shard filtering by ID
    - Worker slicing for distributed training
    - Shuffling for randomization
    - Interleaved reading from multiple files for better I/O
    - Status filtering (only yields rows where status=="success")
    - Configurable tar file path construction

    Example:
        source = ParquetAndTarSource.Config(
            working_dir="/data/recap",
            shard_ids=[0, 1, 2],
        ).make()

        for sample in source:
            # sample has parquet metadata + _tar_handle for image loading
            # ImageLoader will use _tar_handle and delete it, triggering GC
            pass

    """

    class Config(Fig["ParquetAndTarSource"]):
        """Configuration for ParquetAndTarSource."""

        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/datasets/parquet"
        """Logical directory containing parquet and tar files."""

        worker_slice: tuple[int, int] | None = None
        """Worker slice as (worker_id, num_workers) for distributed loading."""

        shard_ids: list[int] | None = None
        """Restrict to these shard IDs (None = all shards)."""

        shuffle: bool = False
        """Shuffle shard order for randomization across epochs."""

        epoch_seed: int = 0
        """Seed folded into the per-epoch shuffle; set by the loader per epoch.

        Epoch state must originate in the main process: under ``num_workers>0``
        each epoch re-forks a fresh source, so a self-incrementing counter would
        reset to zero every epoch. The dataset wrapper injects the current epoch
        here (mirroring ``worker_slice``) so all workers of one epoch share the seed and
        the order changes across epochs.
        """

        num_concurrently_read_shards: int = 4
        """Number of shards to interleave for I/O performance."""

        use_mmap: bool = True
        """Use memory-mapped I/O for tar files."""

        tar_path_fn: Callable[[Path, int], Path] | None = None
        """Custom (working directory, shard index) to tar-path mapping."""

        fail_on_shard_error: bool = False
        """Raise on an unreadable shard instead of skipping it (silent data loss)."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config):
        if config.num_concurrently_read_shards < 1:
            # Zero leaves __iter__'s deque empty, so the source yields nothing
            # at all rather than failing.
            raise ValueError(
                "num_concurrently_read_shards must be >= 1, got "
                f"{config.num_concurrently_read_shards}.",
            )
        self.dataset_dir = Path(config.working_dir)
        self.shuffle = config.shuffle
        self.worker_slice = config.worker_slice
        self.epoch_seed = config.epoch_seed
        self.num_concurrently_read_shards = config.num_concurrently_read_shards
        self.use_mmap = config.use_mmap
        self.fail_on_shard_error = config.fail_on_shard_error

        # Default tar_path_fn: dataset_dir / "{shard_index:08d}.tar"
        if config.tar_path_fn is None:
            self.tar_path_fn: Callable[[Path, int], Path] = (
                lambda dataset_dir, shard_index: dataset_dir / f"{shard_index:08d}.tar"
            )
        else:
            self.tar_path_fn = config.tar_path_fn

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {self.dataset_dir}")

        self.parquet_files = sorted(self.dataset_dir.glob("*.parquet"))

        if not self.parquet_files:
            raise ValueError(f"No parquet files found in {self.dataset_dir}")

        # Filter to specific shard IDs if requested.
        if config.shard_ids is not None:
            shard_id_set = set(config.shard_ids)
            self.parquet_files = [
                p for p in self.parquet_files if int(p.stem) in shard_id_set
            ]
            if len(self.parquet_files) == 0:
                raise ValueError(
                    f"No shards found matching shard_ids={config.shard_ids}",
                )

        # Shuffle-and-slice is deferred to __iter__ so the full file list can be
        # reshuffled with a fresh per-epoch seed every epoch; shuffling here in
        # __init__ would freeze one permutation that every epoch re-iterates.

        # Cache for tar file handles using weak references.
        # This allows TarFileHandle objects to be GC'd when no samples reference them,
        # even though they're still in the cache.
        self._tar_handle_cache: weakref.WeakValueDictionary[int, TarFileHandle] = (
            weakref.WeakValueDictionary()
        )

        tar_mode = "mmap" if self.use_mmap else "standard"
        logger.info(
            "ParquetAndTarSource initialized: %d shards, num_concurrently_read_shards=%d, tar_mode=%s",
            len(self.parquet_files),
            self.num_concurrently_read_shards,
            tar_mode,
        )

    def __len__(self) -> int:
        """Return this worker's shard count (parquet files), not total samples."""
        # Slicing is shuffle-invariant in length, so count without shuffling.
        return len(
            shard_and_shuffle(self.parquet_files, worker_slice=self.worker_slice),
        )

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Iterate with interleaved parallel file reading and tar handle injection.

        Reshuffles the full file list with the loader-injected ``epoch_seed``
        (when ``shuffle`` is set) before slicing, so every epoch sees a new shard
        order while all distributed workers share the same permutation.

        Yields:
            Sample dicts containing metadata from parquet rows + _tar_handle.

        """
        # Shuffle the full list with this epoch's shared seed, then slice. Both
        # happen here (not __init__) so each epoch reshuffles.
        parquet_files = shard_and_shuffle(
            self.parquet_files,
            worker_slice=self.worker_slice,
            shuffle=self.shuffle,
            epoch_seed=self.epoch_seed,
        )

        # Interleave reading from multiple shards for better I/O performance.
        active_shards: deque[Generator[dict[str, object], None, None]] = deque()
        shard_iter = iter(parquet_files)

        # Initialize with num_concurrently_read_shards shards.
        for _ in range(min(self.num_concurrently_read_shards, len(parquet_files))):
            try:
                parquet_path = next(shard_iter)
                active_shards.append(
                    self._read_parquet_shard_with_tar_handle(parquet_path),
                )
            except StopIteration:
                break

        # Round-robin through active shards.
        while active_shards:
            shard = active_shards.popleft()
            try:
                yield next(shard)
                active_shards.append(shard)  # Re-add to end for round-robin.
            except StopIteration:
                # This shard is exhausted, try to load a new one.
                try:
                    parquet_path = next(shard_iter)
                    active_shards.append(
                        self._read_parquet_shard_with_tar_handle(parquet_path),
                    )
                except StopIteration:
                    pass  # No more shards to load.

    def _read_parquet_shard_with_tar_handle(
        self,
        parquet_path: Path,
    ) -> Generator[dict[str, object], None, None]:
        """Read parquet shard and inject tar file handles into samples."""
        try:
            table = pq.read_table(parquet_path)
        except (OSError, ValueError):
            if self.fail_on_shard_error:
                raise
            logger.exception("Failed to read %s, skipping shard", parquet_path.name)
            return

        # Filter to successful samples if status column exists.
        if "status" in table.column_names:
            mask = cast("pa.Array", pc.equal(table.column("status"), "success"))
            table = table.filter(mask)

        if table.num_rows == 0:
            logger.warning("No successful samples in %s, skipping", parquet_path.name)
            return

        # Extract shard index from filename (e.g., "00000000.parquet" -> 0).
        # A non-numeric stem must not silently collapse to 0: every such shard
        # would share index 0, colliding tar handles and clobbering samples.
        try:
            shard_index = int(parquet_path.stem)
        except ValueError as e:
            raise ValueError(
                f"Parquet filename stem is not an integer shard index: "
                f"{parquet_path.name}",
            ) from e

        # Get or create cached tar file handle
        # Use .get() to handle weak references that may have been GC'd.
        tar_handle = self._tar_handle_cache.get(shard_index)
        if tar_handle is None:
            tar_path: Path = self.tar_path_fn(self.dataset_dir, shard_index)
            tar_handle = TarFileHandle(tar_path, use_mmap=self.use_mmap)
            self._tar_handle_cache[shard_index] = tar_handle

        # Yield each row as a sample dict with all columns + tar handle.
        columns = table.column_names
        for i in range(table.num_rows):
            sample: dict[str, object] = {
                "shard_index": shard_index,
                "_tar_handle": tar_handle,
            }
            for col in columns:
                value: object = table.column(col)[i].as_py()  # pyright: ignore[reportAny] -- PyArrow's stub returns Any for arbitrary parquet scalar values.
                sample[col] = value
            yield sample
