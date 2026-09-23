"""Tests for Parquet data source."""

from __future__ import annotations

from pathlib import Path

import io
import tarfile
import tempfile
import weakref

from PIL import Image

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from priml.data.sources.parquet import ParquetAndTarSource
from priml.data.sources.tarhandle import TarFileHandle


def test_default_working_dir_is_opinionated() -> None:
    assert ParquetAndTarSource.Config().working_dir == "/datasets/parquet"


def test_owner_resolves_default_working_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "datasets" / "parquet"
    data_dir.mkdir(parents=True)
    create_test_parquet(data_dir / "00000000.parquet")
    config = ParquetAndTarSource.Config()
    config.base_dir = tmp_path

    source = config.make()

    assert source.dataset_dir == data_dir


def test_source_accepts_fixture_data_inside_a_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    data_dir = checkout / "fixtures"
    (checkout / ".git").mkdir(parents=True)
    data_dir.mkdir()
    create_test_parquet(data_dir / "fixture.parquet")

    source = ParquetAndTarSource.Config(working_dir=data_dir).make()

    assert source.dataset_dir == data_dir


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_test_parquet(path: Path, num_rows: int = 2, status: str = "success") -> None:
    """Create a test parquet file with sample metadata."""
    data = {
        "key": [f"sample_{i:08d}" for i in range(num_rows)],
        "caption": [f"A test caption {i}" for i in range(num_rows)],
        "status": [status] * num_rows,
        "width": [256] * num_rows,
        "height": [256] * num_rows,
        "format": ["jpg"] * num_rows,
    }
    pq.write_table(pa.Table.from_pydict(data), path)


def create_test_tar(path: Path, num_images: int = 2) -> None:
    """Create a test tar file with sample images."""
    with tarfile.open(path, "w") as tar:
        for i in range(num_images):
            img = Image.new("RGB", (4, 4), color="blue")
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG")
            img_bytes = img_buffer.getvalue()

            info = tarfile.TarInfo(name=f"sample_{i:08d}.jpg")
            info.size = len(img_bytes)
            tar.addfile(info, io.BytesIO(img_bytes))


class TestTarFileHandle:
    """Test TarFileHandle lifecycle and garbage collection."""

    def test_tar_file_handle_opens_and_closes(self, temp_dir: Path) -> None:
        """Test that TarFileHandle opens and closes tar files."""
        tar_path = temp_dir / "test.tar"
        create_test_tar(tar_path, num_images=1)

        handle = TarFileHandle(tar_path)
        assert handle.path == tar_path
        # Test that we can get a member (which verifies the file is open)
        member = handle.getmember("sample_00000000.jpg")
        assert member.name == "sample_00000000.jpg"

        # Manual cleanup.
        del handle

    def test_tar_file_handle_provides_methods(self, temp_dir: Path) -> None:
        """Test that TarFileHandle provides getmember and extractfile methods."""
        tar_path = temp_dir / "test.tar"
        create_test_tar(tar_path, num_images=1)

        handle = TarFileHandle(tar_path)

        # Test getmember method.
        member = handle.getmember("sample_00000000.jpg")
        assert member.name == "sample_00000000.jpg"

        # Test extractfile method.
        file_obj = handle.extractfile(member)
        assert file_obj is not None
        data = file_obj.read()
        assert len(data) > 0

        del handle

    def test_tar_file_handle_repr(self, temp_dir: Path) -> None:
        """Test TarFileHandle string representation."""
        tar_path = temp_dir / "test.tar"
        create_test_tar(tar_path, num_images=1)

        handle = TarFileHandle(tar_path)
        # The repr now includes member count and mode for mmap mode.
        assert "TarFileHandle" in repr(handle)
        assert str(tar_path) in repr(handle)
        assert "mode=mmap" in repr(handle)

        del handle

    def test_tar_file_handle_garbage_collection(self, temp_dir: Path) -> None:
        """Test that TarFileHandle is garbage collected properly."""
        tar_path = temp_dir / "test.tar"
        create_test_tar(tar_path, num_images=1)

        # Create handle and weak reference to it.
        handle = TarFileHandle(tar_path)
        weak_ref = weakref.ref(handle)

        # Verify handle exists.
        assert weak_ref() is not None

        # Delete the only strong reference.
        del handle

        # Verify handle was garbage collected.
        assert weak_ref() is None


class TestParquetAndTarSourceInit:
    """Test ParquetAndTarSource initialization."""

    def test_init_basic(self, temp_dir: Path) -> None:
        """Test basic initialization."""
        parquet_path = temp_dir / "00000000.parquet"
        tar_path = temp_dir / "00000000.tar"
        create_test_parquet(parquet_path, num_rows=2)
        create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        assert source.dataset_dir == temp_dir
        assert len(source.parquet_files) == 1
        assert source.shuffle is False
        assert source.num_concurrently_read_shards == 4

    def test_init_missing_dataset_dir(self) -> None:
        """Test error when dataset directory doesn't exist."""
        config = ParquetAndTarSource.Config(working_dir="/nonexistent/path")
        with pytest.raises(ValueError, match="Dataset directory does not exist"):
            ParquetAndTarSource(config)

    def test_init_no_parquet_files(self, temp_dir: Path) -> None:
        """Test error when no parquet files found."""
        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        with pytest.raises(ValueError, match="No parquet files found"):
            ParquetAndTarSource(config)

    def test_init_with_shard_ids(self, temp_dir: Path) -> None:
        """Test initialization with shard ID filtering."""
        for i in range(3):
            parquet_path = temp_dir / f"{i:08d}.parquet"
            tar_path = temp_dir / f"{i:08d}.tar"
            create_test_parquet(parquet_path, num_rows=2)
            create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir, shard_ids=[0, 2])
        source = ParquetAndTarSource(config)

        assert len(source.parquet_files) == 2
        assert source.parquet_files[0].stem == "00000000"
        assert source.parquet_files[1].stem == "00000002"

    def test_init_with_invalid_shard_ids(self, temp_dir: Path) -> None:
        """Test error when shard IDs don't match any files."""
        parquet_path = temp_dir / "00000000.parquet"
        tar_path = temp_dir / "00000000.tar"
        create_test_parquet(parquet_path, num_rows=2)
        create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir, shard_ids=[5, 6])
        with pytest.raises(ValueError, match="No shards found matching"):
            ParquetAndTarSource(config)

    @pytest.mark.parametrize("shards", [0, -1])
    def test_init_rejects_non_positive_concurrency(
        self,
        temp_dir: Path,
        shards: int,
    ) -> None:
        """Zero shards in flight would yield an empty stream, not an error."""
        create_test_parquet(temp_dir / "00000000.parquet", num_rows=2)
        create_test_tar(temp_dir / "00000000.tar", num_images=2)

        config = ParquetAndTarSource.Config(
            working_dir=temp_dir,
            num_concurrently_read_shards=shards,
        )
        with pytest.raises(ValueError, match="num_concurrently_read_shards"):
            ParquetAndTarSource(config)

    def test_init_with_shuffle(self, temp_dir: Path) -> None:
        """Test initialization with shuffle enabled."""
        parquet_path = temp_dir / "00000000.parquet"
        tar_path = temp_dir / "00000000.tar"
        create_test_parquet(parquet_path, num_rows=2)
        create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir, shuffle=True)
        source = ParquetAndTarSource(config)

        assert source.shuffle is True

    def test_init_with_slice(self, temp_dir: Path) -> None:
        """Test initialization with worker slicing."""
        for i in range(4):
            parquet_path = temp_dir / f"{i:08d}.parquet"
            tar_path = temp_dir / f"{i:08d}.tar"
            create_test_parquet(parquet_path, num_rows=2)
            create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir, worker_slice=(0, 2))
        source = ParquetAndTarSource(config)

        # Worker 0 of 2 should get first 2 shards (slice applied in __len__/__iter__)
        assert len(source) == 2

    def test_init_with_slice_last_worker(self, temp_dir: Path) -> None:
        """Test initialization with last worker getting remainder."""
        for i in range(5):
            parquet_path = temp_dir / f"{i:08d}.parquet"
            tar_path = temp_dir / f"{i:08d}.tar"
            create_test_parquet(parquet_path, num_rows=2)
            create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir, worker_slice=(1, 2))
        source = ParquetAndTarSource(config)

        # Worker 1 of 2 should get 3 shards (last worker gets remainder)
        assert len(source) == 3


class TestParquetAndTarSourceIteration:
    """Test ParquetAndTarSource iteration."""

    def test_iter_basic(self, temp_dir: Path) -> None:
        """Test basic iteration over samples."""
        parquet_path = temp_dir / "00000000.parquet"
        tar_path = temp_dir / "00000000.tar"
        create_test_parquet(parquet_path, num_rows=2)
        create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        samples = list(source)
        assert len(samples) == 2
        assert samples[0]["key"] == "sample_00000000"
        assert samples[0]["caption"] == "A test caption 0"
        assert "_tar_handle" in samples[0]
        assert isinstance(samples[0]["_tar_handle"], TarFileHandle)

    def test_iter_filters_failed_status(self, temp_dir: Path) -> None:
        """Test that samples with status != 'success' are filtered."""
        parquet_path = temp_dir / "00000000.parquet"
        tar_path = temp_dir / "00000000.tar"

        # Create parquet with mixed statuses.
        data = {
            "key": ["sample_00000000", "sample_00000001", "sample_00000002"],
            "caption": ["Caption 0", "Caption 1", "Caption 2"],
            "status": ["success", "failed", "success"],
        }
        pq.write_table(pa.Table.from_pydict(data), parquet_path)

        create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        samples = list(source)
        # Should only get 2 samples (with status='success')
        assert len(samples) == 2
        assert samples[0]["key"] == "sample_00000000"
        assert samples[1]["key"] == "sample_00000002"

    def test_iter_multiple_shards(self, temp_dir: Path) -> None:
        """Test iteration over multiple shards."""
        for i in range(3):
            parquet_path = temp_dir / f"{i:08d}.parquet"
            tar_path = temp_dir / f"{i:08d}.tar"
            create_test_parquet(parquet_path, num_rows=2)
            create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        samples = list(source)
        assert len(samples) == 6  # 3 shards * 2 samples each.

    def test_iter_with_concurrent_shards(self, temp_dir: Path) -> None:
        """Test iteration with concurrent shard reading."""
        for i in range(3):
            parquet_path = temp_dir / f"{i:08d}.parquet"
            tar_path = temp_dir / f"{i:08d}.tar"
            create_test_parquet(parquet_path, num_rows=2)
            create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(
            working_dir=temp_dir,
            num_concurrently_read_shards=2,
        )
        source = ParquetAndTarSource(config)

        samples = list(source)
        assert len(samples) == 6


class TestParquetAndTarSourceGarbageCollection:
    """Test garbage collection of tar file handles."""

    def test_tar_handle_cache_uses_weak_references(self, temp_dir: Path) -> None:
        """Test that tar handle cache uses weak references."""
        parquet_path = temp_dir / "00000000.parquet"
        tar_path = temp_dir / "00000000.tar"
        create_test_parquet(parquet_path, num_rows=2)
        create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        # Verify cache is WeakValueDictionary.
        assert isinstance(source._tar_handle_cache, weakref.WeakValueDictionary)

    def test_tar_handle_garbage_collected_after_iteration(self, temp_dir: Path) -> None:
        """Test that tar handles are GC'd after samples are deleted."""
        parquet_path = temp_dir / "00000000.parquet"
        tar_path = temp_dir / "00000000.tar"
        create_test_parquet(parquet_path, num_rows=2)
        create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        # Iterate and keep track of tar handle.
        samples = list(source)
        tar_handle = _tar_handle(samples[0])
        weak_ref = weakref.ref(tar_handle)

        # Verify handle exists.
        assert weak_ref() is not None
        assert len(source._tar_handle_cache) == 1
        # Delete all samples and the handle reference.
        del samples
        del tar_handle
        # Verify handle was garbage collected.
        assert weak_ref() is None
        # Cache should now be empty (weak reference was GC'd)
        assert len(source._tar_handle_cache) == 0

    def test_tar_handle_recreated_after_gc(self, temp_dir: Path) -> None:
        """Test that tar handles are recreated if GC'd and needed again."""
        parquet_path = temp_dir / "00000000.parquet"
        tar_path = temp_dir / "00000000.tar"
        create_test_parquet(parquet_path, num_rows=2)
        create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        # First iteration - keep a weak reference to verify GC.
        samples1 = list(source)
        handle1 = _tar_handle(samples1[0])
        weak_ref = weakref.ref(handle1)

        # Delete the only strong references.
        del samples1
        del handle1

        # Verify cache is now empty and handle was GC'd.
        assert len(source._tar_handle_cache) == 0
        assert weak_ref() is None  # Original handle was garbage collected.

        # Second iteration should recreate handle.
        samples2 = list(source)
        handle2 = _tar_handle(samples2[0])

        # Cache should have new handle.
        assert len(source._tar_handle_cache) == 1
        # New handle should work correctly.
        assert handle2.path == tar_path
        # Verify we can use it.
        member = handle2.getmember("sample_00000000.jpg")
        assert member.name == "sample_00000000.jpg"

        del samples2

    def test_tar_handle_shared_across_samples_from_same_shard(
        self,
        temp_dir: Path,
    ) -> None:
        """Test that samples from same shard share the same tar handle."""
        parquet_path = temp_dir / "00000000.parquet"
        tar_path = temp_dir / "00000000.tar"
        create_test_parquet(parquet_path, num_rows=3)
        create_test_tar(tar_path, num_images=3)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        samples = list(source)
        assert len(samples) == 3

        # All samples should share the same tar handle.
        handle0_id = id(_tar_handle(samples[0]))
        handle1_id = id(_tar_handle(samples[1]))
        handle2_id = id(_tar_handle(samples[2]))

        assert handle0_id == handle1_id == handle2_id

    def test_tar_handle_different_across_shards(self, temp_dir: Path) -> None:
        """Test that samples from different shards have different tar handles."""
        for i in range(2):
            parquet_path = temp_dir / f"{i:08d}.parquet"
            tar_path = temp_dir / f"{i:08d}.tar"
            create_test_parquet(parquet_path, num_rows=2)
            create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        samples = list(source)
        assert len(samples) == 4

        # Group samples by shard.
        shard0_samples = [s for s in samples if s["shard_index"] == 0]
        shard1_samples = [s for s in samples if s["shard_index"] == 1]

        # Within same shard, handles should be the same.
        assert id(_tar_handle(shard0_samples[0])) == id(
            _tar_handle(shard0_samples[1]),
        )
        assert id(_tar_handle(shard1_samples[0])) == id(
            _tar_handle(shard1_samples[1]),
        )

        # Across shards, handles should be different.
        assert id(_tar_handle(shard0_samples[0])) != id(
            _tar_handle(shard1_samples[0]),
        )


class TestParquetAndTarSourceShardErrors:
    """Test shard-error handling and shard-index parsing."""

    def test_non_integer_stem_raises(self, temp_dir: Path) -> None:
        """A non-integer parquet stem raises instead of colliding on index 0 (M23)."""
        parquet_path = temp_dir / "train-shard.parquet"
        tar_path = temp_dir / "train-shard.tar"
        create_test_parquet(parquet_path, num_rows=2)
        create_test_tar(tar_path, num_images=2)

        source = ParquetAndTarSource(ParquetAndTarSource.Config(working_dir=temp_dir))
        with pytest.raises(ValueError, match="not an integer shard index"):
            list(source)

    def test_fail_on_shard_error_raises(self, temp_dir: Path) -> None:
        """fail_on_shard_error surfaces an unreadable shard instead of skipping (M7)."""
        parquet_path = temp_dir / "00000000.parquet"
        parquet_path.write_bytes(b"not a parquet file")
        create_test_tar(temp_dir / "00000000.tar", num_images=1)

        source = ParquetAndTarSource(
            ParquetAndTarSource.Config(working_dir=temp_dir, fail_on_shard_error=True),
        )
        with pytest.raises((OSError, ValueError)):
            list(source)

    def test_shard_error_skipped_by_default(self, temp_dir: Path) -> None:
        """Without fail_on_shard_error an unreadable shard is skipped (default)."""
        bad = temp_dir / "00000000.parquet"
        bad.write_bytes(b"not a parquet file")
        create_test_tar(temp_dir / "00000000.tar", num_images=1)
        create_test_parquet(temp_dir / "00000001.parquet", num_rows=2)
        create_test_tar(temp_dir / "00000001.tar", num_images=2)

        source = ParquetAndTarSource(ParquetAndTarSource.Config(working_dir=temp_dir))
        samples = list(source)
        assert len(samples) == 2  # Only the good shard.


class TestParquetAndTarSourceReshuffle:
    """Test per-epoch reshuffling (epoch_seed folded in __iter__)."""

    @classmethod
    def _shard_order(cls, samples: list[dict[str, object]]) -> list[int]:
        """Shard indices in first-seen order (the shuffled shard ordering)."""
        order: list[int] = []
        for s in samples:
            idx = s["shard_index"]
            assert isinstance(idx, int)
            if idx not in order:
                order.append(idx)
        return order

    def _make_source(
        self,
        temp_dir: Path,
        num_shards: int,
        epoch_seed: int = 0,
    ) -> ParquetAndTarSource:
        for i in range(num_shards):
            create_test_parquet(temp_dir / f"{i:08d}.parquet", num_rows=1)
            create_test_tar(temp_dir / f"{i:08d}.tar", num_images=1)
        config = ParquetAndTarSource.Config(
            working_dir=temp_dir,
            shuffle=True,
            epoch_seed=epoch_seed,
            num_concurrently_read_shards=1,
        )
        return ParquetAndTarSource(config)

    def test_source_reshuffles_across_epochs(self, temp_dir: Path) -> None:
        """Distinct epoch seeds give different shard order, same shard set.

        Epoch state originates outside the source (the loader injects
        ``epoch_seed`` per epoch), so reshuffling is driven by the seed, not by
        re-iterating one instance.
        """
        source = self._make_source(temp_dir, num_shards=8, epoch_seed=0)
        order_a = self._shard_order(list(source))

        source.epoch_seed = 1
        order_b = self._shard_order(list(source))

        assert order_a != order_b  # Reshuffled across epochs.
        assert sorted(order_a) == sorted(order_b) == list(range(8))  # Same set.

    def test_per_epoch_order_identical_across_workers(self, temp_dir: Path) -> None:
        """All workers of one epoch share the seed: union is an exact partition."""
        num_shards = 8
        for i in range(num_shards):
            create_test_parquet(temp_dir / f"{i:08d}.parquet", num_rows=1)
            create_test_tar(temp_dir / f"{i:08d}.tar", num_images=1)

        num_workers = 4
        # Each worker is a fresh source on its first epoch (epoch_seed 0), so all
        # share the same permutation; the slices must partition exactly.
        union: list[int] = []
        for w in range(num_workers):
            config = ParquetAndTarSource.Config(
                working_dir=temp_dir,
                shuffle=True,
                worker_slice=(w, num_workers),
                num_concurrently_read_shards=1,
            )
            worker = ParquetAndTarSource(config)
            union.extend(self._shard_order(list(worker)))

        assert sorted(union) == list(range(num_shards))  # No gap, no dup.
        assert len(union) == num_shards


class TestParquetAndTarSourceLength:
    """Test __len__ method."""

    def test_len_returns_number_of_shards(self, temp_dir: Path) -> None:
        """Test that __len__ returns number of shards, not samples."""
        for i in range(3):
            parquet_path = temp_dir / f"{i:08d}.parquet"
            tar_path = temp_dir / f"{i:08d}.tar"
            create_test_parquet(parquet_path, num_rows=2)
            create_test_tar(tar_path, num_images=2)

        config = ParquetAndTarSource.Config(working_dir=temp_dir)
        source = ParquetAndTarSource(config)

        # Should return number of shards, not total samples.
        assert len(source) == 3


def _tar_handle(sample: dict[str, object]) -> TarFileHandle:
    """Return the typed tar handle recorded in a sample."""
    value = sample["_tar_handle"]
    assert isinstance(value, TarFileHandle)
    return value


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
