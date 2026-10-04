"""Tests for tensorization and stream synchronization."""

from __future__ import annotations

from collections.abc import Generator, Iterator, Mapping
from typing import Protocol, cast

import contextlib

from torch import Tensor

import pytest
import torch

from priml.data.pipeline.tensorize import AsTensor, StreamSync


class TestAsTensor:
    """Tests for AsTensor processor."""

    def test_cpu_transfer(self):
        """Test basic CPU transfer without CUDA."""
        config = AsTensor.Config(device="cpu")
        processor = _make_processor(config)

        sample = {"tensor": torch.randn(3, 4), "label": 42}
        samples = processor(iter([sample]))
        result = next(samples)

        assert isinstance(result["tensor"], Tensor)
        assert _tensor(result, "tensor").device.type == "cpu"
        assert result["label"] == 42

    def test_dtype_conversion(self):
        """Test dtype conversion during transfer."""
        config = AsTensor.Config(device="cpu", dtype=torch.float16)
        processor = _make_processor(config)

        sample = {"tensor": torch.randn(3, 4, dtype=torch.float32)}
        samples = processor(iter([sample]))
        result = next(samples)

        assert _tensor(result, "tensor").dtype == torch.float16

    def test_exclude_fields(self):
        """Test excluding fields from transfer."""
        config = AsTensor.Config(device="cpu", exclude=["raw"])
        processor = _make_processor(config)

        sample = {
            "tensor": torch.randn(3, 4),
            "raw": torch.randn(5, 6),
        }
        samples = processor(iter([sample]))
        result = next(samples)

        assert "tensor" in result
        assert "raw" in result
        # Both should still be present, exclude just skips transfer.

    @pytest.mark.gpu_torch_cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_sync_transfer(self):
        """Test synchronous CUDA transfer."""
        config = AsTensor.Config(device="cuda", non_blocking=False)
        processor = _make_processor(config)

        sample = {"tensor": torch.randn(3, 4)}
        samples = processor(iter([sample]))
        result = next(samples)

        assert _tensor(result, "tensor").device.type == "cuda"
        # Should not have stream field for sync transfer.
        assert "pending_tensorizations" not in result

    @pytest.mark.gpu_torch_cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_async_transfer(self):
        """Test asynchronous CUDA transfer."""
        config = AsTensor.Config(device="cuda", non_blocking=True)
        processor = _make_processor(config)

        sample = {"tensor": torch.randn(3, 4)}
        samples = processor(iter([sample]))
        result = next(samples)

        assert _tensor(result, "tensor").device.type == "cuda"
        # Should have stream field for async transfer.
        assert "pending_tensorizations" in result
        assert isinstance(result["pending_tensorizations"], torch.cuda.Stream)

    @pytest.mark.gpu_torch_cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_stream_field_collision(self):
        """Test error when stream field already exists."""
        config = AsTensor.Config(device="cuda", non_blocking=True)
        processor = _make_processor(config)

        sample = {
            "tensor": torch.randn(3, 4),
            "pending_tensorizations": "existing_value",
        }

        with pytest.raises(
            TypeError,
            match=r"Field .pending_tensorizations. exists but is not",
        ):
            list(processor(iter([sample])))

    @pytest.mark.gpu_torch_cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_custom_stream_field_name(self):
        """Test custom stream field name."""
        config = AsTensor.Config(
            device="cuda",
            non_blocking=True,
            stream_field="my_stream",
        )
        processor = _make_processor(config)

        sample = {"tensor": torch.randn(3, 4)}
        samples = processor(iter([sample]))
        result = next(samples)

        assert "my_stream" in result
        assert "pending_tensorizations" not in result

    def test_nested_tensors(self):
        """Test transfer of nested tensor structures."""
        config = AsTensor.Config(device="cpu")
        processor = _make_processor(config)

        sample = {
            "embeddings": {
                "image": torch.randn(128),
                "text": torch.randn(256),
            },
            "label": 42,
        }
        samples = processor(iter([sample]))
        result = next(samples)

        assert _nested_tensor(result, "embeddings", "image").device.type == "cpu"
        assert _nested_tensor(result, "embeddings", "text").device.type == "cpu"
        assert result["label"] == 42

    def test_multiple_samples(self):
        """Test processing multiple samples."""
        config = AsTensor.Config(device="cpu", dtype=torch.float16)
        processor = _make_processor(config)

        samples = [{"tensor": torch.randn(3, 4), "id": i} for i in range(5)]
        results = list(processor(iter(samples)))

        assert len(results) == 5
        for i, result in enumerate(results):
            assert _tensor(result, "tensor").dtype == torch.float16
            assert result["id"] == i

    def test_list_field_tensorized_as_whole(self):
        """A list field becomes one tensor, not a list of 0-dim tensors (M5)."""
        config = AsTensor.Config(device="cpu")
        processor = _make_processor(config)

        sample = {"label": [1, 2, 3]}
        result = next(processor(iter([sample])))

        assert isinstance(result["label"], Tensor)
        label = _tensor(result, "label")
        assert label.shape == (3,)
        assert label.tolist() == [1, 2, 3]

    def test_nested_list_field_tensorized_as_whole(self):
        """A nested list field is tensorized once, not per-leaf (M5)."""
        config = AsTensor.Config(device="cpu")
        processor = _make_processor(config)

        sample = {"box": [[0, 0, 0], [1, 1, 1]]}
        result = next(processor(iter([sample])))

        assert isinstance(result["box"], Tensor)
        assert _tensor(result, "box").shape == (2, 3)

    def test_include_and_exclude_may_not_overlap(self):
        with pytest.raises(ValueError, match="cannot overlap"):
            _make_processor(AsTensor.Config(include=["a", "b"], exclude=["b"]))

    def test_include_limits_tensorization_to_the_named_fields(self):
        processor = _make_processor(AsTensor.Config(device="cpu", include=["label"]))

        result = next(processor(iter([{"label": [1, 2], "other": [3, 4]}])))

        assert _tensor(result, "label").tolist() == [1, 2]
        assert result["other"] == [3, 4]

    def test_tensorizes_in_place_without_a_device_or_dtype(self):
        processor = _make_processor(AsTensor.Config())

        result = next(processor(iter([{"label": [1, 2, 3]}])))

        label = _tensor(result, "label")
        assert label.dtype == torch.int64
        assert label.tolist() == [1, 2, 3]

    def test_async_cuda_transfer_creates_then_reuses_the_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        entered: list[object] = []

        @contextlib.contextmanager
        def fake_stream_scope(stream: object) -> Generator[None]:
            entered.append(stream)
            yield

        monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
        monkeypatch.setattr(torch.cuda, "stream", fake_stream_scope)
        # ``include`` names no real field, so nothing is moved to a device the
        # host lacks; the stream bookkeeping is the behavior under test.
        processor = _make_processor(
            AsTensor.Config(device="cuda", non_blocking=True, include=["none"]),
        )

        first = next(processor(iter([{"label": 1}])))
        stream = first["pending_tensorizations"]
        assert isinstance(stream, _FakeStream)
        assert first["label"] == 1

        second = next(processor(iter([{"pending_tensorizations": stream}])))
        assert second["pending_tensorizations"] is stream
        assert entered == [stream, stream]

    def test_async_cuda_transfer_rejects_a_foreign_stream_field(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
        processor = _make_processor(
            AsTensor.Config(device="cuda", non_blocking=True, include=["none"]),
        )

        with pytest.raises(
            TypeError,
            match=r"Field 'pending_tensorizations' exists but is not",
        ) as exc_info:
            list(processor(iter([{"pending_tensorizations": "existing"}])))
        assert str(exc_info.value) == (
            "Field 'pending_tensorizations' exists but is not a torch.cuda.Stream "
            "(got str)"
        )

    def test_non_blocking_cuda_transfers_the_sample_inside_its_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        entered: list[object] = []

        @contextlib.contextmanager
        def fake_stream_scope(stream: object) -> Generator[None]:
            entered.append(stream)
            yield

        monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
        monkeypatch.setattr(torch.cuda, "stream", fake_stream_scope)

        def fake_to(tensor: Tensor, **_: object) -> Tensor:
            return tensor

        monkeypatch.setattr(torch.Tensor, "to", fake_to)
        processor = _make_processor(
            AsTensor.Config(device="cuda", non_blocking=True, include=["label"]),
        )
        sample = {"label": [2, 3]}

        result = next(processor(iter([sample])))

        assert isinstance(result["label"], Tensor)
        assert result["label"].tolist() == [2, 3]
        assert entered == [result["pending_tensorizations"]]

    def test_non_blocking_cpu_does_not_create_a_cuda_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
        processor = _make_processor(
            AsTensor.Config(device=None, non_blocking=True, include=["none"]),
        )

        result = next(processor(iter([{"label": 42}])))

        assert result == {"label": 42}
        assert "pending_tensorizations" not in result

    def test_config_sets_and_overlap_error_are_exact(self):
        processor = _make_processor(
            AsTensor.Config(include=["selected", "other"], exclude=["raw"]),
        )
        assert processor.include == {"selected", "other"}
        assert processor.exclude == {"raw"}

        with pytest.raises(ValueError, match="overlap") as exc_info:
            _make_processor(
                AsTensor.Config(include=["shared", "selected"], exclude=["shared"]),
            )
        assert str(exc_info.value) == (
            "'include' and 'exclude' cannot overlap. Overlapping fields: {'shared'}"
        )

    def test_empty_filters_are_disabled(self):
        processor = _make_processor(
            AsTensor.Config(include=[], exclude=[], pin_memory=True),
        )

        assert processor.include is None
        assert processor.exclude is None
        assert processor.pin_memory is True
        result = next(processor(iter([{"label": [2, 3]}])))
        assert _tensor(result, "label").tolist() == [2, 3]

    def test_transfer_passes_device_dtype_and_non_blocking(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls: list[dict[str, object]] = []

        def record_transfer(
            tensor: Tensor,
            *,
            device: object = None,
            dtype: object = None,
            non_blocking: bool = False,
        ) -> Tensor:
            calls.append(
                {"device": device, "dtype": dtype, "non_blocking": non_blocking},
            )
            return tensor

        monkeypatch.setattr(torch.Tensor, "to", record_transfer)
        processor = _make_processor(
            AsTensor.Config(device="cpu", dtype=torch.float64, non_blocking=True),
        )
        result = next(processor(iter([{"label": [2, 3]}])))

        assert _tensor(result, "label").tolist() == [2, 3]
        assert calls == [
            {"device": "cpu", "dtype": torch.float64, "non_blocking": True},
        ]

    def test_omits_noop_transfer_when_device_and_dtype_are_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls: list[object] = []

        def record_transfer(tensor: Tensor, **kwargs: object) -> Tensor:
            calls.append(kwargs)
            return tensor

        monkeypatch.setattr(torch.Tensor, "to", record_transfer)
        processor = _make_processor(AsTensor.Config())

        result = next(processor(iter([{"label": [2, 3]}])))

        assert _tensor(result, "label").tolist() == [2, 3]
        assert calls == []

    def test_pins_before_cuda_transfer(self, monkeypatch: pytest.MonkeyPatch):
        pinned: list[Tensor] = []
        transfers: list[dict[str, object]] = []

        def record_pin(tensor: Tensor) -> Tensor:
            pinned.append(tensor)
            return tensor

        def record_transfer(
            tensor: Tensor,
            *,
            device: object = None,
            dtype: object = None,
            non_blocking: bool = False,
        ) -> Tensor:
            transfers.append(
                {"device": device, "dtype": dtype, "non_blocking": non_blocking},
            )
            return tensor

        monkeypatch.setattr(torch.Tensor, "pin_memory", record_pin)
        monkeypatch.setattr(torch.Tensor, "to", record_transfer)
        processor = _make_processor(
            AsTensor.Config(device="cuda", dtype=torch.float32, pin_memory=True),
        )

        result = next(processor(iter([{"label": [2, 3]}])))

        label = _tensor(result, "label")
        assert label.tolist() == [2.0, 3.0]
        assert pinned == [label]
        assert transfers == [
            {"device": "cuda", "dtype": torch.float32, "non_blocking": False},
        ]

    def test_dtype_only_transfer_and_field_filtering(self):
        processor = _make_processor(AsTensor.Config(dtype=torch.float64))
        sample = {
            "before": [2, 3],
            "raw": [4, 5],
            "after": [6, 7],
        }

        result = next(processor(iter([sample])))

        assert _tensor(result, "before").tolist() == [2.0, 3.0]
        assert _tensor(result, "before").dtype == torch.float64
        assert result["raw"] == [4, 5]
        assert _tensor(result, "after").tolist() == [6.0, 7.0]
        assert _tensor(result, "after").dtype == torch.float64

    def test_excluded_field_does_not_stop_later_fields(self):
        processor = _make_processor(
            AsTensor.Config(device="cpu", exclude=["raw"]),
        )
        sample = {"before": [2, 3], "raw": [4, 5], "after": [6, 7]}

        result = next(processor(iter([sample])))

        assert _tensor(result, "before").tolist() == [2, 3]
        assert result["raw"] == [4, 5]
        assert _tensor(result, "after").tolist() == [6, 7]

    def test_nested_mapping_does_not_stop_later_fields(self):
        processor = _make_processor(AsTensor.Config(device="cpu"))

        result = next(
            processor(iter([{"nested": {"value": [1, 2]}, "later": [3, 4]}])),
        )

        assert _nested_tensor(result, "nested", "value").tolist() == [1, 2]
        assert _tensor(result, "later").tolist() == [3, 4]

    def test_include_field_does_not_stop_later_fields(self):
        processor = _make_processor(
            AsTensor.Config(device="cpu", include=["after"]),
        )
        result = next(
            processor(iter([{"before": [2, 3], "after": [6, 7]}])),
        )

        assert result["before"] == [2, 3]
        assert _tensor(result, "after").tolist() == [6, 7]


class TestStreamSync:
    """Tests for StreamSync processor."""

    def test_no_stream_fields(self):
        """Test pass-through when no stream fields present."""
        config = StreamSync.Config()
        processor = _make_processor(config)

        sample = {"tensor": torch.randn(3, 4), "label": 42}
        samples = processor(iter([sample]))
        result = next(samples)

        assert "tensor" in result
        assert result["label"] == 42

    def test_empty_stream_fields_list(self):
        """Test optimization for empty stream fields list."""
        config = StreamSync.Config(stream_fields=[])
        processor = _make_processor(config)

        samples = [{"tensor": torch.randn(3, 4)}, {"tensor": torch.randn(2)}]
        results = list(processor(iter(samples)))

        assert results == samples

    def test_synchronizes_and_strips_every_named_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
        first, second = _FakeStream(), _FakeStream()
        sample: dict[str, object] = {
            "tensor": torch.randn(2),
            "s1": first,
            "s2": second,
        }
        processor = _make_processor(StreamSync.Config(stream_fields=["s1", "s2"]))

        result = next(processor(iter([sample])))

        assert first.synchronize_calls == 1
        assert second.synchronize_calls == 1
        assert set(result) == {"tensor"}

    @pytest.mark.gpu_torch_cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_sync_and_cleanup(self):
        """Test stream synchronization and cleanup."""
        # Create sample with async transfer.
        to_device = _make_processor(
            AsTensor.Config(device="cuda", non_blocking=True),
        )
        sample = {"tensor": torch.randn(3, 4)}
        transferred = next(to_device(iter([sample])))

        # Verify stream exists.
        assert "pending_tensorizations" in transferred

        # Sync and cleanup.
        sync = _make_processor(StreamSync.Config())
        result = next(sync(iter([transferred])))

        # Stream should be removed.
        assert "pending_tensorizations" not in result
        assert _tensor(result, "tensor").device.type == "cuda"

    @pytest.mark.gpu_torch_cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multiple_streams(self):
        """Test syncing multiple stream fields."""
        sample = {
            "tensor": torch.randn(3, 4).cuda(),
            "stream1": torch.cuda.Stream(),
            "stream2": torch.cuda.Stream(),
        }

        config = StreamSync.Config(stream_fields=["stream1", "stream2"])
        processor = _make_processor(config)

        result = next(processor(iter([sample])))

        assert "stream1" not in result
        assert "stream2" not in result
        assert "tensor" in result

    def test_invalid_stream_type(self):
        """Test error when field exists but isn't a Stream."""
        sample = {
            "tensor": torch.randn(3, 4),
            "pending_tensorizations": "not_a_stream",
        }

        config = StreamSync.Config()
        processor = _make_processor(config)

        with pytest.raises(
            TypeError,
            match=r"Field 'pending_tensorizations' exists but is not a torch\.cuda\.Stream",
        ) as exc_info:
            list(processor(iter([sample])))
        assert str(exc_info.value) == (
            "Field 'pending_tensorizations' exists but is not a torch.cuda.Stream "
            "(got str)"
        )

    def test_string_field_conversion(self):
        """Test that single string field is converted to list."""
        config = StreamSync.Config(stream_fields="my_stream")
        processor = _make_processor(config)

        assert processor.stream_fields == ["my_stream"]

    def test_list_field_preserved(self):
        """Test that list of fields is preserved."""
        config = StreamSync.Config(stream_fields=["stream1", "stream2"])
        processor = _make_processor(config)

        assert processor.stream_fields == ["stream1", "stream2"]

    def test_multiple_samples(self):
        """Test processing multiple samples."""
        samples = [{"tensor": torch.randn(3, 4), "id": i} for i in range(5)]

        config = StreamSync.Config()
        processor = _make_processor(config)

        results = list(processor(iter(samples)))
        assert len(results) == 5


class TestAsTensorStreamSyncIntegration:
    """Integration tests for AsTensor + StreamSync pipeline."""

    @pytest.mark.gpu_torch_cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_async_transfer_pipeline(self):
        """Test async transfer with explicit sync."""
        # Setup pipeline.
        to_device = _make_processor(
            AsTensor.Config(device="cuda", non_blocking=True),
        )
        sync = _make_processor(StreamSync.Config())

        # Create sample.
        sample = {"tensor": torch.randn(3, 4)}

        # Transfer (async)
        transferred = next(to_device(iter([sample])))
        assert "pending_tensorizations" in transferred

        # Sync.
        result = next(sync(iter([transferred])))
        assert "pending_tensorizations" not in result
        assert _tensor(result, "tensor").device.type == "cuda"

    @pytest.mark.gpu_torch_cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multiple_device_transfers(self):
        """Test transferring to different devices with different stream fields."""
        sample = {"tensor": torch.randn(3, 4)}

        # Transfer to cuda:0.
        to_cuda0 = _make_processor(
            AsTensor.Config(
                device="cuda:0",
                non_blocking=True,
                stream_field="stream_cuda0",
            ),
        )
        result = next(to_cuda0(iter([sample])))

        assert "stream_cuda0" in result
        assert _tensor(result, "tensor").device.type == "cuda"

        # Sync.
        sync = _make_processor(StreamSync.Config(stream_fields=["stream_cuda0"]))
        final = next(sync(iter([result])))

        assert "stream_cuda0" not in final


class _FakeStream:
    """Stands in for ``torch.cuda.Stream`` on a host without CUDA."""

    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _Processor(Protocol):
    def __call__(
        self,
        samples: Iterator[Mapping[str, object]],
    ) -> Iterator[dict[str, object]]: ...

    stream_fields: list[str]
    include: set[str] | None
    exclude: set[str] | None
    pin_memory: bool


def _make_processor(
    config: AsTensor.Config | StreamSync.Config,
) -> _Processor:
    return cast(_Processor, config.make())


def _tensor(sample: Mapping[str, object], field: str) -> Tensor:
    value = sample[field]
    assert isinstance(value, Tensor)
    return value


def _nested_tensor(
    sample: Mapping[str, object],
    outer_field: str,
    inner_field: str,
) -> Tensor:
    nested = sample[outer_field]
    assert isinstance(nested, dict)
    typed_nested = cast(Mapping[str, object], nested)
    value = typed_nested[inner_field]
    assert isinstance(value, Tensor)
    return value


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
