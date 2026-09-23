"""Tests for batching processors."""

from __future__ import annotations

from typing import cast

from torch import Tensor

import pytest
import torch

from priml.data.pipeline.batching import (
    Batcher,
    Unbatcher,
)
from priml.data.pipeline.parallel import (
    PrefetchBuffer,
)


# Batcher Tests.


def test_batcher_basic():
    """Test Batcher stacks samples with same shape."""
    config = Batcher.Config(size=2, field_names=["media_tensor"])
    batcher = Batcher(config)

    # Create samples with same shape.
    samples: list[dict[str, object]] = [
        {"id": 0, "media_tensor": torch.randn(1, 8, 8, 3)},
        {"id": 1, "media_tensor": torch.randn(1, 8, 8, 3)},
        {"id": 2, "media_tensor": torch.randn(1, 8, 8, 3)},
    ]

    results = list(batcher(iter(samples)))

    # Should produce 2 batches (size=2 + remaining 1)
    assert len(results) == 2

    # First batch.
    assert _tensor(results[0], "media_tensor").shape == (2, 1, 8, 8, 3)
    assert len(_raw(results[0])) == 2
    assert _raw(results[0])[0]["id"] == 0
    assert _raw(results[0])[1]["id"] == 1

    # Second batch (flush remainder)
    assert _tensor(results[1], "media_tensor").shape == (1, 1, 8, 8, 3)
    assert len(_raw(results[1])) == 1
    assert _raw(results[1])[0]["id"] == 2


def test_batcher_refuses_an_unparseable_device():
    """A device string torch cannot parse fails here, not silently as a no-op.

    ``None`` is the one spelling of "leave the batch where it is". A falsy
    string took that path too, so a typo -- or a value threaded in from a
    config whose device came back empty -- moved nothing and reported nothing,
    and the pipeline ran a GPU stage against CPU tensors.
    """
    with pytest.raises(RuntimeError):
        Batcher(Batcher.Config(size=1, device=""))


def test_batcher_takes_a_torch_device():
    """The batch device is spelled as every other one in priml is."""
    batcher = Batcher(Batcher.Config(size=1, device=torch.device("cpu")))
    assert batcher.device == torch.device("cpu")


def test_batcher_multi_aspect_ratio():
    """Test Batcher handles different shapes separately."""
    config = Batcher.Config(size=2, field_names=["media_tensor"])
    batcher = Batcher(config)

    # Create samples with different shapes.
    samples: list[dict[str, object]] = [
        {"id": 0, "media_tensor": torch.randn(1, 8, 8, 3)},
        {"id": 1, "media_tensor": torch.randn(1, 16, 16, 3)},
        {"id": 2, "media_tensor": torch.randn(1, 8, 8, 3)},
        {"id": 3, "media_tensor": torch.randn(1, 16, 16, 3)},
    ]

    results = list(batcher(iter(samples)))

    # Should produce 2 batches (one for each shape)
    assert len(results) == 2

    # Batches can come in any order, so we need to check by shape.
    batches_by_shape: dict[tuple[int, ...], Batcher.Output] = {}
    for batch in results:
        media_tensor_raw = batch["media_tensor"]
        assert isinstance(media_tensor_raw, Tensor)
        shape = tuple(media_tensor_raw.shape)
        batches_by_shape[shape] = batch

    # Verify 8x8 batch.
    assert (2, 1, 8, 8, 3) in batches_by_shape
    batch_8x8 = batches_by_shape[(2, 1, 8, 8, 3)]
    assert isinstance(_raw(batch_8x8), list)
    assert len(_raw(batch_8x8)) == 2
    assert {s["id"] for s in _raw(batch_8x8)} == {0, 2}

    # Verify 16x16 batch.
    assert (2, 1, 16, 16, 3) in batches_by_shape
    batch_16x16 = batches_by_shape[(2, 1, 16, 16, 3)]
    assert isinstance(_raw(batch_16x16), list)
    assert len(_raw(batch_16x16)) == 2
    assert {s["id"] for s in _raw(batch_16x16)} == {1, 3}


def test_batcher_filter_reasons():
    """Test Batcher passes through samples with filter_reasons."""
    config = Batcher.Config(size=2, field_names=["media_tensor"])
    batcher = Batcher(config)

    samples: list[dict[str, object]] = [
        {"id": 0, "media_tensor": torch.randn(1, 8, 8, 3)},
        {"id": 1, "filter_reasons": ["too_small"]},
        {"id": 2, "media_tensor": torch.randn(1, 8, 8, 3)},
    ]

    results = list(batcher(iter(samples)))

    # First result should be the filtered sample (passed through)
    assert results[0]["id"] == 1
    assert results[0]["filter_reasons"] == ["too_small"]
    assert "raw" not in results[0]

    # Second result should be the batched samples (flush remainder)
    assert _tensor(results[1], "media_tensor").shape == (2, 1, 8, 8, 3)
    assert len(_raw(results[1])) == 2


def test_batcher_missing_field_names():
    """Test Batcher passes through samples missing required fields."""
    config = Batcher.Config(size=2, field_names=["media_tensor"])
    batcher = Batcher(config)

    samples: list[dict[str, object]] = [
        {"id": 0, "media_tensor": torch.randn(1, 8, 8, 3)},
        {"id": 1, "caption": "no tensor here"},
        {"id": 2, "media_tensor": torch.randn(1, 8, 8, 3)},
    ]

    results = list(batcher(iter(samples)))

    # First result should be the sample without media_tensor (passed through)
    assert results[0]["id"] == 1
    assert "raw" not in results[0]

    # Second result should be the batched samples (flush remainder)
    assert _tensor(results[1], "media_tensor").shape == (2, 1, 8, 8, 3)
    assert len(_raw(results[1])) == 2


def test_batcher_drop_remainder():
    """Test Batcher drops incomplete batches when drop_remainder=True."""
    config = Batcher.Config(size=2, field_names=["media_tensor"], drop_remainder=True)
    batcher = Batcher(config)

    samples: list[dict[str, object]] = [
        {"id": 0, "media_tensor": torch.randn(1, 8, 8, 3)},
        {"id": 1, "media_tensor": torch.randn(1, 8, 8, 3)},
        {"id": 2, "media_tensor": torch.randn(1, 8, 8, 3)},
    ]

    results = list(batcher(iter(samples)))

    # Should only produce 1 batch (drop the remainder)
    assert len(results) == 1
    assert _tensor(results[0], "media_tensor").shape == (2, 1, 8, 8, 3)
    assert len(_raw(results[0])) == 2


def test_batcher_custom_field_names():
    """Test Batcher with multiple custom field names."""
    config = Batcher.Config(size=2, field_names=["image", "label"])
    batcher = Batcher(config)

    samples: list[dict[str, object]] = [
        {"id": 0, "image": torch.randn(3, 224, 224), "label": 5},
        {"id": 1, "image": torch.randn(3, 224, 224), "label": 7},
    ]

    results = list(batcher(iter(samples)))

    assert len(results) == 1
    assert _tensor(results[0], "image").shape == (2, 3, 224, 224)
    # Label is not a tensor, so it should be a list.
    assert results[0]["label"] == [5, 7]
    assert len(_raw(results[0])) == 2


def test_batcher_empty_input():
    """Test Batcher handles empty input."""
    config = Batcher.Config(size=2, field_names=["media_tensor"])
    batcher = Batcher(config)

    results = list(batcher(iter([])))
    assert len(results) == 0


def test_batcher_stacks_3d_tensors():
    """Test Batcher stacks 3D image tensors (H,W,C) to (B,H,W,C)."""
    config = Batcher.Config(size=2, field_names=["media_tensor"])
    batcher = Batcher(config)

    # Create image samples without frame dimension (H, W, C)
    samples: list[dict[str, object]] = [
        {"id": 0, "media_tensor": torch.randn(224, 224, 3)},
        {"id": 1, "media_tensor": torch.randn(224, 224, 3)},
    ]

    results = list(batcher(iter(samples)))

    # Should produce 1 batch with shape (B, H, W, C)
    assert len(results) == 1
    assert _tensor(results[0], "media_tensor").shape == (2, 224, 224, 3)
    assert len(_raw(results[0])) == 2

    # Verify raw samples exclude batched fields (media_tensor)
    assert "media_tensor" not in _raw(results[0])[0]
    assert "media_tensor" not in _raw(results[0])[1]
    assert _raw(results[0])[0]["id"] == 0
    assert _raw(results[0])[1]["id"] == 1


# Unbatcher Tests.


def test_unbatcher_basic():
    """Test Unbatcher distributes batch results to individuals."""
    config = Unbatcher.Config()
    unbatcher = Unbatcher(config)

    # Create a batched sample (raw excludes batched fields)
    batch: dict[str, object] = {
        "media_tensor": torch.randn(2, 1, 8, 8, 3),
        "embeddings": torch.randn(2, 512),
        "raw": [
            {"id": 0},
            {"id": 1},
        ],
    }

    results = list(unbatcher(iter([batch])))

    assert len(results) == 2
    assert results[0]["id"] == 0
    assert results[1]["id"] == 1
    # Each should have embeddings distributed (kept as tensors)
    assert isinstance(results[0]["embeddings"], Tensor)
    assert len(results[0]["embeddings"]) == 512
    assert isinstance(results[1]["embeddings"], Tensor)
    assert len(results[1]["embeddings"]) == 512


def test_unbatcher_with_dict_fields():
    """Test Unbatcher distributes dict fields (like embeddings keyed by frame)."""
    config = Unbatcher.Config()
    unbatcher = Unbatcher(config)

    # Create batch with dict field (frame-indexed embeddings, raw excludes batched fields)
    batch: dict[str, object] = {
        "clip_embeddings": {
            0: torch.randn(2, 768),  # 2 samples, each with 768-dim embedding.
        },
        "raw": [
            {"id": 0},
            {"id": 1},
        ],
    }

    results = list(unbatcher(iter([batch])))

    assert len(results) == 2
    # Each should have clip_embeddings distributed (kept as tensors)
    first_embeddings = _dict_field(results[0], "clip_embeddings")
    second_embeddings = _dict_field(results[1], "clip_embeddings")
    assert 0 in first_embeddings
    assert isinstance(first_embeddings[0], Tensor)
    assert len(first_embeddings[0]) == 768
    assert 0 in second_embeddings
    assert isinstance(second_embeddings[0], Tensor)
    assert len(second_embeddings[0]) == 768


def test_unbatcher_filter_reasons():
    """Test Unbatcher merges batch-level filter_reasons into samples."""
    config = Unbatcher.Config()
    unbatcher = Unbatcher(config)

    # Batch with filter reasons.
    batch: dict[str, object] = {
        "filter_reasons": ["batch_level_error"],
        "raw": [
            {"id": 0, "filter_reasons": ["sample_level_error"]},
            {"id": 1},
        ],
    }

    results = list(unbatcher(iter([batch])))

    assert len(results) == 2
    # First sample should have both filter reasons.
    first_reasons = _list_field(results[0], "filter_reasons")
    second_reasons = _list_field(results[1], "filter_reasons")
    assert "batch_level_error" in first_reasons
    assert "sample_level_error" in first_reasons
    # Second sample should have only batch-level filter reason.
    assert second_reasons == ["batch_level_error"]


def test_unbatcher_passthrough():
    """Test Unbatcher passes through non-batched samples."""
    config = Unbatcher.Config()
    unbatcher = Unbatcher(config)

    # Non-batched samples (no 'raw' field)
    samples: list[dict[str, object]] = [
        {"id": 0, "filter_reasons": ["filtered"]},
        {"id": 1, "value": 42},
    ]

    results = list(unbatcher(iter(samples)))

    assert len(results) == 2
    assert results[0] == {"id": 0, "filter_reasons": ["filtered"]}
    assert results[1] == {"id": 1, "value": 42}


def test_unbatcher_empty_input():
    """Test Unbatcher handles empty input."""
    config = Unbatcher.Config()
    unbatcher = Unbatcher(config)

    results = list(unbatcher(iter([])))
    assert len(results) == 0


def test_batcher_unbatcher_roundtrip():
    """Test Batcher + Unbatcher roundtrip preserves data."""
    batcher_config = Batcher.Config(size=2, field_names=["media_tensor"])
    unbatcher_config = Unbatcher.Config()

    batcher = Batcher(batcher_config)
    unbatcher = Unbatcher(unbatcher_config)

    # Original samples.
    samples: list[dict[str, object]] = [
        {"id": 0, "media_tensor": torch.randn(1, 8, 8, 3), "caption": "cat"},
        {"id": 1, "media_tensor": torch.randn(1, 8, 8, 3), "caption": "dog"},
    ]

    # Batch.
    batches = list(batcher(iter(samples)))
    assert len(batches) == 1

    # Unbatch.
    results = list(unbatcher(iter(batches)))

    # Should get back original samples (without media_tensor since it's excluded)
    assert len(results) == 2
    assert results[0]["id"] == 0
    assert results[0]["caption"] == "cat"
    assert results[1]["id"] == 1
    assert results[1]["caption"] == "dog"


def test_unbatcher_passes_through_batch_level_metadata_vector():
    """A (B,) batch-level field is replicated, not split into B scalars (H4)."""
    config = Unbatcher.Config()
    unbatcher = Unbatcher(config)

    batch: dict[str, object] = {
        "_batch_size": 2,
        "scalar_meta": torch.tensor([10.0, 20.0]),  # (B,) but NOT per-sample.
        "embeddings": torch.randn(2, 512),  # Genuinely per-sample.
        "raw": [{"id": 0}, {"id": 1}],
    }

    results = list(unbatcher(iter([batch])))

    assert len(results) == 2
    # Per-sample field is split.
    assert _tensor(results[0], "embeddings").shape == (512,)
    # Batch-level (B,) vector is passed through whole to each sample.
    assert isinstance(results[0]["scalar_meta"], Tensor)
    assert isinstance(results[1]["scalar_meta"], Tensor)
    assert torch.equal(results[0]["scalar_meta"], torch.tensor([10.0, 20.0]))
    assert torch.equal(results[1]["scalar_meta"], torch.tensor([10.0, 20.0]))


def test_unbatcher_strips_batch_size_marker():
    """The _batch_size marker is not leaked into emitted per-sample dicts."""
    config = Unbatcher.Config()
    unbatcher = Unbatcher(config)

    batch: dict[str, object] = {
        "_batch_size": 2,
        "embeddings": torch.randn(2, 4),
        "raw": [{"id": 0}, {"id": 1}],
    }

    results = list(unbatcher(iter([batch])))

    assert all("_batch_size" not in r for r in results)


def test_unbatcher_does_not_split_batchlevel_list_of_batch_len():
    """A batch-level list of length B survives intact on every unbatched sample.

    Red test for #323: the old list branch split ANY length-B list via
    value[idx], sharding shared batch-level metadata across samples.
    """
    config = Unbatcher.Config()
    unbatcher = Unbatcher(config)

    shared_meta = ["a", "b"]  # batch-level list that happens to have length B=2.
    batch: dict[str, object] = {
        "_batch_size": 2,
        "shared_meta": shared_meta,  # NOT a Batcher per-sample field.
        "embeddings": torch.randn(2, 512),  # Genuinely per-sample.
        "raw": [{"id": 0}, {"id": 1}],
    }

    results = list(unbatcher(iter([batch])))

    assert len(results) == 2
    # Per-sample tensor is split.
    assert _tensor(results[0], "embeddings").shape == (512,)
    # Batch-level list is replicated whole, not sharded into value[idx].
    assert results[0]["shared_meta"] == ["a", "b"]
    assert results[1]["shared_meta"] == ["a", "b"]


def test_batcher_unbatcher_roundtrip_splits_tagged_list_field():
    """Batcher-stacked non-tensor per-sample lists still split on unbatch."""
    batcher = Batcher(Batcher.Config(size=2, field_names=["image", "label"]))
    unbatcher = Unbatcher(Unbatcher.Config())

    samples: list[dict[str, object]] = [
        {"id": 0, "image": torch.randn(3, 4, 4), "label": 5},
        {"id": 1, "image": torch.randn(3, 4, 4), "label": 7},
    ]

    batches = list(batcher(iter(samples)))
    assert len(batches) == 1
    # The Batcher tagged 'label' as a per-sample list field.
    assert batches[0]["_batched_list_fields"] == ["label"]

    results = list(unbatcher(iter(batches)))

    assert len(results) == 2
    # Tagged per-sample list field is split back to scalars.
    assert results[0]["label"] == 5
    assert results[1]["label"] == 7


def test_batcher_is_reentrant_across_calls():
    """Batcher holds no per-call queue state; concurrent generators don't share (H3)."""
    config = Batcher.Config(size=2, field_names=["media_tensor"])
    batcher = Batcher(config)

    samples_a: list[dict[str, object]] = [
        {"id": i, "media_tensor": torch.randn(1, 8, 8, 3)} for i in range(3)
    ]
    samples_b: list[dict[str, object]] = [
        {"id": i, "media_tensor": torch.randn(1, 8, 8, 3)} for i in range(3)
    ]

    gen_a = batcher(iter(samples_a))
    gen_b = batcher(iter(samples_b))

    # Interleave: pull first full batch from A, then drive B fully, then finish A.
    first_a = next(gen_a)
    all_b = list(gen_b)
    rest_a = list(gen_a)

    assert len(_raw(first_a)) == 2
    assert len(all_b) == 2  # Full batch + flushed remainder.
    assert len(rest_a) == 1  # Flushed remainder of A, unaffected by B.
    assert _raw(rest_a[0])[0]["id"] == 2


@pytest.mark.parametrize(
    "labels",
    [(torch.zeros(2), 7), (7, torch.zeros(2))],
    ids=["tensor_then_scalar", "scalar_then_tensor"],
)
def test_batcher_rejects_a_field_mixing_tensors_and_scalars(
    labels: tuple[object, object],
):
    """``__call__`` keys queues by tensor shape, so ``_create_batch`` is the
    only entry point that can receive a mixed group; it must still refuse it.
    """
    batcher = Batcher(Batcher.Config(size=2, field_names=["label"]))

    samples: list[dict[str, object]] = [
        {"id": 0, "label": labels[0]},
        {"id": 1, "label": labels[1]},
    ]

    with pytest.raises(TypeError, match="mixed tensor/non-tensor"):
        batcher._create_batch(samples)


def test_batcher_moves_the_stacked_tensor_to_the_configured_device():
    batcher = Batcher(Batcher.Config(size=2, device="cpu"))

    samples: list[dict[str, object]] = [
        {"id": i, "media_tensor": torch.randn(2, 2)} for i in range(2)
    ]

    results = list(batcher(iter(samples)))

    stacked = _tensor(results[0], "media_tensor")
    assert stacked.device == torch.device("cpu")
    assert stacked.shape == (2, 2, 2)


def test_batcher_omits_raw_when_the_stacked_fields_were_everything():
    batcher = Batcher(Batcher.Config(size=2))

    samples: list[dict[str, object]] = [{"media_tensor": torch.zeros(2)}] * 2

    results = list(batcher(iter(samples)))

    assert "raw" not in results[0]
    assert results[0]["_batch_size"] == 2


def test_batcher_logs_a_slow_stack(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    tick = 0.0

    def slow_clock() -> float:
        nonlocal tick
        tick += 0.2
        return tick

    monkeypatch.setattr(
        "priml.data.pipeline.batching.time.perf_counter",
        slow_clock,
    )
    batcher = Batcher(Batcher.Config(size=1))
    caplog.set_level("DEBUG", logger="priml.data.pipeline.batching")

    samples: list[dict[str, object]] = [{"media_tensor": torch.zeros(2)}]
    _ = list(batcher(iter(samples)))

    assert "Batcher: batch_size=1" in caplog.text
    assert "stack=200.0ms" in caplog.text


_MALFORMED_BATCHES: list[tuple[dict[str, object], str]] = [
    ({"raw": "nope"}, "raw must be a list"),
    ({"raw": ["nope"]}, "raw must be a list"),
    ({"raw": [{}], "_batch_size": "1"}, "_batch_size must be an integer"),
    ({"raw": [{}], "filter_reasons": 5}, "filter_reasons must be iterable"),
    ({"raw": [{}], "_batched_list_fields": "x"}, "_batched_list_fields"),
    ({"raw": [{}], "_batched_list_fields": [1]}, "_batched_list_fields"),
]


@pytest.mark.parametrize(
    ("batch", "message"),
    _MALFORMED_BATCHES,
    ids=[
        "raw_not_list",
        "raw_item_not_dict",
        "batch_size_not_int",
        "filter_reasons_not_iterable",
        "list_fields_not_list",
        "list_fields_not_str",
    ],
)
def test_unbatcher_rejects_a_malformed_batch(batch: dict[str, object], message: str):
    unbatcher = Unbatcher(Unbatcher.Config())

    with pytest.raises(TypeError, match=message):
        list(unbatcher(iter([batch])))


def test_unbatcher_replaces_a_non_list_sample_filter_reasons_field():
    unbatcher = Unbatcher(Unbatcher.Config())

    batch: dict[str, object] = {
        "filter_reasons": ["batch_level"],
        "raw": [{"id": 0, "filter_reasons": "corrupt"}],
    }

    results = list(unbatcher(iter([batch])))

    assert results[0]["filter_reasons"] == ["batch_level"]


def test_unbatcher_replicates_dict_values_that_carry_no_batch_axis():
    unbatcher = Unbatcher(Unbatcher.Config())

    batch: dict[str, object] = {
        "_batch_size": 2,
        "scores": {"per_sample": torch.arange(4.0).reshape(2, 2), "scale": 0.5},
        "raw": [{"id": 0}, {"id": 1}],
    }

    results = list(unbatcher(iter([batch])))

    first = _dict_field(results[0], "scores")
    second = _dict_field(results[1], "scores")
    assert first["scale"] == 0.5
    assert second["scale"] == 0.5
    assert isinstance(second["per_sample"], Tensor)
    assert second["per_sample"].tolist() == [2.0, 3.0]


def test_unbatcher_has_batch_dimension_rejects_non_arrays():
    unbatcher = Unbatcher(Unbatcher.Config())

    assert not unbatcher._has_batch_dimension([1, 2], 2)
    assert not unbatcher._has_batch_dimension(torch.zeros(2), 2)
    assert unbatcher._has_batch_dimension(torch.zeros(2, 1), 2)


# PrefetchBuffer Tests
# (These are from the parallel module, but kept here for historical reasons)


def test_prefetch_buffer():
    """Test PrefetchBuffer prefetches samples."""
    config = PrefetchBuffer.Config()
    config.size = 10
    prefetcher = config.make()

    samples: list[dict[str, object]] = [
        {"key": f"test_{i}", "value": i} for i in range(5)
    ]

    results = list(prefetcher(iter(samples)))

    assert len(results) == 5
    for i, result in enumerate(results):
        assert result["key"] == f"test_{i}"
        assert result["value"] == i


def _tensor(batch: Batcher.Output, name: str) -> Tensor:
    """Narrow a dynamic processor field to the tensor this test expects."""
    value = batch[name]
    assert isinstance(value, Tensor)
    return value


def _raw(batch: Batcher.Output) -> list[dict[str, object]]:
    """Narrow the dynamic raw field to the records this test expects."""
    value = batch["raw"]
    assert isinstance(value, list)
    records: list[dict[str, object]] = []
    for raw_record in cast(list[object], value):
        assert isinstance(raw_record, dict)
        record = cast(dict[str, object], raw_record)
        assert all(isinstance(key, str) for key in record)
        records.append(record)
    return records


def _dict_field(batch: Unbatcher.Output, name: str) -> dict[object, object]:
    """Narrow a dynamic processor field to a mapping."""
    value = batch[name]
    assert isinstance(value, dict)
    return cast(dict[object, object], value)


def _list_field(batch: Unbatcher.Output, name: str) -> list[object]:
    """Narrow a dynamic processor field to a list."""
    value = batch[name]
    assert isinstance(value, list)
    return cast(list[object], value)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
