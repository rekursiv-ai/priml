from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import TypeGuard, cast
from unittest.mock import MagicMock, patch

from configgle import Fig
from torch import Tensor

import numpy as np
import pytest
import torch

from priml.data.pipeline.gpu_pipelining import (
    InferenceMode,
    PipelinedGPUProcessor2Stream,
    PipelinedGPUProcessor3Stream,
    Sample,
)


_GPUProcessor = PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream


_GPUProcessorConfig = (
    PipelinedGPUProcessor2Stream.Config | PipelinedGPUProcessor3Stream.Config
)


def _make[P: _GPUProcessor](
    cls: type[P],
    config: _GPUProcessorConfig | None = None,
) -> P:
    return cls(cls.Config() if config is None else config)


# Parametrize over both GPU processor implementations.
@pytest.fixture(params=[PipelinedGPUProcessor2Stream, PipelinedGPUProcessor3Stream])
def gpu_processor_class(
    request: pytest.FixtureRequest,
) -> type[PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream]:
    """Fixture that yields both GPU processor classes for testing."""
    # `request.param` is untyped by construction; the params list above fixes it.
    param = cast(
        type[PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream],
        request.param,
    )
    if param is PipelinedGPUProcessor2Stream:
        return PipelinedGPUProcessor2Stream
    assert param is PipelinedGPUProcessor3Stream
    return PipelinedGPUProcessor3Stream


# A bare ``isinstance(value, Mapping)`` narrows to ``Mapping[Unknown, Unknown]``, so the
# value type is lost at every nested step.
def _is_sample(value: object) -> TypeGuard[Sample]:
    """Narrow to a sample mapping, keeping the parameters both checkers need."""
    return isinstance(value, Mapping)


# These stages declare ``Mapping[str, object]`` because they add no keys of their own; a
# test that knows the payload's shape states that here rather than at every read.
def _at(sample: Sample, *path: str) -> object:
    """Read a nested field a stage moved without naming."""
    value: object = sample
    for step in path:
        assert _is_sample(value)
        value = value[step]
    return value


def _tensor_at(sample: Sample, *path: str) -> Tensor:
    """Read a nested field the specs converted to a tensor."""
    value = _at(sample, *path)
    assert isinstance(value, Tensor)
    return value


class PassthroughProcessor:
    """Processor that doesn't modify samples."""

    class Config(Fig["PassthroughProcessor"]): ...

    def __init__(self, config: Config) -> None:
        pass

    def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        """Pass through samples unchanged."""
        yield from samples


class SimpleProcessor:
    """Simple processor that adds a field."""

    class Config(Fig["SimpleProcessor"]): ...

    def __init__(self, config: Config) -> None:
        pass

    def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        """Add 'processed' field to each sample."""
        for sample in samples:
            assert isinstance(sample, dict)
            sample["processed"] = True
            yield sample


class IncrementProcessor:
    """Processor that increments a counter."""

    class Config(Fig["IncrementProcessor"]): ...

    def __init__(self, config: Config) -> None:
        pass

    def __call__(self, samples: Iterator[Sample]) -> Iterator[Sample]:
        """Increment counter in each sample."""
        for sample in samples:
            assert isinstance(sample, dict)
            counter = sample.get("counter", 0)
            assert isinstance(counter, int)
            sample["counter"] = counter + 1
            yield sample


def test_no_processors_passthrough(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Empty processors list should pass through samples unchanged."""
    config = gpu_processor_class.Config()
    processor = _make(gpu_processor_class, config)

    samples = [{"a": 1}, {"b": 2}]
    result = list(processor(iter(samples)))

    assert result == samples


def test_specs_apply_without_processors(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Transfer/transform specs are independent of the processor list.

    A config used only to move or reshape fields has no processors, and
    skipping the specs then makes it a silent no-op.
    """
    config = gpu_processor_class.Config()
    config.input_require_tensor = False
    config.output_require_tensor = False
    config.input_spec = {"a": _add_one}
    config.output_spec = {"a": _times_ten}

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        result = list(processor(iter([{"a": 1}])))

    assert result[0]["a"] == 20


def test_fields_the_stage_never_declared_ride_through(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """A stage moves whatever keys its processors chose, naming none itself.

    A closed field set here would name a shape neither stage controls, which
    is what forced a cast at every yield and at every call site.
    """
    config = gpu_processor_class.Config()
    processor = _make(gpu_processor_class, config)

    samples: list[Sample] = [{"unnamed_by_any_stage": 7}]
    result = list(processor(iter(samples)))

    assert result == samples


def test_output_spec_uses_its_own_require_tensor(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Each leg reads its OWN ``require_tensor`` flag.

    Sharing one dict between the two spec fields must not make the output leg
    adopt the input leg's setting.
    """
    shared: dict[str, Callable[[object], object]] = {"a": _times_ten}
    config = gpu_processor_class.Config()
    config.input_require_tensor = True
    config.output_require_tensor = False
    config.input_spec = shared
    config.output_spec = shared

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        result = list(processor(iter([{"a": 1}])))

    # Input leg is tensor-only so it skips the int; the output leg is not.
    assert result[0]["a"] == 10


def test_single_processor_cpu_fallback(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Single processor should work on CPU without CUDA."""
    config = gpu_processor_class.Config()
    config.processors.append(SimpleProcessor.Config())

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"a": 1}, {"b": 2}]
        result = list(processor(iter(samples)))

    assert len(result) == 2
    assert all(s["processed"] for s in result)
    assert result[0]["a"] == 1
    assert result[1]["b"] == 2


def test_multiple_processors_composition(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Multiple processors should compose left-to-right."""
    config = gpu_processor_class.Config()
    config.processors.append(IncrementProcessor.Config())
    config.processors.append(IncrementProcessor.Config())
    config.processors.append(IncrementProcessor.Config())

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"value": 0}]
        result = list(processor(iter(samples)))

    assert result[0]["counter"] == 3


def test_input_spec_default(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Default input spec should transfer all tensors to CUDA."""
    config = gpu_processor_class.Config()
    processor = _make(gpu_processor_class, config)

    # Default should be a non-blocking ``.to("cuda")`` applied to every tensor.
    assert "**" in processor.input_spec


def test_output_spec_default(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Default output spec should transfer all tensors to CPU."""
    config = gpu_processor_class.Config()
    processor = _make(gpu_processor_class, config)

    # Default should be a non-blocking ``.to("cpu")`` applied to every tensor.
    assert "**" in processor.output_spec


def test_input_spec_pattern_matching(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Input spec should only transform matching patterns."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"media_tensor": lambda x: torch.tensor(x).float()}
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"media_tensor": [1, 2, 3], "other": [4, 5, 6]}]
        result = list(processor(iter(samples)))

    assert isinstance(_at(result[0], "media_tensor"), Tensor)
    assert _tensor_at(result[0], "media_tensor").dtype == torch.float32
    assert result[0]["other"] == [4, 5, 6]  # Not transformed.


def test_output_spec_pattern_matching(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Output spec should only transform matching patterns."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"**": lambda x: x}
    config.output_spec = {"tensor_field": _tensor_to_list}

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [
            {
                "tensor_field": torch.tensor([1, 2, 3]),
                "other_tensor": torch.tensor([4, 5, 6]),
            },
        ]
        result = list(processor(iter(samples)))

    assert result[0]["tensor_field"] == [1, 2, 3]  # Transformed to list.
    assert isinstance(_at(result[0], "other_tensor"), Tensor)  # Not transformed.


def test_nested_pattern_matching(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Patterns should match nested fields."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"embeddings.*.features": torch.tensor}
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [
            {
                "embeddings": {
                    "clip": {"features": [1, 2, 3]},
                    "dino": {"features": [4, 5, 6]},
                },
            },
        ]
        result = list(processor(iter(samples)))

    assert isinstance(_at(result[0], "embeddings", "clip", "features"), Tensor)
    assert isinstance(_at(result[0], "embeddings", "dino", "features"), Tensor)


def test_wildcard_pattern_all(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Double wildcard should match all fields."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"**": lambda x: torch.tensor(x) if isinstance(x, list) else x}
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [
            {
                "a": [1, 2],
                "nested": {"b": [3, 4]},
            },
        ]
        result = list(processor(iter(samples)))

    assert isinstance(_at(result[0], "a"), Tensor)
    assert isinstance(_at(result[0], "nested", "b"), Tensor)


def test_empty_iterator(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Empty iterator should not crash."""
    config = gpu_processor_class.Config()
    config.processors.append(SimpleProcessor.Config())

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        result = list(processor(iter([])))

    assert result == []


def test_empty_iterator_with_cuda_mock(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Empty iterator should not crash with CUDA mocked."""
    config = gpu_processor_class.Config()
    config.processors.append(SimpleProcessor.Config())

    mock_stream = MagicMock()
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.Stream", return_value=mock_stream),
        patch("torch.cuda.synchronize"),
    ):
        processor = _make(gpu_processor_class, config)
        result = list(processor(iter([])))

    assert result == []


def test_gpu_path_with_mocks(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """GPU path should use streams correctly."""
    config = gpu_processor_class.Config()
    config.processors.append(SimpleProcessor.Config())

    # Different stream creation patterns for 2-stream vs 3-stream.
    if gpu_processor_class.__name__ == "PipelinedGPUProcessor2Stream":
        mock_streams = [MagicMock(), MagicMock()]  # Output, input.
    else:  # 3-stream.
        mock_streams = [MagicMock(), MagicMock(), MagicMock()]  # Input, work, output.

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.Stream", side_effect=mock_streams),
        patch("torch.cuda.synchronize") as mock_sync,
        patch("torch.cuda.stream"),
    ):
        processor = _make(gpu_processor_class, config)
        samples = [{"a": 1}, {"b": 2}]
        result = list(processor(iter(samples)))

    assert len(result) == 2
    assert all(s["processed"] for s in result)
    # Synchronize should be called at the end.
    mock_sync.assert_called()


def test_stream_synchronization(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Streams should be synchronized correctly."""
    config = gpu_processor_class.Config()
    config.processors.append(SimpleProcessor.Config())

    # Different stream creation patterns for 2-stream vs 3-stream.
    if gpu_processor_class.__name__ == "PipelinedGPUProcessor2Stream":
        mock_streams = [MagicMock(), MagicMock()]  # Output, input.
    else:  # 3-stream.
        mock_streams = [MagicMock(), MagicMock(), MagicMock()]  # Input, work, output.

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.Stream", side_effect=mock_streams),
        patch("torch.cuda.synchronize") as mock_sync,
        patch("torch.cuda.stream"),
    ):
        processor = _make(gpu_processor_class, config)
        samples = [{"a": 1}, {"b": 2}]
        list(processor(iter(samples)))

    # 2-stream uses wait_stream, 3-stream uses synchronize directly.
    if gpu_processor_class.__name__ == "PipelinedGPUProcessor2Stream":
        assert mock_streams[0].wait_stream.called
    # Final synchronize should be called.
    mock_sync.assert_called()


def test_require_tensor_input_spec(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Input spec should accept non-tensors (require_tensor=False)."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {
        "list_field": torch.tensor,  # Convert list to tensor.
    }
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"list_field": [1, 2, 3], "other": "string"}]
        result = list(processor(iter(samples)))

    assert isinstance(_at(result[0], "list_field"), Tensor)
    assert result[0]["other"] == "string"


def test_require_tensor_output_spec(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Output spec should only transform tensors (require_tensor=True)."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"**": lambda x: x}
    config.output_spec = {
        "**": _tensor_to_list,  # Only applied to tensors.
    }

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [
            {
                "tensor": torch.tensor([1, 2, 3]),
                "string": "hello",
                "number": 42,
            },
        ]
        result = list(processor(iter(samples)))

    assert result[0]["tensor"] == [1, 2, 3]  # Tensor transformed.
    assert result[0]["string"] == "hello"  # Non-tensor unchanged.
    assert result[0]["number"] == 42  # Non-tensor unchanged.


def test_nested_dict_transformation(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Deeply nested dicts should be transformed correctly."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"a.b.c.d": torch.tensor}
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"a": {"b": {"c": {"d": [1, 2, 3]}}}}]
        result = list(processor(iter(samples)))

    assert isinstance(_at(result[0], "a", "b", "c", "d"), Tensor)


def test_list_in_dict_transformation(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Lists within dicts should be accessible via numeric indices."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"items.0": torch.tensor}
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"items": [[1, 2], [3, 4]]}]
        result = list(processor(iter(samples)))

    items = _at(result[0], "items")
    assert isinstance(items, list)
    assert isinstance(items[0], Tensor)
    assert items[1] == [3, 4]  # Second item unchanged.


def test_in_place_modification(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Transformations should modify sample in-place."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"value": torch.tensor}
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        original = {"value": [1, 2, 3], "id": 123}
        samples = [original]
        result = list(processor(iter(samples)))

    # Result should be the same object (modified in-place)
    assert result[0] is original
    assert isinstance(_at(result[0], "value"), Tensor)
    assert result[0]["id"] == 123


def test_multiple_patterns_first_match_wins(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """When multiple patterns match, first one should win."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {
        "value": lambda x: torch.tensor(x).float(),
        "**": lambda x: torch.tensor(x).long(),
    }
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"value": [1, 2, 3]}]
        result = list(processor(iter(samples)))

    # First pattern should match and create float tensor.
    assert _tensor_at(result[0], "value").dtype == torch.float32


def test_dtype_conversion(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Input spec should handle dtype conversions."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {
        "**": lambda x: (
            torch.tensor(x, dtype=torch.float16) if isinstance(x, list) else x
        ),
    }
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"data": [1.0, 2.0, 3.0]}]
        result = list(processor(iter(samples)))

    assert _tensor_at(result[0], "data").dtype == torch.float16


def test_normalization_in_spec(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Input spec can include normalization operations."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {
        "image": lambda x: torch.tensor(x, dtype=torch.float32).div_(255.0),
    }
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"image": [255, 128, 0]}]
        result = list(processor(iter(samples)))

    assert torch.allclose(
        _tensor_at(result[0], "image"),
        torch.tensor([1.0, 128 / 255, 0.0]),
    )


def test_extraction_to_numpy(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Output spec can convert tensors to numpy."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"**": lambda x: x}
    config.output_spec = {"result": _tensor_to_numpy}

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"result": torch.tensor([1, 2, 3])}]
        result = list(processor(iter(samples)))

    assert isinstance(result[0]["result"], np.ndarray)


def test_pattern_prefix_match(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Pattern without wildcard should match path and children."""
    config = gpu_processor_class.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {
        "embeddings": lambda x: torch.tensor(x) if isinstance(x, list) else x,
    }
    config.input_require_tensor = False  # Allow transforming lists to tensors.

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [
            {
                "embeddings": {"clip": [1, 2], "dino": [3, 4]},
                "other": [5, 6],
            },
        ]
        result = list(processor(iter(samples)))

    # Both embeddings.clip and embeddings.dino should match prefix "embeddings"
    assert isinstance(_at(result[0], "embeddings", "clip"), Tensor)
    assert isinstance(_at(result[0], "embeddings", "dino"), Tensor)
    assert result[0]["other"] == [5, 6]  # Doesn't match prefix.


def test_single_item_iterator(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """Single item should be processed correctly."""
    config = gpu_processor_class.Config()
    config.processors.append(SimpleProcessor.Config())

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples = [{"value": 42}]
        result = list(processor(iter(samples)))

    assert len(result) == 1
    assert result[0]["processed"]
    assert result[0]["value"] == 42


def test_3stream_gpu_output_spec_preserves_user_transform() -> None:
    """3-stream CUDA path applies user output transform, not a forced .to(cpu).

    The previous implementation wrapped every output transform with an
    unconditional ``x.to("cpu", non_blocking=True)`` for tensors, discarding
    the user's dtype/conversion. Mock CUDA so the pipelined branch runs while
    tensors stay on CPU.
    """
    config = PipelinedGPUProcessor3Stream.Config()
    config.processors.append(PassthroughProcessor.Config())
    config.input_spec = {"**": lambda x: x}
    config.output_spec = {"result": _to_float64}

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.Stream", side_effect=[MagicMock(), MagicMock(), MagicMock()]),
        patch("torch.cuda.synchronize"),
        patch("torch.cuda.stream"),
    ):
        processor = PipelinedGPUProcessor3Stream(config)
        samples = [{"result": torch.tensor([1, 2, 3], dtype=torch.float32)}]
        result = list(processor(iter(samples)))

    assert _tensor_at(result[0], "result").dtype == torch.float64


def test_inference_mode() -> None:
    """InferenceMode wraps iterator with torch.inference_mode()."""
    samples = [{"a": 1}, {"b": 2}, {"c": 3}]
    inference_mode = InferenceMode(InferenceMode.Config())

    # Wrap samples with InferenceMode.
    wrapped = inference_mode(iter(samples))
    result = list(wrapped)

    assert len(result) == 3
    assert result[0] == {"a": 1}
    assert result[1] == {"b": 2}
    assert result[2] == {"c": 3}


def test_inference_mode_with_pipelined_processor(
    gpu_processor_class: type[
        PipelinedGPUProcessor2Stream | PipelinedGPUProcessor3Stream
    ],
) -> None:
    """InferenceMode can wrap PipelinedGPUProcessor."""
    config = gpu_processor_class.Config()
    config.processors.append(SimpleProcessor.Config())

    inference_mode = InferenceMode(InferenceMode.Config())

    with patch("torch.cuda.is_available", return_value=False):
        processor = _make(gpu_processor_class, config)
        samples: list[Sample] = [{"value": 42}]

        # Wrap with InferenceMode -> then PipelinedGPUProcessor.
        wrapped = inference_mode(iter(samples))
        result = list(processor(_sample_iterator(wrapped)))

    assert len(result) == 1
    assert result[0]["processed"]
    assert result[0]["value"] == 42


def _sample_iterator(samples: Iterator[object]) -> Iterator[Sample]:
    for sample in samples:
        assert _is_sample(sample)
        yield sample


def _add_one(value: object) -> object:
    assert isinstance(value, int)
    return value + 1


def _times_ten(value: object) -> object:
    assert isinstance(value, int)
    return value * 10


def _tensor_to_list(value: object) -> object:
    assert isinstance(value, Tensor)
    return value.tolist()


def _tensor_to_numpy(value: object) -> object:
    assert isinstance(value, Tensor)
    return value.cpu().numpy()


def _to_float64(value: object) -> object:
    assert isinstance(value, Tensor)
    return value.to(torch.float64)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
