"""Check numerical tracing, the recipe fixtures and the miniature."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import inspect
import sys

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm import testing
from priml.baselines.name_that_llm.calibration import SigmoidCalibration
from priml.baselines.name_that_llm.model import SvmBranch
from priml.baselines.name_that_llm.sparse import CsrMatrix
from priml.baselines.name_that_llm.standardization import StandardScaler
from priml.baselines.name_that_llm.svm import LinearSVC
from priml.baselines.name_that_llm.testing import (
    NumericalTrace,
    centred_branch,
    completed_record,
    float32_comparands,
    miniature,
    miniature_split,
    plain_branch,
    recipe,
    svm_branch,
)
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Generator
    from types import CodeType, FrameType

    from numpy.typing import NDArray


def test_recipe_is_exp000_with_saved_layers_resolved() -> None:
    config = recipe()
    assert config.experiment_name == "exp000"
    assert plain_branch().layers == [0, 1, 2, 3]
    assert centred_branch().heads.layers == [0, 1, 2, 3]


def test_miniature_has_three_classes_in_six_prompt_groups() -> None:
    texts, targets, groups, embeddings = miniature()
    assert len(texts) == len(targets) == len(groups) == len(embeddings) == 18
    assert np.array_equal(np.unique(targets), [0, 1, 2])
    assert len(np.unique(groups)) == 6
    assert embeddings.shape == (18, 4, 2)


def test_trace_captures_fitted_scaler_and_restores_profile() -> None:
    features = np.arange(15, dtype=np.float64).reshape(3, 5)
    with NumericalTrace() as trace:
        StandardScaler.Config().make().fit(features)
    assert torch.equal(
        trace.record["StandardScaler/0/mean_"],
        other=torch.from_numpy(features.mean(axis=0)),
    )
    assert torch.equal(
        trace.record["StandardScaler/0/scale_"],
        other=torch.from_numpy(features.std(axis=0)),
    )


def test_trace_captures_continuing_permutations_and_call_counts() -> None:
    expected = np.random.default_rng(42)
    with NumericalTrace() as trace:
        actual = np.random.default_rng(42)
        first = actual.permutation(np.arange(7))
        second = actual.permutation(np.arange(7))
    assert np.array_equal(first, a2=expected.permutation(np.arange(7)))
    assert np.array_equal(second, a2=expected.permutation(np.arange(7)))
    assert trace.counts == {"permutation": 2}
    assert torch.equal(trace.record["rng/permutation/0"], other=torch.from_numpy(first))
    assert torch.equal(
        trace.record["rng/permutation/1"],
        other=torch.from_numpy(second),
    )


def test_trace_captures_all_fitted_coefficients() -> None:
    config = LinearSVC.Config()
    config.optimizer.max_iter = 3
    with NumericalTrace() as trace:
        model = config.make().fit(
            CsrMatrix.from_dense(np.arange(35, dtype=np.float64).reshape(5, 7)),
            targets=np.array([0, 1, 2, 0, 1]),
        )
    assert trace.record["LinearSVC/0/coef_"].shape == (3, 7)
    assert trace.record["LinearSVC/0/intercept_"].shape == (3,)
    replay = NumericalTrace()
    replay._observe(model)
    assert not mismatches(trace.record, actual=replay.record)


def test_trace_preserves_nested_numeric_values() -> None:
    trace = NumericalTrace()
    trace.put("tensor", value=torch.tensor([1.0], dtype=torch.float64))
    trace.put("nested", value={"rows": [np.arange(3), (2, 3.5)]})
    trace.put("ignored", value="word")
    with pytest.raises(TypeError, match="Unsupported array dtype"):
        trace.put("unsupported", value=np.array(["word"]))
    assert torch.equal(
        trace.record["tensor"],
        other=torch.tensor([1.0], dtype=torch.float64),
    )
    assert trace.record["nested/rows/1/1"].dtype == torch.float64
    assert set(trace.record) == {
        "tensor",
        "nested/rows/0",
        "nested/rows/1/0",
        "nested/rows/1/1",
    }


def test_trace_restores_an_existing_profiler_and_observes_fits() -> None:
    previous = sys.getprofile()
    sys.setprofile(_noop_profile)
    try:
        with NumericalTrace() as trace:
            assert trace.previous is _noop_profile
        assert sys.getprofile() is _noop_profile
    finally:
        sys.setprofile(previous)
    scaler = StandardScaler.Config().make()
    scaler.fit(np.arange(12, dtype=np.float64).reshape(4, 3))
    trace._observe(scaler)
    assert trace.counts["StandardScaler"] == 1


def test_trace_restores_the_profiler_present_at_entry() -> None:
    previous = sys.getprofile()
    trace = NumericalTrace()
    assert trace.previous is None
    sys.setprofile(_noop_profile)
    try:
        with trace:
            pass
        assert sys.getprofile() is _noop_profile
    finally:
        sys.setprofile(previous)


def test_trace_ignores_returns_other_than_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = inspect.currentframe()
    assert frame is not None
    trace = NumericalTrace()
    monkeypatch.setattr(trace, "_observe", value=_unexpected_observation)
    trace._returned(frame.f_code, offset=0, value=None)
    trace._unwound(frame.f_code, offset=0, exception=ValueError())


def test_return_observation_coexists_with_existing_profile_and_monitoring() -> None:
    previous = sys.getprofile()
    frames: list[FrameType] = []
    received: list[object] = []
    profiler = partial(_record_fit_frame, frames=frames)
    code = _fit_scaler.__code__
    tool = next(index for index in range(6) if sys.monitoring.get_tool(index) is None)
    sys.monitoring.use_tool_id(tool, "existing-llm-test-observer")
    sys.monitoring.register_callback(
        tool,
        sys.monitoring.events.PY_RETURN,
        partial(_record_return, received=received),
    )
    sys.monitoring.set_local_events(tool, code, sys.monitoring.events.PY_RETURN)
    tools = [sys.monitoring.get_tool(index) for index in range(6)]
    sys.setprofile(profiler)
    trace = NumericalTrace()
    try:
        with trace:
            _fit_scaler(np.arange(15, dtype=np.float64).reshape(3, 5))
            assert sys.getprofile() is profiler
        assert len(frames) == 1
        assert received == [frames[0].f_locals["self"]]
        assert [sys.monitoring.get_tool(index) for index in range(6)] == tools
        assert (
            sys.monitoring.get_local_events(tool, code)
            == sys.monitoring.events.PY_RETURN
        )
        assert torch.equal(
            trace.record["StandardScaler/0/mean_"],
            other=torch.arange(5) + 5,
        )
    finally:
        sys.setprofile(previous)
        sys.monitoring.set_local_events(tool, code, 0)
        sys.monitoring.register_callback(tool, sys.monitoring.events.PY_RETURN, None)
        sys.monitoring.free_tool_id(tool)


def test_trace_preserves_factory_inputs_and_complete_generator_operations() -> None:
    factory = np.random.default_rng
    expected = factory(42)
    with NumericalTrace() as trace:
        assert isinstance(np.random.default_rng(), np.random.Generator)
        actual = np.random.default_rng(seed=np.random.SeedSequence(42))
        assert np.random.default_rng(actual) is actual
        assert np.array_equal(actual.random(3), a2=expected.random(3))
        assert np.array_equal(
            actual.integers(2, high=13, size=5),
            a2=expected.integers(2, high=13, size=5),
        )
        assert np.array_equal(actual.normal(size=7), a2=expected.normal(size=7))
        assert np.array_equal(actual.permutation(11), a2=expected.permutation(11))
        assert np.array_equal(
            actual.permutation(np.arange(12).reshape(3, 4), axis=1),
            a2=expected.permutation(np.arange(12).reshape(3, 4), axis=1),
        )
        assert actual.bit_generator.state == expected.bit_generator.state
    assert trace.observing is False
    assert np.array_equal(actual.permutation(13), a2=expected.permutation(13))
    assert actual.bit_generator.state == expected.bit_generator.state
    assert trace.counts == {"permutation": 2}


@pytest.mark.parametrize("dtype", argvalues=["S4", "complex64", "complex128"])
def test_trace_rejects_unsupported_array_kinds_explicitly(dtype: str) -> None:
    with pytest.raises(TypeError, match="Unsupported array dtype"):
        NumericalTrace().put("unsupported", value=np.array([1, 2], dtype=dtype))


def test_trace_restores_generators_and_monitoring_after_entry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = np.random.default_rng
    previous = sys.getprofile()
    tools = [sys.monitoring.get_tool(index) for index in range(6)]
    trace = NumericalTrace()
    monkeypatch.setattr(trace, "_enable_returns", value=_unavailable_monitoring)
    with pytest.raises(RuntimeError, match=r"^monitoring unavailable$"), trace:
        pytest.fail("Failed trace entry must not enter the caller body.")
    assert trace.observing is False
    assert np.random.default_rng is factory
    assert sys.getprofile() is previous
    assert [sys.monitoring.get_tool(index) for index in range(6)] == tools


def test_trace_rejects_generator_creation_outside_an_active_context() -> None:
    with pytest.raises(ValueError, match=r"^Expected self\.hooks is not None\.$"):
        NumericalTrace()._generator(42)


def test_trace_observes_a_native_fit_unwind_and_restores_global_state() -> None:
    factory = np.random.default_rng
    with (
        np.errstate(invalid="raise"),
        NumericalTrace() as trace,
        pytest.raises(FloatingPointError),
    ):
        StandardScaler.Config().make().fit(np.full((2, 3), np.inf))
    assert trace.counts == {"StandardScaler": 1}
    assert trace.record["StandardScaler/0/n_features_in_"].item() == 3
    assert np.random.default_rng is factory


def test_return_dispatch_preserves_actual_fit_state_and_unwind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames: list[FrameType] = []
    previous = sys.getprofile()
    sys.setprofile(partial(_record_fit_frame, frames=frames))
    try:
        model = (
            StandardScaler.Config()
            .make()
            .fit(
                np.arange(15, dtype=np.float64).reshape(3, 5),
            )
        )
    finally:
        sys.setprofile(previous)
    assert len(frames) == 1
    monkeypatch.setattr(
        testing,
        "frame_for",
        value=partial(_saved_frame, frame=frames[0]),
    )
    trace = NumericalTrace()
    trace._returned(frames[0].f_code, offset=0, value=model)
    trace._unwound(frames[0].f_code, offset=0, exception=ValueError())
    assert trace.counts == {"StandardScaler": 2}
    assert torch.equal(
        trace.record["StandardScaler/0/mean_"],
        other=torch.arange(5) + 5,
    )
    assert torch.equal(
        trace.record["StandardScaler/0/mean_"],
        other=trace.record["StandardScaler/1/mean_"],
    )


def test_frame_resolution_returns_the_live_operation_frame() -> None:
    frame = inspect.currentframe()
    assert frame is not None
    assert testing.frame_for(frame.f_code) is frame


def test_trace_records_sigmoid_calibration_under_its_class_name() -> None:
    model = (
        SigmoidCalibration.Config()
        .make()
        .fit(
            np.arange(7, dtype=np.float64),
            targets=np.array([0, 1, 0, 1, 0, 1, 0]),
        )
    )
    trace = NumericalTrace()
    trace._observe(model)
    assert trace.counts == {"SigmoidCalibration": 1}
    assert set(trace.record) == {
        "SigmoidCalibration/0/a_",
        "SigmoidCalibration/0/b_",
    }
    assert trace.record["SigmoidCalibration/0/a_"].item() == model.a_
    assert trace.record["SigmoidCalibration/0/b_"].item() == model.b_


def test_svm_branch_is_exp000s_calibrated_lexical_branch() -> None:
    assert isinstance(svm_branch(), SvmBranch.Config)


def test_miniature_split_predicts_its_first_three_rows_in_two_clusters() -> None:
    split = miniature_split()
    texts, _, _, embeddings = miniature()
    assert list(split.predict_texts) == list(texts[:3])
    assert np.array_equal(split.predict_embeddings, embeddings[:3])
    assert split.clusters.tolist() == [2, 2, 3]


def test_completed_record_appends_outputs_and_counts() -> None:
    trace = NumericalTrace()
    trace.counts["draws"] = 3
    record = completed_record(trace, {"out": torch.arange(3)})
    assert record.keys() == {"out", "counts/draws"}
    assert record["out"].tolist() == [0, 1, 2]
    assert record["counts/draws"].item() == 3


def test_float32_comparands_round_only_float64_values() -> None:
    record = float32_comparands(
        {
            "wide": torch.tensor([1 + 2**-30, 3], dtype=torch.float64),
            "narrow": torch.tensor([1 + 2**-23], dtype=torch.float32),
            "count": torch.tensor(7),
        },
    )
    assert record["wide"].dtype == torch.float64
    assert record["wide"].tolist() == [1, 3]
    assert record["narrow"].dtype == torch.float32
    assert record["narrow"].tolist() == [1 + 2**-23]
    assert record["count"].dtype == torch.int64


def test_miniature_is_three_classes_over_six_named_prompt_groups() -> None:
    texts, targets, groups, embeddings = miniature()
    assert texts.dtype == object
    assert texts[:4].tolist() == ["a0 b0", "a1 b0", "a2 b0", "a0 b1"]
    assert targets.dtype == np.int64
    assert targets.tolist() == [0, 1, 2] * 6
    assert groups.tolist() == [name for name in "abcdef" for _ in range(3)]
    assert embeddings.dtype == np.float32
    # Row 1 is class 1; its first value is element 8 of the sine sequence.
    assert embeddings[1, 0, 0] == np.float32(np.sin(8 / 5) + 0.5)


def test_miniature_split_carries_the_miniature_rows_unchanged() -> None:
    split = miniature_split()
    texts, targets, groups, embeddings = miniature()
    assert split.texts.tolist() == texts.tolist()
    assert np.array_equal(split.targets, targets)
    assert np.array_equal(split.groups, groups)
    assert np.array_equal(split.embeddings, embeddings)


@pytest.mark.parametrize(
    "kind",
    [
        "StandardScaler",
        "LogisticRegression",
        "LinearSVC",
        "TfidfTransformer",
        "SigmoidCalibration",
    ],
)
def test_fit_observation_records_each_fitted_attribute_of_a_known_estimator(
    kind: str,
) -> None:
    estimator = type(kind, (), {})()
    vars(estimator).update({"coef_": torch.arange(3), "alpha": torch.arange(2)})
    trace = NumericalTrace()
    trace._observe(estimator)
    assert trace.counts == {kind: 1}
    assert trace.record.keys() == {f"{kind}/0/coef_"}


def test_fit_observation_skips_other_classes() -> None:
    stranger = type("Stranger", (), {})()
    vars(stranger).update({"a_": torch.arange(2)})
    trace = NumericalTrace()
    trace._observe(stranger)
    assert trace.counts == {}
    assert trace.record == {}


def test_a_failed_entry_restores_the_profiler_it_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def install(self: NumericalTrace) -> None:
        del self
        sys.setprofile(_noop_profile)
        raise RuntimeError("entry failed")

    monkeypatch.setattr(NumericalTrace, "_install_observer", install)
    found = sys.getprofile()

    def sentinel(*args: object) -> None:
        del args

    sys.setprofile(sentinel)
    try:
        with pytest.raises(RuntimeError, match="entry failed"):
            NumericalTrace().__enter__()
        assert sys.getprofile() is sentinel
    finally:
        sys.setprofile(found)


@pytest.mark.parametrize("busy", [(), (5,), (5, 4, 3, 2)])
def test_trace_names_and_then_frees_the_highest_free_monitoring_tool(
    busy: tuple[int, ...],
) -> None:
    with _occupied(*busy):
        expected = next(
            tool for tool in range(5, -1, -1) if sys.monitoring.get_tool(tool) is None
        )
        with NumericalTrace():
            assert sys.monitoring.get_tool(expected) == "name-that-llm-trace"
        assert sys.monitoring.get_tool(expected) is None
        assert sys.monitoring.get_events(expected) == 0


def test_trace_fails_when_every_monitoring_tool_is_taken() -> None:
    with _occupied(0, 1, 2, 3, 4, 5), pytest.raises(StopIteration):
        NumericalTrace().__enter__()


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "float16"])
def test_trace_records_boolean_integer_and_floating_arrays(dtype: str) -> None:
    trace = NumericalTrace()
    trace.put("values", value=np.zeros(3, dtype=dtype))
    assert trace.record["values"].shape == (3,)


def _saved_frame(code: CodeType, *, frame: FrameType) -> FrameType:
    assert frame.f_code is code
    return frame


def _unavailable_monitoring(tool: int) -> None:
    del tool
    raise RuntimeError("monitoring unavailable")


def _noop_profile(*args: object) -> None:
    del args


def _unexpected_observation(instance: object) -> None:
    del instance
    pytest.fail("A return other than fit entered numerical capture.")


def _record_fit_frame(
    frame: FrameType,
    event: str,
    arg: object,
    *,
    frames: list[FrameType],
) -> None:
    if (
        event == "return"
        and frame.f_code.co_name == "fit"
        and isinstance(arg, (StandardScaler, LinearSVC))
    ):
        frames.append(frame)


def _record_return(
    code: CodeType,
    offset: int,
    value: object,
    *,
    received: list[object],
) -> None:
    del code, offset
    received.append(value)


def _fit_scaler(features: NDArray[np.float64]) -> StandardScaler:
    scaler = StandardScaler.Config().make()
    scaler.fit(features)
    return scaler


@contextmanager
def _occupied(*tools: int) -> Generator[None]:
    taken = [tool for tool in tools if sys.monitoring.get_tool(tool) is None]
    for tool in taken:
        sys.monitoring.use_tool_id(tool, "occupied-by-test")
    try:
        yield
    finally:
        for tool in taken:
            sys.monitoring.free_tool_id(tool)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
