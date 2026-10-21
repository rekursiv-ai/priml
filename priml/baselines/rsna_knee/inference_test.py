"""Check the released inference stages against their source captures."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial, partialmethod
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

import csv
import hashlib
import json
import math
import os
import re
import shutil

from torch import Tensor, nn
from treekle import ReadError, from_plain, json_loads, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.rsna_knee.data import FINDINGS
from priml.baselines.rsna_knee.experiments import exp000
from priml.baselines.rsna_knee.imaging import (
    StudyData,
    anatomical_mirror,
    read_dicom,
    unchanged,
)
from priml.baselines.rsna_knee.inference import (
    EagerExecution,
    EncoderExecution,
    FoldModel,
    PublicRecipe,
    R7Score,
    R7Stage,
    RankBlend,
    RankGroup,
    RaptorScore,
    RaptorStage,
    ReleasedState,
    StageScores,
    _backend,
    _correlations,
    _mean_fill,
    _ordinal_ranks,
    _read_scores,
    _validate_predictions,
    _write_scores,
    accept_logits,
    percentiles_on,
)
from priml.baselines.rsna_knee.model import R7
from priml.baselines.rsna_knee.support_test import (
    corrupt_deflate,
    postprocess_assets,
    r7_stage_config,
    raptor_stage_config,
    released_state,
    study_inputs,
)
from priml.model.coatnet import CoAtNet
from priml.model.vnet import VNet
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from numpy.typing import NDArray

    from priml.baselines.rsna_knee.imaging import StudyInputs
    from priml.model.ae.checkpoint import LocalFile


_CWD: Final = Path(__file__).resolve().parent


def _skip_flush(descriptor: int) -> None:
    del descriptor


@pytest.fixture(autouse=True)
def _no_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip disk flushes: receipts are read back, not crash-tested, in these tests."""
    monkeypatch.setattr(os, "fsync", _skip_flush)


def test_rank_blend_preserves_average_ties_and_float64_distinctions() -> None:
    raptor384 = np.array(
        [[0.2, 0.9, 0.1], [0.2, 0.7, 0.3], [0.8, 0.5, 0.2], [0.6, 0.3, 0.4]],
    )
    raptor224 = np.array(
        [[0.9, 0.1, 0.6], [0.7, 0.3, 0.8], [0.5, 0.2, 0.2], [0.3, 0.4, 0.2]],
    )
    r7 = np.array(
        [[0.500_000_001, 0.2, 0.9], [0.5, 0.4, 0.7], [0.4, 0.6, 0.5], [0.3, 0.8, 0.3]],
    )
    result = exp000().blend.make()(
        [raptor384, raptor224, r7],
        [np.ones(4, dtype=np.bool_)] * 3,
    )
    expected = (
        0.5
        * (
            0.65
            * np.array(
                [
                    [0.375, 1.0, 0.25],
                    [0.375, 0.75, 0.75],
                    [1.0, 0.5, 0.5],
                    [0.75, 0.25, 1.0],
                ],
            )
            + 0.35
            * np.array(
                [
                    [1.0, 0.25, 0.75],
                    [0.75, 0.75, 1.0],
                    [0.5, 0.5, 0.375],
                    [0.25, 1.0, 0.375],
                ],
            )
        )
        + 0.5
        * np.array(
            [[1.0, 0.25, 1.0], [0.75, 0.5, 0.75], [0.5, 0.75, 0.5], [0.25, 1.0, 0.25]],
        )
    ).astype(np.float32)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("failed_model", ["raptor384", "raptor224", "r7"])
def test_rank_blend_rejects_excessive_model_failures(failed_model: str) -> None:
    scores = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
    inputs = {
        "raptor384": scores.copy(),
        "raptor224": scores.copy(),
        "r7": scores.copy(),
    }
    scored = {name: np.ones(2, dtype=np.bool_) for name in inputs}
    scored[failed_model][0] = False
    scored[failed_model][1] = failed_model != "r7"
    with pytest.raises(RuntimeError, match="failed on"):
        exp000().blend.make()(list(inputs.values()), list(scored.values()))


def test_rank_blend_matches_pinned_source_with_every_study_fallback() -> None:
    # Two failures need at least 67 studies to satisfy the source's 3% stop.
    values = np.arange(67 * 3).reshape(67, 3)
    raw384 = ((values % 13) + 1) / 15.0
    raw224 = ((values % 11) + 1) / 13.0
    ours = ((values % 7) + 1) / 9.0
    raw384[[0, 1]] = 0.5
    raw224[[0, 2]] = 0.5
    scored = np.ones(67, dtype=np.bool_)
    scored[[0, 3]] = False
    golden = _CWD / "testdata" / "rank_blend.pt"
    expected = read_tensors(golden)["predictions"].numpy()
    actual = exp000().blend.make()(
        [raw384, raw224, ours],
        [
            ~np.equal(raw384, 0.5).all(axis=1),
            ~np.equal(raw224, 0.5).all(axis=1),
            scored,
        ],
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_rank_blend_rejects_invalid_config_fractions(value: float) -> None:
    config = exp000().blend
    config.groups[0].weight = value
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        config.make()


def test_rank_blend_rejects_unaligned_study_matrices() -> None:
    with pytest.raises(ValueError, match="align by study"):
        exp000().blend.make()(
            [np.zeros((4, 3)), np.zeros((4, 2)), np.zeros((4, 3))],
            [np.ones(4, dtype=np.bool_)] * 3,
        )


def test_percentiles_ignore_missing_values_and_rank_infinities() -> None:
    scores = np.array([np.nan, -np.inf, 2.0, 2.0, np.inf, 0.0])
    result = percentiles_on(
        scores,
        valid=np.array([True, True, True, True, True, False]),
    )
    np.testing.assert_array_equal(
        result,
        np.array([np.nan, 0.25, 0.625, 0.625, 1.0, np.nan]),
    )
    assert np.isnan(percentiles_on(scores, valid=np.zeros(6, dtype=np.bool_))).all()


def test_blend_rejects_a_group_without_members() -> None:
    config = RankBlend.Config(groups=[RankGroup.Config(members=[])])
    with pytest.raises(ValueError, match="at least one member"):
        config.make()


class _Encoder(nn.Module):
    def __init__(
        self,
        delta: float = 0.0,
        *,
        real_only: bool = False,
        fail: bool = False,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.real_only, self.fail = real_only, fail

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.fail:
            raise RuntimeError("Native encoder failure.")
        delta = self.delta if not self.real_only or x.shape[1] == 5 else 0.0
        return x.mean((-1, -2, -3)) * 0 + delta

    def reset_parameters(self) -> None:
        pass


class _Model(R7):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.enc = _Encoder()
        self.pad_multiple = 32

    @override
    def forward(
        self,
        image: Tensor,
        meta: Tensor,
        mask: Tensor,
        *,
        chunk: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert meta.shape == (1, image.shape[1], 9)
        assert mask.shape == image.shape[:2]
        del chunk
        # R7Score invokes each fold on one study with twelve clinical findings.
        value = self.enc(image.float()).mean().expand(1, 12)
        return value, value.new_zeros(1, 12, 8), value.new_zeros(1, 13)

    @override
    def state_readout(self, states: Tensor, binaries: Tensor) -> Tensor:
        del binaries
        return states.new_zeros(1, 84)


def _compile(encoder: _Encoder, *, dynamic: bool, mode: str) -> _Encoder:
    assert not dynamic
    assert mode == "default"
    return _Encoder(encoder.delta)


def _broken(encoder: _Encoder, *, dynamic: bool, mode: str) -> _Encoder:
    del encoder
    assert not dynamic
    assert mode == "default"
    return _Encoder(1.0)


def _real_difference(encoder: _Encoder, *, dynamic: bool, mode: str) -> _Encoder:
    del encoder
    assert not dynamic
    assert mode == "default"
    return _Encoder(0.02, real_only=True)


class _InspectScore(R7Score):
    def __init__(self, config: R7Score.Config, *, reference: float = 0.0) -> None:
        super().__init__(config)
        self.dtypes: list[torch.dtype | None] = []
        self.reference = reference

    @override
    def forward(
        self,
        model: FoldModel,
        images: Tensor,
        metadata: Tensor,
        mask: Tensor,
        *,
        dtype: torch.dtype | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.dtypes.append(dtype)
        output = super().forward(model, images, metadata, mask, dtype=dtype)
        return (output[0] + self.reference if dtype is None else output[0]), *output[1:]


def _execution() -> EncoderExecution:
    execution = EncoderExecution.Config().make()
    execution.bind(
        [_Model() for _ in range(4)],
        score=R7Score.Config(device="cpu", crop=4, chunk_size=64).make(),
    )
    return execution


def test_compiler_checks_every_padded_shape_and_first_real_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "compile", _compile)
    execution = _execution()
    result = execution(torch.zeros(5, 3, 6, 4, dtype=torch.uint8), torch.zeros(5, 9))
    assert result.logits.shape == (4, 12)
    assert torch.equal(result.logits, torch.zeros(4, 12))
    assert execution.receipt["active"] is True
    assert execution.receipt["checked"] is True
    assert execution.receipt["shapes"] == [32, 64]
    assert execution.receipt["study_dprob"] == 0.0
    assert all(
        model.enc is encoder
        for model, encoder in zip(execution.models, execution.compiled, strict=True)
    )


def test_broken_compiler_restores_every_fold_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "compile", _broken)
    execution = _execution()
    result = execution(torch.zeros(5, 3, 6, 4, dtype=torch.uint8), torch.zeros(5, 9))
    assert torch.equal(result.logits, torch.zeros(4, 12))
    assert execution.receipt["active"] is False
    assert "synthetic" in str(execution.receipt["fallback"])


def test_eager_execution_keeps_all_four_complete_readouts() -> None:
    execution = EagerExecution.Config().make()
    execution.bind(
        [_Model() for _ in range(4)],
        score=R7Score.Config(device="cpu", crop=4).make(),
    )
    result = execution(torch.zeros(5, 3, 6, 4, dtype=torch.uint8), torch.zeros(5, 9))
    assert result.logits.shape == (4, 12)
    assert result.states.shape == (4, 84)
    assert execution.receipt["active"] is False


@pytest.mark.parametrize("reference", [0.0, 0.02])
def test_first_real_study_uses_float32_to_break_autocast_disagreement(
    monkeypatch: pytest.MonkeyPatch,
    reference: float,
) -> None:
    monkeypatch.setattr(torch, "compile", _real_difference)
    execution = _execution()
    score = _InspectScore(
        R7Score.Config(
            device="cpu",
            crop=4,
            chunk_size=64,
            dtype_autocast=torch.bfloat16,
        ),
        reference=reference,
    )
    execution.score = score
    execution(torch.zeros(5, 3, 6, 4, dtype=torch.uint8), torch.zeros(5, 9))
    assert execution.active == (reference > 0)
    assert execution.receipt["fp32_err_compiled"] == (
        0.0049998 if reference == 0 else 0.0
    )
    assert torch.bfloat16 in score.dtypes
    assert None in score.dtypes


def test_checked_forward_preserves_dtype_inputs_and_restores_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "compile", _compile)
    execution = _execution()
    # EncoderExecution._forward receives one-study image, metadata, and mask rows.
    image, meta, mask = (
        torch.zeros(1, 5, 3, 6, 4),
        torch.zeros(1, 5, 9),
        torch.ones(1, 5, dtype=torch.bool),
    )
    execution(image[0].to(torch.uint8), meta[0])
    saved = execution.models[0].enc
    score = _InspectScore(R7Score.Config(device="cpu", crop=4, chunk_size=64))
    execution.score = score
    execution._forward(0, image, meta, mask, compiled=False, dtype=torch.bfloat16)
    assert score.dtypes == [torch.bfloat16]
    assert execution.models[0].enc is saved


@pytest.mark.parametrize("chunk", [0, 31])
def test_invalid_compiled_chunk_falls_back_without_losing_the_study(
    monkeypatch: pytest.MonkeyPatch,
    chunk: int,
) -> None:
    monkeypatch.setattr(torch, "compile", _compile)
    execution = _execution()
    assert execution.score is not None
    execution.score.config.chunk_size = chunk
    execution(torch.zeros(5, 3, 6, 4, dtype=torch.uint8), torch.zeros(5, 9))
    assert not execution.active
    assert "positive multiple" in str(execution.receipt["fallback"])


def test_compiled_finiteness_pattern_must_match_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "compile", _nonfinite)
    execution = _execution()
    execution(torch.zeros(5, 3, 6, 4, dtype=torch.uint8), torch.zeros(5, 9))
    assert not execution.active
    assert "finiteness" in str(execution.receipt["fallback"])


def _nonfinite(encoder: _Encoder, *, dynamic: bool, mode: str) -> _Encoder:
    del encoder, dynamic, mode
    return _Encoder(float("nan"))


@pytest.mark.parametrize("autocast", [True, False])
def test_scoring_failure_falls_back_all_folds_only_for_the_compiled_path(
    monkeypatch: pytest.MonkeyPatch,
    autocast: bool,
) -> None:
    monkeypatch.setattr(torch, "compile", _compile)
    execution = _execution()
    # EncoderExecution._forward receives one-study image, metadata, and mask rows.
    image, meta, mask = (
        torch.zeros(1, 5, 3, 6, 4),
        torch.zeros(1, 5, 9),
        torch.ones(1, 5, dtype=torch.bool),
    )
    execution(image[0].to(torch.uint8), meta[0])
    if autocast:
        execution.compiled[0] = _Encoder(fail=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            # R7Score returns one row containing all twelve clinical findings.
            assert torch.equal(
                execution.folds[0](image, meta, mask, chunk=64)[0],
                torch.zeros(1, 12),
            )
        assert not execution.active
        assert all(
            model.enc is encoder
            for model, encoder in zip(execution.models, execution.eager, strict=True)
        )
    else:
        execution.eager[0] = _Encoder(fail=True)
        with pytest.raises(RuntimeError, match="Native encoder failure"):
            execution.folds[0](image, meta, mask, chunk=64)
        assert execution.active
        assert execution.models[0].enc is execution.compiled[0]


def test_unbound_folds_cannot_provide_a_scorer() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"^Folds must be bound before they are scored\.$",
    ):
        EncoderExecution.Config().make().scorer()


def test_unbound_folds_start_without_models() -> None:
    assert EncoderExecution.Config().make().models == []


def test_released_asset_hash_is_checked_before_deserializing(tmp_path: Path) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    path = tmp_path / "model.pt"
    torch.save({"model": {"logits": expected["raptor/logits"]}}, path)
    asset = released_state(path, key="model").make()
    assert torch.equal(asset.state()["logits"], expected["raptor/logits"])
    path.write_bytes(b"altered checkpoint")
    with pytest.raises(RuntimeError, match="hashes to"):
        asset.state()


@pytest.mark.parametrize("weights_only", [False, True])
def test_released_deserialization_places_every_tensor_on_cpu_with_declared_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    weights_only: bool,
) -> None:
    path = tmp_path / "asset.pt"
    torch.save({"value": torch.arange(3)}, path)
    config = released_state(path)
    config.weights_only = weights_only
    asset = config.make()
    original = torch.load
    calls: list[tuple[str, bool]] = []

    def load(path: Path, *, map_location: str, weights_only: bool) -> object:
        calls.append((map_location, weights_only))
        return cast(
            object,
            original(path, map_location=map_location, weights_only=weights_only),
        )

    monkeypatch.setattr(torch, "load", load)
    assert torch.equal(asset.state()["value"], torch.arange(3))
    assert calls == [("cpu", weights_only)]
    calls.clear()
    path.write_bytes(b"Altered bytes must never reach deserialization.")
    with pytest.raises(RuntimeError, match="hashes to"):
        asset.state()
    assert not calls


def test_checkpoint_paths_resolve_beneath_the_resource_root(tmp_path: Path) -> None:
    config = ReleasedState.Config()
    config.base_dir = tmp_path
    config.file.path = "/datasets/model.pt"
    assert config.copy_tree().finalize().file.path == tmp_path / "datasets/model.pt"
    config.base_dir = None
    assert config.copy_tree().finalize().file.path == Path("/datasets/model.pt")


class _Fold:
    def __init__(self, record: dict[str, Tensor], index: int) -> None:
        self.record, self.index = record, index
        self.calls = 0

    def __call__(
        self,
        images: Tensor,
        metadata: Tensor,
        mask: Tensor,
        *,
        chunk: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert chunk == 192
        assert not torch.is_grad_enabled()
        for label, value in (("inputs", images), ("meta", metadata), ("mask", mask)):
            expected = self.record[f"r7/{label}/{self.index}"][self.calls]
            assert value.dtype == expected.dtype
            assert torch.equal(value, expected)
        self.calls += 1
        logits = self.record["r7/logits"][self.index].clone()
        if self.index == 1 and self.calls == 1:
            logits[0, 0] = float("nan")
        return (
            logits,
            self.record["r7/states"][self.index].clone(),
            self.record["r7/binaries"][self.index].clone(),
        )

    def state_readout(self, states: Tensor, binaries: Tensor) -> Tensor:
        for name, value in (("states", states), ("binaries", binaries)):
            assert value.dtype == torch.float32
            torch.testing.assert_close(
                value,
                self.record[f"r7/{name}"][self.index].float(),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        return self.record["r7/readout"][self.index][None].clone()


class _Raptor(nn.Module):
    def __init__(
        self,
        inputs: Tensor,
        logits: Tensor,
        *,
        failure: type[Exception] | None = None,
    ) -> None:
        super().__init__()
        self.inputs, self.logits = inputs, logits
        self.failure = failure
        self.calls = 0

    @override
    def forward(self, images: Tensor) -> Tensor:
        assert not torch.is_grad_enabled()
        assert images.dtype == self.inputs.dtype
        assert torch.equal(images, self.inputs[self.calls])
        self.calls += 1
        if self.failure is not None and self.calls == 1:
            raise self.failure("Frozen model failure.")
        return self.logits[min(self.calls - 1, len(self.logits) - 1)].clone()


def test_all_four_source_fold_calls_crop_dtype_rescue_and_state_nan() -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    models = [_Fold(expected, index) for index in range(4)]
    score = R7Score.Config(device="cpu", crop=4).make()
    actual = score(models, expected["r7/images"], expected["r7/metadata"])
    assert actual.logits.dtype == torch.float32
    assert torch.equal(actual.logits, expected["r7/stacked"])
    assert torch.equal(actual.logits.mean(0), expected["r7/mean"])
    assert actual.states.dtype == torch.float32
    torch.testing.assert_close(
        actual.states,
        expected["r7/readout"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert torch.equal(
        torch.tensor([model.calls for model in models]),
        expected["r7/counts"],
    )


def test_source_raptor_pair_sigmoids_each_pass_before_float32_numpy_average() -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    model = _Raptor(expected["raptor/inputs"], expected["raptor/logits"]).eval()
    actual = RaptorScore.Config(device="cpu").make()(
        model,
        [expected["raptor/plain"], expected["raptor/mirror"]],
    )
    assert actual.dtype == np.float32
    assert torch.equal(torch.from_numpy(actual), expected["raptor/pair"])
    assert model.calls == 2


def test_source_224_sigmoid_tails_match_all_original_bits() -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    model = _Raptor(
        expected["raptor/inputs"][:1],
        expected["raptor224/tail_logits"][None],
    ).eval()
    score = RaptorScore.Config(device="cpu", check_logits=accept_logits).make()
    actual = torch.from_numpy(score(model, [expected["raptor/plain"]]))
    assert torch.equal(
        actual.view(torch.int32),
        expected["raptor224/tail_probability"].view(torch.int32),
    )


def test_source_224_runtime_error_retries_the_complete_same_input() -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    model = _Raptor(
        expected["raptor224/inputs"],
        expected["raptor/logits"][:1],
        failure=RuntimeError,
    ).eval()
    score = RaptorScore.Config(
        device="cpu",
        dtypes_autocast=[torch.float16, None],
        check_logits=accept_logits,
    ).make()
    actual = score(model, [expected["raptor/plain"]])
    assert actual.dtype == np.float32
    assert torch.equal(torch.from_numpy(actual), expected["raptor224/probability"])
    assert model.calls == 2


def test_first_arm_does_not_rescue_runtime_failures_or_nonfinite_logits() -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    inputs = expected["raptor/inputs"][:1]
    logits = expected["raptor/logits"][:1]
    model = _Raptor(inputs, logits, failure=RuntimeError).eval()
    score = RaptorScore.Config(device="cpu").make()
    with pytest.raises(RuntimeError, match=r"^Frozen model failure\.$"):
        score(model, [expected["raptor/plain"]])
    assert model.calls == 1
    model = _Raptor(inputs, logits * float("nan")).eval()
    with pytest.raises(FloatingPointError, match=r"^Raptor logits are nonfinite\.$"):
        score(model, [expected["raptor/plain"]])
    assert model.calls == 1


def test_disabled_finite_policy_performs_no_finite_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    model = _Raptor(expected["raptor/inputs"], expected["raptor/logits"]).eval()
    calls: list[Tensor] = []
    original = torch.isfinite

    def observe(value: Tensor) -> Tensor:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(torch, "isfinite", observe)
    config = RaptorScore.Config()
    config.device = "cpu"
    config.check_logits = accept_logits
    actual = config.make()(
        model,
        [expected["raptor/plain"], expected["raptor/mirror"]],
    )
    assert calls == []
    assert torch.equal(torch.from_numpy(actual), expected["raptor/pair"])


@contextmanager
def _autocast(
    device_type: str,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
    *,
    calls: list[tuple[str, torch.dtype | None, bool]],
) -> Generator[None]:
    calls.append((device_type, dtype, enabled))
    yield


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scoring_device_autocast_and_cache_rescue_contract(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    transfers: list[torch.device | None] = []
    contexts: list[tuple[str, torch.dtype | None, bool]] = []
    caches: list[int] = []

    def transfer(tensor: Tensor, device: torch.device | None = None) -> Tensor:
        transfers.append(device)
        return tensor

    monkeypatch.setattr(Tensor, "to", transfer)
    monkeypatch.setattr(torch, "autocast", partial(_autocast, calls=contexts))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: caches.append(1))
    model = _Raptor(
        expected["raptor224/inputs"],
        expected["raptor/logits"][:1],
        failure=RuntimeError,
    ).eval()
    config = RaptorScore.Config(
        device=device,
        check_logits=accept_logits,
        dtypes_autocast=[torch.float16, None],
    )
    actual = config.make()(model, [expected["raptor/plain"]])
    assert torch.equal(torch.from_numpy(actual), expected["raptor224/probability"])
    assert transfers == [torch.device(device)]
    assert contexts == [
        (device, torch.float16, device == "cuda"),
        (device, None, False),
    ]
    assert caches == ([1] if device == "cuda" else [])


def test_runtime_error_during_device_to_host_copy_retries_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    original = Tensor.cpu
    copies: list[int] = []

    monkeypatch.setattr(
        Tensor,
        "cpu",
        partialmethod(_copy, copies=copies, original=original),
    )
    model = _Raptor(expected["raptor224/inputs"], expected["raptor/logits"][:1]).eval()
    score = RaptorScore.Config(
        device="cpu",
        check_logits=accept_logits,
        dtypes_autocast=[torch.float16, None],
    ).make()
    actual = score(model, [expected["raptor/plain"]])
    assert torch.equal(torch.from_numpy(actual), expected["raptor224/probability"])
    assert model.calls == len(copies) == 2


def _copy(
    tensor: Tensor,
    *,
    copies: list[int],
    original: Callable[[Tensor], Tensor],
) -> Tensor:
    copies.append(1)
    if len(copies) == 1:
        raise RuntimeError("Frozen transfer failure.")
    return original(tensor)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [None, torch.float16])
@torch.inference_mode()
def test_each_r7_fold_keeps_its_device_precision_and_float32_readout_boundary(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    dtype: torch.dtype | None,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    contexts: list[tuple[str, torch.dtype | None, bool]] = []
    monkeypatch.setattr(torch, "autocast", partial(_autocast, calls=contexts))
    model = _Fold(expected, 0)
    result = (
        R7Score.Config(device=device, crop=4)
        .make()
        .forward(
            model,
            expected["r7/inputs/0"][0],
            expected["r7/meta/0"][0],
            expected["r7/mask/0"][0],
            dtype=dtype,
        )
    )
    assert contexts == [(device, dtype, device == "cuda" and dtype is not None)]
    for actual, name in zip(result, ("logits", "states", "binaries"), strict=True):
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected[f"r7/{name}"][0].float())


def test_scoring_requires_at_least_one_autocast_precision() -> None:
    score = RaptorScore.Config(device="cpu", dtypes_autocast=[]).make()
    with pytest.raises(ValueError, match="at least one autocast precision"):
        score(torch.nn.Identity(), [torch.zeros(2, 3)])


def _path(file: LocalFile.Config) -> Path:
    return Path(file.path)


def test_both_stackers_and_serialization_replay_complete_source_outputs(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "stackers.pt")
    postprocess = postprocess_assets(tmp_path).make()
    for index, volume in enumerate((None, 0.0, 13.7, -1.0, float("nan"), float("inf"))):
        fluid = None if volume is None else {"eff_sag_p99_open": volume}
        result = postprocess(expected["logits"], expected["states"], fluid)
        assert not result.failures
        assert torch.equal(
            torch.from_numpy(result.probabilities),
            expected[f"{index}/probabilities"],
        )
        assert torch.equal(
            torch.from_numpy(result.serialized),
            expected[f"{index}/serialized"],
        )


def test_bad_state_features_preserve_model_and_fluid_probabilities(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "stackers.pt")
    postprocess = postprocess_assets(tmp_path).make()
    # R7Postprocess needs the complete state layout; width one exercises fallback.
    result = postprocess(expected["logits"], torch.zeros(4, 1), None)
    base = torch.sigmoid(expected["logits"].mean(0)).double().numpy()
    assert result.failures
    assert result.probabilities[9] == base[9]
    assert result.probabilities[7] == float(expected["0/probabilities"][7])
    states = expected["states"].clone()
    states[:, 51] = float("nan")
    result = postprocess(expected["logits"], states, None)
    assert result.failures
    assert result.probabilities[9] == base[9]
    logits = expected["logits"].clone()
    logits[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        postprocess(logits, states, None)


def test_asset_layout_prior_and_stacker_overlap_are_verified(tmp_path: Path) -> None:
    config = postprocess_assets(tmp_path)
    _path(config.spec_file).write_text(
        _path(config.spec_file).read_text().replace('"category_0"', '"altered"', 1),
    )
    with pytest.raises(ValueError, match=r"layout|spec"):
        config.make()
    config = postprocess_assets(tmp_path)
    _path(config.fluid_file).write_text(
        _path(config.fluid_file).read_text().replace('"logit"', '"prob"'),
    )
    with pytest.raises(
        ValueError,
        match=r"^Fluid calibration requires logit line mixing\.$",
    ):
        config.make()
    config = postprocess_assets(tmp_path)
    _path(config.state_file).write_text(
        _path(config.state_file).read_text().replace('"Baker\'s": {', '"Effusion": {'),
    )
    with pytest.raises(ValueError, match="overlap"):
        config.make()
    config = postprocess_assets(tmp_path)
    _path(config.fluid_file).write_text(
        _path(config.fluid_file).read_text().replace('"view_id": 4', '"view_id": 5'),
    )
    with pytest.raises(
        ValueError,
        match=r"^Fluid calibration volume and rendered view differ\.$",
    ):
        config.make()


def _change(path: Path, keys: tuple[str, ...], value: object) -> None:
    data = json_loads(path.read_text(), dict[str, object])
    parents = [data]
    for key in keys[:-1]:
        child = from_plain(parents[-1][key], dict[str, object])
        parents[-1][key] = child
        parents.append(child)
    parents[-1][keys[-1]] = value
    path.write_text(json.dumps(to_plain(data)))


@pytest.mark.parametrize(
    ("asset", "keys", "value", "message"),
    [
        (
            "fluid",
            ("feature", "weights"),
            [0.9],
            "The released state recipe requires one model group.",
        ),
        (
            "state",
            ("feature", "lines"),
            [],
            "The released state recipe requires one model group.",
        ),
        ("state", ("kind",), "unknown", "Unsupported state stacker kind."),
        (
            "state",
            ("layout", "width"),
            83,
            "State stacker layout differs from the checkpoint specification.",
        ),
        (
            "state",
            ("state_spec_md5",),
            "incorrect",
            "State stacker was fitted on a different state spec.",
        ),
        (
            "fluid",
            ("feature", "weights_per_class"),
            {"Effusion": [0.8]},
            "Fluid class weights differ from the deployed line mix.",
        ),
        (
            "fluid",
            ("classes", "Effusion", "b0"),
            float("inf"),
            "Stacker coefficients must be finite.",
        ),
        (
            "fluid",
            ("classes", "Effusion", "b_z"),
            float("nan"),
            "Stacker coefficients must be finite.",
        ),
        (
            "state",
            ("classes", "Baker's", "coef"),
            [1.0],
            "State coefficients do not match the ordered feature layout.",
        ),
        (
            "state",
            ("classes", "Baker's", "features"),
            ["z"],
            "State coefficients do not match the ordered feature layout.",
        ),
        (
            "state",
            ("classes", "Baker's", "volume_column"),
            "unknown",
            "State calibration uses an unknown fluid volume.",
        ),
    ],
)
def test_incompatible_calibration_assets_fail_before_scoring(
    tmp_path: Path,
    asset: str,
    keys: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    config = postprocess_assets(tmp_path)
    path = {"fluid": _path(config.fluid_file), "state": _path(config.state_file)}[asset]
    _change(path, keys, value)
    with pytest.raises(ValueError, match="^" + re.escape(message) + "$") as error:
        config.make()
    assert str(error.value) == message


def test_invalid_priors_and_unadopted_classes_have_explicit_behavior(
    tmp_path: Path,
) -> None:
    config = postprocess_assets(tmp_path)
    _change(_path(config.spec_file), ("prior", "ACL"), [0.5])
    _change(
        _path(config.state_file),
        ("state_spec_md5",),
        hashlib.md5(
            _path(config.spec_file).read_bytes(),
            usedforsecurity=False,
        ).hexdigest(),
    )
    with pytest.raises(
        ValueError,
        match=r"^State prior length differs from its category count\.$",
    ):
        config.make()
    config = postprocess_assets(tmp_path)
    _change(_path(config.fluid_file), ("classes", "Effusion", "adopt"), False)
    _change(_path(config.state_file), ("classes", "Baker's", "adopt"), False)
    expected = read_tensors(_CWD / "testdata" / "stackers.pt")
    result = config.make()(expected["logits"], expected["states"], None)
    assert not result.failures
    assert torch.equal(
        torch.from_numpy(result.probabilities),
        torch.sigmoid(expected["logits"].mean(0)).double(),
    )


def test_optional_volume_features_and_dropped_features_keep_coefficient_order(
    tmp_path: Path,
) -> None:
    config = postprocess_assets(tmp_path)
    expected = read_tensors(_CWD / "testdata" / "stackers.pt")
    _change(
        _path(config.state_file),
        ("classes", "Baker's", "volume_column"),
        "bak_sag_p99_open",
    )
    _change(
        _path(config.state_file),
        ("classes", "Baker's", "features"),
        ["z", *[f"state:category_{index}" for index in range(6)], "miss"],
    )
    _change(_path(config.state_file), ("classes", "Baker's", "dropped"), ["lv"])
    _change(
        _path(config.state_file),
        ("classes", "Baker's", "coef"),
        [*[float(value) for value in expected["state_coefficients"][1:]], 0.0],
    )
    model = config.make()
    result = model(expected["logits"], expected["states"], {"bak_sag_p99_open": 2.5})
    assert not result.failures
    assert torch.equal(
        torch.from_numpy(result.probabilities),
        expected["0/probabilities"],
    )


def test_typed_assets_reject_invalid_unused_fields(tmp_path: Path) -> None:
    config = postprocess_assets(tmp_path)
    _change(_path(config.state_file), ("classes", "Baker's", "adopt"), False)
    _change(_path(config.state_file), ("classes", "Baker's", "coef"), ["invalid"])
    with pytest.raises(ReadError):
        config.make()
    config = postprocess_assets(tmp_path)
    _change(_path(config.spec_file), ("prior", "ACL"), ["invalid"])
    with pytest.raises(ReadError):
        config.make()


def test_inactive_rows_do_not_skip_later_adopted_rows_and_baker_volume_is_valid(
    tmp_path: Path,
) -> None:
    config = postprocess_assets(tmp_path)
    for path, name in (
        (_path(config.fluid_file), "Effusion"),
        (_path(config.state_file), "Baker's"),
    ):
        asset = json_loads(path.read_text(), dict[str, object])
        classes = from_plain(asset["classes"], dict[str, dict[str, object]])
        asset["classes"] = {"inactive": dict(classes[name], adopt=False), **classes}
        path.write_text(json.dumps(to_plain(asset)))
    _change(
        _path(config.fluid_file),
        ("classes", "Effusion", "volume_column"),
        "bak_sag_p99_open",
    )
    _change(_path(config.fluid_file), ("classes", "Effusion", "view_id"), 5)
    expected = read_tensors(_CWD / "testdata" / "stackers.pt")
    result = config.make()(
        expected["logits"],
        expected["states"],
        {"bak_sag_p99_open": 13.7},
    )
    assert torch.equal(
        torch.from_numpy(result.probabilities),
        expected["2/probabilities"],
    )


def test_binary_state_features_and_volume_coefficients_affect_scores(
    tmp_path: Path,
) -> None:
    config = postprocess_assets(tmp_path)
    expected = read_tensors(_CWD / "testdata" / "stackers.pt")
    _change(
        _path(config.state_file),
        ("classes", "MCL"),
        {
            "adopt": True,
            "features": [
                "z",
                *[f"state:category_{index}" for index in range(6)],
                "bin:binary_9",
            ],
            "coef": [0.0] * 7 + [1.0],
            "b0": 0.0,
            "volume_column": "",
        },
    )
    model = config.make()
    result = model(expected["logits"], expected["states"], None)
    assert result.probabilities.item(1) == 1 / (
        1 + math.exp(-expected["states"].double().mean(0)[80].item())
    )
    _change(
        _path(config.state_file),
        ("classes", "Baker's", "features"),
        ["z", *[f"state:category_{index}" for index in range(6)], "lv", "miss"],
    )
    _change(
        _path(config.state_file),
        ("classes", "Baker's", "coef"),
        [0.0] * 7 + [1.0, 1.0],
    )
    _change(_path(config.state_file), ("classes", "Baker's", "b0"), 0.0)
    _change(
        _path(config.state_file),
        ("classes", "Baker's", "volume_column"),
        "bak_sag_p99_open",
    )
    result = config.make()(
        expected["logits"],
        expected["states"],
        {"bak_sag_p99_open": 2.5},
    )
    log_volume: NDArray[np.float64] = np.log1p(np.array([2.5], dtype=np.float64))
    assert result.probabilities.item(9) == 1 / (1 + math.exp(-log_volume.item(0)))


def test_spec_checksum_requests_nonsecurity_md5_on_restricted_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = postprocess_assets(tmp_path)

    def restricted_md5(value: bytes, *, usedforsecurity: bool = True) -> object:
        if usedforsecurity is not False:
            raise ValueError("MD5 is available only for nonsecurity checksums.")
        return hashlib.new("md5", value, usedforsecurity=False)

    monkeypatch.setattr(hashlib, "md5", restricted_md5)
    assert config.make().state


def test_full_raptor_stage_runs_both_views_even_for_an_empty_study(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    data = study_inputs(tmp_path).make()
    config = raptor_stage_config(
        tmp_path / "raptor.pt",
        logits=expected["raptor/logits"],
    )
    config.views = [unchanged, anatomical_mirror]
    result = config.make()(data)
    assert result.values.dtype == np.float32
    assert torch.equal(
        torch.from_numpy(result.values),
        expected["raptor/pair"].repeat(2, 1),
    )
    assert result.scored.tolist() == [True, True]
    assert result.failures == {}


def test_four_fold_anatomy_stage_keeps_stackers_and_mean_fills_failed_studies(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    data = study_inputs(tmp_path).make()
    config = r7_stage_config(tmp_path, expected)
    result = config.make()(data)
    assert result.values.dtype == np.float64
    assert result.values.shape == (2, 12)
    assert result.scored.tolist() == [True, False]
    assert set(result.failures) == {"empty"}
    assert torch.equal(
        torch.from_numpy(result.values)[0],
        torch.from_numpy(result.values)[1],
    )
    assert result.notes[0]["uid"] == "study"


def _damaged_study(root: Path, *, damage: str) -> StudyData:
    config = study_inputs(root)
    shutil.copytree(
        root / "test_series" / "study",
        dst=root / "test_series" / "zz-good",
    )
    for name in ("test.csv", "sample_submission.csv", "test_series.csv"):
        path = root / name
        rows = path.read_text().splitlines()
        path.write_text(
            "\n".join(
                [
                    *rows,
                    *(
                        row.replace("study", "zz-good", 1)
                        for row in rows[1:]
                        if row.startswith("study,") or row == "study"
                    ),
                ],
            )
            + "\n",
        )
    if damage == "deflate":
        corrupt_deflate(root)
    else:
        for path in (root / "test_series" / "study").rglob("*.dcm"):
            payload = path.read_bytes()
            spacing = read_dicom(payload).values[0x00280030]
            old = (
                b"\x28\x00\x30\x00DS"
                + len(spacing).to_bytes(2, byteorder="little")
                + spacing
            )
            value = b"1000\\1000 "
            new = (
                b"\x28\x00\x30\x00DS"
                + len(value).to_bytes(2, byteorder="little")
                + value
            )
            assert payload.count(old) == 1
            path.write_bytes(payload.replace(old, new))
    return config.make()


@pytest.mark.parametrize("damage", ["deflate", "spacing"])
def test_native_input_damage_keeps_raptor_zero_bag_and_next_study_scoring(
    tmp_path: Path,
    damage: str,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    data = _damaged_study(tmp_path, damage=damage)
    config = raptor_stage_config(
        tmp_path / "raptor.pt",
        logits=expected["raptor/logits"],
    )
    config.views = [unchanged, anatomical_mirror]
    result = config.make()(data)
    assert data.ids == ["study", "empty", "zz-good"]
    assert result.scored.tolist() == [True, True, True]
    assert result.failures == {}
    assert torch.equal(
        torch.from_numpy(result.values),
        expected["raptor/pair"].repeat(3, 1),
    )
    if damage == "spacing":
        assert len(result.notes) == 1
        assert result.notes[0]["uid"] == "study"
        assert "crop" in str(result.notes[0]["preparation_error"])


def test_corrupt_deflate_mean_fills_r7_failure_and_scores_next_study(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    data = _damaged_study(tmp_path, damage="deflate")
    result = r7_stage_config(tmp_path, expected=expected).make()(data)
    assert result.scored.tolist() == [False, False, True]
    assert list(result.failures) == ["study", "empty"]
    assert torch.equal(
        torch.from_numpy(result.values),
        torch.from_numpy(result.values[2]).repeat(3, 1),
    )


def test_recipe_stages_retain_every_model_and_release_parameter() -> None:
    config = exp000().copy_tree().finalize()
    raptor = config.raptor384
    assert raptor.windows.crop_size == 320
    assert raptor.views == [unchanged, anatomical_mirror]
    assert raptor.windows.windows == 94
    assert raptor.preparation.slots[-1] == ("Axial", -1, 18)
    assert isinstance(raptor.model.backbone, CoAtNet.Config)
    assert raptor.model.backbone.image_size == (320, 320)
    assert raptor.model.chunk_size == 16
    assert (raptor.cudnn_benchmark, raptor.allow_tf32_matmul) == (True, True)
    r7 = config.r7
    assert len(r7.checkpoints) == 4
    assert isinstance(r7.execution, EncoderExecution.Config)
    assert r7.score.crop == 352
    assert r7.score.chunk_size == 192
    assert r7.score.dtype_autocast == torch.bfloat16
    assert (r7.cudnn_benchmark, r7.allow_tf32_matmul) == (False, False)
    assert r7.preparation.eval_stride == 1
    assert r7.preparation.render.image_size == 384
    assert isinstance(r7.model, R7.Config)
    assert r7.model.encoder.blocks_per_stage == (3, 3, 27, 3)
    assert isinstance(r7.anatomy, VNet.Config)
    assert r7.anatomy.channels_in == 1
    assert r7.anatomy.channels_out == 4
    assert r7.anatomy.channels_hidden == (16, 32, 64, 128, 256)


@pytest.mark.parametrize("config_type", [RaptorStage.Config, R7Stage.Config])
def test_stage_backend_flags_apply_inside_the_stage_and_restore_afterwards(
    config_type: type[RaptorStage.Config | R7Stage.Config],
) -> None:
    config = config_type()
    config.cudnn_benchmark, config.allow_tf32_cudnn = True, False
    config.allow_tf32_matmul = True
    benchmark = torch.backends.cudnn.benchmark
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    with _backend(config):
        assert torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32
    assert torch.backends.cudnn.benchmark == benchmark
    assert torch.backends.cudnn.allow_tf32 == cudnn_tf32
    assert torch.backends.cuda.matmul.allow_tf32 == matmul_tf32


def _missing(uid: str) -> StudyInputs:
    del uid
    raise OSError("Native acquisition failure.")


def test_raptor_preparation_errors_score_zero_bags_and_log_inference_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    data = study_inputs(tmp_path).make()
    monkeypatch.setattr(data, "load", _missing)
    config = raptor_stage_config(
        tmp_path / "nonfinite.pt",
        logits=expected["raptor/logits"] * float("nan"),
    )
    config.views = [unchanged, anatomical_mirror]
    result = config.make()(data)
    assert len(result.notes) == 2
    assert set(result.failures) == set(data.ids)
    assert not result.scored.any()
    assert np.equal(result.values, 0.5).all()


def test_raptor_timeout_aborts_the_complete_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    data = study_inputs(tmp_path).make()
    config = raptor_stage_config(
        tmp_path / "raptor.pt",
        logits=expected["raptor/logits"],
    )
    config.time_limit = 1.0
    times = iter((0.0, 2.0))
    monkeypatch.setattr(
        "priml.baselines.rsna_knee.inference.time.monotonic",
        lambda: next(times),
    )
    with pytest.raises(TimeoutError, match="wall-time limit"):
        config.make()(data)


def test_r7_deadline_retains_the_first_study_and_mean_fills_unreached_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    data = study_inputs(tmp_path).make()
    config = r7_stage_config(tmp_path, expected)
    monkeypatch.setattr(
        "priml.baselines.rsna_knee.inference.time.monotonic",
        lambda: 1.0,
    )
    result = config.make()(data, deadline=0.1)
    assert result.scored.tolist() == [True, False]
    assert result.failures == {"empty": "Not reached before the deadline."}


@pytest.mark.parametrize("policy", ["average", "ordinal"])
def test_numpy_rank_consumers_keep_cpu_placement_under_an_ambient_device(
    policy: str,
) -> None:
    values = np.array([0.0, 1.0, 1.0, 3.0])
    with torch.device("meta"):
        if policy == "average":
            actual = percentiles_on(values, valid=np.ones(4, dtype=np.bool_))
        else:
            actual = _ordinal_ranks(values[:, None])[:, 0]
    expected = (
        np.array([0.25, 0.625, 0.625, 1.0])
        if policy == "average"
        else np.array([0.0, 1 / 3, 2 / 3, 1.0], dtype=np.float32)
    )
    assert np.array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def _recipe(root: Path) -> PublicRecipe.Config:
    expected = read_tensors(_CWD / "testdata" / "inference.pt")
    config = PublicRecipe.Config()
    config.base_dir = None
    data = root / "raw"
    data.mkdir()
    config.data = study_inputs(data)
    config.working_dir = root / "output"
    config.raptor384 = raptor_stage_config(
        root / "raptor.pt",
        logits=expected["raptor/logits"],
    )
    config.raptor384.views = [unchanged, anatomical_mirror]
    config.raptor224 = raptor_stage_config(
        root / "raptor.pt",
        logits=expected["raptor/logits"],
    )
    config.r7 = r7_stage_config(root, expected)
    config.blend = exp000().blend
    return config


@pytest.mark.parametrize("layout", ["test_series", "test_images"])
def test_native_recipe_writes_every_stage_and_final_submission(
    tmp_path: Path,
    layout: str,
) -> None:
    config = _recipe(tmp_path)
    if layout == "test_images":
        (Path(config.data.input_dir) / "test_series").rename(
            Path(config.data.input_dir) / layout,
        )
    config.make().run()
    output = Path(config.working_dir)
    assert {path.name for path in output.iterdir()} >= {
        "raptor_raw.csv",
        "raptor_sub.csv",
        "raptor224_raw.csv",
        "raptor224_sub.csv",
        "ours_sub.csv",
        "submission.csv",
        "scored.csv",
        "run_receipt.json",
        "blend_receipt.json",
        "raptor_receipt.json",
        "raptor224_receipt.json",
        "submission_receipt.json",
        "config.txt",
    }
    with (output / "submission.csv").open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["StudyInstanceUID", *reversed(FINDINGS)]
    assert [row[0] for row in rows[1:]] == ["study", "empty"]
    values = np.asarray([row[1:] for row in rows[1:]], dtype=np.float32)
    assert values.shape == (2, 12)
    assert np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    assert b"\r\n" not in (output / "submission.csv").read_bytes()
    assert b"\r\n" in (output / "ours_sub.csv").read_bytes()
    record = json_loads((output / "run_receipt.json").read_text(), dict[str, object])
    receipt = from_plain(record["metadata"], dict[str, object])
    assert record["status"] == "complete"
    assert receipt["historical_training_reproduced"] is False
    assert receipt["implementation"] == "native Priml"
    assert receipt["rows"] == 2
    assert receipt["raptor_scored"] == [2, 2]
    assert not list(output.rglob("*.py"))
    scored = (output / "scored.csv").read_text()
    assert "study" in scored
    assert "empty" not in scored
    assert (output / "raptor_raw.csv").read_text().splitlines()[0].split(",") == [
        "StudyInstanceUID",
        *FINDINGS,
    ]


@pytest.mark.parametrize("layout", ["test_series", "test_images"])
def test_complete_csv_controller_and_final_blend_replay_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "pipeline.pt")
    ids, sample = ["zeta", "alpha", "empty", "beta"], ["beta", "empty", "zeta", "alpha"]
    with (tmp_path / "test.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID"])
        writer.writerows([[uid] for uid in ids])
    with (tmp_path / "sample_submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *reversed(FINDINGS)])
        writer.writerows([[uid, *([0.0] * 12)] for uid in sample])
    (tmp_path / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n",
    )
    data = StudyData.Config(input_dir=tmp_path).make()
    for uid, count in zip(ids, (5, 8, 0, 2), strict=True):
        directory = tmp_path / "test_series" / uid / "series"
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index}.dcm").touch()
    if layout == "test_images":
        (tmp_path / "test_series").rename(tmp_path / layout)
    calls: list[int] = []

    def load(uid: str) -> StudyInputs:
        calls.append(ids.index(uid))
        raise OSError("Frozen worker failure.")

    monkeypatch.setattr(data, "load", load)
    stage = r7_stage_config(
        tmp_path,
        read_tensors(_CWD / "testdata" / "inference.pt"),
    )
    assert not stage.make()(data).scored.any()
    assert torch.equal(torch.tensor(calls), expected["controller/order"])
    values = ((np.arange(48).reshape(4, 12) * 17 % 47) + 0.25) / 48
    values = np.asarray(
        [
            [float(f"{float(values.item(index, column)):.9g}") for column in range(12)]
            for index in range(4)
        ],
        dtype=np.float64,
    )
    result = StageScores(
        values=values,
        scored=np.asarray([True, True, False, True], dtype=np.bool_),
        failures={"empty": "frozen worker failure"},
    )
    _mean_fill(result, ids=ids, sample_ids=sample)
    _write_scores(
        tmp_path / "ours_sub.csv",
        values=result.values,
        data=data,
        order=sample,
        decimals=True,
    )
    mismatches: list[str] = []
    if (tmp_path / "ours_sub.csv").read_bytes() != expected[
        "controller/bytes"
    ].numpy().tobytes():
        mismatches.append("controller/bytes")
    for name, multiplier, modulus, offset, denominator in (
        ("raptor", 7, 41, 0.5, 43),
        ("raptor224", 11, 37, 0.75, 41),
    ):
        raw = np.asarray(
            ((np.arange(48).reshape(4, 12) * multiplier % modulus) + offset)
            / denominator,
            dtype=np.float32,
        )
        for suffix, predictions in (
            ("raw", raw),
            ("sub", (raw.argsort(0).argsort(0) / 3).astype(np.float32)),
        ):
            path = tmp_path / f"{name}_{suffix}.csv"
            _write_scores(
                path,
                values=predictions,
                data=data,
                order=ids,
                decimals=False,
                canonical=suffix == "raw",
            )
            if (
                path.read_bytes()
                != expected[f"{name}_{suffix}/bytes"].numpy().tobytes()
            ):
                mismatches.append(f"{name}_{suffix}/bytes")
            # The native reader must accept the source's complete raw column order.
            path.write_bytes(expected[f"{name}_{suffix}/bytes"].numpy().tobytes())
    parsed: dict[str, NDArray[np.float64]] = {}
    for name in (
        "raptor_raw",
        "raptor_sub",
        "raptor224_raw",
        "raptor224_sub",
        "ours_sub",
    ):
        try:
            parsed[name] = _read_scores(tmp_path / f"{name}.csv", data=data)
        except ValueError:
            mismatches.append(f"{name}/columns")
            continue
        if not np.array_equal(parsed[name], expected[f"{name}/parsed"].numpy()):
            mismatches.append(f"{name}/parsed")
    if len(parsed) == 5:
        correlations = _correlations(
            parsed["raptor_raw"],
            raw224=parsed["raptor224_raw"],
            ours=parsed["ours_sub"],
            sub384=parsed["raptor_sub"],
            sub224=parsed["raptor224_sub"],
        )
        for name, values_by_label in correlations.items():
            if not np.array_equal(
                np.asarray([values_by_label[label] for label in FINDINGS]),
                expected[f"correlation/{name}"].numpy(),
            ):
                mismatches.append(f"correlation/{name}")
        predictions = exp000().blend.make()(
            [parsed["raptor_raw"], parsed["raptor224_raw"], parsed["ours_sub"]],
            [
                ~np.equal(parsed["raptor_raw"], 0.5).all(axis=1),
                ~np.equal(parsed["raptor224_raw"], 0.5).all(axis=1),
                result.scored,
            ],
        )
        _write_scores(
            tmp_path / "submission.csv",
            values=predictions,
            data=data,
            order=ids,
            decimals=False,
        )
        if (tmp_path / "submission.csv").read_bytes() != expected[
            "submission/bytes"
        ].numpy().tobytes():
            mismatches.append("submission/bytes")
        if not np.array_equal(
            _read_scores(tmp_path / "submission.csv", data=data),
            expected["submission/parsed"].numpy(),
        ):
            mismatches.append("submission/parsed")
    assert not mismatches, mismatches


def test_native_recipe_retains_failure_receipt_before_propagating(
    tmp_path: Path,
) -> None:
    config = _recipe(tmp_path)
    Path(config.raptor384.checkpoint.file.path).write_bytes(b"Altered checkpoint")
    with pytest.raises(RuntimeError, match="hashes to"):
        config.make().run()
    record = json_loads(
        (Path(config.working_dir) / "run_receipt.json").read_text(),
        dict[str, object],
    )
    assert record["status"] == "failed"
    assert "hashes to" in str(record["error"])


def test_native_recipe_rejects_arguments_and_protected_or_existing_outputs(
    tmp_path: Path,
) -> None:
    config = _recipe(tmp_path)
    with pytest.raises(ValueError, match="configuration fields"):
        config.make().run("unexpected")
    config.working_dir = Path(config.data.input_dir)
    with pytest.raises(ValueError, match="input"):
        config.make().run()
    config.working_dir = Path(config.data.input_dir) / "output"
    with pytest.raises(ValueError, match="input"):
        config.make().run()
    config.working_dir = tmp_path / "exists"
    Path(config.working_dir).mkdir()
    with pytest.raises(FileExistsError):
        config.make().run()


def test_released_second_raptor_arm_retains_its_distinct_policy() -> None:
    config = exp000().copy_tree().finalize()
    arm = config.raptor224
    assert isinstance(arm.model.backbone, CoAtNet.Config)
    assert arm.model.backbone.image_size == (224, 224)
    assert arm.windows.crop_size == 307
    assert arm.views == [unchanged]
    assert arm.score.dtypes_autocast == [torch.float16, None]
    assert arm.score.check_logits is accept_logits
    assert (
        arm.checkpoint.file.sha256
        == "3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6"
    )
    assert arm.model.chunk_size == math.inf
    assert config.budget.seconds - config.budget.reserve_evaluation_sec == 8.3 * 3_600
    assert config.working_dir == Path("/opt/scratch/runs/rsna-knee/exp000")
    assert arm.checkpoint.file.path == Path(
        "/opt/scratch/datasets/rsna-knee-raptor-swa-v1/"
        "raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt",
    )


def test_prediction_reader_checks_complete_identity_and_column_order(
    tmp_path: Path,
) -> None:
    data = study_inputs(tmp_path).make()
    path = tmp_path / "prediction.csv"
    path.write_text("StudyInstanceUID,wrong\nstudy,0.5\n")
    with pytest.raises(ValueError, match="columns differ"):
        _read_scores(path, data=data)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *data.labels])
        writer.writerows([["study", *([0.5] * 12)]] * 2)
    with pytest.raises(ValueError, match="rows differ"):
        _read_scores(path, data=data)


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1])
def test_final_submission_rejects_nonfinite_and_out_of_range_values(bad: float) -> None:
    with pytest.raises(ValueError, match="invalid probabilities"):
        _validate_predictions(np.full((2, 12), fill_value=bad, dtype=np.float32))


def test_generated_tied_ordinal_csvs_pass_without_changing_average_tie_blend(
    tmp_path: Path,
) -> None:
    raw = np.repeat(
        np.asarray([0.1, 0.1, 0.9, 0.9], dtype=np.float32)[:, None],
        repeats=len(FINDINGS),
        axis=1,
    )
    raw[:, 0] = 0.2
    parsed, ranks = _rank_csvs(tmp_path, raw=raw)
    diagnostic = _correlations(
        parsed,
        raw224=parsed,
        ours=parsed,
        sub384=ranks,
        sub224=ranks,
    )
    assert np.isnan(diagnostic["raptor_raw_vs_sub"][FINDINGS[0]])
    assert diagnostic["raptor_raw_vs_sub"][FINDINGS[1]] == round((4 / 5) ** 0.5, 4)
    blend = exp000().blend.make()(
        [parsed, parsed, parsed],
        [np.ones(len(raw), dtype=np.bool_)] * 3,
    )
    assert np.array_equal(
        blend[:, 0],
        np.full(len(raw), fill_value=2.5 / 4, dtype=np.float32),
    )
    assert np.array_equal(
        blend[:, 1:],
        np.repeat(
            np.asarray([1.5, 1.5, 3.5, 3.5])[:, None] / 4,
            repeats=len(FINDINGS) - 1,
            axis=1,
        ),
    )


@pytest.mark.parametrize("rows", [2, 4])
@pytest.mark.parametrize("arm", ["raptor", "raptor224"])
@pytest.mark.parametrize("constant_column", [0, -1])
def test_monotone_incorrect_ordinal_csvs_fail_for_either_arm(
    tmp_path: Path,
    rows: int,
    arm: str,
    constant_column: int,
) -> None:
    raw = np.repeat(
        np.linspace(0.1, stop=0.9, num=rows, dtype=np.float32)[:, None],
        repeats=len(FINDINGS),
        axis=1,
    )
    raw[:, constant_column] = 0.2
    parsed, ranks = _rank_csvs(tmp_path, raw=raw)
    incorrect = ranks / 2
    with pytest.raises(RuntimeError, match=arm + r"_raw_vs_sub.*submission ranks"):
        _correlations(
            parsed,
            raw224=parsed,
            ours=parsed,
            sub384=incorrect if arm == "raptor" else ranks,
            sub224=incorrect if arm == "raptor224" else ranks,
        )


def test_raw_submission_correlations_preserve_checks_and_tied_missing_cases(
    tmp_path: Path,
) -> None:
    raw, ranks = _rank_csvs(
        tmp_path,
        raw=np.repeat(
            np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32)[:, None],
            repeats=len(FINDINGS),
            axis=1,
        ),
    )
    result = _correlations(
        raw,
        raw224=raw,
        ours=raw[::-1].copy(),
        sub384=ranks,
        sub224=ranks,
    )
    assert set(result["raptor_raw_vs_sub"].values()) == {1.0}
    assert set(result["raptor_vs_ours"].values()) == {-1.0}
    with pytest.raises(RuntimeError, match="submission ranks"):
        _correlations(
            raw,
            raw224=raw,
            ours=raw,
            sub384=ranks[::-1].copy(),
            sub224=ranks,
        )
    constant, constant_ranks = _rank_csvs(
        tmp_path,
        raw=np.full((3, 12), fill_value=0.5, dtype=np.float32),
    )
    missing = np.full((3, 12), fill_value=np.nan)
    result = _correlations(
        constant,
        raw224=constant,
        ours=missing,
        sub384=constant_ranks,
        sub224=constant_ranks,
    )
    assert all(
        np.isnan(value) for values in result.values() for value in values.values()
    )


@pytest.mark.parametrize("constant_column", [0, -1])
@pytest.mark.parametrize("arm", ["raptor", "raptor224"])
def test_undefined_correlations_cannot_mask_finite_rank_failures(
    tmp_path: Path,
    constant_column: int,
    arm: str,
) -> None:
    raw = np.repeat(
        np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)[:, None],
        repeats=len(FINDINGS),
        axis=1,
    )
    raw[:, constant_column] = 0.5
    parsed, ranks = _rank_csvs(tmp_path, raw=raw)
    reversed_ranks = ranks[::-1].copy()
    with pytest.raises(RuntimeError, match=arm + r"_raw_vs_sub.*submission ranks"):
        _correlations(
            parsed,
            raw224=parsed,
            ours=parsed,
            sub384=reversed_ranks if arm == "raptor" else ranks,
            sub224=reversed_ranks if arm == "raptor224" else ranks,
        )


@pytest.mark.parametrize("rows", [3, 7, 11, 257])
def test_vectorized_rank_text_matches_complete_csv_readback(
    tmp_path: Path,
    rows: int,
) -> None:
    raw = np.arange(rows * len(FINDINGS), dtype=np.float32).reshape(
        rows,
        len(FINDINGS),
    )
    parsed, submitted = _rank_csvs(tmp_path, raw=raw)
    ranks = (raw.argsort(0).argsort(0) / (rows - 1)).astype(np.float32)
    assert np.array_equal(submitted, ranks.astype(str).astype(np.float64))
    result = _correlations(
        parsed,
        raw224=parsed,
        ours=parsed,
        sub384=submitted,
        sub224=submitted,
    )
    assert set(result["raptor_raw_vs_sub"].values()) == {1.0}
    assert set(result["raptor224_raw_vs_sub"].values()) == {1.0}


def _rank_csvs(
    root: Path,
    *,
    raw: NDArray[np.float32],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    ids = [f"study{index}" for index in range(len(raw))]
    (root / "test.csv").write_text("StudyInstanceUID\n" + "\n".join(ids) + "\n")
    with (root / "sample_submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *reversed(FINDINGS)])
        writer.writerows([[uid, *([0.0] * len(FINDINGS))] for uid in ids[::-1]])
    (root / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n",
    )
    config = StudyData.Config()
    config.input_dir = root
    data = config.make()
    ranks = (raw.argsort(0).argsort(0) / max(len(raw) - 1, 1)).astype(np.float32)
    for name, values in (("raw", raw), ("sub", ranks)):
        _write_scores(
            root / f"{name}.csv",
            values=values,
            data=data,
            order=data.ids,
            decimals=False,
            canonical=name == "raw",
        )
    return (
        _read_scores(root / "raw.csv", data=data),
        _read_scores(root / "sub.csv", data=data),
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
