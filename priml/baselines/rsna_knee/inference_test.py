"""Check the released inference stages on frozen literal model outputs."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial, partialmethod
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

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
    FLUID_COEFFICIENTS,
    STATE_COEFFICIENTS,
    corrupt_deflate,
    frozen_folds,
    frozen_raptor_logits,
    postprocess_assets,
    r7_stage_config,
    raptor_stage_config,
    released_state,
    study_inputs,
)
from priml.model.coatnet import CoAtNet
from priml.model.vnet import VNet


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from numpy.typing import NDArray

    from priml.baselines.rsna_knee.imaging import StudyInputs
    from priml.model.ae.checkpoint import LocalFile


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


def test_rank_blend_falls_back_on_the_members_that_scored_each_study() -> None:
    config = exp000().blend
    for group in config.groups:
        for member in group.members:
            member.max_failure_fraction = 1.0
    # Rows: all scored; Raptor failed; Raptor-224 failed; R7 failed; all failed.
    scored = [
        np.array([True, False, True, True, False]),
        np.array([True, True, False, True, False]),
        np.array([True, True, True, False, False]),
    ]
    scores = [
        np.array([[0.1], [0.9], [0.3], [0.2], [0.5]]),
        np.array([[0.6], [0.5], [0.9], [0.4], [0.5]]),
        np.array([[0.2], [0.9], [0.5], [0.1], [0.5]]),
    ]
    # Each member ranks only its own scored rows; each weight renormalizes over them.
    raptor = [0.65 * (1 / 3) + 0.35 * 1.0, 2 / 3, 1.0, 0.65 * (2 / 3) + 0.35 * (1 / 3)]
    r7 = [1 / 3, 1.0, 2 / 3]
    expected = [
        0.5 * raptor[0] + 0.5 * r7[0],
        0.5 * raptor[1] + 0.5 * r7[1],
        0.5 * raptor[2] + 0.5 * r7[2],
        raptor[3],
        0.5,
    ]
    np.testing.assert_array_equal(
        config.make()(scores, scored),
        np.array(expected, dtype=np.float32)[:, None],
    )


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
    logits = frozen_raptor_logits()
    path = tmp_path / "model.pt"
    torch.save({"model": {"logits": logits}}, path)
    asset = released_state(path, key="model").make()
    assert torch.equal(asset.state()["logits"], logits)
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


def _r7_inputs() -> tuple[Tensor, Tensor]:
    """Return two six-pixel uint8 windows and their nine metadata columns."""
    # R7Score centers its crop on both axes by the last dim, so windows are square.
    images = (torch.arange(2 * 3 * 6 * 6) % 256).to(torch.uint8).reshape(2, 3, 6, 6)
    return images, torch.arange(2 * 9, dtype=torch.float32).reshape(2, 9) / 8


class _Fold:
    def __init__(self, folds: dict[str, Tensor], index: int, *, rescue: bool) -> None:
        self.folds, self.index, self.rescue = folds, index, rescue
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
        full, meta = _r7_inputs()
        # R7Score crops the centered four pixels and scores one study at a time.
        assert torch.equal(images, full[None, :, :, 1:5, 1:5])
        assert torch.equal(metadata, meta[None])
        assert torch.equal(mask, torch.ones(1, 2, dtype=torch.bool))
        self.calls += 1
        logits = self.folds["logits"][self.index].clone()
        if self.rescue and self.calls == 1:
            logits[0, 0] = float("nan")
        return (
            logits,
            self.folds["states"][self.index].clone(),
            self.folds["binaries"][self.index].clone(),
        )

    def state_readout(self, states: Tensor, binaries: Tensor) -> Tensor:
        for name, value in (("states", states), ("binaries", binaries)):
            assert value.dtype == torch.float32
            torch.testing.assert_close(
                value,
                self.folds[name][self.index].float(),
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        # R7Score scores one study and keeps row 0 of its 84-column state readout.
        return torch.full((1, 84), float(self.index))


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


def _sigmoid(logits: Tensor) -> NDArray[np.float32]:
    return torch.sigmoid(logits.float()).numpy()


def _views() -> list[Tensor]:
    """Return a plain and a mirrored set of 94 normalized 2x2-pixel windows."""
    # RaptorWindows interpolates every window to a square output_size.
    plain = torch.arange(94 * 3 * 2 * 2, dtype=torch.float32).reshape(94, 3, 2, 2) / 64
    return [plain, -plain]


def test_all_four_folds_crop_rescue_nonfinite_findings_and_read_every_state() -> None:
    folds = frozen_folds()
    # A state NaN does not trigger the released finding-logit rescue.
    folds["states"][2, 0, 3, 1] = float("nan")
    models = [_Fold(folds, index, rescue=index == 1) for index in range(4)]
    images, metadata = _r7_inputs()
    actual = R7Score.Config(device="cpu", crop=4).make()(models, images, metadata)
    assert actual.logits.dtype == torch.float32
    assert torch.equal(actual.logits, folds["logits"][:, 0].float())
    assert torch.equal(
        actual.states,
        torch.arange(4, dtype=torch.float32)[:, None].expand(4, 84),
    )
    assert [model.calls for model in models] == [1, 2, 1, 1]


def test_raptor_pair_sigmoids_each_pass_before_the_float32_average() -> None:
    logits, views = frozen_raptor_logits(), _views()
    model = _Raptor(torch.stack([view[None] for view in views]), logits).eval()
    actual = RaptorScore.Config(device="cpu").make()(model, views)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(
        actual,
        (_sigmoid(logits[0, 0]) + _sigmoid(logits[1, 0])) / 2,
    )
    assert model.calls == 2


def test_accepting_policy_passes_nonfinite_and_saturated_logits_to_the_sigmoid() -> (
    None
):
    plain = _views()[0]
    logits = torch.tensor([[-math.inf, -1e4, 0.0, 1e4, math.inf, math.nan] * 2])
    model = _Raptor(plain[None, None], logits[None]).eval()
    score = RaptorScore.Config(device="cpu", check_logits=accept_logits).make()
    np.testing.assert_array_equal(
        score(model, [plain]),
        np.array([0.0, 0.0, 0.5, 1.0, 1.0, math.nan] * 2, dtype=np.float32),
    )


def test_runtime_error_retries_the_complete_same_input_at_the_next_precision() -> None:
    logits, plain = frozen_raptor_logits(), _views()[0]
    model = _Raptor(
        torch.stack([plain[None]] * 2),
        logits[:1],
        failure=RuntimeError,
    ).eval()
    score = RaptorScore.Config(
        device="cpu",
        dtypes_autocast=[torch.float16, None],
        check_logits=accept_logits,
    ).make()
    actual = score(model, [plain])
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, _sigmoid(logits[0, 0]))
    assert model.calls == 2


def test_first_arm_does_not_rescue_runtime_failures_or_nonfinite_logits() -> None:
    plain = _views()[0]
    inputs, logits = plain[None, None], frozen_raptor_logits()[:1]
    model = _Raptor(inputs, logits, failure=RuntimeError).eval()
    score = RaptorScore.Config(device="cpu").make()
    with pytest.raises(RuntimeError, match=r"^Frozen model failure\.$"):
        score(model, [plain])
    assert model.calls == 1
    model = _Raptor(inputs, logits * float("nan")).eval()
    with pytest.raises(FloatingPointError, match=r"^Raptor logits are nonfinite\.$"):
        score(model, [plain])
    assert model.calls == 1


def test_disabled_finite_policy_performs_no_finite_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, views = frozen_raptor_logits(), _views()
    model = _Raptor(torch.stack([view[None] for view in views]), logits).eval()
    calls: list[Tensor] = []
    original = torch.isfinite

    def observe(value: Tensor) -> Tensor:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(torch, "isfinite", observe)
    config = RaptorScore.Config()
    config.device = "cpu"
    config.check_logits = accept_logits
    actual = config.make()(model, views)
    assert calls == []
    np.testing.assert_array_equal(
        actual,
        (_sigmoid(logits[0, 0]) + _sigmoid(logits[1, 0])) / 2,
    )


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
    logits, plain = frozen_raptor_logits(), _views()[0]
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
        torch.stack([plain[None]] * 2),
        logits[:1],
        failure=RuntimeError,
    ).eval()
    config = RaptorScore.Config(
        device=device,
        check_logits=accept_logits,
        dtypes_autocast=[torch.float16, None],
    )
    np.testing.assert_array_equal(
        config.make()(model, [plain]),
        _sigmoid(logits[0, 0]),
    )
    assert transfers == [torch.device(device)]
    assert contexts == [
        (device, torch.float16, device == "cuda"),
        (device, None, False),
    ]
    assert caches == ([1] if device == "cuda" else [])


def test_runtime_error_during_device_to_host_copy_retries_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits, plain = frozen_raptor_logits(), _views()[0]
    original = Tensor.cpu
    copies: list[int] = []

    monkeypatch.setattr(
        Tensor,
        "cpu",
        partialmethod(_copy, copies=copies, original=original),
    )
    model = _Raptor(torch.stack([plain[None]] * 2), logits[:1]).eval()
    score = RaptorScore.Config(
        device="cpu",
        check_logits=accept_logits,
        dtypes_autocast=[torch.float16, None],
    ).make()
    np.testing.assert_array_equal(score(model, [plain]), _sigmoid(logits[0, 0]))
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
    contexts: list[tuple[str, torch.dtype | None, bool]] = []
    monkeypatch.setattr(torch, "autocast", partial(_autocast, calls=contexts))
    folds = frozen_folds()
    images, metadata = _r7_inputs()
    result = (
        R7Score.Config(device=device, crop=4)
        .make()
        .forward(
            _Fold(folds, 0, rescue=False),
            images[None, :, :, 1:5, 1:5],
            metadata[None],
            # R7Score.__call__ masks one study's windows before calling forward.
            torch.ones(1, 2, dtype=torch.bool),
            dtype=dtype,
        )
    )
    assert contexts == [(device, dtype, device == "cuda" and dtype is not None)]
    for actual, name in zip(result, ("logits", "states", "binaries"), strict=True):
        assert actual.dtype == torch.float32
        assert torch.equal(actual, folds[name][0].float())


def test_scoring_requires_at_least_one_autocast_precision() -> None:
    score = RaptorScore.Config(device="cpu", dtypes_autocast=[]).make()
    with pytest.raises(ValueError, match="at least one autocast precision"):
        score(torch.nn.Identity(), [torch.zeros(2, 3)])


def _path(file: LocalFile.Config) -> Path:
    return Path(file.path)


def _stacker_inputs() -> tuple[Tensor, Tensor]:
    """Return four folds' logits and state readouts with dyadic stacker features."""
    logits = ((torch.arange(4 * 12) % 5 - 2) / 8).reshape(4, 12)
    logits[:, [FINDINGS.index("Effusion"), FINDINGS.index("Baker's")]] = 1.0
    states = torch.zeros(4, 84)
    # Baker's six categories follow the first nine findings' 51 categories.
    states[:, 51:57] = torch.tensor([0.5, 0.25, 0.125, 0.0625, 0.5, 0.25])
    states[:, 80] = 0.375
    return logits, states


def _logistic(logit: float) -> float:
    """Return the logistic of a nonnegative logit."""
    return 1 / (1 + math.exp(-logit))


@pytest.mark.parametrize(
    ("volume", "log_volume"),
    [
        (None, None),
        (0.0, 0.0),
        # The same NumPy float64 log1p call as production.
        (13.7, np.log1p(np.array(13.7, dtype=np.float64)).item()),
        (-1.0, None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_both_stackers_replace_their_findings_and_serialize_nine_digits(
    tmp_path: Path,
    volume: float | None,
    log_volume: float | None,
) -> None:
    logits, states = _stacker_inputs()
    fluid = None if volume is None else {"eff_sag_p99_open": volume}
    result = postprocess_assets(tmp_path).make()(logits, states, fluid)
    assert not result.failures
    expected = torch.sigmoid(logits.mean(0)).double().numpy()
    b0, b_z, b_lv, b_miss = FLUID_COEFFICIENTS
    # Missing, negative or nonfinite volumes use the missing-volume weight instead.
    expected[FINDINGS.index("Effusion")] = _logistic(
        b0 + (b_z + (b_miss if log_volume is None else b_lv * log_volume)),
    )
    # Baker's: intercept 0.5, z weight 0.25 and six dyadic category features.
    category = sum(
        weight * value
        for weight, value in zip(
            STATE_COEFFICIENTS[2:],
            (0.5, 0.25, 0.125, 0.0625, 0.5, 0.25),
            strict=True,
        )
    )
    expected[FINDINGS.index("Baker's")] = _logistic(
        STATE_COEFFICIENTS[0] + STATE_COEFFICIENTS[1] + category,
    )
    np.testing.assert_array_equal(result.probabilities, expected)
    np.testing.assert_array_equal(
        result.serialized,
        [float(f"{value:.9g}") for value in expected],
    )


def test_bad_state_features_preserve_model_and_fluid_probabilities(
    tmp_path: Path,
) -> None:
    logits, states = _stacker_inputs()
    postprocess = postprocess_assets(tmp_path).make()
    # R7Postprocess needs the complete state layout; width one exercises fallback.
    result = postprocess(logits, torch.zeros(4, 1), None)
    base = torch.sigmoid(logits.mean(0)).double().numpy()
    effusion = _logistic(0.25 + 1.0 - 0.75)
    assert result.failures
    assert result.probabilities[9] == base[9]
    assert result.probabilities[7] == effusion
    states[:, 51] = float("nan")
    result = postprocess(logits, states, None)
    assert result.failures
    assert result.probabilities[9] == base[9]
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
    logits, states = _stacker_inputs()
    result = config.make()(logits, states, None)
    assert not result.failures
    assert torch.equal(
        torch.from_numpy(result.probabilities),
        torch.sigmoid(logits.mean(0)).double(),
    )


def test_optional_volume_features_and_dropped_features_keep_coefficient_order(
    tmp_path: Path,
) -> None:
    logits, states = _stacker_inputs()
    (tmp_path / "default").mkdir()
    default = postprocess_assets(tmp_path / "default").make()(logits, states, None)
    config = postprocess_assets(tmp_path)
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
        [*STATE_COEFFICIENTS[1:], 0.0],
    )
    result = config.make()(logits, states, {"bak_sag_p99_open": 2.5})
    assert not result.failures
    # A zero-weight missing-volume feature leaves the default probabilities intact.
    np.testing.assert_array_equal(result.probabilities, default.probabilities)


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
    logits, states = _stacker_inputs()
    (tmp_path / "default").mkdir()
    default = postprocess_assets(tmp_path / "default").make()(
        logits,
        states,
        {"eff_sag_p99_open": 13.7},
    )
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
    result = config.make()(logits, states, {"bak_sag_p99_open": 13.7})
    np.testing.assert_array_equal(result.probabilities, default.probabilities)


def test_binary_state_features_and_volume_coefficients_affect_scores(
    tmp_path: Path,
) -> None:
    config = postprocess_assets(tmp_path)
    logits, states = _stacker_inputs()
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
    result = model(logits, states, None)
    assert result.probabilities.item(1) == _logistic(0.375)
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
    result = config.make()(logits, states, {"bak_sag_p99_open": 2.5})
    log_volume: NDArray[np.float64] = np.log1p(np.array([2.5], dtype=np.float64))
    assert result.probabilities.item(9) == _logistic(log_volume.item(0))


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


def _pair() -> Tensor:
    """Return the frozen Raptor's two-pass probability average."""
    logits = frozen_raptor_logits()
    return torch.from_numpy((_sigmoid(logits[0, 0]) + _sigmoid(logits[1, 0])) / 2)


def test_full_raptor_stage_runs_both_views_even_for_an_empty_study(
    tmp_path: Path,
) -> None:
    data = study_inputs(tmp_path).make()
    config = raptor_stage_config(tmp_path / "raptor.pt", logits=frozen_raptor_logits())
    config.views = [unchanged, anatomical_mirror]
    result = config.make()(data)
    assert result.values.dtype == np.float32
    assert torch.equal(torch.from_numpy(result.values), _pair().repeat(2, 1))
    assert result.scored.tolist() == [True, True]
    assert result.failures == {}


def test_four_fold_anatomy_stage_keeps_stackers_and_mean_fills_failed_studies(
    tmp_path: Path,
) -> None:
    data = study_inputs(tmp_path).make()
    result = r7_stage_config(tmp_path).make()(data)
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
    data = _damaged_study(tmp_path, damage=damage)
    config = raptor_stage_config(tmp_path / "raptor.pt", logits=frozen_raptor_logits())
    config.views = [unchanged, anatomical_mirror]
    result = config.make()(data)
    assert data.ids == ["study", "empty", "zz-good"]
    assert result.scored.tolist() == [True, True, True]
    assert result.failures == {}
    assert torch.equal(torch.from_numpy(result.values), _pair().repeat(3, 1))
    if damage == "spacing":
        assert len(result.notes) == 1
        assert result.notes[0]["uid"] == "study"
        assert "crop" in str(result.notes[0]["preparation_error"])


def test_corrupt_deflate_mean_fills_r7_failure_and_scores_next_study(
    tmp_path: Path,
) -> None:
    data = _damaged_study(tmp_path, damage="deflate")
    result = r7_stage_config(tmp_path).make()(data)
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
    data = study_inputs(tmp_path).make()
    monkeypatch.setattr(data, "load", _missing)
    config = raptor_stage_config(
        tmp_path / "nonfinite.pt",
        logits=frozen_raptor_logits() * float("nan"),
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
    data = study_inputs(tmp_path).make()
    config = raptor_stage_config(tmp_path / "raptor.pt", logits=frozen_raptor_logits())
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
    data = study_inputs(tmp_path).make()
    config = r7_stage_config(tmp_path)
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


def test_ordinal_ranks_break_ties_in_study_order() -> None:
    # Over 16 rows: NumPy's default sort keeps short tie runs in order, hiding a bug.
    rows = np.arange(17)
    values = np.stack([np.full(17, 0.5), np.where(rows % 2, 0.25, 0.75)], axis=1)
    expected = np.stack([rows, np.where(rows % 2, rows // 2, 8 + rows // 2)], axis=1)
    assert np.array_equal(_ordinal_ranks(values), (expected / 16).astype(np.float32))
    assert _ordinal_ranks(np.array([[0.5, 0.25]])).tolist() == [[0.0, 0.0]]


def _recipe(root: Path) -> PublicRecipe.Config:
    config = PublicRecipe.Config()
    config.base_dir = None
    data = root / "raw"
    data.mkdir()
    config.data = study_inputs(data)
    config.working_dir = root / "output"
    config.raptor384 = raptor_stage_config(
        root / "raptor.pt",
        logits=frozen_raptor_logits(),
    )
    config.raptor384.views = [unchanged, anatomical_mirror]
    config.raptor224 = raptor_stage_config(
        root / "raptor.pt",
        logits=frozen_raptor_logits(),
    )
    config.r7 = r7_stage_config(root)
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


def _competition(root: Path, *, ids: list[str], sample: list[str]) -> StudyData:
    """Write competition CSVs without series rows and read them as study data."""
    with (root / "test.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID"])
        writer.writerows([[uid] for uid in ids])
    with (root / "sample_submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *reversed(FINDINGS)])
        writer.writerows([[uid, *([0.0] * 12)] for uid in sample])
    (root / "test_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n",
    )
    return StudyData.Config(input_dir=root).make()


@pytest.mark.parametrize("layout", ["test_series", "test_images"])
def test_r7_controller_orders_studies_by_file_count_and_interleaves_halves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    ids = ["zeta", "alpha", "empty", "beta"]
    data = _competition(tmp_path, ids=ids, sample=["beta", "empty", "zeta", "alpha"])
    for uid, count in zip(ids, (5, 8, 0, 2), strict=True):
        directory = tmp_path / "test_series" / uid / "series"
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index}.dcm").touch()
    if layout == "test_images":
        (tmp_path / "test_series").rename(tmp_path / layout)
    calls: list[str] = []

    def load(uid: str) -> StudyInputs:
        calls.append(uid)
        raise OSError("Frozen worker failure.")

    monkeypatch.setattr(data, "load", load)
    assert not r7_stage_config(tmp_path).make()(data).scored.any()
    # Descending file counts alpha, zeta, beta, empty; even positions run first.
    assert calls == ["alpha", "beta", "zeta", "empty"]


def test_mean_fill_and_decimal_csv_follow_the_sample_order(tmp_path: Path) -> None:
    ids, sample = ["zeta", "alpha", "empty", "beta"], ["beta", "empty", "zeta", "alpha"]
    data = _competition(tmp_path, ids=ids, sample=sample)
    columns = np.arange(12) / 64
    result = StageScores(
        values=np.array([0.25, 0.5, 0.0, 0.75])[:, None] + columns,
        scored=np.array([True, True, False, True]),
        failures={"empty": "frozen worker failure"},
    )
    _mean_fill(result, ids=ids, sample_ids=sample)
    # The scored rows 0.75, 0.25 and 0.5 (plus each column offset) average to 0.5.
    np.testing.assert_array_equal(result.values[2], 0.5 + columns)
    path = tmp_path / "ours_sub.csv"
    _write_scores(path, values=result.values, data=data, order=sample, decimals=True)
    rows = {"beta": 0.75, "empty": 0.5, "zeta": 0.25, "alpha": 0.5}
    assert path.read_bytes().decode() == "".join(
        line + "\r\n"
        for line in [
            ",".join(["StudyInstanceUID", *reversed(FINDINGS)]),
            *(
                ",".join([uid, *(str(rows[uid] + offset) for offset in columns[::-1])])
                for uid in sample
            ),
        ]
    )


def test_raw_csvs_use_canonical_columns_and_submissions_the_sample_columns(
    tmp_path: Path,
) -> None:
    data = _competition(tmp_path, ids=["a", "b"], sample=["b", "a"])
    values = np.array([[1 / 3] * 12, [0.5] * 12], dtype=np.float32)
    for name, canonical, labels in (
        ("raw", True, list(FINDINGS)),
        ("sub", False, list(reversed(FINDINGS))),
    ):
        path = tmp_path / f"{name}.csv"
        _write_scores(
            path,
            values=values,
            data=data,
            order=data.ids,
            decimals=False,
            canonical=canonical,
        )
        # Float32 text keeps the shortest round-trip digits and newline endings.
        assert path.read_text() == (
            f"StudyInstanceUID,{','.join(labels)}\n"
            f"a,{','.join(['0.33333334'] * 12)}\n"
            f"b,{','.join(['0.5'] * 12)}\n"
        )
        np.testing.assert_array_equal(
            _read_scores(path, data=data),
            [[float("0.33333334")] * 12, [0.5] * 12],
        )


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
    order = raw.argsort(0, kind="stable")
    ranks = (order.argsort(0) / max(len(raw) - 1, 1)).astype(np.float32)
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
