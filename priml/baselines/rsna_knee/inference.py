"""Released inference: stages, execution, scoring, stackers, blend and pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Protocol, Self, TypedDict, cast, override

import csv
import hashlib
import json
import math
import time

from configgle import Fig, Makeable
from numpy.typing import NDArray
from torch import Tensor, nn
from treekle import MutablePlainTree, from_plain, json_loads, to_plain

import numpy as np
import torch

from priml.baselines.rsna_knee.data import FINDINGS
from priml.baselines.rsna_knee.device import release_memory, synchronize
from priml.baselines.rsna_knee.imaging import (
    REGION_VIEWS,
    RaptorPrepare,
    RaptorWindows,
    ReorientFn,
    StudyData,
    StudyPreparation,
    series_tree,
    unchanged,
)
from priml.baselines.rsna_knee.model import R7, Raptor, ReleasedVNet
from priml.math.stats import average_ranks, ordinal_ranks, spearman
from priml.model.ae.checkpoint import LocalFile
from priml.model.coatnet import CoAtNet
from priml.model.custom_types import TensorModule
from priml.model.resnet import ResNet
from priml.paths import resolve_working_dir, validated_output_path
from priml.train.budget import WallBudget


if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

    from torch import _dynamo

    from priml.math.custom_types import TensorFn
else:
    from wrapt import lazy_import

    _dynamo = lazy_import("torch._dynamo")


class RankMember:
    """One scored model, ranked over its own successful studies."""

    class Config(Fig["RankMember"]):
        name: str = ""
        """Model name reported when too many of its studies failed."""

        weight: float = 1.0
        """Weight within its group, among the members that scored a study."""

        max_failure_fraction: float = 0.03
        """Largest fraction of studies this model may fail to score."""


class RankGroup:
    """Models blended by weight before the groups are blended with each other."""

    class Config(Fig["RankGroup"]):
        weight: float = 1.0
        """Weight among the groups that scored a study."""

        members: list[RankMember.Config] = field(
            default_factory=list[RankMember.Config],
        )
        """Models whose ranks are averaged within the group."""


class RankBlend:
    """Blend per-model percentile ranks, falling back on whichever models scored."""

    class Config(Fig["RankBlend"]):
        groups: list[RankGroup.Config] = field(default_factory=list[RankGroup.Config])
        """Weighted groups; scores are given to the call in group-then-member order."""

    def __init__(self, config: Config) -> None:
        members = [member for group in config.groups for member in group.members]
        if not members or any(not group.members for group in config.groups):
            raise ValueError("Every blend group requires at least one member.")
        for value in (
            *(group.weight for group in config.groups),
            *(member.weight for member in members),
            *(member.max_failure_fraction for member in members),
        ):
            if math.isnan(value) or value < 0.0 or value > 1.0:
                raise ValueError(
                    "Blend weights and failure fractions must lie in [0, 1].",
                )
        self.config = config

    def __call__(
        self,
        scores: Sequence[NDArray[np.float64]],
        scored: Sequence[NDArray[np.bool_]],
    ) -> NDArray[np.float32]:
        """Rank each model on its successful rows and combine finding scores.

        Args:
          scores: Study-by-finding confidence scores of every member, in
            group-then-member order.
          scored: Whether each member scored each study, in the same order.

        Returns:
          predictions: Float32 scores in the supplied study and finding order.

        Raises:
          RuntimeError: A member failed on more studies than it may.

        """
        members = [member for group in self.config.groups for member in group.members]
        rows, findings = cast(tuple[int, int], scores[0].shape)
        if (
            len(scores) != len(members)
            or len(scored) != len(members)
            or any(value.shape != (rows, findings) for value in scores)
            or any(value.shape != (rows,) for value in scored)
        ):
            raise ValueError("Model scores and scored flags must align by study.")
        for member, valid in zip(members, scored, strict=True):
            failed = int((~valid).sum())
            if failed > member.max_failure_fraction * rows:
                raise RuntimeError(
                    f"{member.name} failed on {failed} of {rows} studies.",
                )
        out = np.zeros((rows, findings))
        for j in range(findings):
            offset = 0
            group_ranks: list[NDArray[np.float64]] = []
            group_valid: list[NDArray[np.bool_]] = []
            for group in self.config.groups:
                chosen = range(offset, offset + len(group.members))
                group_ranks.append(
                    _combine(
                        [
                            percentiles_on(scores[i][:, j], valid=scored[i])
                            for i in chosen
                        ],
                        valid=[scored[i] for i in chosen],
                        weights=[member.weight for member in group.members],
                    ),
                )
                group_valid.append(np.any([scored[i] for i in chosen], axis=0))
                offset += len(group.members)
            blended = _combine(
                group_ranks,
                valid=group_valid,
                weights=[group.weight for group in self.config.groups],
            )
            out[:, j] = np.where(np.any(group_valid, axis=0), blended, 0.5)
        return out.astype(np.float32)


def percentiles_on(
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Rank valid non-NaN rows with ascending average ties divided by row count.

    Args:
      values: One finding in original study order.
      valid: Rows with a successful prediction from this model.

    Returns:
      percentiles: Source-normalized ranks, NaN outside valid non-NaN rows.

    """
    ranks = np.full(len(values), np.nan)
    selected = valid & ~np.isnan(values)
    if selected.any():
        present = values[selected]
        with torch.device("cpu"):
            ranks[selected] = (average_ranks(present).numpy() + 1) / len(present)
    return ranks


def _combine(
    ranks: Sequence[NDArray[np.float64]],
    valid: Sequence[NDArray[np.bool_]],
    weights: Sequence[float],
) -> NDArray[np.float64]:
    """Average ranks by weight over the entries valid in each row, NaN if none."""
    masked = [weight * flag for weight, flag in zip(weights, valid, strict=True)]
    total = sum(masked[1:], start=masked[0])
    combined = np.zeros(len(total))
    with np.errstate(divide="ignore", invalid="ignore"):
        for rank, flag, weight in zip(ranks, valid, masked, strict=True):
            combined = combined + np.where(flag, weight / total * rank, 0.0)
    return np.where(total > 0, combined, np.nan)


class FoldExecution(Protocol):
    """Run bound R7 folds over one study and report how they ran."""

    receipt: dict[str, object]

    def bind(self, models: Sequence[R7], *, score: R7Score) -> None:
        """Take ownership of the loaded folds and the scorer that drives them."""
        ...

    def __call__(self, images: Tensor, metadata: Tensor) -> FoldScores:
        """Return every fold's float32 finding and state readouts."""
        ...


class _BoundFolds:
    """Folds and the scorer that drives them, supplied after construction."""

    def __init__(self) -> None:
        self.models: Sequence[R7] = []
        self.score: R7Score | None = None

    def bind(self, models: Sequence[R7], *, score: R7Score) -> None:
        """Take ownership of the loaded folds and the scorer that drives them."""
        self.models, self.score = models, score

    def scorer(self) -> R7Score:
        """Return the scorer given to `bind`."""
        if self.score is None:
            raise RuntimeError("Folds must be bound before they are scored.")
        return self.score


class EagerExecution(_BoundFolds):
    """Score the folds exactly as loaded, with no compilation."""

    class Config(Fig["EagerExecution"]):
        """Eager execution has no tunables."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.receipt: dict[str, object] = {"requested": False, "active": False}

    @torch.inference_mode()
    def __call__(self, images: Tensor, metadata: Tensor) -> FoldScores:
        """Score every fold eagerly with precision rescue.

        Args:
            images: Complete raw uint8 three-slice windows.
            metadata: Complete physical metadata aligned with those windows.

        Returns:
            scores: Every float32 fold finding and state readout.

        """
        return self.scorer()(self.models, images=images, metadata=metadata)


class EncoderExecution(_BoundFolds):
    """Own complete native folds and their checked compiled/eager execution."""

    class Config(Fig["EncoderExecution"]):
        mode: str = "default"
        """torch.compile mode of every encoder."""

        backend: str = "inductor"
        """torch.compile backend recorded with the acceptance receipt."""

        dynamic: bool = False
        """Whether compiled encoders accept varying shapes."""

        probability_tolerance: float = 1e-3
        """Largest first-study probability gap, and its margin to float32 truth."""

        synthetic_tolerance: float = 0.05
        """Largest compiled/eager probability gap on the synthetic padded shapes."""

        seed: int = 20_261_010
        """Seed of the synthetic images that probe each padded batch size."""

        recompile_limit: int = 64
        """Smallest per-encoder recompilation and cache limit while checking."""

        accumulated_recompile_limit: int = 2_048
        """Smallest accumulated recompilation and cache limit while checking."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.active = False
        self.checked = False
        self.eager: list[TensorModule] = []
        self.compiled: list[TensorModule] = []
        self.folds: list[_Fold] = []
        self.receipt: dict[str, object] = {
            "requested": True,
            "active": False,
            "checked": False,
            "mode": config.mode,
            "backend": config.backend,
            "tol_prob": config.probability_tolerance,
            "fallback": None,
        }

    @override
    def bind(self, models: Sequence[R7], *, score: R7Score) -> None:
        """Take ownership of the loaded folds and the scorer that drives them."""
        super().bind(models, score=score)
        self.eager = [model.enc for model in models]
        self.folds = [_Fold(self, index=index) for index in range(len(models))]

    @torch.inference_mode()
    def __call__(self, images: Tensor, metadata: Tensor) -> FoldScores:
        """Check the first study and score all folds with precision rescue.

        Args:
            images: Complete raw uint8 three-slice windows.
            metadata: Complete physical metadata aligned with those windows.

        Returns:
            scores: Every float32 fold finding and state readout.

        """
        config = self.config
        with _dynamo.config.patch(
            cache_size_limit=max(
                config.recompile_limit,
                _dynamo.config.cache_size_limit,
            ),
            recompile_limit=max(
                config.recompile_limit,
                _dynamo.config.recompile_limit,
            ),
            accumulated_cache_size_limit=max(
                config.accumulated_recompile_limit,
                _dynamo.config.accumulated_cache_size_limit,
            ),
            accumulated_recompile_limit=max(
                config.accumulated_recompile_limit,
                _dynamo.config.accumulated_recompile_limit,
            ),
        ):
            return self._run(images, metadata=metadata)

    def _run(self, images: Tensor, metadata: Tensor) -> FoldScores:
        if not self.checked:
            self.checked = True
            self.receipt["checked"] = True
            start = time.perf_counter()
            try:
                for model, encoder in zip(self.models, self.eager, strict=True):
                    assert isinstance(encoder, nn.Module)
                    compiled = torch.compile(
                        encoder,
                        dynamic=self.config.dynamic,
                        mode=self.config.mode,
                    )
                    self.compiled.append(cast(TensorModule, compiled))
                    model.enc = self.compiled[-1]
                self.active = True
                self.receipt["active"] = True
                self._check(images, metadata=metadata)
            except (
                ArithmeticError,
                LookupError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                self.fallback("Compile check failed: " + repr(error))
            self.receipt["check_seconds"] = round(time.perf_counter() - start, 1)
        return self.scorer()(self.folds, images=images, metadata=metadata)

    def fallback(self, reason: str) -> None:
        """Restore every eager encoder after setup, checking, or scoring failure.

        Args:
            reason: Durable diagnostic identifying the rejected compiled path.

        """
        self.active = False
        self.receipt.update(active=False, fallback=reason[:400])
        for model, encoder in zip(self.models, self.eager, strict=True):
            model.enc = encoder
        self.compiled.clear()
        _dynamo.reset()

    def _forward(
        self,
        index: int,
        image: Tensor,
        meta: Tensor,
        mask: Tensor,
        *,
        compiled: bool,
        dtype: torch.dtype | None,
    ) -> Tensor:
        model = self.models[index]
        saved = model.enc
        model.enc = self.compiled[index] if compiled else self.eager[index]
        try:
            return self.scorer().forward(
                model,
                images=image,
                metadata=meta,
                mask=mask,
                dtype=dtype,
            )[0]
        finally:
            model.enc = saved

    def _compare(self, compiled: Tensor, eager: Tensor, *, prefix: str) -> None:
        if not torch.equal(torch.isfinite(compiled), torch.isfinite(eager)):
            raise ValueError("Compiled and eager outputs differ in finiteness.")
        finite = torch.isfinite(compiled) & torch.isfinite(eager)
        if bool(finite.any()):
            for suffix, difference in (
                ("dlogit", (compiled - eager).abs()),
                (
                    "dprob",
                    (compiled.double().sigmoid() - eager.double().sigmoid()).abs(),
                ),
            ):
                key = prefix + "_" + suffix
                previous = self.receipt.get(key, 0.0)
                assert isinstance(previous, float)
                self.receipt[key] = max(previous, float(difference[finite].max()))

    def _group(
        self,
        image: Tensor,
        meta: Tensor,
        mask: Tensor,
        *,
        compiled: bool,
        dtype: torch.dtype | None,
    ) -> Tensor:
        return torch.stack(
            [
                self._forward(
                    index,
                    image=image,
                    meta=meta,
                    mask=mask,
                    compiled=compiled,
                    dtype=dtype,
                )[0]
                for index in range(len(self.models))
            ],
        )

    def _check(self, images: Tensor, metadata: Tensor) -> None:
        config, device = self.scorer().config, self.scorer().device
        multiple = self.models[0].pad_multiple
        if config.chunk_size <= 0 or config.chunk_size % multiple:
            raise ValueError(
                "Compilation requires a positive multiple of the encoder padding.",
            )
        sizes = list(range(multiple, config.chunk_size + 1, multiple))
        self.receipt["shapes"] = sizes
        generator = torch.Generator().manual_seed(self.config.seed)
        for size in sizes:
            image = torch.randint(
                0,
                256,
                (1, size, 3, config.crop, config.crop),
                generator=generator,
                dtype=torch.uint8,
            ).to(device)
            meta = torch.zeros((1, size, metadata.shape[1]), device=device)
            mask = torch.ones((1, size), dtype=torch.bool, device=device)
            for index in range(len(self.models)):
                compiled = self._forward(
                    index,
                    image=image,
                    meta=meta,
                    mask=mask,
                    compiled=True,
                    dtype=config.dtype_autocast,
                )
                eager = self._forward(
                    index,
                    image=image,
                    meta=meta,
                    mask=mask,
                    compiled=False,
                    dtype=config.dtype_autocast,
                )
                self._compare(compiled, eager=eager, prefix="synthetic")
        synthetic = cast(float, self.receipt.get("synthetic_dprob", 0.0))
        if synthetic > self.config.synthetic_tolerance:
            raise ValueError("The synthetic compiled probability bound failed.")
        full = images.to(device, non_blocking=True)
        offset = (full.shape[-1] - config.crop) // 2
        image = full[
            None,
            :,
            :,
            offset : offset + config.crop,
            offset : offset + config.crop,
        ]
        meta = metadata.to(device)[None]
        mask = torch.ones((1, len(full)), dtype=torch.bool, device=device)
        compiled = self._group(
            image,
            meta=meta,
            mask=mask,
            compiled=True,
            dtype=config.dtype_autocast,
        )
        eager = self._group(
            image,
            meta=meta,
            mask=mask,
            compiled=False,
            dtype=config.dtype_autocast,
        )
        self._compare(compiled, eager=eager, prefix="study")
        pc, pe = compiled.mean(0).double().sigmoid(), eager.mean(0).double().sigmoid()
        difference = float((pc - pe).abs().max())
        self.receipt["study_dprob"] = round(difference, 7)
        tolerance = self.config.probability_tolerance
        if difference <= tolerance:
            self.receipt["decision"] = "Within the first-study probability bound."
        else:
            reference = self._group(
                image,
                meta=meta,
                mask=mask,
                compiled=False,
                dtype=None,
            )
            p32 = reference.mean(0).double().sigmoid()
            err_compiled, err_eager = (
                float((pc - p32).abs().max()),
                float((pe - p32).abs().max()),
            )
            self.receipt.update(
                fp32_err_compiled=round(err_compiled, 7),
                fp32_err_eager=round(err_eager, 7),
            )
            if err_compiled <= err_eager + tolerance:
                self.receipt["decision"] = (
                    "Compiled is no further from float32 than eager."
                )
            else:
                self.fallback("Compiled is further from float32 than eager.")
        synchronize(device)


class _Fold:
    def __init__(self, execution: EncoderExecution, *, index: int) -> None:
        self.execution, self.index = execution, index

    def __call__(
        self,
        image: Tensor,
        meta: Tensor,
        mask: Tensor,
        *,
        chunk: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        execution = self.execution
        model = execution.models[self.index]
        if not execution.active:
            return model(image, meta=meta, mask=mask, chunk=chunk)
        autocast = torch.is_autocast_enabled(execution.scorer().device.type)
        model.enc = (
            execution.compiled[self.index] if autocast else execution.eager[self.index]
        )
        try:
            return model(image, meta=meta, mask=mask, chunk=chunk)
        except (
            ArithmeticError,
            LookupError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            if not autocast:
                raise
            execution.fallback("Compiled forward failed: " + repr(error))
            return model(image, meta=meta, mask=mask, chunk=chunk)
        finally:
            if execution.compiled:
                model.enc = execution.compiled[self.index]

    def state_readout(self, states: Tensor, binaries: Tensor) -> Tensor:
        return self.execution.models[self.index].state_readout(
            states,
            binaries=binaries,
        )


class ReleasedState:
    """Read model tensors from a checkpoint whose bytes are verified first."""

    class Config(Fig["ReleasedState"]):
        base_dir: Path | str | None = None
        """Resource root beneath which `file.path` resolves; the owner supplies it."""

        file: LocalFile.Config = field(default_factory=LocalFile.Config)
        """Checkpoint path within the resource root, and its required digest."""

        key: str | None = None
        """Entry holding the model tensors; None means the loaded object itself."""

        weights_only: bool = True
        """Whether deserialization may only build tensors and plain containers."""

        @override
        def finalize(self) -> Self:
            self.file.path = resolve_working_dir(self.base_dir, self.file.path)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def state(self) -> dict[str, Tensor]:
        """Read every model tensor after verifying the complete checkpoint bytes.

        Returns:
            state: All model parameters and buffers without shape filtering.

        """
        loaded = cast(
            object,
            torch.load(
                self.config.file.make().path(),
                map_location="cpu",
                weights_only=self.config.weights_only,
            ),
        )
        if self.config.key is not None:
            loaded = from_plain(loaded, dict[str, object])[self.config.key]
        return from_plain(loaded, dict[str, torch.Tensor])


class FoldModel(Protocol):
    """A native fold providing every finding, state, and binary readout."""

    def __call__(
        self,
        images: Tensor,
        metadata: Tensor,
        mask: Tensor,
        /,
        *,
        chunk: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return complete finding, padded-state, and binary logits."""
        ...

    def state_readout(self, states: Tensor, binaries: Tensor) -> Tensor:
        """Return the complete 84-column categorical and binary probabilities."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class FoldScores:
    """Every float32 fold logit and exported state probability before reduction."""

    logits: Tensor
    states: Tensor


class LogitsCheck(Protocol):
    """Reject logits a scoring arm cannot accept before their sigmoid."""

    def __call__(self, logits: Tensor, /) -> None:
        """Raise when the logits are unusable."""
        ...


def require_finite(logits: Tensor) -> None:
    """Reject nonfinite logits, as the first Raptor arm does."""
    finite = bool(torch.isfinite(logits).all())
    if not finite:
        raise FloatingPointError("Raptor logits are nonfinite.")


def accept_logits(logits: Tensor) -> None:
    """Pass every logit through to the sigmoid, as the 224 arm does."""
    del logits


class RaptorScore:
    """Score each test-time view with the arm's precision ladder and logit check."""

    class Config(Fig["RaptorScore"]):
        device: str = "cuda"
        """Scoring device; CPU supports native numerical replay."""

        dtypes_autocast: list[torch.dtype | None] = field(
            default_factory=lambda: [torch.float16],
        )
        """CUDA autocast precisions tried in order; a RuntimeError moves to the next.

        None is float32; the last entry's RuntimeError propagates.
        """

        check_logits: LogitsCheck = require_finite
        """Validation of each pass's float32 logits before their sigmoid."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device(config.device)

    @torch.inference_mode()
    def __call__(self, model: TensorFn, views: Sequence[Tensor]) -> NDArray[np.float32]:
        """Return float32 probabilities after the released per-pass sigmoid order.

        Args:
            model: Loaded native Raptor in evaluation mode.
            views: Complete normalized window set of each test-time view.

        Returns:
            probabilities: All twelve float32 study probabilities, averaged
                across views in order.

        """
        scores = [self._pass(model, windows=windows) for windows in views]
        total = scores[0]
        for score in scores[1:]:
            total = total + score
        return cast("NDArray[np.float32]", total / len(scores))

    def _pass(self, model: TensorFn, windows: Tensor) -> NDArray[np.float32]:
        images = windows.unsqueeze(0).to(self.device)
        dtypes = self.config.dtypes_autocast
        for index, dtype in enumerate(dtypes):
            try:
                return self._infer(model, images=images, dtype=dtype)
            except RuntimeError:
                if index == len(dtypes) - 1:
                    raise
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        raise ValueError("Raptor scoring requires at least one autocast precision.")

    def _infer(
        self,
        model: TensorFn,
        images: Tensor,
        *,
        dtype: torch.dtype | None,
    ) -> NDArray[np.float32]:
        with torch.autocast(
            self.device.type,
            dtype=dtype,
            enabled=self.device.type == "cuda" and dtype is not None,
        ):
            logits = model(images).float()
            self.config.check_logits(logits)
            probabilities = torch.sigmoid(logits)
        return cast("NDArray[np.float32]", probabilities[0].cpu().numpy())


class R7Score:
    """Score all four folds, rescuing findings before reading every state head."""

    class Config(Fig["R7Score"]):
        device: str = "cuda"
        """Scoring device, with raw uint8 windows transferred before cropping."""

        crop: int = 352
        """Center crop cut from the complete rendered windows on the scoring device."""

        chunk_size: int = 192
        """Maximum padded encoder windows per forward chunk."""

        dtype_autocast: torch.dtype | None = None
        """CUDA autocast precision; None is float32."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device(config.device)

    @torch.inference_mode()
    def __call__(
        self,
        models: Sequence[FoldModel],
        images: Tensor,
        metadata: Tensor,
    ) -> FoldScores:
        """Return complete fold outputs after per-fold nonfinite-logit rescue.

        Args:
            models: Every loaded native fold, in released checkpoint order.
            images: Complete uint8 three-slice study windows.
            metadata: Complete nine-column float32 physical window metadata.

        Returns:
            scores: Every fold logit and complete state probability readout.

        """
        full = images.to(self.device, non_blocking=True)
        offset = (full.shape[-1] - self.config.crop) // 2
        cropped = full[
            None,
            :,
            :,
            offset : offset + self.config.crop,
            offset : offset + self.config.crop,
        ]
        metadata = metadata.to(self.device)[None]
        mask = torch.ones((1, cropped.shape[1]), dtype=torch.bool, device=self.device)
        logits: list[Tensor] = []
        readouts: list[Tensor] = []
        for model in models:
            finding, state, binary = self.forward(
                model,
                images=cropped,
                metadata=metadata,
                mask=mask,
                dtype=self.config.dtype_autocast,
            )
            # State NaNs do not trigger the released finding-logit rescue.
            if not bool(torch.isfinite(finding).all()):
                finding, state, binary = self.forward(
                    model,
                    images=cropped,
                    metadata=metadata,
                    mask=mask,
                    dtype=None,
                )
            logits.append(finding[0])
            readouts.append(model.state_readout(state, binaries=binary)[0])
        return FoldScores(logits=torch.stack(logits), states=torch.stack(readouts))

    def forward(
        self,
        model: FoldModel,
        images: Tensor,
        metadata: Tensor,
        mask: Tensor,
        *,
        dtype: torch.dtype | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one full fold with the selected autocast and float32 readouts.

        Args:
            model: Complete native finding and state model.
            images: Already cropped batch of uint8 three-slice windows.
            metadata: Complete physical window metadata on the scoring device.
            mask: Real-window membership.
            dtype: CUDA autocast precision, or None for eager float32 rescue.

        Returns:
            logits: Float32 finding logits.
            states: Float32 categorical-state logits.
            binaries: Float32 binary-state logits.

        """
        with torch.autocast(
            self.device.type,
            dtype=dtype,
            enabled=self.device.type == "cuda" and dtype is not None,
        ):
            logits, states, binaries = model(
                images,
                metadata,
                mask,
                chunk=self.config.chunk_size,
            )
        return logits.float(), states.float(), binaries.float()


class _Feature(TypedDict):
    weights: list[float]
    lines: list[dict[str, MutablePlainTree]]
    combine: NotRequired[str]
    weights_per_class: NotRequired[dict[str, list[float]]]


class _FluidRow(TypedDict):
    adopt: bool
    volume_column: str
    view_id: int
    b0: float
    b_z: float
    b_lv: float
    b_miss: float


class _StateRow(TypedDict):
    adopt: bool
    features: list[str]
    coef: list[float]
    b0: float
    volume_column: str | None
    dropped: NotRequired[list[str]]


@dataclass(frozen=True, kw_only=True, slots=True)
class CalibratedScores:
    """Full-precision and CSV-roundtripped scores plus nonfatal stacker failures."""

    probabilities: NDArray[np.float64]
    serialized: NDArray[np.float64]
    failures: list[str]


class R7Postprocess:
    """Preserve fold-reduction dtypes, fitted feature order, and CSV quantization."""

    class Config(Fig["R7Postprocess"]):
        base_dir: Path | str | None = None
        """Resource root beneath which the asset paths resolve; the owner supplies it."""

        fluid_file: LocalFile.Config = field(default_factory=LocalFile.Config)
        """Released fluid calibration coefficients and adoption decisions."""

        state_file: LocalFile.Config = field(default_factory=LocalFile.Config)
        """Released state calibration coefficients and exact readout layout."""

        spec_file: LocalFile.Config = field(default_factory=LocalFile.Config)
        """The checkpoint's categorical-state and binary-state specification."""

        @override
        def finalize(self) -> Self:
            for file in (self.fluid_file, self.state_file, self.spec_file):
                file.path = resolve_working_dir(self.base_dir, file.path)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        fluid = json_loads(config.fluid_file.make().path().read_text(), _FluidAsset)
        state = json_loads(config.state_file.make().path().read_text(), _StateAsset)
        spec_bytes = config.spec_file.make().path().read_bytes()
        spec = json_loads(spec_bytes.decode(), _StateSpec)
        if fluid["feature"].get("combine") != "logit":
            raise ValueError("Fluid calibration requires logit line mixing.")
        for feature in (fluid["feature"], state["feature"]):
            if feature["weights"] != [1.0] or len(feature["lines"]) != 1:
                raise ValueError("The released state recipe requires one model group.")
        if state["kind"] != "state_stacker_v1":
            raise ValueError("Unsupported state stacker kind.")
        if (
            state["state_spec_md5"]
            != hashlib.md5(spec_bytes, usedforsecurity=False).hexdigest()
        ):
            raise ValueError("State stacker was fitted on a different state spec.")
        layout: list[dict[str, object]] = []
        features: dict[str, list[tuple[str, str | int]]] = {}
        start = 0
        for name in FINDINGS:
            categories = spec["categories"][name]
            stop = start + len(categories)
            if len(spec["prior"][name]) != len(categories):
                raise ValueError("State prior length differs from its category count.")
            layout.append(
                {"class": name, "categories": categories, "start": start, "stop": stop},
            )
            features[name] = [("z", "z")]
            features[name].extend(
                ("state:" + category, index)
                for index, category in enumerate(categories, start=start)
            )
            start = stop
        for binary, names in spec["binaries"].items():
            for name in names:
                layout.append({"binary": binary, "class": name, "column": start})
                features[name].append(("bin:" + binary, start))
                start += 1
        if state["layout"] != to_plain({"columns": layout, "width": start}):
            raise ValueError(
                "State stacker layout differs from the checkpoint specification.",
            )
        overlap = {name for name, row in fluid["classes"].items() if row["adopt"]} & {
            name for name, row in state["classes"].items() if row["adopt"]
        }
        if overlap:
            raise ValueError(
                f"Fluid and state calibrations overlap: {sorted(overlap)}.",
            )
        self.fluid: list[_Calibration] = []
        self.state: list[_Calibration] = []

        for name, row in fluid["classes"].items():
            if not row["adopt"]:
                continue
            if REGION_VIEWS.get(row["volume_column"]) != row["view_id"]:
                raise ValueError("Fluid calibration volume and rendered view differ.")
            weights = fluid["feature"].get("weights_per_class", {}).get(name, [1.0])
            if weights != [1.0]:
                raise ValueError(
                    "Fluid class weights differ from the deployed line mix.",
                )
            values = [row["b_z"], row["b_lv"], row["b_miss"]]
            self.fluid.append(
                _Calibration(
                    finding=FINDINGS.index(name),
                    features=("z", "lv", "miss"),
                    coefficients=np.asarray(values),
                    intercept=row["b0"],
                    volume=row["volume_column"],
                ),
            )
        for name, row in state["classes"].items():
            if not row["adopt"]:
                continue
            wanted = list(features[name])
            if row["volume_column"] is not None and row["volume_column"] != "":
                if row["volume_column"] not in REGION_VIEWS:
                    raise ValueError("State calibration uses an unknown fluid volume.")
                wanted.append(("lv", "lv"))
                wanted.append(("miss", "miss"))
            kept = [
                (feature, column)
                for feature, column in wanted
                if feature not in row.get("dropped", [])
            ]
            if row["features"] != [feature for feature, _ in kept] or len(
                row["coef"],
            ) != len(kept):
                raise ValueError(
                    "State coefficients do not match the ordered feature layout.",
                )
            self.state.append(
                _Calibration(
                    finding=FINDINGS.index(name),
                    features=tuple(column for _, column in kept),
                    coefficients=np.asarray(row["coef"]),
                    intercept=row["b0"],
                    volume=row["volume_column"],
                ),
            )
        for calibration in [*self.fluid, *self.state]:
            if (
                not math.isfinite(calibration.intercept)
                or np.count_nonzero(np.isfinite(calibration.coefficients))
                != calibration.coefficients.size
            ):
                raise ValueError("Stacker coefficients must be finite.")

    def __call__(
        self,
        logits: Tensor,
        states: Tensor,
        fluid: Mapping[str, float | None] | None,
    ) -> CalibratedScores:
        """Calibrate complete fold outputs and retain the source's decimal boundary.

        Args:
            logits: Float32 fold-by-finding logits after per-fold precision rescue.
            states: Float32 fold-by-state readouts in the validated exported order.
            fluid: Measured D2 fluid volumes, or unavailable measurements.

        Returns:
            scores: Calibrated probabilities, serialized scores, and fallback reasons.

        """
        mean = logits.mean(0)
        logit_features: NDArray[np.float64] = np.asarray(
            mean.double().cpu().numpy(),
            dtype=np.float64,
        )
        if np.count_nonzero(np.isfinite(logit_features)) != logit_features.size:
            raise ValueError("R7 requires finite fold-mean logits.")
        probabilities: NDArray[np.float64] = np.asarray(
            torch.sigmoid(mean).double().cpu().numpy(),
            dtype=np.float64,
        )
        state_features: NDArray[np.float64] = np.asarray(
            states.double().mean(0).cpu().numpy(),
            dtype=np.float64,
        )
        failures: list[str] = []
        for group in (self.fluid, self.state):
            try:
                for calibration in group:
                    volume = (
                        None
                        if fluid is None or calibration.volume is None
                        else fluid.get(calibration.volume)
                    )
                    missing = volume is None or not math.isfinite(volume) or volume < 0
                    log_volume = (
                        0.0
                        if missing
                        else np.log1p(np.array(volume, dtype=np.float64)).item()
                    )
                    values: list[float] = []
                    scalars = {
                        "z": logit_features.item(calibration.finding),
                        "lv": log_volume,
                        "miss": float(missing),
                    }
                    for feature in calibration.features:
                        if isinstance(feature, int):
                            values.append(state_features.item(feature))
                        else:
                            values.append(scalars[feature])
                    logit = calibration.intercept + float(
                        cast(
                            np.float64,
                            np.dot(
                                np.asarray(values, dtype=np.float64),
                                calibration.coefficients,
                            ),
                        ),
                    )
                    if math.isfinite(logit):
                        probability = (
                            1 / (1 + math.exp(-logit))
                            if logit >= 0
                            else math.exp(logit) / (1 + math.exp(logit))
                        )
                        probabilities[calibration.finding] = probability
                    else:
                        failures.append(
                            f"Non-finite stacker logit for {FINDINGS[calibration.finding]}.",
                        )
            except (
                IndexError,
                KeyError,
                ValueError,
                TypeError,
                OverflowError,
            ) as error:
                failures.append(repr(error))
        serialized = np.asarray(
            [
                float(f"{probabilities.item(column):.9g}")
                for column in range(probabilities.size)
            ],
            dtype=np.float64,
        )
        return CalibratedScores(
            probabilities=probabilities,
            serialized=serialized,
            failures=failures,
        )


class _FluidAsset(TypedDict):
    feature: _Feature
    classes: dict[str, _FluidRow]


class _StateAsset(TypedDict):
    kind: str
    feature: _Feature
    layout: dict[str, MutablePlainTree]
    state_spec_md5: str
    classes: dict[str, _StateRow]


class _StateSpec(TypedDict):
    categories: dict[str, list[str]]
    prior: dict[str, list[float]]
    binaries: dict[str, list[str]]


@dataclass(frozen=True, kw_only=True, slots=True)
class _Calibration:
    finding: int
    features: tuple[str | int, ...]
    coefficients: NDArray[np.float64]
    intercept: float
    volume: str | None


@dataclass(kw_only=True, slots=True)
class StageScores:
    """All study probabilities, actual scored membership, and failure reasons."""

    values: NDArray[np.float32] | NDArray[np.float64]
    scored: NDArray[np.bool_]
    failures: dict[str, str]
    notes: list[dict[str, object]] = field(default_factory=list)


class RaptorStage:
    """Run one full released CoAtNet arm over every competition study."""

    class Config(Fig["RaptorStage"]):
        base_dir: Path | str | None = None
        """Resource root beneath which the checkpoint resolves; the owner supplies it."""

        checkpoint: ReleasedState.Config = field(default_factory=ReleasedState.Config)
        """Complete released checkpoint of this arm."""

        model: Raptor.Config = field(default_factory=Raptor.Config)
        """Native complete CoAtNet and finding-specific attention model."""

        preparation: RaptorPrepare.Config = field(default_factory=RaptorPrepare.Config)
        """Raw physical study preparation with every original slice slot."""

        windows: RaptorWindows.Config = field(default_factory=RaptorWindows.Config)
        """All original three-slice windows and arm-specific crop normalization."""

        views: list[ReorientFn] = field(default_factory=lambda: [unchanged])
        """Test-time views scored in order; their probabilities are averaged."""

        score: RaptorScore.Config = field(default_factory=RaptorScore.Config)
        """Per-pass precision ladder and logit check."""

        time_limit: float = math.inf
        """Maximum stage wall seconds; exceeding it aborts the stage."""

        cudnn_benchmark: bool = False
        """Let cuDNN autotune convolution algorithms while this stage runs."""

        allow_tf32_cudnn: bool = True
        """Allow TF32 inside cuDNN convolutions while this stage runs."""

        allow_tf32_matmul: bool = False
        """Allow TF32 in CUDA matrix products while this stage runs."""

        @override
        def finalize(self) -> Self:
            if self.checkpoint.base_dir is None:
                self.checkpoint.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    @torch.inference_mode()
    def __call__(self, data: StudyData) -> StageScores:
        """Score every study, retaining failed inference at twelve values of 0.5.

        Args:
            data: Complete competition study and acquisition inventory.

        Returns:
            scores: Every probability, actual inference membership, and diagnostic.

        """
        config = self.config
        backbone = config.model.backbone
        assert isinstance(backbone, CoAtNet.Config)
        model = config.model.make()
        model.load_checkpoint(config.checkpoint.make().state())
        model.eval().requires_grad_(False).to(config.score.device)
        prepare, windows, score = (
            config.preparation.make(),
            config.windows.make(),
            config.score.make(),
        )
        result = StageScores(
            values=np.full((len(data.ids), len(FINDINGS)), 0.5, dtype=np.float32),
            scored=np.zeros(len(data.ids), dtype=np.bool_),
            failures={},
        )
        start = time.monotonic()
        with _backend(config):
            for index, uid in enumerate(data.ids):
                if time.monotonic() - start > config.time_limit:
                    raise TimeoutError("The Raptor stage exceeded its wall-time limit.")
                try:
                    bag = prepare(data.load(uid).raptor)
                except (
                    ArithmeticError,
                    LookupError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    result.notes.append({"uid": uid, "preparation_error": repr(error)})
                    bag = prepare([])
                try:
                    views = [
                        windows(
                            bag,
                            output_size=backbone.image_size[0],
                            device=config.score.device,
                            reorient=view,
                        )
                        for view in config.views
                    ]
                    result.values[index] = score(model, views)
                    result.scored[index] = True
                except (
                    ArithmeticError,
                    LookupError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    result.failures[uid] = repr(error)
        return result


class R7Stage:
    """Run physical anatomy preparation, four full folds, and both stackers."""

    class Config(Fig["R7Stage"]):
        base_dir: Path | str | None = None
        """Resource root beneath which every asset resolves; the owner supplies it."""

        checkpoints: list[ReleasedState.Config] = field(
            default_factory=list[ReleasedState.Config],
        )
        """Every complete released R7 fold, in immutable checkpoint order."""

        model: Makeable[R7] = field(default_factory=R7.Config)
        """Native ConvNeXt-small with all spatial/state/extent/synovitis heads."""

        anatomy_checkpoint: ReleasedState.Config = field(
            default_factory=ReleasedState.Config,
        )
        """Complete four-tissue MONAI0.6-compatible VNet state."""

        anatomy: Makeable[ReleasedVNet] = field(
            default_factory=lambda: ReleasedVNet.Config(channels_in=1, channels_out=4),
        )
        """Native full-width anatomical segmentation model."""

        side_checkpoint: ReleasedState.Config = field(
            default_factory=ReleasedState.Config,
        )
        """Complete released ResNet34 laterality state."""

        side: Makeable[ResNet] = field(
            default_factory=lambda: ResNet.Config(channels_in=1, channels_out=1),
        )
        """Native postactivation laterality model."""

        preparation: StudyPreparation.Config = field(
            default_factory=StudyPreparation.Config,
        )
        """Full side correction, anatomy, anchors, views, fluid, and stride-one windows."""

        score: R7Score.Config = field(default_factory=R7Score.Config)
        """GPU crop, precision, padded chunking, and complete fold readouts."""

        postprocess: R7Postprocess.Config = field(default_factory=R7Postprocess.Config)
        """Both calibrated readouts, their assets and exact decimal boundary."""

        execution: Makeable[FoldExecution] = field(
            default_factory=EncoderExecution.Config,
        )
        """How the folds run: compiled encoders with acceptance checks, or eager."""

        cudnn_benchmark: bool = False
        """Let cuDNN autotune convolution algorithms while this stage runs."""

        allow_tf32_cudnn: bool = True
        """Allow TF32 inside cuDNN convolutions while this stage runs."""

        allow_tf32_matmul: bool = False
        """Allow TF32 in CUDA matrix products while this stage runs."""

        @override
        def finalize(self) -> Self:
            for asset in (
                *self.checkpoints,
                self.anatomy_checkpoint,
                self.side_checkpoint,
            ):
                if asset.base_dir is None:
                    asset.base_dir = self.base_dir
            if self.postprocess.base_dir is None:
                self.postprocess.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    @torch.inference_mode()
    def __call__(self, data: StudyData, *, deadline: float = math.inf) -> StageScores:
        """Score every reached study and mean-fill failures in sample CSV order.

        Args:
            data: Complete competition study and acquisition inventory.
            deadline: Absolute monotonic deadline, checked after each reached study.

        Returns:
            scores: Calibrated and mean-filled rows with actual scored membership.

        """
        config = self.config
        postprocess = config.postprocess.make()
        anatomy, side = config.anatomy.make(), config.side.make()
        anatomy.load_checkpoint(config.anatomy_checkpoint.make().state())
        side.load_torchvision_state_dict(config.side_checkpoint.make().state())
        for auxiliary in (anatomy, side):
            auxiliary.eval().requires_grad_(False).to(config.score.device)
        models: list[R7] = []
        for checkpoint in config.checkpoints:
            model = config.model.make()
            model.load_checkpoint(checkpoint.make().state())
            model.eval().requires_grad_(False).to(config.score.device)
            model.to(memory_format=torch.channels_last)
            models.append(model)
        execution = config.execution.make()
        execution.bind(models, score=config.score.make())
        preparation = config.preparation.make()
        result = StageScores(
            values=np.full((len(data.ids), len(FINDINGS)), 0.5, dtype=np.float64),
            scored=np.zeros(len(data.ids), dtype=np.bool_),
            failures={},
        )
        index_by_uid = {uid: index for index, uid in enumerate(data.ids)}
        order = sorted(data.ids, key=lambda uid: (-_file_count(data, uid=uid), uid))
        order = order[::2] + order[1::2]
        with _backend(config):
            for uid in order:
                index = index_by_uid[uid]
                try:
                    study = preparation(
                        data.load(uid).physical,
                        anatomy=anatomy,
                        side=side,
                        device=config.score.device,
                    )
                    folds = execution(study.images, metadata=study.metadata)
                    calibrated = postprocess(
                        folds.logits,
                        states=folds.states,
                        fluid=study.fluid.by_region(),
                    )
                    result.values[index] = calibrated.serialized
                    result.scored[index] = True
                    result.notes.append(
                        {
                            "uid": uid,
                            "side": to_plain(study.side),
                            "anchors": to_plain(study.anchors),
                            "fluid": to_plain(study.fluid),
                            "series_errors": study.errors,
                            "stacker_errors": calibrated.failures,
                            "windows": len(study.images),
                        },
                    )
                except (
                    ArithmeticError,
                    LookupError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    result.failures[uid] = repr(error)
                if time.monotonic() >= deadline:
                    break
        for uid in data.ids:
            if not result.scored[index_by_uid[uid]]:
                result.failures.setdefault(uid, "Not reached before the deadline.")
        _mean_fill(result, ids=data.ids, sample_ids=data.sample_ids)
        result.notes.append({"compile": execution.receipt})
        return result


@contextmanager
def _backend(config: RaptorStage.Config | R7Stage.Config) -> Generator[None]:
    matmul = torch.backends.cuda.matmul.allow_tf32
    with torch.backends.cudnn.flags(
        benchmark=config.cudnn_benchmark,
        allow_tf32=config.allow_tf32_cudnn,
    ):
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32_matmul
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul


def _file_count(data: StudyData, uid: str) -> int:
    directory = series_tree(data.input_dir, data.config.split, uid)
    try:
        return sum(len(list(series.iterdir())) for series in directory.iterdir())
    except OSError:
        return 0


def _mean_fill(result: StageScores, *, ids: list[str], sample_ids: list[str]) -> None:
    indices = {uid: index for index, uid in enumerate(ids)}
    reached = [indices[uid] for uid in sample_ids if result.scored[indices[uid]]]
    for column in range(len(FINDINGS)):
        mean = (
            sum(float(result.values.item(index, column)) for index in reached)
            / len(reached)
            if reached
            else 0.5
        )
        result.values[~result.scored, column] = float(f"{mean:.9g}")


class PublicRecipe:
    """Compose the complete released inference recipe from independent native stages."""

    class Config(Fig["PublicRecipe"]):
        study_name: str = ""
        """Run-family name; it fills `working_dir`."""

        experiment_name: str = ""
        """Run name within the study; it fills `working_dir`."""

        base_dir: Path | str | None = "/opt/scratch"
        """Resource root inherited by the data and every stage's assets."""

        working_dir: Path | str = "/runs/{study_name}/{experiment_name}"
        """Logical output directory, new for every run."""

        data: StudyData.Config = field(default_factory=StudyData.Config)
        """Complete competition CSVs and raw DICOM acquisitions."""

        raptor384: RaptorStage.Config = field(default_factory=RaptorStage.Config)
        """First released Raptor arm."""

        raptor224: RaptorStage.Config = field(default_factory=RaptorStage.Config)
        """Second released Raptor arm."""

        r7: R7Stage.Config = field(default_factory=R7Stage.Config)
        """Four ConvNeXt-small folds, physical anatomy, and both calibrated stackers."""

        blend: RankBlend.Config = field(default_factory=RankBlend.Config)
        """Per-model percentile ranks, blend weights, and failure caps."""

        budget: WallBudget.Config = field(default_factory=WallBudget.Config)
        """Outer wall allowance; R7 stops `reserve_evaluation_sec` before it ends."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.working_dir, str):
                self.working_dir = self.working_dir.format(
                    study_name=self.study_name,
                    experiment_name=self.experiment_name,
                )
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            for child in (self.data, self.raptor384, self.raptor224, self.r7):
                if child.base_dir is None:
                    child.base_dir = self.base_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Run the complete native recipe and retain a receipt on stage failure.

        Args:
            *args: Passthrough arguments are rejected; use configuration fields.

        """
        if args:
            raise ValueError("PublicRecipe accepts configuration fields only.")
        config = self.config
        files = [
            config.raptor384.checkpoint.file,
            config.raptor224.checkpoint.file,
            *(state.file for state in config.r7.checkpoints),
            config.r7.side_checkpoint.file,
            config.r7.anatomy_checkpoint.file,
            config.r7.postprocess.fluid_file,
            config.r7.postprocess.state_file,
            config.r7.postprocess.spec_file,
        ]
        data_dir = Path(config.data.input_dir)
        output = validated_output_path(
            config.working_dir,
            protected=[data_dir, *(Path(file.path) for file in files)],
        )
        inputs = [data_dir.resolve(), *(Path(file.path).resolve() for file in files)]
        if any(
            output.is_relative_to(path) or path.is_relative_to(output)
            for path in inputs
        ):
            raise ValueError("The output directory overlaps a protected input.")
        output.mkdir(parents=True)
        (output / "config.txt").write_text(
            config.pformat(hide_default_values=False) + "\n",
        )
        budget = config.budget.make()
        budget.receipt_path = output / "run_receipt.json"
        budget.metadata.update(
            {
                "implementation": "native Priml",
                "historical_training_reproduced": False,
                "assets": [
                    {"path": str(file.path), "sha256": file.sha256} for file in files
                ],
            },
        )
        with budget:
            data = config.data.make()
            raptor384 = _raptor_stage(
                config.raptor384,
                data=data,
                output=output,
                name="raptor",
            )
            raptor224 = _raptor_stage(
                config.raptor224,
                data=data,
                output=output,
                name="raptor224",
            )
            r7 = config.r7.make()(
                data,
                deadline=time.monotonic() + budget.training_seconds(),
            )
            _json(
                output / "submission_receipt.json",
                value=_stage_receipt(r7, data=data),
            )
            _write_scores(
                output / "ours_sub.csv",
                values=r7.values,
                data=data,
                order=data.sample_ids,
                decimals=True,
            )
            _write_scores(
                output / "scored.csv",
                values=r7.values,
                data=data,
                order=[uid for index, uid in enumerate(data.ids) if r7.scored[index]],
                decimals=True,
            )
            raw384 = _read_scores(output / "raptor_raw.csv", data=data)
            raw224 = _read_scores(output / "raptor224_raw.csv", data=data)
            ours = _read_scores(output / "ours_sub.csv", data=data)
            correlations = _correlations(
                raw384,
                raw224=raw224,
                ours=ours,
                sub384=_read_scores(output / "raptor_sub.csv", data=data),
                sub224=_read_scores(output / "raptor224_sub.csv", data=data),
            )
            predictions = config.blend.make()(
                [raw384, raw224, ours],
                [raptor384.scored, raptor224.scored, r7.scored],
            )
            _validate_predictions(predictions)
            _write_scores(
                output / "submission.csv",
                values=predictions,
                data=data,
                order=data.ids,
                decimals=False,
            )
            _json(
                output / "blend_receipt.json",
                value={
                    "weights": to_plain(config.blend),
                    "rows": len(data.ids),
                    "raptor384_failed": int((~raptor384.scored).sum()),
                    "raptor224_failed": int((~raptor224.scored).sum()),
                    "ours_failed": int((~r7.scored).sum()),
                    "correlations": correlations,
                    "rank_basis": "Each model over its own non-failed rows.",
                },
            )
            budget.metadata["rows"] = len(data.ids)
            budget.metadata["raptor_scored"] = [
                int(raptor384.scored.sum()),
                int(raptor224.scored.sum()),
            ]


def _raptor_stage(
    config: RaptorStage.Config,
    *,
    data: StudyData,
    output: Path,
    name: str,
) -> StageScores:
    start = time.monotonic()
    result = config.make()(data)
    _write_scores(
        output / f"{name}_raw.csv",
        values=result.values,
        data=data,
        order=data.ids,
        decimals=False,
        canonical=True,
    )
    _write_scores(
        output / f"{name}_sub.csv",
        values=_ordinal_ranks(result.values),
        data=data,
        order=data.ids,
        decimals=False,
    )
    _json(
        output / f"{name}_receipt.json",
        value={
            **_stage_receipt(result, data=data),
            "seconds": time.monotonic() - start,
        },
    )
    release_memory(torch.device(config.score.device))
    return result


def _ordinal_ranks(
    values: NDArray[np.float32] | NDArray[np.float64],
) -> NDArray[np.float32]:
    inputs = np.asarray(values, dtype=np.float32)
    ranks = np.empty(inputs.shape, dtype=np.float64)
    # Failed studies tie at one score; NumPy's default sort may reorder tied rows.
    order = inputs.argsort(0, kind="stable")
    for column in range(len(inputs.T)):
        ranks[:, column] = ordinal_ranks(
            inputs[:, column],
            order=torch.from_numpy(order[:, column]),
        ).numpy()
    return (ranks / max(len(values) - 1, 1)).astype(np.float32)


def _stage_receipt(result: StageScores, *, data: StudyData) -> dict[str, object]:
    return {
        "rows": len(data.ids),
        "scored": int(result.scored.sum()),
        "scored_studies": [
            uid for index, uid in enumerate(data.ids) if result.scored[index]
        ],
        "failures": result.failures,
        "notes": result.notes,
    }


def _write_scores(
    path: Path,
    values: NDArray[np.float32] | NDArray[np.float64],
    *,
    data: StudyData,
    order: list[str],
    decimals: bool,
    canonical: bool = False,
) -> None:
    indices = {uid: index for index, uid in enumerate(data.ids)}
    labels = FINDINGS if canonical else data.labels
    columns = [FINDINGS.index(label) for label in labels]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n" if decimals else "\n")
        writer.writerow(["StudyInstanceUID", *labels])
        for uid in order:
            row = [float(values.item(indices[uid], column)) for column in columns]
            writer.writerow(
                [
                    uid,
                    *[_score_text(value, decimals=decimals) for value in row],
                ],
            )


def _score_text(value: float, *, decimals: bool) -> str:
    return format(value, ".9g") if decimals else str(np.float32(value))


def _read_scores(path: Path, *, data: StudyData) -> NDArray[np.float64]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames not in (
            ["StudyInstanceUID", *data.labels],
            ["StudyInstanceUID", *FINDINGS],
        ):
            raise ValueError(f"Prediction columns differ from the sample: {path}.")
        rows = list(reader)
    indexed = {row["StudyInstanceUID"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(data.ids):
        raise ValueError(f"Prediction rows differ from the competition: {path}.")
    return np.asarray(
        [[float(indexed[uid][label]) for label in FINDINGS] for uid in data.ids],
        dtype=np.float64,
    )


def _correlations(
    raw384: NDArray[np.float64],
    *,
    raw224: NDArray[np.float64],
    ours: NDArray[np.float64],
    sub384: NDArray[np.float64],
    sub224: NDArray[np.float64],
) -> dict[str, dict[str, float]]:
    for name, raw, submitted in (
        ("raptor_raw_vs_sub", raw384, sub384),
        ("raptor224_raw_vs_sub", raw224, sub224),
    ):
        ranks = _ordinal_ranks(raw)
        expected = ranks.astype(str).astype(np.float64)
        if not np.array_equal(submitted, expected):
            raise RuntimeError(f"{name} does not preserve its own submission ranks.")
    output: dict[str, dict[str, float]] = {}
    if len(raw384) < 3:
        return output
    for name, first, second in (
        ("raptor_raw_vs_sub", raw384, sub384),
        ("raptor224_raw_vs_sub", raw224, sub224),
        ("raptor_vs_ours", raw384, ours),
        ("raptor224_vs_raptor384", raw224, raw384),
    ):
        values: dict[str, float] = {}
        for column, label in enumerate(FINDINGS):
            valid = np.isfinite(first[:, column]) & np.isfinite(second[:, column])
            with torch.device("cpu"):
                correlation = spearman(first[valid, column], second[valid, column])
            values[label] = round(float(correlation), 4)
        output[name] = values
    return output


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(to_plain(value), indent=2) + "\n")


def _validate_predictions(predictions: NDArray[np.float32]) -> None:
    if np.count_nonzero(
        np.isfinite(predictions),
    ) != predictions.size or np.count_nonzero((predictions < 0) | (predictions > 1)):
        raise ValueError("The final submission contains invalid probabilities.")
