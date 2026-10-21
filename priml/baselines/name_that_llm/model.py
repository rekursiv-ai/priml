"""Compose native lexical and frozen-embedding classifiers for model identity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast

import re

from configgle import Fig, InlineConfig, Makeable
from numpy.typing import NDArray

import numpy as np

from priml.baselines.name_that_llm import standardization
from priml.baselines.name_that_llm.calibration import SigmoidCalibration
from priml.baselines.name_that_llm.logistic import LogisticRegression
from priml.baselines.name_that_llm.sparse import CsrMatrix, hstack
from priml.baselines.name_that_llm.svm import LinearSVC
from priml.baselines.name_that_llm.tfidf import (
    CharacterNgrams,
    TfidfVectorizer,
    WordNgrams,
)
from priml.data.folds import grouped_folds


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.random import Generator


@dataclass(frozen=True, slots=True, kw_only=True)
class Split:
    """Rows to fit on and rows to predict, as every ensemble branch receives them."""

    texts: NDArray[np.object_]
    targets: NDArray[np.int64]
    groups: NDArray[np.str_]
    embeddings: NDArray[np.float32]
    predict_texts: NDArray[np.object_]
    predict_embeddings: NDArray[np.float32]
    clusters: (
        NDArray[np.str_] | NDArray[np.int32] | NDArray[np.int64] | NDArray[np.object_]
    )


class Branch(Protocol):
    """One ensemble member mapping a split to class probabilities."""

    def __call__(
        self,
        split: Split,
        earlier: Mapping[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Return probabilities for the split's prediction rows.

        Args:
          split: Complete fitting and prediction rows.
          earlier: Probabilities of the branches run before this one, by name.

        Returns:
          probabilities: Class probabilities in sorted-label order.

        """
        ...


class StyleCounts:
    """Compute the pinned 25 case-sensitive document statistics."""

    def transform(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Return all style statistics, preserving Unicode and empty passages.

        Args:
          texts: Complete original document strings.

        Returns:
          features: Float64 document rows with all 25 style columns.

        """
        result = np.empty((len(texts), 25), dtype=np.float64)
        for row, text in enumerate(texts):
            words = text.split()
            characters = max(len(text), 1)
            result[row, :3] = [
                np.log(len(text) + 1),
                np.log(len(words) + 1),
                text.count("\n") / characters * 100,
            ]
            result[row, 3:18] = [
                text.count(symbol)
                for symbol in (
                    "\n\n",
                    "\N{EM DASH}",
                    "–",
                    ",",
                    ".",
                    ";",
                    ":",
                    '"',
                    "'",
                    "’",
                    "“",
                    "*",
                    "#",
                    "(",
                    "-",
                )
            ]
            result[row, 18] = (
                sum(character.isupper() for character in text) / characters
            )
            result[row, 19] = np.mean([len(word) for word in words]) if words else 0
            result[row, 20] = len(re.findall(r"[.!?]", string=text)) / max(
                len(words),
                1,
            )
            result[row, 21:] = [text.count(symbol) for symbol in ("!", "?", "…", "...")]
        return result


class DocumentFeatures:
    """Fit character, word and standardized style features on complete texts."""

    class Config(Fig["DocumentFeatures"]):
        """Inject all lexical analyzers, style statistics and scaling."""

        char: TfidfVectorizer.Config = field(
            default_factory=lambda: TfidfVectorizer.Config(
                analyzer=CharacterNgrams.Config(),
                preprocess=str,
                min_documents=2,
                max_features=300_000,
            ),
        )
        """Case-sensitive character features with the pinned vocabulary cap."""

        word: TfidfVectorizer.Config = field(
            default_factory=lambda: TfidfVectorizer.Config(
                analyzer=WordNgrams.Config(
                    maximum=2,
                    token_pattern=r"\S+",  # noqa: S106 -- Whitespace token expression.
                ),
                preprocess=str,
                min_documents=2,
            ),
        )
        """Case-sensitive whitespace word features including adjacent pairs."""

        style: Makeable[StyleCounts] = field(
            default_factory=lambda: InlineConfig(StyleCounts),
        )
        """Stateless document statistics before fitted standardization."""

        scaler: standardization.StandardScaler.Config = field(
            default_factory=standardization.StandardScaler.Config,
        )
        """Dense standardization fitted only on the current training fold."""

        style_weight: float = 1.0
        """Style-feature multiplier applied after dense standardization."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.char = config.char.make()
        self.word = config.word.make()
        self.style = config.style.make()
        self.scaler = config.scaler.make()

    def fit_transform(self, texts: Sequence[str]) -> CsrMatrix:
        """Fit every feature transform and combine all training columns.

        Args:
          texts: Complete training passages in source row order.

        Returns:
          features: Character, word and weighted style columns in pinned order.

        """
        style = self.style.transform(texts)
        return self._combine(
            self.char.fit_transform(texts),
            word=self.word.fit_transform(texts),
            style=self.scaler.fit(style).transform(style) * self.config.style_weight,
        )

    def transform(self, texts: Sequence[str]) -> CsrMatrix:
        """Transform held-out passages with the fitted complete vocabulary.

        Args:
          texts: Complete passages in prediction order.

        Returns:
          features: All fitted character, word and weighted style columns.

        """
        return self._combine(
            self.char.transform(texts),
            word=self.word.transform(texts),
            style=self.scaler.transform(self.style.transform(texts))
            * self.config.style_weight,
        )

    def _combine(
        self,
        char: CsrMatrix,
        word: CsrMatrix,
        style: NDArray[np.float64],
    ) -> CsrMatrix:
        return hstack([char, word, CsrMatrix.from_dense(style)])


class SvmBranch:
    """Refit every feature transform within prompt-exclusive calibration folds."""

    class Config(Fig["SvmBranch"]):
        """Entire lexical classifier and held-out sigmoid calibration."""

        features: DocumentFeatures.Config = field(
            default_factory=DocumentFeatures.Config,
        )
        """Character, word and standardized style feature pipeline."""

        classifier: LinearSVC.Config = field(default_factory=LinearSVC.Config)
        """Native regularized squared-hinge classifier and shuffle stream."""

        calibration: SigmoidCalibration.Config = field(
            default_factory=SigmoidCalibration.Config,
        )
        """Native held-out score calibration for each sorted class."""

        inner_folds: int = 2
        """Prompt-exclusive partitions used to fit every calibration target."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.fitted: tuple[DocumentFeatures, LinearSVC] | None = None
        self.calibrators: list[SigmoidCalibration] = []

    def __call__(
        self,
        split: Split,
        earlier: Mapping[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Classify the split's prediction passages from text alone."""
        del earlier
        return self.predict(
            split.texts,
            targets=split.targets,
            groups=split.groups,
            predict_texts=split.predict_texts,
        )

    def predict(
        self,
        texts: NDArray[np.object_],
        targets: NDArray[np.int64],
        groups: NDArray[np.str_],
        predict_texts: NDArray[np.object_],
    ) -> NDArray[np.float64]:
        """Fit grouped score calibration and classify every requested passage.

        Args:
          texts: Training passages in CSV order.
          targets: Integer identities in the sorted class vocabulary.
          groups: Complete prompt identities withheld during calibration.
          predict_texts: All passages to classify with the final fitted model.

        Returns:
          probabilities: Complete calibrated class rows in sorted-label order.

        """
        classes = np.unique(targets)
        held_out = np.empty((len(texts), len(classes)), dtype=np.float64)
        for fit, valid in grouped_folds(groups, count=self.config.inner_folds):
            features = self.config.features.make()
            classifier = self.config.classifier.make().fit(
                features.fit_transform(cast("list[str]", texts[fit].tolist())),
                targets=targets[fit],
            )
            held_out[valid] = classifier.decision_function(
                features.transform(cast("list[str]", texts[valid].tolist())),
            )
        features = self.config.features.make()
        classifier = self.config.classifier.make().fit(
            features.fit_transform(cast("list[str]", texts.tolist())),
            targets=targets,
        )
        self.fitted = features, classifier
        self.calibrators = [
            self.config.calibration.make().fit(
                held_out[:, column],
                targets=cast("NDArray[np.bool_]", targets == label).astype(np.int64),
            )
            for column, label in enumerate(cast("list[int]", classes.tolist()))
        ]
        scores = classifier.decision_function(
            features.transform(cast("list[str]", predict_texts.tolist())),
        )
        probabilities = np.zeros_like(scores)
        for column, calibration in enumerate(self.calibrators):
            probabilities[:, column] = calibration.predict(scores[:, column])
        denominator = probabilities.sum(axis=1)[:, None]
        probabilities = np.divide(
            probabilities,
            denominator,
            out=np.full_like(probabilities, fill_value=1 / len(classes)),
            where=cast("NDArray[np.bool_]", denominator != 0),
        )
        probabilities[(probabilities > 1) & (probabilities <= 1 + 1e-5)] = 1
        return probabilities


class QwenHeads:
    """Average a scaled logistic classifier for every selected hidden layer."""

    class Config(Fig["QwenHeads"]):
        """Layer-wise classifier construction."""

        classifier: LogisticRegression.Config = field(
            default_factory=LogisticRegression.Config,
        )
        """Reference logistic solver, regularization, and iteration budget."""

        scaler: standardization.StandardScaler.Config = field(
            default_factory=standardization.StandardScaler.Config,
        )
        """Layer-specific dense feature standardization fitted on training rows."""

        layers: list[int] = field(default_factory=list[int])
        """Positions in the saved Qwen hidden states; empty takes the encoder's."""

    def __init__(self, config: Config) -> None:
        if not config.layers:
            raise ValueError("Expected at least one saved layer for the heads.")
        self.config = config
        self.fitted: list[
            tuple[standardization.StandardScaler, LogisticRegression]
        ] = []

    def __call__(
        self,
        split: Split,
        earlier: Mapping[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Classify the split's prediction embeddings."""
        del earlier
        return self.predict(
            split.embeddings,
            targets=split.targets,
            predict=split.predict_embeddings,
        )

    def predict(
        self,
        fit: NDArray[np.float32],
        targets: NDArray[np.int64],
        predict: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Fit all heads and average their probabilities in reference order.

        Args:
          fit: Training embeddings shaped rows, saved layers, hidden width.
          targets: Sorted-vocabulary class indices.
          predict: Prediction embeddings with matching layer/width axes.

        Returns:
          probabilities: Mean of the per-layer calibrated class probabilities.

        """
        probabilities = np.zeros((len(predict), len(np.unique(targets))))
        self.fitted = []
        for layer in self.config.layers:
            scaler = self.config.scaler.make().fit(fit[:, layer])
            head = self.config.classifier.make().fit(
                scaler.transform(fit[:, layer]),
                targets=targets,
            )
            probabilities += head.predict_proba(
                scaler.transform(predict[:, layer]),
            ) / len(
                self.config.layers,
            )
            self.fitted.append((scaler, head))
        return probabilities


class CenteredHeads:
    """Train all four layer heads on all four random chunk sizes."""

    class Config(Fig["CenteredHeads"]):
        """Chunking and layer classifiers; singleton predictions retain plain heads."""

        heads: QwenHeads.Config = field(default_factory=QwenHeads.Config)
        """Classifiers fitted on the stacked centred embeddings."""

        chunk_sizes: list[int] = field(default_factory=list[int])
        """Training chunk sizes, all stacked rather than sampled or averaged."""

        minimum_cluster: int = 2
        """Smaller prediction clusters retain the fallback branch's probability."""

        fallback: str = "plain"
        """Earlier branch whose probabilities small clusters keep."""

        seed: int = 0
        """One continuing NumPy generator shared by all training chunk sizes."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.heads = config.heads.make()

    def __call__(
        self,
        split: Split,
        earlier: Mapping[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Classify centred prediction embeddings, keeping the fallback for singletons."""
        return self.predict(
            split.embeddings,
            targets=split.targets,
            groups=split.groups,
            predict=split.predict_embeddings,
            clusters=split.clusters,
            plain=earlier[self.config.fallback],
        )

    def predict(
        self,
        fit: NDArray[np.float32],
        targets: NDArray[np.int64],
        groups: NDArray[np.str_],
        predict: NDArray[np.float32],
        clusters: NDArray[np.str_]
        | NDArray[np.int32]
        | NDArray[np.int64]
        | NDArray[np.object_],
        *,
        plain: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Fit chunk-centred heads and preserve plain predictions for singletons.

        Args:
          fit: Training embeddings, rows by saved layers by hidden width.
          targets: Sorted-vocabulary class indices.
          groups: True training prompt identifiers.
          predict: Prediction embeddings.
          clusters: Recovered topic clusters or validation chunk identifiers.
          plain: Plain-head probabilities, used for small clusters.

        Returns:
          probabilities: Centred-head means with the reference singleton fallback.

        """
        rng = np.random.default_rng(self.config.seed)
        chunkings = [
            random_chunks(groups, size=size, rng=rng)
            for size in self.config.chunk_sizes
        ]
        stacked = np.concatenate([center(fit, groups=chunks) for chunks in chunkings])
        probabilities = self.heads.predict(
            stacked,
            targets=np.tile(targets, reps=len(chunkings)),
            predict=center(predict, groups=clusters),
        )
        _, inverse, counts = np.unique(
            clusters,
            return_inverse=True,
            return_counts=True,
        )
        small = counts[inverse] < self.config.minimum_cluster
        probabilities[small] = plain[small]
        return probabilities


class LogBlend:
    """Blend SVM, plain Qwen, and centred Qwen in log space."""

    class Config(Fig["LogBlend"]):
        """Branch weights and the clipping floor."""

        weights: dict[str, float] = field(default_factory=dict[str, float])
        """Log-probability weight of each branch, summed in this order."""

        minimum_probability: float = 1e-12
        """Lower clip before the logarithm."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(
        self,
        probabilities: Mapping[str, NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """Return unnormalized log scores in the original arithmetic order.

        Args:
          probabilities: Class probabilities of every weighted branch, by name.

        Returns:
          scores: Weighted log probabilities for each passage and class.

        Raises:
          ValueError: The branches and weights name different members.

        """
        config = self.config
        if not config.weights or probabilities.keys() != config.weights.keys():
            raise ValueError(
                "Expected one blend weight for each of one or more branches.",
            )
        terms = [
            weight
            * np.log(
                np.clip(
                    probabilities[name],
                    a_min=config.minimum_probability,
                    a_max=1,
                ),
            )
            for name, weight in config.weights.items()
        ]
        scores = terms[0]
        for term in terms[1:]:
            scores = scores + term
        return scores


def center(
    features: NDArray[np.float32],
    groups: NDArray[np.str_]
    | NDArray[np.int32]
    | NDArray[np.int64]
    | NDArray[np.object_],
) -> NDArray[np.float32]:
    """Subtract each group's mean from a copy of its rows."""
    result = features.copy()
    for group in cast("list[str | int]", np.unique(groups).tolist()):
        index = cast("NDArray[np.bool_]", groups == group)
        result[index] -= result[index].mean(0)
    return result


def random_chunks(
    groups: NDArray[np.str_] | NDArray[np.int32] | NDArray[np.int64],
    size: int,
    rng: Generator,
) -> NDArray[np.object_]:
    """Partition each sorted group into randomly ordered bounded chunks.

    Args:
      groups: True prompt or recovered topic identifiers.
      size: Maximum number of passages per chunk.
      rng: Continuing generator shared by all requested chunk sizes.

    Returns:
      chunks: Object array of the reference's string chunk identifiers.

    Raises:
      ValueError: The chunk size is not positive.

    """
    if size <= 0:
        raise ValueError("Expected a positive chunk size.")
    chunks = np.empty(len(groups), dtype=object)
    for group in cast("list[str | int]", np.unique(groups).tolist()):
        members = rng.permutation(
            np.where(cast("NDArray[np.bool_]", groups == group))[0],
        )
        for chunk, start in enumerate(range(0, len(members), size)):
            chunks[members[start : start + size]] = f"{group}_{chunk}"
    return cast("NDArray[np.object_]", chunks)
