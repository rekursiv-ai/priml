"""Recover text topics with native lexical features and cosine distances."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Literal, cast

from configgle import Fig

import numpy as np

from priml.baselines.name_that_llm.model import random_chunks
from priml.baselines.name_that_llm.tfidf import TfidfVectorizer, english_words


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
else:
    from wrapt import lazy_import

    fcluster = lazy_import("scipy.cluster.hierarchy", "fcluster")
    linkage = lazy_import("scipy.cluster.hierarchy", "linkage")
    squareform = lazy_import("scipy.spatial.distance", "squareform")


class TopicClusters:
    """Recover approximate unseen prompts from prediction text alone."""

    class Config(Fig["TopicClusters"]):
        """Reference cosine-distance clustering cut."""

        threshold: float = 0.9
        """Cut level on the cosine-distance linkage tree."""

        method: Literal[
            "single",
            "complete",
            "average",
            "weighted",
            "centroid",
            "median",
            "ward",
        ] = "average"
        """SciPy linkage method merging clusters."""

        criterion: Literal["distance", "inconsistent"] = "distance"
        """SciPy flat-cluster criterion that `threshold` is measured against."""

        vectorizer: TfidfVectorizer.Config = field(
            default_factory=lambda: TfidfVectorizer.Config(analyzer=english_words()),
        )
        """Content-word TF-IDF fitted only to the passages being clustered."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(
        self,
        texts: NDArray[np.object_],
    ) -> NDArray[np.int32] | NDArray[np.int64]:
        """Return one recovered topic identifier per passage."""
        if len(texts) < 2:
            return np.ones(len(texts), dtype=np.int64)
        features = self.config.vectorizer.make().fit_transform(
            cast("list[str]", texts.tolist()),
        )
        product = features @ features.T
        distances = np.clip(1 - product.toarray(), a_min=0, a_max=None)
        np.fill_diagonal(distances, val=0)
        return cast(
            "NDArray[np.int32]",
            fcluster(
                linkage(
                    squareform(distances, checks=False),
                    method=self.config.method,
                ),
                t=self.config.threshold,
                criterion=self.config.criterion,
            ),
        )


class ValidationGroups:
    """Simulate the hidden set's small unseen prompts on labelled passages."""

    class Config(Fig["ValidationGroups"]):
        """Text-only topics split into seeded small chunks."""

        topics: TopicClusters.Config = field(default_factory=TopicClusters.Config)
        """Topic recovery, shared with the hidden passages' clustering."""

        chunk_size: int = 2
        """Maximum passages per validation group within a recovered topic."""

        seed: int = 0
        """Generator seed for the order in which each topic is chunked."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, texts: NDArray[np.object_]) -> NDArray[np.object_]:
        """Return one validation group identifier per labelled passage."""
        return random_chunks(
            self.config.topics.make()(texts),
            size=self.config.chunk_size,
            rng=np.random.default_rng(self.config.seed),
        )
