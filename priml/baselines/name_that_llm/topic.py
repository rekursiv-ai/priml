"""Recover text topics with native lexical features and cosine distances."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, cast

from configgle import Fig

import numpy as np

from priml.baselines.name_that_llm.model import random_chunks
from priml.baselines.name_that_llm.tfidf import TfidfVectorizer, english_words


if TYPE_CHECKING:
    from numpy.typing import NDArray


class TopicClusters:
    """Recover approximate unseen prompts from prediction text alone."""

    class Config(Fig["TopicClusters"]):
        """Reference cosine-distance clustering cut."""

        threshold: float = 0.9
        """Largest average-linkage merge height inside one topic."""

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
        return distance_clusters(
            average_linkage(np.maximum(1 - features.gram(), 0)),
            threshold=self.config.threshold,
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


def average_linkage(distances: NDArray[np.float64]) -> NDArray[np.float64]:
    """Merge clusters by mean pairwise distance, in nearest-neighbor-chain order.

    Ties resolve as in SciPy's `nn_chain`: a chain starts at the lowest live
    cluster, keeps its previous link on a tie, and otherwise takes the lowest
    index. Each merge rounds the size-weighted products before adding them.

    Args:
      distances: Square pairwise distances; only the upper triangle is read.

    Returns:
      tree: One row per merge sorted stably by height: the merged identifiers,
        smaller first, the height, and the merged size. Leaves are numbered
        from zero, and row `k` forms cluster `leaves + k`.

    """
    upper = np.triu(distances)
    between = upper + upper.T
    # Infinity on the diagonal and for merged-away clusters keeps them unchosen.
    np.fill_diagonal(between, val=np.inf)
    sizes = np.ones(len(distances))
    tree = np.empty((len(distances) - 1, 4))
    chain: list[int] = []
    for merge in range(len(tree)):
        if not chain:
            chain.append(np.flatnonzero(sizes).item(0))
        x, y = sorted(_mutual_neighbors(between, chain=chain))
        nx, ny = sizes.item(x), sizes.item(y)
        tree[merge] = x, y, between.item(x, y), nx + ny
        sizes[x], sizes[y] = 0, nx + ny
        merged = (nx * between[x, :] + ny * between[y, :]) / (nx + ny)
        between[y, :] = between[:, y] = merged
        between[x, :] = between[:, x] = np.inf
    heights = cast("list[float]", tree[:, 2].tolist())
    tree = tree[sorted(range(len(tree)), key=heights.__getitem__)]
    parent: dict[int, int] = {}
    merged_pairs = cast("list[list[float]]", tree[:, :2].tolist())
    for merge, (x, y) in enumerate(merged_pairs):
        first, second = sorted((_root(parent, int(x)), _root(parent, int(y))))
        tree[merge, :2] = first, second
        parent[first] = parent[second] = len(distances) + merge
    return tree


def distance_clusters(
    tree: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.int32]:
    """Assign each leaf a flat cluster whose merges all lie at or below `threshold`.

    Numbering follows SciPy's `cluster_monocrit` walk: depth first, left
    before right, with a node's leaf children numbered after its subtrees.

    Args:
      tree: Merges in the layout `average_linkage` returns.
      threshold: Largest merge height allowed inside one cluster.

    Returns:
      labels: Cluster number of every leaf, counting from one.

    """
    leaves = len(tree) + 1
    children = cast("list[list[int]]", tree[:, :2].astype(np.int64).tolist())
    highest = _subtree_heights(tree, children=children)
    labels = np.zeros(leaves, dtype=np.int32)
    entered = [False] * len(tree)
    stack = [len(tree) - 1]
    clusters, leader = 0, None
    while stack:
        node = stack[-1]
        if leader is None and highest[node] <= threshold:
            leader = node
            clusters += 1
        pending = [
            child - leaves
            for child in children[node]
            if child >= leaves and not entered[child - leaves]
        ]
        if pending:
            entered[pending[0]] = True
            stack.append(pending[0])
            continue
        for child in children[node]:
            if child < leaves:
                if leader is None:
                    clusters += 1
                labels[child] = clusters
        if leader == node:
            leader = None
        stack.pop()
    return labels


def _mutual_neighbors(
    between: NDArray[np.float64],
    chain: list[int],
) -> tuple[int, int]:
    """Extend the chain until its top two are mutual nearest; pop and return them."""
    while True:
        x = chain[-1]
        y = int(np.argmin(between[x, :]))
        if len(chain) > 1 and between.item(x, chain[-2]) <= between.item(x, y):
            return chain.pop(), chain.pop()
        chain.append(y)


def _root(parent: dict[int, int], node: int) -> int:
    """Return the cluster now containing `node`."""
    while node in parent:
        node = parent[node]
    return node


def _subtree_heights(
    tree: NDArray[np.float64],
    children: list[list[int]],
) -> list[float]:
    """Return the highest merge inside each merge's subtree."""
    leaves = len(tree) + 1
    highest = cast("list[float]", tree[:, 2].tolist())
    for node, pair in enumerate(children):
        for child in pair:
            if child >= leaves:
                highest[node] = max(highest[node], highest[child - leaves])
    return highest
