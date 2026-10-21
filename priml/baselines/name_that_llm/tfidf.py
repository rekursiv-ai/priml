"""Build sparse lexical features from configurable character and word analyzers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field, replace
from typing import TYPE_CHECKING, cast

import itertools
import re

from configgle import Fig, Makeable

import numpy as np

from priml.baselines.name_that_llm.sparse import CsrMatrix


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class CharacterNgrams:
    """Enumerate character substrings after repeated-whitespace normalization."""

    class Config(Fig["CharacterNgrams"]):
        """Substring lengths included in the feature vocabulary."""

        minimum: int = 1
        """Shortest substring length."""

        maximum: int = 5
        """Longest substring length."""

    def __init__(self, config: Config) -> None:
        if config.minimum < 1 or config.maximum < config.minimum:
            raise ValueError("Expected positive, ordered character substring lengths.")
        self.config = config

    def __call__(self, text: str) -> list[str]:
        """Enumerate substrings in length-major, left-to-right order."""
        normalized = re.sub(r"\s\s+", repl=" ", string=text)
        return [
            normalized[start : start + size]
            for size in range(
                self.config.minimum,
                min(self.config.maximum, len(normalized)) + 1,
            )
            for start in range(len(normalized) - size + 1)
        ]


class WordNgrams:
    """Tokenize, remove configured words, and enumerate contiguous token groups."""

    class Config(Fig["WordNgrams"]):
        """Token expression, group lengths, and omitted word vocabulary."""

        token_pattern: str = r"(?u)\b\w\w+\b"  # noqa: S105 -- Token regex, not a credential.
        """Regular expression identifying complete word tokens."""

        minimum: int = 1
        """Shortest number of words per feature."""

        maximum: int = 1
        """Longest number of words per feature."""

        stop_words: list[str] = field(default_factory=list[str])
        """Words discarded before constructing multiword features."""

    def __init__(self, config: Config) -> None:
        if config.minimum < 1 or config.maximum < config.minimum:
            raise ValueError("Expected positive, ordered word group lengths.")
        self.config = config
        self.pattern = re.compile(config.token_pattern)
        self.ignored = frozenset(config.stop_words)

    def __call__(self, text: str) -> list[str]:
        """Enumerate retained word groups in length-major, left-to-right order."""
        tokens = [
            match[0]
            for match in self.pattern.finditer(text)
            if match[0] not in self.ignored
        ]
        return [
            " ".join(tokens[start : start + size])
            for size in range(
                self.config.minimum,
                min(self.config.maximum, len(tokens)) + 1,
            )
            for start in range(len(tokens) - size + 1)
        ]


def english_words() -> WordNgrams.Config:
    """Configure the pinned English stop vocabulary.

    Returns:
      config: Word analyzer with all 318 public recipe stop words.

    """
    config = WordNgrams.Config()
    config.stop_words = [
        "a",
        "about",
        "above",
        "across",
        "after",
        "afterwards",
        "again",
        "against",
        "all",
        "almost",
        "alone",
        "along",
        "already",
        "also",
        "although",
        "always",
        "am",
        "among",
        "amongst",
        "amoungst",  # codespell:ignore amoungst
        "amount",
        "an",
        "and",
        "another",
        "any",
        "anyhow",
        "anyone",
        "anything",
        "anyway",
        "anywhere",
        "are",
        "around",
        "as",
        "at",
        "back",
        "be",
        "became",
        "because",
        "become",
        "becomes",
        "becoming",
        "been",
        "before",
        "beforehand",
        "behind",
        "being",
        "below",
        "beside",
        "besides",
        "between",
        "beyond",
        "bill",
        "both",
        "bottom",
        "but",
        "by",
        "call",
        "can",
        "cannot",
        "cant",  # codespell:ignore cant
        "co",
        "con",
        "could",
        "couldnt",  # codespell:ignore couldnt
        "cry",
        "de",
        "describe",
        "detail",
        "do",
        "done",
        "down",
        "due",
        "during",
        "each",
        "eg",
        "eight",
        "either",
        "eleven",
        "else",
        "elsewhere",
        "empty",
        "enough",
        "etc",
        "even",
        "ever",
        "every",
        "everyone",
        "everything",
        "everywhere",
        "except",
        "few",
        "fifteen",
        "fifty",
        "fill",
        "find",
        "fire",
        "first",
        "five",
        "for",
        "former",
        "formerly",
        "forty",
        "found",
        "four",
        "from",
        "front",
        "full",
        "further",
        "get",
        "give",
        "go",
        "had",
        "has",
        "hasnt",  # codespell:ignore hasnt
        "have",
        "he",
        "hence",
        "her",
        "here",
        "hereafter",
        "hereby",
        "herein",
        "hereupon",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "however",
        "hundred",
        "i",
        "ie",
        "if",
        "in",
        "inc",
        "indeed",
        "interest",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "keep",
        "last",
        "latter",
        "latterly",
        "least",
        "less",
        "ltd",
        "made",
        "many",
        "may",
        "me",
        "meanwhile",
        "might",
        "mill",
        "mine",
        "more",
        "moreover",
        "most",
        "mostly",
        "move",
        "much",
        "must",
        "my",
        "myself",
        "name",
        "namely",
        "neither",
        "never",
        "nevertheless",
        "next",
        "nine",
        "no",
        "nobody",
        "none",
        "noone",  # codespell:ignore noone
        "nor",
        "not",
        "nothing",
        "now",
        "nowhere",
        "of",
        "off",
        "often",
        "on",
        "once",
        "one",
        "only",
        "onto",
        "or",
        "other",
        "others",
        "otherwise",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "part",
        "per",
        "perhaps",
        "please",
        "put",
        "rather",
        "re",
        "same",
        "see",
        "seem",
        "seemed",
        "seeming",
        "seems",
        "serious",
        "several",
        "she",
        "should",
        "show",
        "side",
        "since",
        "sincere",
        "six",
        "sixty",
        "so",
        "some",
        "somehow",
        "someone",
        "something",
        "sometime",
        "sometimes",
        "somewhere",
        "still",
        "such",
        "system",
        "take",
        "ten",
        "than",
        "that",
        "the",
        "their",
        "them",
        "themselves",
        "then",
        "thence",
        "there",
        "thereafter",
        "thereby",
        "therefore",
        "therein",
        "thereupon",
        "these",
        "they",
        "thick",
        "thin",
        "third",
        "this",
        "those",
        "though",
        "three",
        "through",
        "throughout",
        "thru",
        "thus",
        "to",
        "together",
        "too",
        "top",
        "toward",
        "towards",
        "twelve",
        "twenty",
        "two",
        "un",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "very",
        "via",
        "was",
        "we",
        "well",
        "were",
        "what",
        "whatever",
        "when",
        "whence",
        "whenever",
        "where",
        "whereafter",
        "whereas",
        "whereby",
        "wherein",
        "whereupon",
        "wherever",
        "whether",
        "which",
        "while",
        "whither",
        "who",
        "whoever",
        "whole",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "within",
        "without",
        "would",
        "yet",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
    return config


class TfidfTransformer:
    """Fit smoothed inverse document weights and normalize sublinear term counts."""

    class Config(Fig["TfidfTransformer"]):
        """Construct smoothed IDF, sublinear term weighting, and Euclidean scaling."""

    def __init__(self, config: Config) -> None:
        del config
        self.n_features_in_ = 0
        self.idf_ = np.zeros(0)

    def fit(self, counts: CsrMatrix) -> TfidfTransformer:
        """Fit the complete smoothed inverse-document-frequency vector.

        Args:
          counts: Sparse nonnegative term counts, documents by features.

        Returns:
          transformer: This fitted numerical transform.

        """
        self.n_features_in_ = counts.shape[1]
        documents = np.bincount(counts.indices, minlength=self.n_features_in_)
        self.idf_ = np.log((counts.shape[0] + 1) / (documents + 1)) + 1
        return self

    def transform(self, counts: CsrMatrix) -> CsrMatrix:
        """Weight and normalize a copy of every sparse document row.

        Args:
          counts: Sparse counts in the fitted feature vocabulary.

        Returns:
          features: Sublinear, inverse-document-weighted unit-norm rows.

        """
        if counts.shape[1] != self.n_features_in_:
            raise ValueError("Expected the fitted number of lexical features.")
        data = np.log(counts.data)
        data += 1
        data *= self.idf_[counts.indices]
        # Every stored weight is at least one, so only empty rows have norm zero,
        # and dividing their empty slice is a no-op.
        for start, stop in itertools.pairwise(counts.indptr.tolist()):
            squared_norm = 0.0
            for value in cast("list[float]", data[start:stop].tolist()):
                squared_norm += value * value
            data[start:stop] /= cast("np.float64", np.sqrt(squared_norm))
        return replace(counts, data=data)


class TfidfVectorizer:
    """Fit a lexical vocabulary and sparse document feature transform."""

    class Config(Fig["TfidfVectorizer"]):
        """Vocabulary selection and injectable text and numerical transforms."""

        analyzer: Makeable[Callable[[str], list[str]]] = field(
            default_factory=WordNgrams.Config,
        )
        """Text-to-feature analyzer applied after preprocessing."""

        preprocess: Callable[[str], str] = str.lower
        """Text transform applied before the analyzer."""

        min_documents: int = 1
        """Minimum number of documents containing a retained feature."""

        max_features: int | None = None
        """Maximum retained features, ranked by corpus term frequency."""

        transformer: TfidfTransformer.Config = field(
            default_factory=TfidfTransformer.Config,
        )
        """Fitted numerical weighting and row scaling."""

    def __init__(self, config: Config) -> None:
        if config.min_documents < 1:
            raise ValueError("Expected a positive minimum document count.")
        if config.max_features is not None and config.max_features < 1:
            raise ValueError("Expected a positive maximum feature count.")
        self.config = config
        self.analyzer = config.analyzer.make()
        self.transformer = config.transformer.make()
        self.vocabulary_: dict[str, int] = {}

    def fit_transform(self, texts: Sequence[str]) -> CsrMatrix:
        """Fit feature identities and return all weighted training rows.

        Args:
          texts: Complete training documents in their original row order.

        Returns:
          features: Sparse, normalized lexical training features.

        """
        self.vocabulary_ = {}
        counts = self._counts(texts, expand=True)
        if not self.vocabulary_:
            raise ValueError("Expected a nonempty lexical vocabulary.")
        if self.config.max_features is not None:
            counts = self._sort_vocabulary(counts)
        counts = self._prune(counts)
        if self.config.max_features is None:
            counts = self._sort_vocabulary(counts)
        self.transformer.fit(counts)
        return self.transformer.transform(counts)

    def transform(self, texts: Sequence[str]) -> CsrMatrix:
        """Map held-out documents into the fitted feature vocabulary."""
        if not self.vocabulary_:
            raise ValueError("Expected a fitted lexical vocabulary.")
        return self.transformer.transform(self._counts(texts))

    @property
    def vocabulary(self) -> list[str]:
        """Return every feature name in column order."""
        return sorted(self.vocabulary_, key=self.vocabulary_.__getitem__)

    @property
    def idf(self) -> NDArray[np.float64]:
        """Return the complete fitted inverse document weights."""
        return self.transformer.idf_

    def _counts(self, texts: Sequence[str], *, expand: bool = False) -> CsrMatrix:
        values: list[float] = []
        indices: list[int] = []
        boundaries = [0]
        for text in texts:
            row: dict[int, float] = {}
            for token in self.analyzer(self.config.preprocess(text)):
                if token not in self.vocabulary_:
                    if not expand:
                        continue
                    self.vocabulary_[token] = len(self.vocabulary_)
                index = self.vocabulary_[token]
                row[index] = row.get(index, 0.0) + 1
            indices.extend(row)
            values.extend(row.values())
            boundaries.append(len(indices))
        return CsrMatrix(
            data=np.array(values),
            indices=np.array(indices, dtype=np.int32),
            indptr=np.array(boundaries, dtype=np.int32),
            shape=(len(texts), len(self.vocabulary_)),
        ).sorted_indices()

    def _sort_vocabulary(self, counts: CsrMatrix) -> CsrMatrix:
        remap = np.empty(len(self.vocabulary_), dtype=np.int32)
        for index, term in enumerate(sorted(self.vocabulary_)):
            remap[self.vocabulary_[term]] = index
            self.vocabulary_[term] = index
        # Rows stay in stored order, unsorted under the new numbering: every
        # stored-order row sum downstream depends on it.
        return replace(counts, indices=remap[counts.indices])

    def _prune(self, counts: CsrMatrix) -> CsrMatrix:
        # Every vocabulary entry was counted, so each bincount spans the vocabulary.
        keep = np.bincount(counts.indices) >= self.config.min_documents
        if self.config.max_features is not None:
            total = np.bincount(counts.indices, weights=counts.data)
            priorities = -total[keep]
            preferred = np.flatnonzero(keep)[
                priorities.argsort(kind="stable")[: self.config.max_features]
            ]
            keep = np.zeros_like(keep)
            keep[preferred] = True
        retained = np.flatnonzero(keep)
        if len(retained) == 0:
            raise ValueError("After pruning, no lexical features remain.")
        remap = {old: new for new, old in enumerate(retained.tolist())}
        self.vocabulary_ = {
            term: remap[index]
            for term, index in self.vocabulary_.items()
            if index in remap
        }
        return counts.take_columns(retained)
