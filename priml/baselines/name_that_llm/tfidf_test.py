"""Pin lexical vocabularies, analyzers and sparse TF-IDF rows."""

from __future__ import annotations

import numpy as np
import pytest

from priml.baselines.name_that_llm.model import DocumentFeatures
from priml.baselines.name_that_llm.sparse import CsrMatrix
from priml.baselines.name_that_llm.tfidf import (
    CharacterNgrams,
    TfidfTransformer,
    TfidfVectorizer,
    WordNgrams,
    english_words,
)


def test_word_features_keep_case_pairs_and_terms_in_every_document() -> None:
    model = DocumentFeatures.Config().word.make()
    fitted = model.fit_transform(["Ab cd ef", "Ab cd EF"])
    # Terms in one document are pruned; each kept term occurs once in both, so its
    # weight is log(3 / 3) + 1 = 1 and every three-term row is 1 / sqrt(3).
    assert model.vocabulary == ["Ab", "Ab cd", "cd"]
    assert model.idf.tolist() == [1, 1, 1]
    assert fitted.shape == (2, 3)
    assert fitted.indptr.tolist() == [0, 3, 6]
    # Rows keep first-seen order (Ab, cd, then the pair) under sorted numbering.
    assert fitted.indices.tolist() == [0, 2, 1] * 2
    assert fitted.data.tolist() == [1 / np.sqrt(3)] * 6
    predicted = model.transform(["Ab EF", ""])
    assert predicted.indptr.tolist() == [0, 1, 1]
    assert predicted.indices.tolist() == [0]
    assert predicted.data.tolist() == [1]


def test_vectorizer_lowercases_documents_before_analysis_by_default() -> None:
    model = TfidfVectorizer.Config().make()
    model.fit_transform(["Cats purr", "cats PURR"])
    assert model.vocabulary == ["cats", "purr"]
    assert model.transform(["CATS"]).indices.tolist() == [0]


def test_vectorizer_sorts_each_row_by_the_numbering_it_counts_under() -> None:
    model = TfidfVectorizer.Config().make()
    # Fitting counts under first-seen numbering (bb, aa), then renumbers aa first.
    assert model.fit_transform(["bb aa", "aa bb"]).indices.tolist() == [1, 0] * 2
    assert model.transform(["bb aa"]).indices.tolist() == [0, 1]


def test_english_words_hold_the_318_distinct_recipe_stop_words() -> None:
    words = english_words().stop_words
    assert len(words) == len(set(words)) == 318
    assert words == sorted(words)
    assert (words[0], words[-1]) == ("a", "yourselves")


def test_english_words_drop_stop_words_before_pairing_terms() -> None:
    config = english_words()
    config.maximum = 2
    # "by" and "the" are stop words, so "sat" pairs with "dogs".
    assert config.make()("cats sat by the dogs") == [
        "cats",
        "sat",
        "dogs",
        "cats sat",
        "sat dogs",
    ]


def test_vectorizer_feature_cap_breaks_ties_by_term_order() -> None:
    config = TfidfVectorizer.Config()
    config.max_features = 3
    model = config.make()
    terms = [f"t{index:02d}" for index in range(20)]
    # Nineteen terms tie behind t19, listed in reverse: past sixteen, NumPy's default
    # sort stops keeping order, and first-seen order would keep t18 and t17.
    model.fit_transform([" ".join(reversed(terms)), "t19"])
    assert model.vocabulary == ["t00", "t01", "t19"]


@pytest.mark.parametrize(
    ("config", "noun"),
    argvalues=[
        (CharacterNgrams.Config(), "character substring"),
        (WordNgrams.Config(), "word group"),
    ],
)
@pytest.mark.parametrize(("minimum", "maximum"), argvalues=[(0, 5), (3, 2)])
def test_analyzers_reject_invalid_lengths(
    config: CharacterNgrams.Config | WordNgrams.Config,
    noun: str,
    minimum: int,
    maximum: int,
) -> None:
    config = config.copy_tree()
    config.minimum, config.maximum = minimum, maximum
    with pytest.raises(
        ValueError,
        match=rf"^Expected positive, ordered {noun} lengths\.$",
    ):
        config.make()


def test_character_ngrams_stop_at_the_configured_length() -> None:
    config = CharacterNgrams.Config()
    config.minimum = config.maximum = 2
    assert config.make()("abc  d") == ["ab", "bc", "c ", " d"]
    config.minimum = 1
    assert config.make()("abc") == ["a", "b", "c", "ab", "bc"]


def test_character_ngrams_keep_lone_whitespace_and_collapse_mixed_runs() -> None:
    config = CharacterNgrams.Config()
    config.minimum = config.maximum = 2
    # The lone tab stays; the run of a newline and a tab becomes one space.
    assert config.make()("a\tb\n\tc") == ["a\t", "\tb", "b ", " c"]


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    argvalues=[
        (0, None, "minimum document count"),
        (1, 0, "maximum feature count"),
    ],
)
def test_vectorizer_rejects_invalid_feature_limits(
    minimum: int,
    maximum: int | None,
    message: str,
) -> None:
    config = TfidfVectorizer.Config()
    config.min_documents, config.max_features = minimum, maximum
    with pytest.raises(ValueError, match=rf"^Expected a positive {message}\.$"):
        config.make()


def test_vectorizer_keeps_the_single_most_frequent_feature() -> None:
    config = TfidfVectorizer.Config()
    config.max_features = 1
    model = config.make()
    model.fit_transform(["rare common", "common"])
    assert model.vocabulary == ["common"]


def test_vectorizer_rejects_unfitted_empty_and_fully_pruned_vocabularies() -> None:
    config = TfidfVectorizer.Config()
    model = config.make()
    assert model.vocabulary == []
    with pytest.raises(ValueError, match=r"^Expected a fitted lexical vocabulary\.$"):
        model.transform(["word"])
    with pytest.raises(ValueError, match=r"^Expected a nonempty lexical vocabulary\.$"):
        model.fit_transform(["", "a"])
    config.min_documents = 2
    with pytest.raises(
        ValueError,
        match=r"^After pruning, no lexical features remain\.$",
    ):
        config.make().fit_transform(["word", "other"])


def test_vectorizer_refit_replaces_vocabulary_and_preserves_empty_prediction_rows() -> (
    None
):
    model = TfidfVectorizer.Config().make()
    model.fit_transform(["previous", "features"])
    model.fit_transform(["current", "tokens"])
    assert model.vocabulary == ["current", "tokens"]
    assert len(model.transform(["previous features", ""]).data) == 0


def test_transformer_rejects_feature_width_mismatch() -> None:
    model = TfidfVectorizer.Config().make()
    counts = model.fit_transform(["two words", "three distinct words"])
    transformer = TfidfTransformer.Config().make().fit(counts)
    with pytest.raises(
        ValueError,
        match=r"^Expected the fitted number of lexical features\.$",
    ):
        transformer.transform(counts.take_columns(np.arange(2)))


def test_transformer_starts_without_features() -> None:
    transformer = TfidfTransformer.Config().make()
    assert transformer.n_features_in_ == 0
    assert transformer.idf_.dtype == np.float64
    assert transformer.idf_.shape == (0,)


def test_transformer_weights_a_column_no_document_contains() -> None:
    counts = CsrMatrix.from_dense(
        np.array([[1, 0, 2, 0], [0, 1, 0, 0], [3, 0, 0, 0]], dtype=np.float64),
    )
    transformer = TfidfTransformer.Config().make().fit(counts)
    assert transformer.n_features_in_ == 4
    # Three documents; the columns appear in two, one, one and none of them.
    expected = np.log((3 + 1) / (np.array([2, 1, 1, 0]) + 1)) + 1
    assert np.array_equal(transformer.idf_, a2=expected)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
