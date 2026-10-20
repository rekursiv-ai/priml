"""Replay whole vocabularies and sparse lexical features from numeric witnesses."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import torch

from priml.baselines.name_that_llm.tfidf import (
    CharacterNgrams,
    TfidfTransformer,
    TfidfVectorizer,
    WordNgrams,
    english_words,
)
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize("name", argvalues=["char", "limit", "word", "topic"])
@pytest.mark.parametrize("corrupt", argvalues=[False, True])
def test_tfidf_replays_every_sparse_value_and_vocabulary(
    name: str,
    corrupt: bool,
) -> None:
    config = TfidfVectorizer.Config()
    if name in {"char", "limit"}:
        config.analyzer = CharacterNgrams.Config()
        config.preprocess = str
        config.min_documents = 2
        config.max_features = 300_000 if name == "char" else 7
    elif name == "word":
        analyzer = WordNgrams.Config()
        analyzer.token_pattern = r"\S+"  # noqa: S105 -- Token regex, not a credential.
        analyzer.maximum = 2
        config.analyzer = analyzer
        config.preprocess = str
        config.min_documents = 2
    else:
        config.analyzer = english_words()
    model = config.make()
    matrices = {
        "fit": model.fit_transform(
            [
                "Ax by",
                "Ax cz",
                "Dx by",
                "Dx cz",
                "Ax\tby",
                "Dx  cz",
                "Äx by",
                "Ax Ax cz",
            ],
        ),
        "predict": model.transform(["Dx by", "", "Ax  cz", "Äx\tby", "new"]),
    }
    actual = {
        "idf": torch.from_numpy(model.idf),
        "vocabulary": torch.tensor(
            list("\0".join(model.vocabulary).encode()),
            dtype=torch.uint8,
        ),
    }
    if name == "topic":
        actual["stop_words"] = torch.tensor(
            list("\0".join(sorted(english_words().stop_words)).encode()),
            dtype=torch.uint8,
        )
    if corrupt:
        actual["idf"] = actual["idf"].clone()
        actual["idf"][:1] = torch.nextafter(
            actual["idf"][:1],
            other=torch.tensor(torch.inf),
        )
    for split, matrix in matrices.items():
        actual[f"{split}/shape"] = torch.tensor(matrix.shape)
        actual[f"{split}/data"] = torch.from_numpy(matrix.data)
        actual[f"{split}/indices"] = torch.from_numpy(matrix.indices)
        actual[f"{split}/indptr"] = torch.from_numpy(matrix.indptr)
    prefix = f"{name}/"
    expected = {
        key.removeprefix(prefix): value
        for key, value in read_tensors(_CWD / "testdata" / "tfidf.pt").items()
        if key.startswith(prefix)
    }
    report = mismatches(expected, actual=actual)
    if corrupt:
        assert any(item.startswith("idf:") for item in report), report
    else:
        assert not report, "\n".join(report)


@pytest.mark.parametrize(
    "config",
    argvalues=[CharacterNgrams.Config(), WordNgrams.Config()],
)
@pytest.mark.parametrize(("minimum", "maximum"), argvalues=[(0, 5), (3, 2)])
def test_analyzers_reject_invalid_lengths(
    config: CharacterNgrams.Config | WordNgrams.Config,
    minimum: int,
    maximum: int,
) -> None:
    config = config.copy_tree()
    config.minimum, config.maximum = minimum, maximum
    with pytest.raises(ValueError, match="positive, ordered"):
        config.make()


@pytest.mark.parametrize(("minimum", "maximum"), argvalues=[(0, None), (1, 0)])
def test_vectorizer_rejects_invalid_feature_limits(
    minimum: int,
    maximum: int | None,
) -> None:
    config = TfidfVectorizer.Config()
    config.min_documents, config.max_features = minimum, maximum
    with pytest.raises(ValueError, match="positive"):
        config.make()


def test_vectorizer_rejects_unfitted_empty_and_fully_pruned_vocabularies() -> None:
    config = TfidfVectorizer.Config()
    model = config.make()
    with pytest.raises(ValueError, match="fitted lexical vocabulary"):
        model.transform(["word"])
    with pytest.raises(ValueError, match="nonempty lexical vocabulary"):
        model.fit_transform(["", "a"])
    config.min_documents = 2
    with pytest.raises(ValueError, match="no lexical features remain"):
        config.make().fit_transform(["word", "other"])


def test_vectorizer_refit_replaces_vocabulary_and_preserves_empty_prediction_rows() -> (
    None
):
    model = TfidfVectorizer.Config().make()
    model.fit_transform(["previous", "features"])
    model.fit_transform(["current", "tokens"])
    assert model.vocabulary == ["current", "tokens"]
    assert model.transform(["previous features", ""]).nnz == 0


def test_transformer_rejects_feature_width_mismatch() -> None:
    model = TfidfVectorizer.Config().make()
    counts = model.fit_transform(["two words", "three distinct words"])
    transformer = TfidfTransformer.Config().make().fit(counts)
    with pytest.raises(ValueError, match="fitted number of lexical features"):
        transformer.transform(counts[:, :2])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
