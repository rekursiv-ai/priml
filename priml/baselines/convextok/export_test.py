"""The exported tokenizer must segment text exactly as upstream's tokenizer does.

Upstream gives every piece the score -1, so a Unigram tokenizer picks the fewest
pieces. The export keeps that model in the nanochat baseline's layout -- the 256
bytes first, then the learned pieces, with reserved IDs appended by the loader --
and must produce upstream's token strings on held-out text.
"""

from pathlib import Path
from typing import Final, cast

import json

from tokenizers import pre_tokenizers

import pytest

from priml.baselines.convextok.export import export_tokenizer
from priml.lib.custom_json import convert


_CWD: Final = Path(__file__).resolve().parent


def test_held_out_text_segments_as_upstream() -> None:
    alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    learned = [
        token
        for token in convert(_read_json("vocab.json").get("det"), list[str])
        if token not in alphabet
    ]
    corpus = _read_json("corpus.json")
    tokenizer = export_tokenizer(
        learned,
        split_pattern=str(corpus.get("split_pattern")),
    )
    texts = convert(corpus.get("heldout"), list[str])
    golden = [
        convert(row, list[str])
        for row in convert(
            _read_json("heldout_tokens.json").get("tokens"),
            list[object],
        )
    ]
    assert [
        tokenizer.encode(text, add_special_tokens=False).tokens for text in texts
    ] == golden


def test_bytes_come_first_then_learned_pieces_in_order() -> None:
    tokenizer = export_tokenizer(["ab", "Ġthe"], split_pattern=r"\s+|\S+")
    assert tokenizer.get_vocab_size() == 258
    assert tokenizer.id_to_token(ord("A")) == "A"
    assert [tokenizer.id_to_token(256), tokenizer.id_to_token(257)] == ["ab", "Ġthe"]


def test_fewest_pieces_win() -> None:
    tokenizer = export_tokenizer(["ab", "bc", "abc"], split_pattern=r"\S+")
    assert tokenizer.encode("abc", add_special_tokens=False).tokens == ["abc"]


def test_model_scores_and_byte_fallback_are_exact() -> None:
    tokenizer = export_tokenizer(["ab"], split_pattern=r"\S+")
    serialized = tokenizer.to_str()
    assert '"unk_id":null' in serialized
    assert '"byte_fallback":false' in serialized
    assert '"use_regex":false' in serialized
    assert '["ab",-1.0]' in serialized
    assert '["A",-1.0]' in serialized


def test_text_round_trips() -> None:
    tokenizer = export_tokenizer(["ab"], split_pattern=r"\s+|\S+")
    text = "naïve ab\tcafé 🦙"
    assert (
        tokenizer.decode(tokenizer.encode(text, add_special_tokens=False).ids) == text
    )


def test_single_bytes_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Learned pieces span two or more bytes; single bytes are built in\.$",
    ):
        export_tokenizer(["a"], split_pattern=r"\S+")


def _read_json(name: str) -> dict[str, object]:
    raw = cast(object, json.loads((_CWD / "testdata" / name).read_text()))
    return dict(convert(raw, dict[str, object]))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
