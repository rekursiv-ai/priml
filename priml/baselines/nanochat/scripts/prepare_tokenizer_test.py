"""Verify fitting text order, deterministic windows, and byte tokenizers."""

from pathlib import Path

from pyarrow import Table, parquet

from priml.baselines.nanochat.scripts.prepare_tokenizer import (
    byte_alphabet,
    frequency_model,
    load_sample,
    sample_window,
)


def test_sample_preserves_text_and_order_without_provenance(tmp_path: Path) -> None:
    texts = ["A café 🦙", " second document\n", "A café 🦙"]
    parquet.write_table(
        Table.from_pydict({"text": texts}),
        tmp_path / "sample.parquet",
    )
    assert load_sample(tmp_path) == texts


def test_utf8_windows_and_byte_vocabulary() -> None:
    raw = ("é🦙abc" * 9).encode()
    start, end = sample_window(raw, max_bytes=9)
    assert 0 <= start < end <= len(raw)
    assert end - start <= 9
    assert raw[start:end].decode().encode() == raw[start:end]
    alphabet = byte_alphabet()
    assert len(alphabet) == len(set(alphabet.values())) == 256
    assert alphabet[32] == "Ġ"
    model = frequency_model(
        [bytes([value]) for value in range(256)],
        counts=[1] * 256,
        split_pattern=r"\S+|\s+",
    )
    text = "café 🦙\n"
    assert model.decode(model.encode(text).ids) == text


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
