"""Tests for nanochat data preparation.

Every case here is about the same hazard: preparation writes the artifact a
score is measured against, so a mismatch it accepts becomes a number nobody can
attribute. None of these tests downloads anything -- the tokenizer and the
shards are the parts that need a network, and each case fails before reaching
them.
"""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pytest

from priml.baselines.nanochat.data import token_bytes_fingerprint
from priml.baselines.nanochat.scripts.prepare_data import prepare


VOCAB = 16
SEQ = 8


def _split(root: Path, name: str, *, vocab_size: int, max_seq_len: int) -> None:
    """Write one prepared split in the on-disk layout."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    np.save(
        directory / "all__tokens.npy",
        np.zeros((4, max_seq_len + 1), dtype=np.uint16),
    )
    lengths = np.ones(vocab_size, dtype=np.int32)
    lengths[0] = 0
    np.save(directory / "all__token_bytes.npy", lengths)
    (directory / "dataset.json").write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "max_seq_len": max_seq_len,
                "token_bytes_sha256": token_bytes_fingerprint(lengths),
            },
        ),
    )


def test_reusing_a_split_prepared_under_other_flags_is_refused(
    tmp_path: Path,
) -> None:
    """Returning it would hand back data that is not what was asked for."""
    for name in ("train", "val"):
        _split(tmp_path, name, vocab_size=VOCAB, max_seq_len=SEQ)
    with pytest.raises(ValueError, match="max_seq_len"):
        prepare(tmp_path, vocab_size=VOCAB, max_seq_len=SEQ * 2)


def test_a_partial_directory_is_checked_too(tmp_path: Path) -> None:
    """A missing split is exactly when the flags are most likely to have moved.

    Checking only when BOTH exist would rebuild the missing one at the new
    geometry beside a stale one at the old, and report the pair as ready.
    """
    _split(tmp_path, "train", vocab_size=VOCAB, max_seq_len=SEQ)
    with pytest.raises(ValueError, match="max_seq_len"):
        prepare(tmp_path, vocab_size=VOCAB, max_seq_len=SEQ * 2)


def test_an_intact_directory_at_the_same_flags_is_reused(tmp_path: Path) -> None:
    """The check must not block the case it exists to make safe."""
    for name in ("train", "val"):
        _split(tmp_path, name, vocab_size=VOCAB, max_seq_len=SEQ)
    assert prepare(tmp_path, vocab_size=VOCAB, max_seq_len=SEQ) == tmp_path


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
