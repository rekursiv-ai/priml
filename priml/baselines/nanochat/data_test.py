"""Tests for nanochat data loading."""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pytest
import torch

from priml.baselines.nanochat.data import NanoChatData, token_bytes_fingerprint


VOCAB = 16
SEQ = 8


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Write a tiny two-split dataset in the on-disk layout."""
    for split, rows in (("train", 6), ("val", 4)):
        directory = tmp_path / split
        directory.mkdir()
        # Distinguishable rows: row i counts up from i.
        tokens = (np.arange(rows)[:, None] + np.arange(SEQ + 1)[None, :]) % VOCAB
        np.save(directory / "all__tokens.npy", tokens.astype(np.uint16))
        # Token 0 is the document marker and carries no bytes; the rest carry
        # one, so a byte count is a token count and assertions stay readable.
        lengths = np.ones(VOCAB, dtype=np.int32)
        lengths[0] = 0
        np.save(directory / "all__token_bytes.npy", lengths)
        (directory / "dataset.json").write_text(
            json.dumps(
                {
                    "vocab_size": VOCAB,
                    "max_seq_len": SEQ,
                    "token_bytes_sha256": token_bytes_fingerprint(lengths),
                },
            ),
        )
    return tmp_path


def _data(dataset_dir: Path, **overrides: object) -> NanoChatData:
    config = NanoChatData.Config()
    config.base_dir = "/"
    config.working_dir = str(dataset_dir)
    config.device = "cpu"
    config.batch_size = 2
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_targets_are_the_inputs_shifted_by_one(dataset_dir: Path) -> None:
    """The whole training signal: position i predicts position i + 1."""
    batch = next(iter(_data(dataset_dir).train_dataloader()))
    assert batch["media"].shape == (2, SEQ)
    assert batch["label"].shape == (2, SEQ)
    assert torch.equal(batch["media"][:, 1:], batch["label"][:, :-1])


def test_every_batch_is_full_width(dataset_dir: Path) -> None:
    """A short final batch is dropped, not padded.

    Every row is a full context by construction, so a partial batch would be
    the only place in a run where the token count per step changes -- and the
    token batch is what the recipe is tuned against.
    """
    batches = list(_data(dataset_dir, batch_size=4).train_dataloader())
    assert [b["media"].shape[0] for b in batches] == [4]  # 6 rows, not 8


def test_evaluation_is_never_shuffled(dataset_dir: Path) -> None:
    """Every candidate must be scored on the identical token stream."""
    data = _data(dataset_dir)
    first = [b["media"].clone() for b in data.eval_dataloader()]
    second = [b["media"].clone() for b in data.eval_dataloader()]
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_seeded_shuffle_is_reproducible_and_pass_varying(
    dataset_dir: Path,
) -> None:
    """One seed fixes each pass's order, and consecutive passes differ."""
    orders = [
        next(iter(_data(dataset_dir, seed=7).train_dataloader()))["media"].clone()
        for _ in range(2)
    ]
    assert torch.equal(orders[0], orders[1])

    loader = _data(dataset_dir, seed=7).train_dataloader()
    first = next(iter(loader))["media"].clone()
    second = next(iter(loader))["media"].clone()
    assert not torch.equal(first, second)


def test_pass_counter_round_trips(dataset_dir: Path) -> None:
    """Resume continues the sequence rather than replaying the first pass."""
    data = _data(dataset_dir, seed=1)
    loader = data.train_dataloader()
    list(loader)
    state = data.state_dict()
    assert state["passes"] == 1

    restored = _data(dataset_dir, seed=1)
    restored.load_state_dict(state)
    assert torch.equal(
        next(iter(restored.train_dataloader()))["media"],
        next(iter(loader))["media"],
    )


def test_the_byte_table_travels_with_the_batch(dataset_dir: Path) -> None:
    """The score divides by it, so it must reach the metric unmediated."""
    batch = next(iter(_data(dataset_dir).eval_dataloader()))
    assert batch["token_bytes"].shape == (VOCAB,)
    assert int(batch["token_bytes"][0]) == 0


def test_evaluation_rows_can_be_capped(dataset_dir: Path) -> None:
    """Mid-training eval reads a prefix; the final number reads everything."""
    batches = list(_data(dataset_dir, num_eval_rows=2).eval_dataloader())
    assert sum(b["media"].shape[0] for b in batches) == 2


def test_a_byte_table_that_does_not_match_its_fingerprint_is_rejected(
    dataset_dir: Path,
) -> None:
    """The byte table IS the score's denominator, so its identity is recorded.

    Two tables of equal length silently reprice every token: a change to byte
    accounting shifts BPB by roughly a real candidate effect while every shape
    check still passes, and two runs become incomparable with nothing on disk
    to tell them apart.
    """
    lengths = np.ones(VOCAB, dtype=np.int32)
    lengths[0] = 0
    lengths[3] = 7  # same shape, different accounting
    np.save(dataset_dir / "val" / "all__token_bytes.npy", lengths)
    with pytest.raises(ValueError, match="fingerprint"):
        list(_data(dataset_dir).eval_dataloader())


def test_a_byte_table_disagreeing_with_the_metadata_is_rejected(
    dataset_dir: Path,
) -> None:
    """A stale table would silently misprice every token in the score.

    A wrong-LENGTH table trips the fingerprint first, since the identity check
    runs before the shape one; either way it never reaches the score.
    """
    np.save(
        dataset_dir / "val" / "all__token_bytes.npy",
        np.ones(VOCAB + 1, dtype=np.int32),
    )
    with pytest.raises(ValueError, match=r"fingerprint|byte table"):
        list(_data(dataset_dir).eval_dataloader())


def test_data_narrower_than_the_model_is_rejected_at_load(dataset_dir: Path) -> None:
    """Prepared rows must be the context the model declares.

    Otherwise the mismatch surfaces deep in the forward as "Input length 2048
    exceeds max_seq_len=128", naming only the model's side and never the
    directory that produced the rows.
    """
    with pytest.raises(ValueError, match="max_seq_len"):
        list(_data(dataset_dir, max_seq_len=SEQ * 2).train_dataloader())


def test_a_token_id_outside_the_vocabulary_is_rejected_at_load(
    dataset_dir: Path,
) -> None:
    """A row indexing past the embedding must fail here, not in a matmul."""
    with pytest.raises(ValueError, match="vocab_size"):
        list(_data(dataset_dir, vocab_size=VOCAB // 2).train_dataloader())


def test_declared_geometry_matching_the_data_loads(dataset_dir: Path) -> None:
    """The check must accept the agreeing case, or it blocks every real run."""
    data = _data(dataset_dir, vocab_size=VOCAB, max_seq_len=SEQ)
    batch = next(iter(data.train_dataloader()))
    assert batch["media"].shape == (2, SEQ)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("eval_batch_size", 0),
        ("eval_batch_size", -1),
        ("num_eval_rows", 0),
        ("num_eval_rows", -1),
    ],
)
def test_a_nonpositive_size_is_rejected_by_name(
    dataset_dir: Path,
    field: str,
    value: int,
) -> None:
    """Each bound names its own field, and none is silently absorbed.

    ``eval_batch_size=0`` is the one that bites hardest: zero is FALSY, so an
    ``or`` fallback would quietly evaluate at the training batch size and still
    report a number. ``num_eval_rows=-1`` would silently mean "all but the last
    row" rather than an invalid cap.
    """
    with pytest.raises(ValueError, match=field):
        _data(dataset_dir, **{field: value})


def test_an_omitted_eval_batch_size_still_reuses_the_train_one(
    dataset_dir: Path,
) -> None:
    """``None`` is the sentinel; rejecting 0 must not break the real default."""
    assert _data(dataset_dir, eval_batch_size=None).eval_batch_size == 2


def test_missing_data_names_the_preparer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        list(_data(tmp_path).train_dataloader())


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
