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
    config.eval_batch_size = 2
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


def test_data_is_verified_even_when_no_geometry_is_declared(
    dataset_dir: Path,
) -> None:
    """A split must agree with its OWN metadata, caller or no caller.

    The model's declaration is an extra check, not the only one: a loader used
    without it -- a probe, a script, a test -- must still refuse arrays that
    contradict the file sitting beside them.
    """
    rows = np.zeros((4, SEQ), dtype=np.uint16)  # one column short
    np.save(dataset_dir / "val" / "all__tokens.npy", rows)
    with pytest.raises(ValueError, match="max_seq_len"):
        list(_data(dataset_dir).eval_dataloader())


def test_a_token_id_past_the_splits_own_vocabulary_is_rejected(
    dataset_dir: Path,
) -> None:
    """Same rule for ids: checked against the split's own declaration."""
    rows = np.zeros((4, SEQ + 1), dtype=np.uint16)
    rows[0, 0] = VOCAB + 3
    np.save(dataset_dir / "val" / "all__tokens.npy", rows)
    with pytest.raises(ValueError, match="vocab_size"):
        list(_data(dataset_dir).eval_dataloader())


def test_metadata_without_a_fingerprint_is_rejected(dataset_dir: Path) -> None:
    """An optional check is no check: pre-fingerprint metadata cannot be read.

    What such a split holds is exactly what cannot be established, so accepting
    it would reintroduce the unidentifiable score the fingerprint exists to
    prevent.
    """
    (dataset_dir / "val" / "dataset.json").write_text(
        json.dumps({"vocab_size": VOCAB, "max_seq_len": SEQ}),
    )
    with pytest.raises(ValueError, match="token_bytes_sha256"):
        list(_data(dataset_dir).eval_dataloader())


def test_a_negative_token_id_is_rejected(dataset_dir: Path) -> None:
    """A negative id indexes the embedding from the BACK.

    That is a silently wrong row rather than a failure, so both ends of the id
    range are checked, not only the top.
    """
    rows = np.zeros((4, SEQ + 1), dtype=np.int32)
    rows[0, 0] = -1
    np.save(dataset_dir / "val" / "all__tokens.npy", rows)
    with pytest.raises(ValueError, match="vocab_size"):
        list(_data(dataset_dir).eval_dataloader())


def test_a_negative_byte_length_is_rejected(dataset_dir: Path) -> None:
    """The metric's mask is ``lengths > 0``, so a negative length drops a token.

    It would vanish from both sums -- excluded from the measurement rather than
    rejected -- which is a quiet change to what the score covers.
    """
    lengths = np.ones(VOCAB, dtype=np.int32)
    lengths[0] = 0
    lengths[3] = -2
    np.save(dataset_dir / "val" / "all__token_bytes.npy", lengths)
    _refingerprint(dataset_dir / "val", lengths)
    with pytest.raises(ValueError, match="negative byte length"):
        list(_data(dataset_dir).eval_dataloader())


def test_a_two_dimensional_byte_table_is_rejected(dataset_dir: Path) -> None:
    """The metric indexes it as one length per id."""
    lengths = np.ones((VOCAB, 1), dtype=np.int32)
    np.save(dataset_dir / "val" / "all__token_bytes.npy", lengths)
    _refingerprint(dataset_dir / "val", lengths)
    with pytest.raises(ValueError, match="one-dimensional"):
        list(_data(dataset_dir).eval_dataloader())


def _refingerprint(directory: Path, table: np.ndarray) -> None:
    """Rewrite the metadata so the fingerprint matches a replaced table.

    Without this the fingerprint check fires first and the test proves only
    that, never reaching the property it means to pin.
    """
    metadata = json.loads((directory / "dataset.json").read_text())
    metadata["token_bytes_sha256"] = token_bytes_fingerprint(table)
    (directory / "dataset.json").write_text(json.dumps(metadata))


def test_a_malformed_token_array_names_the_file(dataset_dir: Path) -> None:
    """A one-dimensional array must not surface as a bare IndexError."""
    np.save(dataset_dir / "val" / "all__tokens.npy", np.zeros(16, dtype=np.uint16))
    with pytest.raises(ValueError, match="two-dimensional"):
        list(_data(dataset_dir).eval_dataloader())


def test_a_split_too_small_for_its_batch_is_rejected_at_construction(
    dataset_dir: Path,
) -> None:
    """An empty TRAINING stream must be named here, not where it shows up.

    Training drops a short batch, so too few rows yields nothing at all, which
    reaches the loop as a generic epoch-reset failure far from its cause.
    Evaluation is the opposite case -- it pads and scores the tail, so it has
    no empty-stream condition to report.
    """
    with pytest.raises(ValueError, match="no batches"):
        _data(dataset_dir, batch_size=64).train_dataloader()


def test_evaluation_scores_every_row_including_a_short_tail(
    dataset_dir: Path,
) -> None:
    """A dropped tail is a score covering fewer rows than the split holds.

    The fixture writes 4 rows; at a batch of 3 the old behavior scored 3 and
    called it the full pass. The tail is padded instead, and the padding is
    marked so the metric excludes it.
    """
    batches = list(_data(dataset_dir, eval_batch_size=3).eval_dataloader())
    assert [b["valid_count"] for b in batches] == [3, 1]
    assert all(b["media"].shape[0] == 3 for b in batches)
    # The padded rows carry a target the byte table cannot index.
    assert int(batches[-1]["label"][1:].min()) < 0


def test_training_still_drops_a_short_batch(dataset_dir: Path) -> None:
    """The token count per optimizer step is what the recipe is tuned against.

    A narrower final step would be the one place in a run where it moves, so
    training drops the tail that evaluation scores.
    """
    # The train split holds 6 rows, so a batch of 4 leaves a remainder of 2.
    batches = list(_data(dataset_dir, batch_size=4).train_dataloader())
    assert [b["media"].shape[0] for b in batches] == [4]


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

    ``num_eval_rows=-1`` is the one that bites hardest: as a slice bound it
    would silently mean "all but the last row" rather than an invalid cap.
    """
    with pytest.raises(ValueError, match=field):
        _data(dataset_dir, **{field: value})


def test_the_eval_batch_does_not_track_the_training_batch(
    dataset_dir: Path,
) -> None:
    """The scored row set must not move with a memory-tuning knob.

    Training's batch size follows device memory, and a short final batch is
    dropped -- so an eval batch tracking it would score a different subset of
    the split on a smaller card and report it as the same metric.
    """
    default = NanoChatData.Config().eval_batch_size
    for batch_size in (2, 8):
        config = NanoChatData.Config()
        config.base_dir = "/"
        config.working_dir = str(dataset_dir)
        config.device = "cpu"
        config.batch_size = batch_size
        assert config.make().eval_batch_size == default


def test_missing_data_names_the_preparer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        list(_data(tmp_path).train_dataloader())


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
