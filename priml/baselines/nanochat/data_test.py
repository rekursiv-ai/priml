"""Tests for nanochat data loading."""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import pytest
import torch

from priml.baselines.nanochat.data import NanoChatData


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
            json.dumps({"vocab_size": VOCAB, "max_seq_len": SEQ}),
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


def test_a_byte_table_disagreeing_with_the_metadata_is_rejected(
    dataset_dir: Path,
) -> None:
    """A stale table would silently misprice every token in the score."""
    np.save(
        dataset_dir / "val" / "all__token_bytes.npy",
        np.ones(VOCAB + 1, dtype=np.int32),
    )
    with pytest.raises(ValueError, match="byte table"):
        list(_data(dataset_dir).eval_dataloader())


def test_missing_data_names_the_preparer(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        list(_data(tmp_path).train_dataloader())


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
