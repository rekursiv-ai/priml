"""Validate competition inputs and prompt-exclusive evaluation partitions."""

from __future__ import annotations

from functools import partialmethod
from pathlib import Path
from typing import cast

import csv
import hashlib

import numpy as np
import pytest

from priml.baselines.name_that_llm.data import CompetitionData
from priml.data.folds import grouped_folds


@pytest.mark.parametrize(
    ("counts", "expected"),
    argvalues=[
        (
            (2, 4, 5, 7),
            [list(range(11, 18)), list(range(6, 11)), list(range(6))],
        ),
        (
            (4, 4, 4, 4),
            [[4, 5, 6, 7, 12, 13, 14, 15], [0, 1, 2, 3, 8, 9, 10, 11]],
        ),
        (
            (3, 6, 3, 6, 3, 6),
            [list(range(18, 27)), list(range(9, 18)), list(range(9))],
        ),
    ],
)
def test_grouped_folds_assign_whole_groups_largest_first(
    counts: tuple[int, ...],
    expected: list[list[int]],
) -> None:
    groups = np.repeat(
        np.array([f"g{index}" for index in range(len(counts))]),
        repeats=counts,
    )
    folds = grouped_folds(groups, count=len(expected))
    assert [valid.tolist() for _, valid in folds] == expected
    for train, valid in folds:
        assert np.array_equal(
            train,
            a2=np.setdiff1d(np.arange(len(groups)), ar2=valid),
        )


@pytest.mark.parametrize("count", argvalues=[0, 1, 7])
def test_grouped_folds_reject_invalid_counts(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        grouped_folds(np.array(["a", "b", "c", "d", "e", "f"]), count=count)


def test_grouped_folds_hold_out_whole_prompts() -> None:
    groups = np.array(["a", "b", "c", "d", "e", "f"] * 3)
    folds = grouped_folds(groups, count=3)
    seen: list[int] = []
    for train, valid in folds:
        assert set(groups[train]).isdisjoint(groups[valid])
        assert len(train) == 12
        seen.extend(cast("list[int]", valid.tolist()))
    assert sorted(seen) == list(range(18))
    assert [valid.tolist() for _, valid in folds] == [
        [2, 5, 8, 11, 14, 17],
        [1, 4, 7, 10, 13, 16],
        [0, 3, 6, 9, 12, 15],
    ]


def test_competition_data_preserves_ids_and_text(tmp_path: Path) -> None:
    with (tmp_path / "train.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "prompt_id", "text", "model"])
        writer.writerows(
            [
                ["001", "p1", "Text\r\nwith punctuation!", "a"],
                ["002", "p2", "Second text", "b"],
            ],
        )
    (tmp_path / "test.csv").write_text("id,text\n003,\n004,Test text\n")
    config = CompetitionData.Config()
    config.directory = tmp_path
    config.labels = ["a", "b"]
    config.expected_rows = 2
    config.expected_prompts = 2
    config.max_prompts = 2
    data = config.make()
    assert data.ids == ["001", "002"]
    assert data.texts == ["Text\r\nwith punctuation!", "Second text"]
    assert data.test_ids == ["003", "004"]
    assert data.test_texts == ["", "Test text"]
    assert data.targets.tolist() == [0, 1]
    assert data.sha256 == {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ["train.csv", "test.csv"]
    }


def test_missing_competition_data_fails_explicitly(tmp_path: Path) -> None:
    config = CompetitionData.Config()
    config.directory = tmp_path
    with pytest.raises(FileNotFoundError, match=r"train\.csv"):
        config.make()


@pytest.mark.parametrize("default_encoding", argvalues=["ascii", "latin-1"])
def test_competition_utf8_text_ignores_the_default_file_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    default_encoding: str,
) -> None:
    (tmp_path / "train.csv").write_bytes(
        "id,prompt_id,text,model\n1,p1,Ä中🙂,a\n2,p2,Second,b\n".encode(),
    )
    (tmp_path / "test.csv").write_bytes("id,text\n3,É文🙂\n".encode())
    config = CompetitionData.Config()
    config.directory = tmp_path
    config.labels = ["a", "b"]
    config.expected_rows = config.expected_prompts = 2
    monkeypatch.setattr(
        Path,
        name="open",
        value=partialmethod(Path.open, encoding=default_encoding),
    )
    data = config.make()
    assert data.texts == ["Ä中🙂", "Second"]
    assert data.test_texts == ["É文🙂"]


def test_duplicate_ids_are_rejected_before_training(tmp_path: Path) -> None:
    (tmp_path / "train.csv").write_text(
        "id,prompt_id,text,model\n001,p1,Text,a\n001,p2,Text,b\n",
    )
    (tmp_path / "test.csv").write_text("id,text\n003,Test\n")
    config = CompetitionData.Config()
    config.directory = tmp_path
    config.labels = ["a", "b"]
    config.expected_rows = config.expected_prompts = 2
    with pytest.raises(ValueError, match="unique"):
        config.make()


@pytest.mark.parametrize(
    ("train", "test", "message"),
    argvalues=[
        (
            "id,prompt_id,text,model\n1,a,Text,a\n2,b,Text,b\n",
            "id,text\n1,Test\n",
            "disjoint",
        ),
        (
            "id,prompt_id,text,model\n1,a,,a\n2,b,Text,b\n",
            "id,text\n3,Test\n",
            "nonempty",
        ),
        (
            "id,prompt_id,text,model\n1,a,Text,a\n",
            "id,text\n3,Test\n",
            "labeled passages",
        ),
        (
            "id,prompt_id,text,model\n1,a,Text,a\n2,a,Text,b\n",
            "id,text\n3,Test\n",
            "distinct prompts",
        ),
        (
            "id,prompt_id,text,model\n1,a,Text,a\n2,b,Text,c\n",
            "id,text\n3,Test\n",
            "class vocabulary",
        ),
        ("id,text\n1,Text\n", "id,text\n3,Test\n", "columns"),
        ("id,prompt_id,text,model\n", "id,text\n3,Test\n", "Malformed"),
        ("id,prompt_id,text,model\n1,a,Text\n", "id,text\n3,Test\n", "Malformed"),
    ],
)
def test_invalid_competition_inputs_fail_before_numerical_work(
    tmp_path: Path,
    train: str,
    test: str,
    message: str,
) -> None:
    (tmp_path / "train.csv").write_text(train)
    (tmp_path / "test.csv").write_text(test)
    config = CompetitionData.Config()
    config.directory = tmp_path
    config.labels = ["a", "b"]
    config.expected_rows = config.expected_prompts = 2
    with pytest.raises(ValueError, match=message):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
