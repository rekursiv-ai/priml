"""Tests for the ETTh1 data loader."""

from pathlib import Path

import csv

import numpy as np
import pytest
import torch

from priml.baselines.etth1.data import Etth1Data, _ForecastBatches
from priml.testing.golden import mismatches


def fixture_config(directory: Path) -> Etth1Data.Config:
    """Build a tiny CSV with distinguishable channels, rows, and split boundaries."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "ETTh1.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", "a", "b", "c", "d"])
        for index in range(36):
            writer.writerow(["2000-01-01", index, index**2, index % 5, 7])
    cfg = Etth1Data.Config()
    cfg.working_dir = directory
    cfg.seq_len = 5
    cfg.pred_len = 3
    cfg.channels = 4
    cfg.train_rows = 17
    cfg.val_rows = 8
    cfg.test_rows = 11
    cfg.batch_size = 2
    cfg.eval_batch_size = 2
    return cfg


def test_split_boundaries_training_only_scaling_and_tail(tmp_path: Path) -> None:
    dataset = fixture_config(tmp_path).make()
    raw = np.asarray([[i, i**2, i % 5, 7] for i in range(36)], dtype=np.float64)
    mean = raw[:17].mean(axis=0)
    scale = raw[:17].std(axis=0)
    scale[-1] = 1
    expected = torch.from_numpy((raw - mean) / scale)
    assert torch.equal(dataset.train, expected[:17])
    assert torch.equal(dataset.val, expected[12:25])
    assert torch.equal(dataset.test, expected[20:36])
    assert len(dataset.train_dataloader()) == 5
    assert len(dataset.val_dataloader()) == 3
    test = list(dataset.test_dataloader())
    assert len(test) == 4
    assert torch.equal(test[0]["media"][0], expected[20:25].float())
    assert torch.equal(test[0]["label"][0], expected[25:28].float())
    assert torch.equal(test[-1]["label"][-1], expected[32:35].float())


@pytest.mark.parametrize("shuffle", [False, True])
def test_batch_order_and_rng_match_torch_dataloader(shuffle: bool) -> None:
    values = torch.arange(17 * 4, dtype=torch.float64).reshape(17, 4)
    x = torch.stack([values[i : i + 5] for i in range(10)])
    y = torch.stack([values[i + 5 : i + 8] for i in range(10)])
    source = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=2,
        shuffle=shuffle,
        drop_last=True,
    )
    torch.manual_seed(23)
    expected = list(source)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(23)
    actual = list(
        _ForecastBatches(
            values,
            seq_len=5,
            pred_len=3,
            batch_size=2,
            shuffle=shuffle,
            device="cpu",
        ),
    )
    for batch, (media, label) in zip(actual, expected, strict=True):
        assert not mismatches({"media": media.float(), "label": label.float()}, batch)
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_validation_preserves_reference_test_iterator_draw(tmp_path: Path) -> None:
    dataset = fixture_config(tmp_path).make()
    torch.manual_seed(31)
    source = torch.utils.data.DataLoader(list(range(6)), batch_size=2, shuffle=True)
    list(source)
    list(
        torch.utils.data.DataLoader(
            list(range(9)),
            batch_size=2,
            shuffle=False,
            drop_last=True,
        ),
    )
    expected = torch.get_rng_state()
    torch.manual_seed(31)
    list(dataset.val_dataloader())
    assert torch.equal(expected, torch.get_rng_state())


def test_resume_preserves_next_batch_and_rng(tmp_path: Path) -> None:
    cfg = fixture_config(tmp_path)
    dataset = cfg.make()
    torch.manual_seed(41)
    iterator = iter(dataset.train_dataloader())
    next(iterator)
    state = dataset.state_dict()
    rng = torch.get_rng_state()
    expected = list(iterator)
    expected_rng = torch.get_rng_state()
    resumed = cfg.make()
    resumed.load_state_dict(state)
    torch.set_rng_state(rng)
    actual = list(resumed.train_dataloader())
    for want, got in zip(expected, actual, strict=True):
        assert not mismatches(want, got)
    assert torch.equal(expected_rng, torch.get_rng_state())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("seq_len", 0),
        ("channels", 3),
        ("train_rows", 6),
        ("val_rows", 2),
        ("test_rows", 99),
    ],
)
def test_invalid_dataset_geometry_fails(tmp_path: Path, field: str, value: int) -> None:
    cfg = fixture_config(tmp_path)
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=r"positive|columns|split|horizon|rows"):
        cfg.make()


def test_empty_missing_and_nonfinite_csv_fail(tmp_path: Path) -> None:
    cfg = fixture_config(tmp_path)
    path = tmp_path / "ETTh1.csv"
    path.unlink()
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        cfg.make()
    path.write_text("date,a,b,c,d\n")
    with pytest.raises(ValueError, match="header"):
        cfg.make()
    cfg = fixture_config(tmp_path)
    path.write_text(path.read_text().replace("0,0,0,7", "nan,0,0,7", 1))
    with pytest.raises(ValueError, match="finite"):
        cfg.make()
