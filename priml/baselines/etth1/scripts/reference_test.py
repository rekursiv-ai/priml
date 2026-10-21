"""Tests for source verification against a faked pinned DLinear checkout.

The fake checkout builds its model, optimizer, and loader from PRIML's own
recipe under upstream names, so parity holds unless a test perturbs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import csv
import hashlib
import subprocess
import sys

from treekle import json_loads

import pytest
import torch

from priml.baselines.etth1 import experiments
from priml.baselines.etth1.experiments import exp000
from priml.baselines.etth1.scripts import data, reference


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from priml.baselines.etth1.train_step import Etth1TrainLoop


_UPSTREAM_PACKAGES: Final = ("exp", "data_provider", "utils", "models")

# 24 windows at 336/96 geometry provide exactly three batches of eight.
_TEST_TRAIN_ROWS: Final = 455
_TEST_VAL_ROWS: Final = 96
_TEST_ROWS: Final = 96
_TEST_CSV_ROWS: Final = _TEST_TRAIN_ROWS + _TEST_VAL_ROWS + _TEST_ROWS

_EXP_MAIN: Final = """
import torch
from torch import nn

from priml.baselines.etth1.experiments import exp000


class _Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decomposition = model.decomposition
        self.Linear_Seasonal = model.seasonal
        self.Linear_Trend = model.trend
        self.Linear_Decoder = model.decoder

    def forward(self, x):
        seasonal, trend = self.decomposition(x)
        output = self.Linear_Seasonal(seasonal.permute(0, 2, 1)) + self.Linear_Trend(
            trend.permute(0, 2, 1),
        )
        return output.permute(0, 2, 1)


class Exp_Main:
    def __init__(self, args):
        self.args = args
        cfg = exp000().step
        cfg.seed = 2021
        cfg.model.seq_len = args.seq_len
        cfg.model.pred_len = args.pred_len
        cfg.model.channels = args.enc_in
        self.model = _Model(cfg.make().model)

    def _select_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()
"""

_DATA_FACTORY: Final = f"""
from priml.baselines.etth1.data import Etth1Data


def data_provider(args, flag):
    cfg = Etth1Data.Config()
    cfg.working_dir = args.root_path
    cfg.seq_len = args.seq_len
    cfg.pred_len = args.pred_len
    cfg.channels = args.enc_in
    cfg.batch_size = args.batch_size
    cfg.train_rows = {_TEST_TRAIN_ROWS}
    cfg.val_rows = {_TEST_VAL_ROWS}
    cfg.test_rows = {_TEST_ROWS}
    loader = cfg.make().train_dataloader()
    return None, ((b["media"], b["label"], b["media"], b["label"]) for b in loader)
"""

_TOOLS: Final = """
# np.Inf
def adjust_learning_rate(optimizer, epoch, args):
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate * {factor} ** ((epoch - 1) // 1)
"""


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("etth1")
    with (directory / "ETTh1.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", *"abcdefg"])
        for row in range(_TEST_CSV_ROWS):
            writer.writerow(
                ["2016-07-01", *((row * (c + 2)) % (31 + c) for c in range(7))],
            )
    return directory


@pytest.fixture
def compact_data_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject matching test splits where the verifier has no split-size hook."""
    # Both factories must change or the source and candidate read different row counts.
    monkeypatch.setattr(experiments, "exp000", _exp000_with_compact_data)
    monkeypatch.setattr(reference, "exp000", _exp000_with_compact_data)


@pytest.fixture
def pinned(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    digest = hashlib.sha256((dataset / "ETTh1.csv").read_bytes()).hexdigest()
    # `reference` binds its own copy at import, so the defining module and the
    # verifier are both pinned to the fixture CSV.
    monkeypatch.setattr(data, "DATASET_SHA256", digest)
    monkeypatch.setattr(reference, "DATASET_SHA256", digest)
    return dataset


@pytest.fixture
def isolated_imports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in _UPSTREAM_PACKAGES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)
    for name in list(sys.modules):
        if name.partition(".")[0] in _UPSTREAM_PACKAGES:
            del sys.modules[name]


def test_reference_args_change_only_geometry_for_the_probe(tmp_path: Path) -> None:
    tiny = vars(reference.reference_args(tmp_path, tiny=True))
    full = vars(reference.reference_args(tmp_path, tiny=False))
    changed = {key for key in full if tiny[key] != full[key]}
    assert changed == {"seq_len", "pred_len", "enc_in"}
    assert (full["seq_len"], full["pred_len"], full["enc_in"]) == (336, 96, 7)
    assert (tiny["seq_len"], tiny["pred_len"], tiny["enc_in"]) == (5, 3, 4)
    assert full["root_path"] == str(tmp_path)


@pytest.mark.usefixtures("isolated_imports", "compact_data_config")
def test_verify_reports_exact_parity(
    tmp_path: Path,
    pinned: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _checkout(tmp_path, monkeypatch=monkeypatch)
    report = reference.verify(checkout, directory=pinned, mint=False)
    assert report["source_commit"] == reference.SOURCE_COMMIT
    assert report["dataset_sha256"] == data.DATASET_SHA256
    assert report["source_compatibility_edits"] == []
    assert report["source_sha256"] == {
        name: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
        for name in reference._SOURCE_FILES
    }
    assert report["canonical_three_step_mismatches"] == 0


@pytest.mark.usefixtures("isolated_imports", "compact_data_config")
def test_capture_returns_portable_source_records(
    tmp_path: Path,
    pinned: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _checkout(tmp_path, monkeypatch=monkeypatch, numpy2=True)
    report, model, training = reference.capture_reference(checkout, directory=pinned)
    assert report["source_compatibility_edits"] == [
        "utils/tools.py: np.Inf -> np.inf (NumPy 2 spelling)",
    ]
    assert "initial/seasonal.weight" in model
    assert not any("Linear_" in key for key in model)
    assert model["output"].shape == (2, 3, 4)
    assert {"step1/loss", "step2/loss", "step3/loss"} <= training.keys()


@pytest.mark.usefixtures("isolated_imports")
def test_diverging_source_schedule_fails_parity(
    tmp_path: Path,
    pinned: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _checkout(tmp_path, monkeypatch=monkeypatch, factor=0.25)
    with pytest.raises(AssertionError, match=r"step3/.*differ"):
        reference.capture_reference(checkout, directory=pinned)


def test_unpinned_dataset_is_rejected(tmp_path: Path, dataset: Path) -> None:
    with pytest.raises(ValueError, match="pinned ETTh1 CSV"):
        reference.capture_reference(tmp_path, directory=dataset)


def test_wrong_revision_is_rejected(
    tmp_path: Path,
    pinned: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _checkout(tmp_path, monkeypatch=monkeypatch, revision="f" * 40)
    with pytest.raises(ValueError, match=f"found {'f' * 40}"):
        reference.capture_reference(checkout, directory=pinned)


def test_modified_source_is_rejected(
    tmp_path: Path,
    pinned: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = _checkout(tmp_path, monkeypatch=monkeypatch)
    with (checkout / "exp" / "exp_main.py").open("a") as stream:
        stream.write("# local edit\n")
    with pytest.raises(ValueError, match=r"modified: exp/exp_main\.py"):
        reference.capture_reference(checkout, directory=pinned)


def test_verify_refuses_to_mint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="runs pytest"):
        reference.verify(tmp_path, directory=tmp_path, mint=True)


def test_main_prints_the_verification_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[Path, Path]] = []

    def capture(
        checkout: Path,
        directory: Path,
    ) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        calls.append((checkout, directory))
        return {"source_commit": "abc"}, {}, {}

    monkeypatch.setattr(reference, "capture_reference", capture)
    _set_argv(monkeypatch, "--reference", "src", "--directory", str(tmp_path))
    assert reference.main() == 0
    assert calls == [(Path("src"), tmp_path)]
    assert json_loads(capsys.readouterr().out, dict[str, object]) == {
        "source_commit": "abc",
    }


def test_main_mints_through_a_fresh_pytest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Sequence[str]] = []

    def call(command: Sequence[str]) -> int:
        calls.append(command)
        return 3

    monkeypatch.setattr(subprocess, "call", call)
    monkeypatch.chdir(tmp_path)
    _set_argv(monkeypatch, "--reference", "src", "--directory", "data", "--mint")
    assert reference.main() == 3
    [command] = calls
    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert Path(command[3]) == Path(reference.__file__).with_name(
        "mint_reference_test.py",
    )
    assert command[-4:] == [
        "--etth1-reference",
        str(tmp_path / "src"),
        "--etth1-directory",
        str(tmp_path / "data"),
    ]


def _checkout(
    root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    factor: float = 0.5,
    numpy2: bool = False,
    revision: str = reference.SOURCE_COMMIT,
) -> Path:
    """Write a fake upstream tree and answer its git queries."""
    checkout = root / "DLinear"
    committed = {name: f"# {name}\n" for name in reference._SOURCE_FILES}
    committed["exp/exp_main.py"] = _EXP_MAIN
    committed["data_provider/data_factory.py"] = _DATA_FACTORY
    committed["utils/tools.py"] = _TOOLS.format(factor=factor)
    for package in _UPSTREAM_PACKAGES:
        (checkout / package).mkdir(parents=True)
        (checkout / package / "__init__.py").write_text("")
    for name, text in committed.items():
        payload = text.replace("np.Inf", "np.inf") if numpy2 else text
        (checkout / name).write_text(payload)

    def check_output(command: Sequence[str], *, text: bool = False) -> str | bytes:
        assert command[:3] == ["git", "-C", str(checkout)]
        if command[3] == "rev-parse":
            assert text
            return f"{revision}\n"
        commit, _, name = command[4].partition(":")
        assert (command[3], commit) == ("show", reference.SOURCE_COMMIT)
        return committed[name].encode()

    monkeypatch.setattr(subprocess, "check_output", check_output)
    return checkout


def _set_argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["verify_reference", *args])


def _exp000_with_compact_data() -> Etth1TrainLoop.Config:
    cfg = exp000()
    cfg.dataset.train_rows = _TEST_TRAIN_ROWS
    cfg.dataset.val_rows = _TEST_VAL_ROWS
    cfg.dataset.test_rows = _TEST_ROWS
    return cfg


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
