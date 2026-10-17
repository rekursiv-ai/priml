"""Tests for one timed pass: draining, saving, comparing, and reporting."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import ast
import multiprocessing
import os
import pickle

from configgle import Fig

import pytest
import torch

from priml.baselines.imagenet.data import ffcv_train_data_pipeline
from priml.baselines.imagenet.throughput import throughput, timed_pass
from priml.data.pipeline.dataset import DataPipeline


if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import ModuleType


IMAGE: Final = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)
LABEL: Final = torch.tensor([3, 5])
FIELDS: Final = ["image", "label"]


class Fixed:
    """Yield two batches of ``IMAGE`` shifted by ``offset``, or fail to build."""

    class Config(Fig["Fixed"]):
        offset: int = 0
        raises: bool = False

    def __init__(self, config: Config) -> None:
        if config.raises:
            raise RuntimeError("The pipeline could not be built.")
        self.config = config

    def __iter__(self) -> Iterator[dict[str, object]]:
        for index in range(2):
            yield {
                "image": IMAGE + self.config.offset + index,
                "label": LABEL,
                "unused": "not compared",
            }


def _record_setpgid(calls: list[tuple[int, int]], pid: int, pgrp: int) -> None:
    calls.append((pid, pgrp))


@pytest.fixture(autouse=True)
def setpgid_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Keep ``drain`` from moving pytest into a process group of its own."""
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "setpgid", partial(_record_setpgid, calls))
    return calls


def _drain(
    config: Fixed.Config,
    *,
    save_dir: Path | None = None,
    compare_dir: Path | None = None,
) -> timed_pass.PassResult | str:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    try:
        timed_pass.drain(
            pickle.dumps(config),
            seed=0,
            fields=FIELDS,
            save_dir=None if save_dir is None else os.fspath(save_dir),
            compare_dir=None if compare_dir is None else os.fspath(compare_dir),
            connection=sender,
        )
    finally:
        sender.close()
    return cast("timed_pass.PassResult | str", receiver.recv())


def test_drain_saves_every_batch_of_the_reference_pass(
    tmp_path: Path,
    setpgid_calls: list[tuple[int, int]],
) -> None:
    result = _drain(Fixed.Config(), save_dir=tmp_path)

    assert isinstance(result, timed_pass.PassResult)
    assert setpgid_calls == [(0, 0)]
    assert (result.num_batches, result.num_images) == (2, 4)
    assert (result.num_reference_batches, result.diffs) == (0, {})
    assert 0 < result.first_batch_sec <= result.elapsed_sec
    assert result.cpu_outside_sec > 0
    saved = cast(
        "dict[str, torch.Tensor]",
        torch.load(tmp_path / "1.pt", weights_only=True),
    )
    assert sorted(saved) == FIELDS
    assert torch.equal(saved["image"], IMAGE + 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["0.pt", "1.pt"]


def test_drain_compares_every_batch_against_the_reference(tmp_path: Path) -> None:
    _ = _drain(Fixed.Config(), save_dir=tmp_path)
    shifted = Fixed.Config()
    shifted.offset = 1

    result = _drain(shifted, compare_dir=tmp_path)

    assert isinstance(result, timed_pass.PassResult)
    assert result.num_reference_batches == 2
    moved = timed_pass.Diff(
        num_differing=24,
        numel=24,
        max_abs_diff=1.0,
        first_index=(0, 0, 0),
    )
    assert result.diffs == {(0, "image"): moved, (1, "image"): moved}


def test_drain_sends_the_traceback_before_reraising() -> None:
    receiver, sender = multiprocessing.Pipe(duplex=False)
    broken = Fixed.Config()
    broken.raises = True

    with pytest.raises(RuntimeError, match="could not be built"):
        timed_pass.drain(
            pickle.dumps(broken),
            seed=0,
            fields=FIELDS,
            save_dir=None,
            compare_dir=None,
            connection=sender,
        )

    sent = cast("str", receiver.recv())
    assert sent.startswith("Traceback (most recent call last):")
    assert sent.rstrip().endswith("RuntimeError: The pipeline could not be built.")


def test_compare_names_the_count_the_largest_move_and_where_it_starts() -> None:
    moved = IMAGE.clone()
    moved[1, 2, :2] += torch.tensor([4, 1], dtype=torch.uint8)

    assert timed_pass.compare(IMAGE, IMAGE.clone()) is None
    assert timed_pass.compare(IMAGE, moved) == timed_pass.Diff(
        num_differing=2,
        numel=IMAGE.numel(),
        max_abs_diff=4.0,
        first_index=(1, 2, 0),
    )
    assert timed_pass.compare(IMAGE, IMAGE.reshape(3, 2, 4)) == (
        "torch.uint8[3, 2, 4] vs torch.uint8[2, 3, 4]"
    )
    assert timed_pass.compare(IMAGE, IMAGE.float()) == (
        "torch.float32[2, 3, 4] vs torch.uint8[2, 3, 4]"
    )


def test_compare_handles_dtypes_numpy_lacks() -> None:
    want = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16)
    got = want.clone()
    got[1, 0] = -4.0

    assert timed_pass.compare(want, want.clone()) is None
    assert timed_pass.compare(want, got) == timed_pass.Diff(
        num_differing=1,
        numel=6,
        max_abs_diff=8.0,
        first_index=(1, 0),
    )


def test_compare_matches_bits_so_nan_equals_itself_and_zeros_differ() -> None:
    want = torch.tensor([float("nan"), 0.0, 1.0])

    assert timed_pass.compare(want, want.clone()) is None
    assert timed_pass.compare(want, torch.tensor([float("nan"), -0.0, 1.0])) == (
        timed_pass.Diff(num_differing=1, numel=3, max_abs_diff=0.0, first_index=(1,))
    )


def test_compare_calls_a_nan_on_one_side_an_unbounded_move() -> None:
    want = torch.tensor([float("nan"), 0.0, 1.0])

    diff = timed_pass.compare(want, torch.tensor([0.0, 0.0, 3.0]))
    infinite = timed_pass.compare(want, torch.tensor([float("nan"), 0.0, float("inf")]))

    assert isinstance(diff, timed_pass.Diff)
    assert (diff.num_differing, diff.max_abs_diff) == (2, float("inf"))
    assert isinstance(infinite, timed_pass.Diff)
    assert infinite.max_abs_diff == float("inf")


def test_compare_moves_integers_exactly() -> None:
    """A float64 detour would round ``2**60 + 1`` to ``2**60``."""
    big = torch.tensor([2**60, 0])
    extremes = torch.tensor([-(2**63), 2**63 - 1])

    nudged = timed_pass.compare(big, big + torch.tensor([1, 0]))
    spanned = timed_pass.compare(extremes, extremes.flip(0))
    narrow = timed_pass.compare(
        LABEL.to(torch.uint8),
        torch.tensor([5, 3]).to(torch.uint8),
    )

    assert isinstance(nudged, timed_pass.Diff)
    assert (nudged.num_differing, nudged.max_abs_diff) == (1, 1.0)
    assert isinstance(spanned, timed_pass.Diff)
    assert spanned.max_abs_diff == float(2**64 - 1)
    assert isinstance(narrow, timed_pass.Diff)
    assert narrow.max_abs_diff == 2.0


def test_a_compared_field_must_be_a_tensor() -> None:
    assert timed_pass._kept({"image": IMAGE, "label": LABEL}, FIELDS) == {
        "image": IMAGE,
        "label": LABEL,
    }
    with pytest.raises(
        TypeError,
        match=r"^Batch field 'label' is list, not a Tensor\.$",
    ):
        _ = timed_pass._kept({"image": IMAGE, "label": [3, 5]}, FIELDS)


def _runtime_imports(module: ModuleType) -> list[str]:
    """Return every module ``module`` imports outside ``if TYPE_CHECKING:``."""
    assert module.__file__ is not None
    tree = ast.parse(Path(module.__file__).read_text())
    typing_only = {
        id(node)
        for block in ast.walk(tree)
        if isinstance(block, ast.If) and ast.unparse(block.test) == "TYPE_CHECKING"
        for statement in block.body
        for node in ast.walk(statement)
    }
    imported: list[str] = []
    for node in ast.walk(tree):
        if id(node) in typing_only:
            continue
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    return imported


def test_the_pass_module_imports_no_pipeline_code() -> None:
    """Pipeline imports before the window would be free for exp000 alone."""
    pipeline_packages = (
        DataPipeline.__module__.removesuffix(".pipeline.dataset"),
        ffcv_train_data_pipeline.__module__.removesuffix(".data"),
    )
    imported = _runtime_imports(timed_pass)

    assert "priml.math.seed" in imported
    assert [m for m in imported if m.startswith(pipeline_packages)] == []


def test_runtime_imports_skips_typing_only_imports() -> None:
    imported = _runtime_imports(throughput)

    assert "priml.baselines.imagenet.data" in imported
    assert "priml.baselines.imagenet.throughput.timed_pass" not in imported


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
