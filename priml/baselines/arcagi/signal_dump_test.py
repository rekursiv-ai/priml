"""Tests for the signal-dump tracker and writer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import logging
import zipfile

import numpy as np
import pytest
import torch

from priml.baselines.arcagi import signal_dump
from priml.baselines.arcagi.augmentation import ArcSpec
from priml.baselines.arcagi.signal_dump import (
    SignalDumpPayload,
    SignalDumpTracker,
    _any_rank,
    _gather_grids,
    _gather_list,
    write_signal_dump,
)


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def test_single_process_gathers_return_their_input() -> None:
    assert _any_rank(True)
    assert _gather_list([1, 2]) == [1, 2]
    assert _gather_grids({"x": np.zeros((2, 3), dtype=np.uint8)})["x"].shape == (2, 3)


def test_signal_dumps_rotate_but_keep_archived_steps(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The newest ``keep_last_n`` stay, and so does every ``keep_every`` multiple."""
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi.signal_dump")
    payload = SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=(1,))
    tracker = SignalDumpTracker.Config(
        working_dir=str(tmp_path / "signals_{global_step}.npz"),
        keep_last_n=2,
        keep_every=20,
    ).make()
    for step in (10, 20, 30, 40, 50):
        tracker.log_metrics({"extras": {"signal_dump": payload}}, step, prefix="eval/")
    kept = sorted(path.name for path in tmp_path.iterdir())
    assert kept == ["signals_20.npz", "signals_40.npz", "signals_50.npz"]
    assert len(caplog.messages) == 7
    assert caplog.messages[3] == (
        f"Deleted signal dump {tmp_path / 'signals_10.npz'} (keep_last_n rotation)."
    )
    assert caplog.messages[6] == (
        f"Deleted signal dump {tmp_path / 'signals_30.npz'} (keep_last_n rotation)."
    )


@pytest.mark.parametrize(
    "template",
    ["{global_step}_{left", "{global_step}_right}"],
)
def test_signal_dump_skips_paths_with_partial_placeholders(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    template: str,
) -> None:
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi.signal_dump")
    write_signal_dump(
        payload=SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=(1,)),
        dump_signals_path=str(tmp_path / template),
        global_step=7,
        spec=ArcSpec(max_grid=3),
    )
    assert list(tmp_path.iterdir()) == []
    assert caplog.messages == [
        f"[eval] signal dump skipped: unresolved path {str(tmp_path / template)!r}",
    ]


def test_signal_dump_archive_preserves_rows_grids_and_steps(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    first_grid = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    second_grid = np.array([[7, 8, 9]], dtype=np.uint8)
    first_hash = "1" * 64
    second_hash = "2" * 64
    payload = SignalDumpPayload(
        rows=[
            ("task-a", "a" * 64, first_hash, 0.25, -1.5, 0.75, 2, 3),
            ("task-b", "b" * 64, second_hash, 0.5, -2.5, 0.25, 1, 3),
        ],
        grids={first_hash: first_grid, second_hash: second_grid},
        steps=[(4, 300, (0.1, 0.2), (1, 0)), (5, 301, (0.3, 0.4), (0, 1))],
        pass_ks=(1, 3),
    )
    path = tmp_path / "nested" / "deeper" / "signals.npz"
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi.signal_dump")
    write_signal_dump(
        payload=payload,
        dump_signals_path=path,
        global_step=7,
        spec=ArcSpec(max_grid=3),
    )

    assert caplog.messages == [
        (
            "[eval] wrote signal dump (2 rows, 2 groups, 2 preds, "
            f"{path.stat().st_size} bytes) to {path}"
        ),
    ]
    archive = cast(np.lib.npyio.NpzFile, np.load(path))
    with archive:
        assert archive.files == [
            "group_id",
            "pred_id",
            "q_halt",
            "logprob",
            "stability",
            "n_rows",
            "n_cols",
            "group_table",
            "pred_table",
            "pred_grids",
            "pred_n_rows",
            "pred_n_cols",
            "pass_ks",
            "converge_step",
            "n_changes",
            "q_halt_steps",
            "correct_step",
        ]
        assert archive["group_id"].dtype == np.int32
        assert archive["group_id"].tolist() == [0, 1]
        assert archive["pred_id"].dtype == np.int32
        assert archive["pred_id"].tolist() == [0, 1]
        assert archive["q_halt"].dtype == np.float32
        assert archive["q_halt"].tolist() == [0.25, 0.5]
        assert archive["logprob"].dtype == np.float32
        assert archive["logprob"].tolist() == [-1.5, -2.5]
        assert archive["stability"].dtype == np.float32
        assert archive["stability"].tolist() == [0.75, 0.25]
        assert archive["n_rows"].dtype == np.uint8
        assert archive["n_rows"].tolist() == [2, 1]
        assert archive["n_cols"].dtype == np.uint8
        assert archive["n_cols"].tolist() == [3, 3]
        assert archive["group_table"].tolist() == [
            "task-a\t" + "a" * 64,
            "task-b\t" + "b" * 64,
        ]
        assert archive["pred_table"].tolist() == [first_hash, second_hash]
        assert archive["pred_grids"].dtype == np.uint8
        assert archive["pred_grids"].shape == (2, 3, 3)
        assert np.array_equal(archive["pred_grids"][0, :2, :3], first_grid)
        assert np.array_equal(archive["pred_grids"][1, :1, :3], second_grid)
        assert archive["pred_n_rows"].dtype == np.uint8
        assert archive["pred_n_rows"].tolist() == [2, 1]
        assert archive["pred_n_cols"].dtype == np.uint8
        assert archive["pred_n_cols"].tolist() == [3, 3]
        assert archive["pass_ks"].dtype == np.int64
        assert archive["pass_ks"].tolist() == [1, 3]
        assert archive["converge_step"].dtype == np.uint8
        assert archive["converge_step"].tolist() == [4, 5]
        assert archive["n_changes"].dtype == np.uint16
        assert archive["n_changes"].tolist() == [300, 301]
        assert (
            archive["q_halt_steps"].tolist()
            == np.asarray(
                [[0.1, 0.2], [0.3, 0.4]],
                dtype=np.float32,
            ).tolist()
        )
        assert archive["correct_step"].dtype == np.uint8
        assert archive["correct_step"].tolist() == [[1, 0], [0, 1]]


def test_signal_dump_continues_past_missing_prediction_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_hash = "a" * 64
    present_hash = "b" * 64
    grid = np.array([[4, 5]], dtype=np.uint8)
    path = tmp_path / "signals.npz"
    arrays_written: dict[str, NDArray[np.generic]] = {}

    def save_arrays(
        output: str | Path,
        **arrays: NDArray[np.generic],
    ) -> None:
        arrays_written.update(arrays)
        assert output == path
        path.write_bytes(b"npz")

    monkeypatch.setattr(np, "savez_compressed", save_arrays)
    write_signal_dump(
        payload=SignalDumpPayload(
            rows=[
                ("task", "c" * 64, missing_hash, 0.1, 0.2, 0.3, 1, 1),
                ("task", "d" * 64, present_hash, 0.4, 0.5, 0.6, 1, 2),
            ],
            grids={present_hash: grid},
            steps=[],
            pass_ks=(1,),
        ),
        dump_signals_path=path,
        global_step=1,
        spec=ArcSpec(max_grid=3),
    )

    assert arrays_written["pred_grids"].shape == (2, 3, 3)
    assert arrays_written["pred_grids"][1, 0, :2].tolist() == [4, 5]
    assert arrays_written["pred_n_rows"].tolist() == [0, 1]


def test_signal_dump_warns_before_overwriting(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=(1,))
    path = tmp_path / "signals.npz"
    write_signal_dump(
        payload=payload,
        dump_signals_path=path,
        global_step=1,
        spec=ArcSpec(max_grid=3),
    )
    caplog.set_level(logging.WARNING, logger="priml.baselines.arcagi.signal_dump")

    write_signal_dump(
        payload=payload,
        dump_signals_path=path,
        global_step=1,
        spec=ArcSpec(max_grid=3),
    )

    assert caplog.messages == [f"[eval] overwriting existing signal dump at {path}"]


def test_signal_dump_reuses_group_and_prediction_indices(tmp_path: Path) -> None:
    prediction_hash = "c" * 64
    grid = np.array([[2, 3]], dtype=np.uint8)
    payload = SignalDumpPayload(
        rows=[
            ("task-a", "a" * 64, prediction_hash, 0.1, 0.2, 0.3, 1, 2),
            ("task-a", "a" * 64, prediction_hash, 0.4, 0.5, 0.6, 1, 2),
            ("task-b", "b" * 64, prediction_hash, 0.7, 0.8, 0.9, 1, 2),
        ],
        grids={prediction_hash: grid},
        steps=[],
        pass_ks=(1,),
    )
    path = tmp_path / "signals.npz"

    write_signal_dump(
        payload=payload,
        dump_signals_path=path,
        global_step=1,
        spec=ArcSpec(max_grid=3),
    )

    archive = cast(np.lib.npyio.NpzFile, np.load(path))
    with archive:
        assert archive["group_id"].tolist() == [0, 0, 1]
        assert archive["pred_id"].tolist() == [0, 0, 0]
        assert archive["group_table"].tolist() == [
            "task-a\t" + "a" * 64,
            "task-b\t" + "b" * 64,
        ]
        assert archive["pred_table"].tolist() == [prediction_hash]


def test_signal_dumps_are_all_kept_by_default(tmp_path: Path) -> None:
    payload = SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=(1,))
    tracker = SignalDumpTracker.Config(
        working_dir=str(tmp_path / "signals_{global_step}.npz"),
    ).make()
    for step in (1, 2, 3):
        tracker.log_metrics({"extras": {"signal_dump": payload}}, step, prefix="eval/")
    assert len(list(tmp_path.iterdir())) == 3


@pytest.mark.parametrize(
    ("keep_last_n", "keep_every", "message"),
    [
        (0, 0, "keep_last_n must be -1 (keep all) or positive; got 0."),
        (-2, 0, "keep_last_n must be -1 (keep all) or positive; got -2."),
        (-1, -1, "keep_every must be >= 0; got -1."),
    ],
)
def test_signal_dump_retention_rejects_nonsense(
    keep_last_n: int,
    keep_every: int,
    message: str,
) -> None:
    config = SignalDumpTracker.Config(keep_last_n=keep_last_n, keep_every=keep_every)
    with pytest.raises(ValueError, match="keep_") as error:
        config.make()
    assert str(error.value) == message


def test_distributed_object_gathers_include_all_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    gathered_flags: list[bool] = []

    def gather_flags(output: list[bool | None], value: bool) -> None:
        gathered_flags.append(value)
        output[0] = value
        output[1] = True

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_flags)
    assert _any_rank(False)
    assert gathered_flags == [False]

    def gather_lists(output: list[list[int] | None], value: list[int]) -> None:
        output[0] = value
        output[1] = [3, 4]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_lists)
    assert _gather_list([1, 2]) == [1, 2, 3, 4]

    first = np.array([[1, 2]], dtype=np.uint8)
    second = np.array([[3, 4]], dtype=np.uint8)

    def gather_grids(
        output: list[dict[str, NDArray[np.uint8]] | None],
        value: dict[str, NDArray[np.uint8]],
    ) -> None:
        output[0] = value
        output[1] = {"remote": second}

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather_grids)
    assert _gather_grids({"local": first}) == {"local": first, "remote": second}


def test_signal_tracker_writes_on_eval_prefix(tmp_path: Path) -> None:
    path = tmp_path / "signals.npz"
    tracker = SignalDumpTracker.Config(working_dir=path).make()
    payload = SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=(1, 3))

    tracker.log_metrics({"extras": {"signal_dump": payload}}, 1, prefix="eval/")

    assert path.exists()
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == [
            "group_id.npy",
            "pred_id.npy",
            "q_halt.npy",
            "logprob.npy",
            "stability.npy",
            "n_rows.npy",
            "n_cols.npy",
            "group_table.npy",
            "pred_table.npy",
            "pred_grids.npy",
            "pred_n_rows.npy",
            "pred_n_cols.npy",
            "pass_ks.npy",
        ]


def test_signal_tracker_branches(tmp_path: Path) -> None:
    tracker = SignalDumpTracker.Config(working_dir=tmp_path / "dump.npz").make()
    tracker.log_metrics({}, 1, prefix="train/")
    # Only the eval stream writes; an unprefixed payload is not an eval result.
    tracker.log_metrics({"extras": 3}, 1, prefix="")
    tracker.log_metrics({}, 1, prefix="eval/")
    with pytest.raises(TypeError) as mapping_error:
        tracker.log_metrics({"extras": 3}, 1, prefix="eval/")
    assert str(mapping_error.value) == (
        "SignalDumpTracker expected metrics['extras'] to be a mapping, got int."
    )
    with pytest.raises(TypeError) as payload_error:
        tracker.log_metrics({"extras": {}}, 1, prefix="eval/")
    assert str(payload_error.value) == (
        "SignalDumpTracker expected extras['signal_dump'] to be "
        "SignalDumpPayload, got NoneType."
    )
    with pytest.raises(TypeError) as wrong_payload_error:
        tracker.log_metrics({"extras": {"signal_dump": 3}}, 1, prefix="eval/")
    assert str(wrong_payload_error.value) == (
        "SignalDumpTracker expected extras['signal_dump'] to be "
        "SignalDumpPayload, got int."
    )
    payload = SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=())
    tracker.log_metrics({"extras": {"signal_dump": payload}}, 2, prefix="eval/")
    tracker.log_images("x", [], 1)
    tracker.log_notes("x")
    tracker.close()


def test_signal_dump_tracker_omits_empty_rank_and_writes_for_remote_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = SignalDumpTracker.Config(
        working_dir=str(tmp_path / "signals_{global_step}.npz"),
    ).make()
    gathered_payload_flags: list[bool] = []

    def capture_payload_flag(has_payload: bool) -> bool:
        gathered_payload_flags.append(has_payload)
        return has_payload

    monkeypatch.setattr(signal_dump, "_any_rank", capture_payload_flag)
    tracker.log_metrics({}, 1, prefix="eval/")
    assert gathered_payload_flags == [False]
    assert not (tmp_path / "signals_1.npz").exists()

    def always_true(has_payload: bool) -> bool:
        del has_payload
        return True

    monkeypatch.setattr(signal_dump, "_any_rank", always_true)
    written: list[SignalDumpPayload] = []

    def capture_dump(
        *,
        payload: SignalDumpPayload,
        dump_signals_path: str | Path,
        global_step: int,
        spec: ArcSpec,
    ) -> None:
        del dump_signals_path, global_step, spec
        written.append(payload)

    monkeypatch.setattr(signal_dump, "write_signal_dump", capture_dump)
    tracker.log_metrics({}, 2, prefix="eval/")
    assert written == [SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=())]


def test_keep_one_signal_dump_and_archive_every_step(
    tmp_path: Path,
) -> None:
    payload = SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=(1,))
    tracker = SignalDumpTracker.Config(
        working_dir=str(tmp_path / "signals_{global_step}.npz"),
        keep_last_n=1,
        keep_every=1,
    ).make()
    for step in (1, 2, 3):
        tracker.log_metrics({"extras": {"signal_dump": payload}}, step, prefix="eval/")
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "signals_1.npz",
        "signals_2.npz",
        "signals_3.npz",
    ]


def test_zero_archive_interval_prunes_to_latest_signal_dump(tmp_path: Path) -> None:
    payload = SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=(1,))
    tracker = SignalDumpTracker.Config(
        working_dir=str(tmp_path / "signals_{global_step}.npz"),
        keep_last_n=1,
    ).make()
    for step in (1, 2, 3):
        tracker.log_metrics({"extras": {"signal_dump": payload}}, step, prefix="eval/")
    assert [path.name for path in tmp_path.iterdir()] == ["signals_3.npz"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
