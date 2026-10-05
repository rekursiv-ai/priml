"""Tests for the pass@K voting metric."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast, override

import hashlib
import json
import logging
import zipfile

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1 import metric
from priml.baselines.arcagi1.augmentation import (
    ArcSpec,
    ColorDihedral,
    SpatialAugmentation,
    grid_hash,
)
from priml.baselines.arcagi1.metric import (
    CanonicalPassK,
    PassK,
    PerOutputPass,
    SignalDumpPayload,
    SignalDumpTracker,
    StrictPass,
    TaskScore,
    _any_rank,
    _digest,
    _floats,
    _gather_grids,
    _gather_list,
    _hash_bytes,
    _json_grid,
    _model_output_header_width,
    _read_u32,
    _shape,
    _uint8_rows,
    _votes_times_max_q,
    _votes_times_mean_q,
    decode_preds,
    encode_preds,
    write_signal_dump,
)
from priml.lib.custom_json import parse


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


def _packed(predictions: Tensor, halt: Tensor) -> Tensor:
    """Pack a halt logit ahead of the predicted tokens, as the step emits."""
    return torch.cat([halt.reshape(-1, 1), predictions.float()], dim=-1)


def _metric(**overrides: object) -> PassK:
    config = PassK.Config(pass_ks=(1, 2))
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_a_puzzle_solved_by_every_view_passes_at_one() -> None:
    metric = _metric()
    labels = torch.full((3, 9), 3, dtype=torch.int64)
    metric.update(
        _packed(labels.clone(), torch.zeros(3)),
        label=labels,
        puzzle_identifiers=torch.zeros(3, dtype=torch.int64),
    )
    assert metric.compute() == {"pass@1": 1.0, "pass@2": 1.0}


def test_the_majority_answer_wins() -> None:
    """Agreement across views is the signal, so two votes beat one."""
    metric = _metric()
    labels = torch.full((3, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[:2] = 7  # Two views agree on a WRONG answer.
    metric.update(
        _packed(predictions, torch.zeros(3)),
        label=labels,
        puzzle_identifiers=torch.zeros(3, dtype=torch.int64),
    )
    # The truth was outvoted, but it is still the second-ranked answer.
    assert metric.compute() == {"pass@1": 0.0, "pass@2": 1.0}


def test_confidence_only_breaks_a_tie() -> None:
    """One vote each: the more confident answer ranks first."""
    metric = _metric()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[0] = 7  # A wrong answer, but stated with low confidence.
    metric.update(
        _packed(predictions, torch.tensor([-5.0, 5.0])),
        label=labels,
        puzzle_identifiers=torch.zeros(2, dtype=torch.int64),
    )
    assert metric.compute()["pass@1"] == 1.0


def test_vote_count_outranks_accumulated_confidence() -> None:
    metric = _metric()
    labels = torch.full((3, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[:2] = 7

    metric.update(
        _packed(predictions, torch.tensor([-5.0, -5.0, 5.0])),
        label=labels,
        puzzle_identifiers=torch.zeros(3, dtype=torch.int64),
    )

    assert metric.compute() == {"pass@1": 0.0, "pass@2": 1.0}


def test_votes_are_grouped_per_puzzle() -> None:
    """One puzzle's views must not vote in another's ballot."""
    metric = _metric()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[2:] = 7  # The second puzzle is answered wrongly.
    metric.update(
        _packed(predictions, torch.zeros(4)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 0, 1, 1]),
    )
    assert metric.compute()["pass@1"] == 0.5


def test_padding_rows_are_not_puzzles() -> None:
    """Rows squaring off a short batch must not enter the denominator."""
    metric = _metric()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    labels[2:] = -100
    metric.update(
        _packed(labels.clone(), torch.zeros(4)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 1, 2, 3]),
    )
    assert metric.compute()["pass@1"] == 1.0


def test_all_ignored_row_does_not_hide_later_puzzle() -> None:
    labels = torch.full((3, 9), 3, dtype=torch.int64)
    labels[1] = -100
    predictions = labels.clone()
    predictions[0] = 7
    metric = _metric()

    metric.update(
        _packed(predictions, torch.zeros(3)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 1, 2]),
    )

    assert metric.compute() == {"pass@1": 0.5, "pass@2": 0.5}


def test_valid_count_truncates_before_voting() -> None:
    metric = _metric()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[2:] = 7
    metric.update(
        _packed(predictions, torch.zeros(4)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 1, 2, 3]),
        valid_count=2,
    )
    assert metric.compute()["pass@1"] == 1.0


def test_valid_count_defaults_to_number_of_batch_rows() -> None:
    labels = torch.tensor([[3, 4], [5, 6], [7, 8]])
    predictions = labels.clone()
    predictions[1] = torch.tensor([0, 0])
    metric = PassK.Config(pass_ks=(1,)).make()

    metric.update(
        _packed(predictions, torch.zeros(3)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 1, 2]),
    )

    assert metric.compute() == {"pass@1": 2 / 3}


def test_pass_k_moves_labels_and_identifiers_to_prediction_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = torch.tensor([[3, 4], [5, 6]])
    predictions = labels.clone()
    metric = PassK.Config(pass_ks=(1,)).make()
    requested_devices: list[object] = []

    def capture_to(self: Tensor, *args: object, **kwargs: object) -> Tensor:
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, torch.device):
            requested_devices.append(device)
        return self

    monkeypatch.setattr(Tensor, "to", capture_to)
    metric.update(
        _packed(predictions, torch.zeros(2)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 1]),
    )

    assert requested_devices == [torch.device("cpu"), torch.device("cpu")]


def test_votes_accumulate_across_batches() -> None:
    """Views of one puzzle arrive in different batches and must still group."""
    metric = _metric()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    wrong = labels.clone()
    wrong[:, 0] = 7
    identifiers = torch.zeros(2, dtype=torch.int64)
    metric.update(
        _packed(wrong, torch.zeros(2)),
        label=labels,
        puzzle_identifiers=identifiers,
    )
    metric.update(
        _packed(wrong, torch.zeros(2)),
        label=labels,
        puzzle_identifiers=identifiers,
    )
    metric.update(
        _packed(labels.clone(), torch.zeros(2)),
        label=labels,
        puzzle_identifiers=identifiers,
    )
    # Two wrong votes against one right: outvoted at K=1, present at K=2.
    assert metric.compute() == {"pass@1": 0.0, "pass@2": 1.0}


def test_grid_is_read_from_the_end() -> None:
    """Diagnostic columns between the halt logit and the grid are ignored."""
    metric = _metric()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    padded = torch.cat([torch.zeros(2, 6), labels.float()], dim=-1)
    metric.update(
        padded,
        label=labels,
        puzzle_identifiers=torch.zeros(2, dtype=torch.int64),
    )
    assert metric.compute()["pass@1"] == 1.0


def test_empty_metric_reports_zero() -> None:
    assert _metric().compute() == {"pass@1": 0.0, "pass@2": 0.0}


def test_pass_k_sums_each_ranks_solved_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each rank scores its own ballots; the counts are summed across ranks."""
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    local = _metric()
    local.update(
        _packed(labels.clone(), torch.zeros(2)),
        label=labels,
        puzzle_identifiers=torch.tensor([0, 1]),
    )
    # The other rank holds three puzzles and solved one of them.
    remote_counts = torch.tensor([3.0, 1.0, 1.0], dtype=torch.float64)

    def all_reduce(counts: Tensor, op: object) -> None:
        del op
        counts += remote_counts

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    assert local.compute() == {"pass@1": 3 / 5, "pass@2": 3 / 5}


def test_canonical_scoring_without_pass_at_one(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(2,)).make()

    candidate.update(logits, **batch)

    assert candidate.compute()["pass@2"] == 0.75


def test_state_round_trips() -> None:
    metric = _metric()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    metric.update(
        _packed(labels.clone(), torch.zeros(2)),
        label=labels,
        puzzle_identifiers=torch.zeros(2, dtype=torch.int64),
    )
    restored = _metric()
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == metric.compute()


def test_pass_k_state_load_defaults_missing_fields_to_empty() -> None:
    metric = _metric()

    metric.load_state_dict({})

    assert metric.state_dict() == {"votes": {}, "truth": {}}
    assert metric.compute() == {"pass@1": 0.0, "pass@2": 0.0}


def test_pass_k_state_records_each_vote_and_reset_clears_it() -> None:
    labels = torch.tensor([[2, 3, 4, 5]] * 3, dtype=torch.int64)
    predictions = torch.tensor(
        [[6, 7, 8, 9], [6, 7, 8, 9], [2, 3, 4, 5]],
        dtype=torch.int64,
    )
    metric = _metric()
    metric.update(
        _packed(predictions, torch.tensor([-1.0, 0.0, 1.0])),
        label=labels,
        puzzle_identifiers=torch.zeros(3, dtype=torch.int64),
    )
    wrong_hash = _digest(predictions[0])
    truth_hash = _digest(labels[0])
    assert metric.state_dict() == {
        "votes": {
            0: {
                wrong_hash: [
                    2.0,
                    float(torch.sigmoid(torch.tensor(-1.0)))
                    + float(torch.sigmoid(torch.tensor(0.0))),
                ],
                truth_hash: [1.0, float(torch.sigmoid(torch.tensor(1.0)))],
            },
        },
        "truth": {0: truth_hash},
    }

    metric.reset()

    assert metric.state_dict() == {"votes": {}, "truth": {}}
    assert metric.compute() == {"pass@1": 0.0, "pass@2": 0.0}


def test_canonical_votes_restore_augmented_views_and_cap_by_confidence(
    tmp_path: Path,
) -> None:
    """One wrong high-confidence view outranks a correct view of the same task."""
    source = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    answer = np.array([[5]], dtype=np.uint8)
    wrong = np.array([[6]], dtype=np.uint8)
    rng = np.random.default_rng(7)
    transform_config = ColorDihedral.Config()
    transform_config.separator = ArcSpec().puzzle_id_separator
    name, transform = transform_config.make().sample("task", rng=rng)
    pack_config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(pack_config.spec, ArcSpec)
    pack_config.spec.max_grid = 3
    pack = pack_config.make()
    original_media, original_answer = pack.pack(
        source,
        answer,
        training=False,
        rng=rng,
    )
    augmented_media, augmented_wrong = pack.pack(
        transform(source),
        transform(wrong),
        training=False,
        rng=rng,
    )
    (tmp_path / "identifiers.json").write_text(
        json.dumps(["<blank>", "task", name]),
    )
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {
                "task": {
                    "test": [{"input": source.tolist(), "output": answer.tolist()}],
                },
                "unseen": {"test": [{"input": [[1]], "output": [[2]]}]},
            },
        ),
    )
    predictions = torch.tensor(np.stack([original_answer, augmented_wrong]))
    batch = {
        "media": torch.tensor(np.stack([original_media, augmented_media])),
        "puzzle_identifiers": torch.tensor([1, 2]),
        "spatial_tags": torch.tensor([[1, 0, 0], [1, 0, 0]]),
    }
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    metric.update(_packed(predictions, torch.tensor([-4.0, 4.0])), **batch)
    assert _pass_at(metric.compute()) == {"pass@1": 0.0, "pass@2": 0.5}

    capped = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 2),
        max_views_per_input=1,
    ).make()
    capped.update(_packed(predictions, torch.tensor([-4.0, 4.0])), **batch)
    assert _pass_at(capped.compute()) == {"pass@1": 0.0, "pass@2": 0.0}

    # SpatialAugmentation packs grids into its square max_grid.
    translated_media = np.pad(
        augmented_media.reshape(3, 3)[:2, :2],
        ((1, 0), (1, 0)),
    ).reshape(-1)
    # SpatialAugmentation packs grids into its square max_grid.
    translated_wrong = np.pad(
        augmented_wrong.reshape(3, 3)[:2, :2],
        ((1, 0), (1, 0)),
    ).reshape(-1)
    tagged_batch = {
        "media": torch.tensor(np.stack([translated_media, original_media])),
        "puzzle_identifiers": torch.tensor([2, 1]),
        "spatial_tags": torch.tensor([[1, 1, 1], [1, 0, 0]]),
    }
    tagged_predictions = torch.tensor(
        np.stack([translated_wrong, original_answer]),
    )
    non_spatial = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 2),
        spatial_views="non_spatial",
    ).make()
    non_spatial.update(
        _packed(tagged_predictions, torch.tensor([4.0, -4.0])),
        **tagged_batch,
    )
    assert _pass_at(non_spatial.compute()) == {"pass@1": 0.5, "pass@2": 0.5}
    all_views = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    all_views.update(
        _packed(tagged_predictions, torch.tensor([4.0, -4.0])),
        **tagged_batch,
    )
    assert _pass_at(all_views.compute()) == {"pass@1": 0.0, "pass@2": 0.5}


def test_canonical_votes_use_configured_transform_separator(tmp_path: Path) -> None:
    source = np.array([[1, 2]], dtype=np.uint8)
    answer = np.array([[8]], dtype=np.uint8)
    transform_config = ColorDihedral.Config(separator="::", transforms=(1,))
    name, transform = transform_config.make().sample(
        "task",
        rng=np.random.default_rng(7),
    )
    pack_config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(pack_config.spec, ArcSpec)
    pack_config.spec.max_grid = 3
    pack = pack_config.make()
    media, prediction = pack.pack(
        transform(source),
        transform(answer),
        training=False,
        rng=np.random.default_rng(7),
    )
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", name]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {"task": {"test": [{"input": source.tolist(), "output": answer.tolist()}]}},
        ),
    )
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        transform=transform_config,
    ).make()
    metric.update(
        _packed(torch.from_numpy(prediction)[None, :], torch.zeros(1)),
        media=torch.from_numpy(media)[None, :],
        puzzle_identifiers=torch.tensor([1]),
    )
    assert _pass_at(metric.compute()) == {"pass@1": 1.0}


@pytest.mark.parametrize("max_views_per_input", [0, 3])
def test_canonical_equal_votes_keep_first_view_order(
    tmp_path: Path,
    max_views_per_input: int,
) -> None:
    source = np.array([[2]], dtype=np.uint8)
    answer = np.array([[8]], dtype=np.uint8)
    pack_config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(pack_config.spec, ArcSpec)
    pack_config.spec.max_grid = 2
    pack = pack_config.make()
    media = pack.pack(source, answer, training=False, rng=np.random.default_rng(0))[0]
    predictions = [
        pack.pack(
            source,
            np.array([[color]], dtype=np.uint8),
            training=False,
            rng=np.random.default_rng(0),
        )[1]
        for color in (0, 8, 1)
    ]
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", "task"]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {"task": {"test": [{"input": source.tolist(), "output": answer.tolist()}]}},
        ),
    )
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 2),
        max_views_per_input=max_views_per_input,
    ).make()
    metric.update(
        _packed(torch.tensor(np.stack(predictions)), torch.zeros(3)),
        media=torch.tensor(np.stack([media] * 3)),
        puzzle_identifiers=torch.ones(3, dtype=torch.long),
    )
    assert _pass_at(metric.compute()) == {"pass@1": 0.0, "pass@2": 1.0}


# ``a``'s first input and ``b``'s only input expect ``[[5]]``; ``a``'s second expects
# ``[[6]]``. So ``a`` is half solved and ``b`` fully solved.
def _two_input_tree(tmp_path: Path) -> dict[str, object]:
    """Tasks ``a`` (two test inputs) and ``b`` (one); every view answers ``[[5]]``."""
    pack_config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(pack_config.spec, ArcSpec)
    pack_config.spec.max_grid = 2
    pack = pack_config.make()
    inputs = [
        np.array([[2]], dtype=np.uint8),
        np.array([[3]], dtype=np.uint8),
        np.array([[4]], dtype=np.uint8),
    ]
    five = np.array([[5]], dtype=np.uint8)
    packed = [
        pack.pack(grid, five, training=False, rng=np.random.default_rng(0))
        for grid in inputs
    ]
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", "a", "b"]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {
                "a": {
                    "test": [
                        {"input": [[2]], "output": [[5]]},
                        {"input": [[3]], "output": [[6]]},
                    ],
                },
                "b": {"test": [{"input": [[4]], "output": [[5]]}]},
            },
        ),
    )
    return {
        "logits": _packed(
            torch.tensor(np.stack([answer for _, answer in packed])),
            torch.zeros(3),
        ),
        "media": torch.tensor(np.stack([media for media, _ in packed])),
        "puzzle_identifiers": torch.tensor([1, 1, 2]),
    }


def test_canonical_rules_score_strict_and_per_output(tmp_path: Path) -> None:
    """A half-solved task earns task-mean credit, no strict credit, one output."""
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        rules=[StrictPass.Config(), PerOutputPass.Config()],
    ).make()
    metric.update(logits, **batch)
    scores = metric.compute()
    assert scores == {
        "pass@1": 0.75,
        "votes_times_mean_q@1": 0.75,
        "votes_times_max_q@1": 0.75,
        "strict@1": 0.5,
        "per_output@1": 2 / 3,
    }


def test_per_task_logs_include_summary_and_hardest_tasks(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    metric = CanonicalPassK.Config().make()
    monkeypatch.setattr(
        "priml.baselines.arcagi1.metric.is_rank_zero",
        lambda: True,
    )
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi1.metric")
    metric._log_per_task(
        {"pass@1": 0.5, "ignored": "not numeric"},
        [("task-a", 0.25), ("task-b", 0.0)],
        n_test_puzzles=2,
        n_no_preds=1,
    )
    assert caplog.messages == [
        "[eval] pass@K over 2 tasks (1 with no predictions): pass@1=0.5000",
        "[eval] per-task pass@1: 1/2 tasks solved (>0). hardest unsolved:",
        "[eval]   task-b pass@1=0.000",
        "[eval]   task-a pass@1=0.250",
    ]


def test_per_task_logs_keep_only_ten_hardest_tasks(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidate = CanonicalPassK.Config().make()
    monkeypatch.setattr(
        "priml.baselines.arcagi1.metric.is_rank_zero",
        lambda: True,
    )
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi1.metric")
    per_task = [(f"task-{index}", index / 10) for index in range(11)]

    candidate._log_per_task({}, per_task, n_test_puzzles=11, n_no_preds=0)

    assert caplog.messages[2:] == [
        f"[eval]   task-{index} pass@1={index / 10:.3f}" for index in range(10)
    ]


def test_scoring_rules_cover_empty_and_partial_tasks() -> None:
    tasks = [
        TaskScore(solved=(1, 0), num_inputs=2),
        TaskScore(solved=(0, 0), num_inputs=0),
    ]
    strict = StrictPass.Config().make()
    per_output = PerOutputPass.Config().make()
    assert strict([], 0) == 0.0
    assert strict(tasks, 0) == 0.0
    assert strict([TaskScore(solved=(2,), num_inputs=2)], 0) == 1.0
    assert per_output([], 0) == 0.0
    assert per_output(tasks, 0) == 0.5
    assert per_output([TaskScore(solved=(1,), num_inputs=1)], 0) == 1.0


def test_canonical_rules_are_off_by_default(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    metric.update(logits, **batch)
    assert not any(key.startswith(("strict", "per_output")) for key in metric.compute())


def test_canonical_excluded_tasks_leave_every_denominator(tmp_path: Path) -> None:
    """Scored as if the excluded task never existed; unknown names are ignored."""
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        rules=[StrictPass.Config(), PerOutputPass.Config()],
        exclude_tasks=["b", "absent"],
    ).make()
    metric.update(logits, **batch)
    scores = metric.compute()
    assert scores["pass@1"] == 0.5
    assert scores["strict@1"] == 0.0
    assert scores["per_output@1"] == 0.5


PER_TASK_GOLDEN: Final = _CWD / "testdata" / "per_task_pass.json"
"""Minted by the reference ``PassKMetric`` (a22cbfa91) on :func:`_per_task_ballots`."""


# ``c``'s wrong answer outvotes its right one, so ``c`` and the unanswered ``d`` tie at
# pass@1 = 0 and keep their ``test_puzzles.json`` order.
def _per_task_ballots(root: Path) -> dict[str, Tensor]:
    """Four tasks on a 2x2 grid: half solved, solved, outvoted, and unanswered."""
    (root / "identifiers.json").write_text(json.dumps(["<blank>", "a", "b", "c"]))
    (root / "test_puzzles.json").write_text(
        json.dumps(
            {
                "a": {
                    "test": [
                        {"input": [[2]], "output": [[5]]},
                        {"input": [[3]], "output": [[6]]},
                    ],
                },
                "b": {"test": [{"input": [[4]], "output": [[5]]}]},
                "c": {"test": [{"input": [[1]], "output": [[6]]}]},
                "d": {"test": [{"input": [[7]], "output": [[7]]}]},
            },
        ),
    )
    # Packed 2x2 tokens: color + 2, then the row/column EOS markers.
    answers = torch.tensor([[7, 1, 1, 0]] * 5 + [[8, 1, 1, 0]])
    return {
        "logits": _packed(answers, torch.tensor([0.5, -1.0, 2.0, 1.0, -0.5, 3.0])),
        "media": torch.tensor(
            [[4, 1, 1, 0], [5, 1, 1, 0], [6, 1, 1, 0]] + [[3, 1, 1, 0]] * 3,
        ),
        "puzzle_identifiers": torch.tensor([1, 1, 2, 3, 3, 3]),
    }


def _per_task_metric(root: Path, *, dump: Path | None) -> CanonicalPassK:
    spec = ArcSpec()
    spec.max_grid = 2
    config = CanonicalPassK.Config(working_dir=root, pass_ks=(1, 2), spec=spec)
    if dump is not None:
        config.dump_per_task_path = dump
    return config.make()


def test_canonical_per_task_dump_matches_the_reference_bytes(tmp_path: Path) -> None:
    batch = _per_task_ballots(tmp_path)
    logits = batch.pop("logits")
    dump = tmp_path / "out" / "per_task_pass.json"
    metric = _per_task_metric(tmp_path, dump=dump)
    metric.update(logits, **batch)
    metric.compute()
    # The repo's end-of-file hook adds the golden's final newline; the dump has none.
    assert dump.read_bytes() + b"\n" == PER_TASK_GOLDEN.read_bytes()


def test_canonical_per_task_dump_off_writes_nothing_and_scores_alike(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _per_task_ballots(tmp_path)
    logits = batch.pop("logits")
    dumping = _per_task_metric(tmp_path, dump=tmp_path / "dump.json")
    dumping.update(logits, **batch)
    plain = _per_task_metric(tmp_path, dump=None)
    plain.update(logits, **batch)
    # A relative write would land in the working directory, so watch it too.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    before = set(tmp_path.rglob("*"))
    scores = plain.compute()
    assert set(tmp_path.rglob("*")) == before
    assert scores == dumping.compute()
    assert scores["pass@1"] == 0.375


def test_canonical_construction_defers_reading_the_tree(tmp_path: Path) -> None:
    """The loop builds metrics before the dataset stages the tree they read."""
    root = tmp_path / "staged-later"
    metric = CanonicalPassK.Config(
        working_dir=root,
        pass_ks=(1,),
        rules=[StrictPass.Config()],
    ).make()
    root.mkdir()
    batch = _two_input_tree(root)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    metric.update(logits, **batch)
    assert metric.compute()["strict@1"] == 0.5


def test_signal_dumps_rotate_but_keep_archived_steps(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The newest ``keep_last_n`` stay, and so does every ``keep_every`` multiple."""
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi1.metric")
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
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi1.metric")
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
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi1.metric")
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
    caplog.set_level(logging.WARNING, logger="priml.baselines.arcagi1.metric")

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


def _pass_at(scores: Mapping[str, object]) -> dict[str, object]:
    """Keep only the ``pass@K`` scores; report-only rankings are tested elsewhere."""
    return {key: value for key, value in scores.items() if key.startswith("pass@")}


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


def test_global_prediction_gather_merges_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CanonicalPassK.Config().make()
    digest = "0" * 64
    remote_digest = "1" * 64
    candidate._preds = {"task": {digest: [(digest, 0.5)]}}
    remote = {
        "task": {digest: [(remote_digest, 0.75)]},
        "other-task": {remote_digest: [(remote_digest, 0.75)]},
    }
    remote_payload = encode_preds(remote)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output: list[Tensor], value: Tensor) -> None:
        output[0].copy_(value)
        if value.dtype == torch.int64:
            output[1].fill_(len(remote_payload))
        else:
            remote_tensor = torch.zeros_like(value)
            remote_tensor[: len(remote_payload)] = torch.tensor(
                list(remote_payload),
                dtype=torch.uint8,
            )
            output[1].copy_(remote_tensor)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    assert candidate._global_preds() == {
        "task": {digest: [(digest, 0.5), (remote_digest, 0.75)]},
        "other-task": {remote_digest: [(remote_digest, 0.75)]},
    }


def test_nccl_gather_places_buffers_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CanonicalPassK.Config().make()
    digest = "a" * 64
    candidate._preds = {"task": {digest: [(digest, 0.5)]}}
    devices: list[tuple[str, int | None]] = []
    factory_devices: list[tuple[object, torch.dtype | None]] = []
    original_tensor = torch.tensor
    original_zeros = torch.zeros
    transfer_devices: list[object] = []

    def capture_to(self: Tensor, *args: object, **kwargs: object) -> Tensor:
        device = kwargs.get("device", args[0] if args else None)
        transfer_devices.append(device)
        return self

    def tensor_factory(
        data: list[int],
        *,
        dtype: torch.dtype | None = None,
        device: str | None = None,
    ) -> Tensor:
        factory_devices.append((device, dtype))
        return original_tensor(data, dtype=dtype, device=device)

    def zeros_factory(
        size: int,
        *,
        dtype: torch.dtype,
        device: str | None,
    ) -> Tensor:
        factory_devices.append((device, dtype))
        return original_zeros(size, dtype=dtype, device=device)

    def fake_device(kind: str, index: int | None = None) -> str:
        devices.append((kind, index))
        return "cpu"

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.distributed, "is_available", lambda: True)
        scoped.setattr(torch.distributed, "is_initialized", lambda: True)
        scoped.setattr(torch.distributed, "get_backend", lambda: "nccl")
        scoped.setattr(torch.distributed, "get_world_size", lambda: 1)

        def all_gather(output: list[Tensor], value: Tensor) -> None:
            output[0].copy_(value)

        scoped.setattr(torch.distributed, "all_gather", all_gather)
        scoped.setattr(torch.cuda, "current_device", lambda: 4)
        scoped.setattr(
            metric,
            "torch",
            SimpleNamespace(
                device=fake_device,
                tensor=tensor_factory,
                cuda=torch.cuda,
                uint8=torch.uint8,
                zeros=zeros_factory,
                frombuffer=torch.frombuffer,
                empty_like=torch.empty_like,
            ),
        )
        scoped.setattr(Tensor, "to", capture_to)
        scoped.setattr(torch, "tensor", tensor_factory)
        scoped.setattr(torch, "zeros", zeros_factory)

        assert candidate._global_preds() == candidate._preds

    assert devices == [("cpu", None), ("cuda", 4)]
    assert factory_devices == [("cpu", None), ("cpu", torch.uint8)]
    assert transfer_devices == ["cpu"]


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


def test_prediction_wire_format_round_trips_multiple_values() -> None:
    predictions = {
        "task-a": {
            "1" * 64: [("a" * 64, 0.25), ("b" * 64, 0.75)],
            "2" * 64: [("c" * 64, 0.5)],
        },
        "task-b": {},
    }
    assert decode_preds(encode_preds(predictions)) == predictions


def test_prediction_wire_format_lengths_are_unsigned_without_large_allocations() -> (
    None
):
    large_length = 2**31

    class LargePreds(dict[str, dict[str, list[tuple[str, float]]]]):
        @override
        def __len__(self) -> int:
            return large_length

    class LargeInputs(dict[str, list[tuple[str, float]]]):
        @override
        def __len__(self) -> int:
            return large_length

    class LargeList(list[tuple[str, float]]):
        @override
        def __len__(self) -> int:
            return large_length

    class LargeBytes(bytes):
        @override
        def __len__(self) -> int:
            return large_length

    class LargeName(str):
        __slots__ = ()

        @override
        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            return LargeBytes(super().encode(encoding, errors))

    outer = encode_preds(LargePreds())
    name = encode_preds({LargeName("t"): {}})
    nested = encode_preds({"t": LargeInputs()})
    values = encode_preds({"t": {"0" * 64: LargeList()}})

    assert int.from_bytes(outer[:4], "little") == large_length
    assert int.from_bytes(name[4:8], "little") == large_length
    assert int.from_bytes(nested[9:13], "little") == large_length
    assert int.from_bytes(values[45:49], "little") == large_length


@pytest.mark.parametrize("cell", [True, 2.0, "2", None])
def test_json_grid_rejects_non_integer_cells(cell: object) -> None:
    with pytest.raises(TypeError, match="ARC grid cell"):
        _json_grid([[cell, 2], [1, 3]], spec=ArcSpec(max_grid=3))


def test_json_grid_reads_integer_cells() -> None:
    grid = _json_grid([[1, 2], [0, 3]], spec=ArcSpec(max_grid=3))

    assert grid.tolist() == [[1, 2], [0, 3]]


def test_metric_private_helpers_and_width_errors() -> None:
    values = torch.tensor([[0, 257], [3, 4]])
    assert [row.tolist() for row in _uint8_rows(values)] == [[0, 1], [3, 4]]
    floats = _floats(torch.tensor([0.25, -1.5]))
    assert floats == [0.25, -1.5]
    assert all(type(value) is float for value in floats)
    assert _read_u32(b"\xff\xff\xff\xff", 0) == (4_294_967_295, 4)
    grid = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
    assert (
        _digest(grid)
        == hashlib.blake2b(
            np.array([[1, 2], [3, 4]], dtype=np.int16).tobytes(),
            digest_size=16,
        ).hexdigest()
    )
    assert _shape(np.zeros((2, 3), dtype=np.uint8)) == (2, 3)
    assert _any_rank(True)
    assert _gather_list([1, 2]) == [1, 2]
    assert _gather_grids({"x": np.zeros((2, 3), dtype=np.uint8)})["x"].shape == (2, 3)
    assert _model_output_header_width(out_width=10, media_len=9, k_steps=0) == 1
    assert _model_output_header_width(out_width=12, media_len=9, k_steps=0) == 3
    assert _model_output_header_width(out_width=13, media_len=6, k_steps=1) == 7
    assert _model_output_header_width(out_width=15, media_len=6, k_steps=2) == 9
    assert _votes_times_mean_q(4, 0.25, 0.9) == 1.0
    assert _votes_times_max_q(4, 0.25, 0.9) == 3.6
    with pytest.raises(ValueError, match="Expected len") as hash_error:
        _hash_bytes("bad")
    assert str(hash_error.value) == "Expected len(value) == 64."
    with pytest.raises(ValueError, match="per_step_acts=") as step_header_error:
        _model_output_header_width(out_width=14, media_len=6, k_steps=1)
    assert str(step_header_error.value) == (
        "per_step_acts=1 expects model_output width 13 (header 7 + grid 6); got 14."
    )
    with pytest.raises(ValueError, match="model_output width") as positive_header_error:
        _model_output_header_width(out_width=11, media_len=9, k_steps=0)
    assert str(positive_header_error.value) == (
        "model_output width 11 minus grid length 9 = header 2; "
        "inferred prediction length 9; expected 1 (baseline) or 3 "
        "(wide signal-dump) leading columns."
    )
    with pytest.raises(ValueError, match="model_output width") as zero_header_error:
        _model_output_header_width(out_width=9, media_len=9, k_steps=0)
    assert str(zero_header_error.value) == (
        "model_output width 9 minus grid length 9 = header 0; "
        "inferred prediction length 8; expected 1 (baseline) or 3 "
        "(wide signal-dump) leading columns."
    )
    with pytest.raises(ValueError, match="model_output width") as negative_header_error:
        _model_output_header_width(out_width=8, media_len=9, k_steps=0)
    assert str(negative_header_error.value) == (
        "model_output width 8 minus grid length 9 = header -1; "
        "inferred prediction length 7; expected 1 (baseline) or 3 "
        "(wide signal-dump) leading columns."
    )


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

    monkeypatch.setattr(metric, "_any_rank", capture_payload_flag)
    tracker.log_metrics({}, 1, prefix="eval/")
    assert gathered_payload_flags == [False]
    assert not (tmp_path / "signals_1.npz").exists()

    def always_true(has_payload: bool) -> bool:
        del has_payload
        return True

    monkeypatch.setattr(metric, "_any_rank", always_true)
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

    monkeypatch.setattr(metric, "write_signal_dump", capture_dump)
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


def test_canonical_empty_tree_reports_every_metric_as_zero(tmp_path: Path) -> None:
    (tmp_path / "identifiers.json").write_text("[]")
    (tmp_path / "test_puzzles.json").write_text("{}")
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 3),
        rules=[StrictPass.Config(), PerOutputPass.Config()],
    ).make()

    assert metric.compute() == {
        "pass@1": 0.0,
        "pass@3": 0.0,
        "votes_times_mean_q@1": 0.0,
        "votes_times_mean_q@3": 0.0,
        "votes_times_max_q@1": 0.0,
        "votes_times_max_q@3": 0.0,
        "strict@1": 0.0,
        "strict@3": 0.0,
        "per_output@1": 0.0,
        "per_output@3": 0.0,
    }


def test_canonical_empty_tree_preserves_signal_dump_extras(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    wide_logits = torch.cat(
        [logits[:, :1], torch.zeros((3, 2)), logits[:, 1:]],
        dim=1,
    )
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    metric.update(wide_logits, **batch)
    (tmp_path / "test_puzzles.json").write_text("{}")

    scores = metric.compute()

    assert scores["pass@1"] == 0.0
    assert scores["votes_times_mean_q@1"] == 0.0
    extras = scores["extras"]
    assert isinstance(extras, dict)
    assert isinstance(extras["signal_dump"], SignalDumpPayload)


def test_canonical_empty_task_does_not_stop_following_tasks(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    tree: dict[str, object] = {
        "empty": {"test": []},
        "a": {
            "test": [
                {"input": [[2]], "output": [[5]]},
                {"input": [[3]], "output": [[6]]},
            ],
        },
        "b": {"test": [{"input": [[4]], "output": [[5]]}]},
    }
    (tmp_path / "test_puzzles.json").write_text(json.dumps(tree))
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    metric.update(logits, **batch)

    assert metric.compute()["pass@1"] == 0.5


def test_canonical_no_predictions_score_zero_and_log_every_task(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        rules=[StrictPass.Config(), PerOutputPass.Config()],
    ).make()
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi1.metric")

    assert metric.compute() == {
        "pass@1": 0.0,
        "votes_times_mean_q@1": 0.0,
        "votes_times_max_q@1": 0.0,
        "strict@1": 0.0,
        "per_output@1": 0.0,
    }
    assert caplog.messages == [
        (
            "[eval] pass@K over 2 tasks (2 with no predictions): pass@1=0.0000 "
            "votes_times_mean_q@1=0.0000 votes_times_max_q@1=0.0000 "
            "strict@1=0.0000 per_output@1=0.0000"
        ),
        "[eval] per-task pass@1: 0/2 tasks solved (>0). hardest unsolved:",
        "[eval]   a pass@1=0.000",
        "[eval]   b pass@1=0.000",
    ]


@pytest.mark.parametrize("identifier", [-1, 3])
def test_canonical_rejects_identifiers_outside_the_map(
    tmp_path: Path,
    identifier: int,
) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    batch["puzzle_identifiers"] = torch.full((3,), identifier)
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    with pytest.raises(ValueError, match="puzzle identifier") as error:
        candidate.update(logits, **batch)

    assert str(error.value) == (
        f"puzzle identifier {identifier} is outside identifier map size 3."
    )


def test_canonical_skips_blank_before_later_puzzles(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    batch["puzzle_identifiers"] = torch.tensor([0, 1, 2])
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    metric.update(logits, **batch)

    assert set(metric._preds) == {"a", "b"}


def test_canonical_discards_noninteger_puzzle_identifiers(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    batch["puzzle_identifiers"] = torch.tensor([1.0, 1.0, 2.0])
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    candidate.update(logits, **batch)

    assert candidate._preds == {}


def test_canonical_accepts_zero_when_blank_identifier_is_nonzero(
    tmp_path: Path,
) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    (tmp_path / "identifiers.json").write_text(json.dumps(["a", "<blank>", "b"]))
    metadata = tmp_path / "test" / "dataset.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({"blank_identifier_id": 1}))
    batch["puzzle_identifiers"] = torch.tensor([0, 0, 2])
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    candidate.update(logits, **batch)

    assert set(candidate._preds) == {"a", "b"}


@pytest.mark.parametrize("spatial_tags", [[1, 1, 0], [1, 0, 1], [2, 0, 0]])
def test_non_spatial_filter_rejects_each_spatial_component(
    tmp_path: Path,
    spatial_tags: list[int],
) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    batch["puzzle_identifiers"] = torch.tensor([1, 1, 2])
    batch["spatial_tags"] = torch.tensor([spatial_tags, [1, 0, 0], [1, 0, 0]])
    candidate = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        spatial_views="non_spatial",
    ).make()

    candidate.update(logits, **batch)

    assert [len(rows) for rows in candidate._preds.values()] == [1, 1]


def test_canonical_rejects_noninteger_spatial_tags(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    batch["spatial_tags"] = torch.tensor([[1.0, 0.0, 0.0]] * 3)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    with pytest.raises(ValueError, match="not enough values to unpack"):
        metric.update(logits, **batch)


def test_canonical_state_round_trip_preserves_ballots(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    source = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    source.update(logits, **batch)
    restored = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    restored.load_state_dict(
        parse(json.dumps(source.state_dict()), dict[str, object]),
    )

    assert restored.state_dict() == source.state_dict()
    assert restored.compute() == source.compute()


def test_canonical_state_load_defaults_missing_fields_to_empty() -> None:
    metric = CanonicalPassK.Config().make()

    metric.load_state_dict({})

    assert metric.state_dict() == {"hmap": {}, "preds": {}}


def test_canonical_state_does_not_accept_boolean_grid_dimensions() -> None:
    metric = CanonicalPassK.Config().make()

    with pytest.raises(TypeError, match="grid shape"):
        metric.load_state_dict({"hmap": {"hash": [True, 3]}})


def test_canonical_state_loads_tuple_grid_shapes() -> None:
    # Checkpoints written before list-valued ``hmap`` hold in-memory tuples.
    metric = CanonicalPassK.Config().make()

    metric.load_state_dict({"hmap": {"hash": (3, 4)}})

    assert metric.state_dict().get("hmap") == {"hash": [3, 4]}


def test_canonical_compute_emits_signal_payload_from_wide_output(
    tmp_path: Path,
) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    wide_logits = torch.cat(
        [logits[:, :1], torch.zeros((3, 2)), logits[:, 1:]],
        dim=1,
    ).to(torch.float64)
    wide_logits[:, 0] = torch.tensor(
        [0.1250000001, 0.25, 0.375],
        dtype=torch.float64,
    )
    wide_logits[:, 1] = torch.tensor(
        [-1.123456789, -2.25, -3.375],
        dtype=torch.float64,
    )
    wide_logits[:, 2] = torch.tensor(
        [0.6250000001, 0.75, 0.875],
        dtype=torch.float64,
    )
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    candidate.update(wide_logits, **batch)

    result = candidate.compute()

    extras = cast(dict[str, object], result["extras"])
    assert isinstance(extras["signal_dump"], SignalDumpPayload)
    payload = extras["signal_dump"]
    prediction_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    assert payload.rows == [
        (
            name,
            grid_hash(np.array([[value]], dtype=np.uint8)),
            prediction_hash,
            float(np.float32(q_halt)),
            float(np.float32(logprob)),
            float(np.float32(stability)),
            1,
            1,
        )
        for name, value, q_halt, logprob, stability in (
            ("a", 2, 0.1250000001, -1.123456789, 0.6250000001),
            ("a", 3, 0.25, -2.25, 0.75),
            ("b", 4, 0.375, -3.375, 0.875),
        )
    ]
    assert payload.grids == {
        prediction_hash: np.array([[5]], dtype=np.uint8),
    }
    assert payload.steps == []
    assert payload.pass_ks == (1,)

    candidate.update(logits, **batch)
    assert isinstance(candidate.compute().get("extras"), dict)
    candidate.reset()
    assert "extras" not in candidate.compute()

    step_headers = torch.tensor(
        [
            [0.0, -1.0, 0.5, 3.0, 4.0, 0.123456789, 1.0],
            [0.0, -2.0, 0.75, 5.0, 6.0, 0.234567891, 0.0],
            [0.0, -3.0, 1.0, 7.0, 8.0, 0.345678912, 1.0],
        ],
        dtype=torch.float64,
    )
    step_metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        per_step_acts=1,
    ).make()
    step_metric.update(torch.cat([step_headers, logits[:, 1:]], dim=1), **batch)
    step_result = step_metric.compute()
    step_extras = cast(dict[str, object], step_result["extras"])
    step_payload = step_extras["signal_dump"]
    assert isinstance(step_payload, SignalDumpPayload)
    assert step_payload.steps == [
        (3, 4, (float(np.float32(0.123456789)),), (1,)),
        (5, 6, (float(np.float32(0.234567891)),), (0,)),
        (7, 8, (float(np.float32(0.345678912)),), (1,)),
    ]


def test_canonical_confidence_keeps_float64_tie_break_precision(
    tmp_path: Path,
) -> None:
    source = np.array([[2, 3], [4, 5]], dtype=np.uint8)
    answer = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    wrong = np.array([[6, 5], [8, 7]], dtype=np.uint8)
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", "task"]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {"task": {"test": [{"input": source.tolist(), "output": answer.tolist()}]}},
        ),
    )
    pack = SpatialAugmentation.Config(spec=ArcSpec(max_grid=3)).make()
    media, correct = pack.pack(
        source,
        answer,
        training=False,
        rng=np.random.default_rng(0),
    )
    _, incorrect = pack.pack(
        source,
        wrong,
        training=False,
        rng=np.random.default_rng(0),
    )
    logits = torch.cat(
        [
            torch.tensor([[19.0], [20.0]], dtype=torch.float32),
            torch.tensor(np.stack([correct, incorrect]), dtype=torch.float32),
        ],
        dim=1,
    )
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    metric.update(
        logits,
        media=torch.tensor(np.stack([media, media])),
        puzzle_identifiers=torch.ones(2, dtype=torch.int64),
    )

    scores = metric.compute()

    assert scores["pass@1"] == 0.0
    assert scores["pass@2"] == 1.0


def test_canonical_vote_means_accumulate_each_repeated_quality(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = grid_hash(np.array([[7]], dtype=np.uint8))
    metric._preds = {
        "a": {
            input_hash: [
                (truth_hash, 0.1),
                (truth_hash, 0.1),
                (wrong_hash, 0.4),
                (wrong_hash, 0.4),
            ],
        },
    }

    assert metric.compute()["pass@1"] == 0.0


def test_canonical_rankings_keep_task_means_and_distinct_confidence_rules(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 2),
    ).make()
    input_a = grid_hash(np.array([[2]], dtype=np.uint8))
    input_a2 = grid_hash(np.array([[3]], dtype=np.uint8))
    input_b = grid_hash(np.array([[4]], dtype=np.uint8))
    truth_a = grid_hash(np.array([[5]], dtype=np.uint8))
    truth_a2 = grid_hash(np.array([[6]], dtype=np.uint8))
    wrong = grid_hash(np.array([[7]], dtype=np.uint8))
    metric._preds = {
        "a": {
            input_a: [(wrong, 0.1), (wrong, 0.3), (truth_a, 0.5)],
            input_a2: [(truth_a2, 0.5)],
        },
        "b": {input_b: [(truth_a, 0.5)]},
    }

    scores = metric.compute()

    assert scores == {
        "pass@1": 0.75,
        "pass@2": 1.0,
        "votes_times_mean_q@1": 1.0,
        "votes_times_mean_q@2": 1.0,
        "votes_times_max_q@1": 0.75,
        "votes_times_max_q@2": 1.0,
    }


def test_canonical_view_cap_breaks_equal_confidence_by_hash(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        max_views_per_input=1,
    ).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = "0" * 64
    metric._preds = {
        "a": {input_hash: [(truth_hash, 0.5), (wrong_hash, 0.5)]},
    }

    assert metric.compute()["pass@1"] == 0.0


def test_canonical_report_mean_ranking_uses_unbiased_confidence_totals(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = grid_hash(np.array([[7]], dtype=np.uint8))
    metric._preds = {
        "a": {
            input_hash: [
                (truth_hash, 0.01),
                (truth_hash, 0.01),
                (truth_hash, 0.01),
                (wrong_hash, 0.9),
            ],
        },
    }

    scores = metric.compute()

    assert scores["pass@1"] == 0.25
    assert scores["votes_times_mean_q@1"] == 0.0
    assert scores["votes_times_max_q@1"] == 0.0


def test_canonical_report_max_uses_highest_repeated_confidence(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = grid_hash(np.array([[7]], dtype=np.uint8))
    metric._preds = {
        "a": {
            input_hash: [
                (truth_hash, 0.9),
                (truth_hash, 0.1),
                (wrong_hash, 0.7),
                (wrong_hash, 0.7),
            ],
        },
    }

    assert metric.compute()["votes_times_max_q@1"] == 0.25


def test_canonical_report_rankings_use_mean_and_max_confidence(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = "0" * 64
    metric._preds = {
        "a": {
            input_hash: [
                (truth_hash, 0.2),
                (truth_hash, 0.9),
                (wrong_hash, 0.5),
                (wrong_hash, 0.5),
            ],
        },
    }

    scores = metric.compute()

    assert scores["pass@1"] == 0.25
    assert scores["votes_times_mean_q@1"] == 0.25
    assert scores["votes_times_max_q@1"] == 0.25


def test_canonical_compute_continues_after_an_input_without_predictions(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_a2 = grid_hash(np.array([[3]], dtype=np.uint8))
    truth_a2 = grid_hash(np.array([[6]], dtype=np.uint8))
    input_b = grid_hash(np.array([[4]], dtype=np.uint8))
    truth_b = grid_hash(np.array([[5]], dtype=np.uint8))
    metric._preds = {
        "a": {input_a2: [(truth_a2, 0.5)]},
        "b": {input_b: [(truth_b, 0.5)]},
    }
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi1.metric")

    scores = metric.compute()

    assert scores["pass@1"] == 0.75
    assert "(0 with no predictions)" in caplog.messages[0]
    assert caplog.messages[2:] == [
        "[eval]   a pass@1=0.500",
        "[eval]   b pass@1=1.000",
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
