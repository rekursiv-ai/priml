"""Tests for the pass@K voting metric."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1.augmentation import (
    ArcSpec,
    ColorDihedral,
    SpatialAugmentation,
)
from priml.baselines.arcagi1.metric import (
    CanonicalPassK,
    PassK,
    PerOutputPass,
    StrictPass,
)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


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
        "media": torch.tensor(np.stack([original_media, translated_media])),
        "puzzle_identifiers": torch.tensor([1, 2]),
        "spatial_tags": torch.tensor([[1, 0, 0], [1, 1, 1]]),
    }
    tagged_predictions = torch.tensor(
        np.stack([original_answer, translated_wrong]),
    )
    non_spatial = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 2),
        spatial_views="non_spatial",
    ).make()
    non_spatial.update(
        _packed(tagged_predictions, torch.tensor([-4.0, 4.0])),
        **tagged_batch,
    )
    assert _pass_at(non_spatial.compute()) == {"pass@1": 0.5, "pass@2": 0.5}
    all_views = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    all_views.update(
        _packed(tagged_predictions, torch.tensor([-4.0, 4.0])),
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
    assert scores["pass@1"] == 0.75
    assert scores["strict@1"] == 0.5
    assert scores["per_output@1"] == 2 / 3


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


def _pass_at(scores: Mapping[str, object]) -> dict[str, object]:
    """Keep only the ``pass@K`` scores; report-only rankings are tested elsewhere."""
    return {key: value for key, value in scores.items() if key.startswith("pass@")}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
