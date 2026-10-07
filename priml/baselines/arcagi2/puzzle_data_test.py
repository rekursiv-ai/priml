"""Tests for the ARC-AGI-2 puzzle dataset staging + loading."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import argparse

import pytest
import torch

from priml.baselines.arcagi2 import puzzle_data
from priml.baselines.arcagi2.data import Arc2Data
from priml.baselines.arcagi2.metric import PassK
from priml.baselines.arcagi2.puzzle_data import Arc2PuzzleDataset
from priml.baselines.arcagi2.scripts import prepare_data
from priml.baselines.arcagi2.scripts.build_dataset import (
    ARC2_DATASET_DIR,
    arc2_num_puzzle_identifiers,
    ensure_arc2_dataset,
)
from priml.baselines.arcagi2.scripts.build_dataset_test import (
    write_tiny_kaggle_source,
)
from priml.paths import resolve_working_dir


def _tiny_tree(tmp_path: Path) -> tuple[Path, str]:
    """Build a tiny arc2 tree in ``tmp_path``; return (root, source prefix)."""
    prefix = write_tiny_kaggle_source(tmp_path)
    target = tmp_path / "arc2concept-tiny"
    root = ensure_arc2_dataset(
        target=target,
        num_aug=2,
        seed=7,
        input_file_prefix=prefix,
    )
    return root, prefix


def test_default_working_dir_is_arc2_logical() -> None:
    cfg = Arc2PuzzleDataset.Config()
    assert cfg.working_dir == "/datasets/arc2concept-aug-1000"


def test_every_arc2_reader_defaults_to_the_directory_the_builder_writes() -> None:
    """A reader defaulting elsewhere would find no tree, or another build's."""
    readers = (
        Arc2PuzzleDataset.Config().working_dir,
        Arc2Data.Config().working_dir,
        PassK.Config().working_dir,
    )
    assert set(readers) == {ARC2_DATASET_DIR}
    parser = argparse.ArgumentParser()
    prepare_data._add_arguments(parser)
    flags = cast(prepare_data._Flags, parser.parse_args(["source"]))
    assert flags.destination == resolve_working_dir("/opt/scratch", ARC2_DATASET_DIR)


def test_finalize_resolves_working_dir_under_base_dir() -> None:
    cfg = Arc2PuzzleDataset.Config()
    cfg.source_dataset_dir = "/datasets/arc2concept-aug-1000-tr0p2"
    cfg.base_dir = "/opt/scratch"
    resolved = cfg.copy_tree().finalize()
    assert resolved.working_dir == Path("/opt/scratch/datasets/arc2concept-aug-1000")
    # A str source_dataset_dir is a logical path joined beneath base_dir.
    assert resolved.source_dataset_dir == Path(
        "/opt/scratch/datasets/arc2concept-aug-1000-tr0p2",
    )


def test_finalize_joins_path_source_dataset_dir_under_base_dir() -> None:
    cfg = Arc2PuzzleDataset.Config()
    # A Path joins under base_dir exactly like a str (no literal passthrough):
    # foreign reads set base_dir, not a bespoke absolute source path.
    cfg.source_dataset_dir = Path("/datasets/arc2concept-aug-1000-tr0p2")
    cfg.base_dir = "/opt/scratch"
    resolved = cfg.copy_tree().finalize()
    assert resolved.source_dataset_dir == Path(
        "/opt/scratch/datasets/arc2concept-aug-1000-tr0p2",
    )


def test_loads_tiny_tree_without_staging(tmp_path: Path) -> None:
    root, _ = _tiny_tree(tmp_path)
    cfg = Arc2PuzzleDataset.Config()
    cfg.working_dir = root
    cfg.num_puzzle_identifiers = 0  # Gate off: load only, never build.
    cfg.batch_size = 2
    cfg.eval_batch_size = 2
    cfg.device = "cpu"
    dataset = cfg.make()
    batch = next(iter(dataset.eval_dataloader()))
    media = batch["media"]
    label = batch["label"]
    puzzle_identifiers = batch["puzzle_identifiers"]
    assert media.shape[-1] == 900
    assert media.dtype == torch.int32
    assert label.shape == media.shape
    assert puzzle_identifiers.shape[0] == media.shape[0]


def test_staging_gate_verifies_identifier_count(tmp_path: Path) -> None:
    root, prefix = _tiny_tree(tmp_path)
    cfg = Arc2PuzzleDataset.Config()
    cfg.working_dir = root
    # Same build params as the staged tiny tree, so ensure no-ops on the
    # sentinel and the loud count check is what fires.
    cfg.augmentation.num_aug = 2
    cfg.augmentation.seed = 7
    cfg.input_file_prefix = prefix
    cfg.num_puzzle_identifiers = 999_999
    cfg.batch_size = 2
    cfg.device = "cpu"
    with pytest.raises(ValueError, match="puzzle identifier"):
        cfg.make()


def test_correct_identifier_count_accepted(tmp_path: Path) -> None:
    root, prefix = _tiny_tree(tmp_path)
    cfg = Arc2PuzzleDataset.Config()
    cfg.working_dir = root
    cfg.augmentation.num_aug = 2
    cfg.augmentation.seed = 7
    cfg.input_file_prefix = prefix
    cfg.num_puzzle_identifiers = arc2_num_puzzle_identifiers(root)
    cfg.batch_size = 2
    cfg.device = "cpu"
    dataset = cfg.make()
    assert dataset.dataset_dir == root


def test_an_arc2_tree_is_never_checked_against_the_arc1_recipe(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The ARC2 tree is stamped by the ARC2 builder; the ARC1 ensure must not run."""
    root, prefix = _tiny_tree(tmp_path)
    cfg = Arc2PuzzleDataset.Config()
    cfg.working_dir = root
    cfg.augmentation.num_aug = 2
    cfg.augmentation.seed = 7
    cfg.input_file_prefix = prefix
    cfg.num_puzzle_identifiers = arc2_num_puzzle_identifiers(root)
    cfg.batch_size = 2
    cfg.device = "cpu"
    caplog.set_level("WARNING")

    assert cfg.make().dataset_dir == root
    assert caplog.messages == []


def test_staging_forwards_the_complete_recipe_and_one_spatial_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    arc2_calls: list[dict[str, object]] = []
    spatial_calls: list[dict[str, object]] = []

    def record_arc2(**kwargs: object) -> None:
        arc2_calls.append(kwargs)

    def record_spatial(**kwargs: object) -> None:
        spatial_calls.append(kwargs)

    monkeypatch.setattr(puzzle_data, "ensure_arc2_dataset", record_arc2)
    monkeypatch.setattr(puzzle_data, "ensure_spatial_eval_data", record_spatial)
    source = tmp_path / "source"
    target = tmp_path / "target"
    cfg = Arc2PuzzleDataset.Config()
    cfg.working_dir = target
    cfg.source_dataset_dir = source
    cfg.spatial_eval_views = 1
    cfg.num_puzzle_identifiers = 2
    cfg.augmentation.spatial.train_scale_weights = {1: 0.25}
    cfg.augmentation.spatial.translation_prob = 0.4
    cfg.augmentation.spatial.scale_prob = 0.7
    cfg.augmentation.num_aug = 3
    cfg.augmentation.seed = 13
    cfg.input_file_prefix = "alternate-"
    # The id-count check after staging reads this; the stubs build nothing.
    target.mkdir()
    (target / "identifiers.json").write_text('["<blank>", "task"]')

    Arc2PuzzleDataset(cfg)

    assert arc2_calls == [
        {
            "target": source,
            "train_scale_weights": {1: 0.25},
            "translation_prob": 0.4,
            "scale_prob": 0.7,
            "num_aug": 3,
            "seed": 13,
            "input_file_prefix": "alternate-",
        },
    ]
    assert spatial_calls == [
        {"source_dir": source, "spatial_views": 1, "target": target},
    ]


def test_spatial_views_require_source_dir(tmp_path: Path) -> None:
    root, _ = _tiny_tree(tmp_path)
    cfg = Arc2PuzzleDataset.Config()
    cfg.working_dir = root
    cfg.num_puzzle_identifiers = 1
    cfg.spatial_eval_views = 1
    with pytest.raises(
        ValueError,
        match="spatial_eval_views > 0 requires source_dataset_dir",
    ) as error:
        cfg.make()
    assert str(error.value) == (
        "spatial_eval_views > 0 requires source_dataset_dir (the base "
        "tree the spatial expansion derives from)."
    )


def test_spatial_staging_lands_at_dataset_dir(tmp_path: Path) -> None:
    root, prefix = _tiny_tree(tmp_path)
    spatial_dir = tmp_path / "arc2concept-tiny-spatialeval-v2"
    cfg = Arc2PuzzleDataset.Config()
    cfg.working_dir = spatial_dir
    cfg.source_dataset_dir = root
    cfg.spatial_eval_views = 2
    cfg.augmentation.num_aug = 2
    cfg.augmentation.seed = 7
    cfg.input_file_prefix = prefix
    cfg.num_puzzle_identifiers = arc2_num_puzzle_identifiers(root)
    cfg.batch_size = 2
    cfg.device = "cpu"
    dataset = cfg.make()
    # Staged at THIS config's dataset_dir (not a scratch-derived path), with the
    # spatial-tags sidecar and the verbatim train copy. The copied train
    # split is part of the staged manifest (a partial tree must never be
    # adopted -- the ARC-1-rebuild leakage guard).
    assert dataset.dataset_dir == spatial_dir
    assert (spatial_dir / "test" / "all__spatial_tags.npy").is_file()
    assert (spatial_dir / "train" / "all__inputs.npy").is_file()
    # Spatial views reuse their source puzzle id, so the id vocab (and the
    # model's embedding size, exp005's 1_191_727) is unchanged -- the count
    # verify above passing against the BASE count proves it.
    assert arc2_num_puzzle_identifiers(spatial_dir) == arc2_num_puzzle_identifiers(root)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
