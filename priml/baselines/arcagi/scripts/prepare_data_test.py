"""Tests for the ARC dataset preparation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import argparse
import json
import logging

from treekle.codec import from_plain, loads

import numpy as np
import pytest

from priml.baselines.arcagi import experiments
from priml.baselines.arcagi.data import PuzzleData
from priml.baselines.arcagi.scripts import prepare_data
from priml.baselines.sudoku.prefix import SparsePuzzleEmbedding
from priml.lib.testing.cli import assert_help_without_docstring


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.arcagi.augmentation import ArcAugmentation


def _write_source(source: Path) -> Path:
    """Write one tiny task per subset; return the ``--input-prefix``."""
    source.mkdir()
    puzzle = {
        "train": [{"input": [[0, 1, 2], [3, 4, 5]], "output": [[5, 4, 3], [2, 1, 0]]}],
        "test": [{"input": [[1, 2, 3], [4, 5, 6]]}],
    }
    for subset in ("training", "evaluation", "concept"):
        (source / f"arc_{subset}_challenges.json").write_text(
            json.dumps({f"{subset}-puzzle": puzzle}),
        )
        (source / f"arc_{subset}_solutions.json").write_text(
            json.dumps({f"{subset}-puzzle": [[[6, 5, 4], [3, 2, 1]]]}),
        )
    return source / "arc"


def test_default_directory_matches_exp000_dataset() -> None:
    assert prepare_data.default_directory() == Path(
        "/opt/scratch/datasets/arc1concept-aug-1000",
    )


def test_dataset_config_uses_experiment_dataset_and_base() -> None:
    config = prepare_data.dataset_config("exp_smoke").finalize()
    assert config.working_dir == Path("/opt/scratch/datasets/arc1concept-aug-1000")


def test_exp005_prepares_the_spatial_views_its_directory_names(tmp_path: Path) -> None:
    """The ``-spatialeval-v2`` tree must carry tagged spatial views.

    Without them every view is the identity, and exp005's ``spatial_eq`` and
    ``spatial_big`` slices silently equal its non-spatial one.
    """
    config = prepare_data.dataset_config("exp005")
    config.base_dir = None
    config.working_dir = tmp_path / "prepared"
    config.augmentation.num_aug = 0
    target = prepare_data.prepare(
        config,
        input_file_prefix=_write_source(tmp_path / "source"),
    )
    loaded = cast(
        "NDArray[np.int32]",
        np.load(target / "test" / "all__spatial_tags.npy"),
    )
    tags = from_plain(cast(object, loaded.tolist()), list[list[int]])
    assert any(tag != [1, 0, 0] for tag in tags)


def test_explicit_directory_is_taken_verbatim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--directory /x`` writes to ``/x``, never beneath the experiment's root."""
    target = tmp_path / "explicit"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--directory",
            str(target),
            "--input-prefix",
            str(_write_source(tmp_path / "source")),
            "--num-aug",
            "0",
        ],
    )
    assert prepare_data.main() == 0
    assert (target / "identifiers.json").is_file()


def test_dataset_declares_its_augmentation_recipe(tmp_path: Path) -> None:
    config = PuzzleData.Config()
    config.working_dir = tmp_path / "absent"
    rendered = config.pformat(hide_default_values=False)
    assert "augmentation" in rendered
    assert "num_aug=1_000" in rendered
    assert "translation_prob=1.0" in rendered
    assert not (tmp_path / "absent").exists()


def test_experiment_flag_forwards_that_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--experiment`` builds the named recipe's tree, not the default one."""
    captured: list[PuzzleData.Config] = []
    monkeypatch.setattr("sys.argv", ["prepare_data.py", "--experiment", "exp005"])

    def fake_prepare(
        config: PuzzleData.Config,
        *,
        input_file_prefix: Path | str | None = None,
    ) -> Path:
        del input_file_prefix
        captured.append(config.copy_tree().finalize())
        return Path(captured[0].working_dir)

    monkeypatch.setattr(prepare_data, "prepare", fake_prepare)

    assert prepare_data.main() == 0
    expected = experiments.exp005().copy_tree().finalize().dataset
    spatial = captured[0].augmentation.spatial
    assert spatial.translation_prob == spatial.scale_prob == 0.2
    assert spatial.train_scale_weights == {2: 1.0}
    assert captured[0].working_dir == expected.working_dir


def test_prepared_tree_loads_and_sizes_exp000(tmp_path: Path) -> None:
    """Build a tiny tree, load it, and size exp000's table from its identifiers."""
    recipe = PuzzleData.Config()
    recipe.working_dir = tmp_path / "dataset"
    recipe.augmentation.num_aug = 0
    directory = prepare_data.prepare(
        recipe,
        input_file_prefix=_write_source(tmp_path / "source"),
    )
    identifiers = from_plain(
        loads((directory / "identifiers.json").read_text()),
        list[str],
    )
    assert len(identifiers) > 1
    config = PuzzleData.Config()
    config.base_dir = "/"
    config.working_dir = str(directory)
    config.device = "cpu"
    config.batch_size = 1
    batch = next(iter(config.make().train_dataloader()))
    assert int(batch["puzzle_identifiers"].max()) > 0
    experiment = experiments.exp000()
    table = experiment.step.model.prefix
    assert isinstance(table, SparsePuzzleEmbedding.Config)
    table.num_puzzles = len(identifiers)
    finalized = experiment.copy_tree().finalize().step.model.prefix
    assert isinstance(finalized, SparsePuzzleEmbedding.Config)
    assert finalized.num_puzzles == len(identifiers)


def test_main_help_works_without_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_help_without_docstring(monkeypatch, prepare_data, prepare_data.main)


def test_prepare_uses_the_pinned_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "prepared"
    config = prepare_data.dataset_config("exp000")
    config.base_dir = None
    config.working_dir = target
    config.augmentation.num_aug = 7
    config.augmentation.seed = 9
    calls: list[tuple[Path, str | None, ArcAugmentation]] = []

    def build_override(
        *,
        target_dir: Path,
        input_file_prefix: str | None,
        augmentation: ArcAugmentation,
    ) -> None:
        calls.append((target_dir, input_file_prefix, augmentation))

    monkeypatch.setattr(prepare_data, "build_arc_dataset", build_override)

    assert prepare_data.prepare(config) == target
    assert len(calls) == 1
    built_target, built_prefix, augmentation = calls[0]
    assert built_target == target
    # None selects the pinned clone inside the builder, which stamps the pin.
    assert built_prefix is None
    assert augmentation.config.num_aug == 7
    assert augmentation.config.seed == 9


def test_main_builds_tiny_local_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    puzzle = {
        "train": [
            {
                "input": [[0, 1, 2], [3, 4, 5]],
                "output": [[5, 4, 3], [2, 1, 0]],
            },
        ],
        "test": [{"input": [[1, 2, 3], [4, 5, 6]]}],
    }
    for subset in ("training", "evaluation", "concept"):
        (source / f"arc_{subset}_challenges.json").write_text(
            json.dumps({f"{subset}-puzzle": puzzle}),
        )
        (source / f"arc_{subset}_solutions.json").write_text(
            json.dumps({f"{subset}-puzzle": [[[6, 5, 4], [3, 2, 1]]]}),
        )

    target = tmp_path / "prepared"
    config = prepare_data.dataset_config("exp000")
    config.base_dir = None

    def dataset_config_override(
        experiment: str,
    ) -> PuzzleData.Config:
        del experiment
        return config

    monkeypatch.setattr(prepare_data, "dataset_config", dataset_config_override)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--directory",
            str(target),
            "--experiment",
            "exp000",
            "--input-prefix",
            str(source / "arc"),
            "--num-aug",
            "0",
            "--seed",
            "3",
        ],
    )

    assert prepare_data.main() == 0
    assert sorted(path.name for path in target.iterdir()) == [
        "_build_params.json",
        "identifiers.json",
        "test",
        "test_puzzles.json",
        "train",
    ]
    assert json.loads((target / "identifiers.json").read_text()) == [
        "<blank>",
        "training-puzzle",
        "evaluation-puzzle",
        "concept-puzzle",
    ]
    assert sorted(path.name for path in (target / "train").iterdir()) == [
        "all__group_indices.npy",
        "all__inputs.npy",
        "all__labels.npy",
        "all__puzzle_identifiers.npy",
        "all__puzzle_indices.npy",
        "dataset.json",
    ]


def test_main_help_shows_the_complete_script_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["prepare_data.py", "--help"])
    with pytest.raises(SystemExit, match="0"):
        prepare_data.main()
    help_text = capsys.readouterr().out
    description = help_text.split("\n\n", 1)[1].split("\noptions:\n", 1)[0]
    assert description == (
        "Prepare the pinned ARC-AGI-1 ``arc1concept-aug-1000`` dataset.\n"
        "\n"
        "The default destination matches ``exp000``'s dataset under a default\n"
        "``TrainLoop``. The source revision is immutable; a local ``--input-prefix``\n"
        "keeps tests and offline rebuilds hermetic. The ARC-AGI-2 rungs build their\n"
        "own tree on first use.\n"
        "\n"
        "Examples:\n"
        "  prepare_data.py\n"
        "  prepare_data.py --base-dir /data/scratch --experiment exp005\n"
        "  prepare_data.py --directory /datasets/my-arcagi\n"
        "  prepare_data.py --experiment exp005\n"
    )


def test_main_applies_overrides_and_passes_local_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "destination"
    prefix = tmp_path / "local-arc"
    config = prepare_data.dataset_config("exp000")
    config.base_dir = None
    configs: list[str] = []
    calls: list[tuple[PuzzleData.Config, Path | str | None]] = []
    logging_options: list[dict[str, object]] = []

    def dataset_config_override(
        experiment: str,
    ) -> PuzzleData.Config:
        configs.append(experiment)
        return config

    def prepare_override(
        received: PuzzleData.Config,
        *,
        input_file_prefix: Path | str | None,
    ) -> None:
        calls.append((received, input_file_prefix))

    def basic_config_override(**kwargs: object) -> None:
        logging_options.append(kwargs)

    monkeypatch.setattr(prepare_data, "dataset_config", dataset_config_override)
    monkeypatch.setattr(prepare_data, "prepare", prepare_override)
    monkeypatch.setattr(logging, "basicConfig", basic_config_override)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--directory",
            str(destination),
            "--experiment",
            "exp000",
            "--input-prefix",
            str(prefix),
            "--num-aug",
            "4",
            "--seed",
            "3",
        ],
    )

    assert prepare_data.main() == 0
    assert configs == ["exp000"]
    assert logging_options == [{"level": logging.INFO, "format": "%(message)s"}]
    assert config.working_dir == destination
    assert config.augmentation.num_aug == 4
    assert config.augmentation.seed == 3
    assert calls == [(config, prefix)]


def test_add_arguments_defaults_and_parses_all_options() -> None:
    parser = argparse.ArgumentParser()
    prepare_data._add_arguments(parser)
    defaults = parser.parse_args([])
    assert vars(defaults) == {
        "base_dir": None,
        "directory": None,
        "experiment": "exp000",
        "input_prefix": None,
        "num_aug": None,
        "seed": None,
    }

    flags = parser.parse_args(
        [
            "--base-dir",
            "/scratch/root",
            "--directory",
            "/scratch/data",
            "--experiment",
            "exp007",
            "--input-prefix",
            "/scratch/input",
            "--num-aug",
            "3",
            "--seed",
            "11",
        ],
    )
    assert vars(flags) == {
        "base_dir": Path("/scratch/root"),
        "directory": Path("/scratch/data"),
        "experiment": "exp007",
        "input_prefix": Path("/scratch/input"),
        "num_aug": 3,
        "seed": 11,
    }


def test_base_dir_relocates_the_tree_the_experiment_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One root moves the build, and the rung reads the same tree beneath it."""
    built: list[PuzzleData.Config] = []

    def prepare_override(
        received: PuzzleData.Config,
        *,
        input_file_prefix: Path | str | None,
    ) -> None:
        del input_file_prefix
        built.append(received)

    monkeypatch.setattr(prepare_data, "prepare", prepare_override)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_data.py", "--base-dir", str(tmp_path), "--experiment", "exp005"],
    )
    assert prepare_data.main() == 0
    expected = experiments.exp005()
    expected.base_dir = tmp_path
    (config,) = built
    assert (
        config.copy_tree().finalize().working_dir
        == expected.copy_tree().finalize().dataset.working_dir
    )
    assert str(config.copy_tree().finalize().working_dir).startswith(str(tmp_path))


@pytest.mark.parametrize("experiment", ["exp007", "exp008"])
def test_an_arc2_rung_is_refused_rather_than_built_from_arc1(experiment: str) -> None:
    """Refuse an ARC-AGI-2 rung: its tree builds through its own loader."""
    with pytest.raises(TypeError, match="first use"):
        prepare_data.prepare(prepare_data.dataset_config(experiment))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
