"""Checks for the experiments module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from priml.baselines.casmi2026.chemistry import OfficialIdentity
from priml.baselines.casmi2026.data import TrainingPreparation
from priml.baselines.casmi2026.dreams import MainBank
from priml.baselines.casmi2026.experiments import (
    exp000,
    train_exp000,
    train_exp001,
    train_exp002,
)
from priml.baselines.casmi2026.metric import MoleculeMRR
from priml.baselines.casmi2026.model import DreamsFP
from priml.baselines.casmi2026.pubchem import PubChemAssets
from priml.baselines.casmi2026.recipe import (
    MainAssets,
    MainResource,
    PubChemResource,
    SecondResource,
    SimulatorStage,
)
from priml.baselines.casmi2026.second_engine import SecondAssets
from priml.baselines.casmi2026.simulators import GlacierBackend, IceBackend
from priml.baselines.casmi2026.train_step import CasmiRun, casmi_loop
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Callable


def test_exp000_config_is_complete_and_hermetic() -> None:
    config = exp000().copy_tree().finalize()
    assert config.device == "cuda"
    assert config.working_dir == Path("/opt/scratch/runs/casmi2026/exp000")
    assert config.dreams == Path(
        "/opt/scratch/datasets/casmi2026/inputs/datasets/hengck23/hengck23-dreams-enveda-casmi26",
    )
    assert isinstance(config.main, MainResource.Config)
    assert isinstance(config.main.assets, MainAssets.Config)
    assert isinstance(config.main.assets.bank, MainBank.Config)
    assert isinstance(config.main.assets.views_bank, MainBank.Config)
    assert config.main.assets.bank.device == "cuda"
    assert config.main.assets.views_bank.device == "cuda"
    assert isinstance(config.second, SecondResource.Config)
    assert isinstance(config.second.assets, SecondAssets.Config)
    assert config.second.assets.bank.device == "cuda"
    assert config.second.assets.rankers.own.tree.rounds == 500
    assert config.second.assets.rankers.pv.tree.rounds == 500
    assert isinstance(config.pubchem, PubChemResource.Config)
    assert isinstance(config.pubchem.assets, PubChemAssets.Config)
    assert config.pubchem.assets.bank.device == "cuda"
    assert isinstance(config.ice, SimulatorStage.Config)
    assert isinstance(config.ice.backend, IceBackend.Config)
    assert isinstance(config.glacier, SimulatorStage.Config)
    assert isinstance(config.glacier.backend, GlacierBackend.Config)
    assert config.ice.backend.device == config.glacier.backend.device == "cuda"
    assert config.ice.budget.seconds == 5400
    assert config.glacier.budget.seconds == 4000
    assert config.ice.batch_size == 96
    assert config.glacier.batch_size == 32
    assert config.ice.chunk_jobs == config.glacier.chunk_jobs == 3000
    assert_pprint_golden(test_file=__file__, name="exp000", config=config)


def test_exp000_pins_every_released_checkpoint_and_simulator_asset() -> None:
    config = exp000().copy_tree().finalize()
    assert isinstance(config.main, MainResource.Config)
    assert isinstance(config.main.assets, MainAssets.Config)
    assert isinstance(config.main.assets.bank, MainBank.Config)
    assert isinstance(config.main.assets.views_bank, MainBank.Config)
    assert [Path(pin.path).name for pin in config.main.assets.bank.checkpoints] == [
        "fpnet_full1.pt",
    ]
    assert [
        Path(pin.path).name for pin in config.main.assets.views_bank.checkpoints
    ] == [
        "fe_A.pt",
        "fe_B.pt",
        "fe_D.pt",
    ]
    assert isinstance(
        config.main.assets.views_bank.checkpoints[2].model,
        DreamsFP.Config,
    )
    assert isinstance(config.second, SecondResource.Config)
    assert isinstance(config.second.assets, SecondAssets.Config)
    bank = config.second.assets.bank
    assert [Path(pin.path).name for pin in bank.merged] == ["fp_merged_m1.pt"]
    assert [Path(pin.path).name for pin in bank.single] == ["fp_single_s2.pt"]
    assert [pin.sha256 for pin in (*bank.merged, *bank.single)] == [
        "261529c823825cbe2028684bbf7214a95a7e631ca9f656336d577c507cce8294",
        "c19aa29016ed4d7a67bac38105b8fa5076c696c29eb66a504558c9b2e52b614a",
    ]
    rankers = config.second.assets.rankers
    assert (rankers.own.priors, rankers.own.seeds) == ([0.35, 0.55], [0, 1])
    assert (rankers.pv.priors, rankers.pv.seeds) == ([0.30, 0.60], [0, 1, 2, 3])
    assert rankers.weight_fingerprint == 0.88
    assert isinstance(config.pubchem, PubChemResource.Config)
    assert isinstance(config.pubchem.assets, PubChemAssets.Config)
    assert [Path(pin.path).name for pin in config.pubchem.assets.bank.checkpoints] == [
        "fpnet_0.pt",
        "fpnet_1.pt",
    ]
    assert isinstance(config.ice, SimulatorStage.Config)
    assert isinstance(config.glacier, SimulatorStage.Config)
    assert set(config.ice.asset_seals.values()) == {
        "deef0451c1870d2125653bb212c9d95c5b5d9bf80156dfa9b22fedda7db49bea",
        "c49e330e049d3a8b0095026ed694fcf70045c24305354853732aaa5f89d55068",
    }
    assert set(config.glacier.asset_seals.values()) == {
        "9192e4c2eeafe83317d5a2c2641e1fc4d68934488757a4faafeb66f2ab397a12",
    }
    votes = (config.policy.legacy, config.policy.control, config.policy.final)
    assert {(vote.alpha, vote.offset, vote.window) for vote in votes} == {
        (0.6, 3.0, 40),
    }
    assert config.policy.fallback is not None
    assert config.policy.fallback.gap_weight == 0.5


@pytest.mark.parametrize("factory", [train_exp000, train_exp001, train_exp002])
def test_training_protocol_is_explicit_and_separate_from_released_checkpoints(
    factory: Callable[[], CasmiRun],
) -> None:
    config = factory()
    assert config.loop.seed is None
    assert casmi_loop(config).step.pretrained is None
    assert casmi_loop(config).step.model.channels_hidden == 128
    assert casmi_loop(config).step.model.num_layers == 2
    assert config.loop.max_time == casmi_loop(config).step.train_budget_sec == 240
    assert config.loop.max_eval_time == 45
    assert config.budget.seconds == 300
    assert config.loop.num_steps_eval == -1
    assert config.loop.eval_stop_on_time_limit is False


def test_train_exp000_config_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="train_exp000", config=train_exp000())


def test_exp001_changes_only_the_declared_peak_cap() -> None:
    parent, child = train_exp000(), train_exp001()
    assert child.loop.experiment_name == "train_exp001"
    assert casmi_loop(child).dataset.max_peaks == 128
    casmi_loop(child).dataset.max_peaks = casmi_loop(parent).dataset.max_peaks
    child.loop.experiment_name = parent.loop.experiment_name
    assert child.pformat(hide_default_values=False) == parent.pformat(
        hide_default_values=False,
    )


def test_exp002_changes_only_the_declared_cold_preparation() -> None:
    parent, child = train_exp000(), train_exp002()
    assert child.loop.experiment_name == "train_exp002"
    assert isinstance(child.preparation, TrainingPreparation.Config)
    child.preparation = parent.preparation
    child.loop.experiment_name = parent.loop.experiment_name
    assert child.pformat(hide_default_values=False) == parent.pformat(
        hide_default_values=False,
    )


def test_exp002_dataset_root_is_the_preparation_work_directory() -> None:
    job = train_exp002().finalized()
    assert isinstance(job.preparation, TrainingPreparation.Config)
    assert casmi_loop(job).dataset.working_dir == job.preparation.working_dir / "work"


def test_dataset_and_retrieval_metric_resolve_one_shared_cache_root(
    tmp_path: Path,
) -> None:
    config = train_exp000()
    config.loop.base_dir = tmp_path
    metric = casmi_loop(config).metrics_eval["retrieval"]
    assert isinstance(metric, MoleculeMRR.Config)
    casmi_loop(config).dataset.working_dir = metric.working_dir = (
        "/datasets/alternate/work"
    )
    finalized = config.copy_tree().finalize()
    resolved = casmi_loop(finalized).metrics_eval["retrieval"]
    assert isinstance(resolved, MoleculeMRR.Config)
    assert (
        resolved.working_dir
        == casmi_loop(finalized).dataset.working_dir
        == tmp_path / "datasets/alternate/work"
    )
    assert isinstance(resolved.identity, OfficialIdentity.Config)
    assert resolved.identity.working_dir == (
        tmp_path / "datasets/casmi2026/native-chemistry/main"
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
