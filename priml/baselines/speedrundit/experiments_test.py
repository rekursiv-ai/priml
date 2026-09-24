"""The experiment ladder: what each recipe sets, and what it must not touch.

Every test here builds configs only. No data, no device, no training -- a
config must finalize with neither the corpus nor a GPU present, and these
tests are what keep that true.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import importlib
import re
import typing

from configgle import apply_overrides
from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.baselines.speedrundit import experiments
from priml.baselines.speedrundit.data import SpeedrunDiTData
from priml.baselines.speedrundit.experiments import (
    SpeedrunDiTLoop,
    exp000,
    exp_smoke,
)
from priml.baselines.speedrundit.metric import VelocityError
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.baselines.speedrundit.train_step import SpeedrunDiTTrainStep
from priml.train.parallelism import NoParallel
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from pathlib import Path

    from priml.testing.experiments import ExperimentFactory


LADDER: Final[list[ExperimentFactory[SpeedrunDiTLoop]]] = [exp000, exp_smoke]
"""Every published factory, in ladder order."""


@pytest.mark.parametrize("factory", LADDER, ids=[f.__name__ for f in LADDER])
def test_every_experiment_finalizes(
    factory: ExperimentFactory[SpeedrunDiTLoop],
) -> None:
    """A recipe must resolve without a corpus or a device."""
    config = factory().copy_tree().finalize()
    assert config.study_name == "speedrundit"
    assert config.experiment_name == factory.__name__


@pytest.mark.parametrize("factory", LADDER, ids=[f.__name__ for f in LADDER])
def test_construction_reads_no_files(
    factory: ExperimentFactory[SpeedrunDiTLoop],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building and finalizing a config must touch no disk.

    A recipe that reads the corpus at config time cannot be inspected on a
    machine that has not staged it, which is most machines.
    """
    monkeypatch.chdir(tmp_path)

    def boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("config construction must not read files")

    monkeypatch.setattr(torch, "load", boom)
    _ = factory().copy_tree().finalize()


def test_the_loop_narrows_both_slots() -> None:
    """Drop the field defaults and the slots silently revert.

    ``TrainLoop.Config``'s generic parameters narrow the STATIC type; only the
    redeclared ``default_factory`` narrows the runtime default. A tree missing
    the second type-checks and then builds the library's ``TrainStep``.
    """
    config = SpeedrunDiTLoop()
    assert isinstance(config.step, SpeedrunDiTTrainStep.Config)
    assert isinstance(config.dataset, SpeedrunDiTData.Config)
    assert isinstance(config.step.model, SpeedrunDiT.Config)


def test_exp000_builds_a_train_loop_not_its_own_config_class() -> None:
    """``Makes["TrainLoop"]`` retargets ``make``; the class name is the
    config's, not the product's.
    """
    assert SpeedrunDiTLoop.parent_class is TrainLoop


def test_exp000_matches_the_reference_geometry() -> None:
    """SR-DiT-B/1 at 256px: the numbers the pinned reference reports.

    Pinned by value rather than by delta because exp000 is the control and a
    drift here silently invalidates every golden beside it.
    """
    cfg = exp000()
    model = cfg.step.model
    assert (model.channels_in, model.channels_hidden) == (32, 768)
    assert (model.image_size, model.patch_size) == (16, 1)
    assert (model.num_layers, model.heads) == (12, 12)
    assert model.num_classes == 1000
    assert model.label_embedder.dropout == 0.1
    assert model.projector_dims == (768,)
    assert model.projector_hidden == 2048
    assert cfg.dataset.batch_size == 256
    assert cfg.dataset.latent_scale == 0.3099


def test_exp000_carries_the_reference_optimizer_recipe() -> None:
    """AdamW at 1e-4 with no decay, clipped at one, averaged at 0.9999."""
    cfg = exp000().copy_tree().finalize()
    assert cfg.step.gradient_clip_norm == 1.0
    assert cfg.step.dtype_autocast is torch.bfloat16
    objective = cfg.step.objective
    assert objective.projection_coeff == 0.5
    assert objective.cls_coeff == 0.03
    assert objective.cfm_coeff == 0.05


def test_exp000_sequence_length_follows_the_latent_grid() -> None:
    """One class token plus one token per patch.

    At 256px the INVAE is f16 and the patch size is one, so the trunk sees 257
    tokens; getting this wrong resizes the position table and nothing else
    complains.
    """
    model = exp000().step.model.copy_tree().finalize()
    assert model.num_patches == 256
    assert model.seq_len == 257


def test_exp000_registers_the_velocity_metric() -> None:
    """Eval publishes a held-out velocity error."""
    cfg = exp000()
    assert isinstance(cfg.metrics_eval["velocity"], VelocityError.Config)


def test_schedule_horizon_matches_the_stop_condition() -> None:
    """A budget that differs from the stop anneals past the end or short of it."""
    for factory in LADDER:
        cfg = factory()
        assert cfg.max_steps == cfg.step.train_budget_steps, factory.__name__


def test_smoke_shrinks_size_and_not_the_recipe() -> None:
    """Smoke cuts every costly axis and leaves the recipe alone.

    Shrinking only the step count still builds the full-width model, which is
    the trap this test exists to catch.
    """
    base, smoke = exp000(), exp_smoke()
    assert smoke.step.model.channels_hidden < base.step.model.channels_hidden
    assert smoke.step.model.num_layers < base.step.model.num_layers
    assert smoke.step.model.image_size < base.step.model.image_size
    assert smoke.step.model.heads < base.step.model.heads
    assert smoke.dataset.batch_size < base.dataset.batch_size
    assert smoke.max_steps < base.max_steps
    # Untouched: the objective's weights, the optimizer, the routing policy.
    assert smoke.step.objective.projection_coeff == base.step.objective.projection_coeff
    assert smoke.step.objective.cfm_coeff == base.step.objective.cfm_coeff
    assert smoke.step.gradient_clip_norm == base.step.gradient_clip_norm
    assert smoke.dataset.latent_scale == base.dataset.latent_scale


def test_smoke_keeps_the_sparse_stage() -> None:
    """A trunk short enough to drop the sparse stage stops covering routing.

    With two encoder and two decoder layers reserved, five layers is the
    shortest trunk that still runs a block on the routed subset -- which is
    where the mask token and the fusion projection get their only gradient.
    """
    cfg = exp_smoke().step.model.copy_tree().finalize()
    assert cfg.sprint is not None
    dense = cfg.sprint.num_encoder_layers + cfg.sprint.num_decoder_layers
    assert cfg.num_layers > dense


def test_smoke_runs_without_autocast_or_compile() -> None:
    """Both are CPU traps: bf16 autocast hits a transposed-matmul cliff and
    compile charges Dynamo tracing for a four-step run.
    """
    cfg = exp_smoke()
    assert cfg.step.dtype_autocast is None
    assert cfg.step.compile is None


def test_forks_do_not_mutate_their_parent() -> None:
    """A factory returns a fresh tree; a fork editing its parent would make
    the ladder order-dependent.
    """
    first = exp000()
    _ = exp_smoke()
    second = exp000()
    assert first.step.model.channels_hidden == second.step.model.channels_hidden
    assert first.max_steps == second.max_steps


def test_smoke_is_marked_as_not_a_result() -> None:
    """Nothing measured at smoke size is comparable with exp000."""
    assert "Not a result" in (exp_smoke.__doc__ or "")


def test_published_experiments_document_themselves() -> None:
    """Hypothesis, References and Results are required on a published rung."""
    doc = exp000.__doc__ or ""
    for section in ("Hypothesis:", "References:", "Results:"):
        assert section in doc, section
    assert "c24c2ff25699cce63174ca56c2afcfeeb225e367" in doc


def test_the_module_pins_the_reference_commit() -> None:
    """The goldens are only meaningful against a named commit."""
    assert "c24c2ff25699cce63174ca56c2afcfeeb225e367" in (experiments.__doc__ or "")


def _configs() -> list[type]:
    """Collect every Config class this baseline defines."""
    found: list[type] = [SpeedrunDiTLoop]
    for name in ("data", "loss", "metric", "model", "sampler", "train_step"):
        module = importlib.import_module(f"priml.baselines.speedrundit.{name}")
        for value in cast(dict[str, object], vars(module)).values():
            config = cast(object, getattr(value, "Config", None))
            if (
                isinstance(value, type)
                and value.__module__ == module.__name__
                and isinstance(config, type)
            ):
                found.append(config)
    return found


CONFIGS: Final = _configs()


@pytest.mark.parametrize("config", CONFIGS, ids=[c.__qualname__ for c in CONFIGS])
def test_every_field_annotation_resolves_at_runtime(config: type) -> None:
    """``--override`` resolves each node's annotations as it walks the path.

    A type imported only under ``TYPE_CHECKING`` passes every static check
    and then raises ``NameError`` the first time a launch overrides a field
    beneath that node.
    """
    assert typing.get_type_hints(config)


def test_the_launcher_can_override_the_step() -> None:
    """Where the step runs is environment, which ``--override`` exists for."""
    config = exp_smoke()
    apply_overrides(config, ["step.parallelism.device=cpu"])
    assert isinstance(config.step.parallelism, NoParallel.Config)
    assert config.step.parallelism.device == "cpu"


def test_the_smoke_corpus_command_matches_the_smoke_model() -> None:
    """The documented preparation must write what ``exp_smoke`` reads.

    ``prepare_data``'s own defaults are exp000's geometry, so a smoke run
    over a corpus prepared without these flags fails at the first forward.
    """
    pairs = cast(
        list[tuple[str, str]],
        re.findall(r"--([a-z-]+) (\d+)", exp_smoke.__doc__ or ""),
    )
    flags = dict(pairs)
    model = exp_smoke().step.model
    assert int(flags["latent-size"]) == model.image_size
    assert int(flags["latent-channels"]) == model.channels_in
    assert int(flags["num-classes"]) == model.num_classes
    assert int(flags["encoder-width"]) == model.projector_dims[0]
    assert int(flags["samples"]) >= exp_smoke().dataset.batch_size


@pytest.mark.compute_large_fixture
def test_exp000_matches_its_golden_config() -> None:
    """The whole finalized tree, frozen.

    Marked rather than shrunk: the claim IS the tree exp000 builds, and every
    lever that would make the render cheap pins a config no run uses.
    """
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
