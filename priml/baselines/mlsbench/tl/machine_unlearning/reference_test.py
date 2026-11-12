"""Replay the ``security-machine-unlearning`` port against the harness's captured values.

``testdata/reference.pt`` holds what the pristine harness, its method and each
baseline's edit produced, written by ``scripts/capture_reference.py``. Equality
is exact.
"""

from __future__ import annotations

from functools import cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

import itertools

from configgle import InlineConfig

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.scripts.capture_reference import (
    CEILING,
    shared_scores,
)
from priml.baselines.mlsbench.tl.machine_unlearning import reference, upstream
from priml.baselines.mlsbench.tl.machine_unlearning.experiments import exp000
from priml.baselines.mlsbench.tl.machine_unlearning.scripts.capture_reference import (
    REQUIRED_PORT,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL import Image
    from torch import Tensor, nn
    from torch.optim import Optimizer

    from priml.baselines.mlsbench.tl.machine_unlearning.methods import Batch
    from priml.baselines.mlsbench.tl.machine_unlearning.unlearning import (
        MachineUnlearning,
    )


_CWD: Final[Path] = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"


@cache
def stored(prefix: str = "") -> dict[str, Tensor]:
    """Return the captured values whose key starts with ``prefix``."""
    record = stored() if prefix else read_tensors(_DATA / "reference.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


def assert_exact(expected: dict[str, Tensor], actual: dict[str, Tensor]) -> None:
    """Require two records to agree in every key, dtype, shape and bit."""
    assert expected
    assert mismatches(expected, actual) == []


def test_exp000_holds_the_launch_scripts_recipe() -> None:
    assert_exact(stored("recipe/"), reference.recipe_record())


def test_the_forget_split_matches_the_harness() -> None:
    assert_exact(stored("split/"), reference.split_record())


def test_the_evaluation_and_auc_match_the_harness() -> None:
    assert_exact(stored("evaluation/"), reference.evaluation_record())


def test_no_stored_auc_ranks_a_score_both_sets_hold() -> None:
    member, nonmember = reference.fixed_scores()
    record = stored("main/") | {
        "fixed/member": torch.from_numpy(member),
        "fixed/nonmember": torch.from_numpy(nonmember),
    }
    pairs = {"fixed/member": "fixed/nonmember"} | {
        f"main/{name}/forget_train_confidence": f"main/{name}/forget_test_confidence"
        for name in reference.VARIANTS
    }
    assert shared_scores(record, pairs=pairs) == []


@pytest.mark.parametrize(
    "key",
    [
        "resnet20-cifar10-class0",
        "mobilenetv2-fmnist-class0",
        # Full-width VGG-16 takes half a second on one CPU thread.
        pytest.param(
            "vgg16bn-cifar100-class0",
            marks=pytest.mark.compute_large_fixture,
        ),
    ],
)
def test_each_settings_network_initializes_and_runs_like_the_harness(key: str) -> None:
    expected = stored(f"full/{key}/") | stored(f"settings/{key}/")
    assert len(expected) == 3
    assert_exact(expected, reference.full_size_record(key)[0])


def test_the_full_values_run_each_mode_from_the_seed() -> None:
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.5))
    draws = torch.arange(2)
    record = reference.full_size_values(model, draws, key="k")
    assert record.keys() == {"full/k/draws", "full/k/train", "full/k/eval"}
    assert record["full/k/draws"] is draws
    media = vision_reference.full_size_media().flatten(1)
    assert torch.equal(record["full/k/eval"], media)
    with vision_reference.seeded(vision_reference.SEED):
        mask = torch.nn.functional.dropout(torch.ones_like(media), 0.5)
    assert torch.equal(record["full/k/train"], media * mask)
    assert not model.training


@pytest.mark.parametrize("name", list(reference.VARIANTS))
def test_three_updates_of_each_method_match_the_harness(name: str) -> None:
    assert_exact(stored(f"methods/{name}/"), reference.method_lockstep(name)[0])


def test_the_method_capture_keeps_replayable_student_and_teacher_states() -> None:
    record, states = reference.method_lockstep("scrub")
    assert len(states) == len(reference.SCHEDULE) + 1
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        model = reference.tiny_model(reference.VARIANTS["scrub"])
        assert_exact(dict(model.state_dict()), states[-1])
        for index, state in enumerate(states[:-1]):
            model.load_state_dict(state)
            assert torch.equal(
                reference.probe(model),
                record["methods/scrub/probe"][index],
            )
            assert all(value.requires_grad is False for value in state.values())
    for before, after in itertools.pairwise(states[:-1]):
        assert mismatches(before, after)
        assert all(before[key].data_ptr() != after[key].data_ptr() for key in before)


@pytest.mark.parametrize(
    "name",
    [
        # SCRUB reaches every path of a run: a teacher, both phases, the
        # recorded optimizer. The rest repeat those paths at half a second each.
        "scrub",
        *(
            pytest.param(name, marks=pytest.mark.compute_training)
            for name in reference.VARIANTS
            if name != "scrub"
        ),
    ],
)
def test_a_whole_run_of_each_variant_matches_the_harness(
    name: str,
    tmp_path: Path,
) -> None:
    trace = reference.port_main(name, tmp_path)
    assert trace.initial
    assert trace.pretrained.keys() == trace.initial.keys()
    assert mismatches(trace.initial, trace.pretrained)
    previous = trace.pretrained
    for step in trace.steps:
        assert step.state.keys() == trace.initial.keys()
        assert mismatches(previous, step.state)
        assert all(value.requires_grad is False for value in step.state.values())
        assert all(
            previous[key].data_ptr() != step.state[key].data_ptr() for key in previous
        )
        previous = step.state
    roles = {"scrub": ("teacher",), "bad_teacher": ("competent", "incompetent")}
    assert set(trace.teachers) == set(roles.get(name, ()))
    for role, state in trace.teachers.items():
        assert state.keys() == trace.pretrained.keys()
        if role in ("teacher", "competent"):
            assert_exact(trace.pretrained, state)
        else:
            assert mismatches(trace.pretrained, state)
    for site in REQUIRED_PORT:
        assert any(
            key.startswith(site) and count > 0 for key, count in trace.sites.items()
        )
    assert_exact(stored(f"main/{name}/"), reference.main_record(name, trace))


def returning(name: str) -> Callable[[], dict[str, Tensor]]:
    """Return a stand-in for a record function: ``name`` mapped to zero."""
    return lambda: {name: torch.zeros(1)}


def fake_lockstep(name: str) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
    """Stand in for :func:`reference.method_lockstep`: one key per variant."""
    return {f"methods/{name}": torch.zeros(1)}, []


def fake_full_size(key: str) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Stand in for :func:`reference.full_size_record`: one key per setting."""
    return {key: torch.zeros(1)}, {"w": torch.ones(1)}


class CaptureInputs:
    """Observe the variant, directory and trace passed between capture components."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.traces = {name: reference.MainTrace() for name in reference.VARIANTS}
        self.captured: list[str] = []
        self.reduced: list[str] = []

    def port_main(self, name: str, directory: Path) -> reference.MainTrace:
        """Return a distinct trace after checking the capture inputs."""
        assert directory == self.directory
        self.captured.append(name)
        return self.traces[name]

    def main_record(self, name: str, trace: reference.MainTrace) -> dict[str, Tensor]:
        """Check that the reducer receives this variant's captured trace."""
        assert trace is self.traces[name]
        self.reduced.append(name)
        return {name: trace.draws}


def test_the_method_record_joins_every_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(reference, "method_lockstep", fake_lockstep)
    assert set(reference.method_record()) == {
        f"methods/{name}" for name in reference.VARIANTS
    }


def test_the_port_record_joins_every_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for name in ("recipe_record", "split_record", "evaluation_record", "method_record"):
        monkeypatch.setattr(reference, name, returning(name))
    monkeypatch.setattr(reference, "full_size_record", fake_full_size)
    capture = CaptureInputs(tmp_path)
    monkeypatch.setattr(reference, "port_main", capture.port_main)
    monkeypatch.setattr(reference, "main_record", capture.main_record)
    record, states = reference.port_record(tmp_path)
    assert capture.captured == list(reference.VARIANTS)
    assert capture.reduced == list(reference.VARIANTS)
    assert all(record[name] is trace.draws for name, trace in capture.traces.items())
    assert set(record) == {
        "recipe_record",
        "split_record",
        "evaluation_record",
        "method_record",
        *reference.SETTINGS,
        *reference.VARIANTS,
    }
    assert set(states) == set(reference.SETTINGS)


class StepDependent:
    """Update real model parameters using both coordinates of the method protocol."""

    def unlearn_step(
        self,
        model: nn.Module,
        retain_batch: Batch,
        forget_batch: Batch,
        optimizer: Optimizer,
        *,
        step: int,
        epoch: int,
    ) -> Tensor:
        """Descend a step-dependent retained loss and an epoch-dependent forget loss.

        Args:
          model: Student whose parameters the optimizer owns.
          retain_batch: Retained media and labels.
          forget_batch: Forget media and labels.
          optimizer: Real optimizer applied after backpropagation.
          step: Multiplier of the retained loss, minus one.
          epoch: Multiplier of the forget loss, minus one.

        Returns:
          loss: Detached scalar from before the update.

        """
        retain_x, retain_y = retain_batch
        forget_x, forget_y = forget_batch
        retain_loss = torch.nn.functional.cross_entropy(
            vision_reference.model_logits(model, retain_x),
            retain_y,
        )
        forget_loss = torch.nn.functional.cross_entropy(
            vision_reference.model_logits(model, forget_x),
            forget_y,
        )
        loss = (step + 1) * retain_loss + (epoch + 1) * forget_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach()


def step_dependent_experiment() -> MachineUnlearning.Config:
    """Inject a valid updating method without changing the experiment's recipe."""
    cfg = exp000()
    model = cfg.pretrain.model
    assert isinstance(model, ResNet.Config)
    model.channels_hidden = (16, 32)
    model.blocks_per_stage = (1, 1)
    cfg.method = InlineConfig(StepDependent)
    return cfg


def test_the_recorder_passes_step_to_a_real_update_and_copies_the_post_state() -> None:
    with vision_reference.seeded(vision_reference.SEED):
        model = torch.nn.Linear(3, 4)
    optimizer = exp000().unlearn_optimizer.make()(model.parameters())
    retain = (torch.arange(6).reshape(2, 3).float() / 8, torch.tensor([0, 2]))
    forget = (torch.arange(15).reshape(5, 3).float() / 8, torch.tensor([1, 3, 0, 2, 1]))
    steps: list[reference.Step] = []
    recorded = reference.Recorded(StepDependent(), steps)
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for epoch, step in ((2, 3), (4, 5)):
        loss = recorded.unlearn_step(
            model,
            retain,
            forget,
            optimizer,
            step=step,
            epoch=epoch,
        )
        captured = steps[-1]
        assert captured.loss == float(loss)
        assert captured.position == (epoch, step)
        assert all(
            actual is expected
            for actual, expected in zip(
                captured.batches,
                (*retain, *forget),
                strict=True,
            )
        )
        assert_exact(dict(model.state_dict()), captured.state)
        assert mismatches(initial, captured.state)
    first = {key: value.clone() for key, value in steps[0].state.items()}
    assert mismatches(steps[0].state, steps[1].state)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    assert_exact(first, steps[0].state)
    assert mismatches(dict(model.state_dict()), steps[-1].state)


def test_lockstep_runs_an_injected_step_dependent_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(reference.VARIANTS, "step_dependent", step_dependent_experiment)
    monkeypatch.setattr(
        vision_reference,
        "tiny_model",
        partial(vision_reference.tiny_model, blocks=(1, 1)),
    )
    monkeypatch.setattr(vision_reference, "PROBE_SHAPE", (2, 3, 4, 5))
    for width, tiny_width in zip((16, 32, 64), (2, 3, 4), strict=True):
        monkeypatch.setitem(vision_reference.TINY_WIDTHS, width, tiny_width)
    record, states = reference.method_lockstep("step_dependent")
    expected_states: list[dict[str, Tensor]] = []
    expected_losses: list[float] = []
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        model = reference.tiny_model(step_dependent_experiment)
        optimizer = exp000().unlearn_optimizer.make()(model.parameters())
        method = StepDependent()
        for index, (epoch, step) in enumerate(reference.SCHEDULE):
            retain, forget = reference.fixed_batches(index)
            loss = method.unlearn_step(
                model,
                retain,
                forget,
                optimizer,
                step=step,
                epoch=epoch,
            )
            expected_losses.append(float(loss))
            expected_states.append(
                {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                },
            )
    assert torch.equal(
        record["methods/step_dependent/loss"],
        torch.tensor(expected_losses, dtype=torch.float64),
    )
    assert len(states) == len(expected_states)
    for expected, actual in zip(expected_states, states, strict=True):
        assert_exact(expected, actual)


def pillow_pixels(image: Image.Image) -> Tensor:
    """Read the actual upstream adapter's Pillow image without changing its pixels."""
    assert image.mode == "RGB"
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


@pytest.mark.parametrize("train", [True, False])
def test_each_synthetic_split_crosses_the_upstream_pillow_boundary(train: bool) -> None:
    media, label = reference.synthetic_dataset(train=train)
    expected_labels = reference.TRAIN_LABELS if train else reference.TEST_LABELS
    # upstream._FakeCifar10 requires CIFAR's RGB 32x32 image schema.
    assert media.shape == (len(expected_labels), 3, 32, 32)
    assert media.dtype == torch.uint8
    assert label.dtype == torch.int64
    assert torch.equal(label, torch.tensor(expected_labels))
    dataset = upstream._FakeCifar10(
        root="",
        train=train,
        download=False,
        transform=pillow_pixels,
    )
    assert len(dataset) == len(expected_labels)
    for index, expected_label in enumerate(expected_labels):
        pixels, actual_label = dataset[index]
        assert pixels.dtype == torch.uint8
        assert torch.equal(pixels, media[index])
        assert actual_label == expected_label


def test_the_end_to_end_run_passes_the_harness_its_tiny_flags() -> None:
    assert reference.tiny_flags() == [
        "--forget-class",
        str(reference.FORGET_CLASS),
        "--pretrain-epochs",
        str(reference.TINY_PRETRAIN_EPOCHS),
        "--unlearn-epochs",
        str(reference.TINY_UNLEARN_EPOCHS),
        "--batch-size",
        str(reference.TINY_BATCH),
        "--seed",
        str(vision_reference.SEED),
    ]


def test_the_golden_fits_the_ceiling() -> None:
    assert (_DATA / "reference.pt").stat().st_size <= CEILING


def test_the_golden_bites_on_one_ulp() -> None:
    key = "methods/scrub/probe"
    value = stored(key)[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(torch.inf))
    port = {key: reference.method_lockstep("scrub")[0][key]}
    assert mismatches({key: value}, port) == [f"{key}: 1/18 differ"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
