"""Replay the port against the values captured from the reference.

``testdata/reference*.pt`` hold what the pristine MLS-Bench script produced,
written by ``scripts/capture_reference.py``, which also compares the port with
the script live and in full: every parameter of the ResNet-56 and of each
update, and every pixel of the loaders. These tests replay the pieces whose
values fit in a golden. Equality is exact; there is no tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import Cifar100Data, normalize
from priml.baselines.mlsbench.dl.vision.experiments import exp000
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"


def stored(prefix: str = "") -> dict[str, torch.Tensor]:
    """Return the captured values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


def assert_exact(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    """Require two records to agree in every key, dtype, shape and bit."""
    assert expected
    assert mismatches(expected, actual) == []


def test_every_loss_and_its_gradient_match_the_reference() -> None:
    assert_exact(stored("loss/"), reference.loss_record())


def test_the_learning_rate_of_all_200_epochs_matches_the_reference() -> None:
    expected = stored("schedule/")
    assert expected["schedule/lr"].shape == (200,)
    assert_exact(expected, reference.schedule_record())


def test_the_optimizer_matches_the_reference() -> None:
    assert_exact(stored("optimizer/"), reference.optimizer_record())


def test_the_best_accuracy_matches_the_reference() -> None:
    assert_exact(stored("best/"), reference.best_record())


def test_the_shallowest_network_initializes_and_runs_like_the_reference() -> None:
    assert_exact(stored("network/"), reference.network_record())


def test_the_full_network_leaves_the_generator_where_the_reference_does() -> None:
    assert_exact(stored("draws/"), reference.draws_record())


@pytest.mark.parametrize("name", list(reference.LOSSES))
def test_two_updates_with_each_loss_match_the_reference(name: str) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name))


def test_the_pipelines_match_the_reference_on_fixed_images() -> None:
    record = reference.transform_record()
    for name in ("train", "test"):
        expected = read_tensors(_DATA / f"reference_{name}_pipeline.pt")
        assert_exact(expected, {f"transform/{name}": record[f"transform/{name}"]})


def test_the_loader_shuffles_and_augments_like_the_reference(tmp_path: Path) -> None:
    assert_exact(stored("loader/"), reference.loader_record(tmp_path))


def test_the_port_reaches_the_same_random_call_sites(tmp_path: Path) -> None:
    lines = (_DATA / "reference_sites.txt").read_text().splitlines()
    expected = {
        key: int(count)
        for side, key, count in (line.split() for line in lines)
        if side == "port"
    }
    assert expected
    assert reference.draw_sites(tmp_path) == expected


def test_the_reference_reached_its_random_call_sites() -> None:
    sides: dict[str, dict[str, int]] = {"reference": {}, "port": {}}
    for line in (_DATA / "reference_sites.txt").read_text().splitlines():
        side, key, count = line.split()
        sides[side][key] = int(count)
    assert capture_reference.check_sites(sides["reference"], sides["port"]) == []


@pytest.mark.parametrize(
    "name",
    ["reference.pt", "reference_train_pipeline.pt", "reference_test_pipeline.pt"],
)
def test_captured_goldens_fit_the_size_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= capture_reference.CEILING


def test_the_comparison_sees_a_change_of_one_ulp() -> None:
    expected = stored("loss/cross_entropy/value")
    bumped = {
        k: torch.nextafter(v, torch.tensor(torch.inf)) for k, v in expected.items()
    }
    assert mismatches(expected, bumped) == ["loss/cross_entropy/value: 1/1 differ"]


def test_decoding_recovers_the_offsets_and_mirror_of_a_crop() -> None:
    source, _ = reference.synthetic_split()
    config = Cifar100Data.Config()
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)

    padded = torch.nn.functional.pad(source, (4,) * 4)
    media: list[torch.Tensor] = []
    for label, (top, left, flip) in enumerate([(0, 8, 1), (4, 4, 0), (8, 0, 1)]):
        crop = padded[label, :, top : top + 32, left : left + 32]
        media.append(
            normalize((crop.flip(-1) if flip else crop)[None], mean=mean, std=std)[0],
        )
    batch = {"media": torch.stack(media), "label": torch.arange(3)}
    record = reference.decode_batches([[batch]], translate_pad=4)
    assert record["loader/top"].tolist() == [[0, 4, 8]]
    assert record["loader/left"].tolist() == [[8, 4, 0]]
    assert record["loader/flip"].tolist() == [[1, 0, 1]]
    assert record["loader/label"].tolist() == [[0, 1, 2]]


def test_decoding_rejects_an_image_that_is_no_crop_of_its_source() -> None:
    batch = {
        # One 32x32 CIFAR image, the size of every crop of the source.
        "media": torch.zeros(1, 3, 32, 32),
        "label": torch.zeros(1, dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="not a crop"):
        reference.decode_batches([[batch]], translate_pad=4)


def test_seeding_is_scoped_to_the_block() -> None:
    before = torch.get_rng_state()
    with reference.seeded(0):
        inside = torch.rand(1)
    assert torch.equal(torch.get_rng_state(), before)
    with reference.seeded(0):
        assert torch.equal(torch.rand(1), inside)


def test_a_lockstep_run_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.lockstep_record("cross_entropy", states=states)
    assert len(states) == reference.STEPS
    assert not torch.equal(states[0]["stem.weight"], states[1]["stem.weight"])


def test_the_tiny_network_keeps_the_reference_widths() -> None:
    step = reference.tiny_step(
        exp000,
        blocks=(1, 1, 1),
        classes=reference.CLASSES,
        epochs=reference.EPOCHS,
    )
    assert [
        p.shape[0]
        for n, p in step.model.named_parameters()
        if n.endswith("conv1.weight")
    ] == [16, 32, 64]


def test_the_record_joins_every_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake(name: str) -> Callable[..., dict[str, torch.Tensor]]:
        return lambda *_: {name: torch.zeros(1)}

    for name in (
        "loss_record",
        "schedule_record",
        "optimizer_record",
        "network_record",
        "draws_record",
        "best_record",
        "transform_record",
        "loader_record",
    ):
        monkeypatch.setattr(reference, name, fake(name))

    def lockstep(name: str) -> dict[str, torch.Tensor]:
        return {f"lockstep/{name}": torch.zeros(1)}

    monkeypatch.setattr(reference, "lockstep_record", lockstep)
    record = reference.port_record(tmp_path)
    assert set(record) == {
        "loss_record",
        "schedule_record",
        "optimizer_record",
        "network_record",
        "draws_record",
        "best_record",
        "transform_record",
        "loader_record",
        *(f"lockstep/{name}" for name in reference.LOSSES),
    }


def test_fixed_inputs_are_the_patterns_the_reference_was_run_on() -> None:
    logits, label = reference.fixed_logits()
    assert logits.shape == (3, 4)
    assert logits[0].tolist() == [-2.5, 0.0, 2.5, -0.5]
    assert label.tolist() == [1, 3, 0]
    assert reference.fixed_media(0).shape == reference.PROBE_SHAPE
    assert reference.fixed_media(0).flatten()[:3].tolist() == [-1.0, -0.625, -0.25]
    assert reference.fixed_media(1).flatten()[:3].tolist() == [-1.0, -0.375, 0.25]
    assert reference.fixed_label(0).tolist() == [0, 1]
    assert reference.fixed_label(2).tolist() == [2, 0]
    images = reference.fixed_images(2)
    assert (images.dtype, images.shape) == (torch.uint8, (2, 3, 32, 32))
    assert images.flatten()[:3].tolist() == [0, 37, 74]


def test_the_synthetic_split_names_the_position_of_every_pixel() -> None:
    media, label = reference.synthetic_split()
    assert media.shape == (reference.LOADER_IMAGES, 3, 32, 32)
    assert media.dtype == torch.uint8
    assert label.tolist() == list(range(reference.LOADER_IMAGES))
    assert media[3, :, 5, 9].tolist() == [6, 10, 4]
    assert torch.equal(
        media[:, 0, :, 0],
        torch.arange(1, 33).expand(reference.LOADER_IMAGES, 32),
    )
    assert torch.equal(
        media[:, 1, 0, :],
        torch.arange(1, 33).expand(reference.LOADER_IMAGES, 32),
    )
    assert (media[2, 2] == 3).all()


def test_the_loss_record_holds_a_value_and_a_gradient_per_loss() -> None:
    record = reference.loss_record()
    assert set(record) == {
        f"loss/{name}/{kind}" for name in reference.LOSSES for kind in ("value", "grad")
    }
    assert record["loss/poly/grad"].shape == (3, 4)
    assert record["loss/poly/value"].shape == ()


def test_scripted_evaluations_are_not_monotone() -> None:
    assert len(reference.SCRIPTED_CORRECT) == 200
    assert max(reference.SCRIPTED_CORRECT) > reference.SCRIPTED_CORRECT[-1]
    assert max(reference.SCRIPTED_CORRECT) < reference.SCRIPTED_TOTAL


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
