"""Replay the port against the values captured from the reference.

``testdata/reference*.pt`` hold what the pristine MLS-Bench script produced,
written by ``scripts/capture_reference.py``, which also compares the port with
the script live and in full: every parameter of the ResNet-56 and every pixel
of the loaders. These tests replay the pieces whose values fit in a golden.
Equality is exact; there is no tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import warnings

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import Cifar100Data, normalize
from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop, exp000
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import (
    BasicBlock,
    GlobalAvgPool,
    ResNet,
)
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
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


def test_the_pipelines_match_the_reference_on_fixed_images() -> None:
    record = reference.transform_record()
    for name in ("train", "test"):
        expected = read_tensors(_DATA / f"reference_{name}_pipeline.pt")
        assert_exact(expected, {f"transform/{name}": record[f"transform/{name}"]})


def test_the_loader_shuffles_and_augments_like_the_reference(tmp_path: Path) -> None:
    assert_exact(stored("loader/"), reference.loader_record(tmp_path))


def test_the_port_reaches_the_same_random_call_sites(tmp_path: Path) -> None:
    expected = capture_reference.read_sites(_DATA)["port"]
    assert expected
    assert reference.draw_sites(tmp_path) == expected


def test_the_reference_reached_its_random_call_sites() -> None:
    sides = capture_reference.read_sites(_DATA)
    assert sides.keys() == {"reference", "port", "reference-fmnist", "port-fmnist"}
    assert capture_reference.check_sites(sides["reference"], sides["port"]) == []
    assert (
        capture_reference.check_sites(sides["reference-fmnist"], sides["port-fmnist"])
        == []
    )


@pytest.mark.parametrize(
    "name",
    [
        "reference.pt",
        "reference_train_pipeline.pt",
        "reference_test_pipeline.pt",
        "reference_settings.pt",
        "reference_fmnist_train_pipeline.pt",
        "reference_fmnist_test_pipeline.pt",
    ],
)
def test_captured_goldens_fit_the_size_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= capture_reference.CEILING


def stored_setting(name: str, suffix: str) -> dict[str, torch.Tensor]:
    """Return the captured values of one setting whose key ends in ``suffix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    prefix = f"settings/{name}/"
    return {
        k: v for k, v in record.items() if k.startswith(prefix) and k.endswith(suffix)
    }


def test_every_setting_this_script_runs_is_captured() -> None:
    assert set(reference.CAPTURED) == set(reference.SETTINGS) - {"resnet110-cifar100"}
    assert "resnet20-cifar10" in reference.CAPTURED


@pytest.mark.parametrize("name", reference.CAPTURED)
def test_each_settings_network_builds_and_runs_like_the_reference(
    name: str,
) -> None:
    expected = {
        k: v
        for k, v in read_tensors(_DATA / "reference_settings.pt").items()
        if k.startswith(f"settings/{name}/") and "/loader/" not in k
    }
    assert set(expected) == {
        f"settings/{name}/{key}" for key in ("draws", "train", "eval")
    }
    assert_exact(expected, reference.setting_record(name))


@pytest.mark.parametrize("name", reference.CAPTURED)
def test_each_settings_loader_shuffles_and_augments_like_the_reference(
    name: str,
    tmp_path: Path,
) -> None:
    expected = {
        k: v
        for k, v in read_tensors(_DATA / "reference_settings.pt").items()
        if k.startswith(f"settings/{name}/loader/")
    }
    assert_exact(expected, reference.setting_loader_record(tmp_path, name))


def test_a_setting_record_can_build_a_tasks_own_experiment_under_its_own_key() -> None:
    def resnet110_with_instance_norm() -> VisionTrainLoop:
        cfg = reference.SETTINGS["resnet110-cifar100"].experiment()
        assert isinstance(cfg.step.model, ResNet.Config)
        cfg.step.model.norm = torch.nn.InstanceNorm2d
        return cfg

    record = reference.setting_record(
        "resnet110-cifar100",
        resnet110_with_instance_norm,
        key="resnet110-cifar100/instance",
    )
    assert set(record) == {
        f"settings/resnet110-cifar100/instance/{part}"
        for part in ("draws", "train", "eval")
    }
    pristine = reference.setting_record("resnet110-cifar100")
    assert not torch.equal(
        record["settings/resnet110-cifar100/instance/train"],
        pristine["settings/resnet110-cifar100/train"],
    )


def test_a_task_settings_record_holds_each_variant_and_each_non_cifar100_loader(
    tmp_path: Path,
) -> None:
    fmnist = reference.SETTINGS["mobilenetv2-fmnist"].experiment
    record = reference.task_settings_record(
        {"mobilenetv2-fmnist": {"a": fmnist, "b": fmnist}},
        tmp_path,
    )
    prefix = "settings/mobilenetv2-fmnist/"
    networks = {k for k in record if not k.startswith(f"{prefix}loader/")}
    assert networks == {
        f"{prefix}{variant}/{part}"
        for variant in ("a", "b")
        for part in ("draws", "train", "eval")
    }
    loader = {k: v for k, v in record.items() if k.startswith(f"{prefix}loader/")}
    assert_exact(
        loader,
        reference.setting_loader_record(tmp_path, "mobilenetv2-fmnist"),
    )


def test_a_task_settings_record_loads_nothing_for_a_cifar100_setting(
    tmp_path: Path,
) -> None:
    # ResNet-110 is the cheapest CIFAR-100 setting to build.
    record = reference.task_settings_record(
        {
            "resnet110-cifar100": {
                "a": reference.SETTINGS["resnet110-cifar100"].experiment,
            },
        },
        tmp_path,
    )
    assert set(record) == {
        f"settings/resnet110-cifar100/a/{part}" for part in ("draws", "train", "eval")
    }


@pytest.mark.parametrize("name", ["vgg16bn-cifar100", "mobilenetv2-fmnist"])
def test_the_tiny_step_of_a_setting_ends_in_the_widths_the_reference_hardcodes(
    name: str,
) -> None:
    step = reference.tiny_step(
        reference.SETTINGS[name].experiment,
        blocks=(1, 1, 1),
        classes=reference.CLASSES,
        epochs=reference.EPOCHS,
    )
    model = step.model
    logits = reference.logits_of(model, reference.fixed_media(0))
    assert logits.shape == (reference.PROBE_SHAPE[0], reference.CLASSES)
    if isinstance(model, VGG16BN):
        convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        narrow = reference.TINY_WIDTHS[512]
        assert [conv.out_channels for conv in convs] == [4, narrow]
        head = [m for m in model.head if isinstance(m, torch.nn.Linear)]
        assert [(m.in_features, m.out_features) for m in head] == [
            (narrow, narrow),
            (narrow, reference.CLASSES),
        ]
    else:
        assert isinstance(model, MobileNetV2)
        assert len(model.blocks) == 2
        stem = cast(torch.nn.Conv2d, model.stem[0])
        last = cast(torch.nn.Conv2d, model.last[0])
        assert stem.out_channels == reference.TINY_WIDTHS[32]
        assert (last.in_channels, last.out_channels) == (
            reference.TINY_WIDTHS[320],
            reference.TINY_WIDTHS[1280],
        )


def test_tiny_model_shrinks_each_network_config_in_place() -> None:
    resnet, vgg, mobilenet = (
        exp000().step.model,
        reference.SETTINGS["vgg16bn-cifar100"].experiment().step.model,
        reference.SETTINGS["mobilenetv2-fmnist"].experiment().step.model,
    )
    for model in (resnet, vgg, mobilenet):
        reference.tiny_model(model)
    assert isinstance(resnet, ResNet.Config)
    widths = reference.TINY_WIDTHS
    assert resnet.blocks_per_stage == (1, 1, 1)
    assert resnet.channels_hidden == (widths[16], widths[32], widths[64])
    assert isinstance(vgg, VGG16BN.Config)
    assert (vgg.channels_hidden, vgg.channels_hidden_head) == (
        ((4,), (widths[512],)),
        widths[512],
    )
    assert isinstance(mobilenet, MobileNetV2.Config)
    assert (mobilenet.channels_hidden, mobilenet.blocks_per_stage) == (
        (4, widths[320]),
        (1, 1),
    )
    assert (mobilenet.channels_stem, mobilenet.channels_last) == (
        widths[32],
        widths[1280],
    )
    assert (mobilenet.expand_ratios, mobilenet.strides) == ((1, 1), (2, 2))


def test_the_tiny_step_builds_the_resnet_depth_it_is_given() -> None:
    step = reference.tiny_step(
        exp000,
        blocks=(2, 1, 1),
        classes=reference.CLASSES,
        epochs=reference.EPOCHS,
    )
    model = step.model
    assert isinstance(model, ResNet)
    assert [len(stage) for stage in model.stages] == [2, 1, 1]


def test_tiny_model_takes_a_resnets_depth() -> None:
    model = exp000().step.model
    reference.tiny_model(model, blocks=(2, 1, 3))
    assert isinstance(model, ResNet.Config)
    assert model.blocks_per_stage == (2, 1, 3)


def test_tiny_model_refuses_a_network_it_cannot_shrink() -> None:
    with pytest.raises(TypeError, match="GlobalAvgPool"):
        reference.tiny_model(GlobalAvgPool.Config())


def test_dataset_batches_runs_the_loaders_of_the_config_it_is_given(
    tmp_path: Path,
) -> None:
    config = Cifar100Data.Config()
    config.mean = (0.25, 0.5, 0.125)
    config.std = (0.5, 0.25, 2.0)
    media, label = reference.synthetic_split()
    train, evaluation = reference.dataset_batches(
        config,
        tmp_path,
        split=(media, label),
    )
    assert len(train) == len(evaluation) == reference.LOADER_PASSES
    assert torch.equal(
        torch.cat([batch["media"] for batch in evaluation[0]]),
        normalize(media, mean=torch.tensor(config.mean), std=torch.tensor(config.std)),
    )
    labels = torch.cat([batch["label"] for batch in train[1]])
    assert labels.sort().values.tolist() == label.tolist()


def test_the_settings_record_joins_the_fmnist_pipelines_and_each_captured_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cheapest setting to build; the others replay above.
    monkeypatch.setattr(reference, "CAPTURED", ("mobilenetv2-fmnist",))
    record = reference.settings_record(tmp_path)
    stored = read_tensors(_DATA / "reference_settings.pt")
    expected = {
        k: v for k, v in stored.items() if k.startswith("settings/mobilenetv2-fmnist/")
    }
    for name in ("train", "test"):
        expected |= read_tensors(_DATA / f"reference_fmnist_{name}_pipeline.pt")
    assert_exact(expected, record)


def test_the_fmnist_pipelines_match_the_reference_on_fixed_images() -> None:
    record = reference.fmnist_transform_record()
    for name in ("train", "test"):
        key = f"transform/fmnist/{name}"
        expected = read_tensors(_DATA / f"reference_fmnist_{name}_pipeline.pt")
        assert_exact(expected, {key: record[key]})


def test_the_fmnist_path_reaches_the_same_random_call_sites(tmp_path: Path) -> None:
    expected = capture_reference.read_sites(_DATA)["port-fmnist"]
    assert expected
    assert reference.fmnist_draw_sites(tmp_path) == expected


def test_the_settings_golden_sees_a_change_of_one_ulp() -> None:
    expected = stored_setting("vgg16bn-cifar100", "/eval")
    (key,) = expected
    bumped = expected[key].clone()
    bumped.view(-1)[0] = torch.nextafter(bumped.view(-1)[0], torch.tensor(torch.inf))
    assert mismatches(expected, {key: bumped}) == [f"{key}: 1/{bumped.numel()} differ"]


def test_the_comparison_sees_a_change_of_one_ulp() -> None:
    expected = stored("best/accuracy")
    bumped = {
        k: torch.nextafter(v, torch.tensor(torch.inf, dtype=v.dtype))
        for k, v in expected.items()
    }
    assert mismatches(expected, bumped) == ["best/accuracy: 1/1 differ"]


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
    record = reference.decode_batches([[batch]], config=config, source=source)
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
        reference.decode_batches(
            [[batch]],
            config=Cifar100Data.Config(),
            source=reference.synthetic_split()[0],
        )


class _WidthKeyedBlock(BasicBlock.Config):
    """A basic block template that keys on one of the reference's widths."""

    reset_width: int = 16


def test_tiny_model_narrows_the_width_a_block_keys_on() -> None:
    model = exp000().step.model
    assert isinstance(model, ResNet.Config)
    model.block = block = _WidthKeyedBlock()
    reference.tiny_model(model)
    assert block.reset_width == reference.TINY_WIDTHS[16]


def test_script_workers_silence_only_the_worker_count_warning() -> None:
    worker_count = "This DataLoader will create 4 worker processes in total."
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with reference.script_workers():
            warnings.warn(worker_count, stacklevel=1)
            warnings.warn("another warning", stacklevel=1)
        warnings.warn(worker_count, stacklevel=1)
    assert [str(w.message) for w in seen] == ["another warning", worker_count]


def test_full_size_media_is_two_reference_sized_images_in_eighths() -> None:
    media = reference.full_size_media()
    assert media.shape == (2, 3, 32, 32)
    assert media.flatten()[:3].tolist() == [-1.0, -0.375, 0.25]
    assert bool(((media * 8) % 1 == 0).all())
    assert media.min().item() == -1.0
    assert media.max().item() == 1.0


def test_seeding_is_scoped_to_the_block() -> None:
    before = torch.get_rng_state()
    with reference.seeded(0):
        inside = torch.rand(1)
    assert torch.equal(torch.get_rng_state(), before)
    with reference.seeded(0):
        assert torch.equal(torch.rand(1), inside)


def test_the_tiny_network_narrows_each_reference_width() -> None:
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
    ] == [reference.TINY_WIDTHS[w] for w in (16, 32, 64)]


def test_the_tiny_network_is_built_on_the_cpu_whatever_the_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Meta stands in for an accelerator a host makes torch's default.
    monkeypatch.setattr(torch, "get_default_device", lambda: torch.device("meta"))
    step = reference.tiny_step(
        exp000,
        blocks=(1, 1, 1),
        classes=reference.CLASSES,
        epochs=reference.EPOCHS,
    )
    assert {p.device.type for p in step.model.parameters()} == {"cpu"}


def test_the_record_joins_every_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake(name: str) -> Callable[..., dict[str, torch.Tensor]]:
        def record(*_args: object) -> dict[str, torch.Tensor]:
            return {name: torch.zeros(1)}

        return record

    components = (
        "schedule_record",
        "optimizer_record",
        "network_record",
        "draws_record",
        "best_record",
        "transform_record",
        "loader_record",
    )
    for name in components:
        monkeypatch.setattr(reference, name, fake(name))
    assert set(reference.port_record(tmp_path)) == set(components)


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


def test_scripted_evaluations_are_not_monotone() -> None:
    assert len(reference.SCRIPTED_CORRECT) == 200
    assert max(reference.SCRIPTED_CORRECT) > reference.SCRIPTED_CORRECT[-1]
    assert max(reference.SCRIPTED_CORRECT) < reference.SCRIPTED_TOTAL


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
