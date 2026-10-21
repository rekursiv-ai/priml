"""Replay the ``security-backdoor-defense`` port against the values captured from its driver.

``testdata/reference*.pt`` hold what the pristine driver and each baseline's
defense produced, written by ``scripts/capture_reference.py``. Equality is
exact; there is no tolerance.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.data import normalize
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.tl.backdoor_defense import reference
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"


def stored(prefix: str, name: str = "reference.pt") -> dict[str, torch.Tensor]:
    """Return the captured values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / name)
    return {k: v for k, v in record.items() if k.startswith(prefix)}


def assert_exact(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    """Require two records to agree in every key, dtype, shape and bit."""
    assert expected
    assert mismatches(expected, actual) == []


def test_the_poisoned_positions_and_pythons_generator_match_the_driver() -> None:
    assert_exact(stored("poison/"), reference.poison_record())


def test_both_triggers_stamp_the_drivers_pixels() -> None:
    assert_exact(stored("trigger/"), reference.trigger_record())


def test_z_score_inputs_spread_on_every_axis_and_svd_inputs_on_one() -> None:
    spread, _, labels = reference.defense_inputs("zscore_outlier")
    axial, _, _ = reference.defense_inputs("spectral_signature")
    classes = torch.from_numpy(labels)
    for cls in classes.unique():
        mask = (classes == cls).numpy()
        varying = (torch.from_numpy(spread[mask]).std(dim=0, correction=0) > 0).sum()
        assert varying == (3 if mask.sum() > 1 else 0)
        assert (torch.from_numpy(axial[mask]).std(dim=0, correction=0) > 0).sum() <= 1


def test_every_defense_scores_as_the_drivers_variant_does() -> None:
    expected = stored("defense/")
    assert {k.split("/")[1] for k in expected} == set(reference.VARIANTS)
    assert_exact(expected, reference.defense_record())


def test_every_defense_filters_mains_features_as_the_driver_does() -> None:
    inputs = stored(f"main/{reference.FIRST}/")
    expected = stored("filter/")
    actual = reference.filter_record(inputs)
    # Which of the images tied at the cut are removed is NumPy's unstable
    # sort's choice, made by the host's SIMD kernel, so the images kept are
    # compared by their scores: exact where the cut is untied.
    for name in reference.VARIANTS:
        scores = expected[f"filter/{name}/scores"]
        want, got = (r.pop(f"filter/{name}/kept") for r in (expected, actual))
        assert torch.equal(scores[want].sort().values, scores[got].sort().values)
    # A trained network's features span every direction, so these defenses'
    # SVD and norms round by the host's OpenBLAS kernel, the driver's too: the
    # capture compares their scores live, and the defense test pins them on
    # features whose SVD and norms are exact.
    for name in ("spectral_signature", "activation_clustering"):
        for record in (expected, actual):
            del record[f"filter/{name}/scores"]
    assert_exact(expected, actual)


def test_activation_clusterings_cut_falls_among_tied_clean_scores() -> None:
    record = stored("filter/") | stored(f"main/{reference.FIRST}/is_poison")
    scores = record["filter/activation_clustering/scores"]
    removed = reference.MAIN_IMAGES - len(record["filter/activation_clustering/kept"])
    assert (scores == scores.max()).sum() > removed
    assert reference.poison_at_tied_cut(record) == []


def test_a_cut_through_a_tie_of_poisoned_and_clean_images_is_reported() -> None:
    # Images 1 and 2 tie at 2.0; the cut removes one of them.
    scores = torch.tensor([0.0, 2.0, 2.0, 1.0], dtype=torch.float64)
    record = {
        f"filter/{name}/{key}": value
        for name in reference.VARIANTS
        for key, value in (("scores", scores), ("kept", torch.tensor([0, 2, 3])))
    }
    reported = [
        f"filter/{name}: a poisoned image is tied at the cut"
        for name in reference.VARIANTS
    ]
    key = f"main/{reference.FIRST}/is_poison"
    for poison in ([0, 1, 0, 0], [0, 0, 1, 0]):
        record[key] = torch.tensor(poison, dtype=torch.uint8)
        assert reference.poison_at_tied_cut(record) == reported
    # Both tied images poisoned, or neither: either order gives one recall.
    for poison in ([0, 1, 1, 0], [0, 0, 0, 1]):
        record[key] = torch.tensor(poison, dtype=torch.uint8)
        assert reference.poison_at_tied_cut(record) == []


def test_two_updates_match_the_drivers_train_epoch() -> None:
    assert_exact(stored("lockstep/"), reference.lockstep_record())


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
def test_main_poisons_evaluates_and_schedules_as_the_driver(setting: str) -> None:
    expected = {
        k: v
        for k, v in stored(f"main/{setting}/").items()
        if k.rsplit("/", 1)[1] not in {"features", "logits", "labels"}
    }
    assert_exact(expected, reference.main_record(setting))


def test_main_evaluates_after_the_first_every_tenth_and_the_last_epoch() -> None:
    # 19 epochs: a range one past the end would add epoch 20, a tenth; the
    # last, 19, is evaluated only because it ends the run.
    record = reference.main_record(reference.FIRST, epochs=19)
    evaluated = record[f"main/{reference.FIRST}/evaluated"]
    assert evaluated.tolist() == [1, 10, 19]
    assert record[f"main/{reference.FIRST}/rates"].shape == (19,)


def test_the_lockstep_hands_back_the_state_after_each_update() -> None:
    states: list[dict[str, torch.Tensor]] = []
    _ = reference.lockstep_record(states=states)
    assert len(states) == vision_reference.STEPS
    first, second = (state["head.weight"] for state in states)
    assert not torch.equal(first, second)


@pytest.mark.parametrize(
    ("dataset", "shape"),
    [("fmnist", (1, 28, 28)), ("cifar10", (3, 32, 32)), ("cifar100", (3, 32, 32))],
)
def test_the_main_split_has_the_datasets_shape_and_distinct_images(
    dataset: str,
    shape: tuple[int, int, int],
) -> None:
    media, label = reference.main_split(dataset)
    assert media.dtype == torch.uint8
    assert media.shape == (reference.MAIN_IMAGES, *shape)
    assert label.tolist() == reference.main_labels()
    assert label.tolist()[:12] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
    assert len({tuple(image.flatten().tolist()) for image in media}) == len(media)
    assert media[0].flatten()[:3].tolist() == [0, 37, 74]
    assert media[1].flatten()[:3].tolist() == [101, 138, 175]


def test_an_optimizers_first_group_is_the_one_whose_settings_are_read() -> None:
    first, second = {"lr": 0.1}, {"lr": 0.2}
    assert reference.optimizer_group([first, second]) is first


def test_decoding_rejects_an_image_that_is_no_crop_of_any_source() -> None:
    batch = {
        # One 32x32 color image, the size every crop has.
        "media": torch.full((1, 3, 32, 32), 9.0),
        "label": torch.zeros(1, dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="no crop"):
        reference.decode_record(reference.FIRST, ([[batch]], []))


@pytest.fixture
def two_source_split(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], tuple[torch.Tensor, torch.Tensor]]:
    """Limit decoder witnesses to two real-shaped synthetic sources."""
    original = reference.synthetic
    first_two = partial(_first_two_synthetic, original)
    monkeypatch.setattr(reference, "synthetic", first_two)
    return first_two


def test_decoding_rejects_a_matching_centre_with_an_off_centre_corruption(
    two_source_split: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
) -> None:
    setting = reference.FIRST
    spec = reference.SETTINGS[setting]
    source, _ = two_source_split(spec.dataset)
    config = reference.SETTINGS[setting].factory().dataset
    pad = config.translate_pad
    height, width = source.shape[-2:]
    padded = torch.nn.functional.pad(source, (pad,) * 4)
    crop = padded[0, :, pad : pad + height, pad : pad + width]
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)
    image = normalize(crop.unsqueeze(0), mean=mean, std=std)
    image[..., 0, 0] = 9.0
    batch = {"media": image, "label": torch.zeros(1, dtype=torch.int64)}

    with pytest.raises(ValueError, match="no crop"):
        reference.decode_record(setting, ([[batch]], []))


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
def test_decoding_matches_clean_and_triggered_crops(
    setting: str,
    two_source_split: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
) -> None:
    source, source_labels = two_source_split(reference.SETTINGS[setting].dataset)
    config = reference.SETTINGS[setting].factory().dataset
    # Target-class source 0 is not selected for poisoning, but remains decodable.
    assert int(source_labels[0]) == config.target_class

    pad = config.translate_pad
    height, width = source.shape[-2:]
    padded = torch.nn.functional.pad(source, (pad,) * 4)
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)
    cases = (
        ((0, 2, 3, 1, True), (1, 1, 2, 1, False)),
        ((0, 0, 1, 0, False), (1, 3, 4, 0, True)),
        ((1, 4, 0, 1, False), (0, 1, 4, 0, True)),
    )
    batches: list[list[dict[str, torch.Tensor]]] = []
    for pass_cases in cases:
        images: list[torch.Tensor] = []
        labels: list[int] = []
        for source_id, top, left, flip, triggered in pass_cases:
            crop = padded[
                source_id,
                :,
                top : top + height,
                left : left + width,
            ]
            if flip:
                crop = crop.flip(-1)
            image = normalize(crop.unsqueeze(0), mean=mean, std=std)
            if triggered:
                image = config.trigger(image, mean=mean, std=std)
            images.append(image[0])
            labels.append(
                config.target_class if triggered else int(source_labels[source_id]),
            )
        batches.append(
            [{"media": torch.stack(images), "label": torch.tensor(labels)}],
        )

    prefix = f"loader/{setting}/"
    expected = {
        f"{prefix}label": torch.tensor(
            [
                [
                    config.target_class if triggered else int(source_labels[source_id])
                    for source_id, _, _, _, triggered in pass_cases
                ]
                for pass_cases in cases
            ],
        ),
        f"{prefix}source": torch.tensor(
            [
                [source_id for source_id, _, _, _, _ in pass_cases]
                for pass_cases in cases
            ],
        ),
        f"{prefix}top": torch.tensor(
            [[top for _, top, _, _, _ in pass_cases] for pass_cases in cases],
        ),
        f"{prefix}left": torch.tensor(
            [[left for _, _, left, _, _ in pass_cases] for pass_cases in cases],
        ),
        f"{prefix}flip": torch.tensor(
            [[flip for _, _, _, flip, _ in pass_cases] for pass_cases in cases],
        ),
    }

    assert_exact(expected, reference.decode_record(setting, (batches[:2], batches[2:])))


def test_the_port_record_joins_every_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake(name: str) -> Callable[..., dict[str, torch.Tensor]]:
        def component(*args: object, **_options: object) -> dict[str, torch.Tensor]:
            return {f"{name}/{args[:1]}": torch.zeros(1)}

        return component

    for name in (
        "poison_record",
        "trigger_record",
        "defense_record",
        "lockstep_record",
        "tiny_record",
        "main_record",
        "loader_batches",
        "decode_record",
    ):
        monkeypatch.setattr(reference, name, fake(name))
    monkeypatch.setattr(vision_reference, "setting_record", fake("setting_record"))
    record = reference.port_record(tmp_path)
    for setting in reference.SETTINGS:
        assert f"main_record/{(setting,)}" in record
        assert f"setting_record/{(reference.SETTINGS[setting].vision,)}" in record
    assert "poison_record/()" in record


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
def test_the_shallowest_network_is_built_and_measured_as_the_driver(
    setting: str,
    tmp_path: Path,
) -> None:
    assert_exact(
        stored(f"tiny/{setting}/", "reference_networks.pt"),
        reference.tiny_record(setting, tmp_path),
    )


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
def test_the_poisoned_loaders_shuffle_augment_and_trigger_as_the_driver(
    setting: str,
    tmp_path: Path,
) -> None:
    expected = _captured_loader_output(setting)
    captured_passes = reference.loader_batches(setting, tmp_path)
    actual_passes = (*captured_passes[0], *captured_passes[1])
    assert len(actual_passes) == len(expected["label"])

    actual_media: list[torch.Tensor] = []
    actual_labels: list[torch.Tensor] = []
    for batches, expected_labels in zip(
        actual_passes,
        expected["label"],
        strict=True,
    ):
        media = torch.cat([batch["media"] for batch in batches])
        labels = torch.cat([batch["label"] for batch in batches])
        assert len(media) == len(expected_labels)
        actual_media.append(media)
        actual_labels.append(labels)

    actual = {"media": torch.stack(actual_media), "label": torch.stack(actual_labels)}
    assert_exact(expected, actual)


@pytest.mark.parametrize("setting", list(reference.SETTINGS))
def test_each_shallow_setting_replay_matches_the_driver(setting: str) -> None:
    spec = reference.SETTINGS[setting]
    assert_exact(
        stored(f"settings/{setting}/", "reference_networks.pt"),
        vision_reference.setting_record(spec.vision, spec.factory, key=setting),
    )


def test_the_port_reaches_the_same_random_call_sites(tmp_path: Path) -> None:
    expected = capture_reference.read_sites(_DATA)["port"]
    assert expected
    assert reference.draw_sites(tmp_path) == expected


@pytest.mark.parametrize("name", ["reference.pt", "reference_networks.pt"])
def test_captured_goldens_fit_the_size_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= capture_reference.CEILING


def test_the_golden_bites_on_one_ulp() -> None:
    expected = stored("defense/spectral_signature/scores")
    value = expected["defense/spectral_signature/scores"].clone()
    value[3] = torch.nextafter(value[3], torch.tensor(0.0, dtype=value.dtype))
    port = {
        "defense/spectral_signature/scores": reference.defense_record()[
            "defense/spectral_signature/scores"
        ],
    }
    assert mismatches({"defense/spectral_signature/scores": value}, port) == [
        "defense/spectral_signature/scores: 1/13 differ",
    ]


def _first_two_synthetic(
    original: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    dataset: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the first two source images and labels from the original split."""
    media, labels = original(dataset)
    return media[:2], labels[:2]


def _captured_loader_output(setting: str) -> dict[str, torch.Tensor]:
    """Rebuild loader pixels from frozen decoder-inferred crop metadata."""
    prefix = f"loader/{setting}/"
    captured = stored(prefix)
    sources = captured[f"{prefix}source"]
    tops = captured[f"{prefix}top"]
    lefts = captured[f"{prefix}left"]
    flips = captured[f"{prefix}flip"]
    labels = captured[f"{prefix}label"]
    source, source_labels = reference.synthetic(reference.SETTINGS[setting].dataset)
    config = reference.SETTINGS[setting].factory().dataset
    pad = config.translate_pad
    height, width = source.shape[-2:]
    padded = torch.nn.functional.pad(source, (pad,) * 4)
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)

    expected_passes: list[torch.Tensor] = []
    for source_row, top_row, left_row, flip_row, label_row in zip(
        sources,
        tops,
        lefts,
        flips,
        labels,
        strict=True,
    ):
        crops: list[torch.Tensor] = []
        triggered_rows: list[int] = []
        for source_id, top, left, flip, label in zip(
            source_row,
            top_row,
            left_row,
            flip_row,
            label_row,
            strict=True,
        ):
            source_index = int(source_id)
            crop_top, crop_left = int(top), int(left)
            crop = padded[
                source_index,
                :,
                crop_top : crop_top + height,
                crop_left : crop_left + width,
            ]
            if int(flip):
                crop = crop.flip(-1)
            crops.append(crop)

            # The captured record has no poison flag: a changed label marks a
            # source image relabeled to this setting's target class.
            if int(label) != int(source_labels[source_index]):
                assert int(label) == config.target_class
                triggered_rows.append(len(crops) - 1)

        media = normalize(torch.stack(crops), mean=mean, std=std)
        if triggered_rows:
            media[triggered_rows] = config.trigger(
                media[triggered_rows],
                mean=mean,
                std=std,
            )
        expected_passes.append(media)

    return {"media": torch.stack(expected_passes), "label": labels}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
