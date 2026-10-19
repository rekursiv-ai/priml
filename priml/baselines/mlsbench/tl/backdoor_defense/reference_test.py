"""Replay the ``security-backdoor-defense`` port against the values captured from its driver.

``testdata/reference*.pt`` hold what the pristine driver and each baseline's
defense produced, written by ``scripts/capture_reference.py``. Equality is
exact; there is no tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision
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


def test_activation_clusterings_cut_falls_among_tied_scores() -> None:
    record = stored("filter/activation_clustering/")
    scores = record["filter/activation_clustering/scores"]
    removed = reference.MAIN_IMAGES - len(record["filter/activation_clustering/kept"])
    assert (scores == scores.max()).sum() > removed


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
    assert len(states) == vision.STEPS
    first, second = (state["head.weight"] for state in states)
    assert not torch.equal(first, second)


def test_decoding_rejects_an_image_that_is_no_crop_of_any_source() -> None:
    batch = {
        # One 32x32 color image, the size every crop has.
        "media": torch.full((1, 3, 32, 32), 9.0),
        "label": torch.zeros(1, dtype=torch.int64),
    }
    with pytest.raises(ValueError, match="no crop"):
        reference.decode_record(reference.FIRST, ([[batch]], []))


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
    monkeypatch.setattr(vision, "setting_record", fake("setting_record"))
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
    assert_exact(
        stored(f"loader/{setting}/"),
        reference.decode_record(setting, reference.loader_batches(setting, tmp_path)),
    )


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("setting", list(reference.SETTINGS))
def test_each_full_network_initializes_and_runs_as_the_driver(setting: str) -> None:
    spec = reference.SETTINGS[setting]
    assert_exact(
        stored(f"settings/{setting}/", "reference_networks.pt"),
        vision.setting_record(spec.vision, spec.factory, key=setting),
    )


def test_the_port_reaches_the_same_random_call_sites(tmp_path: Path) -> None:
    lines = (_DATA / "reference_sites.txt").read_text().splitlines()
    expected = {
        key: int(count)
        for side, key, count in (line.split() for line in lines)
        if side == "port"
    }
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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
