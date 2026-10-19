"""Replay the ``security-poison-robust-learning`` port against the values captured from its driver.

``testdata/reference.pt`` holds what the pristine driver and each baseline's
edit of its loss produced, written by ``scripts/capture_reference.py``, which
also compares every parameter after each update, the full-size label flips and
the full-size initial states live. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import functools

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as vision_capture_reference,
)
from priml.baselines.mlsbench.tl.poison_robust_learning import reference
from priml.baselines.mlsbench.tl.poison_robust_learning.scripts import (
    capture_reference,
)
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"


def stored(prefix: str) -> dict[str, torch.Tensor]:
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


def test_every_loss_and_its_gradient_match_the_driver() -> None:
    assert_exact(stored("loss/"), reference.loss_record())


def test_the_label_flips_and_poisoned_items_match_the_driver() -> None:
    assert_exact(stored("flip/"), reference.flip_record())


def test_both_evaluations_match_the_driver_on_fixed_logits() -> None:
    assert_exact(stored("evaluate/"), reference.evaluate_record())


def test_the_evaluation_logits_are_signed_halves_with_one_tied_row() -> None:
    logits, label = reference.evaluate_logits()
    assert logits.shape == (vision_reference.LOADER_IMAGES, reference.FLIP_CLASSES)
    assert torch.equal(logits * 2, (logits * 2).round())
    assert (float(logits.min()), float(logits.max())) == (-2.0, 2.0)
    # Row 2 ties its two largest, so argmax's first-index rule decides it.
    assert logits[2].tolist() == [1.0, 1.0, 0.0, -1.0]
    others = torch.cat([logits[:2], logits[3:]])
    assert bool(
        (others == others.max(dim=1, keepdim=True).values).sum(dim=1).eq(1).all(),
    )
    assert torch.equal(label, reference.flip_labels())


def test_the_recipe_rates_and_optimizer_match_the_driver() -> None:
    assert_exact(stored("recipe/"), reference.recipe_record())


def test_each_settings_network_matches_the_driver() -> None:
    assert_exact(stored("settings/"), reference.network_record())


@pytest.mark.parametrize("name", list(reference.LOSSES))
def test_a_lockstep_run_with_each_loss_matches_the_driver(
    name: str,
    tmp_path: Path,
) -> None:
    assert_exact(stored(f"lockstep/{name}/"), reference.lockstep_record(name, tmp_path))


def test_a_lockstep_run_hands_back_its_states_and_evaluated_images(
    tmp_path: Path,
) -> None:
    states: list[dict[str, torch.Tensor]] = []
    media: list[torch.Tensor] = []
    record = reference.lockstep_record("pristine", tmp_path, states=states, media=media)
    # The whole split is one batch: one update in each of two epochs.
    assert len(states) == len(record["lockstep/pristine/loss"]) == 2
    assert not torch.equal(states[0]["stem.weight"], states[-1]["stem.weight"])
    # Ten logits per image forwarded: the seven test images and every poisoned
    # batch that holds a flip.
    images = sum(len(m) for m in media)
    assert images * 10 == len(record["lockstep/pristine/eval"])
    assert images > vision_reference.LOADER_IMAGES


def test_the_port_reaches_the_drivers_random_call_sites(tmp_path: Path) -> None:
    vision_capture_reference.assert_sites_reached(
        _DATA,
        port=functools.partial(capture_reference.port_sites, tmp_path),
        check=capture_reference.check_sites,
    )


def test_a_stored_driver_side_missing_a_required_site_fails(tmp_path: Path) -> None:
    sites = vision_capture_reference.read_sites(_DATA)
    sites["reference"] = {
        k: v for k, v in sites["reference"].items() if "run_poison_robust.py" not in k
    }
    vision_capture_reference.write_sites(tmp_path, sites)
    with pytest.raises(AssertionError, match="run_poison_robust"):
        vision_capture_reference.assert_sites_reached(
            tmp_path,
            port=dict,
            check=capture_reference.check_sites,
        )


def test_the_port_record_is_every_record_and_lockstep_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for part in (
        "loss_record",
        "flip_record",
        "evaluate_record",
        "recipe_record",
        "network_record",
    ):
        monkeypatch.setattr(reference, part, lambda part=part: {part: torch.zeros(2)})

    def lockstep(name: str, directory: Path) -> dict[str, torch.Tensor]:
        assert directory == tmp_path
        return {f"lockstep/{name}": torch.zeros(2)}

    monkeypatch.setattr(reference, "lockstep_record", lockstep)
    assert set(reference.port_record(tmp_path)) == {
        "loss_record",
        "flip_record",
        "evaluate_record",
        "recipe_record",
        "network_record",
        *(f"lockstep/{name}" for name in reference.LOSSES),
    }


def test_the_golden_fits_the_size_ceiling() -> None:
    assert (_DATA / "reference.pt").stat().st_size <= vision_capture_reference.CEILING


def test_the_golden_bites_on_one_ulp_of_the_ports_gradient() -> None:
    key = "loss/symmetric_ce/fixed/grad"
    value = reference.loss_record()[key].clone()
    value.view(-1)[0] = torch.nextafter(value.view(-1)[0], torch.tensor(torch.inf))
    assert mismatches(stored(key), {key: value}) == [f"{key}: 1/12 differ"]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
