"""Replay the ``security-machine-unlearning`` port against the harness's captured values.

``testdata/reference.pt`` holds what the pristine harness, its method and each
baseline's edit produced, written by ``scripts/capture_reference.py``. Equality
is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.scripts.capture_reference import (
    CEILING,
    shared_scores,
)
from priml.baselines.mlsbench.tl.machine_unlearning import reference
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"


def stored(prefix: str) -> dict[str, Tensor]:
    """Return the captured values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference.pt")
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
    assert_exact(
        stored(f"main/{name}/"),
        reference.main_record(name, reference.port_main(name, tmp_path)),
    )


def returning(name: str) -> Callable[[], dict[str, Tensor]]:
    """Return a stand-in for a record function: ``name`` mapped to zero."""
    return lambda: {name: torch.zeros(1)}


def fake_lockstep(name: str) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
    """Stand in for :func:`reference.method_lockstep`: one key per variant."""
    return {f"methods/{name}": torch.zeros(1)}, []


def fake_full_size(key: str) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Stand in for :func:`reference.full_size_record`: one key per setting."""
    return {key: torch.zeros(1)}, {"w": torch.ones(1)}


def fake_port_main(name: str, directory: Path) -> reference.MainTrace:
    """Stand in for :func:`reference.port_main`: an empty run."""
    del name, directory
    return reference.MainTrace()


def fake_main_record(name: str, trace: reference.MainTrace) -> dict[str, Tensor]:
    """Stand in for :func:`reference.main_record`: one key per variant."""
    del trace
    return {name: torch.zeros(1)}


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
    monkeypatch.setattr(reference, "port_main", fake_port_main)
    monkeypatch.setattr(reference, "main_record", fake_main_record)
    record, states = reference.port_record(tmp_path)
    assert set(record) == {
        "recipe_record",
        "split_record",
        "evaluation_record",
        "method_record",
        *reference.SETTINGS,
        *reference.VARIANTS,
    }
    assert set(states) == set(reference.SETTINGS)


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
