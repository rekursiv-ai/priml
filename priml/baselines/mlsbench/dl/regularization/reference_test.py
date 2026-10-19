"""Replay the ``dl-regularization`` port against the values captured from its script.

``testdata/reference.pt`` holds what the pristine script and each baseline's
edit produced, written by ``scripts/capture_reference.py``. Equality is exact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.mlsbench.dl.regularization import reference
from priml.baselines.mlsbench.dl.regularization.experiments import exp000
from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.dl.vision.train_step import LossContext
from priml.baselines.mlsbench.dl.vision.upstream_test import StoredPart
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


def test_every_regularizer_and_its_gradients_match_the_script() -> None:
    assert_exact(stored("regularizer/"), reference.regularizer_record())


def test_a_port_call_hands_on_every_input_and_states_its_context() -> None:
    seen = _Seen()
    model = nn.Identity()
    media, logits, label = torch.zeros(2, 3), torch.ones(2, 4), torch.tensor([0, 3])
    term = reference._PortCall(seen)(
        model,
        media=media,
        logits=logits,
        label=label,
        config={"num_classes": 4, "epoch": 3, "total_epochs": 5},
    )
    assert torch.equal(term, torch.tensor(8.0))
    [(got_model, got_media, got_logits, got_label, context)] = seen.calls
    assert got_model is model
    assert got_media is media
    assert got_logits is logits
    assert got_label is label
    assert context == LossContext(num_classes=4, epoch=3, total_epochs=5)


def test_the_schedule_and_optimizer_match_the_setting() -> None:
    assert_exact(
        stored("schedule/") | stored("optimizer/"),
        vision_reference.schedule_record(exp000)
        | vision_reference.optimizer_record(exp000),
    )


def test_the_batch_size_matches_the_setting() -> None:
    assert_exact(
        stored("setting/"),
        {"setting/batch_size": reference.port_batch_size()},
    )


def test_the_full_step_leaves_the_generator_where_the_script_does() -> None:
    assert_exact(stored("draws/"), reference.draws_record())


def test_the_best_accuracy_is_reported_as_the_script_does() -> None:
    assert_exact(stored("best/"), vision_reference.best_record())


@pytest.mark.parametrize("name", list(reference.REGULARIZERS))
def test_two_updates_with_each_regularizer_match_the_script(name: str) -> None:
    states: list[dict[str, torch.Tensor]] = []
    assert_exact(
        stored(f"lockstep/{name}/"),
        reference.lockstep_record(name, states=states),
    )
    assert len(states) == vision_reference.STEPS


def test_the_golden_bites_on_one_ulp() -> None:
    key = "regularizer/orthogonal_reg/value"
    value = stored(key)[key].clone()
    value[1] = torch.nextafter(value[1], torch.tensor(0.0))
    port = {key: reference.regularizer_record()[key]}
    assert [line.split(":")[0] for line in mismatches({key: value}, port)] == [key]


def stored_settings(prefix: str) -> dict[str, torch.Tensor]:
    """Return the captured setting values whose key starts with ``prefix``."""
    record = read_tensors(_DATA / "reference_settings.pt")
    return {k: v for k, v in record.items() if k.startswith(prefix)}


@pytest.mark.parametrize("setting", list(reference.SETTING_REGULARIZERS))
def test_each_settings_recipe_matches_its_launch_script(setting: str) -> None:
    prefix = f"settings/{setting}/"
    expected = {
        k: v
        for k, v in stored_settings(prefix).items()
        if k.startswith(
            (f"{prefix}schedule/", f"{prefix}optimizer/", f"{prefix}setting/"),
        )
    }
    assert len(expected) == 3
    assert_exact(expected, reference.setting_recipe_record(setting))


def test_the_batch_size_of_a_setting_is_read_from_its_command_line() -> None:
    argv = ["python", "main.py", "--epochs", "3", "--batch-size", "64", "--lr", "0.1"]
    assert reference.setting_batch_size(argv).tolist() == [64, 64]


def test_the_vgg_setting_warms_up_as_the_script_does() -> None:
    rates = stored_settings("settings/vgg16bn-cifar100/schedule/lr")
    (lr,) = rates.values()
    assert lr[:5].tolist() == [0.1 * ((e + 1) / 5) for e in range(5)]


@pytest.mark.parametrize(
    "key",
    [
        f"{setting}/{name}"
        for setting, variants in reference.SETTING_REGULARIZERS.items()
        for name in variants
    ],
)
def test_two_updates_at_each_setting_match_the_script(key: str) -> None:
    setting, name = key.split("/")
    assert_exact(
        stored_settings(f"settings/{setting}/lockstep/{name}/"),
        reference.setting_lockstep_record(key),
    )


@pytest.mark.parametrize("setting", list(reference.SETTING_REGULARIZERS))
def test_each_regularizer_on_each_settings_network_matches_the_script(
    setting: str,
) -> None:
    expected = stored_settings(f"settings/{setting}/regularizer/")
    assert len(expected) == 2 * (len(reference.REGULARIZERS) - 1)
    record = reference.setting_regularizer_record(
        setting,
        reference.tiny_network(setting),
    )
    assert_exact(expected, {k: v for k, v in record.items() if k in expected})


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize(
    "key",
    [
        f"{setting}/{name}"
        for setting, variants in reference.SETTING_REGULARIZERS.items()
        for name in variants
    ],
)
def test_each_variants_full_network_at_each_setting_matches_the_script(
    key: str,
) -> None:
    setting, name = key.split("/")
    expected = {
        k: v
        for k, v in stored_settings(f"settings/{key}/").items()
        if k.rsplit("/", 1)[-1] in {"draws", "train", "eval"}
    }
    assert len(expected) == 3
    assert_exact(
        expected,
        vision_reference.setting_record(
            setting,
            reference.SETTING_REGULARIZERS[setting][name],
            key=key,
        ),
    )


def test_the_fmnist_loader_matches_the_script(tmp_path: Path) -> None:
    assert_exact(
        stored_settings("settings/mobilenetv2-fmnist/loader/"),
        vision_reference.setting_loader_record(tmp_path, "mobilenetv2-fmnist"),
    )


def test_the_vgg_orthogonal_term_is_zero_as_the_script_computes_it() -> None:
    key = "settings/vgg16bn-cifar100/regularizer/orthogonal_reg/value"
    assert stored_settings(key)[key].tolist() == [0.0] * len(reference.PROBE_EPOCHS)


@pytest.mark.parametrize("name", ["reference.pt", "reference_settings.pt"])
def test_captured_goldens_fit_the_size_ceiling(name: str) -> None:
    assert (_DATA / name).stat().st_size <= capture_reference.CEILING


def test_the_settings_golden_bites_on_one_ulp() -> None:
    key = "vgg16bn-cifar100/confidence_penalty"
    loss = "settings/vgg16bn-cifar100/lockstep/confidence_penalty/loss"
    expected = stored_settings(loss)
    bumped = expected[loss].clone()
    bumped[1] = torch.nextafter(bumped[1], torch.tensor(torch.inf))
    port = reference.setting_lockstep_record(key)
    assert mismatches({loss: bumped}, {loss: port[loss]}) == [f"{loss}: 1/2 differ"]


class _Seen:
    """A regularizer that keeps what each call handed it."""

    def __init__(self) -> None:
        self.calls: list[tuple[nn.Module, Tensor, Tensor, Tensor, LossContext]] = []

    def __call__(
        self,
        model: nn.Module,
        /,
        *,
        media: Tensor,
        logits: Tensor,
        label: Tensor,
        context: LossContext,
    ) -> Tensor:
        self.calls.append((model, media, logits, label, context))
        return logits.sum()


def test_the_port_record_joins_every_part_the_golden_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden = stored("")
    monkeypatch.setattr(
        reference,
        "lockstep_record",
        StoredPart(golden, "lockstep/{}/"),
    )
    monkeypatch.setattr(reference, "draws_record", StoredPart(golden, "draws/"))
    assert_exact(golden, reference.port_record())


def test_the_settings_record_joins_each_settings_recipe_terms_and_locksteps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden = {
        k: v
        for k, v in stored_settings("").items()
        if k.split("/")[2] in {"schedule", "optimizer", "setting", "lockstep"}
    }
    monkeypatch.setattr(
        reference,
        "setting_lockstep_record",
        StoredPart(golden, "settings/{}/lockstep/{}/"),
    )
    monkeypatch.setattr(reference, "tiny_network", nn.Identity)
    monkeypatch.setattr(reference, "setting_regularizer_record", _no_terms)
    assert_exact(golden, reference.settings_record())


def _no_terms(setting: str, model: nn.Module) -> dict[str, Tensor]:
    """Stand in for a setting's regularizer terms, tested above, with none."""
    del setting, model
    return {}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
