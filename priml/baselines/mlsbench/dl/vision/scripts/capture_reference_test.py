"""Tests for the reference capture, against a stand-in clone of the benchmark.

The stand-in script has the reference's interface and a different network, so a
capture of it must report the network's mismatches and none of the components
whose recipe it shares with the port.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import re
import sys

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.baselines.mlsbench.dl.vision.experiments import exp000
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.dl.vision.upstream_test import write_clone
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD.parent / "testdata"

NO_PASSES: Final[tuple[upstream.Batches, upstream.Batches]] = ([], [])
"""A loader run of no passes, as a stubbed loader returns it."""


def returning[T](value: T) -> Callable[..., T]:
    """Return a function that ignores its arguments and gives back ``value``."""

    def call(*_arguments: object) -> T:
        return value

    return call


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """Write the stand-in clone and return its root."""
    root = tmp_path / "clone"
    write_clone(root)
    return root


def stand_in(clone: Path) -> tuple[upstream.Upstream, upstream.MainTrace]:
    """Return the stand-in script and a trace of its ``main``."""
    script = upstream.load_script(clone / upstream.PRISTINE)
    with upstream.hidden_accelerators():
        return script, upstream.trace_main(script)


def test_the_stand_in_shares_the_recipes_schedule_optimizer_and_loss_wiring(
    clone: Path,
) -> None:
    script, trace = stand_in(clone)
    record = upstream.upstream_record(script, trace, [])
    shared = {
        k: v for k, v in record.items() if k.startswith(("schedule/", "optimizer/"))
    }
    port = reference.schedule_record() | reference.optimizer_record()
    assert shared.keys() == port.keys()
    assert mismatches(shared, port) == []
    assert capture_reference.compare_wiring(trace) == []


def test_the_stand_in_differs_in_the_network_and_the_draws(clone: Path) -> None:
    script, trace = stand_in(clone)
    network = mismatches(
        upstream.upstream_network(script),
        reference.network_record(),
    )
    assert [line.split(":")[0] for line in network] == [
        "network/draws",
        "network/eval",
        "network/train",
    ]
    draws = mismatches(
        {"draws/exp000": trace.draws},
        reference.draws_record(),
    )
    assert draws == ["draws/exp000: 8/8 differ"]


def test_the_stand_in_loader_draws_other_streams_than_the_ports(
    clone: Path,
    tmp_path: Path,
) -> None:
    script, _ = stand_in(clone)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with upstream.hidden_accelerators():
        theirs = upstream.upstream_batches(script)
    report = capture_reference.compare_loaders(
        theirs,
        reference.loader_batches(scratch),
    )
    assert report
    assert all(line.startswith("loader/train/") for line in report)


def test_capture_assembles_both_sides_into_one_report(
    clone: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The port-side runs are tested above; here they are stored values, to test the glue."""
    stored = read_tensors(_DATA / "reference.pt")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    sites = {"uniform_@nn/init.py:f": 1}
    monkeypatch.setattr(reference, "port_record", returning(stored))
    monkeypatch.setattr(reference, "draw_sites", returning(sites))
    monkeypatch.setattr(reference, "fmnist_draw_sites", returning(sites))
    monkeypatch.setattr(
        capture_reference,
        "compare_full_size",
        returning(["full size differs"]),
    )
    monkeypatch.setattr(
        capture_reference,
        "compare_tiny_settings",
        returning(["tiny differs"]),
    )
    settings = {"settings/x": torch.zeros(2)}
    monkeypatch.setattr(
        capture_reference,
        "capture_settings",
        returning((settings, ["setting differs"])),
    )
    captured = capture_reference.capture(clone, scratch)
    assert captured.record.keys() == stored.keys() | settings.keys() | {
        "transform/train",
        "transform/test",
    }
    assert captured.sites["port"] == captured.sites["port-fmnist"] == sites
    assert captured.sites["reference"]
    assert captured.sites["reference-fmnist"]
    lines = captured.report
    assert lines[:2] == ["full size differs", "tiny differs"]
    assert lines[-1] == "setting differs"
    assert any(line.startswith("fmnist port: random call site") for line in lines)
    assert [line for line in lines if line.startswith("network/")]
    assert [line for line in lines if line.startswith("loader/")]
    assert not [line for line in lines if line.startswith(("schedule/", "optimizer/"))]
    assert any(line.startswith("port: random call site") for line in lines)


class Calls:
    """Records each call's arguments under the name it was made through."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def returning[T](self, name: str, value: T) -> Callable[..., T]:
        """Return a stand-in that records its call under ``name`` and gives ``value``."""

        def call(*args: object, **kwargs: object) -> T:
            self.seen.append((name, args, kwargs))
            return value

        return call

    def names(self) -> list[str]:
        """Return the names called, in order."""
        return [name for name, _, _ in self.seen]


def test_capture_settings_joins_each_captured_settings_comparisons(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Calls()
    trace = upstream.MainTrace()
    record = {"settings/a/draws": torch.arange(2)}
    monkeypatch.setattr(reference, "CAPTURED", ("mobilenetv2-fmnist",))
    monkeypatch.setattr(
        upstream,
        "upstream_settings",
        calls.returning("upstream", ({"mobilenetv2-fmnist": trace}, record)),
    )
    monkeypatch.setattr(
        reference,
        "settings_record",
        calls.returning("port", {"settings/a/draws": torch.arange(3)}),
    )
    monkeypatch.setattr(
        capture_reference,
        "compare_wiring",
        calls.returning("wiring", ["wiring differs"]),
    )
    monkeypatch.setattr(
        capture_reference,
        "compare_setting",
        calls.returning("setting", ["state differs"]),
    )
    monkeypatch.setattr(
        upstream,
        "upstream_batches",
        calls.returning("theirs", NO_PASSES),
    )
    monkeypatch.setattr(
        reference,
        "setting_batches",
        calls.returning("ours", NO_PASSES),
    )
    monkeypatch.setattr(
        capture_reference,
        "compare_loaders",
        calls.returning("loaders", ["loader/train/0: differs"]),
    )
    script = cast(upstream.Upstream, object())
    got, report = capture_reference.capture_settings(script, tmp_path)
    assert got is record
    assert report == [
        *mismatches(record, {"settings/a/draws": torch.arange(3)}),
        "wiring differs",
        "state differs",
        "settings/mobilenetv2-fmnist/loader/train/0: differs",
    ]
    assert calls.names() == [
        "upstream",
        "port",
        "wiring",
        "setting",
        "theirs",
        "ours",
        "loaders",
    ]
    _, wiring_args, wiring_kwargs = calls.seen[2]
    assert (wiring_args, wiring_kwargs) == ((trace,), {"classes": 10})
    assert calls.seen[4][2] == {"dataset": "fmnist"}
    assert calls.seen[5][1] == (tmp_path, "mobilenetv2-fmnist")


def weight_sum(model: torch.nn.Module, media: torch.Tensor) -> torch.Tensor:
    """Stand in for a forward: the sum of the model's weights, whatever the images."""
    del media
    return torch.stack([p.sum() for p in model.parameters()]).sum()


def test_compare_tiny_settings_reports_each_setting_under_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Calls()
    built = torch.nn.Linear(2, 3)
    monkeypatch.setattr(upstream, "tiny_model", calls.returning("theirs", built))
    step = SimpleNamespace(model=built)
    monkeypatch.setattr(reference, "tiny_step", calls.returning("ours", step))
    monkeypatch.setattr(
        capture_reference,
        "compare_states",
        calls.returning("states", ["state differs"]),
    )
    monkeypatch.setattr(reference, "logits_of", weight_sum)
    script = SimpleNamespace(initialize_weights=calls.returning("init", None))
    report = capture_reference.compare_tiny_settings(cast(upstream.Upstream, script))
    # Both sides are the same stub, so the draws and both forwards agree and only
    # the stubbed state comparison reports.
    assert report == [
        "tiny/vgg16bn-cifar100: state differs",
        "tiny/mobilenetv2-fmnist: state differs",
    ]
    assert [args[1] for name, args, _ in calls.seen if name == "theirs"] == [
        "vgg16bn",
        "mobilenetv2",
    ]
    assert [kw["blocks"] for name, _, kw in calls.seen if name == "ours"] == [
        (1, 1, 1),
    ] * 2
    assert calls.names().count("init") == 2


def test_compare_task_settings_runs_each_variant_and_each_non_cifar100_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = Calls()
    trace = upstream.MainTrace()
    monkeypatch.setattr(
        upstream,
        "upstream_setting",
        calls.returning("network", (trace, {"settings/n": torch.zeros(2)})),
    )
    monkeypatch.setattr(
        capture_reference,
        "compare_setting",
        calls.returning("setting", ["state differs"]),
    )
    batches: upstream.Batches = [[{"label": torch.arange(2)}]]
    tests: upstream.Batches = []
    monkeypatch.setattr(
        upstream,
        "upstream_batches",
        calls.returning("theirs", (batches, tests)),
    )
    monkeypatch.setattr(
        reference,
        "decode_batches",
        calls.returning("decode", {"loader/label": torch.arange(2)}),
    )
    monkeypatch.setattr(
        reference,
        "setting_batches",
        calls.returning("ours", NO_PASSES),
    )
    monkeypatch.setattr(
        capture_reference,
        "compare_loaders",
        calls.returning("loaders", ["loader/test/0: differs"]),
    )
    monkeypatch.setattr(
        reference,
        "task_settings_record",
        calls.returning("port", {"settings/n": torch.zeros(2)}),
    )
    scripts = {
        "a": cast(upstream.Upstream, object()),
        "b": cast(upstream.Upstream, object()),
    }
    experiments = {
        "vgg16bn-cifar100": {"a": exp000},
        "mobilenetv2-fmnist": {"a": exp000, "b": exp000},
    }
    # The pristine script is not the last variant, so a loader run on whichever
    # variant the loop ended on is caught.
    record, report = capture_reference.compare_task_settings(
        scripts,
        experiments,
        tmp_path,
        pristine="a",
    )
    assert set(record) == {"settings/n", "settings/mobilenetv2-fmnist/loader/label"}
    assert report == [
        "state differs",
        "state differs",
        "state differs",
        "settings/mobilenetv2-fmnist/loader/test/0: differs",
        "missing settings/mobilenetv2-fmnist/loader/label",
    ]
    keys = [kw["key"] for name, _, kw in calls.seen if name == "network"]
    assert keys == [
        "vgg16bn-cifar100/a",
        "mobilenetv2-fmnist/a",
        "mobilenetv2-fmnist/b",
    ]
    (theirs,) = [(args, kw) for name, args, kw in calls.seen if name == "theirs"]
    assert theirs == ((scripts["a"],), {"dataset": "fmnist"})
    assert calls.seen[-1][1] == (experiments, tmp_path)


def test_main_prints_the_report_and_writes_nothing_when_something_differs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = reference.Captured(record={}, sites={}, report=["a differs", "b differs"])
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["capture", "--root", str(tmp_path), "--out", str(out)],
    )
    assert capture_reference.main() == 1
    assert (
        capsys.readouterr().out
        == "a differs\nb differs\n2 mismatches; nothing written.\n"
    )
    assert not out.exists()


def test_main_writes_the_goldens_when_every_comparison_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    record = {
        "transform/train": torch.zeros(2),
        "transform/test": torch.ones(2),
        "transform/fmnist/train": torch.zeros(3),
        "transform/fmnist/test": torch.ones(3),
        "settings/a/draws": torch.arange(2),
        "x": torch.arange(3),
    }
    result = reference.Captured(
        record=record,
        sites={"port": {"a@b.py:c": 2}},
        report=[],
    )
    monkeypatch.setattr(capture_reference, "capture", returning(result))
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        ["capture", "--root", str(tmp_path), "--out", str(out)],
    )
    assert capture_reference.main() == 0
    assert f"0 mismatches; wrote 6 values to {out}." in capsys.readouterr().out
    assert read_tensors(out / "reference.pt").keys() == {"x"}
    assert read_tensors(out / "reference_settings.pt").keys() == {"settings/a/draws"}
    assert read_tensors(out / "reference_fmnist_train_pipeline.pt").keys() == {
        "transform/fmnist/train",
    }
    assert read_tensors(out / "reference_fmnist_test_pipeline.pt").keys() == {
        "transform/fmnist/test",
    }
    assert read_tensors(out / "reference_train_pipeline.pt").keys() == {
        "transform/train",
    }
    assert read_tensors(out / "reference_test_pipeline.pt").keys() == {"transform/test"}
    assert (out / "reference_sites.txt").read_text() == "port a@b.py:c 2\n"


def test_a_golden_over_the_ceiling_is_refused_and_writes_nothing(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "reference.pt"
    previous.write_bytes(b"previous")
    record = {
        "transform/train": torch.zeros(capture_reference.CEILING),
        "transform/test": torch.zeros(2),
        "transform/fmnist/train": torch.zeros(2),
        "transform/fmnist/test": torch.zeros(2),
    }
    with pytest.raises(ValueError, match=re.escape("reference_train_pipeline.pt is")):
        capture_reference.write_goldens(
            tmp_path,
            reference.Captured(record=record, sites={}, report=[]),
        )
    assert previous.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["reference.pt"]


def test_help_describes_the_command(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["capture", "--help"])
    with pytest.raises(SystemExit, match="0"):
        capture_reference.main()
    help_text = capsys.readouterr().out
    assert (
        "Capture the reference's values and prove the port reproduces them."
        in help_text
    )
    assert "--root" in help_text
    assert "--out" in help_text


def test_a_call_site_that_a_side_never_reached_is_reported_for_that_side() -> None:
    port: dict[str, int] = dict.fromkeys(capture_reference.REQUIRED_PORT, 1)
    reference_side: dict[str, int] = dict.fromkeys(
        capture_reference.REQUIRED_UPSTREAM,
        1,
    )
    assert capture_reference.check_sites(reference_side, port) == []
    del reference_side["rand@transforms/transforms.py"]
    port["randint@vision/data.py"] = 0
    assert capture_reference.check_sites(reference_side, port) == [
        "reference: random call site rand@transforms/transforms.py was never reached",
        "port: random call site randint@vision/data.py was never reached",
    ]


def test_states_are_compared_in_registration_order_whatever_the_names() -> None:
    want = {"a": torch.arange(3.0), "b": torch.ones(2)}
    same = {"x": torch.arange(3.0), "y": torch.ones(2)}
    assert capture_reference.compare_states(want, same) == []
    off = {"x": torch.arange(3.0), "y": torch.nextafter(torch.ones(2), torch.zeros(2))}
    assert capture_reference.compare_states(want, off) == ["b: 2/2 differ"]
    assert capture_reference.compare_states(want, {"x": torch.arange(3.0)}) == [
        "state: 1 tensors against 2",
    ]


def test_the_loss_configuration_must_be_zero_based_over_200_epochs() -> None:
    trace = upstream.MainTrace()
    trace.configs = [
        {"num_classes": 100, "epoch": epoch, "total_epochs": 200}
        for epoch in range(200)
    ]
    assert capture_reference.compare_wiring(trace) == []
    trace.configs[0]["epoch"] = 1
    assert capture_reference.compare_wiring(trace) == [
        "main: loss configuration differs",
    ]


def test_the_loss_configuration_is_checked_against_the_settings_classes() -> None:
    trace = upstream.MainTrace()
    trace.configs = [
        {"num_classes": 10, "epoch": epoch, "total_epochs": 200} for epoch in range(200)
    ]
    assert capture_reference.compare_wiring(trace, classes=10) == []
    assert capture_reference.compare_wiring(trace) == [
        "main: loss configuration differs",
    ]


def test_a_setting_whose_network_differs_is_reported_by_name(
    clone: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full MobileNetV2 is compared only in the capture; here the port's is shallow."""

    def shallow(factory: Callable[[], reference.VisionLoop]) -> VisionTrainStep:
        return reference.tiny_step(
            factory,
            blocks=(1, 1, 1),
            classes=reference.CLASSES,
            epochs=reference.EPOCHS,
        )

    monkeypatch.setattr(reference, "full_step", shallow)
    pristine = upstream.load_script(clone / upstream.PRISTINE)
    with upstream.hidden_accelerators():
        trace, _ = upstream.upstream_setting(pristine, "mobilenetv2-fmnist")
    report = capture_reference.compare_setting(trace, "mobilenetv2-fmnist")
    assert report[0].startswith("settings/mobilenetv2-fmnist/state: state: ")
    keyed = capture_reference.compare_setting(
        trace,
        "mobilenetv2-fmnist",
        reference.SETTINGS["mobilenetv2-fmnist"].experiment,
        key="mobilenetv2-fmnist/x",
    )
    assert keyed[0].startswith("settings/mobilenetv2-fmnist/x/state: state: ")


def test_an_upstream_setting_record_is_keyed_as_asked(clone: Path) -> None:
    pristine = upstream.load_script(clone / upstream.PRISTINE)
    with upstream.hidden_accelerators():
        _, record = upstream.upstream_setting(
            pristine,
            "resnet110-cifar100",
            key="resnet110-cifar100/x",
        )
    assert set(record) == {
        f"settings/resnet110-cifar100/x/{part}" for part in ("draws", "train", "eval")
    }


def test_loader_passes_are_compared_pixel_by_pixel() -> None:
    batch = {"media": torch.zeros(2, 3), "label": torch.arange(2)}
    passes = ([[batch]], [[batch]])
    assert capture_reference.compare_loaders(passes, passes) == []
    moved = {
        "media": torch.nextafter(batch["media"], torch.ones(2, 3)),
        "label": batch["label"],
    }
    assert capture_reference.compare_loaders(passes, ([[moved]], [[batch]])) == [
        "loader/train/0: 0/media: 6/6 differ",
    ]


def test_a_lockstep_comparison_runs_the_port_it_is_given() -> None:
    state = {"w": torch.zeros(2)}
    calls: list[str] = []

    def run(
        name: str,
        *,
        states: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        calls.append(name)
        states.append({"w": torch.ones(2)})
        return {}

    report = capture_reference.compare_lockstep({"x": ({}, [state])}, run=run)
    assert calls == ["x"]
    assert report == ["lockstep/x/1: w: 2/2 differ"]
    report = capture_reference.compare_lockstep({"x": ({}, [state] * 3)}, run=run)
    assert report == [
        "lockstep/x: 1 updates against 3",
        "lockstep/x/1: w: 2/2 differ",
    ]


def test_the_full_size_comparison_names_a_network_that_has_another_shape(
    clone: Path,
) -> None:
    pristine = upstream.load_script(clone / upstream.PRISTINE)
    with upstream.hidden_accelerators():
        trace = upstream.trace_main(pristine)
    report = capture_reference.compare_full_size(pristine, trace)
    assert report[0].startswith("state: ")
    assert report[0].endswith(f"against {len(trace.state)}")
    assert [line.split(":")[0] for line in report[1:]] == [
        "forward/train",
        "forward/eval",
    ]


def test_the_sites_file_reads_back_as_written(tmp_path: Path) -> None:
    sites = {"reference": {"a@b.py:c": 2, "d@e.py:f": 1}, "port": {"a@b.py:c": 3}}
    capture_reference.write_sites(tmp_path, sites)
    assert (
        capture_reference.parse_sites((tmp_path / "reference_sites.txt").read_text())
        == sites
    )


def test_a_score_in_both_sets_of_an_auc_is_reported() -> None:
    record = {
        "a/member": torch.tensor([0.25, 0.5, 0.5]),
        "a/nonmember": torch.tensor([0.125, 0.75]),
        "b/member": torch.tensor([0.25, 0.5]),
        "b/nonmember": torch.tensor([0.5, 0.375]),
    }
    pairs = {"a/member": "a/nonmember", "b/member": "b/nonmember"}
    assert capture_reference.shared_scores(record, pairs=pairs) == [
        "b/member: 1 scores also in b/nonmember; their order is the host's",
    ]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
