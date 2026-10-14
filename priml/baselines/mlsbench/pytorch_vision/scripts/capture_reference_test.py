"""Tests for the reference capture, against a stand-in clone of the benchmark.

The stand-in script has the reference's interface and a different network, so a
capture of it must report the network's mismatches and none of the components
whose recipe it shares with the port.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import re
import sys

import pytest
import torch

from priml.baselines.mlsbench.pytorch_vision import reference, upstream
from priml.baselines.mlsbench.pytorch_vision.scripts import capture_reference
from priml.baselines.mlsbench.pytorch_vision.upstream_test import write_clone
from priml.testing.golden import mismatches, read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD.parent / "testdata"


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


def test_the_stand_in_shares_the_recipes_loss_schedule_and_optimizer(
    clone: Path,
    tmp_path: Path,
) -> None:
    script, trace = stand_in(clone)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    variants = {"cross_entropy": script} | {
        name: upstream.edited_script(clone, file, scratch)
        for name, file in upstream.VARIANTS.items()
    }
    record = upstream.upstream_record(variants, trace, {}, [])
    port = (
        reference.loss_record()
        | reference.schedule_record()
        | reference.optimizer_record()
    )
    # The stand-in's focal and poly edits are not the real ones, so those differ.
    same = ("loss/cross_entropy/", "loss/label_smoothing/", "schedule/", "optimizer/")
    shared = {k: v for k, v in record.items() if k.startswith(same)}
    assert (
        mismatches(
            shared,
            {k: v for k, v in port.items() if k in shared},
        )
        == []
    )
    other = {
        k: v for k, v in record.items() if k.startswith(("loss/focal/", "loss/poly/"))
    }
    assert mismatches(
        other,
        {k: v for k, v in port.items() if k in other},
    )
    assert capture_reference.compare_wiring(trace) == []


def test_the_stand_in_differs_in_the_network_the_draws_and_the_updates(
    clone: Path,
) -> None:
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
    lockstep = upstream.upstream_lockstep(script, "cross_entropy", trace)
    assert capture_reference.compare_lockstep({"cross_entropy": lockstep})


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
    monkeypatch.setattr(
        capture_reference,
        "compare_full_size",
        returning(["full size differs"]),
    )
    monkeypatch.setattr(
        capture_reference,
        "compare_lockstep",
        returning(["lockstep differs"]),
    )
    captured = capture_reference.capture(clone, scratch)
    assert captured.record.keys() == stored.keys() | {
        "transform/train",
        "transform/test",
    }
    assert captured.sites["port"] == sites
    assert captured.sites["reference"]
    lines = captured.report
    assert lines[0] == "full size differs"
    assert "lockstep differs" in lines
    assert [line for line in lines if line.startswith("network/")]
    assert [line for line in lines if line.startswith("loader/")]
    assert not [
        line
        for line in lines
        if line.startswith(
            ("loss/cross_entropy", "loss/label_smoothing", "schedule/", "optimizer/"),
        )
    ]
    assert any(line.startswith("port: random call site") for line in lines)


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
    assert f"0 mismatches; wrote 3 values to {out}." in capsys.readouterr().out
    assert read_tensors(out / "reference.pt").keys() == {"x"}
    assert read_tensors(out / "reference_train_pipeline.pt").keys() == {
        "transform/train",
    }
    assert read_tensors(out / "reference_test_pipeline.pt").keys() == {"transform/test"}
    assert (out / "reference_sites.txt").read_text() == "port a@b.py:c 2\n"


def test_a_golden_over_the_ceiling_is_refused(tmp_path: Path) -> None:
    record = {
        "transform/train": torch.zeros(capture_reference.CEILING),
        "transform/test": torch.zeros(2),
    }
    with pytest.raises(ValueError, match=re.escape("reference_train_pipeline.pt is")):
        capture_reference.write_goldens(
            tmp_path,
            reference.Captured(record=record, sites={}, report=[]),
        )


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
    port["randint@pytorch_vision/data.py"] = 0
    assert capture_reference.check_sites(reference_side, port) == [
        "reference: random call site rand@transforms/transforms.py was never reached",
        "port: random call site randint@pytorch_vision/data.py was never reached",
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


def test_a_lockstep_run_that_matches_reports_nothing_and_one_that_drifts_is_named() -> (
    None
):
    ours: list[dict[str, torch.Tensor]] = []
    _ = reference.lockstep_record("cross_entropy", states=ours)
    assert capture_reference.compare_lockstep({"cross_entropy": ({}, ours)}) == []
    drifted = [dict(state) for state in ours]
    weight = drifted[1]["stem.weight"].clone()
    weight.view(-1)[0] = torch.nextafter(weight.view(-1)[0], torch.tensor(torch.inf))
    drifted[1]["stem.weight"] = weight
    assert capture_reference.compare_lockstep({"cross_entropy": ({}, drifted)}) == [
        "lockstep/cross_entropy/2: stem.weight: 1/432 differ",
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


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
