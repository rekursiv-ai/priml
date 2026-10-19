"""Tests for the runs of the ``cv-classification-loss`` script, against a stand-in clone.

The stand-in is the vision package's, whose ``compute_loss`` its baselines'
edits replace. ``write_clone`` is shared with the capture's tests.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c/harbor/tasks-docker/mls-bench__cv-classification-loss/tests/meta/pristine/pytorch-vision/custom_loss.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import torch

from priml.baselines.mlsbench.dl.classification_loss import upstream
from priml.baselines.mlsbench.dl.vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.upstream_test import (
    SCRIPT,
    SETTINGS_SCRIPT,
)
from priml.testing.golden import read_tensors


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.upstream import Upstream

_DATA: Final = _CWD / "testdata"

LOSS: Final = "def compute_loss(logits, targets, config):\n    return F.cross_entropy(logits, targets)\n"
"""The two lines of the stand-in's loss, which the baselines' edits replace."""

BODIES: Final = {
    "label_smoothing": "F.cross_entropy(logits, targets, label_smoothing=0.1)",
    "focal_loss": "F.cross_entropy(logits, targets) * 2",
    "poly_loss": "F.cross_entropy(logits, targets) + 1",
}
"""The return expression each baseline's edit puts in the loss."""


def write_clone(root: Path) -> None:
    """Write the stand-in script and the three baselines' edits under ``root``.

    Args:
      root: Directory to hold the clone.

    """
    script = root / vision_upstream.PRISTINE
    script.parent.mkdir(parents=True)
    # The settings' additions follow ``SCRIPT``, so the loss keeps its lines.
    script.write_text(SETTINGS_SCRIPT)
    edits = root / upstream.EDITS
    edits.mkdir(parents=True)
    first = SCRIPT[: SCRIPT.index(LOSS)].count("\n") + 1
    for name, body in BODIES.items():
        content = LOSS.replace("F.cross_entropy(logits, targets)", body)
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {first + 1}, "
            f"'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, Upstream]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_each_variant_carries_its_baselines_loss(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["cross_entropy", *upstream.VARIANTS]
    logits, targets = reference.fixed_logits()
    config = {"num_classes": 4, "epoch": 0, "total_epochs": 1}
    plain = variants["cross_entropy"].compute_loss(logits, targets, config)
    assert variants["focal"].compute_loss(logits, targets, config) == plain * 2
    assert variants["poly"].compute_loss(logits, targets, config) == plain + 1
    focal = (tmp_path / "scratch" / "focal_loss.py").read_text()
    assert focal == SETTINGS_SCRIPT.replace(
        "return F.cross_entropy(logits, targets)\n",
        "return F.cross_entropy(logits, targets) * 2\n",
        1,
    )


def test_lockstep_records_two_updates_and_leaves_the_script_unpatched(
    tmp_path: Path,
) -> None:
    pristine = scripts(tmp_path)["cross_entropy"]
    original = pristine.compute_loss
    trace = vision_upstream.trace_main(pristine)
    record, states = upstream.upstream_lockstep(pristine, "cross_entropy", trace)
    assert pristine.compute_loss is original
    assert record["lockstep/cross_entropy/context"].tolist() == [
        [reference.CLASSES, 0, reference.EPOCHS],
        [reference.CLASSES, 1, reference.EPOCHS],
    ]
    assert record["lockstep/cross_entropy/loss"].shape == (2,)
    assert record["lockstep/cross_entropy/logits"].shape == (2, 2, reference.CLASSES)
    assert record["lockstep/cross_entropy/probe"].shape == (2, 2, reference.CLASSES)
    assert len(states) == 2
    assert not torch.equal(states[0]["fc.weight"], states[1]["fc.weight"])


def test_setting_losses_run_each_variant_at_each_settings_class_count(
    tmp_path: Path,
) -> None:
    variants = scripts(tmp_path)
    record = upstream.upstream_setting_losses(variants, ["mobilenetv2-fmnist"])
    plain = record["settings/mobilenetv2-fmnist/loss/cross_entropy/value"]
    assert record["settings/mobilenetv2-fmnist/loss/focal/value"] == plain * 2
    assert record["settings/mobilenetv2-fmnist/loss/poly/grad"].shape == (3, 10)
    assert len(record) == 2 * len(variants)


def test_the_record_has_every_key_the_golden_holds(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = vision_upstream.trace_main(variants["cross_entropy"])
    lockstep = {
        name: upstream.upstream_lockstep(variant, name, trace)
        for name, variant in variants.items()
    }
    record = upstream.upstream_record(variants, lockstep)
    assert record.keys() == read_tensors(_DATA / "reference.pt").keys()
    assert record["loss/poly/grad"].shape == (3, 4)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
