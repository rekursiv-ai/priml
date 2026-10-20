"""Share complete captured inputs and record assembly across numerical tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import json

from torch import Tensor
from treekle import to_plain

import numpy as np
import torch

from priml.baselines.name_that_llm.experiments import exp000
from priml.baselines.name_that_llm.model import (
    CenteredHeads,
    QwenHeads,
    Split,
    SvmBranch,
)
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.baselines.name_that_llm.benchmark import Benchmark
    from priml.baselines.name_that_llm.testing.trace_support_test import (
        NumericalTrace,
    )


_CWD: Final = Path(__file__).resolve().parent
_BASELINE: Final = _CWD.parent


def recipe() -> Benchmark.Config:
    """Return exp000 finalized, with its saved layers and input paths resolved."""
    return exp000().finalized()


def svm_branch() -> SvmBranch.Config:
    """Return exp000's calibrated lexical branch."""
    branch = recipe().branches["svm"]
    assert isinstance(branch, SvmBranch.Config)
    return branch


def plain_branch() -> QwenHeads.Config:
    """Return exp000's plain Qwen heads, saved layers resolved."""
    branch = recipe().branches["plain"]
    assert isinstance(branch, QwenHeads.Config)
    return branch


def centred_branch() -> CenteredHeads.Config:
    """Return exp000's chunk-centred Qwen heads, saved layers resolved."""
    branch = recipe().branches["centred"]
    assert isinstance(branch, CenteredHeads.Config)
    return branch


def miniature() -> tuple[
    NDArray[np.object_],
    NDArray[np.int64],
    NDArray[np.str_],
    NDArray[np.float32],
]:
    """Build small geometry with all layers, branches and chunk sizes intact.

    Returns:
      texts: Deterministic short training passages.
      targets: Three repeated class indices.
      groups: Six complete prompt groups.
      embeddings: Complete captured inputs at the configured geometry.

    """
    texts = np.array(
        [f"a{label} b{group}" for group in range(6) for label in range(3)],
        dtype=object,
    )
    targets = np.tile(np.arange(3, dtype=np.int64), reps=6)
    groups = np.repeat(np.array(["a", "b", "c", "d", "e", "f"]), repeats=3)
    embeddings = read_tensors(_BASELINE / "testdata" / "exp000.pt")[
        "input/embeddings"
    ].numpy()
    return texts, targets, groups, embeddings


def miniature_split() -> Split:
    """Fit on the miniature and predict its first three rows in two clusters.

    Returns:
      split: Complete miniature rows with the first three also predicted.

    """
    texts, targets, groups, embeddings = miniature()
    return Split(
        texts=texts,
        targets=targets,
        groups=groups,
        embeddings=embeddings,
        predict_texts=texts[:3],
        predict_embeddings=embeddings[:3],
        clusters=np.array([2, 2, 3]),
    )


def capture_source(component: str) -> dict[str, Tensor]:
    """Read complete portable outputs and captured library-boundary inputs.

    Args:
      component: Portable component whose output is requested.

    Returns:
      outputs: Source outputs and the whole captured branch-probability inputs.

    """
    record = read_tensors(_BASELINE / "testdata" / "exp000.pt")
    return {
        key.split("/", maxsplit=1)[1]: value
        for key, value in record.items()
        if key.startswith("input/")
        or key == f"{component}/{component}"
        or (component == "folds" and key.startswith("folds/fold/"))
    }


def completed_record(
    trace: NumericalTrace,
    outputs: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Retain every output, fitted numeric field, draw, and complete RNG state.

    Args:
      trace: Observer holding all fitted numeric attributes and random draws.
      outputs: Complete component outputs to append to the record.

    Returns:
      record: Every captured value, operation count and full generator state.

    """
    for name, value in outputs.items():
        trace.put(name, value=value)
    for name, count in trace.counts.items():
        trace.put(f"counts/{name}", value=count)
    trace.put(
        "rng/states",
        value=torch.frombuffer(
            bytearray(json.dumps(to_plain(trace.states)).encode()),
            dtype=torch.uint8,
        ),
    )
    return trace.record


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
