"""Replay complete portable paths against the immutable source's stored values."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol

import math

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm.encoder import mean_pool
from priml.baselines.name_that_llm.metric import balanced_accuracy
from priml.baselines.name_that_llm.model import (
    StyleCounts,
    center,
    random_chunks,
)
from priml.baselines.name_that_llm.testing.fixtures_support_test import (
    capture_source,
    centred_branch,
    completed_record,
    miniature,
    plain_branch,
    recipe,
    svm_branch,
)
from priml.baselines.name_that_llm.testing.trace_support_test import NumericalTrace
from priml.baselines.name_that_llm.topic import TopicClusters
from priml.data.folds import grouped_folds
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "component",
    argvalues=[
        "center",
        "chunks",
        "blend",
        "topics",
        "folds",
        "pool",
        "metric",
    ],
)
def test_public_source_exact_outputs(component: str) -> None:
    golden = read_tensors(_CWD / "testdata" / "exp000.pt")
    expected = {
        name.removeprefix(f"{component}/"): value
        for name, value in golden.items()
        if name.startswith(f"{component}/")
    }
    assert expected
    with NumericalTrace() as trace:
        actual = port_outputs(component, dependencies=capture_source("blend"))
    report = mismatches(expected, actual=completed_record(trace, outputs=actual))
    assert not report, "\n".join(report)


def port_outputs(
    component: str,
    *,
    dependencies: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Run the native component with source-captured dependency outputs.

    Args:
      component: Native component to execute with the unchanged recipe.
      dependencies: Whole source-captured probabilities at branch boundaries.

    Returns:
      outputs: Complete component outputs in their original dtypes.

    """
    texts, targets, groups, embeddings = miniature()
    actual: dict[str, Tensor]
    if component == "style":
        actual = {
            "style": torch.from_numpy(
                StyleCounts().transform(["A\N{EM DASH}b!\n\nC...", ""]),
            ),
        }
    elif component == "center":
        actual = {"center": torch.from_numpy(center(embeddings[:, 0], groups=groups))}
    elif component == "chunks":
        chunks = random_chunks(groups, size=2, rng=np.random.default_rng(42))
        actual = {"chunks": torch.from_numpy(np.unique(chunks, return_inverse=True)[1])}
    elif component == "plain":
        actual = {
            "plain": torch.from_numpy(
                plain_branch()
                .make()
                .predict(embeddings, targets=targets, predict=embeddings[:3]),
            ),
        }
    elif component == "centred":
        actual = {
            "centred": torch.from_numpy(
                centred_branch()
                .make()
                .predict(
                    embeddings,
                    targets=targets,
                    groups=groups,
                    predict=embeddings[:3],
                    clusters=np.array([2, 2, 3]),
                    plain=dependencies["plain"].numpy(),
                ),
            ),
        }
    elif component == "svm":
        actual = {
            "svm": torch.from_numpy(
                svm_branch()
                .make()
                .predict(
                    texts,
                    targets=targets,
                    groups=groups,
                    predict_texts=texts[:3],
                ),
            ),
        }
    elif component == "blend":
        actual = {
            "blend": torch.from_numpy(
                recipe().blend.make()(
                    {
                        name: dependencies[name].numpy()
                        for name in ["svm", "plain", "centred"]
                    },
                ),
            ),
        }
    elif component == "pool":
        actual = {
            "pool": mean_pool(
                torch.arange(30, dtype=torch.float16).reshape(2, 3, 5),
                attention_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
            ),
        }
    elif component == "metric":
        actual = {
            "metric": torch.from_numpy(
                np.asarray(
                    balanced_accuracy(
                        targets,
                        predictions=np.arange(54).reshape(18, 3).argmax(1),
                        classes=3,
                    ),
                ),
            ),
        }
    elif component == "topics":
        actual = {"topics": torch.from_numpy(TopicClusters.Config().make()(texts))}
    else:
        actual = {
            f"fold/{index}": torch.from_numpy(valid)
            for index, (_, valid) in enumerate(grouped_folds(groups, count=3))
        }
    return actual


def test_source_golden_detects_one_ulp_in_blend_weight() -> None:
    expected = capture_source("blend")
    config = recipe().blend
    config.weights["svm"] = math.nextafter(config.weights["svm"], math.inf)
    actual = torch.from_numpy(
        config.make()(
            {name: expected[name].numpy() for name in ["svm", "plain", "centred"]},
        ),
    )
    report = mismatches({"blend": expected["blend"]}, actual={"blend": actual})
    assert report
    assert report[0].startswith("blend:")


def test_source_golden_inventory(record_testsuite_property: SuiteProperty) -> None:
    path = _CWD / "testdata" / "exp000.pt"
    record = read_tensors(path)
    assert path.stat().st_size <= 32_768
    assert record["input/embeddings"].shape == (18, 4, 2)
    assert record["input/embeddings"].dtype == torch.float32
    assert {name for name in record if name.startswith("input/")} == {
        "input/embeddings",
        "input/plain",
        "input/centred",
        "input/svm",
    }
    assert {name.split("/", maxsplit=1)[0] for name in record} == {
        "input",
        "center",
        "chunks",
        "topics",
        "folds",
        "blend",
        "pool",
        "metric",
    }
    for name, value in {
        "golden_fields": len(record),
        "golden_elements": sum(value.numel() for value in record.values()),
        "golden_bytes": path.stat().st_size,
        "golden_source_sha256": (
            "946c7138b4e8685fb36327c0a6c4d1ef50a21acdb62746c46c44b99a228ec1aa"
        ),
    }.items():
        record_testsuite_property(name, value=value)


@pytest.mark.parametrize(
    ("component", "golden_name"),
    argvalues=[
        ("plain", "qwen_plain"),
        ("centred", "qwen_centred"),
        ("svm", "svm"),
        ("style", "style"),
    ],
)
def test_complete_fitted_state_replays_from_source_values(
    component: str,
    golden_name: str,
) -> None:
    expected = read_tensors(_CWD / "testdata" / f"{golden_name}.pt")
    with NumericalTrace() as trace:
        actual = port_outputs(component, dependencies=capture_source("blend"))
    report = mismatches(expected, actual=completed_record(trace, outputs=actual))
    assert not report, "\n".join(report)


def test_complete_svm_state_replay_detects_one_ulp() -> None:
    """Reject a one-ULP output perturbation while comparing every committee field."""
    expected = read_tensors(_CWD / "testdata" / "svm.pt")
    with NumericalTrace() as trace:
        outputs = port_outputs("svm", dependencies=capture_source("blend"))
    actual = completed_record(trace, outputs=outputs)
    assert not mismatches(expected, actual=actual)
    actual["svm"][0, 0] = torch.nextafter(
        actual["svm"][0, 0],
        other=torch.tensor(torch.inf),
    )
    report = mismatches(expected, actual=actual)
    assert len(report) == 1
    assert report[0].startswith("svm:")


class SuiteProperty(Protocol):
    """The named callback supplied by pytest's JUnit suite fixture."""

    def __call__(self, name: str, value: object) -> None: ...


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
