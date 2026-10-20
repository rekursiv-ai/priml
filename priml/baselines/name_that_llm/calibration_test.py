"""Replay complete source-minted sigmoid state and extreme-score outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import math

from treekle import from_plain

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm.calibration import (
    SigmoidCalibration,
    _binary_points,
    dot_product,
)
from priml.baselines.name_that_llm.logistic import LBFGS
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize("scale", argvalues=[1, 100])
@pytest.mark.parametrize("corrupt", argvalues=[False, True])
def test_sigmoid_replays_complete_source_state_and_predictions(
    scale: int,
    corrupt: bool,
) -> None:
    scores = np.sin(np.arange(7) / 3) * scale
    targets = np.array([0, 1, 0, 0, 1, 0, 1], dtype=np.int64)
    model = SigmoidCalibration.Config().make().fit(scores, targets=targets)
    if corrupt:
        model.a_ = math.nextafter(model.a_, math.inf)
    actual = {
        "a_": torch.tensor(model.a_, dtype=torch.float64),
        "b_": torch.tensor(model.b_, dtype=torch.float64),
        "predict": torch.from_numpy(
            model.predict(
                np.array([-1000, -37, -2, 0, 18, 1000], dtype=np.float64),
            ),
        ),
    }
    prefix = f"{scale}/"
    expected = {
        name.removeprefix(prefix): value
        for name, value in read_tensors(_CWD / "testdata" / "sigmoid.pt").items()
        if name.startswith(prefix)
    }
    report = mismatches(expected, actual=actual)
    if corrupt:
        assert any(item.startswith("a_:") for item in report), report
    else:
        assert not report, "\n".join(report)


@pytest.mark.parametrize("corrupt", argvalues=[False, True])
def test_binary_likelihood_matches_every_source_boundary_and_tail(
    corrupt: bool,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "binary_points.pt")
    loss, gradient = _binary_points(
        expected["scores"].numpy(),
        targets=expected["targets"].numpy(),
    )
    actual = dict(expected)
    actual["loss"], actual["gradient"] = (
        torch.from_numpy(loss),
        torch.from_numpy(gradient),
    )
    if corrupt:
        actual["gradient"][:1] = torch.nextafter(
            actual["gradient"][:1],
            other=torch.tensor(torch.inf),
        )
    report = mismatches(expected, actual=actual)
    if corrupt:
        assert any(item.startswith("gradient:") for item in report), report
    else:
        assert not report, "\n".join(report)


@pytest.mark.parametrize("column", argvalues=[0, 1, 2])
@pytest.mark.parametrize("strided", argvalues=[False, True])
def test_sigmoid_replays_all_source_grouped_calibration_inputs(
    column: int,
    strided: bool,
) -> None:
    record = read_tensors(_CWD / "testdata" / "svm_calibration.pt")
    prefix = f"{column}/"
    expected = {
        name.removeprefix(prefix): value
        for name, value in record.items()
        if name.startswith(prefix)
    }
    scores = expected["scores"].numpy()
    assert expected["strides"].tolist() == [scores.itemsize]
    if strided:
        columns = np.empty((len(scores), 3), dtype=np.float64)
        columns[:, 0] = scores
        scores = columns[:, 0]
    model = (
        SigmoidCalibration.Config()
        .make()
        .fit(
            scores,
            targets=expected["targets"].numpy(),
        )
    )
    actual = dict(expected)
    actual["a_"] = torch.tensor(model.a_, dtype=torch.float64)
    actual["b_"] = torch.tensor(model.b_, dtype=torch.float64)
    actual["predict"] = torch.from_numpy(model.predict(scores))
    report = mismatches(expected, actual=actual)
    assert not report, "\n".join(report)


@pytest.mark.parametrize("maximum", argvalues=[0, -1, np.nan, np.inf])
def test_sigmoid_rejects_invalid_rescaling_threshold(maximum: float) -> None:
    config = SigmoidCalibration.Config()
    config.max_magnitude = maximum
    with pytest.raises(
        ValueError,
        match=r"^Expected a positive finite calibration rescaling threshold\.$",
    ):
        config.make()


@pytest.mark.parametrize(("rows", "labels"), argvalues=[(0, 0), (2, 3)])
def test_sigmoid_rejects_empty_or_unaligned_scores(rows: int, labels: int) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Expected nonempty calibration scores aligned with targets\.$",
    ):
        SigmoidCalibration.Config().make().fit(
            np.zeros(rows, dtype=np.float64),
            targets=np.zeros(labels, dtype=np.int64),
        )


@pytest.mark.parametrize("corrupt", argvalues=[False, True])
def test_sigmoid_replays_complete_additional_source_state(corrupt: bool) -> None:
    """Replay the complete seven-row witness and detect one-ULP parameter drift.

    Args:
      corrupt: Whether to perturb the fitted slope by one representable value.

    """
    expected = from_plain(
        cast(
            object,
            torch.load(_CWD / "testdata" / "calibration.pt", weights_only=True),
        ),
        target=dict[str, torch.Tensor],
    )
    scores, targets = (
        expected["scores"].numpy().copy(),
        expected["targets"].numpy().copy(),
    )
    model = SigmoidCalibration.Config().make().fit(scores, targets=targets)
    actual = {
        "scores": torch.from_numpy(scores),
        "targets": torch.from_numpy(targets),
        "parameters": torch.tensor([model.a_, model.b_], dtype=torch.float64),
        "predict": torch.from_numpy(model.predict(scores)),
    }
    if corrupt:
        actual["parameters"][0] = torch.nextafter(
            actual["parameters"][0],
            other=torch.tensor(torch.inf),
        )
    report = mismatches(expected, actual=actual)
    if corrupt:
        assert any(item.startswith("parameters:") for item in report), report
    else:
        assert not report, "\n".join(report)


def test_default_calibration_keeps_the_shared_solver_and_source_budgets() -> None:
    config = SigmoidCalibration.Config()
    assert isinstance(config.optimizer.make(), LBFGS)
    assert config.optimizer.max_iter == 15_000
    assert config.optimizer.tolerance_gradient == 1e-6
    assert config.optimizer.max_line_search == 20


def test_dot_product_rejects_misaligned_vectors() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        dot_product(np.ones(2), right=np.ones(3))


@pytest.mark.parametrize("length", argvalues=[2, 7, 15])
def test_dot_product_keeps_fused_binary64_cancellation(length: int) -> None:
    left = np.zeros(length, dtype=np.float64)
    right = np.zeros(length, dtype=np.float64)
    left[:2], right[:2] = [1, 1 + 2**-27], [-1, 1 - 2**-27]
    assert dot_product(left, right=right) == -(2**-54)


@pytest.mark.parametrize("length", argvalues=[16, 17, 31, 32, 33, 63, 64, 65])
def test_dot_product_keeps_the_source_block_reduction_tree(length: int) -> None:
    right = np.zeros(length, dtype=np.float64)
    right[:3] = [1e16, 1, -1e16]
    assert dot_product(np.ones(length, dtype=np.float64), right=right) == 1


def test_dot_product_keeps_the_first_source_calibration_gradient() -> None:
    negative = float.fromhex("-0x1.1c71c71c71c72p-2")
    positive = float.fromhex("0x1.6c16c16c16c17p-2")
    left = np.array(
        [negative, positive, negative, negative, positive, negative, positive],
        dtype=np.float64,
    )
    right = np.array(
        [
            float.fromhex(value)
            for value in (
                "0x0.0p+0",
                "0x1.4f0c2068a80c6p-2",
                "0x1.3c9af78209765p-1",
                "0x1.aed548f090ceep-1",
                "0x1.f1a1d8383254ap-1",
                "0x1.fda61ca171a76p-1",
                "0x1.d18f6ead1b446p-1",
            )
        ],
    )
    assert dot_product(left, right=right) == float.fromhex("0x1.a6bb5c1c5be4ap-4")


def test_dot_product_keeps_an_empty_vector_sum() -> None:
    assert dot_product(np.empty(0), right=np.empty(0)) == 0


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
