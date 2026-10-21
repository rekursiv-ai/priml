"""Check force selection, data separation, and failed candidate handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from configgle import Fig

import pytest
import torch

from priml.baselines.oscillator.data import ForecastData, Oscillator, Windows
from priml.baselines.oscillator.discovery import SparseSimulator, _ForceResiduals
from priml.baselines.oscillator.experiments import exp005
from priml.baselines.oscillator.forces import (
    CorrectedOscillator,
    PowerForce,
    WeightedForce,
)
from priml.baselines.oscillator.search import SearchResult
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


def test_selects_both_missing_forces_and_rejects_a_distractor() -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    validation = data.windows(data.config.validation, simulator=data.reference)
    method = _method_config().make()
    report = method.fit(
        train,
        validation=validation,
        synthetic=train,
        simulator=data.cheap,
    )
    assert [term["name"] for term in report["fitted_forces"]] == [
        "-position*abs(position)^2",
        "-velocity*abs(velocity)^1",
    ]
    assert [term["coefficient"] for term in report["fitted_forces"]] == pytest.approx(
        [0.23, 0.31],
        abs=2e-5,
    )
    assert report["candidates_tried"] == report["candidates_converged"] == 6
    assert report["validation_evaluations"] == 7
    assert report["training_evaluations"] > report["candidates_tried"]
    assert "validation_scaled_mse" in report
    assert report["validation_scaled_mse"] < 1e-10
    assert report["selection_score"] == pytest.approx(
        2 * method.config.complexity_penalty,
        abs=1e-10,
    )
    assert report["updates"] == report["best_update"] == report["pretrain_updates"] == 0
    shifted = data.windows(data.config.shifted, simulator=data.reference)
    prediction = method.predict(shifted.media)
    assert prediction.shape == shifted.label.shape
    assert prediction.dtype == shifted.media.dtype
    torch.testing.assert_close(prediction, shifted.label, atol=1e-5, rtol=1e-5)
    assert data.cheap.config.cubic == 0


def test_validation_windows_not_training_fit_choose_the_terms() -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    # The cheap simulator generated these, so no added force can beat it here.
    validation = data.windows(data.config.validation, simulator=data.cheap)
    method = _method_config().make()
    report = method.fit(
        train,
        validation=validation,
        synthetic=train,
        simulator=data.cheap,
    )
    assert report["fitted_forces"] == []
    assert report["candidates_converged"] == 6


def test_missing_term_leaves_error_on_larger_motion() -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    validation = data.windows(data.config.validation, simulator=data.reference)
    cfg = _method_config()
    cfg.terms = [PowerForce.Config(power=3)]
    cfg.max_terms = 1
    method = cfg.make()
    method.fit(train, validation=validation, synthetic=train, simulator=data.cheap)
    shifted = data.windows(data.config.shifted, simulator=data.reference)
    assert (method.predict(shifted.media) - shifted.label).square().mean() > 1e-5


@pytest.mark.parametrize("penalty", [1e-6, 1e6])
def test_matching_simulator_needs_no_extra_terms(penalty: float) -> None:
    cfg = _data_config()
    cfg.reference = cfg.cheap.copy_tree()
    data = cfg.make()
    train = data.windows(cfg.train, simulator=data.reference)
    method_cfg = _method_config()
    method_cfg.terms = [PowerForce.Config(power=3)]
    method_cfg.max_terms = 1
    method_cfg.complexity_penalty = penalty
    method = method_cfg.make()
    report = method.fit(train, validation=train, synthetic=train, simulator=data.cheap)
    assert report["fitted_forces"] == []
    assert torch.equal(method.predict(train.media), train.label)


def test_zero_term_budget_keeps_the_supplied_simulator() -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    cfg = _method_config()
    cfg.max_terms = 0
    method = cfg.make()
    report = method.fit(train, validation=train, synthetic=train, simulator=data.cheap)
    assert report["candidates_tried"] == report["training_evaluations"] == 0
    assert report["validation_evaluations"] == 1
    assert report["fitted_forces"] == []
    assert torch.equal(
        method.predict(train.media),
        data.cheap.forecast(train.media[:, -1], steps=train.label.shape[1]),
    )


def test_validation_and_synthetic_data_never_enter_coefficient_fitting() -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    validation = data.windows(data.config.validation, simulator=data.reference)
    cfg = _method_config()
    cfg.search = _ScriptedSearch.Config()
    method = cfg.make()
    method.fit(train, validation=validation, synthetic=train, simulator=data.cheap)
    assert isinstance(method.search, _ScriptedSearch)
    first = list(method.search.residuals)
    method.search.residuals.clear()
    unused = Windows(
        media=torch.full_like(train.media, float("nan")),
        label=train.label,
    )
    changed = Windows(media=validation.media, label=validation.label + 0.5)
    method.fit(train, validation=changed, synthetic=unused, simulator=data.cheap)
    assert len(first) == len(method.search.residuals)
    for before, after in zip(first, method.search.residuals, strict=True):
        assert torch.equal(before, after)


def test_test_data_cannot_change_selection_or_coefficients() -> None:
    cfg = exp005()
    cfg.seed = 7
    cfg.data = _data_config()
    cfg.method = _method_config()
    cfg.method.search = _ScriptedSearch.Config()
    before_rng = torch.get_rng_state().clone()
    first = cfg.make().run()
    cfg.data.test.seed += 1000
    cfg.data.shifted.amplitude = (2.6, 3.1)
    second = cfg.make().run()
    assert first["training"] == second["training"]
    assert first["scores"] != second["scores"]
    assert first["reference_budget"] == second["reference_budget"]
    assert torch.equal(before_rng, torch.get_rng_state())


@pytest.mark.parametrize("failure", ["convergence", "nan", "negative", "shape"])
def test_invalid_candidates_fall_back_to_supplied_simulator(failure: str) -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    cfg = _method_config()
    search = _ScriptedSearch.Config()
    if failure == "convergence":
        search.success = False
    elif failure == "nan":
        search.value = float("nan")
    elif failure == "negative":
        search.value = -0.1
    else:
        search.extra = 1
    cfg.search = search
    method = cfg.make()
    report = method.fit(train, validation=train, synthetic=train, simulator=data.cheap)
    assert report["candidates_tried"] == 6
    assert report["candidates_converged"] == 0
    # Each candidate's starting residuals are checked before its one search call.
    assert report["training_evaluations"] == 12
    assert report["validation_evaluations"] == 1
    assert report["fitted_forces"] == []


@pytest.mark.parametrize("bad", ["budget", "penalty", "initial", "duplicate"])
def test_bad_configuration_is_rejected(bad: str) -> None:
    cfg = _method_config()
    if bad == "budget":
        cfg.max_terms = len(cfg.terms) + 1
    elif bad == "penalty":
        cfg.complexity_penalty = -0.1
    elif bad == "initial":
        cfg.initial_coefficient = float("nan")
    else:
        cfg.terms.append(cfg.terms[0])
    with pytest.raises(ValueError, match=r"budget|nonnegative|unique"):
        cfg.make()


@pytest.mark.parametrize("bad", ["nan", "horizon"])
def test_invalid_validation_windows_are_rejected(bad: str) -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    media, label = train.media, train.label
    if bad == "nan":
        media = torch.full_like(media, float("nan"))
    else:
        label = label[:, :3]
    with pytest.raises(ValueError, match=r"finite validation|horizons"):
        _method_config().make().fit(
            train,
            validation=Windows(media=media, label=label),
            synthetic=train,
            simulator=data.cheap,
        )


def test_nonfinite_control_is_rejected_and_does_not_leave_a_fit() -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    cfg = data.cheap.config.copy_tree()
    cfg.stiffness = 1e30
    method = _method_config().make()
    with pytest.raises(ValueError, match="finite validation score"):
        method.fit(train, validation=train, synthetic=train, simulator=cfg.make())
    with pytest.raises(RuntimeError, match="Fit the simulator"):
        method.predict(train.media)


def test_existing_corrections_are_preserved_without_mutation() -> None:
    cfg = CorrectedOscillator.Config()
    cfg.corrections = [WeightedForce.Config(coefficient=0.2)]
    original = cfg.copy_tree()
    states = torch.tensor([[0.3, -0.2], [-0.8, 0.5], [1.1, 0.4]], dtype=torch.float64)
    targets = cfg.make().forecast(states, steps=1)[:, 0]
    loss = _ForceResiduals(
        cfg,
        terms=[PowerForce.Config(axis=1, power=2)],
        states=states,
        targets=targets,
        scale=torch.ones(2),
    )
    zero = torch.zeros(1, dtype=torch.float64)
    assert torch.equal(loss(zero), torch.zeros(states.numel(), dtype=torch.float64))
    corrected = loss.simulator(torch.tensor([0.3], dtype=torch.float64))
    assert [term.coefficient for term in corrected.corrections] == [0.2, 0.3]
    assert cfg == original


@pytest.mark.parametrize("overflow", ["prediction", "scaling"])
def test_overflowing_candidate_is_skipped_before_the_search(overflow: str) -> None:
    if overflow == "prediction":
        data_cfg = _data_config()
        data_cfg.train.amplitude = (2.0, 3.0)
        data = data_cfg.make()
        train = data.windows(data_cfg.train, simulator=data.reference)
        simulator, power = data.cheap, 10_000
    else:
        # 2**1023 is finite, but dividing by the floored 1e-6 scale overflows.
        state = torch.tensor([[[2.0, 0.0]]], dtype=torch.float64)
        train = Windows(media=state, label=state.clone())
        simulator = Oscillator.Config(substeps=1, stiffness=1e-300, damping=0.0).make()
        power = 1023
    cfg = _method_config()
    cfg.terms = [PowerForce.Config(power=power)]
    cfg.max_terms = 1
    cfg.search = _ScriptedSearch.Config()
    method = cfg.make()
    report = method.fit(train, validation=train, synthetic=train, simulator=simulator)
    assert report["candidates_tried"] == report["training_evaluations"] == 1
    assert report["candidates_converged"] == 0
    assert report["fitted_forces"] == []
    assert isinstance(method.search, _ScriptedSearch)
    assert method.search.residuals == []


def test_nonfinite_validation_candidate_cannot_be_selected() -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    cfg = _method_config()
    cfg.search = _ScriptedSearch.Config(value=1e200)
    report = cfg.make().fit(
        train,
        validation=train,
        synthetic=train,
        simulator=data.cheap,
    )
    assert report["candidates_converged"] == 6
    assert report["fitted_forces"] == []


def test_discovery_config_is_fully_printable() -> None:
    assert_pprint_golden(test_file=__file__, name="exp005", config=exp005())


class _ScriptedSearch:
    class Config(Fig["_ScriptedSearch"]):
        success: bool = True
        """Whether the fake search converges."""

        value: float = 0.2
        """Returned strength for every candidate term."""

        extra: int = 0
        """Extra coefficients used to check malformed search results."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.residuals: list[Tensor] = []

    def __call__(
        self,
        fun: Callable[[Tensor], Tensor],
        x0: Tensor,
        /,
    ) -> SearchResult[Tensor]:
        self.residuals.append(fun(x0))
        return SearchResult(
            x=torch.full(
                (len(x0) + self.config.extra,),
                self.config.value,
                dtype=torch.float64,
            ),
            success=self.config.success,
        )


def _data_config() -> ForecastData.Config:
    cfg = ForecastData.Config()
    cfg.seq_len = 4
    cfg.pred_len = 5
    cfg.train.count = 3
    cfg.validation.count = 7
    cfg.test.count = 6
    cfg.shifted.count = 8
    cfg.cheap.substeps = 1
    reference = cfg.reference = CorrectedOscillator.Config()
    reference.substeps = 1
    reference.cubic = 0.23
    reference.corrections = [
        WeightedForce.Config(term=PowerForce.Config(axis=1, power=2), coefficient=0.31),
    ]
    return cfg


def _method_config() -> SparseSimulator.Config:
    cfg = SparseSimulator.Config()
    cfg.terms = [
        PowerForce.Config(power=3),
        PowerForce.Config(axis=1, power=2),
        PowerForce.Config(power=5),
    ]
    return cfg


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
