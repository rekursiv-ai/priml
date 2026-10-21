"""Check the paired construction, causal rollout, and linear-model limit."""

import pytest
import torch

from priml.baselines.oscillator.data import STATE_CHANNELS, Oscillator, Windows
from priml.baselines.oscillator.delayed import (
    DelayedCalibration,
    DelayedData,
    DelayedOscillator,
    DelayedStudy,
    PairedHistory,
    Pulse,
    SignedHistory,
)
from priml.baselines.oscillator.experiments import exp009, exp019
from priml.baselines.oscillator.nonlinear import KernelCorrection, PolynomialKernel
from priml.baselines.oscillator.search import BoundedScalarSearch
from priml.baselines.oscillator.train_step import RidgeCorrection
from priml.testing.golden import assert_pprint_golden


def test_quartets_share_the_last_state_but_have_opposite_responses() -> None:
    data = _data_config().make()
    before = torch.get_rng_state().clone()
    windows = data.windows(data.config.train, simulator=data.reference)
    media = windows.media.reshape(2, 4, 12, STATE_CHANNELS)
    label = windows.label.reshape(2, 4, 3, STATE_CHANNELS)
    assert torch.count_nonzero(media[:, :, -4:]) == 0
    assert torch.equal(media[:, 0] + media[:, 3], torch.zeros_like(media[:, 0]))
    assert torch.equal(media[:, 1] + media[:, 2], torch.zeros_like(media[:, 1]))
    torch.testing.assert_close(label[:, 0], label[:, 3], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(label[:, 0], -label[:, 1], atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(label[:, 1], label[:, 2], atol=1e-7, rtol=1e-7)
    assert label.square().sum() > 0
    assert torch.equal(before, torch.get_rng_state())


def test_best_affine_forecast_cannot_beat_the_zero_response() -> None:
    data = _data_config().make()
    train = data.windows(data.config.train, simulator=data.reference)
    features = train.media.flatten(start_dim=1).double()
    target = train.label.flatten(start_dim=1).double()
    design = torch.cat((features, torch.ones(len(features), 1)), dim=1)
    # A complete sign quartet is orthogonal to its product response.
    torch.testing.assert_close(
        design.T @ target,
        torch.zeros(design.shape[1], target.shape[1]),
        atol=1e-8,
        rtol=0,
        check_dtype=False,
    )
    coefficients = (
        torch.arange(design.shape[1] * target.shape[1])
        .reshape(design.shape[1], target.shape[1])
        .double()
        / 100
    )
    prediction = design @ coefficients
    torch.testing.assert_close(
        (prediction - target).square().mean(),
        prediction.square().mean() + target.square().mean(),
        atol=1e-9,
        rtol=1e-9,
    )
    assert (prediction - target).square().mean() >= target.square().mean() - 1e-12


def test_pulse_peaks_at_unit_height_and_its_analytic_peak_speed() -> None:
    cfg = Pulse.Config()
    cfg.width = 4
    values = cfg.make().sample(steps=7, step_size=0.2)
    assert values[2, 0] == pytest.approx(1.0)
    assert values[1, 1] == pytest.approx(torch.pi / (cfg.width * 0.2))
    assert torch.count_nonzero(values[4:]) == 0


def test_history_sampling_is_reproducible_and_keeps_split_config() -> None:
    cfg = _data_config()
    sampler = cfg.history.make()
    states = cfg.train.copy_tree()
    first = sampler.sample(states, steps=cfg.seq_len, step_size=cfg.cheap.step_size)
    assert states == cfg.train
    assert torch.equal(
        first,
        sampler.sample(states, steps=cfg.seq_len, step_size=cfg.cheap.step_size),
    )
    states.seed += 1000
    assert not torch.equal(
        first,
        sampler.sample(states, steps=cfg.seq_len, step_size=cfg.cheap.step_size),
    )


def test_first_pulse_height_and_later_peak_speed_follow_the_split_bounds() -> None:
    cfg = _data_config()
    media = cfg.history.make().sample(
        cfg.shifted,
        steps=cfg.seq_len,
        step_size=cfg.cheap.step_size,
    )
    assert isinstance(cfg.history, PairedHistory.Config)
    first, second = cfg.history.first, cfg.history.second
    height = media[:, first.start + first.width // 2, 0].abs()
    low, high = cfg.shifted.amplitude
    assert ((height >= low) & (height <= high)).all()
    speed = media[:, second.start : second.start + second.width, 1].abs()
    assert speed.max() <= cfg.shifted.velocity
    assert speed.max() > cfg.shifted.velocity / 2


@pytest.mark.parametrize("width", [0, 2])
def test_short_or_aliased_pulse_is_rejected(width: int) -> None:
    with pytest.raises(ValueError, match="at least three"):
        Pulse.Config(width=width).make()


def test_overlapping_pulses_are_rejected() -> None:
    cfg = PairedHistory.Config()
    cfg.second.start = cfg.first.start
    with pytest.raises(ValueError, match="overlap"):
        cfg.make()


@pytest.mark.parametrize("bad", ["count", "tail"])
def test_histories_need_complete_groups_and_a_quiet_tail(bad: str) -> None:
    cfg = _data_config()
    if bad == "count":
        cfg.train.count = 7
    else:
        assert isinstance(cfg.history, PairedHistory.Config)
        cfg.seq_len = cfg.history.second.start + cfg.history.second.width
    with pytest.raises(ValueError, match=r"four-history|quiet tail"):
        cfg.history.make().sample(
            cfg.train,
            steps=cfg.seq_len,
            step_size=cfg.cheap.step_size,
        )


@pytest.mark.parametrize("bad", ["horizon", "same-pulse", "no-overlap"])
def test_task_rejects_delays_that_break_the_pairing(bad: str) -> None:
    cfg = _data_config()
    assert isinstance(cfg.reference, DelayedOscillator.Config)
    if bad == "horizon":
        cfg.pred_len = min(cfg.reference.lags) + 1
    elif bad == "same-pulse":
        cfg.reference.lags = [6, 6]
    else:
        cfg.reference.lags = [3, 3]
    with pytest.raises(
        ValueError,
        match=r"within all feedback|different history|visible pulse",
    ):
        cfg.make()


def test_delayed_force_reads_positions_at_the_configured_lags() -> None:
    cfg = DelayedOscillator.Config(lags=[3, 2], substeps=1)
    media = torch.tensor([[[0.6, 0.0], [0.5, 0.1], [0.3, -0.2]]])
    position, velocity = media[:, -1].unbind(dim=-1)
    force = cfg.gain * torch.tanh(media[:, -3, 0] * media[:, -2, 0])
    acceleration = -cfg.stiffness * position - cfg.damping * velocity + force
    next_velocity = velocity + cfg.step_size * acceleration
    next_position = position + cfg.step_size * next_velocity
    torch.testing.assert_close(
        cfg.make().forecast_history(media, steps=1)[:, 0],
        torch.stack((next_position, next_velocity), dim=-1),
    )


def test_history_rollout_uses_predictions_after_the_observed_delays() -> None:
    cfg = DelayedOscillator.Config()
    cfg.lags = [3, 2]
    cfg.substeps = 1
    media = torch.tensor(
        [
            [[0.2, 0.1], [0.3, 0.0], [0.4, -0.1]],
            [[-0.3, 0.2], [0.1, -0.2], [0.5, 0.1]],
            [[0.6, 0.0], [0.2, 0.2], [0.1, -0.3]],
            [[0.4, 0.1], [0.2, -0.1], [0.3, 0.0]],
        ],
    )
    full = cfg.make().forecast_history(media, steps=5)
    first = cfg.make().forecast_history(media, steps=2)
    extended = torch.cat((media, first), dim=1)
    tail = cfg.make().forecast_history(extended, steps=3)
    assert torch.equal(full[:, :2], first)
    assert torch.equal(full[:, 2:], tail)


@pytest.mark.parametrize("substeps", [None, 1, 8])
def test_constant_prehistory_is_explicit_and_gain_zero_matches_base(
    substeps: int | None,
) -> None:
    cfg = DelayedOscillator.Config(lags=[4, 3])
    states = torch.tensor([[0.3, -0.2], [-0.8, 0.5], [1.1, 0.4]])
    assert torch.equal(
        cfg.make().forecast(states, steps=5, substeps=substeps),
        cfg.make().forecast_history(
            states[:, None].expand(-1, 4, -1),
            steps=5,
            substeps=substeps,
        ),
    )
    cfg.gain = 0
    base = Oscillator.Config()
    assert torch.equal(
        cfg.make().forecast(states, steps=5, substeps=substeps),
        base.make().forecast(states, steps=5, substeps=substeps),
    )
    with pytest.raises(ValueError, match="position/velocity"):
        cfg.make().forecast(torch.zeros(3, 4, 2), steps=3)


@pytest.mark.parametrize("bad", ["lag", "gain"])
def test_invalid_delay_settings_are_rejected(bad: str) -> None:
    cfg = DelayedOscillator.Config()
    if bad == "lag":
        cfg.lags[0] = 0
    else:
        cfg.gain = float("inf")
    with pytest.raises(ValueError, match=r"lags|gain"):
        cfg.make()


@pytest.mark.parametrize("substeps", [0, -1])
def test_delayed_substep_override_requires_a_positive_count(substeps: int) -> None:
    simulator = DelayedOscillator.Config().make()
    states = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="substeps"):
        simulator.forecast(states, steps=5, substeps=substeps)


@pytest.mark.parametrize(
    "bad",
    ["short", "integer", "rank", "channels", "horizon"],
)
def test_rollout_rejects_invalid_history(bad: str) -> None:
    cfg = DelayedOscillator.Config(lags=[4, 3])
    media = torch.zeros(5, 6, 2)
    steps = 3
    if bad == "short":
        media = media[:, :2]
    elif bad == "integer":
        media = media.long()
    elif bad == "rank":
        media = media[0]
    elif bad == "channels":
        media = torch.cat((media, media), dim=-1)
    else:
        steps = 0
    with pytest.raises(ValueError, match="floating histories"):
        cfg.make().forecast_history(media, steps=steps)


def test_nonfinite_history_propagates_into_the_rollout() -> None:
    media = torch.zeros(5, 6, 2)
    media[0, -4, 0] = float("nan")
    prediction = (
        DelayedOscillator.Config(lags=[4, 3])
        .make()
        .forecast_history(
            media,
            steps=3,
        )
    )
    assert torch.isnan(prediction[0]).all()
    assert torch.equal(prediction[1:], torch.zeros_like(prediction[1:]))


def test_skill_is_zero_for_linear_ridge_and_undefined_for_zero_targets() -> None:
    cfg = DelayedStudy.Config()
    cfg.data = _data_config()
    cfg.method = RidgeCorrection.Config()
    cfg.seed = 7
    record = cfg.make().run()
    assert record["forecast_skill"]["familiar"] == pytest.approx(0, abs=1e-10)
    assert record["forecast_skill"]["shifted"] == pytest.approx(0, abs=1e-10)
    cfg.data.reference = cfg.data.cheap.copy_tree()
    assert cfg.make().run()["forecast_skill"] == {"familiar": None, "shifted": None}


def test_a_delayed_cheap_simulator_forecasts_from_the_observed_history() -> None:
    cfg = DelayedStudy.Config()
    cfg.data = _data_config()
    cfg.data.cheap = cfg.data.reference.copy_tree()
    cfg.method = RidgeCorrection.Config()
    cfg.seed = 7
    scores = cfg.make().run()["scores"]
    assert [split["cheap_simulator"]["mse"] for split in scores.values()] == [0, 0]
    assert [split["method"]["mse"] for split in scores.values()] == [0, 0]


def test_privileged_control_fits_gain_and_forecasts_unseen_magnitudes() -> None:
    cfg = _data_config()
    data = cfg.make()
    train = data.windows(cfg.train, simulator=data.reference)
    method_cfg = DelayedCalibration.Config()
    assert isinstance(cfg.reference, DelayedOscillator.Config)
    method_cfg.simulator = cfg.reference.copy_tree()
    method = method_cfg.make()
    unused = Windows(
        media=torch.full_like(train.media, float("nan")),
        label=train.label,
    )
    before = cfg.reference.copy_tree()
    report = method.fit(
        train,
        validation=unused,
        synthetic=unused,
        simulator=data.cheap,
    )
    assert report["fitted_gain"] == pytest.approx(cfg.reference.gain, abs=2e-6)
    assert report["training_scaled_mse"] < 1e-12
    assert report["simulator_evaluations"] > 1
    assert report["updates"] == report["best_update"] == report["pretrain_updates"] == 0
    test = data.windows(cfg.shifted, simulator=data.reference)
    torch.testing.assert_close(
        method.predict(test.media),
        test.label,
        atol=2e-6,
        rtol=2e-6,
    )
    assert cfg.reference == before


def test_known_equation_keeps_an_exact_zero_gain_without_delayed_feedback() -> None:
    cfg = _data_config()
    cfg.reference = cfg.cheap.copy_tree()
    data = cfg.make()
    train = data.windows(cfg.train, simulator=data.reference)
    method_cfg = DelayedCalibration.Config()
    method_cfg.simulator.lags = [10, 6]
    method = method_cfg.make()
    report = method.fit(train, validation=train, synthetic=train, simulator=data.cheap)
    # The bounded search never evaluates its endpoints, so only the explicit
    # zero-gain candidate can return exactly zero.
    assert report["fitted_gain"] == 0
    assert torch.equal(method.predict(train.media), train.label)


def test_failed_delayed_calibration_has_no_prediction_state() -> None:
    cfg = _data_config()
    data = cfg.make()
    train = data.windows(cfg.train, simulator=data.reference)
    method_cfg = DelayedCalibration.Config()
    assert isinstance(cfg.reference, DelayedOscillator.Config)
    method_cfg.simulator = cfg.reference.copy_tree()
    assert isinstance(method_cfg.search, BoundedScalarSearch.Config)
    method_cfg.search.max_evaluations = 1
    method = method_cfg.make()
    with pytest.raises(RuntimeError, match="converged"):
        method.fit(train, validation=train, synthetic=train, simulator=data.cheap)
    with pytest.raises(RuntimeError, match="Fit the delayed"):
        method.predict(train.media)


def test_test_changes_cannot_change_the_fitted_delay_gain() -> None:
    cfg = DelayedStudy.Config()
    cfg.data = _data_config()
    cfg.seed = 7
    method = DelayedCalibration.Config()
    assert isinstance(cfg.data.reference, DelayedOscillator.Config)
    method.simulator = cfg.data.reference.copy_tree()
    cfg.method = method
    before_rng = torch.get_rng_state().clone()
    first = cfg.make().run()
    assert first["forecast_skill"]["familiar"] == pytest.approx(1, abs=1e-6)
    cfg.data.test.seed += 1000
    cfg.data.shifted.amplitude = (2.5, 3.0)
    second = cfg.make().run()
    assert first["training"] == second["training"]
    assert first["reference_budget"] == second["reference_budget"]
    assert first["scores"] != second["scores"]
    assert torch.equal(before_rng, torch.get_rng_state())


def test_task_configuration_is_fully_printable() -> None:
    assert_pprint_golden(test_file=__file__, name="exp009", config=exp009())


def test_four_way_configuration_is_fully_printable() -> None:
    assert_pprint_golden(test_file=__file__, name="exp019", config=exp019())


def test_fourth_order_kernel_and_known_equation_can_learn_the_four_way_response() -> (
    None
):
    cfg = _four_way_config()
    data = cfg.make()
    train = data.windows(cfg.train, simulator=data.reference)
    quadratic = KernelCorrection.Config().make()
    quadratic.fit(train, validation=train, synthetic=train, simulator=data.cheap)
    torch.testing.assert_close(
        quadratic.predict(train.media),
        torch.zeros_like(train.label),
        atol=1e-8,
        rtol=0,
    )
    method_cfg = KernelCorrection.Config()
    assert isinstance(method_cfg.kernel, PolynomialKernel.Config)
    method_cfg.kernel.degree = 4
    quartic = method_cfg.make()
    quartic.fit(train, validation=train, synthetic=train, simulator=data.cheap)
    assert (quartic.predict(train.media) - train.label).square().mean() < (
        quadratic.predict(train.media) - train.label
    ).square().mean()
    oracle_cfg = DelayedCalibration.Config()
    assert isinstance(cfg.reference, DelayedOscillator.Config)
    oracle_cfg.simulator = cfg.reference.copy_tree()
    oracle = oracle_cfg.make()
    oracle.fit(train, validation=train, synthetic=train, simulator=data.cheap)
    torch.testing.assert_close(
        oracle.predict(train.media),
        train.label,
        atol=1e-7,
        rtol=1e-5,
    )


def test_four_way_response_is_orthogonal_to_every_feature_below_degree_four() -> None:
    cfg = _four_way_config()
    data = cfg.make()
    windows = data.windows(cfg.train, simulator=data.reference)
    history = windows.media.reshape(2, 16, 20, STATE_CHANNELS)
    target = windows.label.flatten(start_dim=1).double()
    features = windows.media.flatten(start_dim=1).double()
    assert torch.count_nonzero(history[:, :, -4:]) == 0
    assert target.square().sum() > 0
    moments = {
        "constant": target.sum(dim=0),
        "linear": torch.einsum("ni,nt->it", features, target),
        "quadratic": torch.einsum("ni,nj,nt->ijt", features, features, target),
        "cubic": torch.einsum(
            "ni,nj,nk,nt->ijkt",
            features,
            features,
            features,
            target,
        ),
    }
    for moment in moments.values():
        torch.testing.assert_close(
            moment,
            torch.zeros_like(moment),
            atol=1e-8,
            rtol=0,
        )
    assert torch.equal(
        windows.media,
        data.windows(cfg.train, simulator=data.reference).media,
    )
    assert torch.equal(
        windows.label,
        data.windows(cfg.train, simulator=data.reference).label,
    )


@pytest.mark.parametrize("bad", ["order", "overlap", "count", "tail", "lags"])
def test_general_sign_groups_reject_incomplete_or_invalid_geometry(bad: str) -> None:
    cfg = _four_way_config()
    assert isinstance(cfg.history, SignedHistory.Config)
    if bad == "order":
        cfg.history.pulses = cfg.history.pulses[:1]
    elif bad == "overlap":
        cfg.history.pulses[1].start = 0
    elif bad == "count":
        cfg.train.count = 17
    elif bad == "tail":
        cfg.seq_len = 16
        cfg.reference = cfg.cheap.copy_tree()
    else:
        assert isinstance(cfg.reference, DelayedOscillator.Config)
        cfg.reference.lags = cfg.reference.lags[:2]
    with pytest.raises(
        ValueError,
        match=r"at least two|nonoverlapping|complete interaction|quiet tail|one feedback",
    ):
        cfg.make().windows(cfg.train, simulator=cfg.reference.make())


def _four_way_config() -> DelayedData.Config:
    cfg = DelayedData.Config()
    cfg.seq_len, cfg.pred_len = 20, 3
    cfg.history = SignedHistory.Config(
        pulses=[Pulse.Config(start=start, width=4) for start in (0, 4, 8, 12)],
    )
    assert isinstance(cfg.reference, DelayedOscillator.Config)
    cfg.reference.lags = [18, 14, 10, 6]
    cfg.train.count, cfg.validation.count = 32, 48
    cfg.test.count, cfg.shifted.count = 64, 80
    return cfg


def _data_config() -> DelayedData.Config:
    cfg = DelayedData.Config()
    cfg.seq_len, cfg.pred_len = 12, 3
    assert isinstance(cfg.history, PairedHistory.Config)
    cfg.history.first.width = 4
    cfg.history.second.start, cfg.history.second.width = 4, 4
    assert isinstance(cfg.reference, DelayedOscillator.Config)
    cfg.reference.lags = [10, 6]
    cfg.train.count, cfg.validation.count = 8, 12
    cfg.test.count, cfg.shifted.count = 20, 24
    return cfg


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
