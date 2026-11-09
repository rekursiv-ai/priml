"""Check the privileged joint calibration on a tiny, clean configuration."""

import pytest

from priml.baselines.oscillator.chained import ChainedData, StuckSensor
from priml.baselines.oscillator.forced import ForcedData, GlitchySensor
from priml.baselines.oscillator.scripts import joint_calibration


def tiny() -> ChainedData:
    cfg = ChainedData.Config(
        seq_len=48,
        pred_len=8,
        recent_steps=(12, 14),
        segment_steps=(8, 10),
        outage=(2, 3),
    )
    cfg.observation = StuckSensor.Config(noise=(0, 0), spike_rate=0, stuck_rate=0)
    cfg.dynamics.substeps = 1
    return cfg.make()


def test_near_truth_start_sits_at_the_truth_and_forecasts() -> None:
    rows = joint_calibration.calibrate(
        tiny(),
        episodes=3,
        restarts=1,
        iterations=0,
        seed=3,
        near_truth=True,
    )
    assert len(rows) == 3
    assert max(r.regimes for r in rows) > 1
    for row in rows:
        assert row.best_loss < 0.1
        assert row.truth_loss < 0.05
        assert row.skill > 0.5


def test_random_restarts_run_and_main_prints_each_episode(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = joint_calibration.calibrate(
        tiny(),
        episodes=2,
        restarts=2,
        iterations=2,
        seed=4,
    )
    assert all(r.best_loss >= 0 for r in rows)
    data = tiny()

    def make(_: object) -> ChainedData:
        return data

    monkeypatch.setattr(ChainedData.Config, "make", make)
    flags = ["--episodes", "2", "--restarts", "1", "--iterations", "1"]
    assert joint_calibration.main(flags) == 0
    assert capsys.readouterr().out.count("episode ") == 2


def test_part_four_runs_with_its_own_library() -> None:
    cfg = ForcedData.Config(
        seq_len=48,
        pred_len=8,
        recent_steps=(12, 14),
        segment_steps=(8, 10),
    )
    cfg.observation = GlitchySensor.Config(noise=(0, 0), spike_rate=0)
    cfg.dynamics.substeps = 1
    rows = joint_calibration.calibrate(
        cfg.make(),
        episodes=2,
        restarts=1,
        iterations=0,
        seed=3,
        near_truth=True,
    )
    assert all(r.skill > 0.5 for r in rows)
