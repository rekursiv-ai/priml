"""Check that logged EMA curves turn back into per-update objectives and compare."""

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import math
import sys

import pytest

from priml.baselines.craftax.world_model.scripts import curves
from priml.lib.codec import PlainTree, from_plain, loads


def test_update_objectives_invert_the_debiased_ema() -> None:
    objectives = [27.2, 26.1, 30.0, 12.5, 3.25, 3.0]
    recovered = curves.update_objectives(_logged(objectives, beta=0.9), beta=0.9)
    assert list(recovered) == [1, 2, 3, 4, 5, 6]
    assert list(recovered.values()) == pytest.approx(objectives)


def test_update_objectives_reject_a_gap_or_a_late_start() -> None:
    logged = _logged([1.0, 2.0, 3.0], beta=0.9)
    with pytest.raises(ValueError, match="without a gap"):
        curves.update_objectives({1: logged[1], 3: logged[3]}, beta=0.9)
    with pytest.raises(ValueError, match="without a gap"):
        curves.update_objectives({2: logged[2], 3: logged[3]}, beta=0.9)


def test_compare_reports_signed_absolute_windowed_and_tail_differences() -> None:
    reference = dict.fromkeys(range(1, 81), 1.0)
    candidate = {step: 1.0 + (0.5 if step <= 40 else -0.25) for step in range(1, 81)}
    difference = curves.compare(reference, candidate)
    assert difference.updates == 80
    assert difference.mean_abs == pytest.approx(0.375)
    assert difference.last100_mean_abs == pytest.approx(0.375)
    assert difference.mean_signed == pytest.approx(0.125)
    assert difference.window40_signed == pytest.approx([0.5, -0.25])
    assert difference.tail50_reference == pytest.approx(1.0)
    # Updates 31-80: ten at 1.5, forty at 0.75.
    assert difference.tail50_candidate == pytest.approx((10 * 1.5 + 40 * 0.75) / 50)


def test_main_writes_the_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs = {
        "ref": _logged([5.0, 4.0, 3.0], beta=0.9),
        "cand": _logged([5.0, 4.5, 2.0], beta=0.9),
    }
    monkeypatch.setattr(curves, "_logged_loss", partial(_fake_loss, runs=runs))
    output = tmp_path / "curves.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["curves.py", "ref", "cand", "--output", str(output)],
    )
    assert curves.main() == 0
    report = from_plain(loads(output.read_text()), dict[str, object])
    candidate = from_plain(
        from_plain(report["candidates"], dict[str, object])["cand"],
        dict[str, object],
    )
    assert from_plain(candidate["updates"], int) == 3
    assert from_plain(candidate["mean_signed"], float) == pytest.approx(-1 / 6)
    assert from_plain(candidate["window40_signed"], list[float]) == pytest.approx(
        [-1 / 6],
    )


def test_a_spike_recovers_when_a_later_loss_returns_near_its_baseline() -> None:
    rule = curves.SpikeRule(history=3, ratio=1.3, horizon=2, recovery=1.05)
    losses = [1.0, 1.0, 1.0, 2.0, 1.04, 1.0, 1.0, 3.0, 2.9, 2.8, 1.0]
    found = curves.spikes(range(1, 12), losses, rule=rule)
    # Update 8's spike never returns near 1.0 within 2 updates; update 9's,
    # also over a baseline of 1.0, does at update 11.
    assert [(s.step, s.recovered) for s in found] == [(4, True), (8, False), (9, True)]
    assert found[0].baseline == found[2].baseline == 1.0
    # A spike on the last logged loss has nothing after it to fail.
    assert curves.spikes([1, 2, 3, 4], [1.0, 1.0, 1.0, 2.0], rule=rule)[0].recovered


def test_training_report_reads_stability_and_each_modalitys_trend() -> None:
    rows: list[dict[str, PlainTree]] = [
        {"_step": step, "train/loss": 10.0 / step, "train/grad_norm": 1.0 / step}
        for step in range(1, 31)
    ]
    rows[5]["train/max_attention_logit"] = 12.5
    rows[6]["train/clip_fraction"] = 0.5
    rows += [
        {"_step": 10, **_val(board=0.3, hud=0.2)},
        {"_step": 20, **_val(board=0.1, hud=0.25)},
        {"_step": 30, **_val(board=0.05, hud=0.15)},
    ]
    report = curves.training_report(rows, rule=curves.SpikeRule())
    assert report["loss_finite"] is True
    assert report["grad_norm_finite"] is True
    assert report["unrecovered_spikes"] == 0
    assert report["last_step"] == 30
    assert report["eval_steps"] == [10, 20, 30]
    assert report["max_attention_logit_max"] == 12.5
    assert report["clip_fraction_last"] == 0.5
    modalities = from_plain(report["modalities"], dict[str, object])
    hud = from_plain(modalities["hud"], dict[str, object])
    assert hud["decreased"] is True
    assert hud["monotone"] is False
    assert from_plain(modalities["board"], dict[str, object])["monotone"] is True
    assert from_plain(report["final"], dict[str, object])["val/bpb/board"] == 0.05
    rows[3]["train/loss"] = math.nan
    assert curves.training_report(rows, rule=curves.SpikeRule())["loss_finite"] is False


def test_timing_reads_evaluations_update_times_and_throughput() -> None:
    updates: list[dict[str, object]] = [
        {
            "_step": step,
            "train/dt": float(step),
            "train/mfu": 0.25,
            "_runtime": 10 * step,
        }
        for step in range(1, 21)
    ]
    updates[4]["train/gpu_mem_allocated_gb"] = 90.0
    evaluations: list[dict[str, object]] = [
        {"_step": 10, "eval/time": 3.0},
        {"_step": 20, "eval/time": 5.0},
    ]
    rows = [*updates, *evaluations]
    timing = curves.timing(rows)
    assert timing["eval_seconds"] == [3.0, 5.0]
    assert timing["eval_total_seconds"] == 8.0
    assert timing["dt_median_seconds"] == 10.5
    # The 95th percentile of 20 sorted updates is the 20th, index int(19.0).
    assert timing["dt_p95_seconds"] == 20.0
    assert timing["mfu_mean"] == 0.25
    assert timing["tok_per_sec_mean"] is None
    assert timing["gpu_mem_max_gb"] == 90.0
    assert timing["runtime_seconds"] == 200.0


def _val(*, board: float, hud: float) -> dict[str, PlainTree]:
    """Return one evaluation's logged ``val/`` series."""
    return {
        "val/bpb": board + hud,
        "val/bpb/board": board,
        "val/bpb/hud": hud,
        "val/bpb/action": 1.0,
        "val/bpb/reward": 0.01,
        "val/bpb/done": 0.001,
    }


def _logged(objectives: list[float], *, beta: float) -> dict[int, float]:
    """Return what the train step logs for these per-update objectives."""
    smooth = 0.0
    logged: dict[int, float] = {}
    for update, value in enumerate(objectives, start=1):
        smooth = beta * smooth + (1 - beta) * value
        logged[update] = smooth / (1 - beta**update)
    return logged


def _fake_loss(
    project: str,
    run: str,
    *,
    runs: Mapping[str, dict[int, float]],
) -> dict[int, float]:
    """Stand in for the W&B fetch with fixed logged curves."""
    del project
    return runs[run]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
