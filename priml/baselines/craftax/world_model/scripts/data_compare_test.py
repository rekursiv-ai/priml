"""Check that seeded data_eval reports compare against a control with pooled noise."""

from pathlib import Path

import json
import sys

import pytest

from priml.baselines.craftax.world_model.scripts import data_compare
from priml.lib.codec import PlainTree, from_plain, loads


def test_headline_reads_totals_arms_modalities_and_the_reweighted_mix() -> None:
    report = _report(arm_nats={0: 2.0, 1: 8.0}, bpb=0.01)
    values = data_compare.headline(report, mix=(3, 1, 0, 0))
    assert values["nats"] == pytest.approx(5.0)
    assert values["bpb"] == pytest.approx(0.01)
    assert values["bpb/board"] == pytest.approx(0.02)
    assert values["arm0"] == pytest.approx(2.0)
    assert values["arm1"] == pytest.approx(8.0)
    # Arms 0 and 1 weigh 3:1 whatever their share of the scored decisions.
    assert values["mix"] == pytest.approx((3 * 2.0 + 8.0) / 4)
    # Board holds half of each class's nats, action the other half.
    assert values["nats/board"] == pytest.approx(2.5)
    assert values["nats/action"] == pytest.approx(2.5)
    assert values["repeated"] == pytest.approx(2.0)
    assert values["fresh"] == pytest.approx(8.0)
    assert values["timeout"] == pytest.approx(2.0)
    assert values["ended"] == pytest.approx(8.0)


def test_compare_pools_seed_noise_and_scores_each_difference() -> None:
    groups = {
        "ctrl": [{"nats": 10.0}, {"nats": 12.0}],
        "arm": [{"nats": 8.0}, {"nats": 8.4}],
        "once": [{"nats": 11.0}],
    }
    rows = data_compare.compare(groups)["nats"]
    noise = ((2.0 + 0.08) / 2) ** 0.5
    assert rows["ctrl"]["mean"] == pytest.approx(11.0)
    assert rows["ctrl"]["sd"] == pytest.approx(2**0.5)
    assert rows["arm"]["delta"] == pytest.approx(-2.8)
    assert rows["arm"]["z"] == pytest.approx(-2.8 / noise)
    assert rows["once"]["z"] == pytest.approx(0.0)
    assert "sd" not in rows["once"]
    assert "delta" not in rows["ctrl"]


def test_compare_without_repeated_seeds_omits_every_z() -> None:
    rows = data_compare.compare({"ctrl": [{"nats": 10.0}], "arm": [{"nats": 8.0}]})
    assert rows["nats"]["arm"]["delta"] == pytest.approx(-2.0)
    assert "z" not in rows["nats"]["arm"]


def test_main_prints_a_flagged_table_and_writes_the_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths: dict[str, list[str]] = {}
    for name, seeds in {"ctrl": (10.0, 12.0), "arm": (8.0, 8.4)}.items():
        for seed, nats in enumerate(seeds):
            path = tmp_path / f"{name}-s{seed}.json"
            path.write_text(json.dumps(_report(arm_nats={0: nats}, bpb=nats / 100)))
            paths.setdefault(name, []).append(str(path))
    output = tmp_path / "compare.json"
    groups = [f"{name}={','.join(p)}" for name, p in paths.items()]
    monkeypatch.setattr(
        sys,
        "argv",
        ["data_compare.py", *groups, "--output", str(output)],
    )
    assert data_compare.main() == 0
    table = capsys.readouterr().out
    (line,) = [row for row in table.splitlines() if row.startswith("nats ")]
    assert "*" in line
    written = from_plain(loads(output.read_text()), dict[str, object])
    arm = from_plain(
        from_plain(written["nats"], dict[str, object])["arm"],
        dict[str, object],
    )
    assert from_plain(arm["delta"], float) == pytest.approx(-2.8)


def _report(*, arm_nats: dict[int, float], bpb: float) -> dict[str, PlainTree]:
    """Return a data_eval report: arm 0 timed-out repeats, other arms fresh endings."""
    classes: dict[str, PlainTree] = {}
    for arm, nats in arm_nats.items():
        kind = "timeout/repeated" if arm == 0 else "ended/fresh"
        classes[f"arm{arm}/{kind}"] = {
            "decisions": 100,
            "nats_per_decision": nats,
            "action": nats / 2,
            "reward": 0.0,
            "done": 0.0,
            "board": nats / 2,
            "hud": 0.0,
        }
    total = sum(arm_nats.values()) / len(arm_nats)
    metric: dict[str, PlainTree] = {
        "bpb": bpb,
        "bpb/board": 2 * bpb,
        "nats_per_decision": total,
    }
    return {"metric": metric, "classes": classes}


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
