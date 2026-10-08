"""Check fidelity report comparisons: paired spans and windows, dream rates, seeds."""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import functools
import json
import sys

import pytest

from priml.baselines.craftax.world_model.metric import MODALITIES
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.scripts import (
    fidelity,
    fidelity_compare,
)
from priml.baselines.craftax.world_model.scripts.fidelity_test import (
    validation_corpus,
)
from priml.baselines.craftax.world_model.testing import tiny_model
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from priml.lib.codec import PlainTree


TARGETS = (2, 4)
"""Target decisions of the two spans of every report."""


def test_compare_pairs_spans_and_pools_their_targets() -> None:
    base = _report("base")
    # Per span, action scores 1 vs 0.5 nats over 2 decisions and 1 vs 1.5 over
    # 4; every other modality 1 vs 1 over 2 and 1 vs 1.25 over 4.
    other = _report("other", action=(0.5, 1.5), rest=(1.0, 1.25))
    result = _compare(base, [other])
    action = _entry(result, "teacher_forced", "action", "other")
    assert action["value"] == pytest.approx((-1.0 + 2.0) / 6)
    assert action["n"] == 2
    assert action["lower"] == 1
    board = _entry(result, "teacher_forced", "board", "other")
    assert board["value"] == pytest.approx(1.0 / 6)
    assert board["lower"] == 0
    total = _entry(result, "teacher_forced", "total", "other")
    assert total["value"] == pytest.approx((1.0 + 1.0 * (len(MODALITIES) - 1)) / 6)


def test_compare_pairs_windows_the_reference_holds() -> None:
    base = _report("base")
    other = _report("other", board={"1": [0.1, 0.4], "8": [0.3, None]})
    result = _compare(base, [other])
    one = _entry(result, "continuations", "board_mismatch", "1", "other")
    assert one["value"] == pytest.approx(-0.05)
    assert one["n"] == 2
    assert one["lower"] == 1
    eight = _entry(result, "continuations", "board_mismatch", "8", "other")
    assert eight["value"] == pytest.approx(-0.2)
    assert eight["n"] == 1
    # Each report's model against its own frozen frame, on the same windows.
    still = _entry(result, "model_minus_frozen", "other", "board_mismatch", "1")
    assert still["value"] == pytest.approx((-0.2 + 0.1) / 2)
    assert still["n"] == 2


def test_compare_differences_dream_rates_per_thousand_decisions() -> None:
    base = _report("base")
    other = _report("other", over_maximum=(5.0, 7.0))
    result = _compare(base, [other])
    over = _entry(result, "dreams", "over_maximum", "other")
    assert over["base"] == pytest.approx(2.0)
    assert over["other"] == pytest.approx(6.0)
    assert over["value"] == pytest.approx(4.0)
    assert from_plain(over["low"], float) <= 4.0 <= from_plain(over["high"], float)
    deaths = _entry(result, "dreams", "deaths", "other")
    assert deaths["value"] == 0.0


def test_compare_averages_seeds_and_scores_them_against_seed_noise() -> None:
    base = _report("base", nll=8.0)
    seeds = [
        _report("s0", action=(0.5, 1.0), over_maximum=(5.0, 7.0), nll=7.0),
        _report("s1", action=(1.0, 0.5), nll=7.2),
    ]
    result = _compare(base, seeds)
    mean = _entry(result, "teacher_forced", "action", "mean")
    # Per span, the seeds' mean differs by -0.25 over 2 decisions and -0.25 over 4.
    assert mean["value"] == pytest.approx((-0.5 - 1.0) / 6)
    assert mean["n"] == 2
    # The seeds' four dream rows pooled: 4 per 1,000 decisions against 2.
    pooled = _entry(result, "dreams", "over_maximum", "mean")
    assert pooled["value"] == pytest.approx(2.0)
    assert pooled["other"] == pytest.approx(4.0)
    nll = _entry(result, "seeds", "nll/nats_per_decision")
    assert nll["delta"] == pytest.approx(-0.9)
    # One degree of freedom: sd 0.1414 over sqrt(1/2 + 1/1).
    assert nll["z"] == pytest.approx(-0.9 / (0.02 * 1.5) ** 0.5)
    assert "continuation/ended/1" not in from_plain(result["seeds"], dict[str, object])


def test_compare_rejects_reports_scored_on_different_data() -> None:
    base = _report("base")
    with pytest.raises(ValueError, match="spans"):
        _compare(base, [_report("other", first=1)])
    with pytest.raises(ValueError, match="settings"):
        _compare(base, [_report("other", seed=1)])
    ended = _report("other", board={"1": [0.1, None], "8": [0.3, None]})
    with pytest.raises(ValueError, match="windows"):
        _compare(base, [ended])


@pytest.mark.compute_large_fixture
def test_compare_finds_no_difference_between_copies_of_a_measured_report(
    tmp_path: Path,
) -> None:
    corpus = validation_corpus(tmp_path, lengths=(9, 12, 10, 14), deaths=True)
    measured = fidelity.measure(
        tiny_model(craftax_schema()),
        sources=fidelity.validation_sources(corpus),
        t_g=16,
        s_max=4,
        settings=fidelity.Settings(
            spans=3,
            targets=3,
            rows=4,
            decisions=4,
            real_episodes=4,
            prefix=2,
            continuation=3,
            resamples=16,
        ),
        precision=nullcontext(),
    )
    base: dict[str, PlainTree] = {**measured, "provenance": {"tag": "a"}}
    copy: dict[str, PlainTree] = {**measured, "provenance": {"tag": "b"}}
    result = _compare(base, [copy])
    for name in (*MODALITIES, "total"):
        assert _entry(result, "teacher_forced", name, "b")["value"] == 0.0, name
    board = _entry(result, "continuations", "board_mismatch", "1", "b")
    assert board["value"] == 0.0
    assert _entry(result, "dreams", "invalid_frame", "b")["value"] == 0.0
    json.dumps(result)


def test_main_prints_and_writes_the_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths: list[str] = []
    for tag, action in (("base", (1.0, 1.0)), ("other", (0.5, 1.5))):
        path = tmp_path / f"{tag}.json"
        path.write_text(json.dumps(_report(tag, action=action)))
        paths.append(str(path))
    output = tmp_path / "compare.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["fidelity_compare.py", *paths, "--output", str(output), "--resamples", "8"],
    )
    assert fidelity_compare.main() == 0
    written = from_plain(loads(output.read_text()), dict[str, object])
    action = _entry(written, "teacher_forced", "action", "other")
    assert action["value"] == pytest.approx(1 / 6)
    printed = capsys.readouterr().out
    assert "teacher_forced/action" in printed
    assert "other" in printed
    with pytest.raises(FileExistsError):
        fidelity_compare.main()


_compare = functools.partial(fidelity_compare.compare, resamples=64, seed=0)


# Per span, ``action`` scores its nats per decision and every other modality ``rest``'s;
# the spans start at decisions ``first`` and ``first + 10``.
def _report(
    tag: str,
    *,
    action: Sequence[float] = (1.0, 1.0),
    rest: Sequence[float] = (1.0, 1.0),
    board: Mapping[str, Sequence[float | None]] | None = None,
    frozen: Mapping[str, Sequence[float | None]] | None = None,
    over_maximum: Sequence[float] = (1.0, 3.0),
    nll: float = 8.0,
    seed: int = 0,
    first: int = 0,
) -> dict[str, object]:
    """Return a fidelity report of two spans, two windows, and two dream rows."""
    board = board or {"1": [0.2, 0.4], "8": [0.5, None]}
    frozen = frozen or {"1": [0.3, 0.3], "8": [0.6, None]}
    spans: list[PlainTree] = [
        {
            "episode": f"val/arm3/w0/shard-000000#{i}",
            "start": first + 10 * i,
            "targets": targets,
            "nats_per_decision": a + r * (len(MODALITIES) - 1),
            "modalities": {m: a if m == "action" else r for m in MODALITIES},
        }
        for i, (targets, a, r) in enumerate(zip(TARGETS, action, rest, strict=True))
    ]
    return {
        "provenance": {"tag": tag},
        "settings": {"seed": seed, "rows": 2, "decisions": 1_000},
        "summary": {"nll/nats_per_decision": nll, "continuation/ended/1": None},
        "teacher_forced": {"per_span": spans},
        "dreams": {
            "decisions": 1_000,
            "per_row": {"over_maximum": list(over_maximum), "deaths": [0.0, 1.0]},
        },
        "continuations": {
            "windows": [{"name": "a@0"}, {"name": "b@7"}],
            "model": {"per_pair": {"board_mismatch": {**board}}},
            "frozen": {"per_pair": {"board_mismatch": {**frozen}}},
        },
    }


def _entry(result: object, *path: str) -> dict[str, object]:
    """Return the comparison entry at a path of nested objects."""
    node = result
    for key in path:
        node = from_plain(node, dict[str, object])[key]
    return from_plain(node, dict[str, object])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
