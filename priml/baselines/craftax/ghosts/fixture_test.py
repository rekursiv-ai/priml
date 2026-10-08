"""Check the committed fixture: its files, its expected values, and how it was made."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import hashlib
import json

from priml.baselines.craftax.ghosts.fixture import (
    FIXTURE_UNBROKEN,
    expected,
    resolved,
    shared,
    write_fixture,
)
from priml.baselines.craftax.ghosts.layout import GroupFile, Manifest
from priml.lib.codec import from_plain, loads


_CWD: Final = Path(__file__).resolve().parent

_FIXTURE: Final = _CWD / "testdata" / "fixture"


def test_every_file_matches_its_manifest_digest() -> None:
    manifest = from_plain(
        loads((_FIXTURE / "site" / "manifest.json").read_text()),
        Manifest,
    )
    for name, digest in manifest.files.items():
        data = (_FIXTURE / "site" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        assert len(data) == manifest.sizes[name], name


def test_the_expected_values_are_what_the_site_decodes_to() -> None:
    decoded = loads(json.dumps(expected(_FIXTURE / "site")))
    assert decoded == loads((_FIXTURE / "expected.json").read_text())


def test_the_fixture_crosses_floors_both_ways() -> None:
    manifest = from_plain(
        loads((_FIXTURE / "site" / "manifest.json").read_text()),
        Manifest,
    )
    tier, _ = manifest.tiers
    every, short = tier.sets
    assert [group.count for group in every.groups] == [3]
    assert short.counts == (2,)
    assert short.composition.qualified == 2
    trip = _expected_episodes()[0]
    floors = [step[0] for step in from_plain(trip["path"], list[list[int]])]
    assert floors[0] == 0
    assert 1 in floors
    assert floors[floors.index(1) :].count(0) > 0


def test_the_fixture_plays_its_pinned_win_unbroken_and_compresses_the_other() -> None:
    manifest = from_plain(
        loads((_FIXTURE / "site" / "manifest.json").read_text()),
        Manifest,
    )
    _, tier = manifest.tiers
    every, _, wins = tier.sets
    assert (tier.name, wins.name, wins.counts) == ("fixture-wins", "wins", (2,))
    assert wins.time_map is not None
    assert wins.time_map.unbroken == 0
    (group,) = wins.groups
    entries = from_plain(
        loads((_FIXTURE / "site" / group.path / "episodes.json").read_text()),
        GroupFile,
    ).episodes
    assert [e.sampling_seed for e in entries] == [str(FIXTURE_UNBROKEN), "0"]
    shown = [e for e in _expected_episodes() if e["set"] == "wins"]
    lengths: list[int] = []
    for episode in shown:
        steps = from_plain(episode["displayed"], list[int])
        decisions = from_plain(episode["decisions"], int)
        assert steps[0] == 0
        assert steps[-1] == decisions - 1
        assert steps == sorted(set(steps))
        lengths.append(len(steps) - decisions)
    unbroken, compressed = lengths
    assert unbroken == 0
    assert compressed < 0
    assert wins.time_map.steps <= wins.time_map.rule.steps
    assert every.time_map is None


def test_the_fixture_rebuilds_byte_for_byte(tmp_path: Path) -> None:
    """On any host: macOS's libm generates world 1 as the minting glibc's did (measured)."""
    manifest = from_plain(
        loads((_FIXTURE / "site" / "manifest.json").read_text()),
        Manifest,
    )
    write_fixture(tmp_path, world_seed=manifest.world_seed)
    rebuilt = from_plain(
        loads((tmp_path / "site" / "manifest.json").read_text()),
        Manifest,
    )
    assert rebuilt.files == manifest.files
    assert (tmp_path / "expected.json").read_text() == (
        _FIXTURE / "expected.json"
    ).read_text()


def test_shared_entries_resolve_to_what_the_site_decodes_each_to() -> None:
    episodes = _expected_episodes()
    stored = from_plain(
        from_plain(
            loads((_FIXTURE / "expected.json").read_text()),
            dict[str, object],
        )["episodes"],
        list[dict[str, object]],
    )
    # The repeats are references, and each expands to a full entry.
    assert sum("same_as" in entry for entry in stored) >= 6
    assert all("path" in entry and "same_as" not in entry for entry in episodes)
    assert shared(episodes) == stored
    # A ghost in two sets decodes alike in both: the all and short sets of the
    # first tier share their dying episodes.
    every, short = (
        [e for e in episodes if (e["tier"], e["set"]) == ("fixture", name)]
        for name in ("all", "short")
    )
    assert short[0]["path"] == every[0]["path"]


def _expected_episodes() -> list[dict[str, object]]:
    data = from_plain(
        loads((_FIXTURE / "expected.json").read_text()),
        dict[str, object],
    )
    return resolved(from_plain(data["episodes"], list[dict[str, object]]))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
