#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Index a frozen corpus, count its coverage, and write the coverage report.

A corpus is a frozen list of published shards, ROOT/corpora/NAME.json. With
--freeze it is first written from every manifest line published under ROOT;
refreezing an existing corpus, or freezing a ROOT holding HALT.json, is an
error. Without --freeze the named corpus is read. freeze_corpus.py freezes a
corpus in the behaviour-mixture shares instead. Every shard's stratum index
(index.py) and coverage counters (coverage.py) are built, or read from the
cache, and timed. The report is coverage.report plus the corpus's shards,
episodes, and decisions, the stored bytes per decision of each kind of shard
file, and those timings. A replay shard's frames are replayed.

Examples:
  priml/baselines/craftax/world_model/scripts/coverage_report.py /opt/scratch/datasets/craftax/world-model/archive-v1 --corpus small-10m --freeze --output /opt/scratch/artifacts/craftax/world-model/coverage/small-10m.json

'''
# fmt: on

from pathlib import Path
from typing import Protocol, cast

import argparse
import collections
import json
import time

from priml.baselines.craftax.world_model.archive import (
    ManifestLine,
    read_corpus,
    read_manifest,
    write_corpus,
)
from priml.baselines.craftax.world_model.capture.control import (
    check_halt,
)
from priml.baselines.craftax.world_model.coverage import (
    Coverage,
    load_coverage,
    report,
)
from priml.baselines.craftax.world_model.index import load_index
from priml.lib.codec import PlainTree, from_plain
from priml.paths import validated_output_path


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the report is written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    output = validated_output_path(flags.output)
    result = build_report(
        flags.root,
        corpus=flags.corpus,
        freeze=flags.freeze,
        cache_dir=flags.root / flags.cache_dir,
        workers=flags.workers,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n")
    seconds = from_plain(result["seconds"], dict[str, object])
    print(
        f"{result['shards']} shards, {result['episodes']} episodes, "
        f"{result['decisions']} decisions; bytes/decision "
        f"{result['bytes_per_decision']}; reach {result['reach']}; "
        f"sufficient {result['sufficient']}.",
    )
    print(f"Seconds: {seconds}. Report: {output}.")
    return 0


def build_report(
    root: Path,
    *,
    corpus: str,
    freeze: bool,
    cache_dir: Path,
    workers: int,
) -> dict[str, PlainTree]:
    """Build the coverage report of one corpus of an archive, with its costs.

    Args:
      root: Archive root holding ``{train,val}/arm{arm}/w{worker}/``.
      corpus: Corpus name; the file is ``root/corpora/{corpus}.json``.
      freeze: Write the corpus from every published shard first.
      cache_dir: Directory of cached indexes and coverage counters.
      workers: Processes per shard; see ``index.map_chunks``.

    Returns:
      result: ``coverage.report`` plus ``corpus``, ``shards``, ``episodes``,
        ``decisions``, ``bytes_per_decision`` of each kind of shard file the
        corpus holds (``bin``, ``meta``, and ``frames`` or ``snap`` or both),
        and ``seconds`` (per-shard ``index`` and ``coverage`` builds, and the
        ``report``).

    Raises:
      FileExistsError: ``freeze`` names a corpus that is already frozen.
      CaptureHaltedError: ``freeze`` names a root that holds ``HALT.json``.
      ValueError: The corpus names no shard.

    """
    path = root / "corpora" / f"{corpus}.json"
    if freeze:
        if path.exists():
            raise FileExistsError(f"Corpus {path} is already frozen.")
        check_halt(root)
        # Absolute, so the corpus reads from any working directory.
        manifests = sorted(root.absolute().glob("*/*/*/MANIFEST.jsonl"))
        entries = [
            (m.parent, line) for m in manifests for line in read_manifest(m.parent)
        ]
        write_corpus(path, entries=entries)
    shards = read_corpus(path)
    if not shards:
        raise ValueError(f"Corpus {path} names no shard.")
    index_seconds: list[PlainTree] = []
    coverage_seconds: list[PlainTree] = []
    parts: list[Coverage] = []
    for directory, line in shards:
        clock = time.monotonic()
        load_index(directory, line, index_dir=cache_dir, workers=workers)
        index_seconds.append(time.monotonic() - clock)
        clock = time.monotonic()
        parts.append(
            load_coverage(
                directory,
                line,
                coverage_dir=cache_dir,
                workers=workers,
            ),
        )
        coverage_seconds.append(time.monotonic() - clock)
    clock = time.monotonic()
    result = report(parts)
    report_seconds = time.monotonic() - clock
    decisions = sum(line.decisions for _, line in shards)
    return {
        **result,
        "corpus": str(path),
        "shards": len(shards),
        "episodes": sum(line.episodes for _, line in shards),
        "decisions": decisions,
        "bytes_per_decision": {
            suffix: size / decisions for suffix, size in _stored_bytes(shards).items()
        },
        "seconds": {
            "index": index_seconds,
            "coverage": coverage_seconds,
            "report": report_seconds,
        },
    }


class Flags(Protocol):
    """Parsed command-line flags."""

    root: Path
    corpus: str
    freeze: bool
    output: Path
    cache_dir: Path
    workers: int


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("root", type=Path, help="Archive root.")
    parser.add_argument("--corpus", required=True, help="Corpus name, e.g. small-10m.")
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze the corpus from every published shard first.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Report JSON to write.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("index"),
        help="Index and coverage cache; relative paths resolve under ROOT.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Processes per shard build.",
    )


def _stored_bytes(shards: list[tuple[Path, ManifestLine]]) -> dict[str, int]:
    """Return the bytes of every kind of shard file the shards hold, by suffix."""
    names = {
        "bin": "bin.zst",
        "frames": "frames.zst",
        "snap": "snap.zst",
        "meta": "meta.jsonl",
    }
    stored: collections.Counter[str] = collections.Counter()
    for directory, line in shards:
        for suffix in line.sha256:
            path = directory / f"{line.shard}.{names[suffix]}"
            stored[suffix] += path.stat().st_size
    return dict(sorted(stored.items()))


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
