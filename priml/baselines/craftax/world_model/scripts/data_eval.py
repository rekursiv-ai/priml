#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Score a world-model checkpoint on validation decisions in their natural mix.

The validation shards of --corpus (by default the experiment's own) are cut
into consecutive windows that tile each shard's episode stream, each cut as
training cuts one (data.window), so every validation decision's action,
reward, done, and next frame, and every episode's first frame, lie in exactly
one tile; a replay shard's episodes are replayed. --tiles tiles drawn at
random with --seed are scored, with the kernels and autocast the experiment
trained under on CUDA (scoring.py). Every decision counts once, so the plain
ratios describe the natural distribution; the training loop's validation
windows are drawn by stratum and episode instead. The same corpus, --t-g,
--tiles, and --seed give the same decisions for every checkpoint, whatever
each was trained on.

The report holds the metric's bits per byte (overall, per modality, floor, and
event) and two breakdowns of the NLL, each with its share of decisions and of
nats and its nats per decision per modality: by stratum (floor x event), and
by class -- the arm, whether the episode timed out, and whether the frame is
"repeated", equal, light aside, to one of the 64 before it.

Examples:
  priml/baselines/craftax/world_model/scripts/data_eval.py /opt/scratch/runs/craftax-world-model/exp001/checkpoints/step_00001525.pt --tiles 512 --output /opt/scratch/artifacts/craftax/world-model/data-eval-s0.json

'''
# fmt: on

from collections.abc import Sequence
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import json
import time

from torch import Tensor

import torch

from priml.baselines.craftax.world_model.archive import (
    EpisodeSummary,
    ManifestLine,
    read_corpus,
    read_summaries,
)
from priml.baselines.craftax.world_model.batch import Segment, pack
from priml.baselines.craftax.world_model.capture.seeds import VALIDATION
from priml.baselines.craftax.world_model.data import EpisodeCache, window
from priml.baselines.craftax.world_model.index import STRATA, Event
from priml.baselines.craftax.world_model.metric import (
    MODALITIES,
    CraftaxBitsPerByte,
    craftax_target_nll,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.scoring import (
    autocast,
    load_trained,
)
from priml.baselines.craftax.world_model.scripts.data_stats import (
    frame_ids,
    repeats,
)
from priml.lib.codec import PlainTree, from_plain
from priml.paths import validated_output_path


CLASSES_PER_ARM: Final = 4
"""Class labels per arm: ended or timed out, times fresh or repeated."""


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
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    device = torch.device(flags.device)
    model, config = load_trained(
        flags.experiment,
        flags.checkpoint,
        overrides=flags.override,
        device=device,
    )
    corpus = flags.corpus or Path(config.dataset.corpus)
    t_g = flags.t_g or config.dataset.t_g
    clock = time.monotonic()
    with autocast(config, device):
        result, available = evaluate(
            model,
            corpus,
            t_g=t_g,
            s_max=config.dataset.s_max,
            count=flags.tiles,
            seed=flags.seed,
            cached_decisions=config.dataset.cached_decisions,
        )
    result["run"] = {
        "checkpoint": str(flags.checkpoint),
        "experiment": flags.experiment,
        "overrides": list(flags.override),
        "corpus": str(corpus),
        "t_g": t_g,
        "seed": flags.seed,
        "tiles_available": available,
        "seconds": time.monotonic() - clock,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Wrote {output}.")
    return 0


def evaluate(
    model: WorldModel,
    corpus: Path,
    *,
    t_g: int,
    s_max: int,
    count: int,
    seed: int,
    cached_decisions: int,
) -> tuple[dict[str, PlainTree], int]:
    """Score ``count`` random tiles of a corpus's validation shards.

    Args:
      model: The world model, on its device, under the caller's autocast.
      corpus: Corpus file whose validation shards are tiled.
      t_g: Global positions per window.
      s_max: Segments allowed per window.
      count: Tiles to score, drawn without replacement.
      seed: Seed of the draw.
      cached_decisions: Decisions of decoded episodes to keep.

    Returns:
      report: ``score``'s report of the drawn tiles.
      available: Tiles the validation shards hold.

    """
    shards = [
        (entry, summaries)
        for entry in read_corpus(corpus)
        if (summaries := read_summaries(*entry))[0].receipt.split == VALIDATION
    ]
    cuts = [
        (shard, episode, start)
        for shard, (_, summaries) in enumerate(shards)
        for episode, start in tiles(
            [s.decisions for s in summaries],
            t_g=t_g,
            s_max=s_max,
        )
    ]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(cuts), generator=generator)[:count]
    chosen = [cuts[i] for i in sorted(int(i) for i in order)]
    report = score(
        model,
        shards,
        chosen,
        t_g=t_g,
        s_max=s_max,
        cached_decisions=cached_decisions,
    )
    return report, len(cuts)


def tiles(decisions: Sequence[int], *, t_g: int, s_max: int) -> list[tuple[int, int]]:
    """Return where each window starts when windows tile a shard's episode stream.

    A window laid out by ``data.window`` from ``(episode, start)`` scores the
    action of every decision whose ``act`` position fits; a decision whose
    ``obs`` lands on the last position is scored by the next window, which
    starts at it. A window that ends on an episode's lone start position scores
    that episode's first frame, which the next window scores again: one frame
    in about ``t_g`` episode boundaries. Every episode is taken to start at its
    world's reset, with a ``start`` position, as validation episodes do;
    ``score`` checks it.

    Args:
      decisions: Decision count of each episode of the shard, in order.
      t_g: Global positions per window.
      s_max: Segments allowed per window.

    Returns:
      starts: Each window's first episode and first decision within it.

    """
    starts: list[tuple[int, int]] = []
    episode, start = 0, 0
    while episode < len(decisions):
        starts.append((episode, start))
        cursor, segments = 0, 0
        while cursor < t_g and segments < s_max and episode < len(decisions):
            head = int(start == 0)
            room = t_g - cursor
            steps = min(decisions[episode] - start, (room - head + 1) // 2)
            length = min(room, head + 2 * steps)
            acts = min(steps, (length - head) // 2)
            cursor, segments = cursor + length, segments + 1
            if start + acts < decisions[episode]:
                start += acts
                break
            episode, start = episode + 1, 0
    return starts


def score(
    model: torch.nn.Module,
    shards: Sequence[tuple[tuple[Path, ManifestLine], list[EpisodeSummary]]],
    chosen: Sequence[tuple[int, int, int]],
    *,
    t_g: int,
    s_max: int,
    cached_decisions: int,
) -> dict[str, PlainTree]:
    """Score the chosen tiles one window at a time.

    Args:
      model: The world model, on its device, under the caller's autocast.
      shards: Each validation shard's corpus entry and summaries.
      chosen: Each tile's shard, first episode, and first decision.
      t_g: Global positions per window.
      s_max: Segments allowed per window.
      cached_decisions: Decisions of decoded episodes to keep.

    Returns:
      report: ``metric`` (the metric's ``compute``), ``strata``, ``classes``,
        ``tiles``, and ``decisions``.

    Raises:
      ValueError: A tiled episode does not start at its world's reset, or the
        arms' class labels outnumber the metric's strata, which hold them.

    """
    arms = 1 + max(s.receipt.arm for _, summaries in shards for s in summaries)
    if arms * CLASSES_PER_ARM > STRATA:
        raise ValueError(f"{arms} arms have more class labels than {STRATA} strata.")
    device = next(model.parameters()).device
    cache = EpisodeCache([entry for entry, _ in shards], capacity=cached_decisions)
    by_stratum = CraftaxBitsPerByte(CraftaxBitsPerByte.Config())
    by_class = CraftaxBitsPerByte(CraftaxBitsPerByte.Config())
    for tally in (by_stratum, by_class):
        tally.measure_reference = False
    labels: dict[tuple[int, int], Tensor] = {}
    for shard, episode, start in chosen:
        parts = window(
            cache,
            shard=shard,
            episode=episode,
            start=start,
            t_g=t_g,
            s_max=s_max,
        )
        classes: list[tuple[Segment, Tensor]] = []
        for offset, (segment, strata) in enumerate(parts):
            key = (shard, episode + offset)
            if key not in labels:
                labels[key] = _classes(cache, shards[shard][1], key)
            first = start if offset == 0 else 0
            classes.append((segment, labels[key][first : first + len(strata)].long()))
        batch, stratum = pack([parts], t_g=t_g, s_max=s_max)
        _, label = pack([classes], t_g=t_g, s_max=s_max)
        with torch.no_grad():
            nll = craftax_target_nll(model, batch.to(device))
        by_stratum.update(nll, media=batch, stratum=stratum)
        by_class.update(nll, media=batch, stratum=label)
    names = [
        f"arm{arm}/{'timeout' if timeout else 'ended'}/"
        f"{'repeated' if repeated else 'fresh'}"
        for arm in range(arms)
        for timeout in (0, 1)
        for repeated in (0, 1)
    ]
    metric: dict[str, PlainTree] = {**by_stratum.compute()}
    return {
        "metric": metric,
        "strata": _breakdown(by_stratum, [_stratum_name(s) for s in range(STRATA)]),
        "classes": _breakdown(by_class, names),
        "tiles": len(chosen),
        "decisions": int(by_stratum.decisions.sum()),
    }


class Flags(Protocol):
    """Parsed command-line flags."""

    checkpoint: Path
    experiment: str
    override: list[str]
    corpus: Path | None
    t_g: int
    tiles: int
    seed: int
    device: str
    output: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("checkpoint", type=Path, help="TrainLoop checkpoint (.pt).")
    parser.add_argument(
        "--experiment",
        default="priml.baselines.craftax.world_model.experiments.exp001",
        help="Dotted path of the experiment factory that trained the checkpoint.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Config override the run was launched with; repeatable.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Corpus file whose validation shards are scored; default the run's.",
    )
    parser.add_argument(
        "--t-g",
        type=int,
        default=0,
        help="Positions per window; default the run's.",
    )
    parser.add_argument(
        "--tiles",
        type=int,
        default=512,
        help="Windows scored; default 512.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the tile draw; default 0.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--output", type=Path, required=True, help="Report JSON.")


def _classes(
    cache: EpisodeCache,
    summaries: Sequence[EpisodeSummary],
    key: tuple[int, int],
) -> Tensor:
    """Return each decision's class, ``arm * 4 + timeout * 2 + repeated``."""
    decoded = cache.get(*key)
    episode = decoded.episode
    if episode.origin:
        raise ValueError(
            f"Validation episode {key} starts from a branch; tiles assume a reset.",
        )
    timeout = from_plain(summaries[key[1]].summary.get("timeout"), int, default=0) > 0
    repeated = repeats(frame_ids(episode)).long()
    label = episode.receipt.arm * CLASSES_PER_ARM + int(timeout) * 2 + repeated
    return label.to(torch.uint8)


def _stratum_name(stratum: int) -> str:
    """Return a stratum's name, e.g. ``floor8/pre_death``."""
    floor, event = divmod(stratum, len(Event))
    return f"floor{floor}/{Event(event).name.lower()}"


def _breakdown(
    metric: CraftaxBitsPerByte,
    names: Sequence[str],
) -> dict[str, PlainTree]:
    """Return decisions, nats, and nats per decision per modality of each label."""
    nats, decisions = metric.nats, metric.decisions
    result: dict[str, PlainTree] = {}
    for index, name in enumerate(names):
        count = float(decisions[index])
        if count == 0:
            continue
        per_modality = {
            modality: float(nats[m, index]) / count
            for m, modality in enumerate(MODALITIES)
        }
        result[name] = {
            "decisions": int(count),
            "share_decisions": count / float(decisions.sum()),
            "share_nats": float(nats[:, index].sum() / nats.sum()),
            "nats_per_decision": float(nats[:, index].sum()) / count,
            **per_modality,
        }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
