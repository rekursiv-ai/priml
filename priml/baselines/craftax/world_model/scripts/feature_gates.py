#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Gate the frozen feature engine on a trained checkpoint against the training forward.

feature_test.py holds the engine to the training forward on tiny random models
in float32; this holds the production engine (bfloat16, the chosen cache
attention, one CUDA graph replayed per step) to it on a trained model and real
frames: the first --decisions decisions of the corpus's first --episodes
validation episodes longer than that, each replayed --replicate times, hooked
every --hook steps from an empty cache, so the rows re-prefill mid-run.

  G2a  every step's feature against the float32 training forward of exactly
       the history the engine holds: rel RMS <= 1.2%, min cosine >= 0.9998,
       action-head argmax agreement >= 99.5%, overall and before and after the
       first re-prefill.
  G2c  per layer, the step's attention against masked SDPA over the engine's
       own final cache, each against the float32 attention of the same inputs:
       passes when it is no further from float32 than bfloat16 SDPA.
  time the graphed step at --timing-rows rows with 3/4 of t_max filled, and
       one re-prefill of half of those rows.

OUTPUT is one JSON report with each gate's numbers and verdict; the process
exits 1 when a gate fails.

Examples:
  priml/baselines/craftax/world_model/scripts/feature_gates.py /opt/scratch/datasets/craftax/world-model-oracle/v1/checkpoints/exp001-s0/step_00001525.pt /opt/scratch/datasets/craftax/world-model-reference/archive-v1-replay/corpora/base.json /opt/scratch/artifacts/craftax/world-model/feature-gates-s0.json --attention fa4 --compile

'''
# fmt: on

from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import dataclasses
import json
import time

from configgle import PartialConfig
from torch import Tensor
from torch.nn import functional

import torch

from priml.baselines.craftax.world_model.archive import (
    read_corpus,
    read_summaries,
)
from priml.baselines.craftax.world_model.capture.seeds import VALIDATION
from priml.baselines.craftax.world_model.checkpoint import (
    load_world_model,
)
from priml.baselines.craftax.world_model.codec import decode
from priml.baselines.craftax.world_model.feature import (
    CacheAttention,
    FeatureEngine,
    Flash4CacheAttention,
    MaskedCacheAttention,
    Refill,
    TrainedWeights,
    WorldModelFeature,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.snapshots import replay_episodes
from priml.baselines.craftax.world_model.testing import (
    final_hidden,
    window_segment,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from priml.lib.codec import PlainTree


@dataclasses.dataclass(slots=True, kw_only=True)
class Span:
    """Consecutive decisions of one row that the engine holds from position 0.

    Attributes:
      row: The engine row.
      first: Env step of the span's first decision.
      starts_episode: Whether the span begins with the ``start`` position.
      steps: Env steps whose feature is the span's obs at ``t - first``.

    """

    row: int
    first: int
    starts_episode: bool
    steps: list[int]


class HeldHistory:
    """Predict, from the engine's rules alone, which decisions each row's cache holds."""

    def __init__(self, rows: int, *, t_max: int, keep: int) -> None:
        self.spans: list[Span] = []
        self._t_max, self._keep = t_max, keep
        self._length = [0] * rows
        self._current: list[Span | None] = [None] * rows

    def hook(self, t: int, *, steps: int) -> None:
        """Record ``ensure_room(steps)`` before env step ``t``.

        Args:
          t: The next env step.
          steps: The call's argument.

        """
        for row, span in enumerate(self._current):
            if span is not None and self._length[row] + 2 * steps > self._t_max:
                self._open(
                    Span(row=row, first=t - self._keep, starts_episode=False, steps=[]),
                )
                self._length[row] = 2 * self._keep - 1

    def step(
        self,
        t: int,
        *,
        resets: Collection[int],
        windows: Collection[int] = (),
    ) -> None:
        """Record one env step.

        Args:
          t: The env step.
          resets: Rows whose observation at ``t`` begins an episode.
          windows: Rows whose observation at ``t`` begins a mid-episode window;
            a reset takes precedence.

        """
        for row, current in enumerate(self._current):
            span = current
            if span is None or row in resets:
                span = self._open(Span(row=row, first=t, starts_episode=True, steps=[]))
                self._length[row] = 0
            elif row in windows:
                span = self._open(
                    Span(row=row, first=t, starts_episode=False, steps=[]),
                )
                # The window's obs alone fills position 0.
                self._length[row] = -1
            span.steps.append(t)
            self._length[row] += 2

    def resume(self, row: int, t: int, *, first: int, anchored: bool) -> None:
        """Record a restore before step ``t`` whose cache holds steps ``first..t-1``.

        Args:
          row: The restored row.
          t: The next env step.
          first: The restored context's first step.
          anchored: Whether a ``start`` leads it.

        """
        self._open(Span(row=row, first=first, starts_episode=anchored, steps=[]))
        self._length[row] = 2 * (t - first) - 1 + int(anchored)

    def _open(self, span: Span) -> Span:
        self._current[span.row] = span
        self.spans.append(span)
        return span


def reference_features(
    model: WorldModel,
    *,
    layers: int,
    cells: Tensor,
    aux: Tensor,
    actions: Tensor,
    spans: Sequence[Span],
) -> Tensor:
    """Return the training forward's final-normed obs hiddens at every held step.

    Args:
      model: The world model, in the precision and on the device to run.
      layers: Global blocks up to the tap; the final norm reads the last one's
        output, as the engine's does.
      cells: Cell values per row and env step ``[R, T, 99, 8]``.
      aux: Aux values ``[R, T, 51]``.
      actions: Executed actions ``[R, T]``.
      spans: The held histories, from ``HeldHistory``.

    Returns:
      hidden: Float32 ``[R, T, C]``; NaN at steps no span covers.

    """
    windows = [
        window_segment(
            cells[s.row],
            aux=aux[s.row],
            actions=actions[s.row],
            first=s.first,
            last=s.steps[-1],
            anchored=s.starts_episode,
        )
        for s in spans
    ]
    with torch.no_grad():
        hidden = final_hidden(model, layers=layers, windows=windows)
    device = model.start.device
    out = torch.full((*actions.shape, hidden.shape[-1]), torch.nan, device=device)
    for window, span in enumerate(spans):
        steps = torch.tensor(span.steps, device=device)
        at = int(span.starts_episode) + 2 * (steps - span.first)
        out[span.row, steps] = hidden[window, at].float()
    return out


class GraphedStep:
    """An engine step graphed per plan over static inputs, as a rollout graphs it."""

    def __init__(self, engine: FeatureEngine, *, rows: int) -> None:
        """Capture the engine's current plan.

        Args:
          engine: The engine; its state is put back after each capture's warmup.
          rows: Its rows.

        """
        device = engine.length.device
        self.engine = engine
        self.observation = torch.zeros(rows, 843, device=device)
        self.terminals = torch.zeros(rows, device=device)
        self.previous_action = torch.zeros(rows, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.feature = engine.feature
        self.graphs[engine.plan] = self._capture()

    def __call__(
        self,
        observation: Tensor,
        terminals: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        """Copy the inputs into the graph's and replay it; return its feature."""
        self.observation.copy_(observation)
        self.terminals.copy_(terminals)
        self.previous_action.copy_(previous_action)
        self.replay()
        return self.feature

    def replay(self) -> None:
        """Replay the engine's current plan's graph, capturing it first if new."""
        plan = self.engine.plan
        if plan not in self.graphs:
            self.graphs[plan] = self._capture()
        self.graphs[plan].replay()

    def _capture(self) -> torch.cuda.CUDAGraph:
        """Warm up on a side stream, capture, then restore the engine's state."""
        engine = self.engine
        saved = [(value, value.clone()) for value in engine.capture_state()]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            for _ in range(3):
                engine(self.observation, self.terminals, self.previous_action)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, stream=stream):
            engine(self.observation, self.terminals, self.previous_action)
        torch.cuda.current_stream().wait_stream(stream)
        for value, original in saved:
            value.copy_(original)
        return graph


def run_steps(
    step: Callable[[Tensor, Tensor, Tensor], Tensor],
    engine: FeatureEngine,
    *,
    obs: Tensor,
    actions: Tensor,
    resets: Mapping[int, Sequence[int]],
    hook: int,
    history: HeldHistory,
    windows: Mapping[int, Sequence[int]] | None = None,
) -> Tensor:
    """Drive ``step`` over env steps as the rollout does; return features ``[R, T, C]``.

    Args:
      step: The engine, or its ``GraphedStep``.
      engine: The engine ``step`` runs.
      obs: Float32 observations per row and step ``[R, T, 843]``.
      actions: Executed actions ``[R, T]``; step ``t`` reads ``t - 1``'s.
      resets: Per env step, the rows whose observation begins an episode.
      hook: Steps between ``ensure_room`` calls; 0 never calls it.
      history: Records what the engine should hold.
      windows: Per env step, the rows that begin a mid-episode window, marked
        after ``ensure_room``.

    Returns:
      features: The float32 feature of every row and step.

    """
    marked = dict[int, Sequence[int]]() if windows is None else windows
    rows = obs.shape[0]
    got: list[Tensor] = []
    for t in range(obs.shape[1]):
        if hook and t % hook == 0:
            engine.ensure_room(hook)
            history.hook(t, steps=hook)
        window = torch.zeros(rows, dtype=torch.bool)
        window[list(marked.get(t, []))] = True
        engine.begin_window(window)
        terminals = torch.zeros(rows, device=obs.device)
        terminals[list(resets.get(t, []))] = 1
        previous = actions[:, t - 1] if t else torch.zeros_like(actions[:, 0])
        feature = step(obs[:, t], terminals, previous.float())
        history.step(t, resets=resets.get(t, []), windows=marked.get(t, []))
        got.append(feature.float().clone())
    return torch.stack(got, dim=1)


def compare(got: Tensor, want: Tensor, head: Tensor) -> dict[str, float]:
    """Compare features row by row.

    Args:
      got: Features under test ``[N, C]``.
      want: Reference features ``[N, C]``.
      head: Action head ``[actions, C]``.

    Returns:
      stats: Rel RMS, max abs, min cosine, action-head argmax agreement, points.

    """
    got, want, head = got.float(), want.float(), head.float()
    agree = (got @ head.T).argmax(-1) == (want @ head.T).argmax(-1)
    return {
        "rel_rms": float(
            (got - want).square().mean().sqrt() / want.square().mean().sqrt(),
        ),
        "max_abs": float((got - want).abs().max()),
        "min_cos": float(functional.cosine_similarity(got, want, dim=-1).min()),
        "argmax_agree": float(agree.float().mean()),
        "points": float(got.shape[0]),
    }


def main() -> int:
    """Run the gates and write the report.

    Returns:
      status: 0 when every gate passes, else 1.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    device = torch.device("cuda")
    reference, _ = load_world_model(
        flags.experiment,
        flags.checkpoint,
        overrides=flags.override,
    )
    reference = reference.to(device)
    config = WorldModelFeature.Config()
    weights = config.weights = TrainedWeights.Config()
    weights.experiment = flags.experiment
    weights.checkpoint = flags.checkpoint
    weights.overrides = list(flags.override)
    refill = config.history = Refill.Config()
    refill.t_max = flags.t_max
    refill.keep = flags.t_max // 4
    config.layers = flags.layers
    config.hook_interval = flags.hook
    if flags.attention == "fa4":
        config.attention = Flash4CacheAttention.Config()
    if flags.compile:
        config.compile = PartialConfig(
            torch.compile,
            fullgraph=True,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
    source = config.make()
    cells, aux, actions = validation_prefixes(
        flags.corpus,
        rows=flags.episodes,
        decisions=flags.decisions,
    )
    obs = decode(cells, aux).to(device)
    rows = flags.episodes * flags.replicate
    engine = source.make_engine(rows=rows, device=device)
    history = HeldHistory(rows, t_max=engine.t_max, keep=engine.keep)
    got = run_steps(
        GraphedStep(engine, rows=rows),
        engine,
        obs=obs.repeat(flags.replicate, 1, 1),
        actions=actions.to(device).repeat(flags.replicate, 1),
        resets={},
        hook=flags.hook,
        history=history,
    )
    spans = [s for s in history.spans if s.row < flags.episodes]
    want = reference_features(
        reference,
        layers=flags.layers,
        cells=cells,
        aux=aux,
        actions=actions,
        spans=spans,
    ).repeat(flags.replicate, 1, 1)
    head = reference.action_head.weight
    split = min((s.steps[0] for s in spans if not s.starts_episode), default=0)
    g2a = {
        "all": compare(got.flatten(0, 1), want.flatten(0, 1), head),
        "before_reprefill": compare(
            got[:, :split].flatten(0, 1),
            want[:, :split].flatten(0, 1),
            head,
        ),
    }
    if split:
        g2a["after_reprefill"] = compare(
            got[:, split:].flatten(0, 1),
            want[:, split:].flatten(0, 1),
            head,
        )
    g2c = attention_gap(engine, attention=source.attention)
    engine.release()
    del engine
    timing = time_step(source, rows=flags.timing_rows, frame=obs[0, 0])
    summary = g2a["all"]
    gates = {
        "g2a": summary["rel_rms"] <= 0.012
        and summary["min_cos"] >= 0.9998
        and summary["argmax_agree"] >= 0.995,
        "g2c": g2c["kernel_vs_fp32"] <= g2c["sdpa_vs_fp32"],
    }
    report: dict[str, PlainTree] = {
        "checkpoint": str(flags.checkpoint),
        "corpus": str(flags.corpus),
        "rows": rows,
        "engine": config.pformat(),
        "g2a": cast("PlainTree", g2a),
        "reprefill_step": split,
        "g2c": cast("PlainTree", g2c),
        "timing": cast("PlainTree", timing),
        "gates": cast("PlainTree", gates),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if all(gates.values()) else 1


def validation_prefixes(
    corpus: Path,
    *,
    rows: int,
    decisions: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return the first ``decisions`` steps of ``rows`` validation episodes long enough.

    Args:
      corpus: Corpus file naming published shards; replay shards are replayed.
      rows: Episodes to return.
      decisions: Steps of each; an episode of no more decisions is skipped.

    Returns:
      cells: Cell values ``[rows, decisions, 99, 8]``.
      aux: Auxiliary token values ``[rows, decisions, 51]``.
      actions: Executed actions ``[rows, decisions]``.

    Raises:
      ValueError: The corpus holds fewer such episodes than ``rows``.

    """
    chosen: list[tuple[Tensor, Tensor, Tensor]] = []
    for directory, line in read_corpus(corpus):
        for summary in read_summaries(directory, line):
            if summary.receipt.split != VALIDATION or summary.decisions <= decisions:
                continue
            (episode,) = replay_episodes(directory, line, summaries=[summary])
            chosen.append(
                (
                    episode.cells[:decisions],
                    episode.aux[:decisions],
                    episode.actions[:decisions],
                ),
            )
            if len(chosen) == rows:
                cells, aux, actions = (
                    torch.stack(part) for part in zip(*chosen, strict=True)
                )
                return cells, aux, actions
    raise ValueError(f"{corpus} has fewer than {rows} episodes over {decisions}.")


# Both kernels run on the same bf16 queries, keys and values; the float32 masked
# attention of those same inputs is the reference each is measured against.
def attention_gap(
    engine: FeatureEngine,
    *,
    attention: CacheAttention,
) -> dict[str, float]:
    """Compare a cache attention with masked SDPA per layer over the engine's own cache.

    Args:
      engine: An engine whose cache a run has filled.
      attention: The kernel under test.

    Returns:
      gap: Per-layer maxima: ``kernel_vs_sdpa``, ``max_out``, and each kernel's
        max-abs to the float32 attention, ``kernel_vs_fp32`` and ``sdpa_vs_fp32``.

    """
    masked = MaskedCacheAttention.Config().make()
    generator = torch.Generator(device=engine.keys.device).manual_seed(0)
    rows, dim = engine.keys.shape[1], engine.keys.shape[-1]
    # Two queries end at each row's length, as the step's do.
    seqused = engine.length.clamp(min=2).to(torch.int32)
    shape = (rows, 2, engine.heads, dim)
    stats = dict.fromkeys(
        ["kernel_vs_sdpa", "max_out", "kernel_vs_fp32", "sdpa_vs_fp32"],
        0.0,
    )
    for keys, values in zip(engine.keys, engine.values, strict=True):
        q = torch.randn(
            shape,
            generator=generator,
            device=keys.device,
            dtype=keys.dtype,
        )
        kernel = attention(q, keys, values, seqused).float()
        sdpa = masked(q, keys, values, seqused).float()
        exact = masked(q.float(), keys.float(), values.float(), seqused)
        layer = {
            "kernel_vs_sdpa": float((kernel - sdpa).abs().max()),
            "max_out": float(sdpa.abs().max()),
            "kernel_vs_fp32": float((kernel - exact).abs().max()),
            "sdpa_vs_fp32": float((sdpa - exact).abs().max()),
        }
        for name, value in layer.items():
            stats[name] = max(stats[name], value)
    return stats


def time_step(
    source: WorldModelFeature,
    *,
    rows: int,
    frame: Tensor,
) -> dict[str, float]:
    """Time the graphed step at 3/4 fill and one re-prefill of half the rows.

    Args:
      source: The feature source.
      rows: Rows of the timed engine.
      frame: One float32 observation every row reads.

    Returns:
      timing: ``step_ms``, ``reprefill_ms`` and the re-prefill's share of a
        ``hook_interval`` block.

    """
    engine = source.make_engine(rows=rows, device=frame.device)
    fill = 3 * engine.t_max // 4
    step = GraphedStep(engine, rows=rows)
    observation = frame.expand(rows, -1)
    zeros = torch.zeros(rows, device=frame.device)
    engine.needs_start.fill_(value=False)
    engine.count.fill_(fill // 2)
    engine.length.fill_(fill)
    step(observation, zeros, zeros)
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    replays = 32
    engine.length.fill_(fill)
    begin.record()
    for _ in range(replays):
        step.replay()
    end.record()
    torch.cuda.synchronize()
    step_ms = begin.elapsed_time(end) / replays
    engine.length.fill_(fill)
    engine.length[: rows // 2] = engine.t_max
    engine.overflow.zero_()
    engine.ensure_room(1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    engine.length[: rows // 2] = engine.t_max
    engine.ensure_room(1)
    torch.cuda.synchronize()
    reprefill_ms = 1e3 * (time.perf_counter() - started)
    block = source.hook_interval * step_ms
    return {
        "rows": float(rows),
        "fill_before": float(fill),
        "step_ms": step_ms,
        "reprefill_rows": float(rows // 2),
        "reprefill_ms": reprefill_ms,
        "reprefill_share_of_a_block": reprefill_ms / (reprefill_ms + block),
    }


class Flags(Protocol):
    """Parsed command-line flags."""

    checkpoint: Path
    corpus: Path
    output: Path
    experiment: str
    override: list[str]
    episodes: int
    decisions: int
    replicate: int
    hook: int
    t_max: int
    layers: int
    attention: str
    compile: bool
    timing_rows: int


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("checkpoint", type=Path, help="TrainLoop checkpoint (.pt).")
    parser.add_argument("corpus", type=Path, help="Corpus of the validation episodes.")
    parser.add_argument("output", type=Path, help="Report JSON.")
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
    parser.add_argument("--episodes", type=int, default=8, help="Episodes; default 8.")
    parser.add_argument(
        "--decisions",
        type=int,
        default=640,
        help="Steps per episode; default 640, past the first re-prefill at 512.",
    )
    parser.add_argument(
        "--replicate",
        type=int,
        default=1,
        help="Engine rows per episode; 32 gives the production 256 rows.",
    )
    parser.add_argument("--hook", type=int, default=128, help="Steps between hooks.")
    parser.add_argument("--t-max", type=int, default=1_024, help="Positions per row.")
    parser.add_argument("--layers", type=int, default=20, help="Blocks to the tap.")
    parser.add_argument(
        "--attention",
        choices=["fa4", "masked"],
        default="fa4",
        help="The step's cache attention; default fa4.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the step's functions with Inductor, as production runs.",
    )
    parser.add_argument(
        "--timing-rows",
        type=int,
        default=256,
        help="Rows of the timed step; default 256.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
