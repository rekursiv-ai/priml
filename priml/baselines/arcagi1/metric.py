"""pass@K voting over each test input's augmented views.

Each ARC task is evaluated many times -- once per augmented view -- and the
model may answer differently on each. A view's prepared id names the VIEW, and
every test input of that view shares it, so the vote cannot be keyed by id:
:class:`CanonicalPassK` inverts each view to the canonical frame and votes per
``(task, canonical test input)``. A test input counts solved if its true grid
is among the top K answers; a task scores the mean over its test inputs, and
the reported number is the mean over tasks in ``test_puzzles.json``.

Ranking is by vote count, with mean halt confidence breaking ties. Count
dominates because agreement across independent views is the stronger signal;
confidence only separates answers that tied.

Predictions are stored as hashes, not grids. A full evaluation is hundreds of
thousands of 900-cell grids, and only equality between them matters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    cast,
    override,
)

import json
import logging
import re
import struct

from configgle import Fig, Makeable
from torch import Tensor

import numpy as np
import torch
import torch.distributed as dist

from priml.baselines.arcagi1.augmentation import (
    ArcSpec,
    ColorDihedral,
    arc_grid_to_np,
    crop_grid,
    grid_hash,
    untranslate_unscale,
)
from priml.lib.custom_json import convert, parse
from priml.paths import resolve_working_dir
from priml.runtime import is_rank_zero


if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)

type _Preds = dict[str, dict[str, list[tuple[str, float]]]]
type _SignalRow = tuple[str, str, str, float, float, float, int, int]
type _StepRow = tuple[int, int, tuple[float, ...], tuple[int, ...]]


class SignalDumpPayload(NamedTuple):
    """Per-rank signal-dump data :meth:`CanonicalPassK.compute` emits in ``extras``."""

    rows: list[_SignalRow]
    grids: dict[str, NDArray[np.uint8]]
    steps: list[_StepRow]
    pass_ks: tuple[int, ...]


class TaskScore(NamedTuple):
    """One task's per-K count of test inputs solved under the pass@K ranking."""

    solved: tuple[int, ...]
    """Solved test inputs, one entry per configured ``pass_ks`` value."""

    num_inputs: int
    """Test inputs the task asks for."""


class ScoringRule(Protocol):
    """Turns per-task pass@K counts into one reported score per K."""

    name: str
    """Metric prefix; each score is reported as ``"{name}@{k}"``."""

    def __call__(self, tasks: Sequence[TaskScore], k_index: int) -> float:
        """Score the ``k_index``-th K over every scored task."""
        ...


class StrictPass:
    """Fraction of tasks with every test input solved: the official ARC Prize rule."""

    class Config(Fig["StrictPass"]):
        """Reported as ``strict@K``."""

        name: str = "strict"
        """Metric prefix."""

    def __init__(self, config: Config) -> None:
        self.name = config.name

    def __call__(self, tasks: Sequence[TaskScore], k_index: int) -> float:
        """Return the all-inputs-solved task fraction at one K."""
        solved = sum(
            task.num_inputs > 0 and task.solved[k_index] == task.num_inputs
            for task in tasks
        )
        return solved / max(1, len(tasks))


class PerOutputPass:
    """Solved test inputs over all test inputs, pooled across tasks."""

    class Config(Fig["PerOutputPass"]):
        """Reported as ``per_output@K``."""

        name: str = "per_output"
        """Metric prefix."""

    def __init__(self, config: Config) -> None:
        self.name = config.name

    def __call__(self, tasks: Sequence[TaskScore], k_index: int) -> float:
        """Return the pooled solved-input fraction at one K."""
        solved = sum(task.solved[k_index] for task in tasks)
        return solved / max(1, sum(task.num_inputs for task in tasks))


class CanonicalPassK:
    """TRM reference pass@K over canonical test inputs, task-normalized.

    ``update`` consumes the packed model output: header columns (1 = halt
    logit; 3 = halt, log-prob, stability; ``5 + 2K`` adds per-ACT-step
    signals) followed by the predicted grid. Each non-blank row is untranslated
    and unscaled by its ``spatial_tags``, cropped, inverted to the canonical
    frame, and recorded under ``(task, canonical input hash)``.

    ``compute`` ranks answers per canonical input by ``(votes, mean
    sigmoid(q))``, averages hits over a task's test inputs, then over tasks.
    Report-only ``votes_times_mean_q`` / ``votes_times_max_q`` rankings never
    feed ``pass@K``.
    """

    class Config(Fig["CanonicalPassK"]):
        """Prepared dataset root, cutoffs, and which views to count."""

        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/arc1concept-aug-1000"
        """Dataset root holding ``identifiers.json`` and ``test_puzzles.json``."""

        pass_ks: tuple[int, ...] = (1, 2, 5, 10, 100, 1000)
        """Attempt budgets to report."""

        spatial_views: Literal["all", "non_spatial"] = "all"
        """Count every view, or only identity-spatial views."""

        max_views_per_input: int = 0
        """Keep the highest-confidence views per canonical input; 0 keeps all."""

        per_step_acts: int = 0
        """Per-ACT-step signal columns in a wide model output; 0 for none."""

        spec: ArcSpec = field(default_factory=ArcSpec)
        """Prepared dataset's packed-grid geometry and vocabulary."""

        transform: Makeable[ColorDihedral] = field(default_factory=ColorDihedral.Config)
        """Color/dihedral policy that encoded the prepared identifiers."""

        rules: list[Makeable[ScoringRule]] = field(
            default_factory=list[Makeable[ScoringRule]],
        )
        """Extra per-task scoring rules, each reported beside ``pass@K``."""

        exclude_tasks: list[str] = field(default_factory=list[str])
        """Tasks dropped before scoring, shrinking every denominator."""

        dump_per_task_path: Path | str = ""
        """Rank 0 writes every scored task's pass@K table here, hardest first.

        Each task maps to its ``pass@K`` rates, ``had_preds``, and the
        report-only rankings' rates. Taken verbatim, never joined under
        ``base_dir``: that root is the shared corpus, not a run directory.
        Empty writes nothing."""

        @override
        def finalize(self) -> Self:
            if (
                isinstance(self.transform, ColorDihedral.Config)
                and not self.transform.separator
            ):
                self.transform.separator = self.spec.puzzle_id_separator
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self._transform = config.transform.make()
        self._root = Path(config.working_dir).expanduser()
        self._rules: list[ScoringRule] = [rule.make() for rule in config.rules]
        self.reset()

    # Read on first use: the loop builds metrics before the dataset, which is what
    # stages this tree on a fresh machine.
    @cached_property
    def _identifier_map(self) -> list[str]:
        """``identifiers.json``: index is puzzle id."""
        return parse((self._root / "identifiers.json").read_text(), list[str])

    @cached_property
    def _test_puzzles(self) -> dict[str, dict[str, object]]:
        """Scored tasks from ``test_puzzles.json``, minus ``exclude_tasks``."""
        excluded = set(self.config.exclude_tasks)
        return {
            name: convert(puzzle, dict[str, object])
            for name, puzzle in parse(
                (self._root / "test_puzzles.json").read_text(),
                dict[str, object],
            ).items()
            if name not in excluded
        }

    @cached_property
    def _blank_identifier_id(self) -> int:
        """The id the loader pads with, so the two never desync."""
        meta_path = self._root / "test" / "dataset.json"
        meta = (
            parse(meta_path.read_text(), dict[str, object])
            if meta_path.is_file()
            else {}
        )
        return convert(meta.get("blank_identifier_id"), int, default=0)

    def reset(self) -> None:
        """Drop every accumulated ballot and dump row."""
        self._hmap: dict[str, tuple[int, int]] = {}
        self._preds: _Preds = {}
        self._dump_rows: list[_SignalRow] = []
        self._dump_grids: dict[str, NDArray[np.uint8]] = {}
        self._dump_steps: list[_StepRow] = []
        self._has_signal_dump = False

    def update(self, logits: Tensor, **batch: object) -> None:
        """Record one batch of canonical ballots.

        Args:
          logits: Packed header columns followed by predicted grid tokens.
          **batch: ``media`` and ``puzzle_identifiers``; optional ``spatial_tags``.

        """
        out = logits.detach().cpu()
        media = batch["media"]
        assert isinstance(media, Tensor)
        media_np = _uint8_rows(media.detach().cpu())
        k_steps = self.config.per_step_acts
        n_header = _model_output_header_width(
            out_width=out.shape[1],
            media_len=media.shape[1],
            k_steps=k_steps,
        )
        dumping = n_header >= 3
        self._has_signal_dump |= dumping
        # float64 before sigmoid, so confident logits do not saturate to 0 or 1.
        q_halt = _floats(out[:, 0].to(torch.float64).sigmoid())
        preds_t = _uint8_rows(out[:, n_header:])
        raw_q_halt = _floats(out[:, 0].to(torch.float32))
        steps = _act_step_rows(out, k_steps)
        puzzle_ids = convert(
            _integer_field(batch, "puzzle_identifiers").tolist(),
            list[int],
        )
        spatial_tags = (
            convert(_integer_field(batch, "spatial_tags").tolist(), list[list[int]])
            if "spatial_tags" in batch
            else [[1, 0, 0]] * len(puzzle_ids)
        )
        for i, ident in enumerate(puzzle_ids):
            if ident == self._blank_identifier_id:
                continue
            if ident < 0 or ident >= len(self._identifier_map):
                raise ValueError(
                    f"puzzle identifier {ident} is outside identifier map size "
                    f"{len(self._identifier_map)}.",
                )
            scale, pad_r, pad_c = spatial_tags[i]
            if self.config.spatial_views == "non_spatial" and (
                scale != 1 or pad_r != 0 or pad_c != 0
            ):
                continue
            orig_name, inv_fn = self._transform.inverse(self._identifier_map[ident])
            input_canon = inv_fn(
                crop_grid(
                    untranslate_unscale(
                        media_np[i],
                        scale=scale,
                        pad_r=pad_r,
                        pad_c=pad_c,
                    ),
                    spec=self.config.spec,
                ),
            )
            input_h = grid_hash(input_canon)
            pred_canon = inv_fn(
                crop_grid(
                    untranslate_unscale(
                        preds_t[i],
                        scale=scale,
                        pad_r=pad_r,
                        pad_c=pad_c,
                    ),
                    spec=self.config.spec,
                ),
            )
            pred_h = grid_hash(pred_canon)
            n_rows, n_cols = _shape(pred_canon)
            self._hmap[pred_h] = (n_rows, n_cols)
            self._preds.setdefault(orig_name, {}).setdefault(input_h, []).append(
                (pred_h, q_halt[i]),
            )
            if dumping:
                logprob = float(out[i, 1].to(torch.float32))
                stability = float(out[i, 2].to(torch.float32))
                self._dump_rows.append(
                    (
                        orig_name,
                        input_h,
                        pred_h,
                        raw_q_halt[i],
                        logprob,
                        stability,
                        n_rows,
                        n_cols,
                    ),
                )
                self._dump_grids.setdefault(pred_h, pred_canon)
                if k_steps:
                    self._dump_steps.append(steps[i])

    def compute(self) -> dict[str, object]:
        """Rank each canonical input's answers and score every K.

        Returns:
          metrics: ``pass@K`` and report-only rankings, plus ``extras`` holding a
            :class:`SignalDumpPayload` when the model output carried signals.

        """
        preds = self._global_preds()
        pass_ks = list(self.config.pass_ks)
        correct = [0.0 for _ in pass_ks]
        correct_report_only = {
            rank_name: [0.0 for _ in pass_ks] for rank_name in _REPORT_ONLY_RANK_SCORERS
        }
        n_test_puzzles = len(self._test_puzzles)
        if n_test_puzzles == 0:
            empty: dict[str, object] = {f"pass@{k}": 0.0 for k in pass_ks}
            empty.update(
                {
                    f"{rank_name}@{k}": 0.0
                    for rank_name in _REPORT_ONLY_RANK_SCORERS
                    for k in pass_ks
                },
            )
            empty.update(self._rule_scores([]))
            if self._has_signal_dump:
                empty["extras"] = {"signal_dump": self._signal_dump_payload()}
            return empty
        per_task_pass1: list[tuple[str, float]] = []
        per_task: dict[str, dict[str, float]] = {}
        task_scores: list[TaskScore] = []
        pass1_idx = pass_ks.index(1) if 1 in pass_ks else None
        n_no_preds = 0
        for name, puzzle in self._test_puzzles.items():
            pairs = convert(
                puzzle.get("test", []),
                list[dict[str, object]],
            )
            per_test_correct = [0 for _ in pass_ks]
            per_test_report = {
                rank_name: [0 for _ in pass_ks]
                for rank_name in _REPORT_ONLY_RANK_SCORERS
            }
            pair_inputs = [
                (
                    pair,
                    grid_hash(_json_grid(pair["input"], spec=self.config.spec)),
                )
                for pair in pairs
            ]
            has_predictions = any(
                preds.get(name, {}).get(input_h) for _, input_h in pair_inputs
            )
            for pair, input_h in pair_inputs:
                label_h = grid_hash(_json_grid(pair["output"], spec=self.config.spec))
                records = preds.get(name, {}).get(input_h, [])
                cap = self.config.max_views_per_input
                if cap > 0 and len(records) > cap:
                    # The prediction hash only breaks ties, keeping the slice deterministic.
                    records = sorted(records, key=lambda r: (-r[1], r[0]))[:cap]
                p_map: dict[str, list[float]] = {}
                max_q_of: dict[str, float] = {}
                for h, q in records:
                    p_map.setdefault(h, []).append(q)
                    max_q_of[h] = max(max_q_of.get(h, 0.0), q)
                if not p_map:
                    continue
                ranked = sorted(
                    p_map.items(),
                    key=lambda item: (
                        len(item[1]),
                        sum(item[1]),
                    ),
                    reverse=True,
                )
                for i, k in enumerate(pass_ks):
                    if any(h == label_h for h, _ in ranked[:k]):
                        per_test_correct[i] += 1
                for rank_name, scorer in _REPORT_ONLY_RANK_SCORERS.items():
                    # The hash breaks exact ties of the collapsed score.
                    report_ranked = sorted(
                        p_map,
                        key=lambda h, s=scorer: (
                            -s(
                                len(p_map[h]),
                                sum(p_map[h]) / len(p_map[h]),
                                max_q_of[h],
                            ),
                            h,
                        ),
                    )
                    for i, k in enumerate(pass_ks):
                        if label_h in report_ranked[:k]:
                            per_test_report[rank_name][i] += 1
            n_test = len(pairs)
            task_scores.append(TaskScore(tuple(per_test_correct), n_test))
            if n_test == 0:
                continue
            if not has_predictions:
                n_no_preds += 1
            for i in range(len(pass_ks)):
                correct[i] += per_test_correct[i] / n_test
            for rank_name, report_correct in per_test_report.items():
                for i in range(len(pass_ks)):
                    correct_report_only[rank_name][i] += report_correct[i] / n_test
            if pass1_idx is not None:
                per_task_pass1.append((name, per_test_correct[pass1_idx] / n_test))
            row = {
                f"pass@{k}": per_test_correct[i] / n_test for i, k in enumerate(pass_ks)
            }
            row["had_preds"] = float(has_predictions)
            for rank_name, report_correct in per_test_report.items():
                for i, k in enumerate(pass_ks):
                    row[f"{rank_name}@{k}"] = report_correct[i] / n_test
            per_task[name] = row
        results: dict[str, object] = {
            f"pass@{k}": correct[i] / n_test_puzzles for i, k in enumerate(pass_ks)
        }
        for rank_name, totals in correct_report_only.items():
            results.update(
                {
                    f"{rank_name}@{k}": totals[i] / n_test_puzzles
                    for i, k in enumerate(pass_ks)
                },
            )
        results.update(self._rule_scores(task_scores))
        self._log_per_task(results, per_task_pass1, n_test_puzzles, n_no_preds)
        if self.config.dump_per_task_path:
            self._dump_per_task(per_task)
        if self._has_signal_dump:
            results["extras"] = {"signal_dump": self._signal_dump_payload()}
        return results

    class StateDict(TypedDict):
        """Checkpointed ballots; ``preds`` records are ``[hash, q]`` lists."""

        hmap: NotRequired[dict[str, list[int]]]
        preds: NotRequired[dict[str, dict[str, list[list[str | float]]]]]

    def state_dict(self) -> StateDict:
        """Return the accumulated ballots in a JSON-friendly shape.

        Returns:
          state: ``hmap`` and ``preds`` with tuples as lists.

        """
        return {
            "hmap": {name: list(shape) for name, shape in self._hmap.items()},
            "preds": {
                name: {ih: [[h, q] for h, q in vs] for ih, vs in by_input.items()}
                for name, by_input in self._preds.items()
            },
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore ballots produced by :meth:`state_dict`.

        Args:
          state_dict: Saved ``hmap`` and ``preds``.

        """
        state = cast(CanonicalPassK.StateDict, state_dict)
        self._hmap = {
            name: _grid_shape(shape)
            for name, shape in convert(
                state.get("hmap"),
                dict[str, object],
                default={},
            ).items()
        }
        self._preds = {
            name: {
                ih: [(str(h), convert(q, float)) for h, q in vs]
                for ih, vs in by_input.items()
            }
            for name, by_input in state.get("preds", {}).items()
        }

    def _rule_scores(self, tasks: Sequence[TaskScore]) -> dict[str, float]:
        """Score every configured rule at every K."""
        return {
            f"{rule.name}@{k}": rule(tasks, index)
            for rule in self._rules
            for index, k in enumerate(self.config.pass_ks)
        }

    def _signal_dump_payload(self) -> SignalDumpPayload:
        return SignalDumpPayload(
            rows=self._dump_rows,
            grids=self._dump_grids,
            steps=self._dump_steps,
            pass_ks=tuple(self.config.pass_ks),
        )

    # Ascending pass@1 (stable, so ties keep test_puzzles order) puts the hardest tasks
    # first: the difficulty ranking proxy-subset selection reads.
    def _dump_per_task(self, per_task: dict[str, dict[str, float]]) -> None:
        """Write the per-task table to ``dump_per_task_path`` on rank 0."""
        if not is_rank_zero():
            return
        ordered = dict(
            sorted(per_task.items(), key=lambda item: item[1].get("pass@1", 0.0)),
        )
        path = Path(self.config.dump_per_task_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ordered, indent=2))
        logger.info(
            "[eval] wrote per-task pass@k table (%d tasks) to %s",
            len(ordered),
            path,
        )

    def _log_per_task(
        self,
        results: Mapping[str, object],
        per_task_pass1: list[tuple[str, float]],
        n_test_puzzles: int,
        n_no_preds: int,
    ) -> None:
        """Log the rank-0 per-task breakdown."""
        if not is_rank_zero():
            return
        summary = " ".join(
            f"{k}={v:.4f}" for k, v in results.items() if isinstance(v, float)
        )
        logger.info(
            "[eval] pass@K over %d tasks (%d with no predictions): %s",
            n_test_puzzles,
            n_no_preds,
            summary,
        )
        if not per_task_pass1:
            return
        solved = sum(1 for _, acc in per_task_pass1 if acc > 0)
        logger.info(
            "[eval] per-task pass@1: %d/%d tasks solved (>0). hardest unsolved:",
            solved,
            len(per_task_pass1),
        )
        for name, acc in sorted(per_task_pass1, key=lambda kv: kv[1])[:10]:
            logger.info("[eval]   %s pass@1=%.3f", name, acc)

    def _global_preds(self) -> _Preds:
        """Merge every rank's ballots in rank order."""
        if not dist.is_available() or not dist.is_initialized():
            return self._preds
        payload = encode_preds(self._preds)
        # NCCL gathers CUDA tensors; gloo and other CPU backends gather on CPU.
        device = torch.device("cpu")
        if dist.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        size = torch.tensor([len(payload)], device=device)
        sizes = [torch.empty_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, size)
        max_size = max(int(size_t.item()) for size_t in sizes)
        send = torch.zeros(max_size, dtype=torch.uint8, device=device)
        send[: len(payload)] = torch.frombuffer(
            bytearray(payload),
            dtype=torch.uint8,
        ).to(
            device,
        )
        recv = [torch.empty_like(send) for _ in sizes]
        dist.all_gather(recv, send)
        merged: _Preds = {}
        for index, tensor in enumerate(recv):
            size_t = sizes[index]
            part = decode_preds(bytes(tensor[: int(size_t.item())].cpu().numpy()))
            for name, by_input in part.items():
                target = merged.setdefault(name, {})
                for input_h, values in by_input.items():
                    target.setdefault(input_h, []).extend(values)
        return merged


class SignalDumpTracker:
    """Tracker writing :class:`CanonicalPassK`'s signal dump from eval ``extras``."""

    class Config(Fig["SignalDumpTracker"]):
        """Signal-dump destination."""

        base_dir: Path | str | None = None
        """Run root supplied during parent finalization."""

        working_dir: Path | str = "/signals_{global_step}.npz"
        """Logical destination; ``{global_step}`` is formatted at write time."""

        spec: ArcSpec = field(default_factory=ArcSpec)
        """Prepared dataset's packed-grid geometry."""

        keep_last_n: int = -1
        """Newest step-stamped dumps kept after each write; ``-1`` keeps all.

        Rotation deletes files an offline analysis may still want: copy a dump
        out of the run directory to pin it, or archive with ``keep_every``."""

        keep_every: int = 0
        """Never delete a dump whose step is a multiple of this; ``0`` is off."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.keep_last_n < -1 or config.keep_last_n == 0:
            raise ValueError(
                "keep_last_n must be -1 (keep all) or positive; got "
                f"{config.keep_last_n}.",
            )
        if config.keep_every < 0:
            raise ValueError(f"keep_every must be >= 0; got {config.keep_every}.")
        self.config = config

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Write the dump carried in ``metrics["extras"]`` for evaluation metrics.

        Args:
          metrics: Evaluation payload; ``extras["signal_dump"]`` holds the dump.
          step: Global step, formatted into ``{global_step}``.
          prefix: Only ``"eval/"`` writes.

        """
        if prefix != "eval/":
            return
        extras = metrics.get("extras")
        if extras is None:
            if not _any_rank(False):
                return
            payload = SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=())
        else:
            try:
                extras_map = convert(extras, dict[str, object])
            except TypeError as err:
                raise TypeError(
                    "SignalDumpTracker expected metrics['extras'] to be "
                    f"a mapping, got {type(extras).__name__}.",
                ) from err
            found = extras_map.get("signal_dump")
            if not isinstance(found, SignalDumpPayload):
                raise TypeError(
                    "SignalDumpTracker expected extras['signal_dump'] to be "
                    f"SignalDumpPayload, got {type(found).__name__}.",
                )
            if not _any_rank(True):
                return
            payload = found
        write_signal_dump(
            payload=payload,
            dump_signals_path=str(self.config.working_dir),
            global_step=step,
            spec=self.config.spec,
        )
        self._prune()

    def _prune(self) -> None:
        """Delete the oldest step-stamped dumps beyond ``keep_last_n``, on rank 0."""
        template = str(self.config.working_dir)
        if self.config.keep_last_n == -1 or "{global_step}" not in template:
            return
        if not is_rank_zero():
            return
        path = Path(template).expanduser()
        pattern = re.compile(
            re.escape(path.name).replace(re.escape("{global_step}"), r"(\d+)"),
        )
        dumps = sorted(
            (int(match.group(1)), candidate)
            for candidate in path.parent.glob(path.name.replace("{global_step}", "*"))
            if (match := pattern.fullmatch(candidate.name))
        )
        every = self.config.keep_every
        for step, doomed in dumps[: max(0, len(dumps) - self.config.keep_last_n)]:
            if every > 0 and step % every == 0:
                continue
            doomed.unlink()
            logger.info("Deleted signal dump %s (keep_last_n rotation).", doomed)

    def log_images(self, key: str, images: list[object], step: int) -> None:
        """Ignore images; this tracker writes only the signal dump."""
        del key, images, step

    def log_notes(self, notes: str) -> None:
        """Ignore run notes; a signal dump has none."""
        del notes

    def close(self) -> None:
        """Release nothing; every write is synchronous."""


def write_signal_dump(
    *,
    payload: SignalDumpPayload,
    dump_signals_path: str | Path,
    global_step: int,
    spec: ArcSpec,
) -> None:
    """Gather every rank's dump and write it as one ``.npz`` from rank 0.

    Args:
      payload: This rank's rows, unique prediction grids, and step rows.
      dump_signals_path: A ``Path`` is literal; a ``str`` formats ``{global_step}``.
      global_step: Step substituted into a ``str`` path.
      spec: Dataset-owned packed-grid geometry.

    """
    if isinstance(dump_signals_path, Path):
        out_path = str(dump_signals_path)
    else:
        out_path = dump_signals_path.replace("{global_step}", str(global_step))
        if "{" in out_path or "}" in out_path:
            # A surviving placeholder means finalize never filled the run context.
            if is_rank_zero():
                logger.info(
                    "[eval] signal dump skipped: unresolved path %r",
                    dump_signals_path,
                )
            return
    rows = _gather_list(payload.rows)
    grids = _gather_grids(payload.grids)
    steps = _gather_list(payload.steps)
    if not is_rank_zero():
        return
    n = len(rows)
    group_ids = np.empty(n, dtype=np.int32)
    pred_ids = np.empty(n, dtype=np.int32)
    q_halt = np.empty(n, dtype=np.float32)
    logprob = np.empty(n, dtype=np.float32)
    stability = np.empty(n, dtype=np.float32)
    n_rows = np.empty(n, dtype=np.uint8)
    n_cols = np.empty(n, dtype=np.uint8)
    group_index: dict[tuple[str, str], int] = {}
    pred_index: dict[str, int] = {}
    group_table: list[str] = []
    pred_table: list[str] = []
    for j, (name, input_h, pred_h, qh, lp, st, nr, nc) in enumerate(rows):
        gid = group_index.get((name, input_h))
        if gid is None:
            gid = group_index[name, input_h] = len(group_table)
            group_table.append(f"{name}\t{input_h}")
        pid = pred_index.get(pred_h)
        if pid is None:
            pid = pred_index[pred_h] = len(pred_table)
            pred_table.append(pred_h)
        group_ids[j] = gid
        pred_ids[j] = pid
        q_halt[j] = qh
        logprob[j] = lp
        stability[j] = st
        n_rows[j] = nr
        n_cols[j] = nc
    n_pred = len(pred_table)
    pred_grids = np.zeros((n_pred, spec.max_grid, spec.max_grid), dtype=np.uint8)
    pred_n_rows = np.zeros(n_pred, dtype=np.uint8)
    pred_n_cols = np.zeros(n_pred, dtype=np.uint8)
    for pid, pred_h in enumerate(pred_table):
        grid = grids.get(pred_h)
        if grid is None:
            continue
        gr, gc = _shape(grid)
        pred_grids[pid, :gr, :gc] = grid
        pred_n_rows[pid] = gr
        pred_n_cols[pid] = gc
    arrays: dict[str, NDArray[np.generic]] = {
        "group_id": group_ids,
        "pred_id": pred_ids,
        "q_halt": q_halt,
        "logprob": logprob,
        "stability": stability,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "group_table": np.array(group_table),
        "pred_table": np.array(pred_table),
        "pred_grids": pred_grids,
        "pred_n_rows": pred_n_rows,
        "pred_n_cols": pred_n_cols,
        "pass_ks": np.array(list(payload.pass_ks)),
    }
    if steps:
        arrays["converge_step"] = np.array([s[0] for s in steps], dtype=np.uint8)
        # A 30x30 grid can change 900 cells per step; uint8 would wrap.
        arrays["n_changes"] = np.array([s[1] for s in steps], dtype=np.uint16)
        arrays["q_halt_steps"] = np.array([s[2] for s in steps], dtype=np.float32)
        arrays["correct_step"] = np.array([s[3] for s in steps], dtype=np.uint8)
    path = Path(out_path).expanduser()
    if path.exists():
        logger.warning("[eval] overwriting existing signal dump at %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The stub's ``**kwds: ArrayLike`` collides with ``allow_pickle`` under unpacking.
    cast(Callable[..., object], np.savez_compressed)(path, **arrays)
    logger.info(
        "[eval] wrote signal dump (%d rows, %d groups, %d preds, %d bytes) to %s",
        n,
        len(group_table),
        n_pred,
        path.stat().st_size,
        path,
    )


def encode_preds(
    preds: Mapping[str, Mapping[str, Sequence[tuple[str, float]]]],
) -> bytes:
    """Encode ballots compactly, keeping every IEEE-754 double bit-for-bit.

    Args:
      preds: Task -> canonical input hash -> ``(prediction hash, confidence)``.

    Returns:
      payload: Length-prefixed names, raw 32-byte digests, little-endian doubles.

    """
    data = bytearray()
    data.extend(struct.pack("<I", len(preds)))
    for name, by_input in preds.items():
        encoded = name.encode()
        data.extend(struct.pack("<I", len(encoded)))
        data.extend(encoded)
        data.extend(struct.pack("<I", len(by_input)))
        for input_h, values in by_input.items():
            data.extend(_hash_bytes(input_h))
            data.extend(struct.pack("<I", len(values)))
            for pred_h, q in values:
                data.extend(_hash_bytes(pred_h))
                data.extend(struct.pack("<d", q))
    return bytes(data)


def decode_preds(payload: bytes) -> _Preds:
    """Decode :func:`encode_preds` bytes.

    Args:
      payload: Bytes from :func:`encode_preds`.

    Returns:
      preds: Task -> canonical input hash -> ``(prediction hash, confidence)``.

    """
    preds: _Preds = {}
    n_names, offset = _read_u32(payload, 0)
    for _ in range(n_names):
        size, offset = _read_u32(payload, offset)
        name = payload[offset : offset + size].decode()
        offset += size
        by_input: dict[str, list[tuple[str, float]]] = {}
        n_inputs, offset = _read_u32(payload, offset)
        for _ in range(n_inputs):
            input_h = payload[offset : offset + 32].hex()
            n_values, offset = _read_u32(payload, offset + 32)
            values: list[tuple[str, float]] = []
            for _ in range(n_values):
                pred_h = payload[offset : offset + 32].hex()
                (q,) = struct.unpack_from("<d", payload, offset + 32)
                offset += 40
                values.append((pred_h, convert(q, float)))
            by_input[input_h] = values
        preds[name] = by_input
    return preds


def _read_u32(payload: bytes, offset: int) -> tuple[int, int]:
    (value,) = struct.unpack_from("<I", payload, offset)
    return convert(value, int), offset + 4


def _votes_times_mean_q(count: float, mean_q: float, max_q: float) -> float:
    del max_q
    return count * mean_q


def _votes_times_max_q(count: float, mean_q: float, max_q: float) -> float:
    del mean_q
    return count * max_q


# Report-only; never replaces the ``(votes, mean_q)`` key outside a held-out comparison.
_REPORT_ONLY_RANK_SCORERS: dict[str, Callable[[float, float, float], float]] = {
    "votes_times_mean_q": _votes_times_mean_q,
    "votes_times_max_q": _votes_times_max_q,
}


def _integer_field(batch: Mapping[str, object], name: str) -> Tensor:
    """Return ``batch[name]`` on CPU, rejecting a floating tensor by field name."""
    value = batch[name]
    assert isinstance(value, Tensor)
    if value.dtype.is_floating_point:
        raise TypeError(f"CanonicalPassK expects integer {name}; got {value.dtype}.")
    return value.detach().cpu()


def _act_step_rows(out: Tensor, k_steps: int) -> list[_StepRow]:
    if k_steps == 0:
        return []
    converge = _ints(out[:, 3])
    n_changes = _ints(out[:, 4])
    q_halt_steps = out[:, 5 : 5 + k_steps].to(torch.float32)
    correct_steps = out[:, 5 + k_steps : 5 + 2 * k_steps]
    return [
        (
            converge[index],
            n_changes[index],
            tuple(_floats(q_halt_steps[index])),
            tuple(_ints(correct_steps[index])),
        )
        for index in range(out.shape[0])
    ]


def _model_output_header_width(*, out_width: int, media_len: int, k_steps: int) -> int:
    """Return the count of leading non-prediction columns."""
    if k_steps > 0:
        n_header = 5 + 2 * k_steps
        if out_width != n_header + media_len:
            raise ValueError(
                f"per_step_acts={k_steps} expects model_output width "
                f"{n_header + media_len} (header {n_header} + grid {media_len}); "
                f"got {out_width}.",
            )
        return n_header
    n_header = out_width - media_len
    if n_header not in (1, 3):
        pred_len = out_width - max(1, n_header)
        raise ValueError(
            f"model_output width {out_width} minus grid length {media_len} = "
            f"header {n_header}; inferred prediction length {pred_len}; expected "
            "1 (baseline) or 3 (wide signal-dump) leading columns.",
        )
    return n_header


# An integer token tensor narrows to uint8 by wrapping, as the reference voted on.
def _uint8_rows(values: Tensor) -> list[NDArray[np.uint8]]:
    array = cast("NDArray[np.integer]", values.numpy()).astype(np.uint8)
    return list(cast("Iterable[NDArray[np.uint8]]", array))


def _floats(values: Tensor) -> list[float]:
    return [float(value) for value in values]


def _ints(values: Tensor) -> list[int]:
    return [int(value) for value in values.to(torch.int64)]


def _json_grid(value: object, *, spec: ArcSpec) -> NDArray[np.uint8]:
    rows = [
        [_json_cell(cell) for cell in convert(row, list[object])]
        for row in convert(value, list[object])
    ]
    return arc_grid_to_np(rows, max_grid=spec.max_grid)


def _json_cell(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"ARC grid cell must be an integer color; got {value!r}.")
    return value


def _shape(grid: NDArray[np.uint8]) -> tuple[int, int]:
    rows, cols = cast(tuple[int, int], grid.shape)
    return rows, cols


# Tuples are accepted alongside lists: checkpoints written before ``state_dict``
# emitted lists hold the in-memory tuples, and a JSON round trip yields lists.
def _grid_shape(value: object) -> tuple[int, int]:
    """Validate a checkpointed ``(rows, cols)`` pair."""
    match value:
        case [int() as rows, int() as cols] if not isinstance(
            rows,
            bool,
        ) and not isinstance(cols, bool):
            return rows, cols
        case _:
            raise TypeError(f"Invalid checkpointed grid shape: {value!r}.")


def _hash_bytes(value: str) -> bytes:
    if len(value) != 64:
        raise ValueError("Expected len(value) == 64.")
    return bytes.fromhex(value)


def _any_rank(has_payload: bool) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return has_payload
    gathered: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, has_payload)
    return any(bool(value) for value in gathered)


def _gather_list[T](items: list[T]) -> list[T]:
    if not dist.is_available() or not dist.is_initialized():
        return items
    gathered: list[list[T] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, items)
    return [item for part in gathered if part is not None for item in part]


def _gather_grids(grids: dict[str, NDArray[np.uint8]]) -> dict[str, NDArray[np.uint8]]:
    if not dist.is_available() or not dist.is_initialized():
        return grids
    gathered: list[dict[str, NDArray[np.uint8]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, grids)
    merged: dict[str, NDArray[np.uint8]] = {}
    for part in gathered:
        if part is not None:
            merged.update(part)
    return merged
