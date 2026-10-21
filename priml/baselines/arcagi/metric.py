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

Vote keys are stored as hashes while candidate grids are retained for
submissions. A full evaluation is hundreds of thousands of 900-cell grids,
and only equality between vote keys matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    NotRequired,
    Protocol,
    Self,
    SupportsFloat,
    SupportsIndex,
    cast,
    override,
)
from typing_extensions import TypedDict

import logging
import operator
import struct

from configgle import Fig, Makeable
from numpy.typing import NDArray
from torch import Tensor
from treekle import from_plain, json_loads

import numpy as np
import torch
import torch.distributed as dist

from priml.baselines.arcagi.augmentation import (
    ArcSpec,
    arc_grid_to_np,
    crop_grid,
    grid_hash,
    untranslate_unscale,
)
from priml.baselines.arcagi.data import read_parquet_source
from priml.baselines.arcagi.signal_dump import (
    SignalDumpPayload,
    SignalRow,
    StepRow,
)
from priml.data.color_dihedral import ColorDihedral
from priml.paths import resolve_working_dir
from priml.runtime import is_rank_zero


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence


logger = logging.getLogger(__name__)

type _Preds = dict[str, dict[str, list[tuple[str, float]]]]


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
        return json_loads((self._root / "identifiers.json").read_text(), list[str])

    @cached_property
    def _test_puzzles(self) -> dict[str, dict[str, object]]:
        """Scored tasks from ``test_puzzles.json``."""
        return {
            name: from_plain(puzzle, dict[str, object])
            for name, puzzle in json_loads(
                (self._root / "test_puzzles.json").read_text(),
                dict[str, object],
            ).items()
        }

    @cached_property
    def _blank_identifier_id(self) -> int:
        """The id the loader pads with, so the two never desync."""
        meta_path = self._root / "test" / "dataset.json"
        meta = (
            json_loads(meta_path.read_text(), dict[str, object])
            if meta_path.is_file()
            else {}
        )
        return from_plain(meta.get("blank_identifier_id"), int, default=0)

    def reset(self) -> None:
        """Drop every accumulated ballot and dump row."""
        self._hmap: dict[str, tuple[int, int]] = {}
        self._preds: _Preds = {}
        self._dump_rows: list[SignalRow] = []
        self._dump_grids: dict[str, NDArray[np.uint8]] = {}
        self._dump_steps: list[StepRow] = []
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
        puzzle_ids = from_plain(
            _integer_field(batch, "puzzle_identifiers").tolist(),
            list[int],
        )
        spatial_tags = (
            from_plain(_integer_field(batch, "spatial_tags").tolist(), list[list[int]])
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
        task_scores: list[TaskScore] = []
        pass1_idx = pass_ks.index(1) if 1 in pass_ks else None
        n_no_preds = 0
        for name, puzzle in self._test_puzzles.items():
            pairs = from_plain(
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
                ranked = _rank_votes(records)
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
            for name, shape in from_plain(
                state.get("hmap"),
                dict[str, object],
                default={},
            ).items()
        }
        self._preds = {
            name: {
                ih: [(str(h), from_plain(q, float)) for h, q in vs]
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


class ParquetPassK:
    """Score augmented ARC predictions against an ordered Parquet source."""

    class Config(Fig["ParquetPassK"]):
        """Evaluation source, packed canvas, pass budgets, and event window."""

        base_dir: Path | str | None = None
        """Resource root used to resolve a relative evaluation source."""

        eval_source: Path | str = ""
        """Ordered Parquet file supplying the evaluation pairs."""

        canvas_size: int = 30
        """Square token canvas side used to decode packed predictions."""

        pass_ks: tuple[int, ...] = (1, 2, 5, 10, 100, 1000)
        """Vote budgets reported as pass@K metrics."""

        recent_window_size: int | None = None
        """Maximum prior event ballots retained; None keeps all events."""

        aggregated_voting: bool = True
        """Whether ballots accumulate across evaluation events."""

        @override
        def finalize(self) -> Self:
            """Resolve the evaluation path without reading its Parquet file."""
            self.eval_source = resolve_working_dir(self.base_dir, self.eval_source)
            return super().finalize()

    class StateDict(TypedDict):
        event_counter: int
        active_event_id: int | None
        events: dict[int, dict[tuple[str, str], list[_ParquetVoteState]]]
        prediction_grids: dict[str, Tensor]
        rank_states: list[_ParquetRankState]

    def __init__(self, config: Config) -> None:
        _validate_parquet_metric_config(config)
        self.config = config
        self._ground_truth: dict[str, tuple[_ParquetGroundTruth, ...]] | None = None
        self._events: dict[int, dict[tuple[str, str], list[_ArcVote]]] = {}
        self._prediction_grids: dict[str, NDArray[np.uint8]] = {}
        self._restored_rank_states: list[_ParquetRankState] = []
        self._event_counter = 0
        self._active_event_id: int | None = None

    def reset(self) -> None:
        """Start a voting event and retain only the configured history window."""
        if not self.config.aggregated_voting:
            self._events.clear()
            self._prediction_grids.clear()
        event_id = self._event_counter
        self._event_counter += 1
        self._events[event_id] = {}
        self._active_event_id = event_id
        self._trim_history()

    def update(self, logits: Tensor, **batch: object) -> None:
        """Invert packed predictions and append votes for canonical inputs.

        Args:
          logits: Halt logits followed by CanvasEOS token grids.
          **batch: Canonical names and inputs, view IDs, tags, and valid row count.

        """
        expected_width = 1 + self.config.canvas_size**2
        if logits.ndim != 2 or logits.shape[1] != expected_width:
            raise ValueError(
                "ParquetPassK expects packed ARC model output shaped "
                f"[batch, {expected_width}] (halt logit and CanvasEOS tokens).",
            )
        names = _parquet_batch_names(batch)
        inputs = _parquet_batch_inputs(batch)
        tags = _tensor_field(batch, name="spatial_tags")
        dihedrals = _tensor_field(batch, name="dih_idxs")
        color_slots = _tensor_field(batch, name="colorperm_slots")
        valid_count = _valid_count(batch.get("valid_count"), batch_size=logits.shape[0])
        if (
            len(names) != valid_count
            or len(inputs) != valid_count
            or tags.shape != (logits.shape[0], 3)
            or dihedrals.shape != (logits.shape[0],)
            or color_slots.shape != (logits.shape[0], 9)
        ):
            raise ValueError("Canonical metadata must match packed batch size.")
        if self._active_event_id is None:
            self.reset()
        assert isinstance(self._active_event_id, int)
        event = self._events.setdefault(self._active_event_id, {})
        confidences = torch.sigmoid(logits[:valid_count, 0].to(torch.float32))
        tag_values = cast(
            NDArray[np.int64],
            tags[:valid_count].to(torch.int64).detach().cpu().numpy(),
        )
        dih_values = cast(
            NDArray[np.int64],
            dihedrals[:valid_count].to(torch.int64).detach().cpu().numpy(),
        )
        slot_values = cast(
            NDArray[np.int64],
            color_slots[:valid_count].to(torch.int64).detach().cpu().numpy(),
        )
        confidence_values = cast(
            NDArray[np.float32],
            confidences.detach().cpu().numpy(),
        )
        spec = ArcSpec(max_grid=self.config.canvas_size)
        transform = ColorDihedral.Config().make()
        for index in range(valid_count):
            scale = operator.index(cast(SupportsIndex, tag_values[index, 0]))
            row = operator.index(cast(SupportsIndex, tag_values[index, 1]))
            col = operator.index(cast(SupportsIndex, tag_values[index, 2]))
            if scale != 1:
                raise ValueError(f"Unsupported online scale {scale}; expected 1.")
            candidate = crop_grid(
                logits[index, 1:].detach().to(torch.uint8).cpu().numpy(),
                spec=spec,
                row=row,
                col=col,
            )
            slots = tuple(
                operator.index(cast(SupportsIndex, slot_values[index, slot]))
                for slot in range(9)
            )
            candidate = transform.inverse_explicit(
                candidate,
                tid=operator.index(cast(SupportsIndex, dih_values[index])),
                colorperm_slots=slots,
            )
            name = names[index]
            input_hash = grid_hash(inputs[index])
            prediction_hash = grid_hash(candidate)
            event.setdefault((name, input_hash), []).append(
                _ArcVote(
                    prediction_hash=prediction_hash,
                    confidence=float(cast(SupportsFloat, confidence_values[index])),
                ),
            )
            self._prediction_grids.setdefault(prediction_hash, candidate.copy())

    def compute(self) -> dict[str, float]:
        """Return source-compatible task-normalized pass@K scores.

        Returns:
          scores: Mean per-task pass@K over the full evaluation source.

        """
        events, _ = self._gather()
        if events is None:
            return {}
        predictions = _merge_parquet_events(
            events,
            event_ids=self._effective_event_ids(events),
        )
        truth = self._ground_truth_pairs()
        if not any(pair.has_output for pairs in truth.values() for pair in pairs):
            return {}
        scores = dict.fromkeys(self.config.pass_ks, 0.0)
        task_count = 0
        for puzzle_name, pairs in truth.items():
            if not pairs:
                continue
            correct = dict.fromkeys(self.config.pass_ks, 0)
            for pair in pairs:
                if pair.output_hash is None:
                    continue
                ranked = _rank_votes(
                    [
                        (vote.prediction_hash, vote.confidence)
                        for vote in predictions.get((puzzle_name, pair.input_hash), [])
                    ],
                )
                for k in self.config.pass_ks:
                    if any(
                        prediction_hash == pair.output_hash
                        for prediction_hash, _ in ranked[:k]
                    ):
                        correct[k] += 1
            for k in self.config.pass_ks:
                scores[k] += correct[k] / len(pairs)
            task_count += 1
        if task_count:
            scores = {k: score / task_count for k, score in scores.items()}
        return {f"pass@{k}": scores[k] for k in self.config.pass_ks}

    def submission(
        self,
        *,
        submission_k: int = 100,
    ) -> dict[str, list[dict[str, list[list[int]]]]]:
        """Return ranked source-schema attempts in canonical task/input order.

        Args:
          submission_k: Number of attempts emitted for each canonical test input.

        Returns:
          submission: Puzzle IDs mapped to ordered attempt dictionaries.

        """
        if submission_k < 1:
            raise ValueError("submission_k must be positive.")
        events, grids = self._gather()
        if events is None:
            return {}
        predictions = _merge_parquet_events(
            events,
            event_ids=self._effective_event_ids(events),
        )
        submission: dict[str, list[dict[str, list[list[int]]]]] = {}
        for puzzle_name, pairs in self._ground_truth_pairs().items():
            attempts_for_task: list[dict[str, list[list[int]]]] = []
            for pair in pairs:
                ranked = _rank_votes(
                    [
                        (vote.prediction_hash, vote.confidence)
                        for vote in predictions.get((puzzle_name, pair.input_hash), [])
                    ],
                )
                attempts: list[list[list[int]]] = [
                    _parquet_grid_to_lists(grids[grid_hash])
                    for grid_hash, _ in ranked[:submission_k]
                    if grid_hash in grids
                ]
                while len(attempts) < submission_k:
                    attempts.append(attempts[0] if attempts else [[0]])
                attempts_for_task.append(
                    {
                        f"attempt_{index + 1}": grid
                        for index, grid in enumerate(attempts)
                    },
                )
            submission[puzzle_name] = attempts_for_task
        return submission

    def state_dict(self) -> StateDict:
        """Return the current ballots and rank-ordered checkpoint state.

        Returns:
          state: Local state and all gathered process-rank streams.

        """
        local_state = self._local_state_dict()
        if dist.is_initialized():
            gathered: list[object] = [object() for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_state)
            current = cast(list[_ParquetRankState], cast(object, gathered))
        else:
            current = [local_state]
        rank_states = _compose_parquet_rank_states(
            self._restored_rank_states,
            current=current,
            recent_window_size=(
                self.config.recent_window_size if self.config.aggregated_voting else 1
            ),
        )
        return {**local_state, "rank_states": rank_states}  # pyrefly: ignore[open-unpacking] -- The spread TypedDict is open; none of its extra keys collide here.

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore rank streams and prediction grids from a metric checkpoint.

        Args:
          state_dict: State returned by :meth:`state_dict`.

        """
        state = cast(ParquetPassK.StateDict, state_dict)
        rank_states = state.get("rank_states")
        if rank_states:
            self._restored_rank_states = cast(
                list[_ParquetRankState],
                cast(object, rank_states),
            )
            self._events = {}
            self._prediction_grids = {}
            self._event_counter = max(
                rank_state["event_counter"] for rank_state in rank_states
            )
            active = [
                rank_state["active_event_id"]
                for rank_state in rank_states
                if rank_state["active_event_id"] is not None
            ]
            self._active_event_id = max(active) if active else None
            return
        self._restored_rank_states = []
        self._event_counter = state["event_counter"]
        self._active_event_id = state["active_event_id"]
        self._events = {
            event_id: {
                key: [_ArcVote(**vote) for vote in votes]
                for key, votes in event.items()
            }
            for event_id, event in state["events"].items()
        }
        self._prediction_grids = _parquet_grids_from_state(state["prediction_grids"])

    def _ground_truth_pairs(self) -> dict[str, tuple[_ParquetGroundTruth, ...]]:
        if self._ground_truth is None:
            source = read_parquet_source(
                Path(self.config.eval_source),
                max_grid_size=self.config.canvas_size,
            )
            ground_truth: dict[str, tuple[_ParquetGroundTruth, ...]] = {}
            for row_index, puzzle_name in enumerate(source.puzzle_names):
                start = operator.index(
                    cast(SupportsIndex, source.row_test_starts[row_index]),
                )
                count = operator.index(
                    cast(SupportsIndex, source.row_test_counts[row_index]),
                )
                pairs: list[_ParquetGroundTruth] = []
                for pair_index in range(start, start + count):
                    input_grid = source.test_inputs.get(pair_index)
                    if input_grid is None:
                        raise ValueError("Expected input_grid is not None.")
                    output_grid = source.test_outputs.get(pair_index)
                    output_hash = None
                    if output_grid is not None and max(output_grid.shape) <= 255:
                        output_hash = grid_hash(output_grid)
                    pairs.append(
                        _ParquetGroundTruth(
                            input_hash=grid_hash(input_grid),
                            output_hash=output_hash,
                            has_output=output_grid is not None,
                        ),
                    )
                ground_truth[puzzle_name] = tuple(pairs)
            self._ground_truth = ground_truth
        return self._ground_truth

    def _effective_event_ids(
        self,
        events: Mapping[int, object] | None = None,
    ) -> list[int]:
        event_ids = list(self._events if events is None else events)
        if not self.config.aggregated_voting:
            return event_ids[-1:]
        window = self.config.recent_window_size
        return event_ids if window is None else event_ids[-window:]

    def _gather(
        self,
    ) -> tuple[
        dict[int, dict[tuple[str, str], list[_ArcVote]]] | None,
        dict[str, NDArray[np.uint8]],
    ]:
        local_state = self._local_state_dict()
        if dist.is_initialized():
            rank = dist.get_rank()
            gathered: list[object] | None = (
                [object() for _ in range(dist.get_world_size())] if rank == 0 else None
            )
            dist.gather_object(local_state, gathered, dst=0)
            if rank != 0:
                return None, {}
            if gathered is None:
                raise ValueError("Expected gathered rank states is not None.")
            rank_states = _compose_parquet_rank_states(
                self._restored_rank_states,
                current=cast(list[_ParquetRankState], cast(object, gathered)),
                recent_window_size=(
                    self.config.recent_window_size
                    if self.config.aggregated_voting
                    else 1
                ),
            )
        else:
            rank_states = _compose_parquet_rank_states(
                self._restored_rank_states,
                current=[local_state],
                recent_window_size=(
                    self.config.recent_window_size
                    if self.config.aggregated_voting
                    else 1
                ),
            )
        events: dict[int, dict[tuple[str, str], list[_ArcVote]]] = {}
        grids: dict[str, NDArray[np.uint8]] = {}
        for rank_state in rank_states:
            for event_id, event in rank_state["events"].items():
                destination = events.setdefault(event_id, {})
                for key, votes in event.items():
                    destination.setdefault(key, []).extend(
                        _parquet_votes_from_state(votes),
                    )
            grids.update(_parquet_grids_from_state(rank_state["prediction_grids"]))
        return events, grids

    def _trim_history(self) -> None:
        window = self.config.recent_window_size
        if not self.config.aggregated_voting or window is None:
            return
        retained = set(list(self._events)[-window:])
        self._events = {
            event_id: event
            for event_id, event in self._events.items()
            if event_id in retained
        }
        referenced = {
            vote.prediction_hash
            for event in self._events.values()
            for votes in event.values()
            for vote in votes
        }
        self._prediction_grids = {
            grid_hash: grid
            for grid_hash, grid in self._prediction_grids.items()
            if grid_hash in referenced
        }

    def _local_state_dict(self) -> _ParquetRankState:
        return {
            "event_counter": self._event_counter,
            "active_event_id": self._active_event_id,
            "events": {
                event_id: {
                    key: [
                        {
                            "prediction_hash": vote.prediction_hash,
                            "confidence": vote.confidence,
                        }
                        for vote in votes
                    ]
                    for key, votes in event.items()
                }
                for event_id, event in self._events.items()
            },
            "prediction_grids": {
                grid_hash: torch.from_numpy(grid.copy())
                for grid_hash, grid in self._prediction_grids.items()
            },
        }


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
                values.append((pred_h, from_plain(q, float)))
            by_input[input_h] = values
        preds[name] = by_input
    return preds


def _read_u32(payload: bytes, offset: int) -> tuple[int, int]:
    (value,) = struct.unpack_from("<I", payload, offset)
    return from_plain(value, int), offset + 4


def _rank_votes(votes: Sequence[tuple[str, float]]) -> list[tuple[str, float]]:
    """Rank grid ballots by count then confidence sum, preserving exact ties."""
    totals: dict[str, tuple[int, float]] = {}
    for prediction_hash, confidence in votes:
        count, confidence_sum = totals.get(prediction_hash, (0, 0.0))
        totals[prediction_hash] = (count + 1, confidence_sum + confidence)
    ranked = sorted(
        totals.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True,
    )
    return [
        (grid_hash, confidence_sum / count)
        for grid_hash, (count, confidence_sum) in ranked
    ]


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


def _act_step_rows(out: Tensor, k_steps: int) -> list[StepRow]:
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
    array = values.numpy().astype(np.uint8)
    return list(cast("Iterable[NDArray[np.uint8]]", array))


def _floats(values: Tensor) -> list[float]:
    return [float(value) for value in values]


def _ints(values: Tensor) -> list[int]:
    return [int(value) for value in values.to(torch.int64)]


def _json_grid(value: object, *, spec: ArcSpec) -> NDArray[np.uint8]:
    rows = [
        [_json_cell(cell) for cell in from_plain(row, list[object])]
        for row in from_plain(value, list[object])
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _ParquetGroundTruth:
    input_hash: str
    output_hash: str | None
    has_output: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class _ArcVote:
    prediction_hash: str
    confidence: float


class _ParquetVoteState(TypedDict, closed=True):
    prediction_hash: str
    confidence: float


class _ParquetRankState(TypedDict):
    event_counter: int
    active_event_id: int | None
    events: dict[int, dict[tuple[str, str], list[_ParquetVoteState]]]
    prediction_grids: dict[str, Tensor]


def _validate_parquet_metric_config(config: ParquetPassK.Config) -> None:
    """Reject invalid voting controls before the metric reads its source."""
    if config.canvas_size < 1:
        raise ValueError("canvas_size must be positive.")
    if config.canvas_size > 255:
        raise ValueError("canvas_size must be at most 255.")
    if not config.pass_ks or any(k < 1 for k in config.pass_ks):
        raise ValueError("pass_ks must contain positive attempt budgets.")
    if len(set(config.pass_ks)) != len(config.pass_ks):
        raise ValueError("pass_ks values must be unique.")
    if config.recent_window_size is not None and config.recent_window_size < 1:
        raise ValueError("recent_window_size must be positive or None.")


def _parquet_batch_names(batch: Mapping[str, object]) -> list[str]:
    """Narrow the canonical puzzle names from one collated evaluation batch."""
    value = batch.get("canonical_puzzle_names")
    if not isinstance(value, list):
        raise TypeError("ParquetPassK requires canonical_puzzle_names as list[str].")
    names: list[str] = []
    for name in cast(list[object], value):
        if not isinstance(name, str):
            raise TypeError(
                "ParquetPassK requires canonical_puzzle_names as list[str].",
            )
        names.append(name)
    return names


def _parquet_batch_inputs(batch: Mapping[str, object]) -> list[NDArray[np.uint8]]:
    """Narrow canonical ARC input arrays from one collated evaluation batch."""
    value = batch.get("canonical_inputs")
    if not isinstance(value, list):
        raise TypeError("ParquetPassK requires canonical_inputs as a list of grids.")
    inputs: list[NDArray[np.uint8]] = []
    for grid in cast(list[object], value):
        if not isinstance(grid, np.ndarray) or grid.ndim != 2:
            raise TypeError(
                "ParquetPassK canonical inputs must be two-dimensional arrays.",
            )
        if grid.dtype != np.uint8:
            raise TypeError("ParquetPassK canonical inputs must have uint8 dtype.")
        inputs.append(cast(NDArray[np.uint8], grid))
    return inputs


def _tensor_field(batch: Mapping[str, object], *, name: str) -> Tensor:
    """Narrow one required tensor field from the metric batch."""
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"ParquetPassK batch field {name!r} must be a Tensor.")
    return value


def _valid_count(value: object, *, batch_size: int) -> int:
    """Return the number of non-padding rows in a collated batch."""
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("valid_count must be a scalar or integer.")
        if value.is_quantized:
            raise TypeError("valid_count tensor must have integer dtype.")
        try:
            torch.iinfo(value.dtype)
        except TypeError:
            raise TypeError("valid_count tensor must have integer dtype.") from None
        count = value.item()
        assert isinstance(count, int)
    elif isinstance(value, int) and not isinstance(value, bool):
        count = value
    else:
        raise TypeError("ParquetPassK requires integer valid_count metadata.")
    if count < 0 or count > batch_size:
        raise ValueError(f"valid_count={count} is outside batch size {batch_size}.")
    return count


def _merge_parquet_events(
    events: Mapping[int, Mapping[tuple[str, str], Sequence[_ArcVote]]],
    *,
    event_ids: Sequence[int],
) -> dict[tuple[str, str], list[_ArcVote]]:
    """Merge ballots in source event, rank, then row insertion order."""
    merged: dict[tuple[str, str], list[_ArcVote]] = {}
    for event_id in event_ids:
        for key, votes in events[event_id].items():
            merged.setdefault(key, []).extend(votes)
    return merged


def _compose_parquet_rank_states(
    restored: Sequence[_ParquetRankState],
    *,
    current: Sequence[_ParquetRankState],
    recent_window_size: int | None,
) -> list[_ParquetRankState]:
    """Append current rows to restored rank streams and retain their event window."""
    if not restored:
        rank_states = list(current)
    elif len(restored) == len(current):
        rank_states = [
            _append_parquet_rank_state(previous, resumed=resumed)
            for previous, resumed in zip(restored, current, strict=True)
        ]
    else:
        rank_states = [*restored, *current]
    if recent_window_size is None:
        return rank_states
    event_ids = sorted(
        {event_id for rank_state in rank_states for event_id in rank_state["events"]},
    )
    retained = set(event_ids[-recent_window_size:])
    return [
        _trim_parquet_rank_state(rank_state, retained_event_ids=retained)
        for rank_state in rank_states
    ]


def _append_parquet_rank_state(
    previous: _ParquetRankState,
    *,
    resumed: _ParquetRankState,
) -> _ParquetRankState:
    """Append resumed event rows to one saved process-rank stream."""
    events = {
        event_id: {key: list(votes) for key, votes in event.items()}
        for event_id, event in previous["events"].items()
    }
    for event_id, event in resumed["events"].items():
        destination = events.setdefault(event_id, {})
        for key, votes in event.items():
            destination.setdefault(key, []).extend(votes)
    active_event_id = resumed["active_event_id"]
    if active_event_id is None:
        active_event_id = previous["active_event_id"]
    return {
        "event_counter": max(previous["event_counter"], resumed["event_counter"]),
        "active_event_id": active_event_id,
        "events": events,
        "prediction_grids": {
            **previous["prediction_grids"],
            **resumed["prediction_grids"],
        },
    }


def _trim_parquet_rank_state(
    rank_state: _ParquetRankState,
    *,
    retained_event_ids: set[int],
) -> _ParquetRankState:
    """Trim one rank's votes and candidate grids to retained events."""
    events = {
        event_id: event
        for event_id, event in rank_state["events"].items()
        if event_id in retained_event_ids
    }
    referenced = {
        vote["prediction_hash"]
        for event in events.values()
        for votes in event.values()
        for vote in votes
    }
    return {
        "event_counter": rank_state["event_counter"],
        "active_event_id": rank_state["active_event_id"],
        "events": events,
        "prediction_grids": {
            name: grid
            for name, grid in rank_state["prediction_grids"].items()
            if name in referenced
        },
    }


def _parquet_votes_from_state(
    votes: Sequence[_ParquetVoteState],
) -> list[_ArcVote]:
    """Restore source-order votes from one checkpointed rank stream."""
    return [
        _ArcVote(
            prediction_hash=vote["prediction_hash"],
            confidence=vote["confidence"],
        )
        for vote in votes
    ]


def _parquet_grids_from_state(
    grids: Mapping[str, Tensor],
) -> dict[str, NDArray[np.uint8]]:
    """Restore candidate tensors as owned CPU ARC arrays."""
    return {
        name: cast(
            NDArray[np.uint8],
            grid.detach().to(torch.uint8).cpu().numpy().copy(),
        )
        for name, grid in grids.items()
    }


def _parquet_grid_to_lists(grid: NDArray[np.uint8]) -> list[list[int]]:
    """Convert a candidate array to source-compatible submission lists."""
    return [
        [
            operator.index(cast(SupportsIndex, grid[row, col]))
            for col in range(operator.index(cast(SupportsIndex, grid.shape[1])))
        ]
        for row in range(operator.index(cast(SupportsIndex, grid.shape[0])))
    ]
