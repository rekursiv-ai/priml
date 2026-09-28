"""pass@K voting over a puzzle's augmented views.

Each ARC puzzle is evaluated many times -- once per augmented view -- and the
model may answer differently on each. The score is the consensus: group every
prediction for one puzzle, rank the distinct answers, and count the puzzle
solved if the true grid is among the top K.

Ranking is by vote count, with mean halt confidence breaking ties. Count
dominates because agreement across independent views is the stronger signal;
confidence only separates answers that tied.

Predictions are stored as hashes, not grids. A full evaluation is hundreds of
thousands of 900-cell grids, and only equality between them matters.

:class:`PassK` votes per prepared puzzle id against its label.
:class:`CanonicalPassK` is the TRM reference evaluator: it inverts each view to
the canonical frame, votes per canonical test input, and scores against
``test_puzzles.json``, averaging within a task and then across tasks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    NotRequired,
    Self,
    TypedDict,
    cast,
    override,
)

import hashlib
import logging
import struct

from configgle import Fig, Makeable
from torch import Tensor

import numpy as np
import torch
import torch.distributed as dist

from priml.baselines.arcagi1.augmentation import (
    ARC,
    ColorDihedral,
    arc_grid_to_np,
    crop_grid,
    grid_hash,
    untranslate_unscale,
)
from priml.lib.custom_json import DictCodec, FloatCodec, IntCodec, ListCodec, loads
from priml.paths import resolve_working_dir
from priml.runtime import is_rank_zero


if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)

type _Preds = dict[str, dict[str, list[tuple[str, float]]]]
type _SignalRow = tuple[str, str, str, float, float, float, int, int]
type _StepRow = tuple[int, int, tuple[float, ...], tuple[int, ...]]


class PassK:
    """Consensus accuracy over each puzzle's augmented views.

    Consumes the packed evaluation output the puzzle train step emits: a halt
    logit in column 0 and the predicted tokens in the last ``grid_len``
    columns, so any diagnostic columns between them are ignored.
    """

    class Config(Fig["PassK"]):
        """Which K values to report, and how a vote is counted."""

        pass_ks: tuple[int, ...] = (1, 2, 5, 10)
        """Report the true grid appearing in the top K ranked answers.

        pass@1 is the headline -- the model's single best guess. Larger K
        measures whether the right answer was present but outvoted, which
        separates a model that cannot solve a task from one that cannot pick
        its own best attempt."""

        ignore_label_id: int = -100
        """Label value marking cells excluded from the comparison.

        Padding rows appended to square off a short batch carry it, so they
        neither count as solved nor as failed."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Drop every accumulated vote."""
        # Puzzle id -> answer hash -> [votes, summed confidence].
        self._votes: dict[int, dict[str, list[float]]] = {}
        # Puzzle id -> the true answer's hash.
        self._truth: dict[int, str] = {}

    def update(self, logits: Tensor, **batch: object) -> None:
        """Record one batch of predictions as votes.

        Args:
          logits: Packed model output; column 0 is the halt logit and the last
            ``grid_len`` columns are the predicted tokens.
          **batch: Must carry ``label`` and ``puzzle_identifiers``;
            ``valid_count`` truncates the padded tail when present.

        """
        label_raw = batch["label"]
        assert isinstance(label_raw, Tensor)
        labels = label_raw.detach().to(torch.int64)
        puzzle_identifiers_raw = batch["puzzle_identifiers"]
        assert isinstance(puzzle_identifiers_raw, Tensor)
        identifiers = puzzle_identifiers_raw.detach().to(torch.int64)
        grid_len = labels.shape[-1]
        packed = logits.detach()
        predictions = packed[:, -grid_len:].to(torch.int64)
        # Confidence in [0, 1] so ties break on a comparable scale.
        confidence = torch.sigmoid(packed[:, 0].float())

        raw_count = batch.get("valid_count", labels.shape[0])
        assert isinstance(raw_count, int)
        valid_count = raw_count
        labels = labels[:valid_count].to(predictions.device)
        predictions = predictions[:valid_count]
        identifiers = identifiers[:valid_count].to(predictions.device)
        confidence = confidence[:valid_count]

        counted = labels != self.config.ignore_label_id
        for row in range(predictions.shape[0]):
            keep = counted[row]
            if not bool(keep.any()):
                continue  # An all-ignored row is padding, not a puzzle.
            puzzle = int(identifiers[row])
            answer = _digest(predictions[row][keep])
            truth = _digest(labels[row][keep])
            self._truth.setdefault(puzzle, truth)
            tally = self._votes.setdefault(puzzle, {}).setdefault(answer, [0.0, 0.0])
            tally[0] += 1.0
            tally[1] += float(confidence[row])

    def compute(self) -> dict[str, float]:
        """Rank each puzzle's answers and score every K.

        Returns:
          metrics: Accuracy at each pass@K threshold, keyed as "pass@{k}".

        """
        solved = dict.fromkeys(self.config.pass_ks, 0)
        for puzzle, tally in self._votes.items():
            truth = self._truth[puzzle]
            # Count first, then mean confidence: agreement across independent
            # views outranks a single confident view.
            ranked = sorted(
                tally.items(),
                key=lambda item: (item[1][0], item[1][1] / item[1][0]),
                reverse=True,
            )
            for k in self.config.pass_ks:
                if any(answer == truth for answer, _ in ranked[:k]):
                    solved[k] += 1
        counts = torch.tensor(
            [float(len(self._votes)), *(float(solved[k]) for k in self.config.pass_ks)],
            dtype=torch.float64,
        )
        if dist.is_available() and dist.is_initialized():
            # NCCL reduces only CUDA tensors; gloo only CPU ones. Move for the
            # former and come back, so ``.tolist()`` works either way.
            if dist.get_backend() != "gloo":
                counts = counts.to(torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            counts = counts.cpu()
        total, *hits = [FloatCodec.coerce(count) for count in counts.tolist()]
        return {
            f"pass@{k}": hit / max(1.0, total)
            for k, hit in zip(self.config.pass_ks, hits, strict=True)
        }

    class StateDict(TypedDict):
        """Per-puzzle vote tallies and the digest of each puzzle's answer."""

        votes: dict[int, dict[str, list[float]]]
        truth: dict[int, str]

    def state_dict(self) -> StateDict:
        """Return the accumulated votes.

        Returns:
          state: The vote tallies and answer digests, keyed by puzzle.

        """
        return {"votes": self._votes, "truth": self._truth}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore votes produced by :meth:`state_dict`.

        Args:
          state_dict: State dict.

        """
        state = cast(PassK.StateDict, state_dict)
        self._votes = state.get("votes", {})
        self._truth = state.get("truth", {})


class SignalDumpPayload(NamedTuple):
    """Per-rank signal-dump data :meth:`CanonicalPassK.compute` emits in ``extras``."""

    rows: list[_SignalRow]
    grids: dict[str, NDArray[np.uint8]]
    steps: list[_StepRow]
    pass_ks: tuple[int, ...]


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

        transform: Makeable[ColorDihedral] = field(default_factory=ColorDihedral.Config)
        """Color/dihedral policy that encoded the prepared identifiers."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self._transform = config.transform.make()
        root = Path(config.working_dir).expanduser()
        self._identifier_map = ListCodec.coerce(
            loads((root / "identifiers.json").read_text()),
            str,
        )
        self._test_puzzles = {
            name: DictCodec.coerce(puzzle)
            for name, puzzle in DictCodec.coerce(
                loads((root / "test_puzzles.json").read_text()),
            ).items()
        }
        # The same blank id the loader pads with, so the two never desync.
        meta_path = root / "test" / "dataset.json"
        meta = (
            DictCodec.coerce(loads(meta_path.read_text()))
            if meta_path.is_file()
            else {}
        )
        self._blank_identifier_id = IntCodec.coerce(meta.get("blank_identifier_id", 0))
        self.reset()

    def reset(self) -> None:
        """Drop every accumulated ballot and dump row."""
        self._hmap: dict[str, tuple[int, int]] = {}
        self._preds: _Preds = {}
        self._dump_rows: list[_SignalRow] = []
        self._dump_grids: dict[str, NDArray[np.uint8]] = {}
        self._dump_steps: list[_StepRow] = []
        self._dump_enabled = False

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
        dumping = n_header >= 3 or k_steps > 0
        self._dump_enabled = self._dump_enabled or dumping
        # float64 before sigmoid, so confident logits do not saturate to 0 or 1.
        q_halt = _floats(out[:, 0].to(torch.float64).sigmoid())
        preds_t = _uint8_rows(out[:, n_header:].to(torch.int64))
        raw_q_halt = _floats(out[:, 0].to(torch.float32))
        if n_header >= 3:
            raw_logprob = _floats(out[:, 1].to(torch.float32))
            raw_stability = _floats(out[:, 2].to(torch.float32))
        else:
            raw_logprob = raw_stability = [float("nan")] * out.shape[0]
        steps: list[_StepRow] = []
        if k_steps > 0:
            converge = ListCodec.coerce(out[:, 3].to(torch.int64).tolist(), int)
            nchg = ListCodec.coerce(out[:, 4].to(torch.int64).tolist(), int)
            qhalt_steps = out[:, 5 : 5 + k_steps].to(torch.float32)
            correct_steps = out[:, 5 + k_steps : 5 + 2 * k_steps].to(torch.int64)
            steps = [
                (
                    converge[i],
                    nchg[i],
                    tuple(_floats(qhalt_steps[i])),
                    tuple(ListCodec.coerce(correct_steps[i].tolist(), int)),
                )
                for i in range(out.shape[0])
            ]
        puzzle_ids_t = batch["puzzle_identifiers"]
        assert isinstance(puzzle_ids_t, Tensor)
        puzzle_ids = ListCodec.coerce(
            puzzle_ids_t.detach().cpu().to(torch.int64).tolist(),
            int,
        )
        spatial_tags_t = batch.get("spatial_tags")
        if spatial_tags_t is None:
            spatial_tags = [[1, 0, 0]] * len(puzzle_ids)
        else:
            assert isinstance(spatial_tags_t, Tensor)
            spatial_tags = [
                ListCodec.coerce(row, int)
                for row in ListCodec.coerce(
                    spatial_tags_t.detach().cpu().to(torch.int64).tolist(),
                )
            ]
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
                ),
            )
            pred_h = grid_hash(pred_canon)
            n_rows, n_cols = _shape(pred_canon)
            self._hmap[pred_h] = (n_rows, n_cols)
            self._preds.setdefault(orig_name, {}).setdefault(input_h, []).append(
                (pred_h, q_halt[i]),
            )
            if dumping:
                self._dump_rows.append(
                    (
                        orig_name,
                        input_h,
                        pred_h,
                        raw_q_halt[i],
                        raw_logprob[i],
                        raw_stability[i],
                        n_rows,
                        n_cols,
                    ),
                )
                self._dump_grids.setdefault(pred_h, pred_canon.astype(np.uint8))
                if k_steps > 0:
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
            if self._dump_enabled:
                empty["extras"] = {"signal_dump": self._signal_dump_payload()}
            return empty
        per_task_pass1: list[tuple[str, float]] = []
        pass1_idx = pass_ks.index(1) if 1 in pass_ks else None
        n_no_preds = 0
        for name, puzzle in self._test_puzzles.items():
            pairs = ListCodec.mappings(puzzle.get("test"))
            per_test_correct = [0 for _ in pass_ks]
            per_test_report = {
                rank_name: [0 for _ in pass_ks]
                for rank_name in _REPORT_ONLY_RANK_SCORERS
            }
            task_had_preds = False
            for pair in pairs:
                input_h = grid_hash(_json_grid(pair["input"]))
                label_h = grid_hash(_json_grid(pair["output"]))
                records = preds.get(name, {}).get(input_h, [])
                cap = self.config.max_views_per_input
                if cap > 0 and len(records) > cap:
                    # The prediction hash only breaks ties, keeping the slice deterministic.
                    records = sorted(records, key=lambda r: (-r[1], r[0]))[:cap]
                p_map: dict[str, list[float]] = {}
                max_q_of: dict[str, float] = {}
                for h, q in records:
                    stat = p_map.setdefault(h, [0.0, 0.0])
                    stat[0] += 1.0
                    stat[1] += q
                    max_q_of[h] = max(max_q_of.get(h, 0.0), q)
                if not p_map:
                    continue
                task_had_preds = True
                for stat in p_map.values():
                    stat[1] /= stat[0]
                ranked = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)
                for i, k in enumerate(pass_ks):
                    if any(h == label_h for h, _ in ranked[:k]):
                        per_test_correct[i] += 1
                for rank_name, scorer in _REPORT_ONLY_RANK_SCORERS.items():
                    # The hash breaks exact ties of the collapsed score.
                    report_ranked = sorted(
                        p_map,
                        key=lambda h, s=scorer: (
                            -s(p_map[h][0], p_map[h][1], max_q_of[h]),
                            h,
                        ),
                    )
                    for i, k in enumerate(pass_ks):
                        if label_h in report_ranked[:k]:
                            per_test_report[rank_name][i] += 1
            n_test = len(pairs)
            if n_test == 0:
                continue
            if not task_had_preds:
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
        self._log_per_task(results, per_task_pass1, n_test_puzzles, n_no_preds)
        if self._dump_enabled:
            results["extras"] = {"signal_dump": self._signal_dump_payload()}
        return results

    class StateDict(TypedDict):
        """Checkpointed ballots; ``preds`` records are ``[hash, q]`` lists."""

        hmap: NotRequired[dict[str, tuple[int, int]]]
        preds: NotRequired[dict[str, dict[str, list[list[str | float]]]]]

    def state_dict(self) -> StateDict:
        """Return the accumulated ballots in a JSON-friendly shape.

        Returns:
          state: ``hmap`` and ``preds`` with tuples as lists.

        """
        return {
            "hmap": dict(self._hmap),
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
        self._hmap = DictCodec.coerce(state.get("hmap", {}), tuple)
        self._preds = {
            name: {
                ih: [(str(h), FloatCodec.coerce(q)) for h, q in vs]
                for ih, vs in by_input.items()
            }
            for name, by_input in state.get("preds", {}).items()
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
        size = torch.tensor([len(payload)], dtype=torch.int64, device=device)
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
        for tensor, size_t in zip(recv, sizes, strict=True):
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

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
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
                extras_map = DictCodec.coerce(extras, default=None)
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
        )

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
) -> None:
    """Gather every rank's dump and write it as one ``.npz`` from rank 0.

    Args:
      payload: This rank's rows, unique prediction grids, and step rows.
      dump_signals_path: A ``Path`` is literal; a ``str`` formats ``{global_step}``.
      global_step: Step substituted into a ``str`` path.

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
    pred_grids = np.zeros((n_pred, ARC.max_grid, ARC.max_grid), dtype=np.uint8)
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
        "pass_ks": np.array(list(payload.pass_ks), dtype=np.int64),
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
        "[eval] wrote signal dump (%d rows, %d groups, %d preds, %.1f MB) to %s",
        n,
        len(group_table),
        n_pred,
        path.stat().st_size / 1e6,
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
                values.append((pred_h, FloatCodec.coerce(q)))
            by_input[input_h] = values
        preds[name] = by_input
    return preds


def _read_u32(payload: bytes, offset: int) -> tuple[int, int]:
    (value,) = struct.unpack_from("<I", payload, offset)
    return IntCodec.coerce(value), offset + 4


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
        pred_len = out_width - 1 if n_header < 1 else out_width - n_header
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
    return ListCodec.coerce(values.tolist(), float)


def _json_grid(value: object) -> NDArray[np.uint8]:
    rows = [ListCodec.coerce(row, int) for row in ListCodec.coerce(value)]
    return arc_grid_to_np(rows, max_grid=ARC.max_grid)


def _shape(grid: NDArray[np.uint8]) -> tuple[int, int]:
    rows, cols = ListCodec.coerce(list(grid.shape), int)
    return rows, cols


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


# Only equality between grids matters, and an evaluation holds hundreds of thousands of
# them, so a digest is stored instead of the grid.
def _digest(grid: Tensor) -> str:
    """Hash one grid's tokens."""
    return hashlib.blake2b(
        grid.to(torch.int16).cpu().numpy().tobytes(),
        digest_size=16,
    ).hexdigest()
