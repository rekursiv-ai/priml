"""Fit a ConvexTok vocabulary once and record what the fit cost and what it chose.

The run directory receives the fitted ``tokenizer.json`` and a ``metrics.json``:

- ``seconds``: wall-clock seconds per stage, and ``total``.
- ``peak_gpu_bytes``: most memory torch held on the solver's CUDA device during the
  fit; null on any other device.
- ``peak_host_bytes``: this process's peak resident memory, excluding the pretoken
  and candidate worker processes.
- ``program`` and ``solved``: rows, columns, and nonzeros of the program as built
  and as given to the solver.
- ``objective``: the program's objective at the solution -- the relaxation's token
  count, weighted by pretoken frequency.
- ``infeasibility``: 2-norm of the solution's violation of the row and variable
  bounds; solvers compare fairly only at matched infeasibility.
- ``iterations`` and ``optimal``: present when the solver reports them.
- ``pieces``: pieces in the fitted tokenizer, its 256 bytes included. With a
  ``reference``, ``missing`` and ``extra`` count the reference's pieces the fit
  lacks and the fit's pieces the reference lacks.
"""

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, override, runtime_checkable

import json
import resource
import sys
import time

from configgle import Fig
from tokenizers import Tokenizer

from priml.baselines.convextok.export import export_tokenizer
from priml.baselines.convextok.prepare import ConvexTokPreparation, Fit
from priml.baselines.convextok.program import LinearProgram
from priml.baselines.convextok.spmv import CsrMatrix
from priml.paths import resolve_working_dir, validated_output_path


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


MAXRSS_UNIT_BYTES: Final = 1 if sys.platform == "darwin" else 1024
"""Bytes per unit of ``ru_maxrss``: Linux reports kibibytes, macOS bytes."""


@runtime_checkable
class IterativeSolution(Protocol):
    """A solution that reports how its solver stopped, as ``PdlpResult`` does."""

    @property
    def iterations(self) -> int:
        """Iterations the solver took."""
        ...

    @property
    def optimal(self) -> bool:
        """Whether the solver met its tolerances."""
        ...


class MeasuredFit:
    """Fit a vocabulary, save it, and write what the fit cost and chose."""

    class Config(Fig["MeasuredFit"]):
        """The fit, where its outputs go, and what its pieces are compared with."""

        study_name: str = ""
        """Run-family name; launch derives it when empty."""

        experiment_name: str = ""
        """Run name within the study; launch derives it when empty."""

        doc: str = ""
        """Experiment description; launch stamps the factory's docstring here."""

        base_dir: Path | str | None = "/opt/scratch"
        """Root the run directory resolves beneath."""

        working_dir: Path | str = "/runs/{study_name}/{experiment_name}"
        """Run directory; the preparation's ``working_dir`` is set to it."""

        preparation: ConvexTokPreparation.Config = field(
            default_factory=ConvexTokPreparation.Config,
        )
        """The fit: shards, vocabulary size, presolve, solver, and rounding."""

        reference: Path | None = None
        """Tokenizer whose pieces the fit's are compared with; None skips it."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.working_dir, str):
                self.working_dir = self.working_dir.format(
                    study_name=self.study_name,
                    experiment_name=self.experiment_name,
                )
            self.working_dir = resolve_working_dir(
                self.base_dir,
                working_dir=self.working_dir,
            )
            self.preparation.working_dir = self.working_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.preparation = config.preparation.make()

    def run(self, *args: str) -> None:
        """Fit, then write ``tokenizer.json`` and ``metrics.json`` to the run directory.

        Args:
          *args: Unused; the launcher forwards flags it does not own.

        """
        del args
        config = self.config
        preparation = config.preparation
        protected = [preparation.raw_dir]
        if config.reference is not None:
            protected.append(config.reference)
        output = validated_output_path(preparation.working_dir, protected=protected)
        output.mkdir(parents=True)
        device = torch.device(preparation.device)
        # pragma: no mutate start -- CUDA-only: CPU tests cannot reach it; the
        # cuda-marked test does.
        on_cuda = device.type == "cuda"
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        # pragma: no mutate end
        fit = self.preparation.fit()
        # pragma: no mutate start -- CUDA-only, as above.
        peak_gpu_bytes = torch.cuda.max_memory_allocated(device) if on_cuda else None
        # pragma: no mutate end
        # Read before the metrics below allocate, so the peak is the fit's own.
        peak_host_bytes = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * MAXRSS_UNIT_BYTES
        )
        start = time.perf_counter()
        tokenizer = export_tokenizer(
            fit.learned,
            split_pattern=preparation.split_pattern,
        )
        tokenizer.save(str(output / "tokenizer.json"))
        seconds = {**fit.seconds, "export": time.perf_counter() - start}
        metrics: dict[str, object] = {
            "seconds": {**seconds, "total": sum(seconds.values())},
            "peak_gpu_bytes": peak_gpu_bytes,
            "peak_host_bytes": peak_host_bytes,
            **_solution_metrics(fit),
            **_vocabulary_metrics(tokenizer, reference=config.reference),
        }
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
        )


def _solution_metrics(fit: Fit) -> dict[str, object]:
    """Measure the programs' sizes and the solution against the program as built."""
    program = fit.program.program
    primal = fit.primal.to(program.objective.dtype)
    matrix = CsrMatrix(program.crow_indices, program.col_indices, program.values)
    activity = matrix @ primal
    violation = torch.cat(
        [
            activity - activity.clamp(min=program.row_lower, max=program.row_upper),
            primal - primal.clamp(min=program.lower, max=program.upper),
        ],
    )
    metrics: dict[str, object] = {
        "program": _size(program),
        "solved": _size(fit.solved),
        "objective": float(torch.dot(program.objective, primal)),
        "infeasibility": float(torch.linalg.vector_norm(violation)),
    }
    if isinstance(fit.solution, IterativeSolution):
        metrics["iterations"] = fit.solution.iterations
        metrics["optimal"] = fit.solution.optimal
    return metrics


def _size(program: LinearProgram) -> dict[str, int]:
    return {
        "rows": program.num_rows,
        "columns": program.num_columns,
        "nonzeros": len(program.values),
    }


def _vocabulary_metrics(
    tokenizer: Tokenizer,
    *,
    reference: Path | None,
) -> dict[str, int]:
    """Count the fitted pieces and, given a reference, how the vocabularies differ."""
    fitted = set(tokenizer.get_vocab())
    if reference is None:
        return {"pieces": len(fitted)}
    expected = set(Tokenizer.from_file(str(reference)).get_vocab())
    return {
        "pieces": len(fitted),
        "missing": len(expected - fitted),
        "extra": len(fitted - expected),
    }
