"""Fit a ConvexTok vocabulary on raw shards and write it as the baseline's tokenizer.

The stages run as upstream runs them: pretokenize, count candidates, build the LP,
presolve it as PSLP does, solve it (PDLP by default; the ``solver`` slot takes any
solver), postsolve, round, and export. With
``reserved_count = 10`` the learned-piece budget ``vocab_size - reserved_count - 256``
is upstream's; its ten special tokens become the baseline's reserved IDs.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import time

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.baselines.convextok.candidates import count_candidates
from priml.baselines.convextok.export import export_tokenizer
from priml.baselines.convextok.pdlp import Pdlp
from priml.baselines.convextok.presolver.presolve import (
    Presolved,
    presolve,
)
from priml.baselines.convextok.pretokens import count_pretokens
from priml.baselines.convextok.program import (
    LinearProgram,
    LpSolution,
    TokenizationProgram,
    build_program,
)
from priml.baselines.convextok.rounding import (
    RoundingFn,
    deterministic_rounding,
)
from priml.baselines.nanochat.scripts.prepare_data import (
    Preparation,
    donor_unigram16k,
)
from priml.baselines.nanochat.scripts.prepare_tokenizer import document_rows
from priml.paths import validated_output_path


@dataclass(frozen=True, slots=True, kw_only=True)
class Fit:
    """One fit: its program, the solver's answer, the chosen pieces, and stage times.

    Attributes:
      program: The program as built, before presolve.
      solved: The program the solver was given: presolved, or as built.
      solution: The solver's answer to ``solved``.
      primal: That answer on ``program``'s columns, on the CPU.
      learned: The chosen pieces, in the order rounding chose them.
      seconds: Wall-clock seconds per stage, in the order the stages ran.

    """

    program: TokenizationProgram
    solved: LinearProgram
    solution: LpSolution
    primal: Tensor
    learned: list[str]
    seconds: dict[str, float]


class ConvexTokPreparation:
    """Solve ConvexTok's LP over the fitting shards and save the resulting tokenizer."""

    class Config(Fig["ConvexTokPreparation"]):
        """Every fitting choice, visible in the preparation recipe."""

        raw_dir: Path = Path("/opt/scratch/datasets/nanochat/raw")
        """Original ClimbMix Parquet shards."""

        shard_indices: list[int] = field(default_factory=lambda: list(range(7)))
        """Shards whose documents are fitted, in order; upstream fits the first seven."""

        working_dir: Path = Path("/opt/scratch/datasets/nanochat/convextok16k")
        """Destination of the tokenizer."""

        vocab_size: int = 16_384
        """Model vocabulary including reserved IDs."""

        reserved_count: int = 10
        """IDs appended after ordinary pieces; ten matches upstream's special tokens."""

        split_pattern: str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
        """Pretokenization expression, upstream's nanochat pattern."""

        num_workers: int = 16
        """Processes for pretokenizing and counting candidates."""

        presolve: Callable[[LinearProgram, torch.device], Presolved] | None = presolve
        """Presolve before solving, as cuOpt does by default; None solves the LP as built."""

        solver: Makeable[Callable[[LinearProgram], LpSolution]] = field(
            default_factory=Pdlp.Config,
        )
        """The LP solver: any program-to-solution callable; cuOpt's PDLP by default."""

        rounding: RoundingFn = deterministic_rounding
        """Chooses the learned pieces from the solution."""

        device: str = "cuda"
        """Device the solver runs on."""

    def __init__(self, config: Config) -> None:
        self.config = config
        if config.vocab_size - config.reserved_count < 256:
            raise ValueError("The ordinary vocabulary must contain every byte.")

    def build(self) -> None:
        """Fit the vocabulary from the shards and save ``tokenizer.json``."""
        config = self.config
        output = validated_output_path(config.working_dir, protected=[config.raw_dir])
        output.mkdir(parents=True)
        export_tokenizer(
            self.fit().learned,
            split_pattern=config.split_pattern,
        ).save(str(output / "tokenizer.json"))

    def fit(self) -> Fit:
        """Read the fitting shards, solve the program, and choose the learned pieces.

        Returns:
          fit: The program, the solution, the pieces, and each stage's seconds.

        """
        config = self.config
        laps = _Laps()
        texts = [
            text
            for shard in config.shard_indices
            for _, text in document_rows(
                config.raw_dir / f"shard_{shard:05d}.parquet",
                shard=shard,
            )
        ]
        laps("read")
        pretokens = count_pretokens(
            texts,
            split_pattern=config.split_pattern,
            num_workers=config.num_workers,
        )
        laps("pretokens")
        candidates = list(count_candidates(pretokens, num_workers=config.num_workers))
        laps("candidates")
        budget = config.vocab_size - config.reserved_count - 256
        built = build_program(pretokens, candidates, budget=budget)
        laps("program")
        device = torch.device(config.device)
        solved, presolved = built.program, None
        if config.presolve is not None:
            presolved = config.presolve(built.program, device)
            solved = presolved.program
            laps("presolve")
        solution = config.solver.make()(solved.to(device))
        # Reading the answer back waits for the device, so the solve's lap is whole.
        primal = solution.primal.cpu()
        laps("solve")
        if presolved is not None:
            primal = presolved.postsolve(primal)
            laps("postsolve")
        chosen = config.rounding(
            primal[built.num_token_edges + built.num_byte_edges :].to(torch.float64),
            candidates,
            budget=budget,
        )
        learned = [candidates[int(position)] for position in chosen]
        laps("rounding")
        return Fit(
            program=built,
            solved=solved,
            solution=solution,
            primal=primal,
            learned=learned,
            seconds=laps.seconds,
        )


def donor_convextok16k() -> Preparation.Config:
    """Fork the donor Unigram recipe with ConvexTok in its tokenizer slot.

    Only the vocabulary changes: the corpus, the donor moves, the packed rows and
    the reference evaluation are the Unigram recipe's. ConvexTok fits upstream's
    first seven raw shards rather than the Unigram sample, and reserves ten IDs.

    Returns:
      config: Source selection, ConvexTok fitting, and packed-row geometry.

    """
    config = donor_unigram16k()
    config.tokenizer = ConvexTokPreparation.Config()
    config.tokenizer_name = "convextok16k"
    return config


@dataclass(slots=True, kw_only=True)
class _Laps:
    """Wall-clock seconds per stage, each timed from the end of the one before."""

    seconds: dict[str, float] = field(default_factory=dict)
    last: float = field(default_factory=time.perf_counter)

    def __call__(self, stage: str) -> None:
        now = time.perf_counter()
        self.seconds[stage] = now - self.last
        self.last = now
