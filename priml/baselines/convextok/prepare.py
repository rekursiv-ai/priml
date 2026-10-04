"""Fit a ConvexTok vocabulary on raw shards and write it as the baseline's tokenizer.

The stages run as upstream runs them: pretokenize, count candidates, build the LP,
presolve it as PSLP does, solve it (PDLP by default; the ``solver`` slot takes any
solver), postsolve, round, and export. With
``reserved_count = 10`` the learned-piece budget ``vocab_size - reserved_count - 256``
is upstream's; its ten special tokens become the baseline's reserved IDs.
"""

from collections.abc import Callable
from dataclasses import field
from pathlib import Path

from configgle import Fig, Makeable

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
        texts = [
            text
            for shard in config.shard_indices
            for _, text in document_rows(
                config.raw_dir / f"shard_{shard:05d}.parquet",
                shard=shard,
            )
        ]
        pretokens = count_pretokens(
            texts,
            split_pattern=config.split_pattern,
            num_workers=config.num_workers,
        )
        candidates = list(count_candidates(pretokens, num_workers=config.num_workers))
        budget = config.vocab_size - config.reserved_count - 256
        built = build_program(pretokens, candidates, budget=budget)
        solver = config.solver.make()
        if config.presolve is None:
            primal = solver(built.program.to(config.device)).primal.cpu()
        else:
            presolved = config.presolve(built.program, torch.device(config.device))
            result = solver(presolved.program.to(config.device))
            primal = presolved.postsolve(result.primal)
        chosen = config.rounding(
            primal[built.num_token_edges + built.num_byte_edges :].to(torch.float64),
            candidates,
            budget=budget,
        )
        export_tokenizer(
            [candidates[int(position)] for position in chosen],
            split_pattern=config.split_pattern,
        ).save(str(output / "tokenizer.json"))


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
