"""Run the complete public recipe, with preprocessing inside the compute budget."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

from configgle import Fig, Makeable

import numpy as np

from priml.baselines.name_that_llm.data import CompetitionData
from priml.baselines.name_that_llm.encoder import FrozenQwen
from priml.baselines.name_that_llm.metric import balanced_accuracy
from priml.baselines.name_that_llm.model import (
    Branch,
    CenteredHeads,
    LogBlend,
    QwenHeads,
    Split,
)
from priml.baselines.name_that_llm.topic import ValidationGroups
from priml.data.folds import grouped_folds
from priml.paths import resolve_working_dir, validated_output_path
from priml.train.budget import WallBudget, provenance


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyarrow import parquet

    import pyarrow as pa
    import torch
else:
    from wrapt import lazy_import

    parquet = lazy_import("pyarrow.parquet")
    pa = lazy_import("pyarrow")
    torch = lazy_import("torch")


class Benchmark:
    """One complete Name That LLM evaluation job for the priml launcher."""

    class Config(Fig["Benchmark"]):
        """Entire dataset, extraction, fitting, evaluation, and output tree."""

        dataset: CompetitionData.Config = field(default_factory=CompetitionData.Config)
        """Authentic competition rows and fixed sorted label vocabulary."""

        encoder: FrozenQwen.Config = field(default_factory=FrozenQwen.Config)
        """Frozen Qwen tokenization and hidden-state extraction."""

        branches: dict[str, Makeable[Branch]] = field(
            default_factory=dict[str, Makeable[Branch]],
        )
        """Ensemble members run in order; each sees the probabilities of earlier ones."""

        validation: ValidationGroups.Config = field(
            default_factory=ValidationGroups.Config,
        )
        """Validation groups simulating the hidden set; its topics cluster the test."""

        blend: LogBlend.Config = field(default_factory=LogBlend.Config)
        """Log-space blending of the branch probabilities."""

        outer_folds: int = 2
        """Fixed prompt-exclusive GroupKFold partitions."""

        budget: WallBudget.Config = field(default_factory=WallBudget.Config)
        """Shared cold wall accounting, checkpoints and durable result receipt."""

        base_dir: Path | str | None = "/opt/scratch"
        """Root that the inputs and the resolved working_dir sit beneath."""

        working_dir: Path | str = "/artifacts/{study_name}/{experiment_name}"
        """Fresh destination for results, probabilities, and the submission."""

        study_name: str = "name-that-llm"
        """Study identity written with every result."""

        experiment_name: str = ""
        """Experiment identity, filled from the launched factory when unset."""

        doc: str = ""
        """Experiment hypothesis stamped by the priml launcher."""

        @override
        def finalize(self) -> Self:
            directory = self.working_dir
            if isinstance(directory, str):
                directory = directory.format(
                    study_name=self.study_name,
                    experiment_name=self.experiment_name,
                )
            self.working_dir = resolve_working_dir(self.base_dir, directory)
            self.dataset.directory = resolve_working_dir(
                self.base_dir,
                self.dataset.directory,
            )
            self.encoder.directory = resolve_working_dir(
                self.base_dir,
                self.encoder.directory,
            )
            for branch in self.branches.values():
                heads = (
                    branch.heads if isinstance(branch, CenteredHeads.Config) else branch
                )
                if isinstance(heads, QwenHeads.Config) and not heads.layers:
                    heads.layers = list(range(len(self.encoder.layers)))
            if self.budget.receipt_path is None:
                self.budget.receipt_path = self.working_dir / "result.json"
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Run every branch and persist the result even when a phase fails.

        Args:
          *args: Additional flags are rejected; the config owns the run.

        Raises:
          ValueError: Unexpected arguments were supplied.
          TimeoutError: Candidate preprocessing, fitting, or scoring exceeds budget.

        """
        if args:
            raise ValueError("This benchmark accepts configuration fields only.")
        config = self.config
        budget = config.budget.make()
        directory = validated_output_path(config.working_dir)
        directory.mkdir(parents=True, exist_ok=False)
        with budget, budget.interruptible():
            result = budget.metadata
            result.update(
                {
                    "study": config.study_name,
                    "experiment": config.experiment_name,
                    "reference": "qasminumber/name-that-llm-v5/1",
                    "script_version_id": 358_565_580,
                    "config": config.serialize(),
                    **provenance(
                        packages=["numpy", "torch", "transformers", "pyarrow"],
                    ),
                    "timing_scope": (
                        "run entry through durable result receipt publication; "
                        "measure outer process time separately"
                    ),
                },
            )
            self._require_device()
            self._evaluate(directory, result=result, budget=budget)

    def _require_device(self) -> None:
        if self.config.encoder.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The public reference requires a CUDA accelerator.")

    def _evaluate(
        self,
        directory: Path,
        result: dict[str, object],
        budget: WallBudget,
    ) -> None:
        config = self.config
        data = config.dataset.make()
        result["input_sha256"] = data.sha256
        embeddings = config.encoder.make().encode(data.texts + data.test_texts)
        train_embeddings, test_embeddings = (
            embeddings[: len(data.ids)],
            embeddings[len(data.ids) :],
        )
        budget.check()
        texts = np.array(data.texts, dtype=object)
        test_texts = np.array(data.test_texts, dtype=object)
        validation_groups = config.validation.make()(texts)
        folds = grouped_folds(data.groups, count=config.outer_folds)
        classes = len(config.dataset.labels)
        oof = np.zeros((len(texts), classes))
        fold_ids = np.zeros(len(texts), dtype=np.int64)
        fold_scores: list[float] = []
        for index, (fit, valid) in enumerate(folds):
            scores = self._predict(
                Split(
                    texts=texts[fit],
                    targets=data.targets[fit],
                    groups=data.groups[fit],
                    embeddings=train_embeddings[fit],
                    predict_texts=texts[valid],
                    predict_embeddings=train_embeddings[valid],
                    clusters=validation_groups[valid],
                ),
            )
            oof[valid] = scores
            fold_ids[valid] = index
            fold_scores.append(
                balanced_accuracy(
                    data.targets[valid],
                    predictions=scores.argmax(1),
                    classes=classes,
                ),
            )
            budget.check()
        result["balanced_accuracy"] = balanced_accuracy(
            data.targets,
            predictions=oof.argmax(1),
            classes=classes,
        )
        result["fold_balanced_accuracy"] = fold_scores
        result["validation_rows"] = len(texts)
        result["validation_prompts"] = len(np.unique(data.groups))
        result["validation_protocol"] = (
            "GroupKFold; text-only topic clusters split into seeded small chunks"
        )
        np.savez(
            directory / "validation.npz",
            ids=np.array(data.ids),
            targets=data.targets,
            fold=fold_ids,
            log_scores=oof,
        )
        scores = self._predict(
            Split(
                texts=texts,
                targets=data.targets,
                groups=data.groups,
                embeddings=train_embeddings,
                predict_texts=test_texts,
                predict_embeddings=test_embeddings,
                clusters=config.validation.topics.make()(test_texts),
            ),
        )
        labels = config.dataset.labels
        predictions = scores.argmax(1)
        table = pa.table(
            {
                "id": data.test_ids,
                "model": [
                    labels[index] for index in cast("list[int]", predictions.tolist())
                ],
            },
        )
        parquet.write_table(table, where=directory / "submission.parquet")
        np.savez(
            directory / "predictions.npz",
            ids=np.array(data.test_ids),
            log_scores=scores,
        )
        budget.check()
        result["public_best_parity"] = (
            "tiny CPU goldens minted in-tree; see README.md for scope"
        )
        result["submission_rows"] = len(predictions)

    def _predict(self, split: Split) -> NDArray[np.float64]:
        probabilities: dict[str, NDArray[np.float64]] = {}
        for name, branch in self.config.branches.items():
            probabilities[name] = branch.make()(split, probabilities)
        return self.config.blend.make()(probabilities)
