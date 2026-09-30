"""ARC-AGI-2 puzzle dataset: the canonical loader staged from the arc2 build.

Thin subclass of :class:`priml.baselines.arcagi1.data.PuzzleData` -- the
on-disk schema, batching, and iterators are identical (ARC-AGI-2 keeps 30x30
grids, 10 colors, vocab 12, seq_len 900). Only the staging differs: creating
the dataset builds the ``arc2concept-aug-1000`` tree (subsets ``training2`` +
``evaluation2`` + ``concept``, test split ``evaluation2``) instead of the
ARC-AGI-1 tree, and can additionally stage the S=N spatial-eval expansion of
an aug-policy variant (the exp005 test-time-augmentation recipe).
"""

from __future__ import annotations

from pathlib import Path
from typing import Self, override

from configgle import Makes

from priml.baselines.arcagi1.data import PuzzleData
from priml.baselines.arcagi1.scripts.build_spatial_eval import (
    ensure_spatial_eval_data,
)
from priml.baselines.arcagi2.scripts.build_dataset import ensure_arc2_dataset
from priml.paths import resolve_working_dir


class Arc2PuzzleDataset(PuzzleData):
    """ARC-AGI-2 puzzle dataset (arc2concept staging)."""

    class Config(Makes["Arc2PuzzleDataset"], PuzzleData.Config):
        """Configure ARC-AGI-2 dataset loading; surface matches the parent."""

        working_dir: Path | str = "/datasets/arc2concept-aug-1000"
        """Root dataset directory containing train/ and test/ splits.

        The default names the plain ARC-AGI-2 build; aug-policy variants use
        :func:`priml.baselines.arcagi2.scripts.build_dataset.arc2_aug_policy_template`
        and spatial-eval expansions
        :func:`priml.baselines.arcagi2.scripts.build_dataset.arc2_spatial_eval_template`.
        The literal must stay byte-identical to the metric config's
        ``working_dir`` (that shared string IS the dataset<->metric sync point).
        The trainer prefixes ``base_dir`` at finalize."""

        spatial_eval_views: int = 0
        """Spatial-eval expansion factor S for the test split (0 = plain).

        When > 0, staging first ensures the base ARC-AGI-2 tree at
        ``source_dataset_dir``, then stages its S-view spatial test expansion
        (translated/scaled variants of every canonical test view, tagged for
        metric-side inversion) directly at ``working_dir``."""

        source_dataset_dir: Path | str = ""
        """Base tree the spatial expansion is built from.

        Only consulted when ``spatial_eval_views > 0``; set it to the
        aug-policy ``working_dir`` the expansion derives from. A logical
        ``/datasets/...`` path resolved beneath ``base_dir`` exactly like
        ``working_dir``. Empty otherwise."""

        input_file_prefix: str = ""
        """Kaggle JSON source prefix for the build; ``""`` clones the pinned source.

        The staged tree's params sentinel records the source, so a tree built
        from a different ARC-AGI-2 packaging -- the revisions differ in their
        evaluation tasks -- is rebuilt rather than silently reused."""

        @override
        def finalize(self) -> Self:
            if self.source_dataset_dir:
                self.source_dataset_dir = resolve_working_dir(
                    self.base_dir,
                    self.source_dataset_dir,
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        # Stage the ARC-AGI-2 tree BEFORE the parent constructor runs: the
        # parent's own ensure step builds ARC-AGI-1 subsets when manifest
        # files are missing, but our ensure (and the spatial expansion's,
        # whose manifest covers the copied train split too) leaves a complete
        # manifest, so the parent's existence-gated ensure no-ops. Gated like
        # the parent (count 0 = tests/smoke with tiny local data, never the
        # real build path).
        if config.num_puzzle_identifiers > 0:
            if config.spatial_eval_views > 0 and not str(config.source_dataset_dir):
                raise ValueError(
                    "spatial_eval_views > 0 requires source_dataset_dir (the base "
                    "tree the spatial expansion derives from).",
                )
            source_dir = Path(
                config.source_dataset_dir
                if config.spatial_eval_views > 0
                else config.working_dir,
            )
            spatial = config.augmentation.spatial
            ensure_arc2_dataset(
                target=source_dir,
                train_scale_weights=spatial.train_scale_weights,
                translation_prob=spatial.translation_prob,
                scale_prob=spatial.scale_prob,
                num_aug=config.augmentation.num_aug,
                seed=config.augmentation.seed,
                input_file_prefix=config.input_file_prefix or None,
            )
            if config.spatial_eval_views > 0:
                # Pin the expansion to this config's own resolved working_dir so
                # the build target and the read path cannot diverge (even
                # under a non-default resource root).
                ensure_spatial_eval_data(
                    source_dir=source_dir,
                    spatial_views=config.spatial_eval_views,
                    target=Path(config.working_dir),
                )
        super().__init__(config)
