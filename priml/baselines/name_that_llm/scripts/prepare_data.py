"""Validate local competition downloads and record an immutable input manifest."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Self, override

import json
import shutil

from configgle import Fig
from treekle import to_plain

from priml.baselines.name_that_llm.data import CompetitionData
from priml.paths import resolve_working_dir, validated_output_path


class Preparation:
    """Prepare authorized local Kaggle files without accepting account terms."""

    class Config(Fig["Preparation"]):
        """Local source files and the prepared output location."""

        dataset: CompetitionData.Config = field(default_factory=CompetitionData.Config)
        """Input schema; directory is the authorized Kaggle download directory."""

        base_dir: Path | str | None = "/opt/scratch"
        """Root that the source and prepared directories resolve beneath."""

        directory: Path | str = "/datasets/name-that-llm/prepared"
        """Fresh output directory for unchanged CSVs and an input manifest."""

        @override
        def finalize(self) -> Self:
            self.dataset.directory = resolve_working_dir(
                self.base_dir,
                self.dataset.directory,
            )
            self.directory = resolve_working_dir(self.base_dir, self.directory)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self, *args: str) -> None:
        """Validate every input before writing prepared data.

        Args:
          *args: Additional flags are rejected; the config owns preparation.

        """
        if args:
            raise ValueError("Preparation accepts configuration fields only.")
        config = self.config
        if config.dataset.max_prompts is not None:
            raise ValueError("Preparation requires the complete input population.")
        data = config.dataset.make()
        source = Path(config.dataset.directory)
        directory = validated_output_path(config.directory, protected=[source])
        directory.mkdir(parents=True, exist_ok=False)
        for name in data.sha256:
            shutil.copyfile(source / name, dst=directory / name)
        manifest = {
            "competition": "name-that-llm",
            "rows": len(data.ids),
            "prompts": len(set(data.groups)),
            "labels": config.dataset.labels,
            "sha256": data.sha256,
            "ids": data.ids,
        }
        (directory / "manifest.json").write_text(
            json.dumps(to_plain(manifest), indent=2) + "\n",
        )


def prepare_data() -> Preparation.Config:
    """Validate authorized downloads and record their digests and row identifiers."""
    return Preparation.Config()
