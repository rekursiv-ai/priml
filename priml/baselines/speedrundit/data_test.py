"""Processed ImageNet pairing and reference sampler order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

from PIL import Image

import numpy as np
import torch

from priml.baselines.speedrundit.data import (
    PairedImageLatentDataset,
    SpeedrunImageNetData,
)


if TYPE_CHECKING:
    from pathlib import Path


def _prepared_pair(root: Path, name: str, label: int) -> None:
    image_dir = root / "images" / "00000"
    latent_dir = root / "vae-in" / "00000"
    image_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(label, 0, 0)).save(image_dir / f"img{name}.png")
    np.save(
        latent_dir / f"img-latents-{name}.npy",
        np.full((1, 32, 2, 2), label, dtype=np.float32),
    )


def test_reference_names_and_tensor_values(tmp_path: Path) -> None:
    labels: list[list[str | int]] = []
    for index in range(10):
        name = f"{index:08d}"
        _prepared_pair(tmp_path, name, index + 5)
        labels.append([f"00000/img-latents-{name}.npy", index + 5])
    (tmp_path / "vae-in" / "dataset.json").write_text(json.dumps({"labels": labels}))
    dataset = PairedImageLatentDataset.Config(working_dir=tmp_path).make()
    assert len(dataset) == 10
    assert torch.equal(dataset[0]["latent"], torch.full((32, 2, 2), 5.0))
    assert dataset[0]["image"][0, 0, 0] == 5
    assert dataset[0]["label"] == 5

    config = SpeedrunImageNetData.Config()
    assert isinstance(config.source, PairedImageLatentDataset.Config)
    config.source.working_dir = tmp_path
    config.batch_size = 2
    config.num_workers = 0
    config.pin_memory = False
    data = config.make()
    torch.manual_seed(0)
    order = [batch["label"].tolist() for batch in data.train_dataloader()]
    assert order == [[11, 12], [6, 9], [7, 5], [14, 13], [8, 10]]
