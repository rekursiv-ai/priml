"""Test support shared by the knee baseline's tests: tiny inputs, doubles, configs.

A test needing these imports them from here rather than from another test module.
Frozen doubles hold numerical outputs minted from the source implementation, so
a stage test can exercise real wiring without a learned model.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, override

import csv
import hashlib
import json

from configgle import Fig, Makes, Makes2
from torch import Tensor, nn
from treekle import to_plain

import pytest
import torch

from priml.baselines.rsna_knee.data import FINDINGS
from priml.baselines.rsna_knee.imaging import StudyData
from priml.baselines.rsna_knee.inference import (
    EagerExecution,
    R7Postprocess,
    R7Stage,
    RaptorStage,
    ReleasedState,
)
from priml.baselines.rsna_knee.model import R7, Raptor, ReleasedVNet
from priml.baselines.rsna_knee.train_loop import (
    DicomPreparation,
    PreparedStudies,
)
from priml.model.coatnet import CoAtNet, CoAtStage
from priml.model.convnext import ConvNeXt
from priml.model.custom_types import ChannelsInOutConfig
from priml.model.resnet import ResNet
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_CWD: Final = Path(__file__).resolve().parent


class SyntheticPreparation(DicomPreparation):
    """Eight tiny studies whose pixels encode their identity, with no DICOM tree."""

    class Config(Makes["SyntheticPreparation"], DicomPreparation.Config):
        pass

    @override
    def run(self) -> PreparedStudies:
        return PreparedStudies(
            media=torch.arange(8 * 3 * 4, dtype=torch.float32).reshape(8, 3, 4) / 97,
            labels={str(i): [float(i % 2)] * 12 for i in range(8)},
            groups={str(i): str(i) for i in range(8)},
            patient_ids_complete=True,
        )


def assert_infers_only_unspecified_widths(configs: Sequence[Fig[object]]) -> None:
    """Require each boundary to infer one unset width and keep set ones.

    Args:
      configs: Composite configs with `channels_in` and `channels_out` fields.

    """
    for config in configs:
        for channels_in, channels_out in [(3, -1), (-1, 5), (3, 6)]:
            resolved: object = config.copy_tree()
            assert isinstance(resolved, ChannelsInOutConfig)
            resolved.channels_in, resolved.channels_out = channels_in, channels_out
            resolved = resolved.finalize()
            if resolved.channels_in != (
                channels_out if channels_in == -1 else channels_in
            ):
                raise ValueError(
                    "Expected resolved.channels_in == (channels_out if channels_in == -1 else channels_in).",
                )
            if resolved.channels_out != (
                channels_in if channels_out == -1 else channels_out
            ):
                raise ValueError(
                    "Expected resolved.channels_out == (channels_in if channels_out == -1 else channels_out).",
                )


def blocks_of[Block: Fig[object]](
    backbone: CoAtNet.Config,
    kind: type[Block],
) -> list[Block]:
    """Return every transition and block config of one kind across the stages.

    Args:
      backbone: A CoAtNet config whose stages are the explicit default list.
      kind: Block config class to select.

    Returns:
      blocks: Each stage's transition, then its block template or blocks.

    """
    assert isinstance(backbone.stage, list)
    found: list[Block] = []
    for stage in backbone.stage:
        blocks = stage.block if isinstance(stage.block, list) else [stage.block]
        found.extend(
            block for block in [stage.transition, *blocks] if isinstance(block, kind)
        )
    return found


def released_state(path: Path, *, key: str | None = None) -> ReleasedState.Config:
    """Name an existing checkpoint file and require its current bytes.

    Args:
      path: Checkpoint written by the test.
      key: Entry holding the model tensors, or None for the object itself.

    Returns:
      config: A released-state config bound to the file's digest.

    """
    config = ReleasedState.Config()
    config.file.path, config.file.sha256 = path, _digest(path)
    config.key = key
    return config


def study_inputs(root: Path) -> StudyData.Config:
    """Write a two-study competition tree and name it as study data.

    Args:
      root: Empty directory receiving the CSVs and series trees.

    Returns:
      config: Study data reading the written tree.

    """
    expected = read_tensors(_CWD / "testdata" / "study.pt")
    with (root / "test.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID"])
        writer.writerows([["study"], ["empty"]])
    with (root / "sample_submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *reversed(FINDINGS)])
        writer.writerows(
            [[uid, *([0.5] * len(FINDINGS))] for uid in ("empty", "study")],
        )
    with (root / "test_series.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Anatomical_Plane",
                "Fat_Suppression",
                "Fluid_Sensitive",
            ],
        )
        writer.writerows(
            [
                [
                    "study",
                    str(index + 1),
                    plane,
                    (1, 0, "")[index],
                    (1, 1, "NaN")[index],
                ]
                for index, plane in enumerate(("Sagittal", "Coronal", "Axial"))
            ],
        )
    for index in range(3):
        directory = root / "test_series" / "study" / str(index + 1)
        directory.mkdir(parents=True)
        for number in expected[f"enumeration/{index}"].tolist():
            (directory / f"{number}.dcm").write_bytes(
                expected[f"input/{index}/{number}"].numpy().tobytes(),
            )
    return StudyData.Config(input_dir=root)


def corrupt_deflate(root: Path) -> None:
    """Replace every file of the first study with an undecodable deflate stream."""
    syntax = b"1.2.840.10008.1.2.1.99\0"
    meta = b"\x02\x00\x10\x00UI" + len(syntax).to_bytes(2, byteorder="little") + syntax
    payload = bytes(128) + b"DICM" + meta + b"invalid deflate stream"
    for path in (root / "test_series" / "study").rglob("*.dcm"):
        path.write_bytes(payload)


def postprocess_assets(directory: Path) -> R7Postprocess.Config:
    """Write synthetic stacker assets and name them as postprocess files.

    Args:
      directory: Empty directory receiving the three asset files.

    Returns:
      config: Postprocess config reading the written files.

    """
    witness = read_tensors(_CWD / "testdata" / "stackers.pt")
    counts = (6, 6, 5, 5, 6, 6, 6, 6, 5, 6, 6, 8)
    parents = (4, 5, 6, 4, 5, 6, 4, 5, 6, 1, 11, 10, 0)
    categories = {
        name: [f"category_{index}" for index in range(count)]
        for name, count in zip(FINDINGS, counts, strict=True)
    }
    binaries = {
        f"binary_{index}": [FINDINGS[parent]] for index, parent in enumerate(parents)
    }
    spec = {
        "categories": categories,
        "prior": {
            name: [0.5] * count for name, count in zip(FINDINGS, counts, strict=True)
        },
        "binaries": binaries,
    }
    columns: list[dict[str, object]] = []
    start = 0
    for name in FINDINGS:
        stop = start + len(categories[name])
        columns.append(
            {
                "class": name,
                "categories": categories[name],
                "start": start,
                "stop": stop,
            },
        )
        start = stop
    for binary, names in binaries.items():
        for name in names:
            columns.append({"binary": binary, "class": name, "column": start})
            start += 1
    config = R7Postprocess.Config()
    config.spec_file.path = directory / "spec.json"
    config.fluid_file.path = directory / "fluid.json"
    config.state_file.path = directory / "state.json"
    Path(config.spec_file.path).write_text(json.dumps(to_plain(spec)))
    feature: dict[str, object] = {
        "weights": [1.0],
        "lines": [{}],
        "combine": "logit",
    }
    Path(config.fluid_file.path).write_text(
        json.dumps(
            to_plain(
                {
                    "feature": feature,
                    "classes": {
                        "Effusion": dict(
                            zip(
                                ("b0", "b_z", "b_lv", "b_miss"),
                                witness["fluid_coefficients"].tolist(),
                                strict=True,
                            ),
                            adopt=True,
                            volume_column="eff_sag_p99_open",
                            view_id=4,
                        ),
                    },
                },
            ),
        ),
    )
    Path(config.state_file.path).write_text(
        json.dumps(
            to_plain(
                {
                    "kind": "state_stacker_v1",
                    "feature": feature,
                    "state_spec_md5": hashlib.md5(
                        Path(config.spec_file.path).read_bytes(),
                        usedforsecurity=False,
                    ).hexdigest(),
                    "layout": {"columns": columns, "width": start},
                    "classes": {
                        "Baker's": {
                            "adopt": True,
                            "features": [
                                "z",
                                *["state:" + name for name in categories["Baker's"]],
                            ],
                            "b0": float(witness["state_coefficients"][0]),
                            "coef": witness["state_coefficients"][1:].tolist(),
                            "volume_column": None,
                        },
                    },
                },
            ),
        ),
    )
    return config


class FrozenRaptor(Raptor):
    """A Raptor whose two passes replay stored logits instead of encoding."""

    class Config(Makes["FrozenRaptor"], Raptor.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config
        nn.Module.__init__(self)
        # Frozen inference.pt stores two passes, one study, twelve findings.
        self.logits = nn.Buffer(torch.zeros(2, 1, 12))
        self.calls = 0

    @override
    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        if set(state) != {"logits"}:
            raise ValueError('Expected state names == {"logits"}.')
        self.logits = nn.Buffer(state["logits"].clone())

    @override
    def forward(self, media: Tensor, **batch: object) -> Tensor:
        if self.training:
            raise ValueError("Expected not self.training.")
        if torch.is_grad_enabled():
            raise ValueError("Expected not torch.is_grad_enabled().")
        if media.shape[:3] != (1, 94, 3):
            raise ValueError("Expected media.shape[:3] == (1, 94, 3).")
        output = self.logits[self.calls % len(self.logits)]
        self.calls += 1
        return output.clone()


class FrozenR7(R7):
    """An R7 fold replaying stored finding, state and binary readouts."""

    class Config(Makes["FrozenR7"], R7.Config):
        encoder: ConvNeXt.Config = field(
            default_factory=lambda: ConvNeXt.Config(blocks_per_stage=(1, 1, 1, 1)),
        )
        """Unused native architecture declaration at the frozen numerical boundary."""

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.state_counts = config.state_counts
        self.pad_multiple = config.pad_multiple
        # Frozen inference.pt keeps one-study finding/state/binary readout schemas.
        self.logits = nn.Buffer(torch.zeros(1, 12))
        self.states = nn.Buffer(torch.zeros(1, 12, 8))
        self.binaries = nn.Buffer(torch.zeros(1, 13))

    @override
    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        if set(state) != {"logits", "states", "binaries"}:
            raise ValueError(
                'Expected state names == {"logits", "states", "binaries"}.',
            )
        self.logits, self.states, self.binaries = (
            nn.Buffer(state[name].clone()) for name in ("logits", "states", "binaries")
        )

    @override
    def forward(
        self,
        image: Tensor,
        meta: Tensor,
        mask: Tensor,
        *,
        chunk: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if image.dtype != torch.uint8:
            raise ValueError("Expected image.dtype == torch.uint8.")
        if self.training:
            raise ValueError("Expected not self.training.")
        if not len(meta) == len(mask) == len(image) == 1:
            raise ValueError("Expected len(meta) == len(mask) == len(image) == 1.")
        if chunk != 192:
            raise ValueError("Expected chunk == 192.")
        if not torch.is_inference_mode_enabled():
            raise ValueError("Expected torch.is_inference_mode_enabled().")
        return self.logits.clone(), self.states.clone(), self.binaries.clone()


class FrozenAnatomy(ReleasedVNet):
    """A segmentation model that predicts one stored logit everywhere."""

    class Config(Makes2["FrozenAnatomy"], ReleasedVNet.Config):
        pass

    def __init__(self, config: Config) -> None:
        del config
        nn.Module.__init__(self)
        self.logit = nn.Buffer(torch.zeros(()))

    @override
    def load_checkpoint(self, state: Mapping[str, Tensor]) -> None:
        if set(state) != {"logit"}:
            raise ValueError('Expected state names == {"logit"}.')
        self.logit = nn.Buffer(state["logit"].clone())

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            raise ValueError("Expected not self.training.")
        if torch.is_grad_enabled():
            raise ValueError("Expected not torch.is_grad_enabled().")
        return x.new_zeros((len(x), 4, *x.shape[-2:])) + self.logit


class FrozenSide(ResNet):
    """A laterality model that predicts one stored logit for every image."""

    class Config(Makes["FrozenSide"], ResNet.Config):
        blocks_per_stage: tuple[int, ...] = (1,)
        """Unused native architecture declaration at the frozen numerical boundary."""

    def __init__(self, config: Config) -> None:
        del config
        nn.Module.__init__(self)
        self.logit = nn.Buffer(torch.zeros(()))

    @override
    def load_torchvision_state_dict(self, state: Mapping[str, Tensor]) -> None:
        if set(state) != {"logit"}:
            raise ValueError('Expected state names == {"logit"}.')
        self.logit = nn.Buffer(state["logit"].clone())

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            raise ValueError("Expected not self.training.")
        if not torch.is_inference_mode_enabled():
            raise ValueError("Expected torch.is_inference_mode_enabled().")
        return x.new_zeros((len(x), 1)) + self.logit


def raptor_stage_config(path: Path, *, logits: Tensor) -> RaptorStage.Config:
    """Name a Raptor stage replaying stored logits on the CPU.

    Args:
      path: Where to write the frozen checkpoint.
      logits: The stored two-pass logits the frozen model returns.

    Returns:
      config: A stage over four-pixel slices cropped to three.

    """
    torch.save({"model": {"logits": logits}}, path)
    config = RaptorStage.Config()
    config.checkpoint = released_state(path, key="model")
    config.model = FrozenRaptor.Config()
    backbone = config.model.backbone
    assert isinstance(backbone, CoAtNet.Config)
    backbone.image_size = (3, 3)
    # The frozen model never builds its backbone; narrow it so the printed config stays short.
    backbone.stem.channels_hidden = (2, 3)
    backbone.channels_hidden = (3,)
    backbone.blocks_per_stage = (1,)
    backbone.stage = [CoAtStage.Config()]
    config.preparation.image_size, config.windows.crop_size = 4, 3
    config.score.device = "cpu"
    return config


def r7_stage_config(root: Path, expected: Mapping[str, Tensor]) -> R7Stage.Config:
    """Name an R7 stage replaying stored fold readouts on the CPU.

    Args:
      root: Empty directory receiving every frozen asset.
      expected: Stored `inference.pt` tensors.

    Returns:
      config: A stage over four-pixel renders with eager execution.

    """
    assets = root / "calibration"
    assets.mkdir()
    config = R7Stage.Config()
    config.postprocess = postprocess_assets(assets)
    for file in (
        config.postprocess.fluid_file,
        config.postprocess.state_file,
        config.postprocess.spec_file,
    ):
        file.sha256 = _digest(Path(file.path))
    config.model = FrozenR7.Config()
    config.anatomy = FrozenAnatomy.Config()
    config.anatomy.channels_in = 1
    config.anatomy.channels_out = 4
    config.anatomy.channels_hidden = (2,)
    config.anatomy.down = []
    config.anatomy.up = []
    config.side = FrozenSide.Config()
    config.side.channels_hidden = (2,)
    config.execution = EagerExecution.Config()
    config.score.device, config.score.crop = "cpu", 4
    config.preparation.render.image_size = config.preparation.side.image_size = 4
    config.checkpoints = []
    for index in range(4):
        path = root / f"fold{index}.pt"
        torch.save(
            {
                name: expected[f"r7/{name}"][index]
                for name in ("logits", "states", "binaries")
            },
            path,
        )
        config.checkpoints.append(released_state(path))
    side, anatomy = root / "side.pt", root / "anatomy.pt"
    torch.save({"logit": expected["raptor/logits"].new_tensor(-3.0).float()}, side)
    torch.save({"state_dict": {"logit": torch.zeros(())}}, anatomy)
    config.side_checkpoint = released_state(side)
    config.anatomy_checkpoint = released_state(anatomy, key="state_dict")
    return config


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _fold() -> FrozenR7:
    config = FrozenR7.Config(state_counts=(2,) * 12, pad_multiple=16)
    return config.make()


def test_frozen_fold_starts_with_one_study_readout_schemas() -> None:
    fold = _fold()
    assert fold.pad_multiple == 16
    assert fold.state_counts == (2,) * 12
    assert (fold.logits.shape, fold.states.shape, fold.binaries.shape) == (
        (1, 12),
        (1, 12, 8),
        (1, 13),
    )


def test_frozen_fold_replays_loaded_readouts_at_the_released_chunk() -> None:
    fold = _fold()
    # FrozenR7.forward serves one study with 12 findings, 8 states and 13 binaries.
    state = {
        "logits": torch.ones(1, 12),
        "states": torch.full((1, 12, 8), 2.0),
        "binaries": torch.full((1, 13), 3.0),
    }
    fold.load_checkpoint(state)
    fold.eval()
    with torch.inference_mode():
        # FrozenR7.forward serves one study with 12 findings, 8 states and 13 binaries.
        out = fold(
            torch.zeros(1, 3, 4, 4, dtype=torch.uint8),
            torch.zeros(1, 9),
            torch.zeros(1, 1),
            chunk=192,
        )
    assert all(
        torch.equal(got, state[name]) for got, name in zip(out, state, strict=True)
    )


def test_frozen_fold_rejects_a_state_with_other_names() -> None:
    with pytest.raises(ValueError, match="state names"):
        # FrozenR7.forward serves one study with 12 findings, 8 states and 13 binaries.
        _fold().load_checkpoint({"logits": torch.zeros(1, 12)})


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
