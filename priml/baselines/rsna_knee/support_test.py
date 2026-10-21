"""Test support shared by the knee baseline's tests: tiny inputs, doubles, configs.

A test needing these imports them from here rather than from another test module.
Frozen doubles replay literal numerical outputs, so a stage test can exercise real
wiring without a learned model.
"""

from __future__ import annotations

from dataclasses import field
from io import BytesIO
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, override

import csv
import hashlib
import json
import zlib

from configgle import Fig, Makes, Makes2
from torch import Tensor, nn
from treekle import to_plain

import numpy as np
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


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray
    from PIL import Image
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")


FLUID_COEFFICIENTS: Final = (0.25, 1.0, 0.5, -0.75)
"""Effusion stacker intercept and its z, log-volume and missing-volume weights."""

STATE_COEFFICIENTS: Final = (0.5, 0.25, -1.0, 0.5, 0.75, -0.5, 1.0, -0.25)
"""Baker's stacker intercept, then its z and six state-category weights."""

STUDY_PLANES: Final = ("Sagittal", "Coronal", "Axial")
"""Plane of each synthetic study series, in CSV order."""


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


def dicom_file(
    pixels: NDArray[np.integer],
    *,
    position: tuple[float, float, float],
    orientation: tuple[float, ...] = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    spacing: tuple[float, float] = (0.5, 0.5),
    instance: int = 1,
    laterality: str = "",
    patient: str = "numeric-patient",
    syntax: str = "1.2.840.10008.1.2.1",
    stored: int = 0,
    photometric: str = "MONOCHROME2",
    slope: str = "",
    intercept: str = "",
) -> bytes:
    """Encode one monochrome frame in a native, deflated or encapsulated syntax.

    Args:
      pixels: Row-major stored samples; their dtype sets bit allocation and sign.
      position: Image position of the first pixel in LPS millimeters.
      orientation: Row then column direction cosines.
      spacing: Row and column pixel spacing in millimeters.
      instance: Acquisition instance number.
      laterality: Series laterality tag, empty when absent.
      patient: Patient identity.
      syntax: Transfer syntax UID: explicit or implicit little-endian, explicit
        big-endian, deflated, RLE, baseline JPEG or JPEG 2000.
      stored: Bits stored, or 0 for every allocated bit.
      photometric: Photometric interpretation.
      slope: Rescale slope decimal string, omitted when empty.
      intercept: Rescale intercept decimal string, omitted when empty.

    Returns:
      payload: The complete file, preamble and meta information included.

    """
    order: Literal["little", "big"] = (
        "big" if syntax == "1.2.840.10008.1.2.2" else "little"
    )
    explicit = syntax != "1.2.840.10008.1.2"
    rows, columns = pixels.shape
    bits = 8 * pixels.dtype.itemsize
    elements = [
        (0x00100020, b"LO", patient.encode()),
        (0x00200013, b"IS", str(instance).encode()),
        (0x00200032, b"DS", _decimals(position)),
        (0x00200037, b"DS", _decimals(orientation)),
        (0x00200060, b"CS", laterality.encode()),
        (0x00280002, b"US", (1).to_bytes(2, order)),
        (0x00280004, b"CS", photometric.encode()),
        (0x00280010, b"US", rows.to_bytes(2, order)),
        (0x00280011, b"US", columns.to_bytes(2, order)),
        (0x00280030, b"DS", _decimals(spacing)),
        (0x00280100, b"US", bits.to_bytes(2, order)),
        (0x00280101, b"US", (stored or bits).to_bytes(2, order)),
    ]
    if pixels.dtype.kind == "i":
        elements.append((0x00280103, b"US", (1).to_bytes(2, order)))
    elements += [
        (tag, b"DS", value.encode())
        for tag, value in ((0x00281052, intercept), (0x00281053, slope))
        if value
    ]
    dataset = b"".join(
        _element(tag, vr=vr, value=value, order=order, explicit=explicit)
        for tag, vr, value in elements
    ) + _pixel_element(pixels, syntax=syntax, order=order, explicit=explicit)
    if syntax == "1.2.840.10008.1.2.1.99":
        deflate = zlib.compressobj(wbits=-15)
        dataset = deflate.compress(dataset) + deflate.flush()
    return _meta(syntax) + dataset


def study_series() -> list[list[tuple[int, bytes]]]:
    """Return the synthetic study's numbered files, per series in enumeration order.

    The sagittal series lies at x 32.6 and 33.6 mm, so its median position
    contradicts the explicit right laterality tag on every frame.

    Returns:
      series: Sagittal (two frames), coronal and axial (four each) files.

    """
    image = np.arange(6 * 8, dtype=np.uint16).reshape(6, 8)
    layouts = (
        ((0.0, 1.0, 0.0, 0.0, 0.0, -1.0), [(32.6, -10.0, 20.0), (33.6, -10.0, 20.0)]),
        (
            (1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
            [(-20.0, float(y), 20.0) for y in range(4)],
        ),
        (
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            [(-20.0, -10.0, 1.5 * z) for z in range(4)],
        ),
    )
    series: list[list[tuple[int, bytes]]] = []
    for index, (orientation, positions) in enumerate(layouts):
        files = [
            (
                number,
                dicom_file(
                    image * (index + 2) + 100 * number,
                    position=position,
                    orientation=orientation,
                    instance=number,
                    laterality="R",
                ),
            )
            for number, position in enumerate(positions, start=1)
        ]
        series.append(files[::-1])
    return series


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
                for index, plane in enumerate(STUDY_PLANES)
            ],
        )
    for index, files in enumerate(study_series()):
        directory = root / "test_series" / "study" / str(index + 1)
        directory.mkdir(parents=True)
        for number, payload in files:
            (directory / f"{number}.dcm").write_bytes(payload)
    return StudyData.Config(input_dir=root)


def corrupt_deflate(root: Path) -> None:
    """Replace every file of the first study with an undecodable deflate stream."""
    payload = _meta("1.2.840.10008.1.2.1.99") + b"invalid deflate stream"
    for path in (root / "test_series" / "study").rglob("*.dcm"):
        path.write_bytes(payload)


def postprocess_assets(directory: Path) -> R7Postprocess.Config:
    """Write synthetic stacker assets and name them as postprocess files.

    Args:
      directory: Empty directory receiving the three asset files.

    Returns:
      config: Postprocess config reading the written files.

    """
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
                                FLUID_COEFFICIENTS,
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
                            "b0": STATE_COEFFICIENTS[0],
                            "coef": list(STATE_COEFFICIENTS[1:]),
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
        # Frozen logits hold two passes over one study and twelve findings.
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
        # Frozen readouts hold one study's finding, padded-state and binary logits.
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


def frozen_raptor_logits() -> Tensor:
    """Return two frozen Raptor passes over one study and twelve findings."""
    # RaptorScore scores one study per pass and reads row 0.
    return ((torch.arange(2 * 12) % 7 - 3) / 4).reshape(2, 1, 12).half()


def frozen_folds() -> dict[str, Tensor]:
    """Return four frozen R7 folds' finding, padded-state and binary logits."""
    # R7Score scores one study per fold and reads row 0 of every readout.
    return {
        "logits": ((torch.arange(4 * 12) % 9 - 4) / 8).reshape(4, 1, 12).half(),
        "states": ((torch.arange(4 * 12 * 8) % 11 - 5) / 4).reshape(4, 1, 12, 8).half(),
        "binaries": ((torch.arange(4 * 13) % 5 - 2) / 2).reshape(4, 1, 13).half(),
    }


def r7_stage_config(root: Path) -> R7Stage.Config:
    """Name an R7 stage replaying frozen fold readouts on the CPU.

    Args:
      root: Empty directory receiving every frozen asset.

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
    folds = frozen_folds()
    for index in range(4):
        path = root / f"fold{index}.pt"
        torch.save({name: value[index] for name, value in folds.items()}, path)
        config.checkpoints.append(released_state(path))
    side, anatomy = root / "side.pt", root / "anatomy.pt"
    torch.save({"logit": torch.tensor(-3.0)}, side)
    torch.save({"state_dict": {"logit": torch.zeros(())}}, anatomy)
    config.side_checkpoint = released_state(side)
    config.anatomy_checkpoint = released_state(anatomy, key="state_dict")
    return config


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _decimals(values: Sequence[float]) -> bytes:
    return "\\".join(format(value, "g") for value in values).encode()


def _element(
    tag: int,
    *,
    vr: bytes,
    value: bytes,
    order: Literal["little", "big"] = "little",
    explicit: bool = True,
) -> bytes:
    value += b" " * (len(value) % 2)
    head = (tag >> 16).to_bytes(2, order) + (tag & 0xFFFF).to_bytes(2, order)
    if not explicit:
        return head + len(value).to_bytes(4, order) + value
    if vr in {b"OB", b"OW"}:
        return head + vr + bytes(2) + len(value).to_bytes(4, order) + value
    return head + vr + len(value).to_bytes(2, order) + value


def _pixel_element(
    pixels: NDArray[np.integer],
    *,
    syntax: str,
    order: Literal["little", "big"],
    explicit: bool,
) -> bytes:
    if syntax == "1.2.840.10008.1.2.5":
        fragment = _rle(pixels)
    elif syntax.startswith("1.2.840.10008.1.2.4."):
        buffer = BytesIO()
        image_format = "JPEG2000" if syntax == "1.2.840.10008.1.2.4.90" else "JPEG"
        Image.fromarray(pixels).save(buffer, format=image_format)
        fragment = buffer.getvalue()
    else:
        encoded = pixels.astype(
            pixels.dtype.newbyteorder("<" if order == "little" else ">"),
        )
        vr = b"OB" if pixels.dtype.itemsize == 1 else b"OW"
        return _element(
            0x7FE00010,
            vr=vr,
            value=encoded.tobytes(),
            order=order,
            explicit=explicit,
        )
    # A one-entry offset table, which readers must skip, one fragment, the delimiter.
    fragment += bytes(len(fragment) % 2)
    return (
        bytes.fromhex("e07f10004f420000ffffffff")
        + bytes.fromhex("feff00e00400000000000000")
        + bytes.fromhex("feff00e0")
        + len(fragment).to_bytes(4, "little")
        + fragment
        + bytes.fromhex("feffdde000000000")
    )


def _rle(pixels: NDArray[np.integer]) -> bytes:
    """Encode each byte plane, most significant first, as PackBits literal runs."""
    width = pixels.dtype.itemsize
    planes = pixels.astype(pixels.dtype.newbyteorder(">")).view(np.uint8)
    segments = [
        b"".join(
            bytes([len(run) - 1]) + run
            for run in (data[start : start + 128] for start in range(0, len(data), 128))
        )
        for data in (
            planes.reshape(-1, width)[:, plane].tobytes() for plane in range(width)
        )
    ]
    offsets = accumulate((len(segment) for segment in segments[:-1]), initial=64)
    header = width.to_bytes(4, "little") + b"".join(
        offset.to_bytes(4, "little") for offset in offsets
    )
    return header.ljust(64, b"\0") + b"".join(segments)


def _meta(syntax: str) -> bytes:
    """Encode the preamble and an MR file meta group holding every required element."""
    uids = {
        0x00020002: "1.2.840.10008.5.1.4.1.1.4",
        0x00020003: "2.25.1",
        0x00020010: syntax,
        0x00020012: "2.25.2",
    }
    group = _element(0x00020001, vr=b"OB", value=bytes([0, 1])) + b"".join(
        _element(tag, vr=b"UI", value=uid.encode() + bytes(len(uid) % 2))
        for tag, uid in uids.items()
    )
    length = len(group).to_bytes(4, "little")
    return bytes(128) + b"DICM" + _element(0x00020000, vr=b"UL", value=length) + group


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
