"""Check cold-study training: receipts, data, loop and budgeted run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import csv
import hashlib
import json
import os
import zlib

from configgle import InlineConfig, traverse
from torch import Tensor, nn
from treekle import ReadError, from_plain, json_loads, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.rsna_knee import train_loop
from priml.baselines.rsna_knee.data import FINDINGS
from priml.baselines.rsna_knee.experiments import train_exp000
from priml.baselines.rsna_knee.imaging import read_dicom
from priml.baselines.rsna_knee.model import ImageNetEncoder
from priml.baselines.rsna_knee.support_test import (
    SyntheticPreparation,
    dicom_file,
)
from priml.baselines.rsna_knee.train_loop import (
    RECEIPT,
    SELECTED_FILES,
    VERIFIED_FILES,
    BudgetReceipt,
    ColdRun,
    ColdStudies,
    DicomPreparation,
    PreparedStudies,
    ProbabilityAuc,
    export_run,
    labeled_studies,
    normalize_slice,
    qualifying_gpu,
    selected_series,
    split_studies,
    verify_inventory,
)
from priml.metrics.roc_auc import MacroRocAuc
from priml.runtime import SingleProcess
from priml.train.parallelism import NoParallel
from priml.train.tracker import TrackerList
from priml.train.train_loop import EvalTimeLimitError, TrainLoop


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "extra",
        "inconsistent",
        "duplicate",
        "unsafe",
        "digest",
        "count",
        "series",
    ],
)
def test_consumer_requires_exact_verified_selected_inventory(
    tmp_path: Path,
    fault: str,
) -> None:
    raw = tmp_path / "raw"
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    names = [
        "train_images/study/series/a.dcm",
        "train_images/study/series/b.dcm",
        "test_images/unused/series/c.dcm",
    ]
    rows: list[list[str | int]] = []
    for name in names:
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = name.encode()
        path.write_bytes(payload)
        rows.append(
            [
                name,
                len(payload),
                zlib.crc32(payload),
                hashlib.sha256(payload).hexdigest(),
            ],
        )
    if fault == "duplicate":
        rows.append(rows[0])
    elif fault == "unsafe":
        rows[0][0] = "../escape.dcm"
    for filename in (SELECTED_FILES, VERIFIED_FILES):
        with (evidence / filename).open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "bytes", "crc32", "sha256"])
            writer.writerows(rows)
    if fault == "inconsistent":
        with (evidence / VERIFIED_FILES).open("a") as handle:
            csv.writer(handle).writerow(["other.dcm", 0, 0, "0" * 64])
    receipt = {
        "complete": True,
        "input_dir": str(raw.resolve()),
        "files": len(rows) + (fault == "count"),
        "inventory_sha256": hashlib.sha256(
            (evidence / SELECTED_FILES).read_bytes(),
        ).hexdigest(),
        "verified_sha256": hashlib.sha256(
            (evidence / VERIFIED_FILES).read_bytes(),
        ).hexdigest(),
    }
    (evidence / RECEIPT).write_text(json.dumps(to_plain(receipt)))
    if fault == "extra":
        (raw / "train_images/study/series/extra.dcm").write_bytes(b"extra")
    elif fault == "digest":
        with (evidence / VERIFIED_FILES).open("a") as handle:
            handle.write("changed")
    # Native verification must not read the unrelated test pixel payload.
    (raw / names[-1]).unlink()
    selected = {
        Path(
            "train_images/study/other"
            if fault == "series"
            else "train_images/study/series",
        ),
    }
    if fault == "none":
        verify_inventory(raw, acquisition_dir=evidence, selected=selected)
        stored = json_loads(
            (evidence / RECEIPT).read_text(),
            dict[str, object],
        )
        assert stored["complete"] is True
    else:
        with pytest.raises(ValueError, match="inventory") as caught:
            verify_inventory(raw, acquisition_dir=evidence, selected=selected)
        assert (
            str(caught.value)
            == {
                "extra": "DICOM series files differ from the acquisition inventory.",
                "inconsistent": "DICOM acquisition inventory is incomplete or inconsistent.",
                "duplicate": "Duplicate or unsafe DICOM acquisition inventory member.",
                "unsafe": "Duplicate or unsafe DICOM acquisition inventory member.",
                "digest": "DICOM acquisition inventory digest differs.",
                "count": "DICOM acquisition inventory is incomplete or inconsistent.",
                "series": "Selected series is absent from the DICOM acquisition inventory.",
            }[fault]
        )


def test_inventory_rejects_a_receipt_without_an_object_schema(tmp_path: Path) -> None:
    (tmp_path / RECEIPT).write_text("[]")
    with pytest.raises(ReadError):
        verify_inventory(
            tmp_path / "raw",
            acquisition_dir=tmp_path,
            selected={Path("train_images/study/series")},
        )


def _skip_flush(descriptor: int) -> None:
    del descriptor


@pytest.fixture(autouse=True)
def _no_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip disk flushes: receipts are read back, not crash-tested, in these tests."""
    monkeypatch.setattr(os, "fsync", _skip_flush)


def test_slice_normalization_exposes_and_preserves_positive_span_floor() -> None:
    config = DicomPreparation.Config(size=2)
    config.lower, config.upper = 0.0, 1.0
    config.minimum_span = 0.5
    config.mean, config.std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    image = torch.tensor([[0.0, 0.1], [0.2, 0.3]])
    assert torch.equal(
        normalize_slice(image, config=config),
        (image / 0.5).expand(3, -1, -1),
    )


@pytest.mark.parametrize("missing", ["", "NaN", "nan"])
def test_labels_exclude_unlabeled_reports_and_preserve_finding_order(
    tmp_path: Path,
    missing: str,
) -> None:
    path = tmp_path / "train.csv"
    with path.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", "Report", *FINDINGS])
        writer.writerow(["weak", "unstructured report", *([missing] * 12)])
        writer.writerow(["second", "must never become a feature", *([1, 0] * 6)])
        writer.writerow(["gold", "must never become a feature", *([0, 1] * 6)])
    studies = labeled_studies(path)
    assert studies == {"gold": [0.0, 1.0] * 6, "second": [1.0, 0.0] * 6}
    assert list(studies) == ["gold", "second"]


def test_split_keeps_patient_groups_together_and_requires_both_classes() -> None:
    labels = {str(i): [float(i % 2), float((i // 2) % 2)] for i in range(12)}
    groups = {str(i): str(i // 2) for i in range(12)}
    train, validation = split_studies(
        labels,
        groups=groups,
        validation_fraction=0.34,
        seed=42,
    )
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(labels)
    assert {groups[uid] for uid in train}.isdisjoint(groups[uid] for uid in validation)
    for selected in (train, validation):
        assert all({labels[uid][i] for uid in selected} == {0.0, 1.0} for i in range(2))
    with pytest.raises(ValueError, match="both classes"):
        split_studies(
            {"a": [0.0], "b": [0.0]},
            groups={"a": "a", "b": "b"},
            validation_fraction=0.5,
            seed=42,
        )


def test_normalization_clips_intensity_and_handles_constant_slices() -> None:
    image = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
    result = normalize_slice(image, config=DicomPreparation.Config(size=6))
    assert result.shape == (3, 6, 6)
    assert torch.isfinite(result).all()
    assert torch.isfinite(
        normalize_slice(torch.zeros(4, 5), config=DicomPreparation.Config(size=6)),
    ).all()


@pytest.mark.parametrize("patient", ["P1", ""])
@pytest.mark.parametrize("image_tree", ["train_images", "train_series"])
@pytest.mark.parametrize("suffix", [".dcm", ".DCM", ".dCm"])
def test_cold_rendering_reads_headers_and_pixels(
    tmp_path: Path,
    patient: str,
    image_tree: str,
    suffix: str,
) -> None:
    config = _raw_fixture(tmp_path)
    config.size = 5
    with (tmp_path / "train.csv").open("a") as handle:
        csv.writer(handle).writerow(["other", *([1, 0] * 6)])
    with (tmp_path / "train_series.csv").open("a") as handle:
        csv.writer(handle).writerow(["other", "series", "Sagittal"])
    for uid in ("gold", "other"):
        directory = tmp_path / image_tree / uid / "series"
        directory.mkdir(parents=True)
        (directory / f"slice{suffix}").write_bytes(_payload(patient))
    _publish_inventory(config)
    result = config.make().run()
    # DicomPreparation.run emits RGB and a fixed square image bag.
    assert result.media.shape == (2, 6, 3, 5, 5)
    assert result.groups == {
        uid: "patient:P1" if patient else f"study:{uid}" for uid in ("gold", "other")
    }
    assert result.patient_ids_complete == bool(patient)
    assert torch.equal(
        result.media[0, 0],
        normalize_slice(
            _pixels(_payload(patient)),
            config=DicomPreparation.Config(size=5),
        ),
    )


def test_cold_rendering_refuses_missing_pixels_and_planes(
    tmp_path: Path,
) -> None:
    config = _raw_fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing DICOM"):
        config.make().run()
    config.planes = ["Coronal"]
    with pytest.raises(ValueError, match="No configured anatomical"):
        config.make().run()
    config.size = 1
    with pytest.raises(ValueError, match="nondegenerate"):
        config.make()


def test_cold_rendering_refuses_conflicting_patient_ids(
    tmp_path: Path,
) -> None:
    config = _raw_fixture(tmp_path)
    directory = tmp_path / "train_images" / "gold" / "series"
    directory.mkdir(parents=True)
    for name in ("a", "b"):
        (directory / f"{name}.dcm").write_bytes(_payload(name))
    _publish_inventory(config)
    with pytest.raises(ValueError, match="Conflicting patient"):
        config.make().run()


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.5] * 12, r"^Gold labels must be binary\.$"),
        ([""] * 12, r"^No completely labeled studies were found\.$"),
    ],
)
def test_label_reader_refuses_soft_or_empty_gold(
    tmp_path: Path,
    values: list[float] | list[str],
    message: str,
) -> None:
    path = tmp_path / "train.csv"
    with path.open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *FINDINGS])
        writer.writerow(["gold", *values])
    with pytest.raises(ValueError, match=message):
        labeled_studies(path)


def test_label_reader_refuses_duplicate_studies(tmp_path: Path) -> None:
    config = _raw_fixture(tmp_path)
    with (tmp_path / "train.csv").open("a") as handle:
        csv.writer(handle).writerow(["gold", *([0, 1] * 6)])
    with pytest.raises(ValueError, match="Duplicate"):
        labeled_studies(Path(config.input_dir) / "train.csv")


@pytest.mark.parametrize("fraction", [0.0, 1.0, float("nan")])
def test_split_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="Validation fraction"):
        split_studies({}, groups={}, validation_fraction=fraction, seed=42)


def test_split_rejects_incomplete_group_inventory() -> None:
    with pytest.raises(ValueError, match="same nonempty"):
        split_studies({"a": [0.0]}, groups={}, validation_fraction=0.25, seed=42)


@pytest.mark.parametrize("fault", ["missing", "changed", "receipt", "root", "recipe"])
def test_rendering_requires_complete_acquisition_before_decoding(
    tmp_path: Path,
    fault: str,
) -> None:
    config = _raw_fixture(tmp_path)
    directory = tmp_path / "train_images/gold/series"
    directory.mkdir(parents=True)
    first = directory / "a.dcm"
    first.write_bytes(b"first")
    (directory / "b.dcm").write_bytes(b"second")
    _publish_inventory(config, complete=fault != "receipt")
    if fault == "missing":
        first.unlink()
    elif fault == "changed":
        first.write_bytes(b"other")
    elif fault == "root":
        other_root = tmp_path / "." / "other"
        config.input_dir = other_root
        other_root.mkdir()
        (other_root / "train.csv").write_bytes(
            (tmp_path / "train.csv").read_bytes(),
        )
        (other_root / "train_series.csv").write_bytes(
            (tmp_path / "train_series.csv").read_bytes(),
        )
        other = other_root / "train_images/gold/series"
        other.mkdir(parents=True)
        for name in ("a.dcm", "b.dcm"):
            (other / name).write_bytes((directory / name).read_bytes())
    elif fault == "recipe":
        (tmp_path / "train_series.csv").write_text(
            "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\ngold,unacquired,Sagittal\n",
        )
    with pytest.raises((ValueError, FileNotFoundError), match=r"acquisition|inventory"):
        config.make().run()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_normalization_derives_quantile_dtype_from_pixels(dtype: torch.dtype) -> None:
    original = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64 if dtype == torch.float32 else torch.float32)
    try:
        result = normalize_slice(
            torch.arange(6, dtype=dtype).reshape(2, 3),
            config=DicomPreparation.Config(size=4),
        )
        assert result.dtype == dtype
        # normalize_slice emits RGB on a square output grid.
        assert result.shape == (3, 4, 4)
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("batch_size", [0, -2])
def test_invalid_batch_sizes_fail_before_raw_preparation(batch_size: int) -> None:
    config = ColdStudies.Config()
    config.batch_size = batch_size
    with pytest.raises(ValueError, match="batch size must be positive"):
        config.make()


def test_cold_dataset_builds_from_declared_paths_and_keeps_every_study(
    tmp_path: Path,
) -> None:
    config = ColdStudies.Config()
    config.preparation = SyntheticPreparation.Config()
    config.preparation.input_dir = tmp_path / "raw"
    config.device = "cpu"
    config.batch_size = 4
    assert list(traverse(config, torch.Tensor)) == []
    with torch.device("meta"):
        data = config.make()
    assert set(data.train_studies).isdisjoint(data.validation_studies)
    assert set(data.train_studies) | set(data.validation_studies) == set(
        data.prepared.labels,
    )
    for _ in range(2):
        batches = list(data.train_dataloader())
        assert [batch["valid_count"] for batch in batches] == [4, 2]
        media = torch.cat([batch["media"] for batch in batches])
        labels = torch.cat([batch["label"] for batch in batches])
        rows = (media[:, 0, 0] * 97 / 12).round().long()
        assert sorted(int(value) for value in rows) == [
            int(uid) for uid in data.train_studies
        ]
        assert torch.equal(labels[:, 0], (rows % 2).float())
    batches = list(data.eval_dataloader())
    assert torch.equal(
        torch.cat([batch["media"] for batch in batches]),
        data.media[data.validation_index],
    )
    assert torch.equal(
        torch.cat([batch["label"] for batch in batches]),
        data.labels[data.validation_index],
    )
    data.timer_epoch.global_count = 3
    restored = config.make()
    restored.load_state_dict(data.state_dict())
    assert restored.timer_epoch.global_count == 3


def test_series_selection_prefers_fluid_sensitive_then_lowest_uid() -> None:
    records = [
        {"StudyInstanceUID": "s", "SeriesInstanceUID": uid, "Anatomical_Plane": plane}
        | ({"Fluid_Sensitive": fluid} if fluid else {})
        for uid, plane, fluid in [
            ("a", "Sagittal", "0"),
            ("c", "Sagittal", "1"),
            ("b", "Sagittal", "1"),
            ("z", "Coronal", ""),
            ("y", "Coronal", ""),
        ]
    ] + [
        {
            "StudyInstanceUID": "t",
            "SeriesInstanceUID": "d",
            "Anatomical_Plane": "Axial",
        },
    ]
    assert selected_series(
        records,
        studies=["s", "u"],
        planes=("Sagittal", "Coronal", "Axial"),
    ) == {"s": ["b", "y"], "u": []}


def _payload(patient: str) -> bytes:
    return dicom_file(
        np.arange(15, dtype=np.uint16).reshape(3, 5),
        position=(0.0, 0.0, 0.0),
        patient=patient,
        photometric="MONOCHROME1",
        slope="-0.5",
        intercept="17",
    )


def _pixels(payload: bytes) -> torch.Tensor:
    frame = read_dicom(payload)
    image = torch.from_numpy(frame.pixel_array).float()
    image = image * float(frame.text(0x00281053) or "1") + float(
        frame.text(0x00281052) or "0",
    )
    return image.max() - image if frame.text(0x00280004) == "MONOCHROME1" else image


def _raw_fixture(path: Path) -> DicomPreparation.Config:
    with (path / "train.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *FINDINGS])
        writer.writerow(["gold", *([0, 1] * 6)])
    with (path / "train_series.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane"])
        writer.writerow(["gold", "series", "Sagittal"])
    config = DicomPreparation.Config()
    config.input_dir = path
    config.acquisition_dir = path / "acquisition"
    config.size = 6
    config.slices_per_plane = 2
    _publish_inventory(config)
    return config


def _publish_inventory(
    config: DicomPreparation.Config,
    *,
    complete: bool = True,
) -> None:
    input_dir, acquisition_dir = Path(config.input_dir), Path(config.acquisition_dir)
    acquisition_dir.mkdir(exist_ok=True)
    paths = sorted(input_dir.glob("train_*/*/*/*.[dD][cC][mM]"))
    if not paths:
        paths = [input_dir / "train_images/gold/series/slice.dcm"]
    with (
        (acquisition_dir / SELECTED_FILES).open("w") as selected,
        (acquisition_dir / VERIFIED_FILES).open("w") as verified,
    ):
        expected = csv.writer(selected)
        actual = csv.writer(verified)
        expected.writerow(["name", "bytes", "crc32"])
        actual.writerow(["name", "bytes", "crc32", "sha256"])
        for path in paths:
            payload = path.read_bytes() if path.is_file() else b""
            row = [
                str(path.relative_to(input_dir)),
                len(payload),
                zlib.crc32(payload),
            ]
            expected.writerow(row)
            actual.writerow([*row, hashlib.sha256(payload).hexdigest()])
    (acquisition_dir / RECEIPT).write_text(
        json.dumps(
            to_plain(
                {
                    "complete": complete,
                    "input_dir": str(input_dir.resolve()),
                    "files": len(paths),
                    "inventory_sha256": hashlib.sha256(
                        (acquisition_dir / SELECTED_FILES).read_bytes(),
                    ).hexdigest(),
                    "verified_sha256": hashlib.sha256(
                        (acquisition_dir / VERIFIED_FILES).read_bytes(),
                    ).hexdigest(),
                },
            ),
        ),
    )


def test_split_rejects_studies_that_cannot_be_separated_by_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(train_loop, "_SPLIT_ATTEMPTS", 3)
    labels = {"a": [0.0], "b": [0.0], "c": [1.0], "d": [1.0]}
    with pytest.raises(ValueError, match="grouped split"):
        split_studies(
            labels,
            groups={uid: uid for uid in labels},
            validation_fraction=0.25,
            seed=42,
        )


def test_probability_auc_retains_float32_sigmoid_ties_and_every_prediction() -> None:
    metric = ProbabilityAuc.Config().make()
    logits = torch.tensor([[20, -20], [21, -21]], dtype=torch.bfloat16)
    labels = torch.tensor([[0, 1], [1, 0]])
    metric.update(logits, label=labels)
    assert torch.equal(torch.cat(metric.predictions), logits.float().sigmoid())
    assert metric.compute() == {"macro_auc": 0.75}


def test_shared_loop_evaluates_only_once_after_crossing_an_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    original = ProbabilityAuc.compute

    def compute(metric: ProbabilityAuc) -> dict[str, float]:
        calls.append(1)
        return original(metric)

    monkeypatch.setattr(ProbabilityAuc, "compute", compute)
    config = train_exp000().cold_loop
    config.base_dir = None
    config.working_dir = tmp_path / "loop"
    config.runtime = SingleProcess.Config()
    config.runtime.device = "cpu"
    config.dataset.preparation = SyntheticPreparation.Config()
    config.dataset.preparation.input_dir = tmp_path / "raw"
    config.step.model.backbone = InlineConfig(nn.Identity)
    config.step.model.head.channels_in = 4
    config.step.model.head.channels_hidden = 5
    config.step.dtype_autocast = None
    config.max_steps = 4
    loop = config.make()
    loop.train()
    metric = loop.metrics_eval[""]
    assert isinstance(metric, ProbabilityAuc)
    assert loop.step.global_step == 4
    assert loop.current_epoch == 1
    assert calls == [1]
    assert len(torch.cat(metric.predictions)) == 2
    tracked = json_loads((tmp_path / "loop/metrics.json").read_text(), dict[str, float])
    assert tracked["eval/macro_auc"] == MacroRocAuc.compute(metric)["macro_auc"]


def test_finalize_roots_the_pretrained_encoder_beneath_the_loop_base_dir(
    tmp_path: Path,
) -> None:
    config = train_exp000().cold_loop
    config.base_dir = tmp_path
    backbone = config.step.model.backbone
    assert isinstance(backbone, ImageNetEncoder.Config)
    assert backbone.base_dir is None
    config.finalize()
    assert backbone.base_dir == tmp_path
    assert backbone.checkpoint.path == (
        tmp_path / "datasets/rsna-knee-imagenet/convnext_tiny-983f1562.pth"
    )


def test_cold_protocol_declares_the_shared_training_tree() -> None:
    loops = list(traverse(ColdRun.Config(), TrainLoop.Config, recurse=True))
    assert [match.fqn for match in loops] == ["loop"]


def test_missing_raw_inputs_write_a_failed_cold_receipt(tmp_path: Path) -> None:
    config = train_exp000()
    config.cold_loop.base_dir = None
    config.cold_loop.working_dir = tmp_path / "run"
    config.cold_loop.runtime = SingleProcess.Config()
    config.cold_loop.runtime.device = "cpu"
    config.cold_loop.step.model.backbone = InlineConfig(nn.Identity)
    config.cold_loop.step.model.head.channels_in = 4
    config.cold_loop.step.model.head.channels_hidden = 5
    config.cold_loop.dataset.preparation.input_dir = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        config.make().run()
    receipt = _receipt(Path(config.cold_loop.working_dir) / "budget.json")
    assert receipt["completed"] is False
    assert receipt["cache_reused"] is False
    assert receipt["steps"] == 0
    with pytest.raises(ValueError, match="configuration fields"):
        config.make().run("unexpected")


def test_native_training_evaluates_and_exports_the_entire_holdout(
    tmp_path: Path,
) -> None:
    config = train_exp000()
    config.cold_loop.base_dir = None
    config.cold_loop.working_dir = tmp_path / "run"
    config.cold_loop.runtime = SingleProcess.Config()
    config.cold_loop.runtime.device = "cpu"
    config.cold_loop.dataset.preparation = SyntheticPreparation.Config()
    config.cold_loop.dataset.preparation.input_dir = tmp_path / "raw"
    config.cold_loop.step.model.backbone = InlineConfig(nn.Identity)
    config.cold_loop.step.model.head.channels_in = 4
    config.cold_loop.step.model.head.channels_hidden = 5
    config.cold_loop.step.dtype_autocast = None
    config.cold_loop.max_steps = 3
    config.make().run()
    receipt = _receipt(Path(config.cold_loop.working_dir) / "budget.json")
    assert receipt["completed"] is True
    assert receipt["steps"] == 3
    assert receipt["qualified_hardware"] is False
    assert (Path(config.cold_loop.working_dir) / "model.pt").is_file()
    with (Path(config.cold_loop.working_dir) / "validation.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["StudyInstanceUID"] for row in rows] == receipt["validation_studies"]
    assert len(rows) == 2
    assert all(len(row) == 13 for row in rows)
    metric = MacroRocAuc.Config().make()
    metric.update(
        torch.tensor([[float(row[name]) for name in FINDINGS] for row in rows]),
        label=torch.tensor(
            [[float(int(row["StudyInstanceUID"]) % 2)] * 12 for row in rows],
        ),
    )
    assert receipt["metrics"] == metric.compute()


def test_hardware_receipt_uses_actual_training_placement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = train_exp000()
    config.cold_loop.base_dir = None
    config.cold_loop.working_dir = tmp_path / "run"
    config.cold_loop.runtime = SingleProcess.Config()
    config.cold_loop.runtime.device = "cpu"
    config.cold_loop.dataset.preparation = SyntheticPreparation.Config()
    config.cold_loop.dataset.preparation.input_dir = tmp_path / "raw"
    config.cold_loop.step.model.backbone = InlineConfig(nn.Identity)
    config.cold_loop.step.model.head.channels_in = 4
    config.cold_loop.step.model.head.channels_hidden = 5
    config.cold_loop.step.dtype_autocast = None
    assert isinstance(config.cold_loop.step.parallelism, NoParallel.Config)
    config.cold_loop.step.parallelism.device = "cpu"
    config.cold_loop.max_steps = 2
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", _h200_name)
    config.make().run()
    receipt = _receipt(Path(config.cold_loop.working_dir) / "budget.json")
    assert receipt["completed"] is True
    assert receipt["qualified_hardware"] is False


def _h200_name(device: object) -> str:
    del device
    return "H200"


def test_export_writes_exact_weights_holdout_rows_and_split(tmp_path: Path) -> None:
    model = nn.Linear(2, 3)
    prepared = PreparedStudies(
        media=torch.zeros(2, 3),
        labels={"a": [0.0] * 12, "b": [1.0] * 12},
        groups={"a": "patient:7", "b": "study:b"},
        patient_ids_complete=False,
    )
    predictions = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 32
    export_run(
        tmp_path,
        model=model,
        steps=5,
        prepared=prepared,
        validation=["a", "b"],
        predictions=predictions,
    )
    saved = cast(
        dict[str, object],
        torch.load(tmp_path / "model.pt", weights_only=True),
    )
    assert saved.keys() == {"model", "findings", "steps"}
    assert saved["findings"] == list(FINDINGS)
    assert saved["steps"] == 5
    weights = cast(dict[str, Tensor], saved["model"])
    assert weights.keys() == model.state_dict().keys()
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    with (tmp_path / "validation.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["StudyInstanceUID", *FINDINGS]
    assert [[row[0], *map(float, row[1:])] for row in rows[1:]] == [
        ["a", *predictions[0].tolist()],
        ["b", *predictions[1].tolist()],
    ]
    assert (tmp_path / "split.json").read_text() == (
        '{\n  "patient_ids_complete": false,\n  "groups": {\n'
        '    "a": "patient:7",\n    "b": "study:b"\n  },\n'
        '  "validation": [\n    "a",\n    "b"\n  ]\n}\n'
    )
    with pytest.raises(ValueError, match="zip"):
        export_run(
            tmp_path,
            model=model,
            steps=5,
            prepared=prepared,
            validation=["a"],
            predictions=predictions,
        )


@pytest.mark.parametrize(
    ("gpus", "name"),
    [(0, ""), (2, "H200"), (1, "T4"), (1, "H200")],
)
def test_gpu_qualification_checks_actual_device_and_hardware(
    monkeypatch: pytest.MonkeyPatch,
    gpus: int,
    name: str,
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpus)

    def get_name(device: torch.device) -> str:
        assert device == torch.device("cuda")
        return name

    monkeypatch.setattr(torch.cuda, "get_device_name", get_name)
    if gpus == 1 and name == "H200":
        assert qualifying_gpu(torch.device("cuda")) == name
    else:
        with pytest.raises(ValueError, match="requires"):
            qualifying_gpu(torch.device("cuda"))
    with pytest.raises(ValueError, match="requires"):
        qualifying_gpu(torch.device("cpu"))


@pytest.mark.parametrize("field", ["budget_sec", "max_time", "evaluation_reserve_sec"])
@pytest.mark.parametrize("seconds", [-1.0, 0.0, float("nan"), float("inf")])
def test_budget_rejects_unmeasurable_time_limits(field: str, seconds: float) -> None:
    config = train_exp000()
    if field == "budget_sec":
        config.budget.seconds = seconds
    elif field == "max_time":
        config.cold_loop.max_time = seconds
    else:
        config.budget.reserve_evaluation_sec = seconds
    with pytest.raises(ValueError, match="Time budgets"):
        config.make()


def test_budget_rejects_zero_updates() -> None:
    config = train_exp000()
    config.cold_loop.max_steps = 0
    with pytest.raises(ValueError, match="Training steps"):
        config.make()


def test_no_updates_cannot_be_reported_as_a_completed_cold_run(tmp_path: Path) -> None:
    config = _training_fixture(tmp_path)
    config.cold_loop.max_epochs = 0
    with pytest.raises(TimeoutError, match=r"[Nn]o training time|no time"):
        config.make().run()
    receipt = _receipt(tmp_path / "run/budget.json")
    assert receipt["completed"] is False
    assert receipt["steps"] == 0


def test_failed_final_evaluation_retains_updates_and_a_failed_receipt(
    tmp_path: Path,
) -> None:
    config = _training_fixture(tmp_path)
    config.cold_loop.max_eval_time = 0
    with pytest.raises(EvalTimeLimitError):
        config.make().run()
    receipt = _receipt(tmp_path / "run/budget.json")
    assert receipt["completed"] is False
    assert receipt["steps"] == 3
    assert not (tmp_path / "run/validation.csv").exists()


def _training_fixture(tmp_path: Path) -> ColdRun.Config:
    config = train_exp000()
    config.cold_loop.base_dir = None
    config.cold_loop.working_dir = tmp_path / "run"
    config.cold_loop.runtime = SingleProcess.Config()
    config.cold_loop.runtime.device = "cpu"
    config.cold_loop.dataset.preparation = SyntheticPreparation.Config()
    config.cold_loop.dataset.preparation.input_dir = tmp_path / "raw"
    config.cold_loop.step.model.backbone = InlineConfig(nn.Identity)
    config.cold_loop.step.model.head.channels_in = 4
    config.cold_loop.step.model.head.channels_hidden = 5
    config.cold_loop.step.dtype_autocast = None
    config.cold_loop.max_steps = 3
    return config


def test_exhausted_preparation_budget_cannot_publish_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = train_exp000()
    config.cold_loop.base_dir = None
    config.cold_loop.working_dir = tmp_path / "run"
    config.cold_loop.runtime = SingleProcess.Config()
    config.cold_loop.runtime.device = "cpu"
    config.cold_loop.dataset.preparation = SyntheticPreparation.Config()
    config.cold_loop.dataset.preparation.input_dir = tmp_path / "raw"
    config.cold_loop.step.model.backbone = InlineConfig(nn.Identity)
    config.cold_loop.step.model.head.channels_in = 4
    config.cold_loop.step.model.head.channels_hidden = 5
    config.cold_loop.step.dtype_autocast = None
    clock = _Clock(seconds=0.0)
    config.budget.clock = clock.perf_counter
    original = Path.write_text

    def write(path: Path, text: str) -> int:
        clock.seconds = config.budget.seconds + 1
        return original(path, text)

    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(TimeoutError, match=r"^Outer wall budget exceeded\.$"):
        config.make().run()
    receipt = _receipt(Path(config.cold_loop.working_dir) / "budget.json")
    assert receipt["completed"] is False
    assert receipt["steps"] == 0


@pytest.mark.parametrize("write_sec", [0.0, 0.6, 2.0])
def test_receipt_publication_counts_against_cold_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_sec: float,
) -> None:
    clock = _Clock(seconds=0.0)
    config = ColdRun.Config().budget
    config.clock = clock.perf_counter
    config.receipt_path = tmp_path / "budget.json"
    measured = config.make()
    clock.seconds = 299.0
    original = Path.replace

    def publish(path: Path, target: Path) -> Path:
        clock.seconds += write_sec
        return original(path, target)

    monkeypatch.setattr(Path, "replace", publish)
    if write_sec > 1.0:
        with pytest.raises(TimeoutError, match="wall budget"), measured:
            measured.metadata["qualified_hardware"] = True
    else:
        with measured:
            measured.metadata["qualified_hardware"] = True
    record = json_loads((tmp_path / "budget.json").read_text(), dict[str, object])
    assert record["status"] == ("failed" if write_sec > 1.0 else "complete")
    assert from_plain(record["elapsed_seconds"], float) >= 299.0
    assert record["metadata"] == {"qualified_hardware": True}


def _receipt(path: Path) -> BudgetReceipt:
    record = json_loads(path.read_text(), dict[str, object])
    receipt = from_plain(record["metadata"], BudgetReceipt)
    assert record["status"] == ("complete" if receipt["completed"] else "failed")
    return receipt


@dataclass(slots=True, kw_only=True)
class _Clock:
    seconds: float

    def perf_counter(self) -> float:
        return self.seconds


def test_budget_requires_a_file_tracker() -> None:
    config = train_exp000()
    config.cold_loop.tracker = TrackerList.Config()
    with pytest.raises(TypeError, match="file tracker"):
        config.make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
