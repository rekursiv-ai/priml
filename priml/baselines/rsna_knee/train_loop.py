"""Cold-study training: acquisition receipts, cold data, the training loop and its budget."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, TypedDict, override

import csv
import hashlib
import json
import math
import zlib

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch.nn import functional
from treekle import from_plain, json_loads, to_plain

import torch

from priml.baselines.rsna_knee.data import FINDINGS
from priml.baselines.rsna_knee.device import synchronize
from priml.baselines.rsna_knee.imaging import dicom_files, read_dicom
from priml.baselines.rsna_knee.model import ImageNetEncoder, Raptor
from priml.data.passes import Passes
from priml.metrics.roc_auc import MacroRocAuc
from priml.paths import resolve_working_dir
from priml.runtime import get_device
from priml.timer import CheckpointableStepTimer
from priml.train.budget import BudgetedRun, Export, WallBudget
from priml.train.tracker import FileTracker
from priml.train.train_loop import TrainLoop
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence


RECEIPT: Final = "acquisition.json"
SELECTED_FILES: Final = "selected-files.csv"
VERIFIED_FILES: Final = "files.csv"
SERIES_FILES: Final = "series.csv"
ORIGINAL_RECEIPT: Final = "original-acquisition.json"
TRANSFER: Final = "transfer.json"
_SPLIT_ATTEMPTS: Final = 10_000


def verify_inventory(
    input_dir: Path,
    *,
    acquisition_dir: Path,
    selected: set[Path],
) -> None:
    """Verify every member of the selected complete series before decoding.

    Args:
      input_dir: Resolved pixel consumer root, also recorded by acquisition.
      acquisition_dir: Acquisition receipt and complete member manifests.
      selected: Series directories relative to the input root, from this recipe.

    Raises:
      ValueError: Acquisition evidence or selected pixel contents differ.
      FileNotFoundError: Acquisition evidence or an expected DICOM is missing.

    """
    receipt, verified = verified_inventory(acquisition_dir)
    if Path(from_plain(receipt["input_dir"], str)) != input_dir.resolve():
        raise ValueError("DICOM acquisition inventory names another input root.")
    members = {
        name: row for name, row in verified.items() if Path(name).parent in selected
    }
    if {Path(name).parent for name in members} != selected:
        raise ValueError(
            "Selected series is absent from the DICOM acquisition inventory.",
        )
    verify_members(input_dir, members=members, selected=selected)


def verified_inventory(
    acquisition_dir: Path,
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    """Read a complete digest-bound receipt and its verified member inventory.

    Args:
      acquisition_dir: Directory containing the receipt and both manifests.

    Returns:
      receipt: Original complete acquisition metadata.
      members: Every verified member in the declared inventory.

    """
    receipt = json_loads(
        (acquisition_dir / RECEIPT).read_text(),
        dict[str, object],
    )
    if not from_plain(receipt["complete"], bool):
        raise ValueError("DICOM acquisition is incomplete.")
    for name, key in (
        (SELECTED_FILES, "inventory_sha256"),
        (VERIFIED_FILES, "verified_sha256"),
    ):
        with (acquisition_dir / name).open("rb") as handle:
            digest = hashlib.file_digest(handle, hashlib.sha256).hexdigest()
        if digest != from_plain(receipt[key], str):
            raise ValueError("DICOM acquisition inventory digest differs.")
    expected = _read_inventory(acquisition_dir / SELECTED_FILES)
    verified = _read_inventory(acquisition_dir / VERIFIED_FILES)
    if (
        expected.keys() != verified.keys()
        or len(expected) != from_plain(receipt["files"], int)
        or any(
            expected[name][key] != verified[name][key]
            for name in expected
            for key in ("bytes", "crc32")
        )
    ):
        raise ValueError("DICOM acquisition inventory is incomplete or inconsistent.")
    return receipt, verified


def verify_members(
    input_dir: Path,
    *,
    members: dict[str, dict[str, str]],
    selected: set[Path],
) -> None:
    """Require every selected member and verify its length, CRC32 and SHA256.

    Args:
      input_dir: Consumer root containing complete acquired pixels.
      members: Expected per-file identities from the digest-bound manifest.
      selected: Complete series directories relative to input_dir.

    """
    for name, row in members.items():
        path = input_dir / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing DICOM acquisition inventory member: {name}.",
            )
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            crc32 = 0
            for chunk in iter(lambda: handle.read(2**20), b""):
                digest.update(chunk)
                crc32 = zlib.crc32(chunk, crc32)
        if (
            path.stat().st_size != int(row["bytes"])
            or digest.hexdigest() != row["sha256"]
            or crc32 != int(row["crc32"])
        ):
            raise ValueError(f"DICOM acquisition inventory member differs: {name}.")
    actual = {
        str(path.relative_to(input_dir))
        for series in selected
        for path in dicom_files(input_dir / series)
    }
    if actual != members.keys():
        raise ValueError("DICOM series files differ from the acquisition inventory.")


def _read_inventory(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            name = row["name"]
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or name in result:
                raise ValueError(
                    "Duplicate or unsafe DICOM acquisition inventory member.",
                )
            result[name] = row
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedStudies:
    """Rendered studies and the patient identity evidence available in DICOM headers."""

    media: Tensor
    labels: dict[str, list[float]]
    groups: dict[str, str]
    patient_ids_complete: bool


class DicomPreparation:
    """Render the labeled studies from raw DICOM pixels on every invocation."""

    class Config(Fig["DicomPreparation"]):
        lower: float = 0.02
        """Lower intensity quantile."""

        upper: float = 0.98
        """Upper intensity quantile."""

        minimum_span: float = 1e-6
        """Floor applied to every quantile span before division."""

        mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
        """RGB centering values, constructed in the image dtype."""

        std: tuple[float, float, float] = (0.229, 0.224, 0.225)
        """RGB divisors, constructed in the image dtype."""

        base_dir: Path | str | None = None
        """Resource root beneath which the logical directories resolve; the loop supplies it."""

        input_dir: Path | str = "/datasets/rsna-knee/raw"
        """Logical official train.csv, train_series.csv and image tree."""

        acquisition_dir: Path | str = "/artifacts/rsna-knee/acquisition"
        """Logical acquisition receipt and expected/verified member inventories."""

        size: int = 128
        """Square slice size after native percentile normalization."""

        slices_per_plane: int = 4
        """Evenly spaced slices from one series in each available anatomical plane."""

        planes: list[str] = field(
            default_factory=lambda: ["Sagittal", "Coronal", "Axial"],
        )
        """Ordered planes preferred for the image bag."""

        @override
        def finalize(self) -> Self:
            self.input_dir = resolve_working_dir(self.base_dir, self.input_dir)
            self.acquisition_dir = resolve_working_dir(
                self.base_dir,
                self.acquisition_dir,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.size < 2 or config.slices_per_plane < 2 or not config.planes:
            raise ValueError(
                "Image size, slice count, and anatomical planes must be nondegenerate.",
            )
        self.config = config

    def run(self) -> PreparedStudies:
        """Decode raw studies and retain available patient grouping evidence.

        Returns:
          prepared: Ordered image bags, gold targets and patient-or-study groups.

        """
        config = self.config
        input_dir = Path(config.input_dir)
        labels = labeled_studies(input_dir / "train.csv")
        with (input_dir / "train_series.csv").open() as handle:
            records = list(csv.DictReader(handle))
        images = input_dir / "train_images"
        if not images.is_dir():
            images = input_dir / "train_series"
        selected = selected_series(records, studies=list(labels), planes=config.planes)
        if any(not series for series in selected.values()):
            raise ValueError("No configured anatomical planes for a labeled study.")
        if not images.is_dir():
            raise FileNotFoundError("Missing DICOM acquisition inventory image tree.")
        verify_inventory(
            input_dir,
            acquisition_dir=Path(config.acquisition_dir),
            selected={
                Path(images.name) / uid / series
                for uid, series_list in selected.items()
                for series in series_list
            },
        )
        media: list[Tensor] = []
        groups: dict[str, str] = {}
        patient_ids_complete = True
        for uid in labels:
            windows: list[Tensor] = []
            patients: set[str] = set()
            for series in selected[uid]:
                files = dicom_files(images / uid / series)
                headers = [(path, read_dicom(path.read_bytes())) for path in files]
                headers.sort(
                    key=lambda pair: pair[1].instance_number or 0,
                )
                patients.update(header.patient_id.strip() for _, header in headers)
                for index in (
                    torch.linspace(0, len(headers) - 1, config.slices_per_plane)
                    .round()
                    .long()
                ):
                    dataset = read_dicom(headers[int(index)][0].read_bytes())
                    image = torch.from_numpy(dataset.modality_pixels())
                    windows.append(normalize_slice(image, config=config))
            if len(patients - {""}) > 1:
                raise ValueError(f"Conflicting patient identifiers in study {uid}.")
            known = patients - {""}
            complete = bool(known) and "" not in patients
            patient_ids_complete &= complete
            groups[uid] = "patient:" + next(iter(known)) if known else "study:" + uid
            # Missing planes repeat existing views; no fabricated zero images enter attention.
            media.append(
                torch.stack(
                    [
                        windows[i % len(windows)]
                        for i in range(len(config.planes) * config.slices_per_plane)
                    ],
                ),
            )
        return PreparedStudies(
            media=torch.stack(media),
            labels=labels,
            groups=groups,
            patient_ids_complete=patient_ids_complete,
        )


def labeled_studies(path: Path) -> dict[str, list[float]]:
    """Read only complete binary gold labels, without using report text.

    Args:
      path: Official training table.

    Returns:
      labels: Ordered study identifiers and finding targets.

    """
    result: dict[str, list[float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if any(row[finding] in ("", "NaN", "nan") for finding in FINDINGS):
                continue
            values = [float(row[finding]) for finding in FINDINGS]
            if any(value not in (0.0, 1.0) for value in values):
                raise ValueError("Gold labels must be binary.")
            uid = row["StudyInstanceUID"]
            if uid in result:
                raise ValueError(f"Duplicate labeled study {uid}.")
            result[uid] = values
    if not result:
        raise ValueError("No completely labeled studies were found.")
    return dict(sorted(result.items()))


def split_studies(
    labels: dict[str, list[float]],
    *,
    groups: dict[str, str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Choose a reproducible grouped split with both classes in every finding.

    Args:
      labels: Complete binary targets keyed by study identity.
      groups: Patient identities where available, otherwise explicit study identities.
      validation_fraction: Fraction of groups assigned to validation.
      seed: Local split RNG seed, independent of model initialization.

    Returns:
      train: Sorted training study identifiers.
      validation: Sorted validation study identifiers.

    """
    if (
        math.isnan(validation_fraction)
        or validation_fraction <= 0
        or validation_fraction >= 1
    ):
        raise ValueError("Validation fraction must lie strictly between zero and one.")
    if labels.keys() != groups.keys() or not labels:
        raise ValueError(
            "Labels and patient groups must name the same nonempty studies.",
        )
    identifiers = sorted(set(groups.values()))
    count = max(1, round(len(identifiers) * validation_fraction))
    if any(
        len({groups[uid] for uid in labels if labels[uid][j] == value}) < 2
        for j in range(len(next(iter(labels.values()))))
        for value in (0.0, 1.0)
    ):
        raise ValueError(
            "Cannot construct a grouped split with both classes in every finding.",
        )
    rng = torch.Generator().manual_seed(seed)
    for _ in range(_SPLIT_ATTEMPTS):
        chosen = {
            identifiers[int(index)]
            for index in torch.randperm(len(identifiers), generator=rng)[:count]
        }
        validation = sorted(uid for uid in labels if groups[uid] in chosen)
        train = sorted(set(labels) - set(validation))
        if (
            train
            and validation
            and all(
                {labels[uid][j] for uid in split} == {0.0, 1.0}
                for split in (train, validation)
                for j in range(len(next(iter(labels.values()))))
            )
        ):
            return train, validation
    raise ValueError(
        "Cannot construct a grouped split with both classes in every finding.",
    )


def selected_series(
    records: list[dict[str, str]],
    *,
    studies: list[str],
    planes: Sequence[str],
) -> dict[str, list[str]]:
    """Pick, per study and plane, a fluid-sensitive series before the lowest UID.

    Args:
      records: Rows of the official series table.
      studies: Labeled study identifiers, in output order.
      planes: Anatomical planes in image-bag order.

    Returns:
      selected: Chosen series identifiers per study, one per available plane.

    """
    selected: dict[str, list[str]] = {uid: [] for uid in studies}
    for uid in studies:
        for plane in planes:
            candidates = [
                row
                for row in records
                if row["StudyInstanceUID"] == uid and row["Anatomical_Plane"] == plane
            ]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda row: (
                        row.get("Fluid_Sensitive") != "1",
                        row["SeriesInstanceUID"],
                    ),
                )
                selected[uid].append(chosen["SeriesInstanceUID"])
    return selected


def normalize_slice(image: Tensor, *, config: DicomPreparation.Config) -> Tensor:
    """Apply the declared slice policy in its original operation order.

    Args:
      image: Two-dimensional modality-rescaled pixels.
      config: Rendering geometry, percentile floor and RGB normalization values.

    Returns:
      media: Three RGB channels on the declared square image grid.

    """
    bounds = torch.quantile(
        image.flatten(),
        image.new_tensor([config.lower, config.upper]),
    )
    scaled = (
        (image - bounds[0]) / (bounds[1] - bounds[0]).clamp_min(config.minimum_span)
    ).clamp(0, 1)
    resized = functional.interpolate(
        scaled[None, None],
        size=(config.size, config.size),
        mode="bilinear",
        align_corners=False,
    )[0]
    mean = image.new_tensor(config.mean).view(3, 1, 1)
    std = image.new_tensor(config.std).view(3, 1, 1)
    return (resized - mean) / std


class ColdStudies:
    """Render from declared paths and hold study tensors only on runtime state.

    The cold protocol disables checkpointing. Restoring the timer starts a new
    pass rather than resuming a partially consumed permutation.
    """

    class Config(Fig["ColdStudies"]):
        preparation: DicomPreparation.Config = field(
            default_factory=DicomPreparation.Config,
        )
        """Raw input paths and the independent cold rendering policy."""

        batch_size: int = 2
        """Studies per training and complete validation batch."""

        validation_fraction: float = 0.25
        """Fraction of patient groups reserved for validation."""

        split_seed: int = 42
        """Local holdout search seed, independent of model initialization."""

        device: torch.device | str | None = None
        """Resident tensor device; None inherits the loop runtime."""

    class Batch(TypedDict):
        """One complete supervised study batch."""

        media: Tensor
        label: Tensor
        valid_count: int

    class StateDict(TypedDict):
        """Completed pass count; cold runs do not resume a partial pass."""

        timer_epoch: CheckpointableStepTimer.StateDict

    def __init__(self, config: Config) -> None:
        if config.batch_size <= 0:
            raise ValueError("Study batch size must be positive.")
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        # Raw rendering and the split's CPU generator precede the resident-device boundary.
        with torch.device("cpu"):
            self.prepared = config.preparation.make().run()
            self.train_studies, self.validation_studies = split_studies(
                self.prepared.labels,
                groups=self.prepared.groups,
                validation_fraction=config.validation_fraction,
                seed=config.split_seed,
            )
        device = get_device(config.device)
        self.media = self.prepared.media.to(device)
        self.labels = torch.tensor(list(self.prepared.labels.values()), device=device)
        positions = {uid: index for index, uid in enumerate(self.prepared.labels)}
        self.train_index = torch.tensor(
            [positions[uid] for uid in self.train_studies],
            device=device,
        )
        self.validation_index = torch.tensor(
            [positions[uid] for uid in self.validation_studies],
            device=device,
        )

    def train_dataloader(self) -> Passes[Batch]:
        """Return a fresh shuffled pass whenever the shared loop starts an epoch."""
        return Passes(draw=self._train_batches)

    def eval_dataloader(self) -> Passes[Batch]:
        """Return every held-out study once, preserving sorted study identity."""
        return Passes(draw=self._eval_batches)

    def state_dict(self) -> StateDict:
        """Return the dataset-owned completed-pass timer."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the pass timer before starting a fresh loader pass."""
        self.timer_epoch.load_state_dict(
            from_plain(state_dict["timer_epoch"], CheckpointableStepTimer.StateDict),
        )

    def _train_batches(self) -> Iterator[Batch]:
        order = self.train_index[
            torch.randperm(len(self.train_index), device=self.media.device)
        ]
        for index in order.split(self.config.batch_size):
            yield {
                "media": self.media[index],
                "label": self.labels[index],
                "valid_count": len(index),
            }

    def _eval_batches(self) -> Iterator[Batch]:
        for index in self.validation_index.split(self.config.batch_size):
            yield {
                "media": self.media[index],
                "label": self.labels[index],
                "valid_count": len(index),
            }


class ProbabilityAuc(MacroRocAuc):
    """Preserve the previous float32 sigmoid before scoring or CSV export."""

    class Config(Makes["ProbabilityAuc"], MacroRocAuc.Config):
        """Use the original probability-based holdout scoring boundary."""

    @override
    def update(self, logits: Tensor, **batch: object) -> None:
        super().update(logits.float().sigmoid(), **batch)


class KneeTrainLoop(TrainLoop):
    """Use the shared training loop on cold DICOM studies and a finding-pooled model."""

    class Config(
        Makes["KneeTrainLoop"],
        TrainLoop.Config[TrainStep.Config[Raptor.Config], ColdStudies.Config],
    ):
        step: TrainStep.Config[Raptor.Config] = field(
            default_factory=lambda: TrainStep.Config(
                model=Raptor.Config(backbone=ImageNetEncoder.Config()),
            ),
        )
        """Native supervised optimization tree."""

        dataset: ColdStudies.Config = field(default_factory=ColdStudies.Config)
        """Path-built cold DICOM data and patient-disjoint holdout."""

        @override
        def finalize(self) -> Self:
            if self.dataset.preparation.base_dir is None:
                self.dataset.preparation.base_dir = self.base_dir
            backbone = self.step.model.backbone
            if (
                isinstance(backbone, ImageNetEncoder.Config)
                and backbone.base_dir is None
            ):
                backbone.base_dir = self.base_dir
            return super().finalize()


def _cold_loop() -> TrainLoop.Config:
    return KneeTrainLoop.Config()


class ColdExport:
    """Publish the trained weights, holdout probabilities and split."""

    class Config(Fig["ColdExport"]):
        """No fields; the export reads everything from the finished loop."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__(self, loop: TrainLoop, output: Path) -> None:
        """Write the model artefacts of the finished cold run.

        Args:
          loop: The trained loop over cold studies, scored on the full holdout.
          output: Existing run directory.

        """
        step, data = loop.step, loop.dataset
        assert isinstance(step, TrainStep)
        assert isinstance(data, ColdStudies)
        metric = loop.metrics_eval[""]
        assert isinstance(metric, ProbabilityAuc)
        export_run(
            output,
            model=step.model,
            steps=step.global_step,
            prepared=data.prepared,
            validation=data.validation_studies,
            predictions=torch.cat(metric.predictions),
        )
        synchronize(step.device)


class ColdRun(BudgetedRun):
    """Measure a separate native recipe against the full five-minute wall budget."""

    class Config(Makes["ColdRun"], BudgetedRun.Config):
        loop: TrainLoop.Config = field(default_factory=_cold_loop)
        """The complete declared data, model, optimization and evaluation tree."""

        budget: WallBudget.Config = field(
            default_factory=lambda: WallBudget.Config(
                seconds=300.0,
                reserve_evaluation_sec=45.0,
            ),
        )
        """Shared cold clock, evaluation reserve and durable receipt ownership."""

        export: Makeable[Export] | None = field(default_factory=ColdExport.Config)
        """Weights, holdout probabilities and split, charged to the budget."""

        @property
        def cold_loop(self) -> KneeTrainLoop.Config:
            """The declared loop, narrowed to its cold-study slots."""
            assert isinstance(self.loop, KneeTrainLoop.Config)
            return self.loop

        @override
        def finalize(self) -> Self:
            self = super().finalize()
            if not self.protected:
                self.protected = [Path(self.cold_loop.dataset.preparation.input_dir)]
            return self

    def __init__(self, config: Config) -> None:
        if (
            any(
                not math.isfinite(value) or value <= 0
                for value in (
                    config.budget.seconds,
                    config.cold_loop.max_time,
                    config.budget.reserve_evaluation_sec,
                )
            )
            or config.budget.reserve_evaluation_sec >= config.budget.seconds
        ):
            raise ValueError(
                "Time budgets must be finite, positive, and leave training time.",
            )
        if config.cold_loop.max_steps <= 0 or config.cold_loop.dataset.batch_size <= 0:
            raise ValueError("Training steps and batch size must be positive.")
        if not isinstance(config.cold_loop.tracker, FileTracker.Config):
            raise TypeError(
                "The cold experiment reads its metrics from a file tracker.",
            )
        super().__init__(config)
        self.cold = config
        receipt: BudgetReceipt = {
            "completed": False,
            "cache_reused": False,
            "steps": 0,
            "gpu": "",
            "qualified_hardware": False,
            "patient_ids_complete": False,
            "train_studies": [],
            "validation_studies": [],
            "metrics": {},
        }
        self.budget.metadata.update(receipt)

    @override
    def run(self, *args: str) -> None:
        """Perform the entire cold experiment and write a receipt on success or failure.

        Args:
          *args: Passthrough arguments are rejected.

        """
        if args:
            raise ValueError("ColdRun accepts configuration fields only.")
        super().run()

    @override
    def _train(self, loop: TrainLoop, output: Path, args: tuple[str, ...]) -> None:
        step, data = loop.step, loop.dataset
        assert isinstance(step, TrainStep)
        assert isinstance(data, ColdStudies)
        metadata = self.budget.metadata
        gpu = qualifying_gpu(step.device) if step.device.type == "cuda" else ""
        metadata.update(
            gpu=gpu,
            qualified_hardware=bool(gpu),
            patient_ids_complete=data.prepared.patient_ids_complete,
            train_studies=data.train_studies,
            validation_studies=data.validation_studies,
        )
        super()._train(loop, output, args)
        tracker = self.cold.cold_loop.tracker
        assert isinstance(tracker, FileTracker.Config)
        tracked = json_loads(Path(tracker.working_dir).read_text(), dict[str, float])
        metadata["metrics"] = {"macro_auc": tracked["eval/macro_auc"]}


def qualifying_gpu(device: torch.device) -> str:
    """Require actual placement on the sole visible H100/H200 and return its name.

    Args:
      device: The constructed training step's placement.

    Returns:
      name: The qualifying device's reported hardware name.

    """
    if device.type != "cuda" or torch.cuda.device_count() != 1:
        raise ValueError("The cold experiment requires exactly one visible GPU.")
    name = torch.cuda.get_device_name(device)
    if not any(model in name for model in ("H100", "H200")):
        raise ValueError("The cold experiment requires an H100 or H200.")
    return name


class BudgetReceipt(TypedDict):
    """Cold experiment outcome, hardware qualification, and measured budget."""

    completed: bool
    cache_reused: bool
    steps: int
    gpu: str
    qualified_hardware: bool
    patient_ids_complete: bool
    train_studies: list[str]
    validation_studies: list[str]
    metrics: dict[str, float]


def export_run(
    output: Path,
    *,
    model: nn.Module,
    steps: int,
    prepared: PreparedStudies,
    validation: list[str],
    predictions: Tensor,
) -> None:
    """Write the trained weights, holdout probabilities and the patient split.

    Args:
      output: Existing run directory.
      model: Trained model whose state is saved on CPU.
      steps: Optimizer updates behind `model`.
      prepared: Studies whose groups and identity evidence are recorded.
      validation: Holdout study identifiers, aligned with `predictions` rows.
      predictions: One probability row per holdout study.

    """
    torch.save(
        {
            "model": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "findings": list(FINDINGS),
            "steps": steps,
        },
        output / "model.pt",
    )
    with (output / "validation.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", *FINDINGS])
        for uid, prediction in zip(validation, predictions, strict=True):
            writer.writerow([uid, *(float(value) for value in prediction)])
    (output / "split.json").write_text(
        json.dumps(
            to_plain(
                {
                    "patient_ids_complete": prepared.patient_ids_complete,
                    "groups": prepared.groups,
                    "validation": validation,
                },
            ),
            indent=2,
        )
        + "\n",
    )
