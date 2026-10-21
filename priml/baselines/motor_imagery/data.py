"""Cue-aligned public-reference preprocessing and subject-disjoint loaders.

Filtering and epoch standardization run from raw CSV inside the campaign timer.
Standardization has no fitted state: each trial uses only its own samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, cast, override

import csv
import math
import re

from configgle import Fig
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from treekle import from_plain

import numpy as np
import torch

from priml.data.na_values import PANDAS_NA_VALUES
from priml.paths import resolve_working_dir, validated_output_path
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from numpy.typing import NDArray


NUM_CLASSES: Final = 2
"""Rest and move."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Trial:
    """One cue and all of its overlapping temporal windows."""

    subject: str
    trial_id: int
    label: int
    windows: Tensor


class Preprocessing:
    """Reproduce the notebook's two filters, per-trial zscore, and windows."""

    class Config(Fig["Preprocessing"]):
        channels: tuple[str, ...] = ()
        """Named electrodes in the reference's exact order."""

        sample_rate: float = 0.0
        """Native samples per second."""

        notch_hz: float = 0.0
        """Mains notch center frequency."""

        notch_quality: float = 0.0
        """Reference IIR notch quality factor."""

        band_hz: tuple[float, float] = (0.0, 0.0)
        """Reference Butterworth passband; retained even though prose says broad."""

        filter_order: int = 0
        """Butterworth design order."""

        eps: float = 0.0
        """Added to the population standard deviation before division."""

        window_size: int = 0
        """Samples in each model input."""

        window_step: int = 0
        """Samples between overlapping inputs."""

    def __init__(self, config: Config) -> None:
        if config.window_size <= 0 or config.window_step <= 0:
            raise ValueError("window_size and window_step must be positive.")
        self.config = config

    def filter(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the reference's zero-phase notch followed by bandpass.

        Args:
          values: Continuous samples by named EEG channel.

        Returns:
          filtered: Float64 zero-phase filtered samples.

        """
        cfg = self.config
        notch = _notch(
            cfg.notch_hz,
            quality=cfg.notch_quality,
            sample_rate=cfg.sample_rate,
        )
        band = _butterworth_bandpass(
            cfg.band_hz,
            order=cfg.filter_order,
            sample_rate=cfg.sample_rate,
        )
        notched = _zero_phase_filter(values, b=notch[0], a=notch[1])
        return _zero_phase_filter(notched, b=band[0], a=band[1])

    def standardize(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize one cue interval without consulting another trial."""
        mean = np.mean(values, axis=0, keepdims=True)
        std = np.std(values, axis=0, keepdims=True)
        return (values - mean) / (std + self.config.eps)

    def windows(
        self,
        values: NDArray[np.float64],
        *,
        pad_short: bool,
    ) -> NDArray[np.float64]:
        """Extract complete windows, padding a short inference cue only.

        Args:
          values: Normalized samples by channels.
          pad_short: Apply the public notebook's inference-only zero padding.

        Returns:
          windows: Windows by channels by samples, retaining float64 storage.

        """
        size = self.config.window_size
        windows = [
            values[start : start + size].T
            for start in range(
                0,
                len(values) - size + 1,
                self.config.window_step,
            )
        ]
        if pad_short and not windows:
            padded = np.zeros((size, values.shape[-1]))
            padded[: len(values)] = values
            windows.append(padded.T)
        if not windows:
            return np.empty((0, values.shape[-1], size), dtype=np.float64)
        return np.array(windows)

    def read(
        self,
        paths: Sequence[Path],
        *,
        labelled: bool,
    ) -> list[Trial]:
        """Segment named CSV electrodes between consecutive nonempty markers.

        Args:
          paths: Recording paths, sorted lexicographically before ingestion.
          labelled: Read move/rest cues; otherwise read masked cue_start cues.

        Returns:
          trials: Cue intervals with monotonically increasing trial identifiers.

        Raises:
          ValueError: No recording or usable cue exists.

        """
        if not paths:
            raise ValueError("No EEG CSV recordings found; stage the competition data.")
        trials: list[Trial] = []
        for path in sorted(paths):
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            markers = [row["Marker_val"] for row in rows]
            triggers = [
                index
                for index, marker in enumerate(markers)
                if marker not in PANDAS_NA_VALUES
            ]
            values = self.filter(
                np.asarray(
                    [
                        [float(row[channel]) for channel in self.config.channels]
                        for row in rows
                    ],
                    dtype=np.float64,
                ),
            )
            for index in range(len(triggers) - 1):
                marker = markers[triggers[index]]
                if marker not in (("move", "rest") if labelled else ("cue_start",)):
                    continue
                trial = self.standardize(values[triggers[index] : triggers[index + 1]])
                trials.append(
                    Trial(
                        subject=subject_id(path),
                        trial_id=len(trials),
                        label=(1 if marker == "move" else 0) if labelled else -1,
                        windows=torch.from_numpy(
                            self.windows(trial, pad_short=not labelled),
                        ),
                    ),
                )
        if not trials:
            raise ValueError("No EEG cue intervals found in the recording markers.")
        return trials


def subject_id(path: Path) -> str:
    """Extract the person from both hyphenated and underscored filenames."""
    match = re.match(r"^(S\d+)[_-]", string=path.name)
    if match is None:
        raise ValueError(f"Recording filename has no subject identifier: {path.name}.")
    return from_plain(match.group(1), target=str)


class WindowData:
    """Serve the reference's CPU TensorDataset through its torch DataLoader."""

    class Config(Fig["WindowData"]):
        base_dir: Path | str | None = None
        """Resource root supplied during parent finalization."""

        working_dir: Path | str = "/datasets/motor-imagery"
        """Logical raw competition directory containing the Training set CSV files.

        Resolved beneath ``base_dir`` at finalize."""

        preprocessing: Preprocessing.Config = field(
            default_factory=Preprocessing.Config,
        )
        """Cold filtering, cue normalization, and window extraction."""

        validation_subjects: list[str] = field(default_factory=list[str])
        """People excluded completely from training."""

        batch_size: int = 1
        """Training and evaluation batch size."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config, *, trials: list[Trial] | None = None) -> None:
        """Build a declared fold, optionally sharing the campaign's processed cues.

        Args:
          config: Raw paths, preprocessing, fold membership, and batch geometry.
          trials: Runtime cues already processed inside the campaign wall budget.
            None reads the declared recordings, allowing a fold to be rebuilt
            directly from its printed configuration.

        """
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        if trials is None:
            trials = config.preprocessing.make().read(
                sorted((Path(config.working_dir) / "Training set").glob("*.csv")),
                labelled=True,
            )
        validation = set(config.validation_subjects)
        self.train_dataset = _tensor_dataset(
            [trial for trial in trials if trial.subject not in validation],
        )
        self.eval_dataset = _tensor_dataset(
            [trial for trial in trials if trial.subject in validation],
            shape=self.train_dataset.tensors[0].shape[1:],
        )

    def train_dataloader(self) -> _Loader:
        """Shuffle windows and drop the short training batch exactly upstream.

        Returns:
          loader: Source-compatible training batches.

        """
        if len(self.train_dataset) < self.config.batch_size:
            raise ValueError("Training windows do not fill one reference batch.")
        return _Loader(
            DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,
            ),
        )

    def eval_dataloader(self) -> _Loader:
        """Visit every held-out window in stable file/cue/window order.

        Returns:
          loader: Complete validation batches without shuffling.

        """
        return _Loader(
            DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
            ),
        )

    def state_dict(self) -> Mapping[str, object]:
        """Return completed epoch progress; mid-epoch resume is unsupported."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Reject resume because upstream DataLoader has no cursor checkpoint."""
        del state_dict
        raise NotImplementedError("The public reference does not support exact resume.")


def write_submission(
    path: Path,
    *,
    probabilities: Sequence[float],
    expected_rows: int,
) -> None:
    """Write official sequential identifiers and the source's threshold rule.

    Args:
      path: Output CSV destination.
      probabilities: P(move) in sorted file and original cue order.
      expected_rows: Required sample-submission length, never inferred from predictions.

    Raises:
      ValueError: The count or probability range is invalid.

    """
    if len(probabilities) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} submission rows; got {len(probabilities)}.",
        )
    if any(not np.isfinite(value) or value < 0 or value > 1 for value in probabilities):
        raise ValueError("Submission probabilities must be finite and in [0, 1].")
    destination = validated_output_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["ID", "TARGET"])
        writer.writerows(
            (index, "move" if value >= 0.5 else "rest")
            for index, value in enumerate(probabilities)
        )


def _notch(
    center_hz: float,
    *,
    quality: float,
    sample_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Design a second-order IIR notch with unit gain at DC and Nyquist."""
    center = 2 * center_hz / sample_rate
    if center > 1.0 or center < 0.0:
        raise ValueError("Notch frequency must lie between zero and Nyquist.")
    gain = 1.0 / (1.0 + math.tan(center / quality * math.pi / 2.0))
    cosine = math.cos(center * math.pi)
    return (
        np.asarray([gain, -2.0 * gain * cosine, gain], dtype=np.float64),
        np.asarray([1.0, -2.0 * gain * cosine, 2.0 * gain - 1.0], dtype=np.float64),
    )


def _butterworth_bandpass(
    band_hz: tuple[float, float],
    *,
    order: int,
    sample_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Design a digital Butterworth bandpass by the prewarped bilinear transform."""
    nyquist = sample_rate / 2
    low, high = (edge / nyquist for edge in band_hz)
    if low <= 0 or high >= 1 or low >= high:
        raise ValueError("Band edges must increase strictly between zero and Nyquist.")
    # Prewarped for the bilinear map s = 4 (z - 1) / (z + 1).
    warped_low, warped_high = (
        4.0 * math.tan(math.pi * edge / 2.0) for edge in (low, high)
    )
    bandwidth = warped_high - warped_low
    prototype = -np.exp(1j * np.pi * np.arange(1 - order, order, 2) / (2 * order))
    lowpass = prototype * bandwidth / 2
    shift = np.sqrt(lowpass * lowpass - warped_low * warped_high)
    analog = np.concatenate((lowpass + shift, lowpass - shift))
    gain = bandwidth**order * np.real(4.0**order / np.prod(4.0 - analog))
    return (
        np.asarray(gain * np.poly(np.repeat([1.0, -1.0], order)), dtype=np.float64),
        np.asarray(np.poly((4.0 + analog) / (4.0 - analog)).real, dtype=np.float64),
    )


def _filter_initial_state(
    b: NDArray[np.float64],
    *,
    a: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the transposed direct-form delays of a unit step at steady state."""
    numerator = cast("NDArray[np.float64]", b / a[0])
    denominator = cast("NDArray[np.float64]", a / a[0])
    # The steady output is the DC gain; a pole at z = 1 has none and divides by zero.
    steady = math.fsum(numerator) / math.fsum(denominator)
    residual = numerator - denominator * steady
    return np.cumsum(residual[:0:-1])[::-1]


def _zero_phase_filter(
    values: NDArray[np.float64],
    *,
    b: NDArray[np.float64],
    a: NDArray[np.float64],
) -> NDArray[np.float64]:
    edge = 3 * max(len(a), len(b))
    if len(values) <= edge:
        raise ValueError(f"Zero-phase filtering requires more than {edge} samples.")
    extended = np.concatenate(
        (
            2 * values[:1] - values[edge:0:-1],
            values,
            2 * values[-1:] - values[-2 : -edge - 2 : -1],
        ),
        axis=0,
    )
    zi = _filter_initial_state(b, a=a)[:, None]
    forward = _causal_filter(
        extended,
        b=b,
        a=a,
        initial=cast("NDArray[np.float64]", zi * extended[0]),
    )
    backward = _causal_filter(
        forward[::-1],
        b=b,
        a=a,
        initial=cast("NDArray[np.float64]", zi * forward[-1]),
    )
    return backward[::-1][edge:-edge]


def _causal_filter(
    values: NDArray[np.float64],
    *,
    b: NDArray[np.float64],
    a: NDArray[np.float64],
    initial: NDArray[np.float64],
) -> NDArray[np.float64]:
    numerator = cast("list[float]", cast("NDArray[np.float64]", b / a[0]).tolist())
    denominator = cast("list[float]", cast("NDArray[np.float64]", a / a[0]).tolist())
    output = np.empty(values.shape, dtype=np.float64)
    for channel, samples in enumerate(cast("list[list[float]]", values.T.tolist())):
        state = cast("list[float]", initial[:, channel].tolist())
        for index, value in enumerate(samples):
            result = state[0] + numerator[0] * value
            for position in range(len(state) - 1):
                total = value * numerator[position + 1] + state[position + 1]
                state[position] = total - result * denominator[position + 1]
            state[-1] = value * numerator[-1] - result * denominator[-1]
            output[index, channel] = result
    return output


def _tensor_dataset(
    trials: list[Trial],
    *,
    shape: Sequence[int] = (),
) -> TensorDataset:
    if not trials:
        return TensorDataset(
            torch.empty((0, *shape), device="cpu"),
            torch.empty(0, dtype=torch.long, device="cpu"),
            torch.empty(0, dtype=torch.long, device="cpu"),
        )
    media = torch.cat([trial.windows.to(device="cpu") for trial in trials])[:, None]
    label = torch.cat(
        [
            torch.full((len(trial.windows),), trial.label, device="cpu")
            for trial in trials
        ],
    )
    identifiers = torch.cat(
        [
            torch.full((len(trial.windows),), trial.trial_id, device="cpu")
            for trial in trials
        ],
    )
    return TensorDataset(media, label, identifiers)


class _Loader:
    def __init__(self, loader: DataLoader[tuple[Tensor, ...]]) -> None:
        self.loader = loader

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        for media, label, trial in self.loader:
            yield {
                "media": media,
                "label": label,
                "trial": trial,
                "valid_count": torch.tensor(len(media), device="cpu"),
            }

    def __len__(self) -> int:
        return len(self.loader)
