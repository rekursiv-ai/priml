"""ETTh1 data loading for long-horizon forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict, cast, override

import csv

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.paths import resolve_working_dir
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class Etth1Data:
    """ETTh1 multivariate forecasting dataset."""

    class Config(Fig["Etth1Data"]):
        """Dataset path and forecasting geometry."""

        base_dir: Path | str | None = None
        """Resource root supplied by the training loop."""

        working_dir: Path | str = "/datasets/etth1"
        """Directory holding the prepared ETTh1.csv, beneath base_dir."""

        channels: int = 7
        """Number of numeric columns after the timestamp."""

        train_rows: int = 12 * 30 * 24
        """Rows in the training split, used to fit normalization."""

        val_rows: int = 4 * 30 * 24
        """Rows forecast in the validation split."""

        test_rows: int = 4 * 30 * 24
        """Rows forecast in the held-out test split."""

        seq_len: int = 336
        """Historical timesteps in each input window."""

        pred_len: int = 96
        """Future timesteps in each target window."""

        batch_size: int = 8
        """Training windows per batch."""

        eval_batch_size: int = 8
        """Validation windows per batch."""

        device: str = "cpu"
        """Device on which batches are returned."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(
                self.base_dir,
                working_dir=self.working_dir,
            )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        self._live: _ForecastBatches | None = None
        self._pending: _ForecastBatches.StateDict | None = None

        if (
            min(
                config.seq_len,
                config.pred_len,
                config.channels,
                config.batch_size,
                config.eval_batch_size,
            )
            <= 0
        ):
            raise ValueError(
                "Window lengths, channels, and batch sizes must be positive.",
            )
        if config.train_rows < config.seq_len + config.pred_len:
            raise ValueError(
                "Training split must contain a complete forecasting window.",
            )
        if min(config.val_rows, config.test_rows) < config.pred_len:
            raise ValueError(
                "Validation and test splits must contain the forecast horizon.",
            )

        values = _read_values(Path(config.working_dir) / "ETTh1.csv")
        if values.shape[1] != config.channels:
            raise ValueError(
                "ETTh1 must contain the configured number of numeric columns.",
            )

        train_end = config.train_rows
        val_end = train_end + config.val_rows
        test_end = val_end + config.test_rows

        if len(values) < test_end:
            raise ValueError(
                f"ETTh1 needs at least {test_end} rows; found {len(values)}.",
            )

        train_values = values[:train_end]

        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0)
        std[std == 0] = 1

        scaled = (values - mean) / std

        self.train = torch.from_numpy(
            scaled[:train_end],
        )

        self.val = torch.from_numpy(
            scaled[train_end - config.seq_len : val_end],
        )

        self.test = torch.from_numpy(
            scaled[val_end - config.seq_len : test_end],
        )

    def train_dataloader(self) -> _ForecastBatches:
        """Return shuffled training windows.

        Returns:
          batches: The live training loader, resumed from any pending state.

        """
        self._live = _ForecastBatches(
            self.train,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            batch_size=self.config.batch_size,
            shuffle=True,
            device=self.config.device,
            consume_test_loader_seed=False,
        )
        if self._pending is not None:
            self._live.load_state_dict(self._pending)
            self._pending = None
        return self._live

    def val_dataloader(self) -> _ForecastBatches:
        """Return shuffled validation windows matching the reference loader.

        Returns:
          batches: Validation windows that also consume the test-loader seed.

        """
        return _ForecastBatches(
            self.val,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            batch_size=self.config.eval_batch_size,
            shuffle=True,
            device=self.config.device,
            consume_test_loader_seed=True,
        )

    def eval_dataloader(self) -> _ForecastBatches:
        """Return validation windows for PRIML evaluation."""
        return self.val_dataloader()

    def test_dataloader(self) -> _ForecastBatches:
        """Return ordered held-out windows, dropping the reference's short tail.

        Returns:
          batches: Unshuffled held-out windows in complete batches.

        """
        return _ForecastBatches(
            self.test,
            seq_len=self.config.seq_len,
            pred_len=self.config.pred_len,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            device=self.config.device,
        )

    class StateDict(TypedDict):
        """Checkpointed dataset state."""

        timer_epoch: CheckpointableStepTimer.StateDict
        loader: _ForecastBatches.StateDict | None

    def state_dict(self) -> StateDict:
        """Return checkpoint state.

        Returns:
          state: The epoch timer and the live or pending loader cursor.

        """
        return {
            "timer_epoch": self.timer_epoch.state_dict(),
            "loader": self._live.state_dict()
            if self._live is not None
            else self._pending,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore checkpoint state."""
        state = cast(Etth1Data.StateDict, state_dict)
        self.timer_epoch.load_state_dict(state["timer_epoch"])
        self._pending = state.get("loader")
        self._live = None

    @property
    def train_epoch_complete(self) -> bool:
        """Check whether the saved cursor has reached the last full batch."""
        state = self._live.state_dict() if self._live is not None else self._pending
        if state is None or state["order"] is None:
            return False
        usable = len(state["order"]) // self.config.batch_size * self.config.batch_size
        return state["position"] == usable

    def finish_train_epoch(self) -> None:
        """Advance the epoch without drawing the next shuffle."""
        if not self.train_epoch_complete:
            raise RuntimeError("The training epoch still has unread batches.")
        self._live = None
        self._pending = None
        self.timer_epoch.global_count += 1
        self.timer_epoch.local_count += 1


class _ForecastBatches:
    """Sliding forecasting windows over one ETTh1 split."""

    def __init__(
        self,
        values: Tensor,
        *,
        seq_len: int,
        pred_len: int,
        batch_size: int,
        shuffle: bool,
        device: str,
        consume_test_loader_seed: bool = False,
    ) -> None:
        self.values = values
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device(device)
        self.consume_test_loader_seed = consume_test_loader_seed
        self._order: Tensor | None = None
        self._position = 0

        self.num_windows = len(values) - seq_len - pred_len + 1

        if self.num_windows < batch_size:
            raise ValueError(
                "Split is too short for a complete batch with "
                f"seq_len={seq_len}, pred_len={pred_len}.",
            )

    def __len__(self) -> int:
        """Return number of complete batches."""
        return self.num_windows // self.batch_size

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """Yield complete forecasting batches."""
        if self._order is None:
            self._order = _reference_order(self.num_windows, shuffle=self.shuffle)
            self._position = 0
        order = self._order

        usable = len(order) // self.batch_size * self.batch_size
        order = order[:usable]

        for start in range(self._position, usable, self.batch_size):
            indices = order[start : start + self.batch_size]

            media = torch.stack(
                [
                    self.values[i : i + self.seq_len]
                    for i in cast(list[int], cast(object, indices.tolist()))
                ],
            )

            label = torch.stack(
                [
                    self.values[i + self.seq_len : i + self.seq_len + self.pred_len]
                    for i in cast(list[int], cast(object, indices.tolist()))
                ],
            )

            self._position = start + self.batch_size
            yield {
                "media": media.to(
                    device=self.device,
                    dtype=torch.float32,
                ),
                "label": label.to(
                    device=self.device,
                    dtype=torch.float32,
                ),
            }

        if self.consume_test_loader_seed:
            # Keep the test-loader seed draw in the same place as the reference.
            _ = torch.empty((), dtype=torch.int64).random_().item()
        self._order = None
        self._position = 0

    class StateDict(TypedDict):
        """Permutation and next window offset for exact mid-epoch resume."""

        order: Tensor | None
        position: int

    def state_dict(self) -> StateDict:
        """Snapshot the active training permutation and cursor."""
        return {"order": self._order, "position": self._position}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore a permutation without consuming another RNG draw."""
        state = cast(_ForecastBatches.StateDict, state_dict)
        self._order = state["order"]
        self._position = state["position"]


def _reference_order(
    count: int,
    *,
    shuffle: bool,
) -> Tensor:
    """Reproduce PyTorch DataLoader and RandomSampler RNG behavior."""
    _ = torch.empty((), dtype=torch.int64).random_().item()

    if not shuffle:
        return torch.arange(count)

    seed = int(
        torch.empty(
            (),
            dtype=torch.int64,
        )
        .random_()
        .item(),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.randperm(
        count,
        generator=generator,
    )


def _read_values(path: Path) -> np.ndarray:
    """Read the seven numeric ETTh1 variables from the CSV file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"ETTh1 CSV not found at {path}. Run "
            "uv --quiet run --frozen python -m "
            "priml.baselines.etth1.scripts.prepare_data first.",
        )

    with path.open(newline="") as file:
        reader = csv.reader(file)
        next(reader, None)

        rows = [[float(value) for value in row[1:]] for row in reader]

    values = np.asarray(
        rows,
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("ETTh1 CSV must have a header and numeric data rows.")
    return values
