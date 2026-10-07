"""Two-rank ARC2 sampling and optimizer collectives against source records.

Each rank records its own view -- rank-local batches, gradients after the
collective, dense and sparse updates, a resumed update, and both optimizer
checkpoints -- and the record must equal the one the implementation this
recipe was ported from produced on the same rank. Mutation controls drop one
distributed mechanism at a time and must fail at the checkpoint that mechanism
owns.

Training, including construction, is recorded under ``host_agnostic_numerics``
for every fault arm alike, so the golden is portable across CPU ISAs.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, Protocol, cast, override

import math
import traceback

from torch import Tensor, nn
from torch.optim import Optimizer

import numpy as np
import pytest
import torch

from priml.baselines.arcagi2.data import Arc2Data
from priml.baselines.arcagi2.record_test import load, reduce
from priml.baselines.arcagi2.train_step import ArcDataParallel, ArcTrainStep
from priml.baselines.arcagi2.train_step_test import (
    capture_gradients,
    rows,
    training_batch,
    training_config,
)
from priml.baselines.sudoku.embedding import GridEmbedding
from priml.lib.codec import from_plain
from priml.runtime import MultiProcess
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import joined, mismatches, put_steps
from priml.train.parallelism import DataParallel, NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter
    from priml.train.custom_types import TrainStepOutput


type Fault = Literal["none", "broadcast", "dense", "sparse", "clipping", "ulp"]


class Loader(Protocol):
    """The dataset surface the sampling recorder drives."""

    def train_dataloader(self) -> Iterable[Mapping[str, object]]:
        """Return a training pass."""
        ...

    def eval_dataloader(self) -> Iterable[Mapping[str, object]]:
        """Return the evaluation pass."""
        ...


class Subject(Protocol):
    """One implementation of the ARC2 step, seen through the compared surface."""

    @property
    def model(self) -> nn.Module:
        """The dense body."""
        ...

    @property
    def latent_inits(self) -> tuple[Tensor, Tensor]:
        """The slow and fast latent inits."""
        ...

    @property
    def dense_optimizer(self) -> Optimizer:
        """The dense optimizer."""
        ...

    @property
    def sparse_local(self) -> Tensor:
        """The per-batch sparse rows."""
        ...

    @property
    def sparse_table(self) -> Tensor:
        """The master sparse table."""
        ...

    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one training call."""
        ...

    def optimizer_states(self) -> dict[str, object]:
        """Dense and sparse optimizer checkpoints."""
        ...

    def resumed(self) -> Subject:
        """Return a fresh subject at double sparse rate, loaded from this state."""
        ...


def prepared_tree(root: Path) -> None:
    """Two tasks, four rows, one tree for both splits."""
    for split in ("train", "test"):
        directory = root / split
        directory.mkdir()
        (directory / "dataset.json").write_text(
            '{"ignore_label_id": 0, "blank_identifier_id": 0}',
        )
        arrays = {
            "inputs": np.arange(12, dtype=np.int32).reshape(4, 3) % 2 + 2,
            "labels": (np.arange(12, dtype=np.int32).reshape(4, 3) + 1) % 2 + 2,
            "puzzle_indices": np.array([0, 2, 4], dtype=np.int64),
            "group_indices": np.array([0, 1, 2], dtype=np.int64),
            "puzzle_identifiers": np.array([1, 2], dtype=np.int32),
        }
        for name, array in arrays.items():
            np.save(directory / f"all__{name}.npy", array)


def record_sampling(loader: Loader) -> dict[str, Tensor]:
    """Record three training passes and the evaluation pass of one rank."""
    train = loader.train_dataloader()
    record: dict[str, Tensor] = {}
    passes = [reduce(_batch(batch)) for _ in range(3) for batch in train]
    put_steps(record, "train", passes)
    put_steps(record, "eval", [reduce(_batch(b)) for b in loader.eval_dataloader()])
    return record


def record_training(
    subject: Subject,
    *,
    rank: int,
    clip: float,
) -> dict[str, Tensor]:
    """Record three updates, the trained state, then a resumed update.

    Args:
      subject: The implementation, already built on this rank.
      rank: This rank, choosing its batches.
      clip: Gradient clipping norm; ``inf`` disables it.

    Returns:
      record: Name-to-tensor record of this rank.

    """
    with host_agnostic_numerics():
        slow, fast = subject.latent_inits
        out: dict[str, object] = {"latent_init": joined([slow, fast])}
        gradients: dict[str, Tensor] = {}
        subject.dense_optimizer.register_step_pre_hook(
            partial(
                capture_gradients,
                model=subject.model,
                sparse=subject.sparse_local,
                target=gradients,
            ),
        )
        steps: list[dict[str, Tensor]] = []
        # Dense gradients reach the final parameters and optimizer state stored
        # below, so only the sparse rows, which each rank keeps local, are kept.
        for index in range(3):
            result = subject.train_step(**training_batch(index, rank))
            step: dict[str, object] = {
                "grad_sparse": rows(gradients.pop("sparse")),
                "loss": result["loss"],
                "model": result["model"],
            }
            if math.isfinite(clip):
                step["grad_norm"] = result.get("metrics", {}).get("grad_norm")
            steps.append(reduce(step))
        # Parameters and the sparse table are carried state: their final value
        # already depends on every update before it.
        out["param"] = _parameters(subject.model)
        out["sparse"] = rows(subject.sparse_table)
        # A resumed update reads every restored optimizer moment, so its loss,
        # output, and sparse rows pin the round trip; the dense state it lands
        # on is implied by those and stays unstored.
        resumed = subject.resumed()
        result = resumed.train_step(**training_batch(3, rank))
        out["resumed/loss"] = result["loss"]
        out["resumed/model"] = result["model"]
        out["resumed/sparse"] = rows(resumed.sparse_table)
        out["resumed/optimizers"] = _optimizer_structure(resumed.optimizer_states())
        record = reduce(out)
        put_steps(record, "step", steps)
        return record


class PortSubject:
    """The exported ARC2 step."""

    def __init__(self, config: ArcTrainStep.Config) -> None:
        self.config = config
        self.step: ArcTrainStep = config.make()

    @property
    def model(self) -> nn.Module:
        """The dense body."""
        return self.step.model

    @property
    def latent_inits(self) -> tuple[Tensor, Tensor]:
        """The slow and fast latent inits."""
        return self.step.net.slow_init, self.step.net.fast_init

    @property
    def dense_optimizer(self) -> Optimizer:
        """The dense optimizer."""
        assert isinstance(self.step.optimizer, Optimizer)
        return self.step.optimizer

    @property
    def sparse_local(self) -> Tensor:
        """The per-batch sparse rows."""
        return self.step.puzzle_embedding.local_weights

    @property
    def sparse_table(self) -> Tensor:
        """The master sparse table."""
        return self.step.puzzle_embedding.weights

    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one training call."""
        return self.step.train_step(**batch)

    def optimizer_states(self) -> dict[str, object]:
        """Dense and sparse optimizer checkpoints."""
        return {
            "dense": self.dense_optimizer.state_dict(),
            "sparse": self.step.sparse_optimizer.state_dict(),
        }

    def resumed(self) -> PortSubject:
        """Return a fresh step at double sparse rate, loaded from this state."""
        config = self.config.copy_tree()
        config.sparse_optimizer.lr *= 2
        other = PortSubject(config)
        other.step.load_state_dict(self.step.state_dict())
        return other


class GeometrySubject(PortSubject):
    """Repeat the same prepared tokens across a configurable packed grid."""

    def __init__(self, config: ArcTrainStep.Config, grid_len: int) -> None:
        super().__init__(config)
        self.grid_len = grid_len

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        media = cast("Tensor", batch["media"])
        label = cast("Tensor", batch["label"])
        return super().train_step(
            **{
                **batch,
                "media": media.repeat(1, self.grid_len // media.shape[1]),
                "label": label.repeat(1, self.grid_len // label.shape[1]),
            },
        )

    @override
    def resumed(self) -> GeometrySubject:
        config = self.config.copy_tree()
        config.sparse_optimizer.lr *= 2
        other = GeometrySubject(config, self.grid_len)
        other.step.load_state_dict(self.step.state_dict())
        return other


def port_training_config(*, clip: float, fault: Fault) -> ArcTrainStep.Config:
    """Return the port's two-rank recipe, one mechanism removed under ``fault``."""
    candidate = training_config(4, torch.bfloat16)
    candidate.gradient_clip_norm = clip
    candidate.parallelism = ArcDataParallel.Config(gradient_as_bucket_view=True)
    if fault == "broadcast":
        candidate.parallelism = DataParallel.Config(gradient_as_bucket_view=True)
    elif fault == "dense":
        candidate.parallelism = NoParallel.Config(device="cpu")
    elif fault == "sparse":
        candidate.sparse_optimizer.aggregate_distributed = False
    elif fault == "clipping":
        candidate.gradient_clip_norm = math.inf
    return candidate


def seed_of(rank: int, fault: Fault) -> int:
    """Per-rank seeds expose a missing broadcast; the other controls share one."""
    return rank if fault in ("none", "broadcast") else 0


def case_name(kind: str, *, clip: float, seed: str, rank: int) -> str:
    """Key of one rank's record in the frozen source table."""
    return f"{kind}/clip{clip}/seed-{seed}/rank{rank}"


def run_distributed(
    worker: Callable[[DeviceMesh], None],
    root: Path,
    getter: WarmPoolGetter,
) -> list[dict[str, Tensor] | str]:
    """Run ``worker`` on two CPU ranks; return each rank's record or traceback."""
    getter({"dp": 2})(worker)
    out: list[dict[str, Tensor] | str] = []
    for rank in range(2):
        failure = root / f"record_{rank}.txt"
        if failure.is_file():
            out.append(failure.read_text())
            continue
        loaded = cast(
            object,
            torch.load(root / f"record_{rank}.pt", weights_only=True),
        )
        out.append(from_plain(loaded, dict[str, Tensor]))
    return out


def _sampling_worker(root: Path, mesh: DeviceMesh) -> None:
    rank = mesh.get_rank()
    try:
        config = Arc2Data.Config(working_dir=root, batch_size=2, device="cpu")
        port = config.make()
        record = record_sampling(port)
        actual_loader = port.train_dataloader()
        iterator = iter(actual_loader)
        next(iterator)
        resumed = config.make()
        resumed.load_state_dict(port.state_dict())
        assert len(resumed.train_dataloader()) == len(actual_loader)
        for left, right in zip(iterator, resumed.train_dataloader(), strict=True):
            assert not mismatches(reduce(_batch(left)), reduce(_batch(right))), "resume"
        torch.save(record, root / f"record_{rank}.pt")
    except (AssertionError, RuntimeError, ValueError, TypeError, KeyError):
        (root / f"record_{rank}.txt").write_text(traceback.format_exc())


def _training_worker(root: Path, clip: float, fault: Fault, mesh: DeviceMesh) -> None:
    runtime = MultiProcess.Config(
        device="cpu",
        mesh_topology={"dp": 2, "pp": 1, "tp": 1},
    ).make()
    rank = mesh.get_rank()
    try:
        runtime.initialize()
        torch.default_generator.manual_seed(seed_of(rank, fault))
        with host_agnostic_numerics():
            subject = PortSubject(port_training_config(clip=clip, fault=fault))
            if fault == "ulp":
                with torch.no_grad():
                    weight = next(subject.model.parameters())
                    assert weight.dtype == torch.float32
                    weight.view(torch.int32).view(-1)[0] += 1
        record = record_training(subject, rank=rank, clip=clip)
        torch.save(record, root / f"record_{rank}.pt")
    except (AssertionError, RuntimeError, ValueError, TypeError, KeyError):
        (root / f"record_{rank}.txt").write_text(traceback.format_exc())
    finally:
        runtime.destroy()


def _parameters(model: nn.Module) -> Tensor:
    return joined(model.parameters())


# Structure and scalars are what a checkpoint round trip can lose; the state
# tensors are exercised by the resumed update, which reads every one.
def _optimizer_structure(value: object) -> dict[str, Tensor]:
    """Return an optimizer checkpoint's non-tensor leaves, ordered by path."""
    flat = reduce(_without_tensors(value))
    text = {k: v for k, v in flat.items() if v.dtype == torch.uint8}
    numeric = [flat[k].double() for k in sorted(flat.keys() - text.keys())]
    return {**text, "scalars": torch.stack(numeric)}


def _without_tensors(value: object) -> object:
    if isinstance(value, dict):
        items = cast("dict[object, object]", value).items()
        return {k: _without_tensors(v) for k, v in items if not isinstance(v, Tensor)}
    if isinstance(value, list | tuple):
        return [_without_tensors(item) for item in cast("list[object]", value)]
    return value


def _batch(batch: Mapping[str, object]) -> dict[str, object]:
    keys = ("valid_count", "media", "label", "puzzle_identifiers", "spatial_tags")
    return {key: batch[key] for key in keys}


def _expect(
    table: str,
    case: Callable[[int], str],
    records: list[dict[str, Tensor] | str],
) -> list[str]:
    frozen = load(table)
    report: list[str] = []
    for rank, record in enumerate(records):
        assert isinstance(record, dict), record
        report += [
            f"rank{rank} {line}" for line in mismatches(frozen[case(rank)], record)
        ]
    return report


@pytest.mark.compute_distributed
def test_source_global_sampling(tmp_path: Path, warm_pools: WarmPoolGetter) -> None:
    """Plan globally, slice locally, preserve resume, and align empty eval tails."""
    prepared_tree(tmp_path)
    records = run_distributed(partial(_sampling_worker, tmp_path), tmp_path, warm_pools)
    report = _expect("distributed", lambda rank: f"sampling/rank{rank}", records)
    assert not report, "\n".join(report)


@pytest.mark.compute_distributed
@pytest.mark.parametrize("clip", [math.inf, 0.01])
def test_source_distributed_training(
    tmp_path: Path,
    warm_pools: WarmPoolGetter,
    clip: float,
) -> None:
    """Compare distinct rank inputs and shared sparse IDs through three updates."""
    worker = partial(_training_worker, tmp_path, clip, "none")
    records = run_distributed(worker, tmp_path, warm_pools)
    report = _expect(
        "distributed",
        lambda rank: case_name("training", clip=clip, seed="rank", rank=rank),
        records,
    )
    assert not report, "\n".join(report)


@pytest.mark.compute_distributed
@pytest.mark.parametrize(
    ("fault", "failure"),
    [
        ("broadcast", "latent_init"),
        ("dense", "param"),
        ("sparse", "sparse"),
        ("clipping", "step/grad_norm"),
    ],
)
def test_oracle_rejects_missing_distributed_mechanisms(
    tmp_path: Path,
    warm_pools: WarmPoolGetter,
    fault: Fault,
    failure: str,
) -> None:
    """Mutation controls fail at the checkpoint belonging to each mechanism."""
    clip = 0.01 if fault == "clipping" else math.inf
    seed = "rank" if fault == "broadcast" else "zero"
    worker = partial(_training_worker, tmp_path, clip, fault)
    records = run_distributed(worker, tmp_path, warm_pools)
    report = _expect(
        "distributed",
        lambda rank: case_name("training", clip=clip, seed=seed, rank=rank),
        records,
    )
    assert any(failure in line for line in report), "\n".join(report)


@pytest.mark.compute_distributed
def test_one_ulp_weight_bites_distributed_golden(
    tmp_path: Path,
    warm_pools: WarmPoolGetter,
) -> None:
    worker = partial(_training_worker, tmp_path, math.inf, "ulp")
    records = run_distributed(worker, tmp_path, warm_pools)
    report = _expect(
        "distributed",
        lambda rank: case_name("training", clip=math.inf, seed="zero", rank=rank),
        records,
    )
    assert any("param" in line for line in report), "\n".join(report)


# A 900-cell arm (ARC's canvas, vocab 12) took 5-9s and covered no line or branch
# this one misses (coverage.py, 2026-10-03); a 9-cell grid still repeats the
# 3-token batch, which is the geometry under test.
@pytest.mark.parametrize(("grid_len", "vocab"), [(3, 4), (9, 12)])
def test_training_geometry_coverage_probe(grid_len: int, vocab: int) -> None:
    config = port_training_config(clip=math.inf, fault="none")
    config.parallelism = NoParallel.Config(device="cpu")
    config.model.vocab_size = vocab
    assert isinstance(config.model.embedding, GridEmbedding.Config)
    config.model.embedding.grid_shape = (grid_len,)
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.default_generator.manual_seed(0)
        subject = GeometrySubject(config, grid_len)
        record = record_training(subject, rank=0, clip=math.inf)
    assert record["step/loss"].shape[0] == 3
    assert record["step/model"].shape[-1] == vocab


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
