"""Replay frozen reference outputs for every ARC data-side entry point.

``testdata/data.pt`` holds what the reference implementation produced for
these seeded inputs. One capture body runs against the priml port
(``PortBackend``) here, and against the reference by the minting test beside
that implementation -- so both compare exactly the same keys. Values are short
digests of arrays (dtype and shape included), whole file trees, batch streams,
and generator positions, plus exact reprs of scalar results. Each capture is
stored as one tensor of ``key<TAB>value`` lines.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import hashlib
import json
import shutil
import zlib

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.arcagi1 import augmentation
from priml.baselines.arcagi1.augmentation import (
    ArcAugmentation,
    ColorDihedral,
    SpatialAugmentation,
)
from priml.baselines.arcagi1.data import (
    PuzzleBatches,
    PuzzleData,
    load_puzzle_dataset,
)
from priml.baselines.arcagi1.metric import (
    CanonicalPassK,
    SignalDumpPayload,
    SignalDumpTracker,
    decode_preds,
    encode_preds,
    write_signal_dump,
)
from priml.baselines.arcagi1.scripts import build_dataset, build_spatial_eval
from priml.lib.custom_json import DictCodec, ListCodec, loads
from priml.testing.golden import read_tensors, write_tensors


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from numpy.lib.npyio import NpzFile
    from numpy.typing import NDArray


type Grid = NDArray[np.uint8]
type Batch = Mapping[str, object]
type Ballots = Mapping[str, Mapping[str, Sequence[tuple[str, float]]]]

_CWD: Final = Path(__file__).resolve().parent

GOLDEN: Final = _CWD / "testdata" / "data.pt"
"""Frozen reference digests, keyed by capture then case."""


@dataclass(slots=True, kw_only=True, frozen=True)
class AugPolicy:
    """Arguments locating an aug-policy tree."""

    translation_prob: float
    scale_prob: float
    train_scale_weights: Mapping[int, float]
    num_aug: int
    seed: int


class Stream(Protocol):
    """A sized, re-iterable batch stream."""

    def __iter__(self) -> Iterator[Batch]: ...
    def __len__(self) -> int: ...


class DataLike(Protocol):
    """The loader surface both implementations share."""

    def train_dataloader(self) -> Stream: ...
    def eval_dataloader(self) -> Stream: ...
    def full_eval_dataloader(self) -> Stream: ...
    def state_dict(self) -> Mapping[str, object]: ...
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None: ...


class MetricLike(Protocol):
    """The metric surface both implementations share."""

    def reset(self) -> None: ...
    def update(self, logits: Tensor, **batch: object) -> None: ...
    def compute(self) -> Mapping[str, object]: ...
    def state_dict(self) -> Mapping[str, object]: ...
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None: ...


class TrackerLike(Protocol):
    """The signal-dump tracker surface."""

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None: ...


@dataclass(slots=True, kw_only=True, frozen=True)
class BuildCase:
    """One builder parameterization."""

    seed: int
    prob: float
    weights: Mapping[int, float]
    num_aug: int
    subsets: tuple[str, ...] = ("training", "evaluation", "concept")
    test_set_name: str = "evaluation"
    with_solutions: bool = True

    @property
    def key(self) -> str:
        """Stable case name."""
        return (
            f"s{self.seed}-p{self.prob}-w{sorted(self.weights.items())}-n{self.num_aug}"
            f"-{'+'.join(self.subsets)}-{self.test_set_name}-sol{self.with_solutions}"
        )


class Backend(Protocol):
    """Every data-side entry point, under one name per role."""

    def dihedral(self, arr: Grid, tid: int) -> Grid: ...
    def inverse_dihedral(self, arr: Grid, tid: int) -> Grid: ...
    def scale_grid(self, grid: Grid, scale: int) -> Grid: ...
    def grid_hash(self, grid: Grid) -> str: ...
    def to_np(self, grid: list[list[int]]) -> Grid: ...
    def crop(self, flat: Grid) -> Grid: ...
    def untranslate(self, flat: Grid, scale: int, pad_r: int, pad_c: int) -> Grid: ...
    def inverse_aug(self, name: str) -> tuple[str, Callable[[Grid], Grid]]: ...
    def square_side(self, length: int) -> int: ...
    def normalize(self, weights: Mapping[int, float]) -> dict[int, float]: ...
    def slug(self, weights: Mapping[int, float]) -> str: ...
    def parse(self, values: Sequence[str]) -> dict[int, float]: ...
    def no_train_scale_weights(self) -> Mapping[int, float]: ...
    def bernoulli(self, prob: float, rng: np.random.Generator) -> bool: ...
    def scale_factor(
        self,
        inp: Grid,
        out: Grid,
        weights: Mapping[int, float],
        rng: np.random.Generator,
        max_grid: int,
    ) -> int: ...
    def pack(
        self,
        inp: Grid,
        out: Grid,
        *,
        training: bool,
        rng: np.random.Generator,
        max_grid: int,
        prob: float,
        weights: Mapping[int, float],
    ) -> list[Grid]: ...
    def color_sample(
        self,
        name: str,
        rng: np.random.Generator,
    ) -> tuple[str, Callable[[Grid], Grid]]: ...
    def build(self, case: BuildCase, prefix: Path, output: Path) -> None: ...
    def ensure_build(self, case: BuildCase, prefix: Path, output: Path) -> None: ...
    def manifest(self) -> list[str]: ...
    def build_policy(
        self,
        *,
        dataset_dir: Path,
        prefix: Path,
        translation_prob: float,
        scale_prob: float,
        weights: Mapping[int, float],
    ) -> None: ...
    def spatial_build(
        self,
        *,
        views: int,
        source: Path,
        target: Path,
        weights: Mapping[int, float],
        seed: int,
    ) -> None: ...
    def spatial_ensure(self, *, source: Path, views: int, target: Path) -> None: ...
    def aug_slug(self, policy: AugPolicy) -> str: ...
    def aug_template(self, policy: AugPolicy) -> str: ...
    def aug_dir(self, policy: AugPolicy) -> Path: ...
    def aug_dir_default(self) -> Path: ...
    def default_scale_weights(self) -> Mapping[int, float]: ...
    def spatial_slug(self, views: int) -> str: ...
    def spatial_dir(self, views: int, source: str | None, base: str) -> Path: ...
    def data_dir(self, base: str | None) -> str: ...
    def metric_dir(self, base: str | None) -> str: ...
    def tracker_dir(self, base: str | None) -> str: ...
    def data(self, fields: Mapping[str, object]) -> DataLike: ...
    def batches(
        self,
        root: Path,
        *,
        train: bool,
        offset: int,
        remap: NDArray[np.int64] | None,
    ) -> Stream: ...
    def load(
        self,
        root: Path,
        split: str,
        max_samples: int | None,
    ) -> Mapping[str, object]: ...
    def metric(self, fields: Mapping[str, object]) -> MetricLike: ...
    def metric_preds(self, scorer: MetricLike) -> Ballots: ...
    def encode(self, preds: Ballots) -> bytes: ...
    def decode(self, payload: bytes) -> Ballots: ...
    def payload(
        self,
        *,
        rows: list[tuple[str, str, str, float, float, float, int, int]],
        grids: dict[str, Grid],
        steps: list[tuple[int, int, tuple[float, ...], tuple[int, ...]]],
        pass_ks: tuple[int, ...],
    ) -> tuple[object, ...]: ...
    def write_dump(self, payload: object, path: Path, step: int) -> None: ...
    def tracker(self, working_dir: str) -> TrackerLike: ...


@dataclass(slots=True, kw_only=True, frozen=True)
class PortBackend:
    """The priml port."""

    def dihedral(self, arr: Grid, tid: int) -> Grid:
        return augmentation.dihedral_transform(arr, tid=tid)

    def inverse_dihedral(self, arr: Grid, tid: int) -> Grid:
        return augmentation.inverse_dihedral_transform(arr, tid=tid)

    def scale_grid(self, grid: Grid, scale: int) -> Grid:
        return augmentation.scale_grid(grid, scale)

    def grid_hash(self, grid: Grid) -> str:
        return augmentation.grid_hash(grid)

    def to_np(self, grid: list[list[int]]) -> Grid:
        return augmentation.arc_grid_to_np(grid, max_grid=augmentation.ARC.max_grid)

    def crop(self, flat: Grid) -> Grid:
        return augmentation.crop_grid(flat)

    def untranslate(self, flat: Grid, scale: int, pad_r: int, pad_c: int) -> Grid:
        return augmentation.untranslate_unscale(
            flat,
            scale=scale,
            pad_r=pad_r,
            pad_c=pad_c,
        )

    def inverse_aug(self, name: str) -> tuple[str, Callable[[Grid], Grid]]:
        return augmentation.inverse_aug(name)

    def square_side(self, length: int) -> int:
        return augmentation.square_side(length, who="capture")

    def normalize(self, weights: Mapping[int, float]) -> dict[int, float]:
        return augmentation.normalize_scale_weights(weights)

    def slug(self, weights: Mapping[int, float]) -> str:
        return augmentation.scale_weights_slug(weights)

    def parse(self, values: Sequence[str]) -> dict[int, float]:
        return augmentation.parse_scale_weights(values)

    def no_train_scale_weights(self) -> Mapping[int, float]:
        return augmentation.NO_TRAIN_SCALE_WEIGHTS

    def bernoulli(self, prob: float, rng: np.random.Generator) -> bool:
        return augmentation.bernoulli(prob, rng)

    def scale_factor(
        self,
        inp: Grid,
        out: Grid,
        weights: Mapping[int, float],
        rng: np.random.Generator,
        max_grid: int,
    ) -> int:
        return augmentation.sample_scale_factor(
            inp,
            out,
            weights,
            rng,
            max_grid=max_grid,
        )

    def pack(
        self,
        inp: Grid,
        out: Grid,
        *,
        training: bool,
        rng: np.random.Generator,
        max_grid: int,
        prob: float,
        weights: Mapping[int, float],
    ) -> list[Grid]:
        config = SpatialAugmentation.Config()
        config.max_grid = max_grid
        config.translation_prob = prob
        config.scale_prob = prob
        config.train_scale_weights = dict(weights)
        return config.make().pack(inp, out, training=training, rng=rng)

    def color_sample(
        self,
        name: str,
        rng: np.random.Generator,
    ) -> tuple[str, Callable[[Grid], Grid]]:
        return ColorDihedral.Config().make().sample(name, rng=rng)

    def build(self, case: BuildCase, prefix: Path, output: Path) -> None:
        build_dataset._build_arc_dataset(
            input_file_prefix=str(prefix),
            output_dir=output,
            augmentation=_recipe(case),
            subsets=case.subsets,
            test_set_name=case.test_set_name,
        )

    def ensure_build(self, case: BuildCase, prefix: Path, output: Path) -> None:
        build_dataset.ensure_arc_dataset(
            target_dir=output,
            augmentation=_recipe(case),
            input_file_prefix=str(prefix),
        )

    def manifest(self) -> list[str]:
        return [spec.rel_path for spec in build_dataset.arc_manifest()]

    def build_policy(
        self,
        *,
        dataset_dir: Path,
        prefix: Path,
        translation_prob: float,
        scale_prob: float,
        weights: Mapping[int, float],
    ) -> None:
        build_dataset.build(
            dataset_dir=str(dataset_dir),
            input_file_prefix=str(prefix),
            translation_prob=translation_prob,
            scale_prob=scale_prob,
            train_scale_weights=weights,
            num_aug=2,
            seed=5,
        )

    def spatial_build(
        self,
        *,
        views: int,
        source: Path,
        target: Path,
        weights: Mapping[int, float],
        seed: int,
    ) -> None:
        build_spatial_eval.build_spatial_eval(
            spatial_views=views,
            source_dir=source,
            target_dir=target,
            scale_weights=weights,
            seed=seed,
        )

    def spatial_ensure(self, *, source: Path, views: int, target: Path) -> None:
        build_spatial_eval.ensure_spatial_eval_data(
            source_dir=source,
            spatial_views=views,
            target=target,
        )

    def aug_slug(self, policy: AugPolicy) -> str:
        return build_dataset.aug_policy_slug(
            translation_prob=policy.translation_prob,
            scale_prob=policy.scale_prob,
            train_scale_weights=policy.train_scale_weights,
            num_aug=policy.num_aug,
            seed=policy.seed,
        )

    def aug_template(self, policy: AugPolicy) -> str:
        return build_dataset.aug_policy_template(
            translation_prob=policy.translation_prob,
            scale_prob=policy.scale_prob,
            train_scale_weights=policy.train_scale_weights,
            num_aug=policy.num_aug,
            seed=policy.seed,
        )

    def aug_dir(self, policy: AugPolicy) -> Path:
        return build_dataset.aug_policy_dataset_dir(
            translation_prob=policy.translation_prob,
            scale_prob=policy.scale_prob,
            train_scale_weights=policy.train_scale_weights,
            num_aug=policy.num_aug,
            seed=policy.seed,
            base_dir="/opt/scratch",
        )

    def aug_dir_default(self) -> Path:
        return build_dataset.aug_policy_dataset_dir(
            translation_prob=0.1,
            scale_prob=0.1,
        )

    def default_scale_weights(self) -> Mapping[int, float]:
        return build_dataset.DEFAULT_SCALE_WEIGHTS

    def spatial_slug(self, views: int) -> str:
        return build_spatial_eval.spatial_eval_slug(spatial_views=views)

    def spatial_dir(self, views: int, source: str | None, base: str) -> Path:
        return build_spatial_eval.spatial_eval_dataset_dir(
            spatial_views=views,
            source_name=source,
            base_dir=base,
        )

    def data_dir(self, base: str | None) -> str:
        config = PuzzleData.Config()
        config.base_dir = base
        return str(config.copy_tree().finalize().working_dir)

    def metric_dir(self, base: str | None) -> str:
        config = CanonicalPassK.Config()
        config.base_dir = base
        return str(config.copy_tree().finalize().working_dir)

    def tracker_dir(self, base: str | None) -> str:
        config = SignalDumpTracker.Config()
        config.base_dir = base
        return str(config.copy_tree().finalize().working_dir)

    def data(self, fields: Mapping[str, object]) -> DataLike:
        config = PuzzleData.Config()
        for key, value in fields.items():
            setattr(config, key, value)
        return config.make()

    def batches(
        self,
        root: Path,
        *,
        train: bool,
        offset: int,
        remap: NDArray[np.int64] | None,
    ) -> Stream:
        return PuzzleBatches(
            dataset_dir=root,
            device="cpu",
            batch_size=4,
            rank=0,
            num_replicas=1,
            train=train,
            seed=9,
            puzzle_identifier_offset=offset,
            puzzle_identifier_remap=remap,
        )

    def load(
        self,
        root: Path,
        split: str,
        max_samples: int | None,
    ) -> Mapping[str, object]:
        return load_puzzle_dataset(root, split, max_samples=max_samples)

    def metric(self, fields: Mapping[str, object]) -> MetricLike:
        config = CanonicalPassK.Config()
        for key, value in fields.items():
            setattr(config, key, value)
        return config.make()

    def metric_preds(self, scorer: MetricLike) -> Ballots:
        assert isinstance(scorer, CanonicalPassK)
        return scorer._preds

    def encode(self, preds: Ballots) -> bytes:
        return encode_preds(preds)

    def decode(self, payload: bytes) -> Ballots:
        return decode_preds(payload)

    def payload(
        self,
        *,
        rows: list[tuple[str, str, str, float, float, float, int, int]],
        grids: dict[str, Grid],
        steps: list[tuple[int, int, tuple[float, ...], tuple[int, ...]]],
        pass_ks: tuple[int, ...],
    ) -> tuple[object, ...]:
        return SignalDumpPayload(rows=rows, grids=grids, steps=steps, pass_ks=pass_ks)

    def write_dump(self, payload: object, path: Path, step: int) -> None:
        assert isinstance(payload, SignalDumpPayload)
        write_signal_dump(payload=payload, dump_signals_path=path, global_step=step)

    def tracker(self, working_dir: str) -> TrackerLike:
        config = SignalDumpTracker.Config()
        config.working_dir = working_dir
        return config.make()


WEIGHTS: Final = ({1: 1.0}, {1: 0.5, 2: 0.5}, {2: 1.0}, {1: 1.0, 2: 2.0, 3: 1.0})
"""Scale distributions: identity, mixed, forced, and three-way."""

PROBS: Final = (0.0, 0.5, 1.0)
"""Gate probabilities: never, sampled, and always."""

BUILD_CASES: Final = (
    *(
        BuildCase(seed=seed, prob=prob, weights=weights, num_aug=3)
        for seed in (0, 42)
        for prob in PROBS
        for weights in WEIGHTS[:3]
    ),
    BuildCase(seed=1, prob=1.0, weights={1: 1.0}, num_aug=0),
    BuildCase(seed=2, prob=0.5, weights=WEIGHTS[3], num_aug=40),
    BuildCase(
        seed=3,
        prob=1.0,
        weights={1: 1.0},
        num_aug=2,
        subsets=("training", "evaluation"),
    ),
    BuildCase(
        seed=4,
        prob=1.0,
        weights={1: 1.0},
        num_aug=2,
        test_set_name="__none__",
        with_solutions=False,
    ),
)
"""Every gate/weight branch, no augmentation, retry exhaustion, subset overrides."""


def digest(*values: object) -> str:
    """Hash arrays, tensors, and scalars with dtype and shape, 16 hex chars."""
    h = hashlib.sha256()
    for value in values:
        array = (
            cast("NDArray[np.generic]", value.detach().cpu().numpy())
            if isinstance(value, Tensor)
            else value
        )
        if isinstance(array, np.ndarray):
            typed = cast("NDArray[np.generic]", array)
            h.update(
                f"{typed.dtype.str}{tuple(ListCodec.coerce(list(typed.shape), int))}".encode(),
            )
            h.update(np.ascontiguousarray(typed).tobytes())
        else:
            h.update(repr(array).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def rng_state(rng: np.random.Generator) -> str:
    """Fingerprint a generator's position."""
    return digest(json.dumps(rng.bit_generator.state, sort_keys=True))


def outcome(fn: Callable[[], object]) -> str:
    """Digest ``fn()`` (verbatim if a string), or name its exception."""
    try:
        value = fn()
    except (ValueError, TypeError, KeyError, FileNotFoundError, OverflowError) as err:
        return f"{type(err).__name__}: {err}"
    return value if isinstance(value, str) else digest(value)


def tree_digests(root: Path) -> dict[str, str]:
    """Hash every file under ``root`` except the ensure completion marker."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".ensure_complete"
    }


def write_source(prefix: Path, *, with_solutions: bool = True) -> None:
    """Write a tiny three-subset ARC source at ``prefix``."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for subset in ("training", "evaluation", "concept"):
        puzzles = {
            f"{subset}-{index}": {
                "train": [
                    {"input": [[0, index + 1, 2], [3, 4, 5]], "output": [[6], [7]]},
                    {"input": [[8, 9]], "output": [[index, 0], [2, 3]]},
                    {"input": [[1] * (index + 1)], "output": [[9, 8, 7]]},
                ],
                "test": [{"input": [[1, 2], [3, 4]]}, {"input": [[index, 5]]}],
            }
            for index in range(3)
        }
        # Every view of an all-zero square grid hashes alike, exhausting the retries.
        puzzles[f"{subset}-blank"] = {
            "train": [{"input": [[0, 0], [0, 0]], "output": [[0]]}],
            "test": [{"input": [[0]]}],
        }
        Path(f"{prefix}_{subset}_challenges.json").write_text(json.dumps(puzzles))
        if with_solutions or subset != "concept":
            solutions: dict[str, object] = {
                name: [[[4, 3], [2, 1]], [[5, 5, 5]]] for name in puzzles
            }
            solutions[f"{subset}-blank"] = [[[0]]]
            Path(f"{prefix}_{subset}_solutions.json").write_text(json.dumps(solutions))


def random_grids(seed: int, count: int) -> list[Grid]:
    """Return seeded color grids of random shape up to 6x6."""
    rng = np.random.default_rng(seed)
    return [
        rng.integers(
            0,
            10,
            size=(int(rng.integers(1, 7)), int(rng.integers(1, 7))),
        ).astype(np.uint8)
        for _ in range(count)
    ]


def random_tokens(seed: int, count: int, side: int = 30) -> list[Grid]:
    """Return flat token grids with a color block and noise past it."""
    rng = np.random.default_rng(seed)
    rows: list[Grid] = []
    for _ in range(count):
        grid = rng.integers(0, 12, size=(side, side)).astype(np.uint8)
        nr, nc = int(rng.integers(1, side)), int(rng.integers(1, side))
        grid[:nr, :nc] = rng.integers(2, 12, size=(nr, nc))
        rows.append(grid.flatten())
    return rows


def pairs(grids: list[Grid]) -> Iterable[tuple[Grid, Grid]]:
    """Pair consecutive grids."""
    return zip(grids[::2], grids[1::2], strict=True)


def capture_grid_ops(b: Backend, tmp: Path) -> dict[str, str]:
    """Grid transforms, inverses, hashing, cropping, and error branches."""
    del tmp
    out: dict[str, str] = {}
    for i, grid in enumerate(random_grids(0, 12)):
        for tid in range(8):
            forward = b.dihedral(grid, tid)
            out[f"dihedral/{i}/{tid}"] = digest(forward)
            out[f"inverse_dihedral/{i}/{tid}"] = digest(
                b.inverse_dihedral(forward, tid),
            )
        for k in (1, 2, 3):
            out[f"scale/{i}/{k}"] = digest(b.scale_grid(grid, k))
        out[f"hash/{i}"] = b.grid_hash(grid)
        rows = [
            ListCodec.coerce(row, int)
            for row in ListCodec.coerce(cast(object, grid.tolist()))
        ]
        out[f"to_np/{i}"] = digest(b.to_np(rows))
    rng = np.random.default_rng(3)
    for i, flat in enumerate(random_tokens(1, 12)):
        out[f"crop/{i}"] = digest(b.crop(flat))
        scale = int(rng.integers(1, 4))
        pad_r, pad_c = int(rng.integers(0, 5)), int(rng.integers(0, 5))
        out[f"untranslate/{i}"] = digest(b.untranslate(flat, scale, pad_r, pad_c))
        out[f"untranslate_identity/{i}"] = digest(b.untranslate(flat, 1, 0, 0))
    out["error/dihedral_tid"] = outcome(lambda: b.dihedral(random_grids(0, 1)[0], 8))
    out["crop/small"] = digest(b.crop(np.array([2, 3, 1, 0], dtype=np.uint8)))
    out["crop/empty"] = digest(b.crop(np.zeros(9, dtype=np.uint8)))
    grid = random_grids(5, 1)[0]
    rng = np.random.default_rng(9)
    for i in range(16):
        name, forward = b.color_sample(f"task{i}", rng)
        original, inverse = b.inverse_aug(name)
        out[f"inverse_aug/{i}"] = digest(name, original, inverse(forward(grid)))
    out["inverse_aug/bare"] = digest(
        b.inverse_aug("bare")[0],
        b.inverse_aug("bare")[1](grid),
    )
    zeros4, zeros5 = np.zeros(4, np.uint8), np.zeros(5, np.uint8)
    errors: dict[str, Callable[[], object]] = {
        "square": lambda: b.square_side(10),
        "square_ok": lambda: b.square_side(900),
        "bad_scale": lambda: b.untranslate(zeros4, 0, 0, 0),
        "bad_pad": lambda: b.untranslate(zeros4, 1, -1, 0),
        "nonsquare_identity": lambda: b.untranslate(zeros5, 1, 0, 0),
        "bad_perm": lambda: b.inverse_aug("a|||t1|||0123456788"),
        "short_perm": lambda: b.inverse_aug("a|||t1|||012"),
        "hash_ndim": lambda: b.grid_hash(np.zeros(3, np.uint8)),
        "hash_dtype": lambda: b.grid_hash(cast("Grid", np.zeros((1, 1), np.int32))),
        "to_np_range": lambda: b.to_np([[256]]),
        "to_np_neg": lambda: b.to_np([[-1]]),
        "to_np_wide": lambda: b.to_np([[0] * 31]),
        "to_np_tall": lambda: b.to_np([[0]] * 31),
        "to_np_ndim": lambda: b.to_np(cast("list[list[int]]", cast(object, [1, 2]))),
    }
    out.update({f"error/{k}": outcome(fn) for k, fn in errors.items()})
    return out


SCALE_WEIGHT_INPUTS: Final[tuple[dict[int, float], ...]] = (
    {1: 1.0},
    {2: 1.0, 1: 3.0},
    {3: 0.0, 2: 2.0},
    {1: 0.1, 2: 0.2},
    {},
    {0: 1.0},
    {1: -1.0},
    {1: float("nan")},
    {1: float("inf")},
    {1: 0.0},
)
"""Valid and invalid scale weights."""


def capture_scale_weights(b: Backend, tmp: Path) -> dict[str, str]:
    """Scale-weight normalization, slugs, and CLI parsing."""
    del tmp
    out: dict[str, str] = {"no_train": digest(dict(b.no_train_scale_weights()))}
    for i, weights in enumerate(SCALE_WEIGHT_INPUTS):
        out[f"normalize/{i}"] = outcome(lambda w=weights: b.normalize(w))
        out[f"slug/{i}"] = outcome(lambda w=weights: b.slug(w))
    for i, values in enumerate((["2=0.5", "4=1.0"], ["1=1"], ["2"], ["x=1"])):
        out[f"parse/{i}"] = outcome(lambda v=values: b.parse(v))
    return out


def capture_draws(b: Backend, tmp: Path) -> dict[str, str]:
    """Bernoulli and scale-factor draws with generator positions."""
    del tmp
    out: dict[str, str] = {}
    rng = np.random.default_rng(7)
    for i, prob in enumerate((0.0, 0.3, 1.0, 0.7, -1.0, 2.0) * 3):
        out[f"bernoulli/{i}"] = digest(b.bernoulli(prob, rng), rng_state(rng))
    for i, (inp, target) in enumerate(pairs(random_grids(11, 8))):
        for w, weights in enumerate(WEIGHTS):
            for max_grid in (6, 12, 30):
                out[f"scale_factor/{i}/{w}/{max_grid}"] = digest(
                    b.scale_factor(inp, target, weights, rng, max_grid),
                    rng_state(rng),
                )
    return out


def capture_pack(b: Backend, tmp: Path) -> dict[str, str]:
    """Spatial packing over every gate, weight, grid size, and training branch."""
    del tmp
    out: dict[str, str] = {}
    grids = random_grids(21, 8)
    for seed in (0, 42):
        for prob in PROBS:
            for w, weights in enumerate(WEIGHTS):
                for max_grid in (8, 30):
                    for training in (True, False):
                        rng = np.random.default_rng(seed)
                        rows = [
                            row
                            for inp, target in pairs(grids)
                            for row in b.pack(
                                inp,
                                target,
                                training=training,
                                rng=rng,
                                max_grid=max_grid,
                                prob=prob,
                                weights=weights,
                            )
                        ]
                        key = f"{seed}/{prob}/{w}/{max_grid}/{training}"
                        out[key] = digest(*rows, rng_state(rng))
    out["error/oversized"] = outcome(
        lambda: b.pack(
            np.zeros((4, 1), np.uint8),
            np.zeros((1, 1), np.uint8),
            training=False,
            rng=np.random.default_rng(0),
            max_grid=3,
            prob=1.0,
            weights={1: 1.0},
        ),
    )
    return out


def capture_color_dihedral(b: Backend, tmp: Path) -> dict[str, str]:
    """Color permutation and dihedral sampling with generator positions."""
    del tmp
    out: dict[str, str] = {}
    grid = random_grids(31, 1)[0]
    for seed in (0, 42, 1234):
        rng = np.random.default_rng(seed)
        for i in range(8):
            name, forward = b.color_sample("puzzle", rng)
            out[f"{seed}/{i}"] = digest(name, forward(grid), rng_state(rng))
    return out


def capture_build(b: Backend, tmp: Path) -> dict[str, str]:
    """Whole trees for every builder case, plus the ensure entry and manifest."""
    out: dict[str, str] = {}
    for n, case in enumerate(BUILD_CASES):
        prefix = tmp / f"src{n}" / "arc-agi"
        write_source(prefix, with_solutions=case.with_solutions)
        root = tmp / f"tree{n}"
        b.build(case, prefix, root)
        for rel, value in tree_digests(root).items():
            out[f"{case.key}/{rel}"] = value
    root = tmp / "ensure-tree"
    for _ in range(2):  # The second call is a process-cache hit.
        b.ensure_build(BUILD_CASES[1], tmp / "src1" / "arc-agi", root)
    for rel, value in tree_digests(root).items():
        out[f"ensure/{rel}"] = value
    out["manifest"] = digest(b.manifest())
    for i, (tr, sc, weights) in enumerate(
        ((0.2, 0.2, {2: 1.0}), (1.0, 0.0, {1: 1.0}), (0.5, 0.5, {1: 1.0})),
    ):
        root = tmp / f"policy{i}"
        out[f"policy/{i}"] = outcome(
            lambda r=root, t=tr, s=sc, w=weights: b.build_policy(
                dataset_dir=r,
                prefix=tmp / "src1" / "arc-agi",
                translation_prob=t,
                scale_prob=s,
                weights=w,
            ),
        )
        for rel, value in tree_digests(root).items() if root.exists() else ():
            out[f"policy/{i}/{rel}"] = value
    return out


def base_tree(b: Backend, tmp: Path) -> Path:
    """Build, once, the tree the spatial, loader, and metric captures read."""
    root = tmp / "base"
    if not root.exists():
        prefix = tmp / "src-base" / "arc-agi"
        write_source(prefix)
        b.build(
            BuildCase(seed=0, prob=0.5, weights={1: 1.0, 2: 1.0}, num_aug=3),
            prefix,
            root,
        )
    return root


def spatial_tree(b: Backend, tmp: Path) -> Path:
    """Build, once, a three-view spatial expansion of :func:`base_tree`."""
    target = tmp / "base-spatial"
    if not target.exists():
        b.spatial_build(
            views=3,
            source=base_tree(b, tmp),
            target=target,
            weights={1: 1.0, 2: 1.0},
            seed=42,
        )
    return target


SPATIAL_WEIGHTS: Final = ({2: 1.0}, {1: 1.0, 2: 1.0, 3: 1.0}, {1: 1.0})
"""Spatial-eval scale distributions."""


def capture_spatial_eval(b: Backend, tmp: Path) -> dict[str, str]:
    """Spatial-eval expansions of a built tree, plus the ensure entry."""
    out: dict[str, str] = {}
    source = base_tree(b, tmp)
    for views in (1, 2, 3):
        for w, weights in enumerate(SPATIAL_WEIGHTS):
            for seed in (0, 7):
                target = tmp / f"spatial-{views}-{w}-{seed}"
                b.spatial_build(
                    views=views,
                    source=source,
                    target=target,
                    weights=weights,
                    seed=seed,
                )
                for rel, value in tree_digests(target).items():
                    out[f"{views}/{w}/{seed}/{rel}"] = value
    target = tmp / "spatial-ensure"
    for _ in range(2):  # The second call is a process-cache hit.
        b.spatial_ensure(source=source, views=2, target=target)
    for rel, value in tree_digests(target).items():
        out[f"ensure/{rel}"] = value
    test_only = tmp / "test-only"
    shutil.copytree(source, test_only, ignore=shutil.ignore_patterns("train"))
    edge = test_only / "test"
    # An empty grid never varies, and a full 30x30 grid only fits identity.
    inputs = load_npy(edge / "all__inputs.npy")
    labels = load_npy(edge / "all__labels.npy")
    full = np.full((1, inputs.shape[1]), 5, dtype=inputs.dtype)
    blank = np.zeros((1, inputs.shape[1]), dtype=inputs.dtype)
    np.save(edge / "all__inputs.npy", np.concatenate([inputs, blank, full]))
    np.save(edge / "all__labels.npy", np.concatenate([labels, blank, full]))
    for name, extra in (("puzzle_indices", 2), ("group_indices", 2)):
        values = load_npy(edge / f"all__{name}.npy")
        np.save(
            edge / f"all__{name}.npy",
            np.concatenate(
                [values, values[-1] + np.arange(1, extra + 1, dtype=values.dtype)],
            ),
        )
    ids = load_npy(edge / "all__puzzle_identifiers.npy")
    np.save(edge / "all__puzzle_identifiers.npy", np.concatenate([ids, ids[:2]]))
    for views, weights in ((2, {2: 1.0}), (4, {1: 1.0})):
        b.spatial_build(
            views=views,
            source=test_only,
            target=tmp / f"test-only-spatial-{views}",
            weights=weights,
            seed=0,
        )
        for rel, value in tree_digests(tmp / f"test-only-spatial-{views}").items():
            out[f"test_only/{views}/{rel}"] = value
    b.spatial_ensure(source=test_only, views=2, target=tmp / "test-only-ensure")
    for rel, value in tree_digests(tmp / "test-only-ensure").items():
        out[f"test_only_ensure/{rel}"] = value
    out["error/views"] = outcome(
        lambda: b.spatial_build(
            views=0,
            source=source,
            target=tmp / "never",
            weights={2: 1.0},
            seed=0,
        ),
    )
    return out


AUG_POLICIES: Final = (
    (0.1, 0.1, {2: 1.0}, 1_000, 42),
    (0.2, 0.2, {2: 1.0}, 1_000, 42),
    (1.0, 0.0, {2: 1.0}, 1_000, 42),
    (0.1 + 0.2, 0.5, {1: 1.0, 2: 3.0}, 100, 7),
    (0.5, 0.5, {1: 1.0}, 1_000, 42),
    (1.5, 0.5, {2: 1.0}, 1_000, 42),
    (float("nan"), 0.5, {2: 1.0}, 1_000, 42),
)
"""Scratch trees in use, a dropped-weights slug, repr noise, and invalid policies."""


def capture_paths(b: Backend, tmp: Path) -> dict[str, str]:
    """Directory names and slugs that locate existing trees on scratch."""
    del tmp
    out: dict[str, str] = {}
    for i, (tr, sc, weights, num_aug, seed) in enumerate(AUG_POLICIES):
        policy = AugPolicy(
            translation_prob=tr,
            scale_prob=sc,
            train_scale_weights=weights,
            num_aug=num_aug,
            seed=seed,
        )
        out[f"aug_slug/{i}"] = outcome(lambda p=policy: b.aug_slug(p))
        out[f"aug_template/{i}"] = outcome(lambda p=policy: b.aug_template(p))
        out[f"aug_dir/{i}"] = outcome(lambda p=policy: str(b.aug_dir(p)))
    out["aug_dir/default"] = str(b.aug_dir_default())
    out["default_scale_weights"] = digest(dict(b.default_scale_weights()))
    for views in (0, 1, 2, 5):
        out[f"spatial_slug/{views}"] = outcome(lambda v=views: b.spatial_slug(v))
        for source in (None, "arc1concept-aug-1000-tr0p2-sc0p2-2w1p0-n1000-s42"):
            out[f"spatial_dir/{views}/{source}"] = outcome(
                lambda v=views, s=source: str(b.spatial_dir(v, s, "/opt/scratch")),
            )
    for base in (None, "/opt/scratch"):
        out[f"data_dir/{base}"] = b.data_dir(base)
        out[f"metric_dir/{base}"] = b.metric_dir(base)
        out[f"tracker_dir/{base}"] = b.tracker_dir(base)
    return out


@dataclass(slots=True, kw_only=True, frozen=True)
class LoaderCase:
    """One loader parameterization."""

    batch_size: int
    num_replicas: int
    epochs_per_iter: int
    max_samples: int | None
    seed: int
    spatial: bool

    @property
    def key(self) -> str:
        """Stable case name."""
        return (
            f"b{self.batch_size}-r{self.num_replicas}-e{self.epochs_per_iter}"
            f"-m{self.max_samples}-s{self.seed}-sp{self.spatial}"
        )


LOADER_CASES: Final = (
    LoaderCase(
        batch_size=4,
        num_replicas=1,
        epochs_per_iter=1,
        max_samples=None,
        seed=0,
        spatial=False,
    ),
    LoaderCase(
        batch_size=3,
        num_replicas=2,
        epochs_per_iter=2,
        max_samples=None,
        seed=5,
        spatial=False,
    ),
    LoaderCase(
        batch_size=5,
        num_replicas=1,
        epochs_per_iter=1,
        max_samples=37,
        seed=1,
        spatial=True,
    ),
    LoaderCase(
        batch_size=2,
        num_replicas=3,
        epochs_per_iter=1,
        max_samples=10_000,
        seed=2,
        spatial=True,
    ),
    LoaderCase(
        batch_size=7,
        num_replicas=2,
        epochs_per_iter=3,
        max_samples=11,
        seed=3,
        spatial=False,
    ),
)
"""Widths, sharding, multi-epoch passes, prefix caps, and spatial tags."""

EVAL_CAPS: Final = (
    ("prefix", 13, None, None),
    ("augs", None, 1, None),
    ("augs2", None, 2, None),
    ("augs_big", None, 50, None),
    ("groups", None, None, 3),
    ("groups_big", None, None, 500),
)
"""(name, eval_max_samples, eval_max_augs_per_puzzle, eval_max_examples_per_group)."""


def batch_digest(batch: Batch) -> str:
    """Hash one batch's tensors and valid count."""
    return digest(
        batch["media"],
        batch["label"],
        batch["puzzle_identifiers"],
        batch["spatial_tags"],
        batch["valid_count"],
    )


# One digest per stream, not per batch: the stream name already localizes a mismatch,
# and a golden's size is its key count.
def stream_digests(prefix: str, batches: Iterable[Batch]) -> dict[str, str]:
    """Digest a stream's batches in order, and the count."""
    digests = [batch_digest(batch) for batch in batches]
    return {prefix: digest(*digests), f"{prefix}/count": str(len(digests))}


def loader_fields(root: Path, case: LoaderCase, rank: int) -> dict[str, object]:
    """Config fields for one rank of ``case``."""
    return {
        "working_dir": root,
        "device": "cpu",
        "batch_size": case.batch_size,
        "num_replicas": case.num_replicas,
        "rank": rank,
        "epochs_per_iter": case.epochs_per_iter,
        "max_samples": case.max_samples,
        "seed": case.seed,
    }


def capture_loader(b: Backend, tmp: Path) -> dict[str, str]:
    """Training passes, resume, evaluation scans, caps, and hooks per rank."""
    out: dict[str, str] = {}
    for case in LOADER_CASES:
        root = spatial_tree(b, tmp) if case.spatial else base_tree(b, tmp)
        for rank in range(case.num_replicas):
            key = f"{case.key}/rank{rank}"
            fields = loader_fields(root, case, rank)
            data = b.data({**fields})
            loader = data.train_dataloader()
            out[f"{key}/len"] = str(len(loader))
            for p in range(3):
                out.update(stream_digests(f"{key}/train{p}", loader))
            out[f"{key}/state"] = digest(data.state_dict()["train_iters"])
            out.update(stream_digests(f"{key}/recreated", data.train_dataloader()))
            resumed = b.data({**fields})
            resumed.load_state_dict(data.state_dict())
            out.update(stream_digests(f"{key}/resumed", resumed.train_dataloader()))
            offset = b.data({**fields, "iters_offset": 4})
            out.update(stream_digests(f"{key}/offset", offset.train_dataloader()))
            live = b.data({**fields})
            live_loader = live.train_dataloader()
            live.load_state_dict({"train_iters": 2})
            live.load_state_dict({})
            out.update(stream_digests(f"{key}/live_load", live_loader))
            out[f"{key}/live_state"] = digest(live.state_dict()["train_iters"])
            out.update(stream_digests(f"{key}/eval_fallback", data.eval_dataloader()))
            out.update(stream_digests(f"{key}/eval_full", data.full_eval_dataloader()))
            for name, prefix_cap, augs, groups in EVAL_CAPS:
                stream = b.data(
                    {
                        **fields,
                        "max_samples": None,
                        "eval_batch_size": case.batch_size + 1,
                        "eval_max_samples": prefix_cap,
                        "eval_max_augs_per_puzzle": augs,
                        "eval_max_examples_per_group": groups,
                    },
                ).eval_dataloader()
                out[f"{key}/eval_{name}/len"] = str(len(stream))
                out.update(stream_digests(f"{key}/eval_{name}", stream))
    root = base_tree(b, tmp)
    remap = np.arange(1_000, dtype=np.int64)[::-1].copy()
    for name, offset, table in (("offset", 100, None), ("remap", 0, remap)):
        for train in (True, False):
            stream = b.batches(root, train=train, offset=offset, remap=table)
            out.update(stream_digests(f"hook_{name}_{train}", stream))
    fields = loader_fields(root, LOADER_CASES[0], 0)
    errors: dict[str, Callable[[], object]] = {
        "both_caps": lambda: b.data(
            {**fields, "eval_max_augs_per_puzzle": 1, "eval_max_examples_per_group": 1},
        ).eval_dataloader(),
        "prefix_and_proxy": lambda: b.data(
            {**fields, "max_samples": 3, "eval_max_augs_per_puzzle": 1},
        ).eval_dataloader(),
        "zero_augs": lambda: b.data(
            {**fields, "eval_max_augs_per_puzzle": 0},
        ).eval_dataloader(),
        "zero_groups": lambda: b.data(
            {**fields, "eval_max_examples_per_group": 0},
        ).eval_dataloader(),
        "epochs": lambda: b.data({**fields, "epochs_per_iter": 0}),
        "half_rank": lambda: b.data({**fields, "num_replicas": -1}),
        "rank_range": lambda: b.data({**fields, "rank": 2, "num_replicas": 2}),
        "missing_split": lambda: b.load(tmp / "absent", "train", None),
        "missing_meta": lambda: b.load(
            _corrupt(root, tmp / "no-meta", "meta"),
            "test",
            None,
        ),
        "bad_tags": lambda: b.load(
            _corrupt(root, tmp / "bad-tags", "tags"),
            "test",
            None,
        ),
        "empty_group": lambda: b.load(
            _corrupt(root, tmp / "empty-group", "group"),
            "test",
            None,
        ),
    }
    out.update(
        {
            f"error/{k}": outcome(fn).replace(str(tmp), "<tmp>")
            for k, fn in errors.items()
        },
    )
    zero = _corrupt(root, tmp / "zero-puzzle", "zero")
    for name, augs, groups in (("augs", 1, None), ("groups", None, 2)):
        stream = b.data(
            {
                **loader_fields(zero, LOADER_CASES[0], 0),
                "eval_max_augs_per_puzzle": augs,
                "eval_max_examples_per_group": groups,
            },
        ).eval_dataloader()
        out.update(stream_digests(f"zero_{name}", stream))
    for split in ("train", "test"):
        for cap in (None, 5, 100_000):
            loaded = b.load(spatial_tree(b, tmp), split, cap)
            out[f"load/{split}/{cap}"] = digest(
                *(loaded[k] for k in ("inputs", "labels", "puzzle_indices")),
                *(loaded[k] for k in ("group_indices", "puzzle_identifiers")),
                loaded["spatial_tags"],
                loaded["metadata"],
            )
    return out


def _corrupt(root: Path, target: Path, kind: str) -> Path:
    """Copy ``root`` with one test-split defect: no metadata, bad tags, empty group."""
    if not target.exists():
        shutil.copytree(root, target)
        split = target / "test"
        if kind == "meta":
            (split / "dataset.json").unlink()
        elif kind == "tags":
            np.save(split / "all__spatial_tags.npy", np.ones((1, 3), dtype=np.int32))
        elif kind == "zero":
            # A leading zero-row puzzle, alone in its own group.
            puzzles = load_npy(split / "all__puzzle_indices.npy")
            np.save(
                split / "all__puzzle_indices.npy",
                np.concatenate([puzzles[:1], puzzles]),
            )
            ids = load_npy(split / "all__puzzle_identifiers.npy")
            np.save(
                split / "all__puzzle_identifiers.npy",
                np.concatenate([ids[:1], ids]),
            )
            groups = load_npy(split / "all__group_indices.npy")
            np.save(
                split / "all__group_indices.npy",
                np.concatenate([groups[:1], groups + 1]),
            )
        else:
            groups = load_npy(split / "all__group_indices.npy")
            np.save(
                split / "all__group_indices.npy",
                np.concatenate([groups[:1], groups]),
            )
    return target


def capture_verify(b: Backend, tmp: Path) -> dict[str, str]:
    """Identifier-count verification over an existing tree, match and mismatch."""
    out: dict[str, str] = {}
    root = base_tree(b, tmp)
    count = len(ListCodec.coerce(loads((root / "identifiers.json").read_text())))
    for name, expected in (("match", count), ("mismatch", count + 1)):
        target = tmp / f"verify-{name}"
        if not target.exists():
            shutil.copytree(root, target)
        fields = loader_fields(target, LOADER_CASES[0], 0)
        out[name] = outcome(
            lambda f=fields, e=expected: (
                b.data({**f, "num_puzzle_identifiers": e}).train_dataloader().__len__()
            ),
        ).replace(str(target), "<root>")
    return out


def packed_output(batch: Batch, header: str, index: int) -> Tensor:
    """Build a model output from labels with seeded corruption and coarse logits."""
    label = batch["label"]
    assert isinstance(label, Tensor)
    generator = torch.Generator().manual_seed(index)
    rows = label.shape[0]
    media = batch["media"]
    assert isinstance(media, Tensor)
    # Each view answers correctly, echoes its input, answers nothing, or answers
    # noise: the first three agree across views, so candidates compete for votes.
    correct = torch.where(label == -100, torch.zeros_like(label), label)
    noise = torch.randint(
        0,
        12,
        correct.shape,
        generator=generator,
        dtype=correct.dtype,
    )
    options = torch.stack(
        [correct, media.to(correct.dtype), torch.zeros_like(correct), noise],
    )
    choice = torch.randint(0, 4, (rows,), generator=generator)
    predictions = options[choice, torch.arange(rows)]
    # Coarse logits make exact vote ties, which exercise every tiebreak.
    columns = [(torch.randint(-3, 4, (rows, 1), generator=generator) * 2.0)]
    if header in ("wide", "steps"):
        columns.append(torch.randn(rows, 2, generator=generator))
    if header == "steps":
        columns.append(torch.randint(0, 16, (rows, 2), generator=generator).float())
        columns.append(torch.randn(rows, 2, generator=generator))
        columns.append(torch.randint(0, 2, (rows, 2), generator=generator).float())
    columns.append(predictions.float())
    return torch.cat(columns, dim=1)


def eval_batches(b: Backend, root: Path) -> list[dict[str, object]]:
    """Return every evaluation batch of ``root`` at width 7, so the tail pads."""
    stream = b.data(
        {"working_dir": root, "device": "cpu", "batch_size": 7},
    ).eval_dataloader()
    return [dict(batch) for batch in stream]


def result_digests(prefix: str, results: Mapping[str, object]) -> dict[str, str]:
    """Record each scalar result's exact repr, and the signal payload's digest."""
    out: dict[str, str] = {}
    for name, value in results.items():
        if name != "extras":
            out[f"{prefix}/{name}"] = repr(value)
            continue
        raw = DictCodec.coerce(value)["signal_dump"]
        assert isinstance(raw, tuple)
        payload = ListCodec.coerce(list(cast("tuple[object, ...]", raw)))
        grid_map = DictCodec.coerce(payload[1])
        out[f"{prefix}/payload"] = digest(
            payload[0],
            sorted(grid_map),
            *(grid_map[k] for k in sorted(grid_map)),
            *payload[2:],
        )
    return out


def npz_digest(path: Path) -> str:
    """Hash an ``.npz``'s named arrays, ignoring zip metadata."""
    with cast("NpzFile", np.load(path)) as loaded:
        names = sorted(loaded.files)
        return digest(*(item for name in names for item in (name, loaded[name])))


def load_npy(path: Path) -> NDArray[np.int64]:
    """Load a small index array written by a builder."""
    return cast("NDArray[np.int64]", np.load(path))


def capture_metric(b: Backend, tmp: Path) -> dict[str, str]:
    """pass@K, report-only rankings, signal dumps, state, and the gather codec."""
    out: dict[str, str] = {}
    for spatial in (False, True):
        root = spatial_tree(b, tmp) if spatial else base_tree(b, tmp)
        batches = eval_batches(b, root)
        for header in ("halt", "wide", "steps"):
            outputs = [
                packed_output(batch, header, i) for i, batch in enumerate(batches)
            ]
            for views in ("all", "non_spatial"):
                for cap in (0, 2):
                    key = f"{header}/{views}/{cap}/{spatial}"
                    fields: dict[str, object] = {
                        "working_dir": root,
                        "pass_ks": (1, 2, 3, 100),
                        "spatial_views": views,
                        "max_views_per_input": cap,
                        "per_step_acts": 2 if header == "steps" else 0,
                    }
                    metric = b.metric({**fields})
                    for output, batch in zip(outputs, batches, strict=True):
                        metric.update(output, **batch)
                    results = metric.compute()
                    out.update(result_digests(key, results))
                    restored = b.metric({**fields})
                    restored.load_state_dict(
                        DictCodec.coerce(loads(json.dumps(metric.state_dict()))),
                    )
                    out[f"{key}/restored"] = digest(
                        {k: v for k, v in restored.compute().items() if k != "extras"},
                    )
                    extras = results.get("extras")
                    if extras is not None:
                        path = tmp / f"dump-{key.replace('/', '_')}.npz"
                        b.write_dump(DictCodec.coerce(extras)["signal_dump"], path, 3)
                        out[f"{key}/npz"] = npz_digest(path)
                    if cap == 0 and views == "all":
                        encoded = b.encode(b.metric_preds(metric))
                        out[f"{key}/codec"] = digest(encoded)
                        out[f"{key}/codec_roundtrip"] = str(
                            b.decode(encoded) == b.metric_preds(metric),
                        )
    root = base_tree(b, tmp)
    batches = eval_batches(b, root)
    empty = tmp / "empty-metric"
    empty.mkdir(exist_ok=True)
    (empty / "identifiers.json").write_text(json.dumps(["<blank>"]))
    (empty / "test_puzzles.json").write_text("{}")
    out.update(
        result_digests(
            "empty",
            b.metric({"working_dir": empty, "pass_ks": (1, 2)}).compute(),
        ),
    )
    wide = b.metric({"working_dir": empty, "pass_ks": (1, 2)})
    blank = {**batches[0], "puzzle_identifiers": torch.zeros(7, dtype=torch.int64)}
    wide.update(packed_output(blank, "wide", 0), **blank)
    out.update(result_digests("empty_wide", wide.compute()))
    errors: dict[str, Callable[[], object]] = {
        "width": lambda: b.metric({"working_dir": root}).update(
            torch.zeros(7, 5),
            **batches[0],
        ),
        "steps_width": lambda: b.metric(
            {"working_dir": root, "per_step_acts": 2},
        ).update(
            torch.zeros(7, 903),
            **batches[0],
        ),
        "ident": lambda: b.metric({"working_dir": root}).update(
            packed_output(batches[0], "halt", 0),
            media=batches[0]["media"],
            puzzle_identifiers=torch.full((7,), 10**6),
        ),
    }
    out.update({f"error/{k}": outcome(fn) for k, fn in errors.items()})
    # Views of only the first batch leave most tasks without ballots; no K=1.
    partial = b.metric({"working_dir": root, "pass_ks": (2, 5)})
    for i, batch in enumerate(batches):
        partial.update(packed_output(batch, "wide", i), **batch)
    partial.reset()
    partial.update(packed_output(batches[0], "halt", 0), **batches[0])
    out.update(result_digests("partial", partial.compute()))
    no_tests = tmp / "no-tests-metric"
    shutil.copytree(root, no_tests, ignore=shutil.ignore_patterns("train", "test"))
    puzzles = DictCodec.coerce(loads((no_tests / "test_puzzles.json").read_text()))
    first = next(iter(puzzles))
    puzzles[first] = {**DictCodec.coerce(puzzles[first]), "test": []}
    (no_tests / "test_puzzles.json").write_text(json.dumps(puzzles))
    out.update(
        result_digests("no_tests", b.metric({"working_dir": no_tests}).compute()),
    )
    out["error/codec_hash"] = outcome(lambda: b.encode({"t": {"short": []}}))
    out.update(capture_tracker(b, tmp))
    return out


def capture_tracker(b: Backend, tmp: Path) -> dict[str, str]:
    """Signal-dump tracker routing, path formatting, and payload validation."""
    out: dict[str, str] = {}
    payload = b.payload(
        rows=[("t", "a" * 64, "b" * 64, 0.5, -1.0, 0.25, 2, 3)],
        grids={"b" * 64: np.arange(6, dtype=np.uint8).reshape(2, 3)},
        steps=[(1, 900, (0.5, 0.25), (1, 0))],
        pass_ks=(1, 2),
    )
    folder = tmp / "tracker"
    tracker = b.tracker(str(folder / "signals_{global_step}.npz"))
    tracker.log_metrics({"extras": {"signal_dump": payload}}, 5, prefix="eval/")
    out["tracker/written"] = npz_digest(folder / "signals_5.npz")
    tracker.log_metrics({"extras": {"signal_dump": payload}}, 6, prefix="train/")
    tracker.log_metrics({}, 7, prefix="eval/")
    out["tracker/bad_extras"] = outcome(
        lambda: tracker.log_metrics({"extras": 3}, 8, prefix="eval/"),
    )
    out["tracker/bad_payload"] = outcome(
        lambda: tracker.log_metrics({"extras": {"signal_dump": 3}}, 8, prefix="eval/"),
    )
    b.tracker(str(folder / "{run}_{global_step}.npz")).log_metrics(
        {"extras": {"signal_dump": payload}},
        9,
        prefix="eval/",
    )
    out["tracker/files"] = str(sorted(p.name for p in folder.iterdir()))
    literal = tmp / "literal_{global_step}.npz"
    for _ in range(2):  # The second write overwrites.
        b.write_dump(payload, literal, 1)
    out["tracker/literal"] = npz_digest(literal)
    missing = b.payload(
        rows=[("t", "a" * 64, "c" * 64, 0.5, -1.0, 0.25, 2, 3)],
        grids={},
        steps=[],
        pass_ks=(1,),
    )
    b.write_dump(missing, tmp / "missing.npz", 1)
    out["tracker/missing_grid"] = npz_digest(tmp / "missing.npz")
    return out


CAPTURES: Final[dict[str, Callable[[Backend, Path], dict[str, str]]]] = {
    "grid_ops": capture_grid_ops,
    "scale_weights": capture_scale_weights,
    "draws": capture_draws,
    "pack": capture_pack,
    "color_dihedral": capture_color_dihedral,
    "build": capture_build,
    "spatial_eval": capture_spatial_eval,
    "paths": capture_paths,
    "loader": capture_loader,
    "verify": capture_verify,
    "metric": capture_metric,
}
"""Every capture, keyed as in the golden."""


def mismatches(expected: Mapping[str, str], actual: Mapping[str, str]) -> list[str]:
    """List every differing, missing, or extra key."""
    return [
        f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]


def load_golden() -> dict[str, dict[str, str]]:
    """Read the frozen reference digests."""
    return {
        name: dict(
            line.split("\t", 1)
            for line in zlib.decompress(lines.numpy().tobytes()).decode().split("\n")
            if line
        )
        for name, lines in read_tensors(GOLDEN).items()
    }


def save_golden(table: Mapping[str, Mapping[str, str]]) -> None:
    """Write capture-keyed digests, one zlib tensor of ``key<TAB>value`` lines each.

    Args:
      table: Capture name to its key-to-digest map; no key holds a tab and no
        value a newline.

    """
    write_tensors(
        GOLDEN,
        {
            name: torch.frombuffer(
                bytearray(
                    zlib.compress(
                        "\n".join(
                            f"{k}\t{v}" for k, v in sorted(section.items())
                        ).encode(),
                        level=9,
                    ),
                ),
                dtype=torch.uint8,
            )
            for name, section in sorted(table.items())
        },
    )


@pytest.fixture(scope="module")
def golden() -> dict[str, dict[str, str]]:
    """Decode the frozen reference digests once per module."""
    return load_golden()


@pytest.mark.parametrize("name", sorted(CAPTURES))
def test_port_replays_reference_golden(
    name: str,
    tmp_path: Path,
    golden: dict[str, dict[str, str]],
) -> None:
    """The port reproduces every frozen reference output exactly."""
    diff = mismatches(golden[name], CAPTURES[name](PortBackend(), tmp_path))
    assert not diff, f"{len(diff)} mismatches:\n" + "\n".join(diff[:40])


def _recipe(case: BuildCase) -> ArcAugmentation:
    config = ArcAugmentation.Config()
    config.num_aug = case.num_aug
    config.seed = case.seed
    config.spatial.translation_prob = case.prob
    config.spatial.scale_prob = case.prob
    config.spatial.train_scale_weights = dict(case.weights)
    return config.make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
