"""Materialize the dataset-owned ARC augmentation recipe from local source JSON."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import hashlib
import json
import tempfile

from numpy.typing import NDArray

import numpy as np

from priml.baselines.arcagi1.augmentation import (
    ArcAugmentation,
    arc_grid_to_np,
    grid_hash,
)
from priml.lib.custom_json import DictCodec, ListCodec, loads


if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(slots=True, kw_only=True)
class _Puzzle:
    name: str
    examples: list[tuple[NDArray[np.uint8], NDArray[np.uint8]]]


def build_arc_dataset(
    *,
    target_dir: Path,
    input_file_prefix: str,
    augmentation: ArcAugmentation,
) -> None:
    """Build a fresh ARC tree without overwriting an existing dataset.

    Args:
      target_dir: Destination for the complete dataset tree.
      input_file_prefix: Local prefix of the pinned challenge/solution JSON files.
      augmentation: Dataset-owned augmentation recipe and transforms.

    Raises:
      FileExistsError: The destination already contains data.

    """
    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"Dataset destination is not empty: {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=target_dir.parent,
        prefix=f".{target_dir.name}-",
    ) as temporary:
        staging = Path(temporary) / "dataset"
        _build_arc_dataset(
            input_file_prefix=input_file_prefix,
            output_dir=staging,
            augmentation=augmentation,
        )
        staging.replace(target_dir)


def _build_arc_dataset(
    *,
    input_file_prefix: str,
    output_dir: Path,
    augmentation: ArcAugmentation,
) -> None:
    rng = np.random.default_rng(augmentation.config.seed)
    results: dict[str, dict[str, list[list[_Puzzle]]]] = {}
    test_puzzles: dict[str, object] = {}
    train_dest = ("train", "all")
    test_dest = ("test", "all")
    for subset_name in ("training", "evaluation", "concept"):
        raw_puzzles = DictCodec.coerce(
            loads(
                Path(f"{input_file_prefix}_{subset_name}_challenges.json").read_text(),
            ),
        )
        puzzles = {
            pid: {
                key: [
                    DictCodec.coerce(example) for example in ListCodec.coerce(examples)
                ]
                for key, examples in DictCodec.coerce(puzzle).items()
            }
            for pid, puzzle in raw_puzzles.items()
        }
        solutions = Path(f"{input_file_prefix}_{subset_name}_solutions.json")
        if solutions.is_file():
            raw_solutions = DictCodec.coerce(loads(solutions.read_text()))
            for pid, puzzle in puzzles.items():
                for index, grid in enumerate(ListCodec.coerce(raw_solutions[pid])):
                    puzzle["test"][index]["output"] = _int_grid(grid)
        else:
            for puzzle in puzzles.values():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])
        puzzle_list = list(puzzles.items())
        rng.shuffle(puzzle_list)
        for name, puzzle in puzzle_list:
            is_test = subset_name == "evaluation"
            if is_test:
                test_puzzles[name] = puzzle
            _convert_puzzle(
                results,
                name,
                puzzle,
                dest_map={
                    "train": train_dest,
                    "test": test_dest if is_test else train_dest,
                },
                augmentation=augmentation,
                rng=rng,
            )

    id_map: dict[str, int] = {}
    for split in results.values():
        for subset in split.values():
            for group in subset:
                for puzzle in group:
                    if puzzle.name not in id_map:
                        id_map[puzzle.name] = len(id_map) + 1
    for split_name, split in results.items():
        _write_split(
            output_dir,
            split_name,
            split,
            id_map,
            augmentation=augmentation,
            rng=rng,
        )
    inverse_id_map = {value: key for key, value in id_map.items()}
    identifiers = [inverse_id_map.get(i, "<blank>") for i in range(len(id_map) + 1)]
    (output_dir / "identifiers.json").write_text(json.dumps(identifiers))
    (output_dir / "test_puzzles.json").write_text(json.dumps(test_puzzles))


def _write_split(
    out_path: Path,
    split_name: str,
    split: dict[str, list[list[_Puzzle]]],
    id_map: dict[str, int],
    *,
    augmentation: ArcAugmentation,
    rng: np.random.Generator,
) -> None:
    split_path = out_path / split_name
    split_path.mkdir(parents=True)
    total_examples = total_puzzles = total_groups = 0
    subset_names: list[str] = []
    for subset_name, subset in split.items():
        subset_names.append(subset_name)
        all_inputs: list[NDArray[np.uint8]] = []
        all_labels: list[NDArray[np.uint8]] = []
        all_puzzle_ids: list[int] = []
        puzzle_indices = [0]
        group_indices = [0]
        puzzle_count = example_count = 0
        for group in subset:
            for puzzle in group:
                no_aug_idx = int(rng.integers(0, len(puzzle.examples)))
                for index, (inp, out) in enumerate(puzzle.examples):
                    inp_seq, out_seq = augmentation.spatial.pack(
                        inp,
                        out,
                        training=split_name == "train" and index != no_aug_idx,
                        rng=rng,
                    )
                    all_inputs.append(inp_seq)
                    all_labels.append(out_seq)
                    example_count += 1
                    total_examples += 1
                puzzle_indices.append(example_count)
                all_puzzle_ids.append(id_map[puzzle.name])
                puzzle_count += 1
                total_puzzles += 1
            group_indices.append(puzzle_count)
            total_groups += 1
        np.save(split_path / f"{subset_name}__inputs.npy", np.stack(all_inputs))
        np.save(split_path / f"{subset_name}__labels.npy", np.stack(all_labels))
        np.save(
            split_path / f"{subset_name}__puzzle_indices.npy",
            np.array(puzzle_indices, dtype=np.int32),
        )
        np.save(
            split_path / f"{subset_name}__group_indices.npy",
            np.array(group_indices, dtype=np.int32),
        )
        np.save(
            split_path / f"{subset_name}__puzzle_identifiers.npy",
            np.array(all_puzzle_ids, dtype=np.int32),
        )
    (split_path / "dataset.json").write_text(
        json.dumps(
            {
                "pad_id": 0,
                "ignore_label_id": 0,
                "blank_identifier_id": 0,
                "vocab_size": 12,
                "seq_len": augmentation.spatial.config.max_grid**2,
                "num_puzzle_identifiers": len(id_map) + 1,
                "total_groups": total_groups,
                "mean_puzzle_examples": total_examples / total_puzzles
                if total_puzzles
                else 0.0,
                "total_puzzles": total_puzzles,
                "sets": subset_names,
            },
        ),
    )


def _convert_puzzle(
    results: dict[str, dict[str, list[list[_Puzzle]]]],
    name: str,
    puzzle: dict[str, list[dict[str, object]]],
    *,
    dest_map: dict[str, tuple[str, str]],
    augmentation: ArcAugmentation,
    rng: np.random.Generator,
) -> None:
    destinations = set(dest_map.values())
    converted = {
        destination: _Puzzle(name=name, examples=[]) for destination in destinations
    }
    for example_type, examples in puzzle.items():
        converted[dest_map[example_type]].examples.extend(
            [
                (
                    arc_grid_to_np(
                        _int_grid(example["input"]),
                        max_grid=augmentation.spatial.config.max_grid,
                    ),
                    arc_grid_to_np(
                        _int_grid(example["output"]),
                        max_grid=augmentation.spatial.config.max_grid,
                    ),
                )
                for example in examples
            ],
        )
    group = [converted]
    if augmentation.config.num_aug > 0:
        hashes = {_puzzle_group_hash(converted)}
        for _ in range(
            augmentation.config.retries_factor * augmentation.config.num_aug,
        ):
            aug_name, apply = augmentation.transform.sample(name, rng=rng)
            augmented = {
                destination: _Puzzle(
                    name=aug_name,
                    examples=[(apply(inp), apply(out)) for inp, out in value.examples],
                )
                for destination, value in converted.items()
            }
            digest = _puzzle_group_hash(augmented)
            if digest not in hashes:
                hashes.add(digest)
                group.append(augmented)
            if len(group) >= augmentation.config.num_aug + 1:
                break
    for destination in destinations:
        split, subset = destination
        results.setdefault(split, {}).setdefault(subset, []).append(
            [view[destination] for view in group],
        )


def _puzzle_group_hash(group: Mapping[tuple[str, str], _Puzzle]) -> str:
    hashes = sorted(
        f"{grid_hash(inp)}|{grid_hash(out)}"
        for puzzle in group.values()
        for inp, out in puzzle.examples
    )
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def _int_grid(value: object) -> list[list[int]]:
    return [ListCodec.coerce(row, int) for row in ListCodec.coerce(value)]
