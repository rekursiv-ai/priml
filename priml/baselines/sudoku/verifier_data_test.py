from __future__ import annotations

from typing import TYPE_CHECKING, cast

import json
import re

from torch import Tensor

import numpy as np
import pytest
import torch

from priml.baselines.sudoku.runner import file_sha256
from priml.baselines.sudoku.testing import write_dataset
from priml.baselines.sudoku.verifier_data import (
    VerifierData,
    _corrupt_hamming,
    _HarvestShardEntry,
    _load_harvest,
    _swap_cells,
)
from priml.data.arrays import load_array


if TYPE_CHECKING:
    from pathlib import Path


def _write_harvest_shard(directory: Path, name: str, *, groups: list[int]) -> str:
    """Write one harvest shard and return its sha256."""
    rows = len(groups)
    grid = np.full((rows, 81), 2, dtype=np.uint8)
    np.savez(
        directory / name,
        base_group_id=np.array(groups, dtype=np.int64),
        original=grid,
        candidate=grid,
        flat_view_id=np.arange(rows, dtype=np.int64),
    )
    return file_sha256(directory / name)


def _write_manifest(
    manifest: Path,
    shards: list[dict[str, object]],
    **overrides: object,
) -> None:
    fields: dict[str, object] = {
        "schema_version": 1,
        "groups_per_shard": 2,
        "shards": shards,
        "shard_count": 2,
        "row_count": 5,
    }
    manifest.write_text(json.dumps(fields | overrides))


def test_load_harvest_rejects_a_malformed_manifest(tmp_path: Path) -> None:
    cpu = torch.device("cpu")
    manifest = tmp_path / "manifest.json"
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        _load_harvest(tmp_path, device=cpu)
    entry = {"file": "a/b", "rows": 1, "sha256": "0"}
    counts = {"groups_per_shard": 2, "shard_count": 1, "row_count": 1}
    cases: tuple[tuple[str, str], ...] = (
        ("{", "invalid harvest manifest"),
        ("[]", r"expected object for _HarvestManifest, got \[\]"),
        (
            json.dumps({"schema_version": 2, "shards": [], **counts}),
            r"first at \['schema_version'\]: cannot read 2 as typing\.Literal\[1\]",
        ),
        (
            json.dumps({"schema_version": 1, "shards": {}, **counts}),
            # The export rewrites the module path, so the type names itself here.
            r"first at \['shards'\]: cannot read \{\} as "
            + re.escape(repr(list[_HarvestShardEntry])),
        ),
        (
            json.dumps({"schema_version": 1, "shards": [3], **counts}),
            r"first at \['shards', 0\]: expected object for _HarvestShardEntry, got 3",
        ),
        (
            json.dumps({"schema_version": 1, "shards": [entry], **counts}),
            "invalid shard name",
        ),
    )
    for text, message in cases:
        manifest.write_text(text)
        with pytest.raises(ValueError, match=message):
            _load_harvest(tmp_path, device=cpu)


def test_load_harvest_binds_shards_to_the_manifest(tmp_path: Path) -> None:
    cpu = torch.device("cpu")
    manifest = tmp_path / "manifest.json"
    first = _write_harvest_shard(tmp_path, "shard-00000.npz", groups=[0, 1])
    second = _write_harvest_shard(tmp_path, "shard-00001.npz", groups=[1, 0, 1])
    good: list[dict[str, object]] = [
        {"file": "shard-00000.npz", "sha256": first, "rows": 2},
        {"file": "shard-00001.npz", "sha256": second, "rows": 3},
    ]
    broken: tuple[tuple[list[dict[str, object]], dict[str, object], str], ...] = (
        (good[:1], {}, "shard set mismatch"),
        (good[::-1], {}, "canonical shard order"),
        ([{**good[0], "sha256": "0"}, good[1]], {}, "digest mismatch"),
        ([{**good[0], "rows": 4}, good[1]], {}, "row count mismatch"),
        (good, {"row_count": 6}, "aggregate count mismatch"),
        (good, {"groups_per_shard": 1}, "base_group_id must lie"),
    )
    for shards, overrides, message in broken:
        _write_manifest(manifest, shards, **overrides)
        with pytest.raises(ValueError, match=message):
            _load_harvest(tmp_path, device=cpu)
    _write_manifest(manifest, good)
    originals, candidates, views, groups = _load_harvest(tmp_path, device=cpu)
    assert originals.shape == candidates.shape == (5, 81)
    assert views.tolist() == [0, 1, 0, 1, 2]
    assert groups.tolist() == [0, 1, 3, 2, 3]


def _write_shard(
    shard: Path,
    *,
    original: np.ndarray,
    candidate: np.ndarray,
    flat_views: list[int],
    groups: list[int],
) -> None:
    np.savez(
        shard,
        base_group_id=np.array(groups, dtype=np.int64),
        original=original,
        candidate=candidate,
        flat_view_id=np.array(flat_views, dtype=np.int64),
    )


def _bound_harvest(
    data: Path,
    harvest: Path,
    *,
    flat_views: list[int],
    groups: list[int],
) -> Path:
    """Write a one-shard harvest whose originals are the named train rows."""
    inputs = load_array(data / "train" / "all__inputs.npy", np.generic)
    labels = load_array(data / "train" / "all__labels.npy", np.generic)
    harvest.mkdir(parents=True)
    shard = harvest / "shard-00000.npz"
    _write_shard(
        shard,
        original=inputs[flat_views],
        candidate=labels[flat_views],
        flat_views=flat_views,
        groups=groups,
    )
    digest = file_sha256(shard)
    entry = {"file": shard.name, "sha256": digest, "rows": len(groups)}
    manifest = {
        "schema_version": 1,
        "groups_per_shard": max(groups) + 1,
        "shards": [entry],
        "shard_count": 1,
        "row_count": len(groups),
    }
    (harvest / "manifest.json").write_text(json.dumps(manifest))
    return harvest


def _verifier_data_config(root: Path, harvest: Path) -> VerifierData.Config:
    cfg = VerifierData.Config()
    cfg.working_dir = root / "data"
    cfg.harvest_dir = harvest
    cfg.device = "cpu"
    cfg.batch_size = 6
    cfg.contradiction_fraction = 0.2
    cfg.train_group_end = 2
    return cfg


def test_verifier_data_rejects_a_broken_harvest_binding(tmp_path: Path) -> None:
    write_dataset(tmp_path / "data")
    harvest = _bound_harvest(
        tmp_path / "data",
        tmp_path / "harvest",
        flat_views=[0, 1, 2, 3],
        groups=[0, 1, 2, 3],
    )
    cfg = _verifier_data_config(tmp_path, harvest)
    cfg.real_fraction = 0.9
    with pytest.raises(ValueError, match="batch fractions"):
        cfg.make()
    cfg = _verifier_data_config(tmp_path, harvest)
    cfg.train_group_end = -1
    with pytest.raises(ValueError, match="train_group_end must be >= 0"):
        cfg.make()
    cfg = _verifier_data_config(tmp_path, harvest)
    cfg.train_group_end = 0
    with pytest.raises(ValueError, match="no harvest rows"):
        cfg.make()
    inputs = load_array(tmp_path / "data" / "train" / "all__inputs.npy", np.generic)
    shard = harvest / "shard-00000.npz"
    for flat_views, message in (
        ([0, 1, 2, 9], "flat_view_id exceeds"),
        ([3, 1, 2, 0], "pairing contract"),
    ):
        _write_shard(
            shard,
            original=inputs[[0, 1, 2, 3]],
            candidate=inputs[[0, 1, 2, 3]],
            flat_views=flat_views,
            groups=[0, 1, 2, 3],
        )
        _rebind_digest(harvest)
        with pytest.raises(ValueError, match=message):
            _verifier_data_config(tmp_path, harvest).make()


def _rebind_digest(harvest: Path) -> None:
    manifest = harvest / "manifest.json"
    fields = cast(
        "dict[str, list[dict[str, object]]]",
        json.loads(manifest.read_text()),
    )
    fields["shards"][0]["sha256"] = file_sha256(
        harvest / "shard-00000.npz",
    )
    manifest.write_text(json.dumps(fields))


def test_verifier_data_streams_seeded_epochs_with_contradictions(
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path / "data")
    harvest = _bound_harvest(
        tmp_path / "data",
        tmp_path / "harvest",
        flat_views=[0, 1, 2, 3],
        groups=[0, 1, 2, 3],
    )
    cfg = _verifier_data_config(tmp_path, harvest)
    cfg.steps_per_epoch = 2
    data = cfg.make()
    batch = data.generate_batch(torch.Generator().manual_seed(cfg.seed))
    assert batch["puzzle"].shape == batch["candidate"].shape == (6, 81)
    first, second = data.train_dataloader(), data.train_dataloader()
    assert len(first) == len(second) == 2
    assert torch.equal(next(iter(first))["candidate"], batch["candidate"])
    assert not torch.equal(next(iter(second))["candidate"], batch["candidate"])


def test_contradiction_rows_follow_two_seeded_draws_and_align_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = object.__new__(VerifierData)
    data.device = torch.device("meta")
    data.train_inputs = torch.tensor([[2, 1], [3, 1], [4, 1]])
    data.train_labels = torch.tensor([[5, 6], [7, 8], [9, 10]])
    generator = torch.Generator().manual_seed(91)
    expected_generator = torch.Generator().manual_seed(91)
    row_indices = torch.randint(3, (4,), generator=expected_generator)
    sibling_indices = torch.randint(3, (4,), generator=expected_generator)
    original_randint = torch.randint
    devices: list[object] = []

    def record_randint(*args: object, **kwargs: object) -> Tensor:
        devices.append(kwargs.get("device"))
        return original_randint(
            cast("int", args[0]),
            cast("tuple[int, ...]", args[1]),
            device="cpu",
            generator=cast("torch.Generator", kwargs["generator"]),
        )

    monkeypatch.setattr(torch, "randint", record_randint)
    puzzle, candidate, solution = data._contradiction_rows(4, generator=generator)
    assert devices == [data.device, data.device]
    assert torch.equal(puzzle, data.train_inputs[row_indices].long())
    assert torch.equal(candidate, data.train_labels[sibling_indices].long())
    assert torch.equal(solution, data.train_labels[row_indices].long())


def test_hamming_corruption_changes_only_non_given_cells() -> None:
    solution = (torch.arange(2 * 9).reshape(2, 9) % 9 + 2).long()
    given = torch.zeros(2, 9, dtype=torch.bool)
    given[:, :4] = True
    # Row 1 asks for more cells than its 5 blanks: the givens must still hold.
    k = torch.tensor([2, 7])
    corrupted = _corrupt_hamming(
        solution,
        given,
        k=k,
        generator=torch.Generator().manual_seed(0),
    )
    changed = corrupted != solution
    assert changed.sum(dim=-1).tolist() == [2, 5]
    assert not changed[given].any()
    assert corrupted.min() >= 2
    assert corrupted.max() <= 10


def test_swaps_with_no_pairs_return_the_candidate_untouched() -> None:
    candidate = torch.arange(2 * 9).reshape(2, 9)
    out = _swap_cells(
        candidate,
        torch.zeros(2, 9, dtype=torch.bool),
        pairs=torch.zeros(2, dtype=torch.int64),
        generator=torch.Generator().manual_seed(0),
    )
    assert out is candidate


def test_swaps_of_an_empty_batch_return_the_candidate_itself() -> None:
    candidate = torch.zeros(0, 9, dtype=torch.int64)
    out = _swap_cells(
        candidate,
        torch.zeros(0, 9, dtype=torch.bool),
        pairs=torch.tensor([2]),
        generator=torch.Generator().manual_seed(0),
    )
    assert out is candidate


@pytest.mark.parametrize("steps", [0, -1])
def test_verifier_data_rejects_a_nonpositive_epoch(steps: int) -> None:
    """``VerifierFit.run`` would otherwise loop forever on an empty epoch."""
    with pytest.raises(ValueError, match="steps_per_epoch must be >= 1"):
        VerifierData.Config(steps_per_epoch=steps).make()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
