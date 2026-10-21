"""Checks for the dreams module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

import hashlib
import math
import pickle

from torch import Tensor
from treekle import from_plain

import numpy as np
import pytest
import torch

from priml.baselines.casmi2026 import dreams
from priml.baselines.casmi2026.dreams import (
    BankCheckpoint,
    DreamsStore,
    MainBank,
    clamp_count,
    collate_main,
    dreams_attention_fp32,
    keep_count,
    merged_adduct,
    modal_adduct,
    safe_tail,
)
from priml.baselines.casmi2026.model import DreamAttention, DreamsFP, FPNet
from priml.baselines.casmi2026.model_test import set_block_heads, tiny_dreams
from priml.baselines.casmi2026.peaks import prepare_peaks
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from priml.baselines.casmi2026.custom_types import SpectrumRequest


def bank_spectra() -> list[SpectrumRequest]:
    """Retain empty, unknown, NaN-energy, cap and both-polarity source inputs."""
    return [
        {
            "mz": [0, 25, 45, 60, 102, 103],
            "it": [1, 4, 2, 1, 0.004, 9],
            "prec": 100.0,
            "adduct": "[M+H]+",
            "mode": 1,
            "ce": 10.0,
        },
        {
            "mz": [25.003, 45, 70],
            "it": [2, 5, 1],
            "prec": 120.0,
            "adduct": "[M+Na]+",
            "mode": 1,
            "ce": None,
        },
        {
            "mz": [20, 35],
            "it": [3, 1],
            "prec": 80.0,
            "adduct": "unknown",
            "mode": -1,
            "ce": float("nan"),
        },
        {"mz": [], "it": [], "prec": 90.0, "adduct": "[M-H]-", "mode": -1, "ce": 50.0},
    ]


def source_checkpoint(*, index: int) -> dict[str, object]:
    """Wrap a tiny model, seeded by its index, in the released checkpoint envelope."""
    with torch.random.fork_rng(devices=[]):
        _ = torch.manual_seed(index)
        model = (tiny_released_dreams() if index == 2 else tiny_fpnet()).make()
    checkpoint: dict[str, object] = {
        "model": model.released_state(),
        "nbits": 7,
        "d": 8,
        "layers": 1,
        "heads": 2,
    }
    if index == 2:
        checkpoint.update(
            arch="dreams_fp",
            fcfg={
                "nbits": 7,
                "n_peaks": 4,
                "neg_shift": True,
                "cond_target": "prec",
                "head_hidden": 3,
                "head_dropout": 0.1,
                "n_adduct": 60,
                "n_elem": 10,
            },
            bcfg={
                "d_fourier": 6,
                "d_peak": 2,
                "n_layers": 1,
                "n_heads": 2,
                "ff_fourier_depth": 2,
                "ff_fourier_d": 2,
                "ff_peak_depth": 1,
                "dropout": 0.1,
                "att_dropout": 0.1,
                "ff_dropout": 0.1,
                "residual_dropout": 0.1,
                "graphormer": "key",
                "pad_mode": "faithful",
                "fourier_fp32": True,
                "fourier_chunk": 4096,
                "fourier_ckpt": True,
                "attn_fp32": True,
                "ssl_heads": False,
            },
        )
    return checkpoint


def tiny_fpnet() -> FPNet.Config:
    """Declare the tiny FPNet that the source checkpoints at indices 0 and 1 hold."""
    config = FPNet.Config()
    config.channels_out = 7
    config.channels_hidden = 8
    config.num_layers = 1
    set_block_heads(config.block, 2)
    config.head.channels_hidden = 3
    config.head_elements.channels_hidden = 2
    return config


def tiny_released_dreams() -> DreamsFP.Config:
    """Declare the tiny DreamsFP that the source checkpoint at index 2 holds."""
    config = tiny_dreams()
    config.num_elements = 10
    config.adduct.channels_in = 60
    return config


def checkpoint_config(path: Path, *, index: int) -> BankCheckpoint.Config:
    """Pin a saved source checkpoint and declare the architecture it holds.

    Args:
      path: A checkpoint written from ``source_checkpoint``.
      index: Its source index; 2 holds the DreamsFP, the others the tiny FPNet.

    Returns:
      config: The checkpoint with its digest and literal architecture.

    """
    config = BankCheckpoint.Config()
    config.path = path
    config.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    config.model = tiny_released_dreams() if index == 2 else tiny_fpnet()
    return config


def save_bank(directory: Path, indices: Sequence[int]) -> MainBank.Config:
    """Write source checkpoints and declare a bank over them in the given order."""
    config = MainBank.Config()
    for index in indices:
        path = directory / f"model{index}.pt"
        torch.save(source_checkpoint(index=index), path)
        config.checkpoints.append(checkpoint_config(path, index=index))
    return config


def test_bank_keeps_or_averages_the_model_axis_of_single_and_merged_items(
    tmp_path: Path,
) -> None:
    config = save_bank(tmp_path, [0, 1, 2])
    config.batch_size = 2
    bank = config.make()
    spectra = bank_spectra()
    single, merged = bank.items(spectra, merged_counts=[0, 12, 2, 1])
    assert [item.merged_count for item in single] == [1, 8, 2, 1]
    # One merged item per present polarity, positive first; counts cap at eight.
    assert [(item.mode, item.merged_count) for item in merged] == [(1, 8), (-1, 3)]
    assert [(item.precursor, item.energy) for item in merged] == [(110, 10), (85, 50)]
    mz, _, pad, *_, count, _ = collate_main(single)
    # Peaks above the precursor by more than two drop; empty items keep one slot.
    assert mz.shape == pad.shape == (4, 4)
    assert pad.tolist()[3] == [False, True, True, True]
    assert count.tolist() == [0.25, 2.0, 0.5, 0.25]
    for items in (single, merged):
        logits, elements = bank.predict(items, each=True)
        assert logits.shape == (3, len(items), 7)
        assert elements.shape == (3, len(items), 10)
        mean = bank.predict(items)
        assert np.array_equal(mean[0], torch.from_numpy(logits).mean(0).numpy())
        assert np.array_equal(mean[1], torch.from_numpy(elements).mean(0).numpy())
    views = bank.views(spectra, merged_counts=[0, 12, 2, 1])
    assert np.array_equal(views.single, bank.predict(single, each=True)[0])
    assert np.array_equal(views.merged_elements, bank.predict(merged, each=True)[1])
    assert views.modes.tolist() == [1, -1]
    assert [array.shape for array in bank.predict([], each=True)] == [
        (3, 0, 7),
        (3, 0, 10),
    ]
    assert [array.shape for array in bank.predict([])] == [(0, 7), (0, 10)]


def test_bank_metadata_adduct_and_validation_boundaries(tmp_path: Path) -> None:
    assert merged_adduct(["z", "a"]) == "a"
    assert merged_adduct(["[M+Na]+", "[M+H]+", "[M+Na]+", "[M+H]+ "]) == "[M+Na]+"
    assert modal_adduct(["[M+H]+", "[M+Na]+", "[M+H]+"]) == "[M+H]+"
    assert (clamp_count(0), clamp_count(3)) == (1, 3)
    assert (keep_count(0), keep_count(3)) == (0, 3)
    spectra = bank_spectra()
    spectra[0]["mode"] = "positive"
    spectra[1]["mode"] = "negative"
    config = save_bank(tmp_path, [0])
    config.merged_adduct = modal_adduct
    config.single_count = keep_count
    bank = config.make()
    single, merged = bank.items(spectra, merged_counts=[0, 12, 2, 1])
    assert single[0].merged_count == 0
    assert len(merged) == 2
    assert len(bank.items([])[1]) == 0
    config = MainBank.Config()
    with pytest.raises(ValueError, match="at least one"):
        config.make()
    config.checkpoints = [BankCheckpoint.Config()]
    config.batch_size = 0
    with pytest.raises(ValueError, match="positive"):
        config.make()


def test_bank_checkpoints_must_share_one_fingerprint_width(tmp_path: Path) -> None:
    config = save_bank(tmp_path, [0])
    narrow = source_checkpoint(index=0)
    state = from_plain(narrow["model"], dict[str, Tensor])
    for name in ("head.3.weight", "head.3.bias"):
        state[name] = state[name][:6]
    narrow["model"], narrow["nbits"] = state, 6
    path = tmp_path / "narrow.pt"
    torch.save(narrow, path)
    checkpoint = checkpoint_config(path, index=0)
    assert isinstance(checkpoint.model, FPNet.Config)
    checkpoint.model.channels_out = 6
    config.checkpoints.append(checkpoint)
    with pytest.raises(ValueError, match="disagree on fingerprint width"):
        _ = config.make()


def test_checkpoint_pin_geometry_and_architecture_are_verified(tmp_path: Path) -> None:
    checkpoint = source_checkpoint(index=2)
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    pinned = checkpoint_config(path, index=2)
    wrong = BankCheckpoint.Config()
    wrong.path, wrong.sha256 = path, "wrong"
    with pytest.raises(RuntimeError, match="hashes to"):
        _ = wrong.make()
    wrong.sha256 = ""
    with pytest.raises(ValueError, match="requires its SHA256 pin"):
        _ = wrong.make()
    wrong.sha256, wrong.model = pinned.sha256, tiny_fpnet()
    with pytest.raises(ValueError, match="Unknown neural checkpoint architecture"):
        _ = wrong.make()
    declared = pinned.model
    assert isinstance(declared, DreamsFP.Config)
    declared.max_peaks += 1
    with pytest.raises(ValueError, match="differ from the declared"):
        _ = pinned.make()
    declared.max_peaks -= 1
    assert pinned.make().model is not None
    for name, value in (
        ("arch", "unknown"),
        ("fcfg", {"cond_target": "all"}),
        ("bcfg", {"graphormer": "direct"}),
    ):
        original = checkpoint[name]
        checkpoint[name] = value
        torch.save(checkpoint, path)
        pinned.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match=r"Unknown|precursor|faithful"):
            _ = pinned.make()
        checkpoint[name] = original
    fpnet = source_checkpoint(index=0)
    torch.save(fpnet, path)
    declared_fpnet = checkpoint_config(path, index=0)
    assert isinstance(declared_fpnet.model, FPNet.Config)
    declared_fpnet.model.channels_hidden = 16
    with pytest.raises(ValueError, match="geometry"):
        _ = declared_fpnet.make()


def test_checkpoint_path_resolves_beneath_its_base_dir(tmp_path: Path) -> None:
    config = BankCheckpoint.Config()
    config.base_dir, config.path = tmp_path, "inputs/model.pt"
    assert config.copy_tree().finalize().path == tmp_path / "inputs/model.pt"


def test_main_preparation_filters_before_normalizing() -> None:
    mass, intensity = prepare_peaks(
        np.array([-1, 0, 12, 10, 102, 103], np.float64),
        np.array([100, 200, 1, 4, 0.004, 1_000], np.float64),
        precursor=100,
        max_peaks=160,
        floor=0.001,
    )
    assert np.array_equal(mass, np.array([10, 12, 102], np.float32))
    assert np.array_equal(
        intensity,
        np.sqrt(np.array([1, 0.25, 0.001])).astype(np.float32),
    )


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [None, torch.float16, torch.bfloat16],
)
def test_dreams_precision_recipe_preserves_training_only_under_bfloat16(
    trained: bool,
    dtype: torch.dtype | None,
    tmp_path: Path,
) -> None:
    checkpoint = source_checkpoint(index=2)
    bcfg = from_plain(checkpoint["bcfg"], dict[str, object])
    bcfg["attn_fp32"] = trained
    checkpoint["bcfg"] = bcfg
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    expected = trained if dtype == torch.bfloat16 else True
    assert dreams_attention_fp32(dtype or torch.float32, trained=trained) is expected
    config = checkpoint_config(path, index=2)
    config.dtype_autocast = dtype
    assert isinstance(config.model, DreamsFP.Config)
    block = cast(object, config.model.backbone.encoder.block)
    assert isinstance(block, TransformerBlock.Config)
    attention = block.attn
    assert isinstance(attention, DreamAttention.Config)
    attention.precision.dtype = torch.float32 if expected else None
    assert config.make().model is not None
    attention.precision.dtype = None if expected else torch.float32
    with pytest.raises(ValueError, match="attention precision"):
        _ = config.make()


def test_neural_items_consume_source_ce_n_without_an_override(tmp_path: Path) -> None:
    spectra = bank_spectra()
    for spectrum, count in zip(spectra, (0, 12, 2, 1), strict=True):
        spectrum["ce_n"] = count
    bank = save_bank(tmp_path, [0]).make()
    explicit = bank.items(bank_spectra(), merged_counts=[0, 12, 2, 1])
    for recorded, overridden in zip(bank.items(spectra), explicit, strict=True):
        for left, right in zip(
            collate_main(recorded),
            collate_main(overridden),
            strict=True,
        ):
            assert torch.equal(left, right)
    assert [item.merged_count for item in bank.items(bank_spectra())[0]] == [1] * 4


LIMIT: Final = 25


def test_store_selects_the_relative_window_in_distance_order(tmp_path: Path) -> None:
    np.save(
        tmp_path / "dreams_np_mass.npy",
        np.array([100.0005, 99.9999, 100.002, 101.0], np.float32),
    )
    np.save(tmp_path / "dreams_np_smiles.npy", np.array(["a", "b", "c", "d"]))
    store = DreamsStore.load(tmp_path)
    assert store is not None
    selection = store.select(100.0)
    assert store.index is not None
    assert store.index.mass.dtype == np.float64
    assert store.index.order.tolist() == [1, 0, 2, 3]
    # The 10 ppm window around 100 spans [99.999, 100.001].
    assert (selection.low, selection.high) == (0, 2)
    assert selection.rows.tolist() == [1, 0]
    assert selection.smiles == ("b", "a")
    result, audit = safe_tail(
        "m",
        ["c"],
        [],
        target=100.0,
        identity=identity,
        limit=2,
        dreams=store,
    )
    assert result == ["c", "b"]
    assert audit is not None
    assert audit["dreams_added"] == 1
    # The original keeps its first rank, so nothing counts as promoted.
    assert audit["promoted_original"] == 0


def identity(value: str) -> str | None:
    """Inject complete known identities independently of any chemistry runtime."""
    return None if value in {"invalid", ""} else value.lower()


def test_safe_tail_retains_all_ten_audit_fields_and_original_identity_ranks() -> None:
    assert safe_tail(
        "m",
        ["invalid", " A ", "a", "B", ""],
        ["B", "C"],
        target=100.0,
        identity=identity,
        limit=LIMIT,
    ) == (
        ["A", "B", "C"],
        {
            "molecule_id": "m",
            "original_count": 4,
            "original_unique": 2,
            "invalid_count": 1,
            "duplicate_count": 1,
            "revised_count": 3,
            "ranked_added": 1,
            "dreams_added": 0,
            "promoted_original": 2,
            "changed": True,
        },
    )
    missing: dict[str, str] = {}
    assert safe_tail(
        "inactive",
        [" A ", "", "invalid"],
        ["absent"],
        target=100.0,
        identity=missing.__getitem__,
        limit=LIMIT,
        processed=False,
    ) == ([" A ", "", "invalid"], None)
    with pytest.raises(KeyError, match="absent"):
        safe_tail(
            "missing",
            ["absent"],
            [],
            target=100.0,
            identity=missing.__getitem__,
            limit=LIMIT,
        )
    with pytest.raises(AssertionError, match="No valid candidates for empty"):
        safe_tail(
            "empty",
            ["invalid"],
            [],
            target=100.0,
            identity=identity,
            limit=LIMIT,
        )
    with pytest.raises(AssertionError):
        safe_tail(
            "overflow",
            [f"id/{index}" for index in range(26)],
            [],
            target=100.0,
            identity=identity,
            limit=LIMIT,
        )


def test_missing_store_is_optional_and_present_invalid_store_is_deferred(
    tmp_path: Path,
) -> None:
    assert DreamsStore.load(tmp_path) is None
    np.save(tmp_path / "dreams_np_smiles.npy", np.array(["C", "O"]))
    assert DreamsStore.load(tmp_path) is None
    (tmp_path / "dreams_np_mass.npy").write_bytes(b"malformed")
    store = DreamsStore.load(tmp_path)
    assert store is not None
    for target in (-1.0, 0.0, math.nan, math.inf):
        candidates, audit = safe_tail(
            "skip",
            ["C", "O"],
            [],
            target=target,
            identity=identity,
            limit=LIMIT,
            dreams=store,
        )
        assert candidates == ["C", "O"]
        assert audit is not None
        assert audit["dreams_added"] == 0
    full = [f"id/{index}" for index in range(25)]
    assert (
        safe_tail(
            "full",
            full,
            [],
            target=100.0,
            identity=identity,
            limit=LIMIT,
            dreams=store,
        )[0]
        == full
    )
    with pytest.raises(ValueError, match=r"pickle|pickled|magic"):
        store.candidates(100.0)


def test_positive_subdalton_targets_use_the_optional_store(tmp_path: Path) -> None:
    np.save(tmp_path / "dreams_np_mass.npy", np.array([0.5]))
    np.save(tmp_path / "dreams_np_smiles.npy", np.array(["new"]))
    result, audit = safe_tail(
        "small",
        ["original"],
        [],
        target=0.5,
        identity=identity,
        limit=LIMIT,
        dreams=DreamsStore.load(tmp_path),
    )
    assert result == ["original", "new"]
    assert audit is not None
    assert audit["dreams_added"] == 1


def test_tail_assertions_reject_extension_contract_violations() -> None:
    # These source assertions guard safe_extend's output, whose normal path caps
    # and deduplicates before returning. Inject a faulty boundary to reach them.
    for candidates in (
        [f"id/{index}" for index in range(26)],
        ["id/0", "id/0"],
    ):
        with (
            patch.object(dreams, "safe_extend", return_value=candidates),
            pytest.raises(AssertionError),
        ):
            safe_tail(
                "broken",
                ["id/0"],
                [],
                target=100.0,
                identity=identity,
                limit=LIMIT,
            )


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_numpy_object_versions_use_only_restricted_scientific_reconstruction(
    tmp_path: Path,
    version: tuple[int, int],
) -> None:
    np.save(tmp_path / "dreams_np_mass.npy", np.array([100.0, 100.0], np.float32))
    with (tmp_path / "dreams_np_smiles.npy").open("wb") as handle:
        np.lib.format.write_array(
            handle,
            np.array([b"C", "O"], dtype=object),
            version=version,
        )
    store = DreamsStore.load(tmp_path)
    assert store is not None
    assert store.candidates(100.0) == ("C", "O")
    assert store.index is not None
    assert store.index.mass.dtype == np.float64
    with (tmp_path / "dreams_np_smiles.npy").open("wb") as handle:
        np.save(handle, np.array([range(3), "O"], dtype=object))
    forbidden = DreamsStore.load(tmp_path)
    assert forbidden is not None
    with pytest.raises(pickle.UnpicklingError, match=r"builtins\.range"):
        forbidden.candidates(100.0)


def test_present_array_failures_propagate_and_only_selected_bytes_decode(
    tmp_path: Path,
) -> None:
    np.save(tmp_path / "dreams_np_mass.npy", np.array([100.0, 200.0]))
    store = DreamsStore.load(tmp_path)
    assert store is not None
    with pytest.raises(FileNotFoundError):
        store.candidates(100.0)
    (tmp_path / "dreams_np_smiles.npy").write_bytes(b"malformed")
    with pytest.raises(ValueError, match=r"pickle|pickled|magic"):
        store.candidates(100.0)
    np.save(tmp_path / "dreams_np_smiles.npy", np.array([b"C", b"\xff"]))
    assert store.candidates(100.0) == ("C",)
    with pytest.raises(UnicodeDecodeError):
        store.candidates(200.0)
    np.save(tmp_path / "dreams_np_smiles.npy", np.array(["C", "O", "N"]))
    mismatched = DreamsStore.load(tmp_path)
    assert mismatched is not None
    with pytest.raises(AssertionError):
        mismatched.candidates(100.0)
    with (tmp_path / "dreams_np_smiles.npy").open("wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle,
            {"descr": "|O", "fortran_order": False, "shape": (2,)},
        )
        handle.write(pickle.dumps(["C", "O"]))
    malformed = DreamsStore.load(tmp_path)
    assert malformed is not None
    with pytest.raises(TypeError, match="NumPy array"):
        malformed.candidates(100.0)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
