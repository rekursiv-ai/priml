"""Check the shared wellbore test fixtures and replay tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import csv

import numpy as np
import pytest
import torch

from priml.baselines.wellbore_geology.data import discover_wells, load_split
from priml.baselines.wellbore_geology.experiments import exp000
from priml.baselines.wellbore_geology.scripts.prepare_data import geographic_map
from priml.baselines.wellbore_geology.testing import (
    BatchedFoldData,
    EvalIdentityDropout,
    ProbeModel,
    capture_training,
    freeze_record,
    numerical_tensors,
    probe_batches,
    probe_recipe,
    tiny_batches,
    tiny_family,
    tiny_model_config,
    tiny_recipe,
    write_study_inputs,
    write_tiny_wells,
    write_training_wells,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import PinnedDropout, assert_tensor_golden, mismatches
from priml.train.epoch_loop import EarlyStopping


if TYPE_CHECKING:
    from priml.baselines.wellbore_geology.custom_types import Batch
    from priml.baselines.wellbore_geology.train_loop import WellboreFoldLoop


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize("rows", [None, 13])
def test_complete_study_fixture_keeps_every_original_target_identifier(
    tmp_path: Path,
    rows: int | None,
) -> None:
    if rows is None:
        write_study_inputs(tmp_path)
    else:
        write_study_inputs(tmp_path, rows=rows)
    expected_rows = 33 if rows is None else rows
    assert discover_wells(tmp_path / "train") == tuple(
        f"well{index}" for index in range(4)
    )
    assert discover_wells(tmp_path / "test") == tuple(
        f"well{index}" for index in range(4)
    )
    wells = load_split(tmp_path / "train")
    assert [len(well.horizontal["MD"]) for well in wells.values()] == [
        expected_rows,
    ] * 4
    with (tmp_path / "sample_submission.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["id", "tvt"]
        records = list(reader)
        assert {record["tvt"] for record in records} == {"0"}
        identifiers = [row["id"] for row in records]
    assert identifiers == [
        f"well{number}_{row}"
        for number in range(3, -1, -1)
        for row in range(expected_rows - 1, 3, -1)
    ]
    assert (tmp_path / "sample_submission.csv").read_bytes() == (
        "id,tvt\n" + "".join(f"{identifier},0\n" for identifier in identifiers)
    ).encode()
    assert (tmp_path / "train_png_typewell_map.csv").read_text() == (
        "well_id,horizontal_avg_X,horizontal_avg_Y\n"
        "well0,0,0\nwell1,1,1\nwell2,40,14\nwell3,41,15\n"
    )


def test_tiny_native_geometry_preserves_all_stages_and_complete_input_shapes(
    tmp_path: Path,
) -> None:
    for particle in (False, True):
        family = tiny_family(particle=particle).finalized()
        assert family.cv.folds == family.cv.clusters == 2
        assert family.cv.repeats == family.training.max_epochs == 1
        assert family.training.dataset.loader.num_workers == 0
        assert (
            family.training.dataset.loader.batch_size
            == family.training.dataset.validation_batch_size
            == 2
        )
        assert family.training.step.model.unet.channels_hidden == (2, 3, 4, 5)
        assert family.training.step.model.unet.backbone.network.blocks_per_stage == (
            1,
            1,
            1,
            1,
        )
        assert family.training.step.model.kernel_size == 1
        stages = family.training.step.model.unet.backbone.network.stage
        assert isinstance(stages, list)
        for stage in stages:
            assert isinstance(stage.block, list)
            assert all(block.conv_dw.kernel_size == 3 for block in stage.block)
        assert family.training.step.model.channels_hidden == 2
        assert family.training.step.model.unet.backbone.checkpoint is None
        if particle:
            assert family.particle is not None
            assert family.particle.particles == 3
            assert family.particle.seeds == 5
            assert family.particle.paths == 2
        else:
            assert family.particle is None
        assert family.training.step.model.reference_columns == 32
        assert family.prediction.expansion.prefix_rows == 4
        assert family.prediction.expansion.target_rows == 28
        assert family.prediction.expansion.downsample == 2
        assert family.training.step.dtype_autocast is None
    write_training_wells(tmp_path / "train", count=2)
    wells = load_split(tmp_path / "train")
    assert list(wells) == ["well0", "well1"]
    assert len(wells["well0"].horizontal["MD"]) == 13
    assert wells["well1"].horizontal["TVT_input"][:4].tolist() == [101, 109, 117, 125]
    assert wells["well1"].horizontal["TVT"][4] == 125.03125
    assert len(wells["well1"].reference_depth) == 600
    write_tiny_wells(tmp_path / "map")
    assert discover_wells(tmp_path / "map") == ("aaaaaaaa", "bbbbbbbb")


def test_tiny_wells_map_to_sorted_six_decimal_centroids(tmp_path: Path) -> None:
    write_tiny_wells(tmp_path)
    assert geographic_map(tmp_path) == (
        b"well_id,horizontal_avg_X,horizontal_avg_Y\n"
        b"aaaaaaaa,3.666667,3.500000\n"
        b"bbbbbbbb,7.333333,7.666667\n"
    )


def test_collect_every_field_and_one_ulp_including_float64() -> None:
    fp32 = torch.tensor([1.0, -2.0, 0.0])
    fp64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
    expected = {
        "fp32": fp32,
        "fp64": fp64,
        "missing": 3,
        "shape": torch.zeros(2, 3),
    }
    actual = {
        "fp32": torch.nextafter(fp32, torch.tensor(float("inf"))),
        "fp64": torch.nextafter(fp64, torch.tensor(float("inf"))),
        "extra": 4,
        "shape": torch.zeros(3, 2),
    }
    report = mismatches(
        numerical_tensors(expected),
        numerical_tensors(actual),
        ulps=True,
    )
    assert set(report) == {
        "missing missing",
        "unexpected extra",
        "fp32: 3/3 differ; max_ulp=1",
        "fp64: 2/2 differ; max_ulp=1",
        "shape: torch.float32[3, 2] vs torch.float32[2, 3]",
    }


def test_compare_bits_retains_nan_payload_and_signed_zero() -> None:
    expected = torch.tensor([0, -2_147_483_648, 2_143_289_345], dtype=torch.int32)
    actual = expected.clone()
    actual[0] = -2_147_483_648
    actual[2] += 1
    report = mismatches(
        numerical_tensors({"value": expected.view(torch.float32)}),
        numerical_tensors({"value": actual.view(torch.float32)}),
        ulps=True,
    )
    assert len(report) == 1
    assert report[0] == "value: 2/3 differ; max_ulp=nan"
    assert (
        mismatches(
            numerical_tensors({"value": expected.view(torch.float32)}),
            numerical_tensors({"value": expected.view(torch.float32).clone()}),
            ulps=True,
        )
        == []
    )


def test_records_copy_every_element_and_survive_weights_only_load(
    tmp_path: Path,
) -> None:
    value = torch.arange(30, dtype=torch.float64).reshape(2, 3, 5)
    record = freeze_record({"nested": [{"full": value}], "empty": (), "none": None})
    value.add_(100)
    saved = record["nested/0/full"]
    assert isinstance(saved, torch.Tensor)
    assert torch.equal(saved, value - 100)
    path = tmp_path / "record.pt"
    torch.save(record, path)
    assert (
        mismatches(
            numerical_tensors(record),
            numerical_tensors(
                cast("dict[str, object]", torch.load(path, weights_only=True)),
            ),
            ulps=True,
        )
        == []
    )
    assert record["empty"] == "tuple:0"
    assert record["none"] is None


def test_records_preserve_numpy_precision_scalars_and_empty_mappings() -> None:
    record = freeze_record(
        {
            "array": np.asarray([1, 2], dtype=np.float64),
            "scalar": np.int64(7),
            "scalar_float": np.float32(0.5),
            "map": {},
        },
    )
    array = record["array"]
    assert isinstance(array, torch.Tensor)
    assert array.dtype is torch.float64
    assert record["scalar"] == 7
    scalar_float = record["scalar_float"]
    assert isinstance(scalar_float, torch.Tensor)
    assert scalar_float.dtype is torch.float32
    assert record["map"] == "dict:0"
    with pytest.raises(
        TypeError,
        match=r"^Unsupported numerical trace value at : <class 'object'>\.$",
    ):
        freeze_record(object())


def test_record_paths_strip_only_the_leading_separator() -> None:
    assert freeze_record([5]) == {"@sequence": "list:1", "0": 5}
    assert freeze_record({"Xs": [5], "Xa": 1}) == {
        "Xs/@sequence": "list:1",
        "Xs/0": 5,
        "Xa": 1,
    }


def test_scalar_types_float_bits_and_nonfloating_tensor_differences() -> None:
    report = mismatches(
        numerical_tensors(
            {
                "scalar": 1,
                "zero": -0.0,
                "dtype": torch.zeros(2),
                "int": torch.tensor([3, 4]),
            },
        ),
        numerical_tensors(
            {
                "scalar": 1.0,
                "zero": 0.0,
                "dtype": torch.zeros(2, dtype=torch.float64),
                "int": torch.tensor([3, 5]),
            },
        ),
        ulps=True,
    )
    assert len(report) == 4
    assert (
        mismatches(numerical_tensors({"x": 3}), numerical_tensors({"x": 3}), ulps=True)
        == []
    )


def test_complete_native_record_retains_every_tensor_scalar_and_container() -> None:
    vector = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    values = numerical_tensors(
        {
            "value": vector,
            "rate": 0.000_000_000_000_000_123,
            "empty": {},
            "unset": None,
            "enabled": True,
            "count": 7,
            "name": "native",
        },
    )
    assert torch.equal(values["value"], vector)
    assert values["value"].data_ptr() != vector.data_ptr()
    assert values["rate"].dtype == torch.float64
    assert values["rate"].item() == 0.000_000_000_000_000_123
    assert values["empty/@str"].tolist() == list(b"dict:0")
    assert values["unset/@none"].numel() == 0
    assert values["enabled"].dtype == torch.bool
    assert values["enabled"].item() is True
    assert values["count"].dtype == torch.int64
    assert values["count"].item() == 7
    assert values["name/@str"].tolist() == list(b"native")


def test_nexus_family_training_golden() -> None:
    with host_agnostic_numerics():
        record = capture_training(recipe=_family_recipe("0803_V2"))
    assert_tensor_golden(_CWD / "testdata" / "train_step.pt", record)


@pytest.mark.parametrize(
    "name",
    ["0719_V1", "0724_V1", "0729_V3", "0801_V1", "0801_V2"],
)
def test_each_family_trains_like_the_nexus_except_for_dropout(name: str) -> None:
    """Every other setting a three-update replay reads matches the nexus family's.

    Early stopping is excluded: no family may stop inside three epochs.
    """
    nexus, other = _family_recipe("0803_V2"), _family_recipe(name)
    nexus.step.model.dropout = other.step.model.dropout
    nexus, other = nexus.finalized(), other.finalized()
    for recipe in (nexus, other):
        assert isinstance(recipe.early_stopping, EarlyStopping.Config)
        assert recipe.early_stopping.min_epochs > recipe.max_epochs
        recipe.early_stopping = None
    assert repr(nexus) == repr(other)


def _family_recipe(name: str) -> WellboreFoldLoop.Config:
    return tiny_recipe(
        next(family for family in exp000().families if family.name == name),
    )


def test_pinned_dropout_draws_at_each_decoder_site_and_never_in_evaluation() -> None:
    model = tiny_model_config(dropout=0.05).make()
    media = tiny_batches()[0]["media"]
    assert isinstance(media, torch.Tensor)
    pins = EvalIdentityDropout(model)
    with pins:
        model(media)
    blocks = [f"up_blocks.{index}.block" for index in range(3)]
    assert pins.sites == [
        f"unet.{block}.{part}"
        for block in ("stem", *blocks, "full_up_block.block", "out")
        for part in ("conv1.drop", "drop")
    ] + ["unet_gr_rmse_head.3"]
    model.eval()
    with torch.inference_mode():
        expected, _ = model(media)
        with pins:
            actual, _ = model(media)
    assert pins.calls == 13
    assert torch.equal(actual, expected)


def test_lowest_random_pin_preserves_scaling_gradients_and_call_order() -> None:
    value = torch.ones(2, 3, 4, 5, requires_grad=True)
    pins = PinnedDropout(torch.nn.Identity())
    with pins:
        actual = torch.nn.functional.dropout2d(value, p=0.8, training=True)
        second = torch.nn.functional.dropout2d(value, p=0.8, training=True)
        actual.sum().backward()
    expected = (
        torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.float32)[
            ...,
            None,
            None,
        ].expand_as(value)
        * 5
    )
    assert torch.equal(actual, expected)
    assert value.grad is not None
    assert torch.equal(value.grad, expected)
    assert torch.equal(second, torch.zeros_like(second))
    assert pins.calls == 2
    assert pins.sites == ["functional", "functional"]
    with pins, pytest.raises(ValueError, match=r"aten\.rand"):
        torch.rand(2, 3)


def test_complete_record_preserves_float64_and_structural_leaves() -> None:
    values = numerical_tensors(
        {"rate": 0.000_000_000_000_000_123, "empty": {}, "unset": None},
    )
    assert values["rate"].dtype == torch.float64
    assert values["rate"].item() == 0.000_000_000_000_000_123
    assert values["empty/@str"].tolist() == list(b"dict:0")
    assert values["unset/@none"].numel() == 0


def test_probe_retains_unequal_complete_samples_and_all_loss_inputs() -> None:
    batches = [cast("Batch", value) for value in probe_batches()]
    assert [len(batch["meta"]) for batch in batches] == [3, 3, 2]
    for batch in batches:
        count = len(batch["meta"])
        assert batch["media"].shape == (count, 2)
        assert batch["label"].shape == (count, 4)
        assert batch["typewell_target_probs"].shape == (count, 4, 6)
        assert batch["aux_targets"]["GR_penalty_error"].shape == (count, 4, 6)
        assert batch["target_mask"].dtype is torch.bool


def test_probe_makes_fresh_runtime_objects_from_named_constructors() -> None:
    config = probe_recipe()
    assert isinstance(config.dataset, BatchedFoldData.Config)
    first, second = config.dataset.make(), config.dataset.make()
    assert first.train is not second.train
    assert first.validation is not None
    assert len(first.train) == len(first.validation) == 3
    assert first.train_dataloader() is first.train
    assert first.eval_dataloader() is first.validation
    assert first.state_dict() == {"timer_epoch": {"global_count": 0, "global_sec": 0.0}}


def test_probe_step_runs_eager_without_inductor() -> None:
    step = probe_recipe().step.make()
    assert step._compile_fn is None


def test_probe_model_maps_each_sample_to_four_depths_and_six_logits() -> None:
    model = ProbeModel.Config().make()
    depths, extra = model(torch.zeros(3, 2))
    assert depths.shape == (3, 4)
    assert extra["alignment_logits"].shape == (3, 4, 6)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
