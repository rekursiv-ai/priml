"""Check the shared wellbore test fixtures and replay tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import csv

from torch import nn

import numpy as np
import pytest
import torch

from priml.baselines.wellbore_geology.data import discover_wells, load_split
from priml.baselines.wellbore_geology.experiments import exp000
from priml.baselines.wellbore_geology.inference import GeographicGate, WellBlend
from priml.baselines.wellbore_geology.recipes import family_0719_v1
from priml.baselines.wellbore_geology.scripts.prepare_data import geographic_map
from priml.baselines.wellbore_geology.testing import (
    BatchedFoldData,
    ProbeModel,
    capture_training,
    freeze_record,
    historical_cosine_state,
    numerical_tensors,
    probe_batches,
    probe_recipe,
    raise_first_alignment_weight_one_ulp,
    tiny_family,
    tiny_model_config,
    tiny_predictions,
    tiny_recipe,
    write_study_inputs,
    write_tiny_wells,
    write_training_wells,
)
from priml.optimizers.lr_schedule import RecursiveCosine
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    PinnedDropout,
    mismatches,
    read_tensors,
    write_tensors,
)
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
    assert list(tiny_predictions()) == [
        "0719_V1",
        "0724_V1",
        "0729_V3",
        "0801_V1",
        "0801_V2",
        "0803_V2",
    ]
    write_tiny_wells(tmp_path / "map")
    assert discover_wells(tmp_path / "map") == ("aaaaaaaa", "bbbbbbbb")


def test_complete_component_record_requires_no_original_source(tmp_path: Path) -> None:
    safe = {
        "geo_nbr_distance": 100.0,
        "geo_nbr_distance_q10": 50.0,
        "geo_nbr_path_alignment": 0.99,
        "geo_radial_extrap_score": 1.0,
        "geo_prefix_weight_ratio": 0.1,
    }
    cases = [dict(safe)]
    for name, value in (
        ("geo_nbr_distance", 2459),
        ("geo_nbr_distance_q10", 1426),
        ("geo_nbr_path_alignment", 0.96),
        ("geo_radial_extrap_score", 1.45),
        ("geo_prefix_weight_ratio", 0.315),
        ("geo_nbr_distance", float("nan")),
    ):
        cases.append({**safe, name: value})
    predictions = tiny_predictions()
    blend = WellBlend.Config().make()
    gate = GeographicGate.Config().make()
    ensemble = np.concatenate(
        [blend(predictions, geographic=gate(case)) for case in cases],
    )
    write_tiny_wells(tmp_path)
    expected = read_tensors(_CWD / "testdata/components.pt")
    assert (
        mismatches(
            expected,
            {
                "ensemble": torch.from_numpy(ensemble.astype(np.float64)),
                "geographic_csv": torch.tensor(
                    list(geographic_map(tmp_path)),
                    dtype=torch.uint8,
                ),
            },
        )
        == []
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
    with pytest.raises(TypeError, match="Unsupported numerical trace"):
        freeze_record(object())


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


# Each released family's three-update record equals its golden twin's byte for byte,
# so one replay per distinct record covers all six.
_TWINS: Final = {
    "0719_V1": ("0724_V1", "0729_V3"),
    "0801_V1": ("0801_V2", "0803_V2"),
}


@pytest.mark.parametrize("name", list(_TWINS))
def test_complete_native_training_replays_original_golden(name: str) -> None:
    assert _replay(name, mutation="none")[1] == []


@pytest.mark.parametrize(
    ("name", "twin"),
    [(name, twin) for name, twins in _TWINS.items() for twin in twins],
)
def test_a_family_trains_like_the_twin_its_record_shares(name: str, twin: str) -> None:
    """Every numerical setting a three-update replay reads matches the twin's.

    Early stopping is excluded: no family may stop inside three epochs.
    """
    replayed, other = (_family_recipe(family).finalized() for family in (name, twin))
    assert _CWD.joinpath("testdata", name.lower(), "train_step.pt").read_bytes() == (
        _CWD.joinpath("testdata", twin.lower(), "train_step.pt").read_bytes()
    )
    for recipe in (replayed, other):
        assert isinstance(recipe.early_stopping, EarlyStopping.Config)
        assert recipe.early_stopping.min_epochs > recipe.max_epochs
        recipe.early_stopping = None
    assert repr(replayed) == repr(other)


def _family_recipe(name: str) -> WellboreFoldLoop.Config:
    return tiny_recipe(
        next(family for family in exp000().families if family.name == name),
    )


@pytest.mark.parametrize("mutation", ["one_ulp", "draw_order"])
def test_a_single_perturbation_breaks_the_original_replay(mutation: str) -> None:
    actual, report = _replay("0719_V1", mutation=mutation)
    assert actual["pinned_bernoulli_calls"].item() > 0
    assert report
    assert any("model/" in line or "optimizer/" in line for line in report)


def _replay(name: str, mutation: str) -> tuple[dict[str, torch.Tensor], list[str]]:
    family = next(family for family in exp000().families if family.name == name)
    record = _CWD / "testdata" / name.lower() / "train_step.pt"
    with host_agnostic_numerics():
        actual = capture_training(
            initial_record=record,
            recipe=tiny_recipe(family),
            one_ulp=mutation == "one_ulp",
            draw_offset=int(mutation == "draw_order"),
        )
    wanted = {
        key: value
        for key, value in read_tensors(record).items()
        if not key.startswith("initial/")
    }
    return actual, mismatches(wanted, actual)


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


@pytest.mark.parametrize("horizon", [3, 5, 11])
def test_every_historical_cosine_state_matches_complete_torch_state(
    horizon: int,
) -> None:
    optimizer = torch.optim.AdamW(
        nn.Linear(2, 3).parameters(),
        lr=0.0002,
        weight_decay=0.02,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=horizon,
        eta_min=5e-7,
    )
    schedule = RecursiveCosine.Config(T_max=horizon, eta_min=5e-7).make()
    for completed in range(horizon + 1):
        rate = schedule.learning_rate(0.0002, step=completed)
        assert rate == cast("float", optimizer.param_groups[0]["lr"])
        actual = historical_cosine_state(
            horizon=horizon,
            eta_min=5e-7,
            initial_lrs=[0.0002],
            last_lrs=[rate],
            completed_steps=completed,
        )
        assert (
            mismatches(
                numerical_tensors(scheduler.state_dict()),
                numerical_tensors(actual),
                ulps=True,
            )
            == []
        )
        optimizer.step()
        scheduler.step()


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


def test_capture_training_loads_the_initial_record_strictly(tmp_path: Path) -> None:
    source = _CWD / "testdata/0719_v1/train_step.pt"
    record = {
        name: value
        for name, value in read_tensors(source).items()
        if name != "initial/alignment_head.weight"
    }
    write_tensors(tmp_path / "missing.pt", record)
    with pytest.raises(RuntimeError, match=r"proj_out\.weight"):
        capture_training(
            initial_record=tmp_path / "missing.pt",
            recipe=tiny_recipe(family_0719_v1()),
            one_ulp=False,
            draw_offset=0,
        )


def test_one_ulp_perturbation_moves_only_the_first_alignment_weight() -> None:
    model = tiny_model_config(dropout=0).make()
    before = model.get_parameter("proj_out.weight").detach().clone()
    raise_first_alignment_weight_one_ulp(model)
    after = model.get_parameter("proj_out.weight").detach()
    delta = after.view(torch.int32).flatten() - before.view(torch.int32).flatten()
    assert delta[0].item() == 1
    assert torch.count_nonzero(delta[1:]) == 0


def test_probe_model_maps_each_sample_to_four_depths_and_six_logits() -> None:
    model = ProbeModel.Config().make()
    depths, extra = model(torch.zeros(3, 2))
    assert depths.shape == (3, 4)
    assert extra["alignment_logits"].shape == (3, 4, 6)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
