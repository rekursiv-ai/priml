"""Tests for the pass@K voting metric."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast, override

import json
import logging

from torch import Tensor
from treekle import json_loads

import numpy as np
import pytest
import torch

from priml.baselines.arcagi import metric
from priml.baselines.arcagi.augmentation import (
    ArcSpec,
    ColorDihedral,
    SpatialAugmentation,
    grid_hash,
)
from priml.baselines.arcagi.metric import (
    CanonicalPassK,
    PerOutputPass,
    StrictPass,
    TaskScore,
    _floats,
    _hash_bytes,
    _json_grid,
    _model_output_header_width,
    _read_u32,
    _shape,
    _uint8_rows,
    _votes_times_max_q,
    _votes_times_mean_q,
    decode_preds,
    encode_preds,
)
from priml.baselines.arcagi.signal_dump import (
    SignalDumpPayload,
)


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from collections.abc import Mapping


def _packed(predictions: Tensor, halt: Tensor) -> Tensor:
    """Pack a halt logit ahead of the predicted tokens, as the step emits."""
    return torch.cat([halt.reshape(-1, 1), predictions.float()], dim=-1)


def test_canonical_scoring_without_pass_at_one(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(2,)).make()

    candidate.update(logits, **batch)

    assert candidate.compute()["pass@2"] == 0.75


def test_canonical_votes_restore_augmented_views_and_cap_by_confidence(
    tmp_path: Path,
) -> None:
    """One wrong high-confidence view outranks a correct view of the same task."""
    source = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    answer = np.array([[5]], dtype=np.uint8)
    wrong = np.array([[6]], dtype=np.uint8)
    rng = np.random.default_rng(7)
    transform_config = ColorDihedral.Config()
    transform_config.separator = ArcSpec().puzzle_id_separator
    name, transform = transform_config.make().sample("task", rng=rng)
    pack_config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(pack_config.spec, ArcSpec)
    pack_config.spec.max_grid = 3
    pack = pack_config.make()
    original_media, original_answer = pack.pack(
        source,
        answer,
        training=False,
        rng=rng,
    )
    augmented_media, augmented_wrong = pack.pack(
        transform(source),
        transform(wrong),
        training=False,
        rng=rng,
    )
    (tmp_path / "identifiers.json").write_text(
        json.dumps(["<blank>", "task", name]),
    )
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {
                "task": {
                    "test": [{"input": source.tolist(), "output": answer.tolist()}],
                },
                "unseen": {"test": [{"input": [[1]], "output": [[2]]}]},
            },
        ),
    )
    predictions = torch.tensor(np.stack([original_answer, augmented_wrong]))
    batch = {
        "media": torch.tensor(np.stack([original_media, augmented_media])),
        "puzzle_identifiers": torch.tensor([1, 2]),
        "spatial_tags": torch.tensor([[1, 0, 0], [1, 0, 0]]),
    }
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    metric.update(_packed(predictions, torch.tensor([-4.0, 4.0])), **batch)
    assert _pass_at(metric.compute()) == {"pass@1": 0.0, "pass@2": 0.5}

    capped = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 2),
        max_views_per_input=1,
    ).make()
    capped.update(_packed(predictions, torch.tensor([-4.0, 4.0])), **batch)
    assert _pass_at(capped.compute()) == {"pass@1": 0.0, "pass@2": 0.0}

    # SpatialAugmentation packs grids into its square max_grid.
    translated_media = np.pad(
        augmented_media.reshape(3, 3)[:2, :2],
        ((1, 0), (1, 0)),
    ).reshape(-1)
    # SpatialAugmentation packs grids into its square max_grid.
    translated_wrong = np.pad(
        augmented_wrong.reshape(3, 3)[:2, :2],
        ((1, 0), (1, 0)),
    ).reshape(-1)
    tagged_batch = {
        "media": torch.tensor(np.stack([translated_media, original_media])),
        "puzzle_identifiers": torch.tensor([2, 1]),
        "spatial_tags": torch.tensor([[1, 1, 1], [1, 0, 0]]),
    }
    tagged_predictions = torch.tensor(
        np.stack([translated_wrong, original_answer]),
    )
    non_spatial = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 2),
        spatial_views="non_spatial",
    ).make()
    non_spatial.update(
        _packed(tagged_predictions, torch.tensor([4.0, -4.0])),
        **tagged_batch,
    )
    assert _pass_at(non_spatial.compute()) == {"pass@1": 0.5, "pass@2": 0.5}
    all_views = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    all_views.update(
        _packed(tagged_predictions, torch.tensor([4.0, -4.0])),
        **tagged_batch,
    )
    assert _pass_at(all_views.compute()) == {"pass@1": 0.0, "pass@2": 0.5}


def test_canonical_votes_use_configured_transform_separator(tmp_path: Path) -> None:
    source = np.array([[1, 2]], dtype=np.uint8)
    answer = np.array([[8]], dtype=np.uint8)
    transform_config = ColorDihedral.Config(separator="::", transforms=(1,))
    name, transform = transform_config.make().sample(
        "task",
        rng=np.random.default_rng(7),
    )
    pack_config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(pack_config.spec, ArcSpec)
    pack_config.spec.max_grid = 3
    pack = pack_config.make()
    media, prediction = pack.pack(
        transform(source),
        transform(answer),
        training=False,
        rng=np.random.default_rng(7),
    )
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", name]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {"task": {"test": [{"input": source.tolist(), "output": answer.tolist()}]}},
        ),
    )
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        transform=transform_config,
    ).make()
    metric.update(
        _packed(torch.from_numpy(prediction)[None, :], torch.zeros(1)),
        media=torch.from_numpy(media)[None, :],
        puzzle_identifiers=torch.tensor([1]),
    )
    assert _pass_at(metric.compute()) == {"pass@1": 1.0}


@pytest.mark.parametrize("max_views_per_input", [0, 3])
def test_canonical_equal_votes_keep_first_view_order(
    tmp_path: Path,
    max_views_per_input: int,
) -> None:
    source = np.array([[2]], dtype=np.uint8)
    answer = np.array([[8]], dtype=np.uint8)
    pack_config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(pack_config.spec, ArcSpec)
    pack_config.spec.max_grid = 2
    pack = pack_config.make()
    media = pack.pack(source, answer, training=False, rng=np.random.default_rng(0))[0]
    predictions = [
        pack.pack(
            source,
            np.array([[color]], dtype=np.uint8),
            training=False,
            rng=np.random.default_rng(0),
        )[1]
        for color in (0, 8, 1)
    ]
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", "task"]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {"task": {"test": [{"input": source.tolist(), "output": answer.tolist()}]}},
        ),
    )
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 2),
        max_views_per_input=max_views_per_input,
    ).make()
    metric.update(
        _packed(torch.tensor(np.stack(predictions)), torch.zeros(3)),
        media=torch.tensor(np.stack([media] * 3)),
        puzzle_identifiers=torch.ones(3, dtype=torch.long),
    )
    assert _pass_at(metric.compute()) == {"pass@1": 0.0, "pass@2": 1.0}


# ``a``'s first input and ``b``'s only input expect ``[[5]]``; ``a``'s second expects
# ``[[6]]``. So ``a`` is half solved and ``b`` fully solved.
def _two_input_tree(tmp_path: Path) -> dict[str, object]:
    """Tasks ``a`` (two test inputs) and ``b`` (one); every view answers ``[[5]]``."""
    pack_config = SpatialAugmentation.Config(spec=ArcSpec())
    assert isinstance(pack_config.spec, ArcSpec)
    pack_config.spec.max_grid = 2
    pack = pack_config.make()
    inputs = [
        np.array([[2]], dtype=np.uint8),
        np.array([[3]], dtype=np.uint8),
        np.array([[4]], dtype=np.uint8),
    ]
    five = np.array([[5]], dtype=np.uint8)
    packed = [
        pack.pack(grid, five, training=False, rng=np.random.default_rng(0))
        for grid in inputs
    ]
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", "a", "b"]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {
                "a": {
                    "test": [
                        {"input": [[2]], "output": [[5]]},
                        {"input": [[3]], "output": [[6]]},
                    ],
                },
                "b": {"test": [{"input": [[4]], "output": [[5]]}]},
            },
        ),
    )
    return {
        "logits": _packed(
            torch.tensor(np.stack([answer for _, answer in packed])),
            torch.zeros(3),
        ),
        "media": torch.tensor(np.stack([media for media, _ in packed])),
        "puzzle_identifiers": torch.tensor([1, 1, 2]),
    }


def test_canonical_rules_score_strict_and_per_output(tmp_path: Path) -> None:
    """A half-solved task earns task-mean credit, no strict credit, one output."""
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        rules=[StrictPass.Config(), PerOutputPass.Config()],
    ).make()
    metric.update(logits, **batch)
    scores = metric.compute()
    assert scores == {
        "pass@1": 0.75,
        "votes_times_mean_q@1": 0.75,
        "votes_times_max_q@1": 0.75,
        "strict@1": 0.5,
        "per_output@1": 2 / 3,
    }


def test_per_task_logs_include_summary_and_hardest_tasks(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    metric = CanonicalPassK.Config().make()
    monkeypatch.setattr(
        "priml.baselines.arcagi.metric.is_rank_zero",
        lambda: True,
    )
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi.metric")
    metric._log_per_task(
        {"pass@1": 0.5, "ignored": "not numeric"},
        [("task-a", 0.25), ("task-b", 0.0)],
        n_test_puzzles=2,
        n_no_preds=1,
    )
    assert caplog.messages == [
        "[eval] pass@K over 2 tasks (1 with no predictions): pass@1=0.5000",
        "[eval] per-task pass@1: 1/2 tasks solved (>0). hardest unsolved:",
        "[eval]   task-b pass@1=0.000",
        "[eval]   task-a pass@1=0.250",
    ]


def test_per_task_logs_keep_only_ten_hardest_tasks(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidate = CanonicalPassK.Config().make()
    monkeypatch.setattr(
        "priml.baselines.arcagi.metric.is_rank_zero",
        lambda: True,
    )
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi.metric")
    per_task = [(f"task-{index}", index / 10) for index in range(11)]

    candidate._log_per_task({}, per_task, n_test_puzzles=11, n_no_preds=0)

    assert caplog.messages[2:] == [
        f"[eval]   task-{index} pass@1={index / 10:.3f}" for index in range(10)
    ]


def test_scoring_rules_cover_empty_and_partial_tasks() -> None:
    tasks = [
        TaskScore(solved=(1, 0), num_inputs=2),
        TaskScore(solved=(0, 0), num_inputs=0),
    ]
    strict = StrictPass.Config().make()
    per_output = PerOutputPass.Config().make()
    assert strict([], 0) == 0.0
    assert strict(tasks, 0) == 0.0
    assert strict([TaskScore(solved=(2,), num_inputs=2)], 0) == 1.0
    assert per_output([], 0) == 0.0
    assert per_output(tasks, 0) == 0.5
    assert per_output([TaskScore(solved=(1,), num_inputs=1)], 0) == 1.0


def test_canonical_rules_are_off_by_default(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    metric.update(logits, **batch)
    assert not any(key.startswith(("strict", "per_output")) for key in metric.compute())


# ``c``'s wrong answer outvotes its right one, so ``c`` and the unanswered ``d`` tie at
# pass@1 = 0 and keep their ``test_puzzles.json`` order.
def _per_task_ballots(root: Path) -> dict[str, Tensor]:
    """Four tasks on a 2x2 grid: half solved, solved, outvoted, and unanswered."""
    (root / "identifiers.json").write_text(json.dumps(["<blank>", "a", "b", "c"]))
    (root / "test_puzzles.json").write_text(
        json.dumps(
            {
                "a": {
                    "test": [
                        {"input": [[2]], "output": [[5]]},
                        {"input": [[3]], "output": [[6]]},
                    ],
                },
                "b": {"test": [{"input": [[4]], "output": [[5]]}]},
                "c": {"test": [{"input": [[1]], "output": [[6]]}]},
                "d": {"test": [{"input": [[7]], "output": [[7]]}]},
            },
        ),
    )
    # Packed 2x2 tokens: color + 2, then the row/column EOS markers.
    answers = torch.tensor([[7, 1, 1, 0]] * 5 + [[8, 1, 1, 0]])
    return {
        "logits": _packed(answers, torch.tensor([0.5, -1.0, 2.0, 1.0, -0.5, 3.0])),
        "media": torch.tensor(
            [[4, 1, 1, 0], [5, 1, 1, 0], [6, 1, 1, 0]] + [[3, 1, 1, 0]] * 3,
        ),
        "puzzle_identifiers": torch.tensor([1, 1, 2, 3, 3, 3]),
    }


def test_canonical_scores_each_task_by_its_ranked_ballots(tmp_path: Path) -> None:
    batch = _per_task_ballots(tmp_path)
    logits = batch.pop("logits")
    spec = ArcSpec()
    spec.max_grid = 2
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2), spec=spec)
    scorer = metric.make()
    scorer.update(logits, **batch)
    assert scorer.compute()["pass@1"] == 0.375


def test_canonical_construction_defers_reading_the_tree(tmp_path: Path) -> None:
    """The loop builds metrics before the dataset stages the tree they read."""
    root = tmp_path / "staged-later"
    metric = CanonicalPassK.Config(
        working_dir=root,
        pass_ks=(1,),
        rules=[StrictPass.Config()],
    ).make()
    root.mkdir()
    batch = _two_input_tree(root)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    metric.update(logits, **batch)
    assert metric.compute()["strict@1"] == 0.5


def _pass_at(scores: Mapping[str, object]) -> dict[str, object]:
    """Keep only the ``pass@K`` scores; report-only rankings are tested elsewhere."""
    return {key: value for key, value in scores.items() if key.startswith("pass@")}


def test_global_prediction_gather_merges_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CanonicalPassK.Config().make()
    digest = "0" * 64
    remote_digest = "1" * 64
    candidate._preds = {"task": {digest: [(digest, 0.5)]}}
    remote = {
        "task": {digest: [(remote_digest, 0.75)]},
        "other-task": {remote_digest: [(remote_digest, 0.75)]},
    }
    remote_payload = encode_preds(remote)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output: list[Tensor], value: Tensor) -> None:
        output[0].copy_(value)
        if value.dtype == torch.int64:
            output[1].fill_(len(remote_payload))
        else:
            remote_tensor = torch.zeros_like(value)
            remote_tensor[: len(remote_payload)] = torch.tensor(
                list(remote_payload),
                dtype=torch.uint8,
            )
            output[1].copy_(remote_tensor)

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    assert candidate._global_preds() == {
        "task": {digest: [(digest, 0.5), (remote_digest, 0.75)]},
        "other-task": {remote_digest: [(remote_digest, 0.75)]},
    }


def test_nccl_gather_places_buffers_on_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = CanonicalPassK.Config().make()
    digest = "a" * 64
    candidate._preds = {"task": {digest: [(digest, 0.5)]}}
    devices: list[tuple[str, int | None]] = []
    factory_devices: list[tuple[object, torch.dtype | None]] = []
    original_tensor = torch.tensor
    original_zeros = torch.zeros
    transfer_devices: list[object] = []

    def capture_to(self: Tensor, *args: object, **kwargs: object) -> Tensor:
        device = kwargs.get("device", args[0] if args else None)
        transfer_devices.append(device)
        return self

    def tensor_factory(
        data: list[int],
        *,
        dtype: torch.dtype | None = None,
        device: str | None = None,
    ) -> Tensor:
        factory_devices.append((device, dtype))
        return original_tensor(data, dtype=dtype, device=device)

    def zeros_factory(
        size: int,
        *,
        dtype: torch.dtype,
        device: str | None,
    ) -> Tensor:
        factory_devices.append((device, dtype))
        return original_zeros(size, dtype=dtype, device=device)

    def fake_device(kind: str, index: int | None = None) -> str:
        devices.append((kind, index))
        return "cpu"

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.distributed, "is_available", lambda: True)
        scoped.setattr(torch.distributed, "is_initialized", lambda: True)
        scoped.setattr(torch.distributed, "get_backend", lambda: "nccl")
        scoped.setattr(torch.distributed, "get_world_size", lambda: 1)

        def all_gather(output: list[Tensor], value: Tensor) -> None:
            output[0].copy_(value)

        scoped.setattr(torch.distributed, "all_gather", all_gather)
        scoped.setattr(torch.cuda, "current_device", lambda: 4)
        scoped.setattr(
            metric,
            "torch",
            SimpleNamespace(
                device=fake_device,
                tensor=tensor_factory,
                cuda=torch.cuda,
                uint8=torch.uint8,
                zeros=zeros_factory,
                frombuffer=torch.frombuffer,
                empty_like=torch.empty_like,
            ),
        )
        scoped.setattr(Tensor, "to", capture_to)
        scoped.setattr(torch, "tensor", tensor_factory)
        scoped.setattr(torch, "zeros", zeros_factory)

        assert candidate._global_preds() == candidate._preds

    assert devices == [("cpu", None), ("cuda", 4)]
    assert factory_devices == [("cpu", None), ("cpu", torch.uint8)]
    assert transfer_devices == ["cpu"]


def test_prediction_wire_format_round_trips_multiple_values() -> None:
    predictions = {
        "task-a": {
            "1" * 64: [("a" * 64, 0.25), ("b" * 64, 0.75)],
            "2" * 64: [("c" * 64, 0.5)],
        },
        "task-b": {},
    }
    assert decode_preds(encode_preds(predictions)) == predictions


def test_prediction_wire_format_lengths_are_unsigned_without_large_allocations() -> (
    None
):
    large_length = 2**31

    class LargePreds(dict[str, dict[str, list[tuple[str, float]]]]):
        @override
        def __len__(self) -> int:
            return large_length

    class LargeInputs(dict[str, list[tuple[str, float]]]):
        @override
        def __len__(self) -> int:
            return large_length

    class LargeList(list[tuple[str, float]]):
        @override
        def __len__(self) -> int:
            return large_length

    class LargeBytes(bytes):
        @override
        def __len__(self) -> int:
            return large_length

    class LargeName(str):
        __slots__ = ()

        @override
        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            return LargeBytes(super().encode(encoding, errors))

    outer = encode_preds(LargePreds())
    name = encode_preds({LargeName("t"): {}})
    nested = encode_preds({"t": LargeInputs()})
    values = encode_preds({"t": {"0" * 64: LargeList()}})

    assert int.from_bytes(outer[:4], "little") == large_length
    assert int.from_bytes(name[4:8], "little") == large_length
    assert int.from_bytes(nested[9:13], "little") == large_length
    assert int.from_bytes(values[45:49], "little") == large_length


@pytest.mark.parametrize("cell", [True, 2.0, "2", None])
def test_json_grid_rejects_non_integer_cells(cell: object) -> None:
    with pytest.raises(TypeError, match="ARC grid cell"):
        _json_grid([[cell, 2], [1, 3]], spec=ArcSpec(max_grid=3))


def test_json_grid_reads_integer_cells() -> None:
    grid = _json_grid([[1, 2], [0, 3]], spec=ArcSpec(max_grid=3))

    assert grid.tolist() == [[1, 2], [0, 3]]


def test_metric_private_helpers_and_width_errors() -> None:
    values = torch.tensor([[0, 257], [3, 4]])
    assert [row.tolist() for row in _uint8_rows(values)] == [[0, 1], [3, 4]]
    floats = _floats(torch.tensor([0.25, -1.5]))
    assert floats == [0.25, -1.5]
    assert all(type(value) is float for value in floats)
    assert _read_u32(b"\xff\xff\xff\xff", 0) == (4_294_967_295, 4)
    assert _shape(np.zeros((2, 3), dtype=np.uint8)) == (2, 3)
    assert _model_output_header_width(out_width=10, media_len=9, k_steps=0) == 1
    assert _model_output_header_width(out_width=12, media_len=9, k_steps=0) == 3
    assert _model_output_header_width(out_width=13, media_len=6, k_steps=1) == 7
    assert _model_output_header_width(out_width=15, media_len=6, k_steps=2) == 9
    assert _votes_times_mean_q(4, 0.25, 0.9) == 1.0
    assert _votes_times_max_q(4, 0.25, 0.9) == 3.6
    with pytest.raises(ValueError, match="Expected len") as hash_error:
        _hash_bytes("bad")
    assert str(hash_error.value) == "Expected len(value) == 64."
    with pytest.raises(ValueError, match="per_step_acts=") as step_header_error:
        _model_output_header_width(out_width=14, media_len=6, k_steps=1)
    assert str(step_header_error.value) == (
        "per_step_acts=1 expects model_output width 13 (header 7 + grid 6); got 14."
    )
    with pytest.raises(ValueError, match="model_output width") as positive_header_error:
        _model_output_header_width(out_width=11, media_len=9, k_steps=0)
    assert str(positive_header_error.value) == (
        "model_output width 11 minus grid length 9 = header 2; "
        "inferred prediction length 9; expected 1 (baseline) or 3 "
        "(wide signal-dump) leading columns."
    )
    with pytest.raises(ValueError, match="model_output width") as zero_header_error:
        _model_output_header_width(out_width=9, media_len=9, k_steps=0)
    assert str(zero_header_error.value) == (
        "model_output width 9 minus grid length 9 = header 0; "
        "inferred prediction length 8; expected 1 (baseline) or 3 "
        "(wide signal-dump) leading columns."
    )
    with pytest.raises(ValueError, match="model_output width") as negative_header_error:
        _model_output_header_width(out_width=8, media_len=9, k_steps=0)
    assert str(negative_header_error.value) == (
        "model_output width 8 minus grid length 9 = header -1; "
        "inferred prediction length 7; expected 1 (baseline) or 3 "
        "(wide signal-dump) leading columns."
    )


def test_canonical_empty_tree_reports_every_metric_as_zero(tmp_path: Path) -> None:
    (tmp_path / "identifiers.json").write_text("[]")
    (tmp_path / "test_puzzles.json").write_text("{}")
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 3),
        rules=[StrictPass.Config(), PerOutputPass.Config()],
    ).make()

    assert metric.compute() == {
        "pass@1": 0.0,
        "pass@3": 0.0,
        "votes_times_mean_q@1": 0.0,
        "votes_times_mean_q@3": 0.0,
        "votes_times_max_q@1": 0.0,
        "votes_times_max_q@3": 0.0,
        "strict@1": 0.0,
        "strict@3": 0.0,
        "per_output@1": 0.0,
        "per_output@3": 0.0,
    }


def test_canonical_empty_tree_preserves_signal_dump_extras(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    wide_logits = torch.cat(
        [logits[:, :1], torch.zeros((3, 2)), logits[:, 1:]],
        dim=1,
    )
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    metric.update(wide_logits, **batch)
    (tmp_path / "test_puzzles.json").write_text("{}")

    scores = metric.compute()

    assert scores["pass@1"] == 0.0
    assert scores["votes_times_mean_q@1"] == 0.0
    extras = scores["extras"]
    assert isinstance(extras, dict)
    assert isinstance(extras["signal_dump"], SignalDumpPayload)


def test_canonical_empty_task_does_not_stop_following_tasks(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    tree: dict[str, object] = {
        "empty": {"test": []},
        "a": {
            "test": [
                {"input": [[2]], "output": [[5]]},
                {"input": [[3]], "output": [[6]]},
            ],
        },
        "b": {"test": [{"input": [[4]], "output": [[5]]}]},
    }
    (tmp_path / "test_puzzles.json").write_text(json.dumps(tree))
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    metric.update(logits, **batch)

    assert metric.compute()["pass@1"] == 0.5


def test_canonical_no_predictions_score_zero_and_log_every_task(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        rules=[StrictPass.Config(), PerOutputPass.Config()],
    ).make()
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi.metric")

    assert metric.compute() == {
        "pass@1": 0.0,
        "votes_times_mean_q@1": 0.0,
        "votes_times_max_q@1": 0.0,
        "strict@1": 0.0,
        "per_output@1": 0.0,
    }
    assert caplog.messages == [
        (
            "[eval] pass@K over 2 tasks (2 with no predictions): pass@1=0.0000 "
            "votes_times_mean_q@1=0.0000 votes_times_max_q@1=0.0000 "
            "strict@1=0.0000 per_output@1=0.0000"
        ),
        "[eval] per-task pass@1: 0/2 tasks solved (>0). hardest unsolved:",
        "[eval]   a pass@1=0.000",
        "[eval]   b pass@1=0.000",
    ]


@pytest.mark.parametrize("identifier", [-1, 3])
def test_canonical_rejects_identifiers_outside_the_map(
    tmp_path: Path,
    identifier: int,
) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    batch["puzzle_identifiers"] = torch.full((3,), identifier)
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    with pytest.raises(ValueError, match="puzzle identifier") as error:
        candidate.update(logits, **batch)

    assert str(error.value) == (
        f"puzzle identifier {identifier} is outside identifier map size 3."
    )


def test_canonical_skips_blank_before_later_puzzles(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    batch["puzzle_identifiers"] = torch.tensor([0, 1, 2])
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    metric.update(logits, **batch)

    assert set(metric._preds) == {"a", "b"}


@pytest.mark.parametrize("field", ["puzzle_identifiers", "spatial_tags"])
def test_canonical_rejects_floating_integer_fields(tmp_path: Path, field: str) -> None:
    """A float id or tag is a caller bug; dropping or misreading it hides ballots."""
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    batch["spatial_tags"] = torch.tensor([[1, 0, 0]] * 3)
    integral = batch[field]
    assert isinstance(integral, Tensor)
    batch[field] = integral.float()
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    with pytest.raises(TypeError) as error:
        candidate.update(logits, **batch)

    assert str(error.value) == (
        f"CanonicalPassK expects integer {field}; got torch.float32."
    )


def test_canonical_accepts_zero_when_blank_identifier_is_nonzero(
    tmp_path: Path,
) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    (tmp_path / "identifiers.json").write_text(json.dumps(["a", "<blank>", "b"]))
    metadata = tmp_path / "test" / "dataset.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps({"blank_identifier_id": 1}))
    batch["puzzle_identifiers"] = torch.tensor([0, 0, 2])
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    candidate.update(logits, **batch)

    assert set(candidate._preds) == {"a", "b"}


@pytest.mark.parametrize("spatial_tags", [[1, 1, 0], [1, 0, 1], [2, 0, 0]])
def test_non_spatial_filter_rejects_each_spatial_component(
    tmp_path: Path,
    spatial_tags: list[int],
) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    batch["puzzle_identifiers"] = torch.tensor([1, 1, 2])
    batch["spatial_tags"] = torch.tensor([spatial_tags, [1, 0, 0], [1, 0, 0]])
    candidate = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        spatial_views="non_spatial",
    ).make()

    candidate.update(logits, **batch)

    assert [len(rows) for rows in candidate._preds.values()] == [1, 1]


def test_canonical_state_round_trip_preserves_ballots(tmp_path: Path) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    source = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    source.update(logits, **batch)
    restored = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()

    restored.load_state_dict(
        json_loads(json.dumps(source.state_dict()), dict[str, object]),
    )

    assert restored.state_dict() == source.state_dict()
    assert restored.compute() == source.compute()


def test_canonical_state_load_defaults_missing_fields_to_empty() -> None:
    metric = CanonicalPassK.Config().make()

    metric.load_state_dict({})

    assert metric.state_dict() == {"hmap": {}, "preds": {}}


def test_canonical_state_does_not_accept_boolean_grid_dimensions() -> None:
    metric = CanonicalPassK.Config().make()

    with pytest.raises(TypeError, match="grid shape"):
        metric.load_state_dict({"hmap": {"hash": [True, 3]}})


def test_canonical_state_loads_tuple_grid_shapes() -> None:
    # Checkpoints written before list-valued ``hmap`` hold in-memory tuples.
    metric = CanonicalPassK.Config().make()

    metric.load_state_dict({"hmap": {"hash": (3, 4)}})

    assert metric.state_dict().get("hmap") == {"hash": [3, 4]}


def test_canonical_compute_emits_signal_payload_from_wide_output(
    tmp_path: Path,
) -> None:
    batch = _two_input_tree(tmp_path)
    logits = batch.pop("logits")
    assert isinstance(logits, Tensor)
    wide_logits = torch.cat(
        [logits[:, :1], torch.zeros((3, 2)), logits[:, 1:]],
        dim=1,
    ).to(torch.float64)
    wide_logits[:, 0] = torch.tensor(
        [0.1250000001, 0.25, 0.375],
        dtype=torch.float64,
    )
    wide_logits[:, 1] = torch.tensor(
        [-1.123456789, -2.25, -3.375],
        dtype=torch.float64,
    )
    wide_logits[:, 2] = torch.tensor(
        [0.6250000001, 0.75, 0.875],
        dtype=torch.float64,
    )
    candidate = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    candidate.update(wide_logits, **batch)

    result = candidate.compute()

    extras = cast(dict[str, object], result["extras"])
    assert isinstance(extras["signal_dump"], SignalDumpPayload)
    payload = extras["signal_dump"]
    prediction_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    assert payload.rows == [
        (
            name,
            grid_hash(np.array([[value]], dtype=np.uint8)),
            prediction_hash,
            float(np.float32(q_halt)),
            float(np.float32(logprob)),
            float(np.float32(stability)),
            1,
            1,
        )
        for name, value, q_halt, logprob, stability in (
            ("a", 2, 0.1250000001, -1.123456789, 0.6250000001),
            ("a", 3, 0.25, -2.25, 0.75),
            ("b", 4, 0.375, -3.375, 0.875),
        )
    ]
    assert payload.grids == {
        prediction_hash: np.array([[5]], dtype=np.uint8),
    }
    assert payload.steps == []
    assert payload.pass_ks == (1,)

    candidate.update(logits, **batch)
    assert isinstance(candidate.compute().get("extras"), dict)
    candidate.reset()
    assert "extras" not in candidate.compute()

    step_headers = torch.tensor(
        [
            [0.0, -1.0, 0.5, 3.0, 4.0, 0.123456789, 1.0],
            [0.0, -2.0, 0.75, 5.0, 6.0, 0.234567891, 0.0],
            [0.0, -3.0, 1.0, 7.0, 8.0, 0.345678912, 1.0],
        ],
        dtype=torch.float64,
    )
    step_metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        per_step_acts=1,
    ).make()
    step_metric.update(torch.cat([step_headers, logits[:, 1:]], dim=1), **batch)
    step_result = step_metric.compute()
    step_extras = cast(dict[str, object], step_result["extras"])
    step_payload = step_extras["signal_dump"]
    assert isinstance(step_payload, SignalDumpPayload)
    assert step_payload.steps == [
        (3, 4, (float(np.float32(0.123456789)),), (1,)),
        (5, 6, (float(np.float32(0.234567891)),), (0,)),
        (7, 8, (float(np.float32(0.345678912)),), (1,)),
    ]


def test_canonical_confidence_keeps_float64_tie_break_precision(
    tmp_path: Path,
) -> None:
    source = np.array([[2, 3], [4, 5]], dtype=np.uint8)
    answer = np.array([[5, 6], [7, 8]], dtype=np.uint8)
    wrong = np.array([[6, 5], [8, 7]], dtype=np.uint8)
    (tmp_path / "identifiers.json").write_text(json.dumps(["<blank>", "task"]))
    (tmp_path / "test_puzzles.json").write_text(
        json.dumps(
            {"task": {"test": [{"input": source.tolist(), "output": answer.tolist()}]}},
        ),
    )
    pack = SpatialAugmentation.Config(spec=ArcSpec(max_grid=3)).make()
    media, correct = pack.pack(
        source,
        answer,
        training=False,
        rng=np.random.default_rng(0),
    )
    _, incorrect = pack.pack(
        source,
        wrong,
        training=False,
        rng=np.random.default_rng(0),
    )
    logits = torch.cat(
        [
            torch.tensor([[19.0], [20.0]], dtype=torch.float32),
            torch.tensor(np.stack([correct, incorrect]), dtype=torch.float32),
        ],
        dim=1,
    )
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    metric.update(
        logits,
        media=torch.tensor(np.stack([media, media])),
        puzzle_identifiers=torch.ones(2, dtype=torch.int64),
    )

    scores = metric.compute()

    assert scores["pass@1"] == 0.0
    assert scores["pass@2"] == 1.0


def test_canonical_vote_means_accumulate_each_repeated_quality(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = grid_hash(np.array([[7]], dtype=np.uint8))
    metric._preds = {
        "a": {
            input_hash: [
                (truth_hash, 0.1),
                (truth_hash, 0.1),
                (wrong_hash, 0.4),
                (wrong_hash, 0.4),
            ],
        },
    }

    assert metric.compute()["pass@1"] == 0.0


def test_canonical_rankings_keep_task_means_and_distinct_confidence_rules(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1, 2),
    ).make()
    input_a = grid_hash(np.array([[2]], dtype=np.uint8))
    input_a2 = grid_hash(np.array([[3]], dtype=np.uint8))
    input_b = grid_hash(np.array([[4]], dtype=np.uint8))
    truth_a = grid_hash(np.array([[5]], dtype=np.uint8))
    truth_a2 = grid_hash(np.array([[6]], dtype=np.uint8))
    wrong = grid_hash(np.array([[7]], dtype=np.uint8))
    metric._preds = {
        "a": {
            input_a: [(wrong, 0.1), (wrong, 0.3), (truth_a, 0.5)],
            input_a2: [(truth_a2, 0.5)],
        },
        "b": {input_b: [(truth_a, 0.5)]},
    }

    scores = metric.compute()

    assert scores == {
        "pass@1": 0.75,
        "pass@2": 1.0,
        "votes_times_mean_q@1": 1.0,
        "votes_times_mean_q@2": 1.0,
        "votes_times_max_q@1": 0.75,
        "votes_times_max_q@2": 1.0,
    }


def test_canonical_view_cap_breaks_equal_confidence_by_hash(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(
        working_dir=tmp_path,
        pass_ks=(1,),
        max_views_per_input=1,
    ).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = "0" * 64
    metric._preds = {
        "a": {input_hash: [(truth_hash, 0.5), (wrong_hash, 0.5)]},
    }

    assert metric.compute()["pass@1"] == 0.0


def test_canonical_report_mean_ranking_uses_unbiased_confidence_totals(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = grid_hash(np.array([[7]], dtype=np.uint8))
    metric._preds = {
        "a": {
            input_hash: [
                (truth_hash, 0.01),
                (truth_hash, 0.01),
                (truth_hash, 0.01),
                (wrong_hash, 0.9),
            ],
        },
    }

    scores = metric.compute()

    assert scores["pass@1"] == 0.25
    assert scores["votes_times_mean_q@1"] == 0.0
    assert scores["votes_times_max_q@1"] == 0.0


def test_canonical_report_max_uses_highest_repeated_confidence(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = grid_hash(np.array([[7]], dtype=np.uint8))
    metric._preds = {
        "a": {
            input_hash: [
                (truth_hash, 0.9),
                (truth_hash, 0.1),
                (wrong_hash, 0.7),
                (wrong_hash, 0.7),
            ],
        },
    }

    assert metric.compute()["votes_times_max_q@1"] == 0.25


def test_canonical_report_rankings_use_mean_and_max_confidence(
    tmp_path: Path,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1, 2)).make()
    input_hash = grid_hash(np.array([[2]], dtype=np.uint8))
    truth_hash = grid_hash(np.array([[5]], dtype=np.uint8))
    wrong_hash = "0" * 64
    metric._preds = {
        "a": {
            input_hash: [
                (truth_hash, 0.2),
                (truth_hash, 0.9),
                (wrong_hash, 0.5),
                (wrong_hash, 0.5),
            ],
        },
    }

    scores = metric.compute()

    assert scores["pass@1"] == 0.25
    assert scores["votes_times_mean_q@1"] == 0.25
    assert scores["votes_times_max_q@1"] == 0.25


def test_canonical_compute_continues_after_an_input_without_predictions(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _two_input_tree(tmp_path)
    metric = CanonicalPassK.Config(working_dir=tmp_path, pass_ks=(1,)).make()
    input_a2 = grid_hash(np.array([[3]], dtype=np.uint8))
    truth_a2 = grid_hash(np.array([[6]], dtype=np.uint8))
    input_b = grid_hash(np.array([[4]], dtype=np.uint8))
    truth_b = grid_hash(np.array([[5]], dtype=np.uint8))
    metric._preds = {
        "a": {input_a2: [(truth_a2, 0.5)]},
        "b": {input_b: [(truth_b, 0.5)]},
    }
    caplog.set_level(logging.INFO, logger="priml.baselines.arcagi.metric")

    scores = metric.compute()

    assert scores["pass@1"] == 0.75
    assert "(0 with no predictions)" in caplog.messages[0]
    assert caplog.messages[2:] == [
        "[eval]   a pass@1=0.500",
        "[eval]   b pass@1=1.000",
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
