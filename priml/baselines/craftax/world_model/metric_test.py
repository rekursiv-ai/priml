"""Check Craftax bits per byte, its per-target NLL, and the zstd reference."""

from collections.abc import Mapping
from pathlib import Path

import math
import multiprocessing

from configgle import Fig

import pytest
import torch
import torch.distributed as dist

from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.batch import (
    Kind,
    PackedBatch,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.data import _pad_counts
from priml.baselines.craftax.world_model.loss import cross_entropy
from priml.baselines.craftax.world_model.metric import (
    MODALITIES,
    ZSTD_LEVEL,
    CraftaxBitsPerByte,
    NanochatSeries,
    canonical_records,
    counted_nll,
    craftax_target_nll,
)
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    FrameEncoder,
    WorldModel,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.lib import zstd_compat
from priml.lib.codec import from_plain
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer
from priml.train.tracker import AsyncTracker, FileTracker


RECORD: int = 1 + 152
"""Columns of one job's record: the action, then its 152 local slots."""

WINDOW: tuple[int, int] = (1, 7)
"""One packed window's ``[B, t_g]``: ``update`` takes a stratum and a weight per
position, and ``_one_episode`` packs one window of 7."""


def _tiny_model() -> WorldModel:
    config = WorldModel.Config()
    config.encoder.channels_in = 16
    config.decoder.channels_in = 16
    assert isinstance(config.encoder, FrameEncoder.Config)
    for stack in (config.encoder.stack, config.decoder.stack):
        _shrink_stack(stack)
    config.transformer.channels_in = 36
    config.transformer.num_layers = 1
    block = config.transformer.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.attn.channels_head = 4
    block.ffn.channels_hidden = 48
    torch.manual_seed(0)
    return config.make()


def _shrink_stack(stack: Transformer.Config) -> None:
    stack.num_layers = 1
    block = stack.block
    assert isinstance(block, TransformerBlock.Config | DecoderBlock.Config)
    assert isinstance(block.ffn, SwiGLU.Config)
    block.ffn.channels_hidden = 32


def _segment(decisions: int, *, frames: int, starts: bool, terminal: bool) -> Segment:
    schema = craftax_schema()
    generator = torch.Generator().manual_seed(decisions + 10 * frames)
    cells = torch.stack(
        [
            torch.randint(0, field.valid, (frames, 99), generator=generator)
            for field in schema.cell_fields
        ],
        dim=-1,
    )
    aux = torch.stack(
        [
            torch.randint(low - 155, high - 154, (frames,), generator=generator)
            for low, high in schema.scalar_ranges
        ],
        dim=-1,
    )
    done = torch.zeros(decisions, dtype=torch.bool)
    done[-1] = terminal
    return Segment(
        cells=cells.to(torch.uint8),
        aux=aux.to(torch.int16),
        actions=torch.randint(0, 43, (decisions,), generator=generator).byte(),
        reward=torch.randint(-1, 4, (decisions,), generator=generator).short(),
        done=done,
        starts_episode=starts,
    )


def _batch() -> PackedBatch:
    """Two windows: a start, a terminal, a mid-episode cut, and padding."""
    return pack_windows(
        [
            [
                _segment(3, frames=3, starts=True, terminal=True),
                _segment(4, frames=5, starts=True, terminal=False),
            ],
            [_segment(2, frames=3, starts=False, terminal=False)],
        ],
        t_g=16,
        s_max=4,
    )


def _one_episode(*, terminal: bool = False) -> PackedBatch:
    """Return a start and three decisions; without a terminal every frame exists."""
    frames = 3 if terminal else 4
    segment = _segment(3, frames=frames, starts=True, terminal=terminal)
    return pack_windows([[segment]], t_g=7, s_max=1)


def _ones(batch: PackedBatch) -> torch.Tensor:
    """Return one nat for every record column of every job."""
    return torch.ones(len(batch.job_at) * RECORD)


def test_craftax_target_nll_is_the_models_scored_terms_per_job() -> None:
    model = _tiny_model()
    batch = _batch()
    with torch.no_grad():
        record = craftax_target_nll(model, batch).view(len(batch.job_at), RECORD)
        loss = model(batch)
    assert torch.allclose(record[:, 0].sum(), loss.nll["action"])
    assert torch.allclose(record[:, 1].sum(), loss.nll["reward"])
    assert torch.allclose(record[:, 2].sum(), loss.nll["done"])
    assert torch.allclose(record[:, 3:102].sum(), loss.nll["board"])
    assert torch.allclose(record[:, 102:].sum(), loss.nll["hud"])


def test_each_job_records_the_nll_of_its_own_action() -> None:
    model = _tiny_model()
    batch = _batch()
    with torch.no_grad():
        record = craftax_target_nll(model, batch).view(len(batch.job_at), RECORD)
        logits = model.logits(batch).action.flatten(0, 1)
    acts = ~batch.job_is_start
    at = batch.job_at[acts].long()
    # The act at ``at`` is predicted from the obs position just before it.
    expected = cross_entropy(logits[at - 1], batch.action.flatten()[at].long()).nll
    assert torch.allclose(record[acts, 0], expected)
    assert bool((record[~acts, 0] == 0).all())


def test_records_are_exactly_the_targets_the_loss_scores() -> None:
    model = _tiny_model()
    batch = _batch()
    with torch.no_grad():
        terms = model.target_terms(batch, model.logits(batch))
    acts = ~batch.job_is_start
    kind = batch.kind.flatten()
    scored_action = terms["action"][1].flatten()
    assert torch.equal(scored_action.nonzero().flatten() + 1, batch.job_at[acts].long())
    assert bool((kind[batch.job_at[acts].long()] == Kind.ACT).all())
    assert torch.equal(terms["reward"][1], acts)
    assert torch.equal(terms["done"][1], acts)
    has_next = batch.job_next >= 0
    assert torch.equal(terms["board"][1], has_next[:, None].expand(-1, 99))
    assert torch.equal(terms["hud"][1], has_next[:, None].expand(-1, 51))


def test_a_full_decision_is_898_bytes() -> None:
    batch = _one_episode()
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(_ones(batch), media=batch, stratum=torch.zeros(WINDOW).long())
    # One nat per scored target, so bits per byte recovers the byte count. The
    # start job scores one frame; each decision adds action, reward, and done.
    targets = 3 * 3 + 4 * 150
    assert targets / (math.log(2) * metric.compute()["bpb"]) == pytest.approx(
        894 + 3 * 898,
    )
    assert len(canonical_records(batch)) == 894 + 3 * 898


def test_bits_per_byte_divides_nats_by_canonical_bytes() -> None:
    batch = _one_episode()
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(_ones(batch), media=batch, stratum=torch.zeros(WINDOW).long())
    result = metric.compute()
    # One nat per scored target: 3 actions, 3 rewards, 3 dones, 4 frames.
    targets = 3 * 3 + 4 * 150
    bits = math.log(2)
    assert result["bpb"] == pytest.approx(targets / (bits * (894 + 3 * 898)))
    assert result["bpb/board"] == pytest.approx(4 * 99 / (bits * 4 * 792))
    assert result["bpb/hud"] == pytest.approx(1 / (2 * bits))
    assert result["bpb/action"] == pytest.approx(1 / bits)
    assert result["bpb/reward"] == pytest.approx(1 / (2 * bits))
    assert result["bpb/done"] == pytest.approx(1 / bits)
    assert result["nats_per_decision"] == pytest.approx(targets / 3)
    assert "bpb_natural" not in result


def test_a_terminal_decision_scores_no_next_frame() -> None:
    batch = _one_episode(terminal=True)
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(_ones(batch), media=batch, stratum=torch.zeros(WINDOW).long())
    # The start frame and two next frames; the terminal decision has none.
    targets = 3 * 3 + 3 * 150
    expected = targets / (math.log(2) * (3 * 894 + 3 * 4))
    assert metric.compute()["bpb"] == pytest.approx(expected)


def test_canonical_records_hold_action_reward_and_done_bytes() -> None:
    batch = _one_episode(terminal=True)
    data = canonical_records(batch)
    assert len(data) == 3 * 894 + 3 * 4
    acts = (~batch.job_is_start).nonzero().flatten()
    heads = [data[894:898], data[2 * 894 + 4 : 2 * 894 + 8], data[-4:]]
    for job, head in zip(from_plain(acts.tolist(), list[int]), heads, strict=True):
        action = int(batch.action.flatten()[batch.job_at[job]])
        reward = int(batch.job_reward[job])
        done = int(batch.job_done[job])
        assert head == bytes([action]) + reward.to_bytes(2, "little", signed=True) + (
            bytes([done])
        )
    assert data[-1] == 1


def test_canonical_records_hold_each_jobs_head_then_the_frame_it_generates() -> None:
    batch = _batch()
    expected = bytearray()
    for job, at in enumerate(from_plain(batch.job_at.tolist(), list[int])):
        if not batch.job_is_start[job]:
            expected += bytes([int(batch.action.flatten()[at])])
            expected += int(batch.job_reward[job]).to_bytes(2, "little", signed=True)
            expected += bytes([int(batch.job_done[job])])
        frame = int(batch.job_next[job])
        if frame >= 0:
            expected += batch.cells[frame].numpy().tobytes()
            expected += batch.aux[frame].numpy().astype("<i2").tobytes()
    assert canonical_records(batch) == bytes(expected)


def test_strata_split_by_floor_and_by_event() -> None:
    batch = _one_episode()
    # The start and the first decision (positions 0-2) are floor 0 pre_death,
    # stratum 1; the two later decisions floor 1 ordinary, stratum 5.
    stratum = torch.tensor([[1, 1, 1, 5, 5, 5, 5]])
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(_ones(batch), media=batch, stratum=stratum)
    result = metric.compute()
    bits = math.log(2)
    first = (150 + 153) / (bits * (894 + 898))
    later = 2 * 153 / (bits * 2 * 898)
    assert result["bpb/floor_0"] == pytest.approx(first)
    assert result["bpb/pre_death"] == pytest.approx(first)
    assert result["bpb/floor_1"] == pytest.approx(later)
    assert result["bpb/ordinary"] == pytest.approx(later)
    assert "bpb/floor_2" not in result
    assert "bpb/entry" not in result


def test_each_target_counts_under_its_decisions_stratum() -> None:
    batch = _one_episode()
    stratum = torch.tensor([[0, 0, 0, 4, 4, 7, 7]])
    nll = torch.zeros(len(batch.job_at), RECORD)
    nll[:, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(nll.flatten(), media=batch, stratum=stratum)
    action = MODALITIES.index("action")
    assert metric.nats[action, [0, 4, 7]].tolist() == [1.0, 2.0, 3.0]
    assert metric.bytes[action, [0, 4, 7]].tolist() == [1.0, 1.0, 1.0]
    assert metric.decisions[[0, 4, 7]].tolist() == [1.0, 1.0, 1.0]


def test_a_start_mid_row_counts_under_its_own_episodes_stratum() -> None:
    batch = _batch()
    # Row 0 packs a terminal episode into positions 0-6, then starts another at
    # position 7, right after the first episode's last act.
    assert batch.kind[0, 6] == Kind.ACT
    assert batch.kind[0, 7] == Kind.START
    stratum = torch.zeros(2, 16, dtype=torch.long)
    stratum[0, :7], stratum[0, 7:] = 2, 9
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(_ones(batch), media=batch, stratum=stratum)
    board = MODALITIES.index("board")
    # The first episode generates three frames (its start and two acts; the
    # terminal act has none), the second five (its start and four acts).
    assert metric.bytes[board, [2, 9]].tolist() == [3 * 792.0, 5 * 792.0]


def test_strata_reweight_to_natural_counts() -> None:
    batch = _one_episode()
    stratum = torch.tensor([[0, 0, 0, 4, 4, 4, 4]])
    nll = torch.ones(len(batch.job_at), RECORD)
    nll[:, 0] = 0.0
    metric = CraftaxBitsPerByte.Config().make()
    natural = torch.zeros(27, dtype=torch.int64)
    natural[0], natural[4] = 1, 9
    metric.natural_decisions = natural.double()
    metric.update(nll.flatten(), media=batch, stratum=stratum)
    result = metric.compute()
    # Stratum 0 holds the start job and the first decision, stratum 4 the two
    # later decisions; each is weighted by natural over sampled decisions.
    weights = torch.tensor([1.0 / 1.0, 9.0 / 2.0])
    nats = torch.tensor([150.0 + 152.0, 2 * 152.0])
    size = torch.tensor([894.0 + 898.0, 2 * 898.0])
    decided = torch.tensor([1.0, 2.0])
    per_decision = float((weights * nats).sum() / (weights * decided).sum())
    bpb = float((weights * nats).sum() / (math.log(2) * (weights * size).sum()))
    assert result["nats_per_decision_natural"] == pytest.approx(per_decision)
    assert result["bpb_natural"] == pytest.approx(bpb)
    assert result["nats_per_decision"] != pytest.approx(per_decision)


def test_each_record_counts_by_its_positions_weight() -> None:
    batch = _one_episode()
    # The start job at position 0, then act jobs at 2, 4, and 6; the first act
    # is unscored history.
    weight = torch.tensor([[1.0, 5.0, 0.0, 5.0, 2.0, 5.0, 3.0]])
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(
        _ones(batch),
        media=batch,
        stratum=torch.zeros(WINDOW).long(),
        weight=weight,
    )
    result = metric.compute()
    # One nat per target: 150 for the start's frame, 153 per decision.
    nats = 1 * 150 + (2 + 3) * 153
    assert result["nats_per_decision"] == pytest.approx(nats / (2 + 3))
    bytes_ = 1 * 894 + (2 + 3) * 898
    assert result["bpb"] == pytest.approx(nats / (math.log(2) * bytes_))


def test_unscored_records_stay_out_of_the_zstd_reference() -> None:
    batch = _one_episode()
    weight = torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0]])
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(
        _ones(batch),
        media=batch,
        stratum=torch.zeros(WINDOW).long(),
        weight=weight,
    )
    # Records run in job order: the start's frame, then one per decision.
    scored = canonical_records(batch)[894 + 898 :]
    assert len(scored) == 2 * 898
    assert metric.compute()["zstd19_bpb"] == pytest.approx(_zstd_bpb(scored))


def test_counted_nll_is_the_nll_the_metric_counts_padding_included() -> None:
    # Opened mid-episode, then padded: the padding jobs sit at position 0, an obs
    # whose action is scored, so craftax_target_nll gives them its NLL.
    segment = _segment(3, frames=4, starts=False, terminal=False)
    batch = _pad_counts(pack_windows([[segment]], t_g=7, s_max=1), 8)
    assert len(batch.job_at) == 8
    with torch.no_grad():
        nll = craftax_target_nll(_tiny_model(), batch)
    weight = torch.full(WINDOW, 2.0)
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(nll, media=batch, stratum=torch.zeros(WINDOW).long(), weight=weight)
    counted = counted_nll(nll, media=batch, weight=weight)
    torch.testing.assert_close(counted.double(), metric.nats.sum())
    assert (nll.view(8, -1) * 2.0).sum() > counted


def test_the_zstd_reference_does_not_depend_on_how_ranks_split_micro_batches() -> None:
    batches = [_one_episode(), _one_episode(terminal=True)]
    stratum = torch.zeros(WINDOW).long()
    together = CraftaxBitsPerByte.Config().make()
    alone = [CraftaxBitsPerByte.Config().make() for _ in batches]
    for batch, metric in zip(batches, alone, strict=True):
        together.update(_ones(batch), media=batch, stratum=stratum)
        metric.update(_ones(batch), media=batch, stratum=stratum)
    together.compute()
    for metric in alone:
        metric.compute()
    # Two ranks' references are summed; one rank holding both must agree.
    assert torch.equal(together.reference, alone[0].reference + alone[1].reference)


def test_a_micro_batch_with_no_counted_record_changes_nothing() -> None:
    # A rank that ``EvalSpans`` leaves short repeats a micro-batch at weight 0.
    batch, stratum = _one_episode(), torch.zeros(WINDOW).long()
    repeated = CraftaxBitsPerByte.Config().make()
    repeated.update(_ones(batch), media=batch, stratum=stratum)
    repeated.update(
        _ones(batch),
        media=batch,
        stratum=stratum,
        weight=torch.zeros(WINDOW),
    )
    single = CraftaxBitsPerByte.Config().make()
    single.update(_ones(batch), media=batch, stratum=stratum)
    assert repeated.compute() == single.compute()


def test_metric_reports_the_zstd_reference_of_its_records() -> None:
    batch = _one_episode()
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(_ones(batch), media=batch, stratum=torch.zeros(WINDOW).long())
    expected = _zstd_bpb(canonical_records(batch))
    assert metric.compute()["zstd19_bpb"] == pytest.approx(expected)


def test_a_metric_that_measures_no_reference_reports_none() -> None:
    batch = _one_episode()
    metric = CraftaxBitsPerByte.Config().make()
    metric.measure_reference = False
    metric.update(_ones(batch), media=batch, stratum=torch.zeros(WINDOW).long())
    result = metric.compute()
    assert "zstd19_bpb" not in result
    assert result["bpb"] > 0


def test_state_dict_round_trips() -> None:
    batch = _one_episode()
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(_ones(batch), media=batch, stratum=torch.zeros(WINDOW).long())
    restored = CraftaxBitsPerByte.Config().make()
    restored.load_state_dict(metric.state_dict())
    assert restored.compute()["bpb"] == metric.compute()["bpb"]


def test_a_restored_metric_measures_the_reference_of_its_own_records() -> None:
    # An evaluation-only run restores a training run's metric and scores other
    # windows; the saved reference describes the training run's records.
    trained, scored = _one_episode(), _one_episode(terminal=True)
    metric = CraftaxBitsPerByte.Config().make()
    metric.update(_ones(trained), media=trained, stratum=torch.zeros(WINDOW).long())
    metric.compute()
    restored = CraftaxBitsPerByte.Config().make()
    restored.load_state_dict(metric.state_dict())
    restored.reset()
    restored.update(_ones(scored), media=scored, stratum=torch.zeros(WINDOW).long())
    expected = _zstd_bpb(canonical_records(scored))
    assert restored.compute()["zstd19_bpb"] == pytest.approx(expected)


def test_empty_evaluation_raises() -> None:
    with pytest.raises(ValueError, match="no scored"):
        CraftaxBitsPerByte.Config().make().compute()


@pytest.mark.compute_distributed
def test_ranks_sum_before_dividing(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ranks = [
        context.Process(target=_check_rank_sums, args=(rank, 2, str(tmp_path / "s")))
        for rank in range(2)
    ]
    try:
        for process in ranks:
            process.start()
        for process in ranks:
            process.join(timeout=120)
            assert process.exitcode == 0
    finally:
        for process in ranks:
            if process.is_alive():
                process.kill()
                process.join()


class _Recorder:
    """A tracker remembering every call."""

    class Config(Fig["_Recorder"]):
        """No options."""

    def __init__(self, config: Config) -> None:
        del config
        self.calls: list[tuple[str, dict[str, object]]] = []

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        del step
        self.calls.append((prefix, dict(metrics)))

    def log_images(self, key: str, images: list[object], step: int) -> None:
        del key, images, step

    def log_notes(self, notes: str) -> None:
        del notes

    def close(self) -> None:
        """Do nothing."""


def test_series_use_nanochat_names() -> None:
    config = NanochatSeries.Config()
    config.tracker = _Recorder.Config()
    series = config.make()
    recorder = series.tracker
    assert isinstance(recorder, _Recorder)
    series.log_metrics(
        {"loss": 1.0, "total_training_flops": 2.0, "total_training_time": 3.0},
        5,
        prefix="train/",
    )
    series.log_metrics(
        {
            "val_bpb": 0.5,
            "val_bpb/board": 0.4,
            "time": 9.0,
            "train_split_val_bpb": 0.3,
            "train_split_total_loss": 7.0,
        },
        5,
        prefix="eval/",
    )
    flat = {
        prefix + key: value
        for prefix, metrics in recorder.calls
        for key, value in metrics.items()
    }
    assert flat == {
        "train/loss": 1.0,
        "total_training_flops": 2.0,
        "total_training_time": 3.0,
        "val/bpb": 0.5,
        "val/bpb/board": 0.4,
        "eval/time": 9.0,
        "train_split/bpb": 0.3,
        "eval/train_split_total_loss": 7.0,
    }


def test_series_hand_their_directory_to_a_tracker_beneath_an_async_wrapper(
    tmp_path: Path,
) -> None:
    wrapper = AsyncTracker.Config()
    wrapper.tracker = FileTracker.Config()
    config = NanochatSeries.Config()
    config.tracker = wrapper
    config.base_dir = tmp_path
    final = config.copy_tree().finalize().tracker
    assert isinstance(final, AsyncTracker.Config)
    assert isinstance(final.tracker, FileTracker.Config)
    assert Path(final.tracker.working_dir) == tmp_path / "metrics.json"


def _check_rank_sums(rank: int, world: int, store: str) -> None:
    """Score one batch per rank and compare with one process scoring both."""
    batches = [_one_episode(), _one_episode(terminal=True)]
    strata = [torch.zeros(WINDOW).long(), torch.full(WINDOW, 5)]
    natural = torch.arange(1, 28, dtype=torch.float64)
    together = CraftaxBitsPerByte.Config().make()
    together.natural_decisions = natural
    for batch, stratum in zip(batches, strata, strict=True):
        together.update(_ones(batch) * 2.0, media=batch, stratum=stratum)
    expected = together.compute()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=world,
    )
    try:
        metric = CraftaxBitsPerByte.Config().make()
        metric.natural_decisions = natural
        batch = batches[rank]
        metric.update(_ones(batch) * 2.0, media=batch, stratum=strata[rank])
        result = metric.compute()
    finally:
        dist.destroy_process_group()
    assert result.keys() == expected.keys()
    for key, value in expected.items():
        assert result[key] == pytest.approx(value), key
    assert result["zstd19_bpb"] == pytest.approx(
        _zstd_bpb(*(canonical_records(batch) for batch in batches)),
    )


def _zstd_bpb(*micro_batches: bytes) -> float:
    """Return the zstd reference of micro-batches' records, each compressed alone."""
    compressed = sum(
        len(zstd_compat.compress(data, level=ZSTD_LEVEL)) for data in micro_batches
    )
    return 8 * compressed / sum(len(data) for data in micro_batches)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
