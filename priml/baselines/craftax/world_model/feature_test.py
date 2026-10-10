"""Check the frozen feature engine against the world model's training forward.

A two-layer ``exp_smoke`` model over the full Craftax schema, in float32 on
the CPU with masked attention: every feature the engine steps out is the
training forward's final-normed ``obs`` hidden state of exactly the history
the engine holds (``feature_gates.HeldHistory``), across episode starts,
mid-episode windows and re-prefills of full rows (G1), within 2e-6 on hiddens
of unit RMS: the largest difference measured is 9.5e-7 on an arm64 Mac and
2.4e-7 on x86. ``Sliding`` is held to the same forward over each step's own
window, before, at and after it slides, and a practice restore to the
context its donor's history or a fresh window gives. On a GPU, a graphed
step replays the eager one bit for bit, per fallback plan under ``Sliding``,
and one row's reset leaves the other rows' features bit for bit (G3a, G3b).
"""

from __future__ import annotations

from functools import cache, partial
from typing import TYPE_CHECKING, Final

import copy
import itertools

from configgle import Fig, PartialConfig
from torch import nn

import pytest
import torch

from priml.baselines.craftax.world_model.codec import decode
from priml.baselines.craftax.world_model.feature import (
    DonorHistory,
    FeatureEngine,
    Flash4CacheAttention,
    InitialWeights,
    Refill,
    Sliding,
    TrainedWeights,
    WorldModelFeature,
    context_inputs,
)
from priml.baselines.craftax.world_model.model import FrameEncoder
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.scripts.feature_gates import (
    GraphedStep,
    HeldHistory,
    reference_features,
    run_steps,
)
from priml.baselines.craftax.world_model.testing import (
    context_features,
    random_segment,
    window_segment,
)
from priml.model.attention.kernel import SdpaNaive
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from torch import Tensor

    from priml.baselines.craftax.world_model.model import WorldModel


T_MAX: Final = 16
"""Positions per row: rows fill within a dozen steps."""

KEEP: Final = 4
"""Decisions a re-prefill keeps."""

HOOK: Final = 4
"""Steps between ``ensure_room`` calls, the largest ``T_MAX`` and ``KEEP`` allow."""

DECISIONS: Final = 4
"""Decisions per ``Sliding`` window: a row slides at its fifth step."""

SMOKE: Final = "priml.baselines.craftax.world_model.experiments.exp_smoke"
"""The minimum-size world model, over the full schema."""

TWO_LAYERS: Final = ("step.model.transformer.num_layers=2",)
"""``exp_smoke`` with a second global block, so a tap can sit below the last."""


def test_steps_match_the_training_forward_across_resets_and_reprefills(
    source: WorldModelFeature,
) -> None:
    # Rows 1 and 2 fill by the hook at step 8 and re-prefill; row 1 then resets.
    rows, steps = 3, 10
    cells, aux, actions = _frames(rows, steps)
    engine = _engine(source, rows=rows)
    history = HeldHistory(rows, t_max=T_MAX, keep=KEEP)
    got = run_steps(
        engine,
        engine,
        obs=decode(cells, aux),
        actions=actions,
        resets={5: [0], 9: [1]},
        hook=HOOK,
        history=history,
    )
    reprefilled = {(s.row, s.first) for s in history.spans if not s.starts_episode}
    assert reprefilled == {(1, 8 - KEEP), (2, 8 - KEEP)}
    want = reference_features(
        source.model,
        layers=2,
        cells=cells,
        aux=aux,
        actions=actions,
        spans=history.spans,
    )
    torch.testing.assert_close(got, want, rtol=0, atol=2e-6)


def test_windows_begun_mid_episode_match_mid_episode_training_segments(
    source: WorldModelFeature,
) -> None:
    # A row whose history is gone (a resume) reads as training's mid-episode
    # window from its next observation, and a reset on the same step still
    # begins an episode.
    rows, steps = 3, 12
    cells, aux, actions = _frames(rows, steps)
    engine = _engine(source, rows=rows)
    history = HeldHistory(rows, t_max=T_MAX, keep=KEEP)
    got = run_steps(
        engine,
        engine,
        obs=decode(cells, aux),
        actions=actions,
        resets={5: [0]},
        windows={3: [1], 5: [0, 2]},
        hook=HOOK,
        history=history,
    )
    fresh = [(s.row, s.first) for s in history.spans if not s.starts_episode]
    assert fresh[:2] == [(1, 3), (2, 5)]
    assert (0, 5) not in fresh
    want = reference_features(
        source.model,
        layers=2,
        cells=cells,
        aux=aux,
        actions=actions,
        spans=history.spans,
    )
    torch.testing.assert_close(got, want, rtol=0, atol=2e-6)


def test_a_tap_below_the_last_block_matches_the_forward_cut_there(
    shallow_source: WorldModelFeature,
) -> None:
    source = shallow_source
    rows, steps = 2, 8
    cells, aux, actions = _frames(rows, steps)
    engine = _engine(source, rows=rows)
    assert engine.keys.shape[0] == 1
    history = HeldHistory(rows, t_max=T_MAX, keep=KEEP)
    got = run_steps(
        engine,
        engine,
        obs=decode(cells, aux),
        actions=actions,
        resets={3: [0]},
        hook=HOOK,
        history=history,
    )
    want = reference_features(
        source.model,
        layers=1,
        cells=cells,
        aux=aux,
        actions=actions,
        spans=history.spans,
    )
    torch.testing.assert_close(got, want, rtol=0, atol=2e-6)
    full = reference_features(
        source.model,
        layers=2,
        cells=cells,
        aux=aux,
        actions=actions,
        spans=history.spans,
    )
    assert not torch.allclose(got, full, atol=1e-2)


def test_the_obs_input_is_the_frame_encoders_pooled_output(
    source: WorldModelFeature,
) -> None:
    cells, aux, _ = _frames(3, 1)
    engine = _engine(source, rows=3)
    model = source.model
    with torch.no_grad():
        pooled, _ = model.encoder(model.frame_slots(cells[:, 0], aux[:, 0]))
        want = model.obs_proj(pooled)
    engine(decode(cells[:, 0], aux[:, 0]), torch.zeros(3), torch.zeros(3))
    torch.testing.assert_close(engine.obs_ring[:, 0], want, rtol=0, atol=1e-6)
    assert int(engine.invalid) == 0


def test_ensure_room_reports_the_context_and_reprefills_full_rows(
    source: WorldModelFeature,
) -> None:
    cells, aux, actions = _frames(2, 8)
    engine = _engine(source, rows=2)
    run_steps(
        engine,
        engine,
        obs=decode(cells, aux),
        actions=actions,
        resets={4: [1]},
        hook=HOOK,
        history=HeldHistory(2, t_max=T_MAX, keep=KEEP),
    )
    assert engine.length.tolist() == [16, 8]
    telemetry = engine.ensure_room(HOOK)
    assert telemetry["feature/reprefill_rows"] == 1
    assert telemetry["feature/context_decisions_mean"] == (8 + 4) / 2
    assert engine.length.tolist() == [2 * KEEP - 1, 8]
    with pytest.raises(ValueError, match=r"steps=5 must be in 1\.\.4"):
        engine.ensure_room(HOOK + 1)


def test_the_context_telemetry_reads_the_rows_that_continue(
    source: WorldModelFeature,
) -> None:
    cells, aux, actions = _frames(2, 3)
    obs = decode(cells, aux)
    engine = _engine(source, rows=2)
    for t in range(3):
        engine(obs[:, t], torch.zeros(2), _led_to(actions, t))
    engine.begin_window(torch.tensor([True, False]))
    telemetry = engine.ensure_room(1)
    assert telemetry["feature/context_decisions_mean"] == 3
    assert telemetry["feature/context_decisions_p5"] == 3
    # A resume restarts every row: none holds a context.
    engine.begin_window(torch.tensor([True, True]))
    telemetry = engine.ensure_room(1)
    assert telemetry["feature/context_decisions_mean"] == 0
    assert telemetry["feature/context_decisions_p95"] == 0


def test_a_missed_hook_raises_overflow_and_spares_the_other_rows(
    source: WorldModelFeature,
) -> None:
    rows, steps = 3, 14
    cells, aux, actions = _frames(rows, steps)
    obs = decode(cells, aux)
    # Row 1 restarts every third step, so it never fills; rows 0 and 2 do.
    resets = {t: [1] for t in range(0, steps, 3)}
    features: list[Tensor] = []
    for hook in (HOOK, 0):
        engine = _engine(source, rows=rows)
        features.append(
            run_steps(
                engine,
                engine,
                obs=obs,
                actions=actions,
                resets=resets,
                hook=hook,
                history=HeldHistory(rows, t_max=T_MAX, keep=KEEP),
            ),
        )
        if not hook:
            assert int(engine.overflow) > 0
            with pytest.raises(RuntimeError, match="overflow"):
                engine.ensure_room(HOOK)
    assert torch.equal(features[0][1], features[1][1])


def test_an_out_of_schema_frame_is_counted_then_raised_by_name(
    source: WorldModelFeature,
) -> None:
    cells, aux, _ = _frames(2, 1)
    obs = decode(cells[:, 0], aux[:, 0])
    obs[1, 0] = 0.5
    engine = _engine(source, rows=2)
    feature = engine(obs, torch.zeros(2), torch.zeros(2))
    assert bool(feature.isfinite().all())
    assert int(engine.invalid) == 1
    with pytest.raises(ValueError, match=r"1 frames held .* Latest frame: .*index"):
        engine.ensure_room(HOOK)


def test_reset_begins_an_episode_in_every_row(source: WorldModelFeature) -> None:
    cells, aux, actions = _frames(2, 3)
    obs = decode(cells, aux)
    fresh, reset = _engine(source, rows=2), _engine(source, rows=2)
    for t in range(3):
        reset(obs[:, t], torch.zeros(2), actions[:, t].float())
    reset.reset()
    for engine in (fresh, reset):
        engine(obs[:, 0], torch.zeros(2), torch.zeros(2))
    assert torch.equal(fresh.feature, reset.feature)
    assert reset.length.tolist() == [2, 2]


def test_a_frozen_engine_refuses_a_rebuild(source: WorldModelFeature) -> None:
    """Only a joint source's engine keeps the frames a rebuild re-encodes."""
    with pytest.raises(ValueError, match="keeps no frames"):
        _engine(source, rows=2).rebuild()


def test_a_window_begun_on_a_fresh_engine_begins_a_window(
    source: WorldModelFeature,
) -> None:
    # A resume begins windows in a fresh engine's rows; a terminal still
    # begins an episode.
    cells, aux, _ = _frames(3, 1)
    engine = _engine(source, rows=3)
    engine.begin_window(torch.tensor([True, True, False]))
    engine(
        decode(cells[:, 0], aux[:, 0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.zeros(3),
    )
    decisions, anchored = engine.context()
    assert decisions.tolist() == [1, 1, 1]
    assert anchored.tolist() == [False, True, True]


def test_context_inputs_lay_out_training_windows_from_position_0() -> None:
    # Width-1 observations, so each slot's value names its step.
    obs = torch.arange(1.0, 7.0).view(2, 3, 1)
    actions = -obs
    start = torch.tensor([100.0])
    x = context_inputs(
        obs,
        actions=actions,
        start=start,
        length=torch.tensor([2, 3]),
        anchored=torch.tensor([True, False]),
        tokens=6,
    )
    assert x[..., 0].tolist() == [
        [100.0, 2.0, -2.0, 3.0, 0.0, 0.0],
        [4.0, -4.0, 5.0, -5.0, 6.0, 0.0],
    ]


def test_sliding_steps_match_the_training_forward_of_each_steps_window(
    sliding_source: WorldModelFeature,
) -> None:
    # Row 0 reads its episode's prefix through step 3, slides at its fifth
    # decision, then resets at step 6 and reads its new prefix; row 1 begins a
    # window at step 1, which slides at step 5.
    rows, steps = 2, 8
    cells, aux, actions = _frames(rows, steps)
    starts: list[dict[int, bool]] = [{6: True}, {1: False}]
    engine = _engine(sliding_source, rows=rows)
    got, decisions, anchored = _drive(
        engine,
        engine,
        obs=decode(cells, aux),
        actions=actions,
        starts=starts,
    )
    firsts, anchors = _sliding_contexts(starts, steps=steps)
    t = torch.arange(steps)
    assert torch.equal(decisions, t - torch.tensor(firsts) + 1)
    assert anchored.tolist() == anchors
    assert decisions[0, 3:5].tolist() == [DECISIONS, DECISIONS]
    assert anchored[0, 3:5].tolist() == [True, False]
    want = _window_reference(
        sliding_source.model,
        cells=cells,
        aux=aux,
        actions=actions,
        firsts=firsts,
        anchors=anchors,
    )
    torch.testing.assert_close(got, want, rtol=0, atol=2e-6)


def test_ensure_room_plans_the_rows_that_can_slide_in_whole_chunks() -> None:
    config = _sliding_config()
    sliding = config.history
    assert isinstance(sliding, Sliding.Config)
    sliding.fallback_rows = 16
    engine = _engine(config.make(), rows=64)
    engine.needs_start.fill_(value=False)
    capacities = {}
    for able in (0, 1, 16, 33, 64):
        # The first ``able`` rows slide at their next step, 2 of them already.
        engine.count.fill_(0)
        engine.count[:able] = DECISIONS
        engine.count[: min(able, 2)] = DECISIONS + 1
        telemetry = engine.ensure_room(1)
        capacities[able] = engine.plan
        assert telemetry["feature/fallback_capacity"] == engine.plan
        assert telemetry["feature/sliding_fraction"] == min(able, 2) / 64
    assert capacities == {0: 0, 1: 16, 16: 16, 33: 48, 64: 64}
    # A row whose next step begins a window cannot slide in the block.
    engine.begin_window(torch.tensor([True] * 33 + [False] * 31))
    engine.ensure_room(1)
    assert engine.plan == 32


@pytest.mark.compute_torch_compile
@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_a_sliding_engine_compiles_one_fallback_shape_whatever_its_capacity() -> None:
    """Fallbacks of 1, 2, 4 and 8 rows and a restore fit a budget of three shapes.

    Each per-block function compiles once per shape -- the step's, the
    fallback chunk's, the prefill's -- and the actor, the evaluation and a
    joint learner share its budget; past it, a full-graph compile raises.
    """
    config = _sliding_config()
    config.compile = PartialConfig(
        torch.compile,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    torch.compiler.reset()
    try:
        with torch._dynamo.config.patch(recompile_limit=3):
            _fall_back_then_restore(config, device=torch.device("cuda"))
    finally:
        torch.compiler.reset()


def test_a_sliding_engine_runs_each_kernel_at_three_shapes_whatever_its_capacity() -> (
    None
):
    """The test above's budget, eagerly: each kernel sees three input shapes at most.

    A compiled kernel compiles once per shape, so a fallback chunk sized by how
    many rows fall back would compile once per count and overrun the budget.
    """
    config = _sliding_config()
    config.compile = _ShapeRecordingCompile.Config()
    kernels = _fall_back_then_restore(config, device=torch.device("cpu")).kernels
    shapes: dict[str, set[tuple[torch.Size, ...]]] = {}
    for name, kernel in (
        ("pre", kernels.pre),
        ("post", kernels.post),
        ("encode", kernels.encode),
    ):
        assert isinstance(kernel, _ShapeRecording), name
        shapes[name] = kernel.shapes
    assert all(1 <= len(seen) <= 3 for seen in shapes.values()), shapes


def test_a_joint_restore_encodes_the_restored_rows_frames_alone() -> None:
    config = _sliding_config()
    config.joint = True
    config.compile = _CountingCompile.Config()
    source = config.make()
    encode = source.kernels.encode
    assert isinstance(encode, _CountedCalls)
    cells, aux, actions = _frames(3, 3)
    donor, restored = _engine(source, rows=3), _engine(source, rows=3)
    for t in range(3):
        donor(decode(cells[:, t], aux[:, t]), torch.zeros(3), actions[:, t].float())
    archive = (
        DonorHistory.Config()
        .make()
        .archive(
            entries=3,
            ring=source.ring,
            width=source.width,
            frames=True,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )
    assert "obs" not in archive
    donor.save_history(
        archive,
        torch.arange(3),
        slice(0, 3),
        previous_action=actions[:, 2].float(),
        fresh=torch.zeros(3),
    )
    encode.calls = 0
    assert restored.restore_history(archive, torch.full((3,), -1)) is None
    assert encode.calls == 0
    restored.restore_history(archive, torch.tensor([-1, 0, -1]))
    # The restored row's 4 frames, in batches of the step's 3, the last padded.
    assert encode.calls == 2
    # Of its ring's 4 slots, the donor's 3 steps filled the first 3.
    torch.testing.assert_close(
        restored.obs_ring[1, :3],
        donor.obs_ring[0, :3],
        rtol=0,
        atol=1e-6,
    )
    assert not restored.obs_ring[[0, 2]].any()


def test_a_row_that_slides_outside_the_plan_raises_overflow(
    sliding_source: WorldModelFeature,
) -> None:
    cells, aux, actions = _frames(2, DECISIONS + 1)
    obs = decode(cells, aux)
    engine = _engine(sliding_source, rows=2)
    for t in range(DECISIONS + 1):
        engine(obs[:, t], torch.zeros(2), _led_to(actions, t))
    assert int(engine.overflow) == 1
    with pytest.raises(RuntimeError, match="the fallback plan of 0 rows left out"):
        engine.ensure_room(HOOK)


@pytest.mark.parametrize("saved_at", [3, 7])
@pytest.mark.parametrize("donor", [False, True], ids=["fresh_window", "donor"])
@pytest.mark.parametrize("sliding", [False, True], ids=["refill", "sliding"])
def test_a_restored_row_reads_its_donors_history_or_a_fresh_window(
    source: WorldModelFeature,
    sliding_source: WorldModelFeature,
    *,
    sliding: bool,
    donor: bool,
    saved_at: int,
) -> None:
    """Row 0 of one engine saves before step ``saved_at``; row 1 of another resumes it.

    With the donor's history the restored row reads what its donor reads
    (``Sliding``), or what a re-prefill at the save would have kept
    (``Refill``: at step 7 the donor's 7 anchored decisions restart from the
    last 4); with a fresh window, the branch alone. Both engines make room at
    the same block boundaries, the restore's among them.
    """
    used = sliding_source if sliding else source
    steps = 9
    cells, aux, actions = _frames(2, steps)
    own_cells, own_aux, own_actions = _frames(3, steps, seed=7)
    obs, own_obs = decode(cells, aux), decode(own_cells, own_aux)
    archive = (
        DonorHistory.Config()
        .make()
        .archive(
            entries=2,
            ring=used.ring,
            width=used.width,
            frames=False,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )
    bounds = sorted(
        {0, saved_at, *range(0, steps, HOOK), *range(saved_at, steps, HOOK)},
    )
    donor_engine, restored = _engine(used, rows=2), _engine(used, rows=3)
    donor_history = HeldHistory(2, t_max=T_MAX, keep=KEEP)
    history = HeldHistory(3, t_max=T_MAX, keep=KEEP)
    got: list[Tensor] = []
    contexts: list[tuple[int, bool]] = []
    for first, stop in itertools.pairwise([*bounds, steps]):
        if first == saved_at:
            donor_engine.save_history(
                archive,
                torch.tensor([0, 1]),
                slice(0, 2),
                previous_action=actions[:, first - 1].float(),
                fresh=torch.zeros(2),
            )
            previous = restored.restore_history(
                archive if donor else None,
                torch.tensor([-1, 0, -1]),
            )
            own_obs[1], own_actions[1] = obs[0], actions[0]
            assert previous is None or previous[1] == actions[0, first - 1]
            if donor:
                span = next(s for s in donor_history.spans if first - 1 in s.steps)
                held = first - span.first
                kept = min(held, KEEP)
                history.resume(
                    1,
                    t=first,
                    first=first - kept,
                    anchored=span.starts_episode and held <= KEEP,
                )
        for engine, held_history in (
            (donor_engine, donor_history),
            (restored, history),
        ):
            engine.ensure_room(stop - first)
            held_history.hook(first, steps=stop - first)
        for t in range(first, stop):
            donor_engine(obs[:, t], torch.zeros(2), _led_to(actions, t))
            feature = restored(own_obs[:, t], torch.zeros(3), _led_to(own_actions, t))
            got.append(feature[1].float().clone())
            decisions, anchored = restored.context()
            contexts.append((int(decisions[1]), bool(anchored[1])))
            windows = [1] if t == saved_at and not donor else []
            donor_history.step(t, resets=[])
            history.step(t, resets=[], windows=windows)
    if sliding:
        firsts, anchors = _sliding_contexts(
            [{} if donor else {saved_at: False}],
            steps=steps,
        )
        held = list(zip(firsts[0], anchors[0], strict=True))
    else:
        spans = [s for s in history.spans if s.row == 1]
        held = [
            next((s.first, s.starts_episode) for s in spans if t in s.steps)
            for t in range(steps)
        ]
    later = range(saved_at, steps)
    assert [contexts[t] for t in later] == [
        (t - held[t][0] + 1, held[t][1]) for t in later
    ]
    want = context_features(
        used.model,
        layers=2,
        windows=[
            window_segment(
                cells[0],
                aux=aux[0],
                actions=actions[0],
                first=held[t][0],
                last=t,
                anchored=held[t][1],
            )
            for t in later
        ],
    )
    torch.testing.assert_close(torch.stack(got[saved_at:]), want, rtol=0, atol=2e-6)


def test_a_history_saved_as_an_episode_or_window_begins_restores_as_a_window(
    source: WorldModelFeature,
) -> None:
    # Row 0 saves its two decisions; row 1's next step begins a window and row
    # 2's an episode, so neither has a history to save.
    cells, aux, actions = _frames(3, 3)
    obs = decode(cells, aux)
    archive = (
        DonorHistory.Config()
        .make()
        .archive(
            entries=3,
            ring=KEEP,
            width=source.width,
            frames=False,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )
    donor, restored = _engine(source, rows=3), _engine(source, rows=3)
    for engine in (donor, restored):
        for t in range(2):
            engine(obs[:, t], torch.zeros(3), _led_to(actions, t))
    donor.begin_window(torch.tensor([False, True, False]))
    donor.save_history(
        archive,
        torch.tensor([2, 0, 1]),
        slice(0, 3),
        previous_action=actions[:, 1].float(),
        fresh=torch.tensor([0.0, 0.0, 1.0]),
    )
    assert archive["length"].tolist() == [0, 0, 4]
    restored.restore_history(archive, torch.tensor([2, 0, 1]))
    restored(obs[:, 2], torch.zeros(3), actions[:, 1].float())
    decisions, anchored = restored.context()
    assert decisions.tolist() == [3, 1, 1]
    assert anchored.tolist() == [True, False, False]


@pytest.mark.parametrize("sliding", [False, True], ids=["refill", "sliding"])
def test_after_new_weights_a_rebuilt_engine_reads_every_context_under_them(
    *,
    sliding: bool,
) -> None:
    """The weights change before step 7, and the engine rebuilds.

    Then row 0's context is cached (``Refill``: 7 anchored decisions, which
    re-prefill at step 8) or slides; row 1's is a window begun at step 5, and
    row 2's next step begins one.
    """
    config = _sliding_config() if sliding else _config(layers=2)
    config.weights = _TwoLayerWeights.Config()
    config.joint = True
    source = config.make()
    rows, steps, changed = 3, 9, 7
    cells, aux, actions = _frames(rows, steps)
    engine = _engine(source, rows=rows)
    assert engine.ring == (DECISIONS if sliding else T_MAX // 2)
    old = copy.deepcopy(source.model)
    got, decisions, anchored = _drive(
        engine,
        engine,
        obs=decode(cells, aux),
        actions=actions,
        starts=[{}, {5: False}, {7: False}],
        before={changed: partial(_publish, source.model, engine)},
    )
    held = DECISIONS if sliding else 7
    assert decisions[:, changed - 1].tolist() == [held, 2, held]
    firsts = torch.arange(steps) - decisions + 1
    windows = [
        [
            window_segment(
                cells[row],
                aux=aux[row],
                actions=actions[row],
                first=int(firsts[row, t]),
                last=t,
                anchored=bool(anchored[row, t]),
            )
            for t in range(steps)
        ]
        for row in range(rows)
    ]
    # Earlier steps are each the old weights' forward, as the tests above check; the
    # last of them, then every step after the rebuild, whose contexts reach before it.
    for model, times in (
        (old, range(changed - 1, changed)),
        (source.model, range(changed, steps)),
    ):
        want = context_features(
            model,
            layers=2,
            windows=[windows[row][t] for row in range(rows) for t in times],
        )
        torch.testing.assert_close(
            got[:, times].flatten(0, 1),
            want,
            rtol=0,
            atol=2e-6,
        )


def test_last_decisions_hold_each_contexts_frames_and_the_actions_before_them() -> None:
    config = _sliding_config()
    config.joint = True
    engine = _engine(config.make(), rows=4)
    cells, aux, actions = _frames(4, 6)
    # Row 1's episode begins at step 4, row 3's at step 5, its ring full of
    # actions by then; row 2 begins a window before its next step.
    starts: list[dict[int, bool]] = [{}, {4: True}, {}, {5: True}]
    _drive(engine, engine, obs=decode(cells, aux), actions=actions, starts=starts)
    current = engine.last_decisions(1)
    assert torch.equal(current.cells[:, 0], cells[:, 5])
    led = actions[:, 4].to(torch.uint8)
    assert current.previous_actions[:, 0].tolist() == [*led[:3].tolist(), 0]
    engine.begin_window(torch.tensor([False, False, True, False]))
    recent = engine.last_decisions(DECISIONS - 1)
    assert recent.count.tolist() == [3, 2, 0, 1]
    assert torch.equal(recent.cells[0], cells[0, 3:])
    assert torch.equal(recent.aux[0], aux[0, 3:])
    assert torch.equal(recent.previous_actions[0], actions[0, 2:5].to(torch.uint8))
    assert torch.equal(recent.cells[1, 1:], cells[1, 4:])
    assert recent.previous_actions[1].tolist() == [0, 0, int(actions[1, 4])]
    assert not recent.cells[1, 0].any()
    assert not recent.cells[2].any()
    assert int(actions[3, 3]) != 0
    assert recent.previous_actions[3].tolist() == [0, 0, 0]
    with pytest.raises(ValueError, match="n=5 exceeds the ring's 4"):
        engine.last_decisions(DECISIONS + 1)
    with pytest.raises(ValueError, match="keeps no frames"):
        _engine(_sliding_config().make(), rows=1).last_decisions(1)


def test_making_a_source_leaves_the_global_generator_where_it_was() -> None:
    torch.manual_seed(5)
    before = torch.get_rng_state()
    _make_source(layers=2)
    assert torch.equal(torch.get_rng_state(), before)


def test_initial_weights_are_their_seeds_init(tmp_path: Path) -> None:
    config = InitialWeights.Config()
    config.experiment = SMOKE
    config.seed = 3
    first, second = config.make()(), config.make()()
    for (name, a), b in zip(
        first.state_dict().items(),
        second.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(a, b), name
    config.seed = 4
    assert not torch.equal(config.make()().start, first.start)
    path = tmp_path / "step.pt"
    torch.save({"step": {"model": first.state_dict()}}, path)
    trained = TrainedWeights.Config()
    trained.experiment = SMOKE
    # No default run to read: the experiment names the checkpoint of its parent's.
    with pytest.raises(ValueError, match="needs its experiment's checkpoint"):
        trained.make()
    trained.checkpoint = path
    loaded = trained.make()()
    assert not loaded.training
    for (name, a), b in zip(
        loaded.state_dict().items(),
        first.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(a, b), name


@pytest.mark.parametrize(
    ("t_max", "keep", "hook", "match"),
    [
        (16, 4, 5, "hook_interval=5"),
        (16, 0, 1, "keep=0"),
        (16, 9, 1, "keep=9"),
    ],
)
def test_a_window_that_cannot_hold_a_block_is_refused(
    t_max: int,
    keep: int,
    hook: int,
    match: str,
) -> None:
    config = _config(layers=2)
    refill = config.history = Refill.Config()
    refill.t_max, refill.keep = t_max, keep
    config.hook_interval = hook
    with pytest.raises(ValueError, match=match):
        config.make()


@pytest.mark.parametrize(
    ("decisions", "fallback_rows", "match"),
    [(0, 2, "decisions=0"), (4, 0, "fallback_rows=0")],
)
def test_a_window_or_chunk_that_is_not_positive_is_refused(
    decisions: int,
    fallback_rows: int,
    match: str,
) -> None:
    config = _sliding_config()
    sliding = config.history
    assert isinstance(sliding, Sliding.Config)
    sliding.decisions, sliding.fallback_rows = decisions, fallback_rows
    with pytest.raises(ValueError, match=match):
        config.make()


@pytest.mark.parametrize(
    ("hook", "reprefill_rows", "match"),
    [
        (DECISIONS + 1, 1, "hook_interval=5 must be in 1..4 for Sliding"),
        (DECISIONS, 0, "reprefill_rows=0"),
    ],
)
def test_a_block_past_the_window_or_an_empty_prefill_is_refused(
    hook: int,
    reprefill_rows: int,
    match: str,
) -> None:
    config = _sliding_config()
    config.hook_interval, config.reprefill_rows = hook, reprefill_rows
    with pytest.raises(ValueError, match=match):
        config.make()


def test_a_tap_past_the_models_blocks_is_refused() -> None:
    config = _config(layers=3)
    with pytest.raises(ValueError, match=r"layers=3 must be in 1\.\.2"):
        config.make()


@pytest.mark.parametrize("encoder", [True, False])
def test_an_attention_kernel_the_engine_does_not_run_is_refused(
    *,
    encoder: bool,
) -> None:
    """The engine attends with its own SDPA calls, whatever a kernel slot holds."""
    source = _make_source(layers=1)
    model = source.model
    assert isinstance(model.encoder, FrameEncoder)
    block = (model.encoder.stack if encoder else model.transformer).blocks[0]
    assert isinstance(block, TransformerBlock)
    assert isinstance(block.attn, nn.Module)
    block.attn.register_module("attn_kernel", SdpaNaive.Config().make())
    with pytest.raises(ValueError, match="not as SdpaNaive"):
        _engine(source, rows=1)


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_a_graphed_step_replays_the_eager_step_bit_for_bit() -> None:
    source = _make_source(layers=2, dtype=torch.bfloat16)
    rows, steps = 3, 12
    cells, aux, actions = _frames(rows, steps)
    device = torch.device("cuda")
    obs, actions = decode(cells, aux).to(device), actions.to(device)
    runs: list[Tensor] = []
    for graphed in (False, True):
        engine = source.make_engine(rows=rows, device=device)
        runs.append(
            run_steps(
                GraphedStep(engine, rows=rows) if graphed else engine,
                engine,
                obs=obs,
                actions=actions,
                resets={5: [0], 9: [1]},
                hook=HOOK,
                history=HeldHistory(rows, t_max=T_MAX, keep=KEEP),
            ),
        )
    assert torch.equal(runs[0], runs[1])


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_one_rows_reset_leaves_the_other_rows_bit_for_bit() -> None:
    source = _make_source(layers=2, dtype=torch.bfloat16)
    rows, steps = 3, 12
    cells, aux, actions = _frames(rows, steps)
    device = torch.device("cuda")
    obs, actions = decode(cells, aux).to(device), actions.to(device)
    runs: list[Tensor] = []
    for resets in ({}, {6: [1]}):
        engine = source.make_engine(rows=rows, device=device)
        runs.append(
            run_steps(
                GraphedStep(engine, rows=rows),
                engine,
                obs=obs,
                actions=actions,
                resets=resets,
                hook=HOOK,
                history=HeldHistory(rows, t_max=T_MAX, keep=KEEP),
            ),
        )
    assert torch.equal(runs[0][[0, 2]], runs[1][[0, 2]])
    assert torch.equal(runs[0][1, :6], runs[1][1, :6])
    assert not torch.equal(runs[0][1, 6:], runs[1][1, 6:])


@pytest.mark.compute_torch_compile
@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_a_compiled_step_replays_as_a_graph_and_stays_near_the_eager_one() -> None:
    rows, steps = 3, 12
    cells, aux, actions = _frames(rows, steps)
    device = torch.device("cuda")
    obs, actions = decode(cells, aux).to(device), actions.to(device)
    runs: list[Tensor] = []
    for compiled, graphed in ((False, False), (True, False), (True, True)):
        config = _config(layers=2)
        config.dtype = torch.bfloat16
        if compiled:
            config.compile = PartialConfig(torch.compile, fullgraph=True, dynamic=False)
        source = config.make()
        engine = source.make_engine(rows=rows, device=device)
        runs.append(
            run_steps(
                GraphedStep(engine, rows=rows) if graphed else engine,
                engine,
                obs=obs,
                actions=actions,
                resets={5: [0], 9: [1]},
                hook=HOOK,
                history=HeldHistory(rows, t_max=T_MAX, keep=KEEP),
            ),
        )
    assert torch.equal(runs[1], runs[2])
    torch.testing.assert_close(runs[1], runs[0], rtol=0, atol=0.05)


@pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:(flash_attn|cutlass|quack)",
    "ignore::UserWarning:cutlass",
)
@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_flash4_steps_match_the_masked_steps_in_bfloat16() -> None:
    rows, steps = 3, 12
    cells, aux, actions = _frames(rows, steps)
    device = torch.device("cuda")
    obs, actions = decode(cells, aux).to(device), actions.to(device)
    runs: list[Tensor] = []
    for flash in (False, True):
        config = _config(layers=2)
        # FA4 takes heads of 64 channels or more; the smoke model's are 4.
        assert isinstance(config.weights, InitialWeights.Config)
        config.weights.overrides.append(
            "step.model.transformer.block.attn.channels_head=64",
        )
        config.dtype = torch.bfloat16
        if flash:
            config.attention = Flash4CacheAttention.Config()
        source = config.make()
        engine = source.make_engine(rows=rows, device=device)
        runs.append(
            run_steps(
                GraphedStep(engine, rows=rows),
                engine,
                obs=obs,
                actions=actions,
                resets={5: [0], 9: [1]},
                hook=HOOK,
                history=HeldHistory(rows, t_max=T_MAX, keep=KEEP),
            ),
        )
    torch.testing.assert_close(runs[1], runs[0], rtol=0, atol=0.05)


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_a_graphed_sliding_step_replays_the_eager_one_under_every_plan() -> None:
    config = _sliding_config()
    config.dtype = torch.bfloat16
    sliding = config.history
    assert isinstance(sliding, Sliding.Config)
    sliding.fallback_rows = 1
    source = config.make()
    # Planned at hooks 0..10, in chunks of one row: no row, then row 0, then
    # rows 0 and 1, then all four, as rows 1-3 restart at step 3 and rows 2-3
    # again at step 5.
    rows, steps = 4, 12
    cells, aux, actions = _frames(rows, steps)
    device = torch.device("cuda")
    obs, actions = decode(cells, aux).to(device), actions.to(device)
    starts: list[dict[int, bool]] = [
        {},
        {3: True},
        {3: True, 5: True},
        {3: True, 5: False},
    ]
    runs: list[Tensor] = []
    plans: list[int] = []
    for graphed in (False, True):
        engine = source.make_engine(rows=rows, device=device)
        step = GraphedStep(engine, rows=rows) if graphed else engine
        got, _, _ = _drive(
            step,
            engine,
            obs=obs,
            actions=actions,
            starts=starts,
            hook=2,
        )
        runs.append(got)
        if isinstance(step, GraphedStep):
            plans = sorted(step.graphs)
    assert torch.equal(runs[0], runs[1])
    assert plans == [0, 1, 2, 4]


def _config(*, layers: int) -> WorldModelFeature.Config:
    """Return a float32 masked source over the two-layer smoke model."""
    config = WorldModelFeature.Config()
    weights = config.weights = InitialWeights.Config()
    weights.experiment = SMOKE
    weights.overrides = list(TWO_LAYERS)
    refill = config.history = Refill.Config()
    refill.t_max = T_MAX
    refill.keep = KEEP
    config.layers = layers
    config.hook_interval = HOOK
    config.dtype = torch.float32
    return config


class _TwoLayerWeights:
    """``_config``'s two-layer smoke model, built once a process and copied for each source.

    For a source whose weights a test moves; the others share a fixture's.
    """

    class Config(Fig["_TwoLayerWeights"]):
        """Nothing to configure: the weights are ``_config``'s."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self) -> WorldModel:
        """Return a copy of the model, at the seeded init and in eval mode."""
        return copy.deepcopy(_two_layer_model())


@cache
def _two_layer_model() -> WorldModel:
    """Build ``_config``'s two-layer smoke model."""
    return _config(layers=2).weights.make()()


def _make_source(
    *,
    layers: int,
    dtype: torch.dtype = torch.float32,
) -> WorldModelFeature:
    """Build a source of ``layers`` taps over the smoke model, in ``dtype``."""
    config = _config(layers=layers)
    config.dtype = dtype
    return config.make()


# A source's weights are read-only and its engines own their state, so the CPU
# tests of a module share one per tap; building it resolves and initializes an
# experiment, 40 ms on x86.
@pytest.fixture(scope="module")
def source() -> WorldModelFeature:
    """Return the shared float32 CPU source tapping both blocks."""
    return _make_source(layers=2)


@pytest.fixture(scope="module")
def shallow_source() -> WorldModelFeature:
    """Return the shared float32 CPU source tapping the first block."""
    return _make_source(layers=1)


@pytest.fixture(scope="module")
def sliding_source() -> WorldModelFeature:
    """Return the shared float32 CPU ``Sliding`` source tapping both blocks."""
    return _sliding_config().make()


def _sliding_config() -> WorldModelFeature.Config:
    """Return ``DECISIONS``-decision windows over the smoke model, 2 rows a chunk."""
    config = _config(layers=2)
    sliding = config.history = Sliding.Config()
    sliding.decisions = DECISIONS
    sliding.fallback_rows = 2
    return config


def _engine(source: WorldModelFeature, *, rows: int) -> FeatureEngine:
    """Return a fresh CPU engine of ``rows`` rows."""
    return source.make_engine(rows=rows, device=torch.device("cpu"))


def _drive(
    step: Callable[[Tensor, Tensor, Tensor], Tensor],
    engine: FeatureEngine,
    *,
    obs: Tensor,
    actions: Tensor,
    starts: Sequence[dict[int, bool]],
    hook: int = HOOK,
    before: Mapping[int, Callable[[], object]] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Step every row as the rollout does, ``before[t]`` first; return the features."""
    steps = obs.shape[1]
    outputs: list[tuple[Tensor, Tensor, Tensor]] = []
    for t in range(steps):
        if before is not None and t in before:
            before[t]()
        if t % hook == 0:
            engine.ensure_room(min(hook, steps - t))
        engine.begin_window(torch.tensor([row.get(t) is False for row in starts]))
        terminals = torch.tensor([float(row.get(t, False)) for row in starts])
        previous = actions[:, t - 1] if t else torch.zeros_like(actions[:, 0])
        feature = step(obs[:, t], terminals.to(obs.device), previous.float())
        decisions, anchored = engine.context()
        outputs.append((feature.float().clone(), decisions.clone(), anchored.clone()))
    features, decisions, anchored = (
        torch.stack(part, dim=1) for part in zip(*outputs, strict=True)
    )
    return features, decisions.cpu(), anchored.cpu()


def _publish(model: WorldModel, engine: FeatureEngine) -> None:
    """Move every weight of ``model`` in place, as a publication does, then rebuild."""
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.add_(0.2 * parameter.abs().mean() * noise)
    engine.rebuild()


def _sliding_contexts(
    starts: Sequence[dict[int, bool]],
    *,
    steps: int,
) -> tuple[list[list[int]], list[list[bool]]]:
    """Return each row's and step's ``Sliding`` context: its first step and anchor."""
    firsts: list[list[int]] = []
    anchors: list[list[bool]] = []
    for row in starts:
        first, anchored = 0, True
        firsts.append([])
        anchors.append([])
        for t in range(steps):
            if t in row:
                first, anchored = t, row[t]
            firsts[-1].append(max(first, t + 1 - DECISIONS))
            anchors[-1].append(anchored and t - first < DECISIONS)
    return firsts, anchors


def _window_reference(
    model: WorldModel,
    *,
    cells: Tensor,
    aux: Tensor,
    actions: Tensor,
    firsts: list[list[int]],
    anchors: list[list[bool]],
) -> Tensor:
    """Return the training forward's feature of each step's context ``[R, T, C]``."""
    rows, steps = len(firsts), len(firsts[0])
    windows = [
        window_segment(
            cells[row],
            aux=aux[row],
            actions=actions[row],
            first=firsts[row][t],
            last=t,
            anchored=anchors[row][t],
        )
        for row in range(rows)
        for t in range(steps)
    ]
    return context_features(model, layers=2, windows=windows).view(rows, steps, -1)


class _CountedCalls:
    """A function, and how often it was called."""

    def __init__(self, function: Callable[..., object]) -> None:
        self.function = function
        self.calls = 0

    def __call__(self, *args: object) -> object:
        """Count the call, then make it."""
        self.calls += 1
        return self.function(*args)


class _CountingCompile:
    """A ``compile`` slot that counts each kernel's calls rather than compiling it."""

    class Config(Fig["_CountingCompile"]):
        """No options."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, function: Callable[..., object]) -> Callable[..., object]:
        """Return ``function``, counted."""
        return _CountedCalls(function)


class _ShapeRecording:
    """A function, and the input shapes it was called at: a compiled kernel's graphs."""

    def __init__(self, function: Callable[..., object]) -> None:
        self.function = function
        self.shapes: set[tuple[torch.Size, ...]] = set()

    def __call__(self, *args: object) -> object:
        """Record the shapes of the tensor arguments, then make the call."""
        self.shapes.add(
            tuple(arg.shape for arg in args if isinstance(arg, torch.Tensor)),
        )
        return self.function(*args)


class _ShapeRecordingCompile:
    """A ``compile`` slot that records each kernel's input shapes rather than compiling it."""

    class Config(Fig["_ShapeRecordingCompile"]):
        """No options."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, function: Callable[..., object]) -> Callable[..., object]:
        """Return ``function``, recording."""
        return _ShapeRecording(function)


# Eight rows, fallback chunks of four: 1, 2, 4 and then all 8 rows fall back in turn.
def _fall_back_then_restore(
    config: WorldModelFeature.Config,
    *,
    device: torch.device,
) -> WorldModelFeature:
    """Step a ``Sliding`` source's engine past each fallback count, then save and restore."""
    sliding = config.history
    assert isinstance(sliding, Sliding.Config)
    sliding.fallback_rows = 4
    source = config.make()
    rows = 8
    cells, aux, actions = _frames(rows, 1)
    obs = decode(cells[:, 0], aux[:, 0]).to(device)
    zeros = torch.zeros(rows, device=device)
    archive = (
        DonorHistory.Config()
        .make()
        .archive(
            entries=1,
            ring=DECISIONS,
            width=36,
            frames=False,
            dtype=torch.float32,
            device=device,
        )
    )
    engine = source.make_engine(rows=rows, device=device)
    for able in (1, 2, 4, 8):
        engine.needs_start.fill_(value=False)
        engine.count.fill_(1)
        engine.length.fill_(2)
        engine.count[:able] = DECISIONS
        engine.length[:able] = 2 * DECISIONS
        engine.ensure_room(1)
        engine(obs, zeros, actions[:, 0].float().to(device))
    engine.save_history(
        archive,
        torch.tensor([0], device=device),
        slice(rows - 1, rows),
        previous_action=zeros[:1],
        fresh=zeros[:1],
    )
    engine.restore_history(
        archive,
        torch.tensor([0] + [-1] * (rows - 1), device=device),
    )
    return source


def _led_to(actions: Tensor, t: int) -> Tensor:
    """Return the actions that led to step ``t``'s observations: none at step 0."""
    return (actions[:, t - 1] if t else torch.zeros_like(actions[:, 0])).float()


def _frames(
    rows: int,
    steps: int,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return schema-valid random cells, aux values and actions ``[rows, steps, ...]``."""
    segments = [
        random_segment(craftax_schema(), steps, seed=seed + row) for row in range(rows)
    ]
    return (
        torch.stack([s.cells[:steps] for s in segments]),
        torch.stack([s.aux[:steps] for s in segments]),
        torch.stack([s.actions for s in segments]).long(),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
