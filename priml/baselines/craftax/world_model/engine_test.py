"""Check the sampling engine against the training forward (the design's five tests).

The fourth, Gumbel-max frequencies, lives with ``gumbel_max`` in priml's
``math/probability_test.py``.
"""

from collections.abc import Callable, Generator
from functools import partial
from pathlib import Path
from typing import Final, cast

import contextlib
import dataclasses

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.craftax.testing import fill_portable
from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.batch import (
    Kind,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.engine import (
    Control,
    Decision,
    Engine,
    GraphedStep,
    Prefix,
    StartResult,
)
from priml.baselines.craftax.world_model.loss import cell_nll, scalar_nll
from priml.baselines.craftax.world_model.model import (
    DecoderBlock,
    WorldModel,
)
from priml.baselines.craftax.world_model.schema import done_id
from priml.baselines.craftax.world_model.scripts.engine_checks import (
    forced_control,
    no_done,
    prefix_of,
)
from priml.baselines.craftax.world_model.testing import (
    random_segment,
    small_schema,
    state_of,
    tiny_model,
)
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive, SdpaVarlen
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_tensor_golden


_CWD: Final = Path(__file__).resolve().parent


@pytest.mark.parametrize("untied_output", [False, True])
def test_teacher_forced_log_probs_match_training_forward(untied_output: bool) -> None:
    schema = small_schema()
    # Two layers in each stack, so a cache or cross-attention memory shared
    # across layers diverges from the training forward.
    model = tiny_model(
        schema,
        global_layers=2,
        local_layers=2,
        untied_output=untied_output,
    )
    # Built norms scale by 1, under which a norm the engine applies in the wrong
    # place, or two swapped, changes nothing.
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.uniform_(0.5, 1.5, generator=generator)
    segment = random_segment(schema, 2, seed=1, terminal=True)
    batch = pack_windows([[segment]], t_g=5, s_max=1)
    with torch.no_grad():
        logits = model.logits(batch)
    expected_action = logits.action[0].log_softmax(-1)
    expected = _local_log_probs(model, logits.local, batch_targets=_targets(segment))
    engine = Engine(model, rows=1, t_max=16, generator=torch.Generator().manual_seed(0))
    begun = engine.start(forced_control(engine, segment, 0))
    torch.testing.assert_close(begun.job.logp[0, 2:], expected[0, 2:])
    torch.testing.assert_close(begun.job.logp[0, :2], torch.zeros(2))
    decisions = [
        engine.decide(forced_control(engine, segment, step)) for step in range(2)
    ]
    for step, decision in enumerate(decisions):
        obs = int((batch.kind[0] == Kind.OBS).nonzero()[step, 0])
        action = int(segment.actions[step])
        torch.testing.assert_close(
            decision.action_logp[0],
            expected_action[obs, action],
            rtol=0,
            atol=1e-5,
        )
        scored = 2 if step == 1 else schema.local_slots
        torch.testing.assert_close(
            decision.job.logp[0, :scored],
            expected[step + 1, :scored],
            rtol=0,
            atol=1e-4,
        )
    # The terminal job's frame slots are skipped, not scored.
    assert bool((decisions[1].job.logp[0, 2:] == 0).all())
    assert bool(engine.state.needs_start[0])


def test_the_engine_matches_its_golden() -> None:
    """A tiny model's first free-running decision in two rows, frozen.

    The engine this one was ported from matched it bit for bit on a trained
    checkpoint (a check retired with its full-size fixture). The golden holds
    the port's own values from portable weights, computed in float64, so they
    hold on every host: the start and act jobs' tokens and log-probabilities,
    and the actions.
    """
    with host_agnostic_numerics():
        model = tiny_model(small_schema())
        fill_portable(model, seed=73)
        engine = Engine(
            model,
            rows=2,
            t_max=16,
            generator=torch.Generator().manual_seed(5),
        )
        start, decision = engine.step(engine.control())
    assert_tensor_golden(
        _CWD / "testdata" / "engine_tiny.pt",
        {
            "start/tokens": start.job.tokens,
            "start/logp": start.job.logp,
            "actions": decision.action,
            "action_logp": decision.action_logp,
            "tokens": decision.job.tokens,
            "logp": decision.job.logp,
        },
    )


@pytest.mark.parametrize("t_max", [4, 10])
def test_a_window_too_short_or_not_whole_decisions_is_refused(t_max: int) -> None:
    with pytest.raises(ValueError, match="multiple of 4, at least 8"):
        Engine(
            tiny_model(small_schema()),
            rows=1,
            t_max=t_max,
            generator=torch.Generator(),
        )


def test_a_schema_without_the_reward_and_done_prefix_is_refused() -> None:
    model = tiny_model(small_schema())
    model.schema = dataclasses.replace(model.schema, prefix_names=(), prefix_ranges=())
    with pytest.raises(ValueError, match="reward and done prefix"):
        Engine(model, rows=1, t_max=8, generator=torch.Generator())


def test_a_cross_attention_with_grouped_key_heads_is_refused() -> None:
    model = tiny_model(small_schema())
    block = model.decoder.stack.blocks[0]
    assert isinstance(block, DecoderBlock)
    block.cross_attn = Attention.Config(
        channels_in=16,
        num_heads=4,
        num_heads_kv=2,
    ).make()
    with pytest.raises(ValueError, match="as many key heads as query heads"):
        Engine(model, rows=1, t_max=8, generator=torch.Generator())


@pytest.mark.parametrize("cross", [False, True])
def test_decoder_attention_on_another_kernel_is_refused(cross: bool) -> None:
    model = tiny_model(small_schema())
    block = model.decoder.stack.blocks[0]
    assert isinstance(block, DecoderBlock)
    attention = block.cross_attn if cross else block.attn
    assert isinstance(attention, Attention)
    attention.attn_kernel = SdpaNaive()
    with pytest.raises(ValueError, match="decoder attention as SdpaFused"):
        Engine(model, rows=1, t_max=8, generator=torch.Generator())


def test_global_attention_on_another_kernel_is_refused() -> None:
    model = tiny_model(small_schema())
    block = model.transformer.blocks[0]
    assert isinstance(block, TransformerBlock)
    assert isinstance(block.attn, VarlenAttention)
    block.attn.attn_kernel = _OtherVarlen(SdpaVarlen.Config())
    with pytest.raises(ValueError, match="global attention as SdpaVarlen"):
        Engine(model, rows=1, t_max=8, generator=torch.Generator())


def test_a_decoder_stack_with_an_input_projection_is_refused() -> None:
    model = tiny_model(small_schema())
    model.decoder.stack.proj_in = nn.Linear(16, 16)
    with pytest.raises(ValueError, match="without an input projection"):
        Engine(model, rows=1, t_max=8, generator=torch.Generator())


def test_a_post_norm_global_block_is_refused() -> None:
    model = tiny_model(small_schema())
    block = model.transformer.blocks[0]
    assert isinstance(block, TransformerBlock)
    block.prenorm = False
    with pytest.raises(ValueError, match="pre-norm global blocks"):
        Engine(model, rows=1, t_max=8, generator=torch.Generator())


def test_a_decoder_self_attention_with_split_projections_is_refused() -> None:
    model = tiny_model(small_schema())
    block = model.decoder.stack.blocks[0]
    assert isinstance(block, DecoderBlock)
    assert isinstance(block.attn, Attention)
    block.attn.split_qkv_projection = True
    with pytest.raises(ValueError, match="decoder self-attention with a fused QKV"):
        Engine(model, rows=1, t_max=8, generator=torch.Generator())


def test_a_prefix_past_the_window_keeps_its_last_quarter() -> None:
    schema = small_schema()
    model = tiny_model(schema, global_layers=2)
    segment = random_segment(schema, 7, seed=29)
    # 1 + 2 * 7 positions overrun 16 - 2, so the last 4 decisions are kept.
    long = Engine(model, rows=1, t_max=16, generator=torch.Generator())
    long.prefill(torch.tensor([0]), prefix_of(segment))
    assert int(long.state.length[0]) == 8
    assert int(long.state.count[0]) == 4
    kept = Engine(model, rows=1, t_max=16, generator=torch.Generator())
    kept.prefill(
        torch.tensor([0]),
        Prefix(
            cells=segment.cells[None, 3:],
            aux=segment.aux[None, 3:],
            actions=segment.actions[None, 3:],
            starts_episode=False,
        ),
    )
    _assert_same_decision(long, kept, seed=30)


def test_a_row_run_past_its_window_never_writes_another_rows_cache() -> None:
    engine = Engine(
        tiny_model(small_schema()),
        rows=2,
        t_max=8,
        generator=torch.Generator().manual_seed(27),
    )
    engine.step(no_done(engine))
    before = state_of(engine, row=1)
    control = no_done(engine)
    control.active[1] = False
    # Row 0 takes four decisions without ``ensure_room``: 3 positions to 11 of 8.
    for _ in range(4):
        engine.decide(control)
    assert int(engine.state.length[0]) == 11
    after = state_of(engine, row=1)
    for name, value in before.items():
        assert torch.equal(after[name], value), name


def test_prefill_then_step_equals_step_only() -> None:
    schema = small_schema()
    model = tiny_model(schema, global_layers=2)
    segment = random_segment(schema, 3, seed=2)
    stepped = Engine(model, rows=1, t_max=16, generator=torch.Generator())
    stepped.start(forced_control(stepped, segment, 0))
    for step in range(3):
        stepped.decide(forced_control(stepped, segment, step))
    prefilled = Engine(model, rows=1, t_max=16, generator=torch.Generator())
    prefilled.prefill(torch.tensor([0]), prefix_of(segment))
    assert torch.equal(prefilled.state.length, stepped.state.length)
    assert torch.equal(prefilled.state.cells, stepped.state.cells)
    _assert_same_decision(prefilled, stepped, seed=7)


def test_resetting_one_row_leaves_other_rows_unchanged() -> None:
    model = tiny_model(small_schema())
    plain, reset = (
        Engine(model, rows=2, t_max=32, generator=torch.Generator().manual_seed(3))
        for _ in range(2)
    )
    _assert_reset_is_row_local(
        _reset_run(plain, plain.step, reset=False),
        _reset_run(reset, reset.step, reset=True),
    )


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_graphed_reset_leaves_other_rows_unchanged() -> None:
    model = tiny_model(small_schema()).cuda()
    generator = torch.cuda.default_generators[0]
    runs: list[list[tuple[StartResult, Decision]]] = []
    for reset in (False, True):
        engine = Engine(model, rows=2, t_max=32, generator=generator)
        replay = GraphedStep(engine)
        generator.manual_seed(3)
        runs.append(_reset_run(engine, replay, reset=reset))
    _assert_reset_is_row_local(*runs)


def test_a_graphed_step_on_another_generator_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "default_generators", (torch.Generator(),))
    engine = Engine(
        tiny_model(small_schema()),
        rows=2,
        t_max=16,
        generator=torch.Generator(),
    )
    with pytest.raises(
        ValueError,
        match=r"^A graphed engine samples from its default CUDA generator\.$",
    ):
        GraphedStep(engine)


def test_a_graphed_step_captures_once_and_replays_eager_steps_into_its_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tiny_model(small_schema())
    eager = Engine(model, rows=2, t_max=16, generator=torch.Generator())
    graphed = Engine(model, rows=2, t_max=16, generator=torch.Generator())
    cuda = _FakeCuda(graphed)
    cuda.install(monkeypatch)
    replay = GraphedStep(graphed, warmup=2)
    # Warmup runs idle steps on the side stream, then each graph captures there.
    assert cuda.log == [
        "enter side",
        "warmup start",
        "warmup decide",
        "warmup start",
        "warmup decide",
        "exit side",
        "capture 0 on side",
        "capture 1 on side",
        "capture 2 on side",
    ]
    for graph, method, buffer in zip(
        cuda.graphs,
        ("start", "skip_start", "decide"),
        (replay._start, replay._skip, replay._decision),
        strict=True,
    ):
        graph.run = partial(_rerun, getattr(graphed, method), replay._control, buffer)
    cuda.log.clear()
    outputs: list[list[tuple[StartResult, Decision]]] = []
    returned: list[tuple[StartResult, Decision]] = []
    for engine, step in ((eager, eager.step), (graphed, replay)):
        engine.generator.manual_seed(21)
        ending = no_done(engine)
        ending.job_tokens[1, 1, 0] = done_id(done=True)
        controls = [no_done(engine), no_done(engine), ending, no_done(engine)]
        # Cloned as returned: a graphed step overwrites its outputs at the next call.
        results = [(result, _clone_step(*result)) for result in map(step, controls)]
        if engine is graphed:
            returned = [result for result, _ in results]
        outputs.append([clone for _, clone in results])
    assert len(cuda.graphs) == 3
    assert [graph.captures for graph in cuda.graphs] == [1, 1, 1]
    assert cuda.log == [
        "replay 0",
        "replay 2",
        "replay 1",
        "replay 2",
        "replay 1",
        "replay 2",
        "replay 0",
        "replay 2",
    ]
    assert [start for start, _ in returned] == [
        replay._start,
        replay._skip,
        replay._skip,
        replay._start,
    ]
    assert all(decision is replay._decision for _, decision in returned)
    for (start_a, decision_a), (start_b, decision_b) in zip(
        outputs[0],
        outputs[1],
        strict=True,
    ):
        assert torch.equal(start_a.started, start_b.started)
        assert torch.equal(start_a.job.tokens, start_b.job.tokens)
        assert torch.equal(start_a.job.logp, start_b.job.logp)
        assert torch.equal(decision_a.action, decision_b.action)
        assert torch.equal(decision_a.job.tokens, decision_b.job.tokens)
        assert torch.equal(decision_a.job.logp, decision_b.job.logp)


def test_step_runs_the_start_job_only_when_a_row_begins_an_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = Engine(
        tiny_model(small_schema()),
        rows=2,
        t_max=16,
        generator=torch.Generator().manual_seed(20),
    )
    calls: list[Tensor] = []
    start = engine.start

    def spy(control: Control) -> StartResult:
        calls.append(engine.state.needs_start.clone())
        return start(control)

    monkeypatch.setattr(engine, "start", spy)
    begun = [engine.step(no_done(engine))[0] for _ in range(2)]
    engine.reset(torch.tensor([False, True]))
    idle = no_done(engine)
    idle.active[1] = False
    begun.append(engine.step(idle)[0])
    begun.append(engine.step(no_done(engine))[0])
    assert [row.tolist() for row in calls] == [[True, True], [False, True]]
    assert [b.started.tolist() for b in begun] == [
        [True, True],
        [False, False],
        [False, False],
        [False, True],
    ]


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_graphed_step_replays_eager_steps() -> None:
    model = tiny_model(small_schema()).cuda()
    generator = torch.cuda.default_generators[0]
    eager = Engine(model, rows=2, t_max=16, generator=generator)
    graphed = Engine(model, rows=2, t_max=16, generator=generator)
    replay = GraphedStep(graphed)
    outputs: list[list[tuple[StartResult, Decision]]] = []
    for engine, step in ((eager, eager.step), (graphed, replay)):
        generator.manual_seed(21)
        ending = no_done(engine)
        ending.job_tokens[1, 1, 0] = done_id(done=True)
        # Steps 2 and 3 begin no episode; step 3 ends row 1's, which step 4
        # restarts.
        controls = [no_done(engine), no_done(engine), ending, no_done(engine)]
        outputs.append([_clone_step(*step(control)) for control in controls])
    for (start_a, decision_a), (start_b, decision_b) in zip(
        outputs[0],
        outputs[1],
        strict=True,
    ):
        assert torch.equal(start_a.started, start_b.started)
        assert torch.equal(start_a.job.tokens, start_b.job.tokens)
        torch.testing.assert_close(start_a.job.logp, start_b.job.logp)
        assert torch.equal(decision_a.action, decision_b.action)
        assert torch.equal(decision_a.job.tokens, decision_b.job.tokens)
        torch.testing.assert_close(decision_a.job.logp, decision_b.job.logp)
    assert [start.started.tolist() for start, _ in outputs[1]] == [
        [True, True],
        [False, False],
        [False, False],
        [False, True],
    ]


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_graphed_step_on_a_device_not_current_replays_eager_steps() -> None:
    assert torch.cuda.current_device() == 0
    model = tiny_model(small_schema()).to("cuda:1")
    generator = torch.cuda.default_generators[model.start.device.index or 0]
    eager = Engine(model, rows=2, t_max=16, generator=generator)
    graphed = Engine(model, rows=2, t_max=16, generator=generator)
    replay = GraphedStep(graphed)
    outputs: list[list[tuple[StartResult, Decision]]] = []
    for engine, step in ((eager, eager.step), (graphed, replay)):
        generator.manual_seed(28)
        outputs.append([_clone_step(*step(no_done(engine))) for _ in range(3)])
    for (start_a, decision_a), (start_b, decision_b) in zip(
        outputs[0],
        outputs[1],
        strict=True,
    ):
        assert torch.equal(start_a.job.tokens, start_b.job.tokens)
        assert torch.equal(decision_a.action, decision_b.action)
        assert torch.equal(decision_a.job.tokens, decision_b.job.tokens)


def test_fixed_seed_reproduces_identical_tokens() -> None:
    model = tiny_model(small_schema())
    first = Engine(model, rows=2, t_max=32, generator=torch.Generator().manual_seed(5))
    second = Engine(model, rows=2, t_max=32, generator=torch.Generator().manual_seed(5))
    for _ in range(3):
        start_a, decision_a = first.step(first.control())
        start_b, decision_b = second.step(second.control())
        assert torch.equal(start_a.job.tokens, start_b.job.tokens)
        assert torch.equal(decision_a.action, decision_b.action)
        assert torch.equal(decision_a.job.tokens, decision_b.job.tokens)
        assert torch.equal(decision_a.job.logp, decision_b.job.logp)


def test_long_episode_reprefills_last_quarter_from_position_zero() -> None:
    schema = small_schema()
    model = tiny_model(schema, global_layers=2)
    engine = Engine(model, rows=1, t_max=8, generator=torch.Generator().manual_seed(6))
    engine.start(engine.control())
    frames: list[tuple[Tensor, Tensor]] = []
    actions: list[Tensor] = []
    for _ in range(3):
        assert engine.ensure_room() >= 1
        frames.append((engine.state.cells[0].clone(), engine.state.aux[0].clone()))
        actions.append(engine.decide(no_done(engine)).action[0])
    assert int(engine.state.length[0]) == 7
    assert engine.ensure_room() == 2
    assert int(engine.state.length[0]) == 4
    fresh = Engine(model, rows=1, t_max=8, generator=torch.Generator())
    kept_cells = [cells for cells, _ in frames[1:]] + [engine.state.cells[0]]
    kept_aux = [aux for _, aux in frames[1:]] + [engine.state.aux[0]]
    fresh.prefill(
        torch.tensor([0]),
        Prefix(
            cells=torch.stack(kept_cells)[None],
            aux=torch.stack(kept_aux)[None],
            actions=torch.stack(actions[1:])[None],
            starts_episode=False,
        ),
    )
    _assert_same_decision(fresh, engine, seed=8)


def test_reprefill_after_a_wrapped_prefill_keeps_the_last_quarter() -> None:
    schema = small_schema()
    model = tiny_model(schema, global_layers=2)
    engine = Engine(
        model,
        rows=1,
        t_max=16,
        generator=torch.Generator().manual_seed(15),
    )
    segment = random_segment(schema, 7, seed=16, starts=False)
    # Seven decisions wrap the ring of four: decision 4 lands in slot 0.
    engine.prefill(torch.tensor([0]), prefix_of(segment))
    assert engine.ensure_room() == 1
    action = engine.decide(no_done(engine)).action
    assert int(engine.state.length[0]) == 16
    assert engine.ensure_room() == 4
    assert int(engine.state.length[0]) == 8
    fresh = Engine(model, rows=1, t_max=16, generator=torch.Generator())
    fresh.prefill(
        torch.tensor([0]),
        Prefix(
            cells=torch.cat([segment.cells[4:].long(), engine.state.cells])[None],
            aux=torch.cat([segment.aux[4:].long(), engine.state.aux])[None],
            actions=torch.cat([segment.actions[4:].long(), action])[None],
            starts_episode=False,
        ),
    )
    _assert_same_decision(fresh, engine, seed=17)


def test_ensure_room_reprefills_full_rows_a_few_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = small_schema()
    model = tiny_model(schema, global_layers=2)
    segments = [random_segment(schema, 7, seed=s, starts=False) for s in (20, 21, 22)]
    calls: list[int] = []
    states: list[list[dict[str, Tensor]]] = []
    for prefill_rows in (2, 3):
        engine = Engine(
            model,
            rows=3,
            t_max=16,
            generator=torch.Generator().manual_seed(23),
            prefill_rows=prefill_rows,
        )
        for row, segment in enumerate(segments):
            engine.prefill(torch.tensor([row]), prefix_of(segment))
        engine.decide(no_done(engine))
        # One decision after 7 prefilled ones fills every row of 16 positions.
        assert engine.state.length.tolist() == [16] * 3
        monkeypatch.setattr(engine, "prefill", _counting(engine.prefill, calls))
        engine.ensure_room()
        states.append([state_of(engine, row=row) for row in range(3)])
    # Re-prefilling every full row in one forward holds all their windows at once.
    assert calls == [2, 1, 3]
    for chunked, whole in zip(states[0], states[1], strict=True):
        for name, value in whole.items():
            torch.testing.assert_close(chunked[name], value, msg=name)


def test_full_rows_keep_their_cache_through_the_start_step() -> None:
    schema = small_schema()
    engine = Engine(
        tiny_model(schema),
        rows=2,
        t_max=8,
        generator=torch.Generator().manual_seed(12),
    )
    segments = [random_segment(schema, 3, seed=seed, starts=False) for seed in (13, 14)]
    engine.prefill(
        torch.tensor([0, 1]),
        Prefix(
            cells=torch.stack([s.cells for s in segments]),
            aux=torch.stack([s.aux for s in segments]),
            actions=torch.stack([s.actions for s in segments]),
            starts_episode=False,
        ),
    )
    assert engine.ensure_room() == 1
    engine.decide(no_done(engine))
    assert engine.state.length.tolist() == [8, 8]
    full = state_of(engine, row=0), state_of(engine, row=1)
    engine.start(engine.control())
    for row, before in enumerate(full):
        after = state_of(engine, row=row)
        for name, value in before.items():
            assert torch.equal(after[name], value), (row, name)


def test_inactive_rows_keep_their_state_when_their_job_ends() -> None:
    engine = Engine(
        tiny_model(small_schema()),
        rows=2,
        t_max=16,
        generator=torch.Generator().manual_seed(9),
    )
    engine.step(no_done(engine))
    before = state_of(engine, row=1)
    control = no_done(engine)
    control.job_tokens[:, 1, 0] = done_id(done=True)
    control.active[1] = False
    engine.step(control)
    after = state_of(engine, row=1)
    for name, value in before.items():
        assert torch.equal(after[name], value), name
    assert int(engine.state.length[0]) == 5
    assert engine.start(engine.control()).started.tolist() == [True, False]


def test_start_forces_tokens_only_in_starting_rows() -> None:
    schema = small_schema()
    engine = Engine(
        tiny_model(schema),
        rows=2,
        t_max=16,
        generator=torch.Generator().manual_seed(18),
    )
    engine.step(no_done(engine))
    engine.reset(torch.tensor([True, False]))
    control = forced_control(engine, random_segment(schema, 1, seed=19), 0)
    begun = engine.start(control)
    assert begun.started.tolist() == [True, False]
    assert torch.equal(begun.job.tokens[0], control.start_tokens[0])
    # The idle row's masked job samples its frame instead of copying the force.
    assert not torch.equal(begun.job.tokens[1, 2:], control.start_tokens[1, 2:])


def test_copied_row_scores_forced_tokens_like_its_source() -> None:
    schema = small_schema()
    engine = Engine(
        tiny_model(schema),
        rows=2,
        t_max=16,
        generator=torch.Generator().manual_seed(10),
    )
    engine.step(no_done(engine))
    engine.copy_row(0, 1)
    segment = random_segment(schema, 1, seed=11)
    decision = engine.decide(forced_control(engine, segment, 0))
    torch.testing.assert_close(decision.job.logp[0], decision.job.logp[1])
    torch.testing.assert_close(decision.action_logp[0], decision.action_logp[1])


def test_set_frame_replaces_the_frame_the_next_decision_encodes() -> None:
    schema = small_schema()
    engine = Engine(
        tiny_model(schema),
        rows=2,
        t_max=16,
        generator=torch.Generator().manual_seed(24),
    )
    engine.step(no_done(engine))
    segment = random_segment(schema, 1, seed=25)
    engine.set_frame(1, cells=segment.cells[0].long(), aux=segment.aux[0].long())
    assert torch.equal(engine.state.cells[1], segment.cells[0].long())
    assert torch.equal(engine.state.aux[1], segment.aux[0].long())
    assert not torch.equal(engine.state.cells[0], engine.state.cells[1])


def test_action_logits_are_the_distribution_decide_samples_from() -> None:
    schema = small_schema()
    engine = Engine(
        tiny_model(schema, global_layers=2),
        rows=2,
        t_max=16,
        generator=torch.Generator().manual_seed(26),
    )
    engine.step(no_done(engine))
    before = [state_of(engine, row=row) for row in range(2)]
    logp = engine.action_logits().log_softmax(-1)
    for row, state in enumerate(before):
        for name, value in state_of(engine, row=row).items():
            assert torch.equal(value, state[name]), (row, name)
    control = no_done(engine)
    control.action_forced[:] = True
    control.action[:] = torch.tensor([3, 40])
    decision = engine.decide(control)
    torch.testing.assert_close(decision.action_logp, logp[[0, 1], [3, 40]])


class _FakeGraph:
    """A CUDA graph on the CPU: counts its captures; ``replay`` calls ``run``."""

    def __init__(self, log: list[str], index: int) -> None:
        self.log = log
        self.index = index
        self.captures = 0
        self.run: Callable[[], None] = _unwired

    def replay(self) -> None:
        """Log the replay and rerun what the graph captured."""
        self.log.append(f"replay {self.index}")
        self.run()


class _FakeStream:
    """A CUDA stream on the CPU, named for the log."""

    def __init__(self, name: str) -> None:
        self.name = name

    def wait_stream(self, stream: object) -> None:
        """Order nothing: the CPU runs in program order."""
        del stream


class _FakeCuda:
    """The CUDA calls ``GraphedStep`` makes, recorded on the CPU for one engine.

    A capture runs its body eagerly, as the CPU must, then restores the engine's
    state: a real capture records kernels without running them.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.log: list[str] = []
        self.graphs: list[_FakeGraph] = []
        self._capturing: _FakeGraph | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace ``torch.cuda``'s graph, stream and device calls with this fake's."""
        monkeypatch.setattr(torch.cuda, "default_generators", (self.engine.generator,))
        monkeypatch.setattr(torch.cuda, "device", _no_device)
        monkeypatch.setattr(torch.cuda, "Stream", lambda: _FakeStream("side"))
        monkeypatch.setattr(torch.cuda, "current_stream", lambda: _FakeStream("main"))
        monkeypatch.setattr(torch.cuda, "stream", self.stream)
        monkeypatch.setattr(torch.cuda, "CUDAGraph", self.new_graph)
        monkeypatch.setattr(torch.cuda, "graph", self.graph)
        for name in ("start", "decide"):
            monkeypatch.setattr(
                self.engine,
                name,
                partial(self._logged, name, getattr(self.engine, name)),
            )

    def new_graph(self) -> _FakeGraph:
        """Return a fresh graph."""
        self.graphs.append(_FakeGraph(self.log, len(self.graphs)))
        return self.graphs[-1]

    @contextlib.contextmanager
    def stream(self, stream: _FakeStream) -> Generator[None]:
        """Log entering and leaving ``stream``."""
        self.log.append(f"enter {stream.name}")
        yield
        self.log.append(f"exit {stream.name}")

    @contextlib.contextmanager
    def graph(self, graph: _FakeGraph, *, stream: _FakeStream) -> Generator[None]:
        """Capture into ``graph`` on ``stream``, leaving the engine's state as it was."""
        graph.captures += 1
        self.log.append(f"capture {graph.index} on {stream.name}")
        saved = [
            cast("Tensor", getattr(self.engine.state, f.name)).clone()
            for f in dataclasses.fields(self.engine.state)
        ]
        self._capturing = graph
        yield
        self._capturing = None
        for f, value in zip(dataclasses.fields(self.engine.state), saved, strict=True):
            cast("Tensor", getattr(self.engine.state, f.name)).copy_(value)

    def _logged(
        self,
        name: str,
        method: Callable[[Control], object],
        control: Control,
    ) -> object:
        """Log a warmup call of ``method``; captures and replays are logged above."""
        if self._capturing is None and not self.graphs:
            self.log.append(f"warmup {name}")
        return method(control)


def _unwired() -> None:
    """Fail a replay of a graph the test did not wire."""
    raise AssertionError("A graph replayed without a captured body.")


@contextlib.contextmanager
def _no_device(device: object) -> Generator[None]:
    """Enter no CUDA device: the CPU has one."""
    del device
    yield


def _rerun(
    method: Callable[[Control], object],
    control: Control,
    buffer: object,
) -> None:
    """Run ``method`` on ``control`` and copy its output into ``buffer``'s tensors."""
    _copy_into(buffer, method(control))


def _copy_into(target: object, source: object) -> None:
    """Copy every tensor of a dataclass tree ``source`` into ``target``'s."""
    if isinstance(target, Tensor):
        assert isinstance(source, Tensor)
        target.copy_(source)
        return
    assert dataclasses.is_dataclass(target)
    for f in dataclasses.fields(target):
        _copy_into(
            cast("object", getattr(target, f.name)),
            cast("object", getattr(source, f.name)),
        )


class _OtherVarlen(SdpaVarlen):
    """A varlen kernel the engine was never held to: a subclass may change the math."""


def _counting(
    prefill: Callable[[Tensor, Prefix], None],
    calls: list[int],
) -> Callable[[Tensor, Prefix], None]:
    """Wrap ``prefill`` to record how many rows each call loads."""

    def spy(rows: Tensor, prefix: Prefix) -> None:
        calls.append(len(rows))
        prefill(rows, prefix)

    return spy


def _reset_run(
    engine: Engine,
    step: Callable[[Control], tuple[StartResult, Decision]],
    *,
    reset: bool,
) -> list[tuple[StartResult, Decision]]:
    """Take three free steps; with ``reset``, reset row 0 after the first."""
    steps = [_clone_step(*step(no_done(engine)))]
    if reset:
        engine.reset(torch.tensor([True, False]))
    return steps + [_clone_step(*step(no_done(engine))) for _ in range(2)]


def _assert_reset_is_row_local(
    plain: list[tuple[StartResult, Decision]],
    reset: list[tuple[StartResult, Decision]],
) -> None:
    """Require a reset of row 0 to change row 0's outputs and no others."""
    # Steps 1 and 2 of the plain run begin no episode, so they skip the start.
    assert [start.started.tolist() for start, _ in plain] == [
        [True, True],
        [False, False],
        [False, False],
    ]
    assert bool(reset[1][0].started[0])
    # A near-uniform random model samples mostly from the shared noise, so the
    # reset shows in the log-probabilities rather than the tokens.
    assert not torch.equal(reset[1][1].job.logp[0], plain[1][1].job.logp[0])
    for (plain_start, plain_decision), (reset_start, reset_decision) in zip(
        plain,
        reset,
        strict=True,
    ):
        _assert_row_equal(
            plain_start,
            reset_start,
            plain_decision,
            reset_decision,
            row=1,
        )


def _clone_step(start: StartResult, decision: Decision) -> tuple[StartResult, Decision]:
    """Copy a step's outputs, which ``GraphedStep`` overwrites at its next call."""
    return (
        StartResult(
            started=start.started.clone(),
            job=dataclasses.replace(
                start.job,
                tokens=start.job.tokens.clone(),
                logp=start.job.logp.clone(),
            ),
        ),
        dataclasses.replace(
            decision,
            action=decision.action.clone(),
            job=dataclasses.replace(
                decision.job,
                tokens=decision.job.tokens.clone(),
                logp=decision.job.logp.clone(),
            ),
        ),
    )


def _targets(segment: Segment) -> list[tuple[int, bool, Tensor, Tensor]]:
    """Return each job's reward, done, and next frame: the start job first."""
    frames = len(segment.cells)
    jobs = [(0, False, segment.cells[0], segment.aux[0])]
    for step in range(len(segment.actions)):
        following = min(step + 1, frames - 1)
        jobs.append(
            (
                int(segment.reward[step]),
                bool(segment.done[step]),
                segment.cells[following],
                segment.aux[following],
            ),
        )
    return jobs


def _local_log_probs(
    model: WorldModel,
    local: Tensor,
    *,
    batch_targets: list[tuple[int, bool, Tensor, Tensor]],
) -> Tensor:
    """Return the training forward's log-probability of every local slot."""
    schema = model.schema
    reward = torch.tensor([t[0] for t in batch_targets]) + model.scalar_offset
    done = torch.tensor([t[1] for t in batch_targets]).long() + done_id(done=False)
    cells = torch.stack([t[2] for t in batch_targets]).long()
    aux = torch.stack([t[3] for t in batch_targets]).long() + model.scalar_offset
    scalars = scalar_nll(
        local[:, model.scalar_rows],
        allowed=model.scalar_allowed,
        target=torch.cat([reward[:, None], done[:, None], aux], dim=-1),
    ).nll
    board = cell_nll(
        local[:, 2 : 2 + schema.cell_slots],
        index_table=model.cell_index,
        target=cells,
    ).nll
    return -torch.cat([scalars[:, :2], board, scalars[:, 2:]], dim=-1)


def _assert_same_decision(first: Engine, second: Engine, *, seed: int) -> None:
    """Draw one free decision from each engine under one seed; require equality."""
    first.generator.manual_seed(seed)
    second.generator.manual_seed(seed)
    one = first.decide(no_done(first))
    two = second.decide(no_done(second))
    assert torch.equal(one.action, two.action)
    torch.testing.assert_close(one.action_logp, two.action_logp)
    assert torch.equal(one.job.tokens, two.job.tokens)
    torch.testing.assert_close(one.job.logp, two.job.logp)


def _assert_row_equal(
    start_a: StartResult,
    start_b: StartResult,
    decision_a: Decision,
    decision_b: Decision,
    *,
    row: int,
) -> None:
    """Require one row's start and decision outputs to be identical."""
    assert torch.equal(start_a.started[row], start_b.started[row])
    # A row's start job is an output only where the row began an episode.
    if bool(start_a.started[row]):
        assert torch.equal(start_a.job.tokens[row], start_b.job.tokens[row])
    assert torch.equal(decision_a.action[row], decision_b.action[row])
    assert torch.equal(decision_a.job.tokens[row], decision_b.job.tokens[row])
    assert torch.equal(decision_a.job.logp[row], decision_b.job.logp[row])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
