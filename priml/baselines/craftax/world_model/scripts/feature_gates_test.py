"""Check the feature gates script on the CPU: its history, reference, graph and report.

The script's CUDA calls run through ``_CpuTorch``, a stand-in for its ``torch``
whose graphs re-run the captured engine call on replay, so every graphed step
is the eager step it captured.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import partial, partialmethod
from typing import TYPE_CHECKING

import copy
import sys

from treekle import from_plain, json_loads

import pytest
import torch

from priml.baselines.craftax.world_model import feature
from priml.baselines.craftax.world_model.codec import decode
from priml.baselines.craftax.world_model.feature import (
    FeatureEngine,
    MaskedCacheAttention,
    Sliding,
    WorldModelFeature,
)
from priml.baselines.craftax.world_model.feature_test import (
    HOOK,
    KEEP,
    T_MAX,
    _drive,
    _Flash4Interface,
    _frames,
    _make_source,
    _sliding_config,
    _two_layer_model,
)
from priml.baselines.craftax.world_model.scripts import feature_gates
from priml.baselines.craftax.world_model.scripts.engine_checks_test import (
    _corpus,
)
from priml.baselines.craftax.world_model.scripts.feature_gates import (
    GraphedStep,
    HeldHistory,
    Span,
    attention_gap,
    compare,
    reference_features,
    run_steps,
    time_step,
    validation_prefixes,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence
    from pathlib import Path

    from torch import Tensor

    from priml.baselines.craftax.world_model.model import WorldModel


def test_held_history_opens_a_span_per_episode_window_reprefill_and_resume() -> None:
    history = HeldHistory(2, t_max=8, keep=2)
    for t in range(3):
        history.hook(t, steps=1)
        history.step(t, resets=[1] if t == 2 else [], windows=[0] if t == 1 else [])
    # Row 0: an episode, then a window at step 1; row 1: an episode, reset at 2.
    assert [(s.row, s.first, s.starts_episode, s.steps) for s in history.spans] == [
        (0, 0, True, [0]),
        (1, 0, True, [0, 1]),
        (0, 1, False, [1, 2]),
        (1, 2, True, [2]),
    ]
    assert history._length == [3, 2]
    # Rows of 3 and 2 positions: a block of 4 more steps overflows 8 in both.
    history.hook(3, steps=4)
    assert [(s.row, s.first) for s in history.spans[4:]] == [(0, 1), (1, 1)]
    assert history._length == [3, 3]
    history.resume(1, 9, first=5, anchored=True)
    assert history.spans[-1] == Span(row=1, first=5, starts_episode=True, steps=[])
    assert history._length[1] == 2 * 4 - 1 + 1


def test_eager_steps_match_the_reference_of_the_held_history(
    source: WorldModelFeature,
) -> None:
    rows, steps = 2, 10
    cells, aux, actions = _frames(rows, steps)
    engine = source.make_engine(rows=rows, device=torch.device("cpu"))
    history = HeldHistory(rows, t_max=T_MAX, keep=KEEP)
    got = run_steps(
        engine,
        engine,
        obs=decode(cells, aux),
        actions=actions,
        resets={5: [0]},
        windows={3: [1]},
        hook=HOOK,
        history=history,
    )
    assert [(s.row, s.first, s.starts_episode) for s in history.spans] == [
        (0, 0, True),
        (1, 0, True),
        (1, 3, False),
        (0, 5, True),
        (1, 8 - KEEP, False),
    ]
    want = reference_features(
        source.model,
        layers=2,
        cells=cells,
        aux=aux,
        actions=actions,
        spans=history.spans,
    )
    torch.testing.assert_close(got, want, rtol=0, atol=2e-6)
    # A row no span covers stays NaN.
    row_zero = [s for s in history.spans if s.row == 0]
    partial_want = reference_features(
        source.model,
        layers=2,
        cells=cells,
        aux=aux,
        actions=actions,
        spans=row_zero,
    )
    assert bool(partial_want[1].isnan().all())
    torch.testing.assert_close(partial_want[0], want[0], rtol=0, atol=2e-6)


def test_a_graphed_step_replays_the_eager_step_exactly(
    source: WorldModelFeature,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _CpuTorch.install(monkeypatch)
    rows, steps = 3, 10
    cells, aux, actions = _frames(rows, steps)
    runs: list[Tensor] = []
    for graphed in (False, True):
        engine = source.make_engine(rows=rows, device=torch.device("cpu"))
        runs.append(
            run_steps(
                GraphedStep(engine, rows=rows) if graphed else engine,
                engine,
                obs=decode(cells, aux),
                actions=actions,
                resets={5: [0], 9: [1]},
                hook=HOOK,
                history=HeldHistory(rows, t_max=T_MAX, keep=KEEP),
            ),
        )
    assert torch.equal(runs[0], runs[1])
    # One capture of the one plan, replayed once a step.
    assert fake.cuda.replays == steps
    assert len(fake.cuda.graphs) == 1


def test_a_graphed_step_captures_each_new_plan_on_first_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _CpuTorch.install(monkeypatch)
    config = _sliding_config()
    sliding = config.history
    assert isinstance(sliding, Sliding.Config)
    sliding.fallback_rows = 1
    source = config.make()
    rows, steps = 4, 12
    cells, aux, actions = _frames(rows, steps)
    obs = decode(cells, aux)
    starts: list[dict[int, bool]] = [
        {},
        {3: True},
        {3: True, 5: True},
        {3: True, 5: False},
    ]
    runs: list[Tensor] = []
    graphed: GraphedStep | None = None
    for wrap in (False, True):
        engine = source.make_engine(rows=rows, device=torch.device("cpu"))
        step = GraphedStep(engine, rows=rows) if wrap else engine
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
            graphed = step
    assert torch.equal(runs[0], runs[1])
    assert graphed is not None
    assert sorted(graphed.graphs) == [0, 1, 2, 4]
    assert len(fake.cuda.graphs) == 4


def test_compare_reports_rms_max_cosine_and_argmax_agreement() -> None:
    want = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    got = torch.tensor([[3.0, 0.0], [4.0, 0.0]])
    head = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    stats = compare(got, want, head)
    assert stats == {
        # The squared error 32 over 4 entries against 25 / 4.
        "rel_rms": pytest.approx((32 / 25) ** 0.5),
        "max_abs": 4.0,
        "min_cos": 0.0,
        "argmax_agree": 0.5,
        "points": 2.0,
    }
    same = compare(want, want, head)
    assert (same["rel_rms"], same["min_cos"], same["argmax_agree"]) == (0, 1, 1)


def test_the_prefixes_are_the_first_long_enough_validation_episodes(
    tmp_path: Path,
) -> None:
    corpus = _corpus(tmp_path)
    cells, aux, actions = validation_prefixes(corpus, rows=2, decisions=9)
    # Episode 3 (5 decisions) is too short; 8 and 9 qualify; training is skipped.
    assert actions.tolist() == [[8] * 9, [9] * 9]
    assert cells.shape == (2, 9, 99, 8)
    assert aux.shape == (2, 9, 51)
    # An episode of exactly ``decisions`` decisions, episode 8's 10, is too short.
    (_, _, longer) = validation_prefixes(corpus, rows=1, decisions=10)
    assert longer.tolist() == [[9] * 10]
    with pytest.raises(ValueError, match="fewer than 3 episodes over 9"):
        validation_prefixes(corpus, rows=3, decisions=9)


def test_attention_gap_measures_a_kernel_against_masked_sdpa_per_layer() -> None:
    source = _make_source(layers=2, dtype=torch.bfloat16)
    engine = source.make_engine(rows=2, device=torch.device("cpu"))
    cells, aux, actions = _frames(2, 3)
    run_steps(
        engine,
        engine,
        obs=decode(cells, aux),
        actions=actions,
        resets={},
        hook=0,
        history=HeldHistory(2, t_max=T_MAX, keep=KEEP),
    )
    masked = attention_gap(engine, attention=MaskedCacheAttention.Config().make())
    assert masked["kernel_vs_sdpa"] == 0
    assert masked["kernel_vs_fp32"] == masked["sdpa_vs_fp32"] > 0
    assert masked["max_out"] > 0
    zero = attention_gap(engine, attention=_zero_attention)
    # Zeros sit max_out from SDPA, so no further from float32 than that plus its gap.
    assert zero["kernel_vs_sdpa"] == masked["max_out"]
    assert zero["kernel_vs_fp32"] > zero["sdpa_vs_fp32"]
    # A kernel wrong in the first layer alone: the maximum over layers keeps it.
    first_wrong = attention_gap(engine, attention=_FirstLayerZero())
    assert first_wrong["kernel_vs_sdpa"] > 0


def test_time_step_divides_the_replays_and_shares_the_reprefill(
    source: WorldModelFeature,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _CpuTorch.install(monkeypatch)
    monkeypatch.setattr(feature_gates, "time", _Clock(0.5))
    reprefills: list[int] = []
    original = FeatureEngine._reprefill
    monkeypatch.setattr(
        FeatureEngine,
        "_reprefill",
        partialmethod(_counted_reprefill, original, reprefills),
    )
    frame = decode(*(part[0, 0] for part in _frames(1, 1)[:2]))
    timing = time_step(source, rows=2, frame=frame)
    # 32 replays timed at 0.25 ms each by the fake events; one capture at the start.
    assert timing == {
        "rows": 2.0,
        "fill_before": 3 * T_MAX // 4,
        "step_ms": 0.25,
        "reprefill_rows": 1.0,
        "reprefill_ms": 500.0,
        "reprefill_share_of_a_block": 500 / (500 + HOOK * 0.25),
    }
    assert fake.cuda.replays == 1 + 32
    # A warmup re-prefill, then the timed one, each of the first row.
    assert reprefills == [1, 1]


@pytest.mark.parametrize("attention", ["fa4", "masked"])
def test_main_writes_every_gate_and_exits_as_they_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attention: str,
) -> None:
    fake = _CpuTorch.install(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "flash_attn.cute.interface",
        _Flash4Interface(returns_tuple=False),
    )
    loads: list[tuple[str, Path, list[str]]] = []
    for module in (feature, feature_gates):
        monkeypatch.setattr(module, "load_world_model", partial(_load, loads))
    checkpoint, output = tmp_path / "step.pt", tmp_path / "out" / "gates.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "feature_gates.py",
            *(str(checkpoint), str(_corpus(tmp_path)), str(output)),
            *("--experiment", "exp.smoke", "--override", "a=1"),
            *("--episodes", "2", "--decisions", "9", "--replicate", "2"),
            *("--hook", str(HOOK), "--t-max", str(T_MAX), "--layers", "2"),
            *("--attention", attention, "--timing-rows", "2", "--compile"),
        ],
    )
    status = feature_gates.main()
    # The reference, then the engine's weights.
    assert loads == [("exp.smoke", checkpoint, ["a=1"])] * 2
    assert fake.compiled == [
        {"fullgraph": True, "dynamic": False, "mode": "max-autotune-no-cudagraphs"},
    ] * len(fake.compiled)
    assert fake.compiled
    report = json_loads(output.read_text(), dict[str, object])
    assert report["rows"] == 4
    # Every row re-prefills at the hook of step 8: steps 8.. come after it.
    assert report["reprefill_step"] == 2 * HOOK
    g2a = from_plain(report["g2a"], dict[str, dict[str, float]])
    assert list(g2a) == ["all", "before_reprefill", "after_reprefill"]
    assert g2a["all"]["points"] == 4 * 9
    assert g2a["before_reprefill"]["points"] == 4 * 8
    assert g2a["after_reprefill"]["points"] == 4
    assert g2a["all"]["argmax_agree"] == 1
    g2c = from_plain(report["g2c"], dict[str, float])
    assert (g2c["kernel_vs_sdpa"] == 0) is (attention == "masked")
    timing = from_plain(report["timing"], dict[str, float])
    assert (timing["rows"], timing["step_ms"]) == (2, 0.25)
    gates = from_plain(report["gates"], dict[str, bool])
    summary = g2a["all"]
    assert gates == {
        "g2a": summary["rel_rms"] <= 0.012
        and summary["min_cos"] >= 0.9998
        and summary["argmax_agree"] >= 0.995,
        "g2c": g2c["kernel_vs_fp32"] <= g2c["sdpa_vs_fp32"],
    }
    assert status == (0 if all(gates.values()) else 1)
    assert ("Flash4CacheAttention" in str(report["engine"])) is (attention == "fa4")


def test_main_exits_1_when_a_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _CpuTorch.install(monkeypatch)
    loads: list[tuple[str, Path, list[str]]] = []
    for module in (feature, feature_gates):
        monkeypatch.setattr(module, "load_world_model", partial(_load, loads))
    monkeypatch.setattr(feature_gates, "attention_gap", _failing_gap)
    output = tmp_path / "gates.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "feature_gates.py",
            *(str(tmp_path / "step.pt"), str(_corpus(tmp_path)), str(output)),
            *("--episodes", "1", "--decisions", "6", "--hook", str(HOOK)),
            *("--t-max", str(T_MAX), "--layers", "1", "--attention", "masked"),
            *("--timing-rows", "2"),
        ],
    )
    assert feature_gates.main() == 1
    report = json_loads(output.read_text(), dict[str, object])
    # Six steps fill 12 of 16 positions: no re-prefill, one G2a block before it.
    assert report["reprefill_step"] == 0
    g2a = from_plain(report["g2a"], dict[str, dict[str, float]])
    assert list(g2a) == ["all", "before_reprefill"]
    assert g2a["before_reprefill"] == g2a["all"]
    assert g2a["all"]["points"] == 6
    assert from_plain(report["gates"], dict[str, bool])["g2c"] is False


@pytest.fixture(scope="module")
def source() -> WorldModelFeature:
    """Return a float32 CPU source tapping both blocks of the smoke model."""
    return _make_source(layers=2)


class _FakeStream:
    """A CUDA stream on the CPU, which runs in program order."""

    def wait_stream(self, stream: _FakeStream) -> None:
        """Order nothing."""
        del stream


class _FakeGraph:
    """A CUDA graph whose replay re-runs the engine call captured into it."""

    def __init__(self, cuda: _FakeCuda) -> None:
        self.cuda = cuda
        self.body: Callable[[], object] | None = None

    def replay(self) -> None:
        """Count the replay, then run the captured call."""
        assert self.body is not None, "A graph replayed with nothing captured."
        self.cuda.replays += 1
        self.body()


class _FakeEvent:
    """A timing event that reads the replays since its start: 0.25 ms apiece."""

    def __init__(self, cuda: _FakeCuda, *, enable_timing: bool) -> None:
        assert enable_timing
        self.cuda = cuda
        self.replays = -1

    def record(self) -> None:
        """Note the replays so far."""
        self.replays = self.cuda.replays

    def elapsed_time(self, end: _FakeEvent) -> float:
        """Return 0.25 ms per replay between this record and ``end``'s."""
        assert 0 <= self.replays <= end.replays
        return 0.25 * (end.replays - self.replays)


class _FakeCuda:
    """The ``torch.cuda`` calls the script makes, on the CPU."""

    def __init__(self) -> None:
        self.graphs: list[_FakeGraph] = []
        self.capturing: _FakeGraph | None = None
        self.replays = 0
        # torch.cuda's class names, bound to this fake's factories.
        self.CUDAGraph = self.new_graph
        self.Stream = _FakeStream
        self.Event = self.new_event

    def new_graph(self) -> _FakeGraph:
        """Return a new graph."""
        self.graphs.append(_FakeGraph(self))
        return self.graphs[-1]

    def new_event(self, *, enable_timing: bool) -> _FakeEvent:
        """Return a timing event."""
        return _FakeEvent(self, enable_timing=enable_timing)

    def current_stream(self) -> _FakeStream:
        """Return the main stream."""
        return _FakeStream()

    def stream(self, stream: _FakeStream) -> nullcontext[None]:
        """Enter ``stream``: nothing to do on the CPU."""
        del stream
        return nullcontext()

    @contextmanager
    def graph(self, graph: _FakeGraph, *, stream: _FakeStream) -> Generator[None]:
        """Capture the engine call made inside into ``graph``."""
        del stream
        self.capturing = graph
        yield
        self.capturing = None
        assert graph.body is not None, "The capture made no engine call."

    def synchronize(self) -> None:
        """Wait for nothing: the CPU is done."""


class _CpuTorch:
    """The script's ``torch`` on the CPU: CUDA faked, ``compile`` the identity."""

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.compiled: list[dict[str, object]] = []
        # The rest of torch the script reads, as torch's own.
        self.no_grad = torch.no_grad
        self.full, self.tensor, self.zeros = torch.full, torch.tensor, torch.zeros
        self.zeros_like, self.stack, self.randn = (
            torch.zeros_like,
            torch.stack,
            torch.randn,
        )
        self.Generator = torch.Generator
        self.nan, self.bool, self.int32 = torch.nan, torch.bool, torch.int32

    @classmethod
    def install(cls, monkeypatch: pytest.MonkeyPatch) -> _CpuTorch:
        """Swap the script's ``torch`` for one, recording engine calls; return it."""
        fake = cls()
        monkeypatch.setattr(feature_gates, "torch", fake)
        monkeypatch.setattr(
            FeatureEngine,
            "__call__",
            partialmethod(
                _recorded_call,
                cuda=fake.cuda,
                original=FeatureEngine.__call__,
            ),
        )
        return fake

    def device(self, kind: str) -> torch.device:
        """Return the CPU for the script's ``cuda``."""
        assert kind == "cuda"
        return torch.device("cpu")

    def compile(
        self,
        function: Callable[..., object],
        **options: object,
    ) -> Callable[..., object]:
        """Record the options and return ``function`` itself."""
        self.compiled.append(options)
        return function


class _Clock:
    """A ``time`` module whose clock advances ``step`` seconds a read."""

    def __init__(self, step: float) -> None:
        self.step = step
        self.now = 0.0

    def perf_counter(self) -> float:
        """Return the clock, then advance it."""
        self.now += self.step
        return self.now - self.step


def _recorded_call(
    engine: FeatureEngine,
    observation: Tensor,
    terminals: Tensor,
    previous_action: Tensor,
    *,
    cuda: _FakeCuda,
    original: Callable[[FeatureEngine, Tensor, Tensor, Tensor], Tensor],
) -> Tensor:
    """Run the engine's step; during a capture, also keep it as the graph's body."""
    call = partial(original, engine, observation, terminals, previous_action)
    if cuda.capturing is not None:
        cuda.capturing.body = call
    return call()


def _counted_reprefill(
    engine: FeatureEngine,
    original: Callable[[FeatureEngine, Tensor], None],
    counts: list[int],
    rows: Tensor,
) -> None:
    """Note how many rows re-prefill, then re-prefill them."""
    counts.append(len(rows))
    original(engine, rows)


def _zero_attention(q: Tensor, keys: Tensor, values: Tensor, seqused: Tensor) -> Tensor:
    """Attend to nothing: zeros shaped like ``q``."""
    del keys, values, seqused
    return torch.zeros_like(q)


class _FirstLayerZero:
    """Attend to nothing at the first call, a layer's, then as masked SDPA does."""

    def __init__(self) -> None:
        self.calls = 0
        self.masked = MaskedCacheAttention.Config().make()

    def __call__(
        self,
        q: Tensor,
        keys: Tensor,
        values: Tensor,
        seqused: Tensor,
    ) -> Tensor:
        """Return zeros on the first call, masked SDPA after."""
        self.calls += 1
        if self.calls == 1:
            return torch.zeros_like(q)
        return self.masked(q, keys, values, seqused)


def _failing_gap(
    engine: FeatureEngine,
    *,
    attention: object,
) -> dict[str, float]:
    """Return a gap whose kernel is further from float32 than SDPA."""
    del engine, attention
    return {
        "kernel_vs_sdpa": 1.0,
        "max_out": 1.0,
        "kernel_vs_fp32": 2.0,
        "sdpa_vs_fp32": 1.0,
    }


def _load(
    loads: list[tuple[str, Path, list[str]]],
    experiment: str,
    checkpoint: Path,
    *,
    overrides: Sequence[str],
) -> tuple[WorldModel, None]:
    """Stand in for ``load_world_model``: the smoke model, freshly copied."""
    loads.append((experiment, checkpoint, list(overrides)))
    return copy.deepcopy(_two_layer_model()), None


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
