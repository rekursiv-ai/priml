"""Tests for the live rollout-to-learner seam."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING, NoReturn, cast

import pytest
import torch

from priml.baselines.tmax import live as live_module
from priml.baselines.tmax.data import TMaxRolloutData
from priml.baselines.tmax.live import LiveTMaxRolloutData, _records_from_payload
from priml.baselines.tmax.rollouts import RolloutRecord
from priml.baselines.tmax.scripts.rollout_writer import RolloutWriter
from priml.baselines.tmax.scripts.runner import (
    CollectionSession,
    RunnerConfig,
)
from priml.baselines.tmax.scripts.weight_sync import WeightSync


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from priml.train.custom_types import TrainStepProtocol


def _records(step: int) -> list[RolloutRecord]:
    """Make one two-sample group with a nonzero centered advantage."""
    return [
        RolloutRecord(
            step=step,
            sample_idx=0,
            prompt_idx=0,
            prompt_tokens=(1,),
            response_tokens=(2, 3),
            logprobs=(-0.1, -0.2),
            reward=0.0,
            finish_reason="stop",
            tool_mask=(1, 1),
        ),
        RolloutRecord(
            step=step,
            sample_idx=1,
            prompt_idx=0,
            prompt_tokens=(1,),
            response_tokens=(2, 3),
            logprobs=(-0.1, -0.2),
            reward=1.0,
            finish_reason="stop",
            tool_mask=(1, 1),
        ),
    ]


def test_live_loader_syncs_between_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The new policy reaches the engines before the next update is collected.
    runner = RunnerConfig(
        num_updates=2,
        num_prompts=1,
        samples_per_prompt=2,
    )
    config = LiveTMaxRolloutData.Config(
        working_dir="/unused",
        records_per_update=2,
        num_samples_per_prompt=2,
        runner=runner,
    )
    data = config.make()
    step = SimpleNamespace(pad_token_id=0, global_step=0)
    data.bind_step(cast("TrainStepProtocol", step))
    events: list[str] = []

    def start() -> None:
        data._next_step = 0
        data._first_missing_step = 0
        data._synced_model_step = 0
        data._runtime_started = True

    def collect() -> list[RolloutRecord]:
        events.append(f"collect{data._next_step}")
        return _records(data._next_step)

    def sync() -> None:
        events.append(f"sync{data._next_step}")

    monkeypatch.setattr(data, "_ensure_runtime", start)
    monkeypatch.setattr(data, "_records_for_next_step", collect)
    monkeypatch.setattr(data, "_sync_latest_policy", sync)

    loader = data.train_dataloader()
    updates = [next(loader)]
    step.global_step = 1
    updates.append(next(loader))
    with pytest.raises(StopIteration):
        next(loader)

    assert len(updates) == 2
    assert events == ["collect0", "sync1", "collect1"]
    assert all(
        isinstance(update["rows"], tuple)
        and len(cast(tuple[object, ...], update["rows"])) == 1
        for update in updates
    )
    assert data.state_dict()["next_step"] == 2


def test_live_loader_does_not_sync_for_replayed_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Replayed steps skip the sync; only fresh collection resyncs.
    runner = RunnerConfig(num_updates=2, num_prompts=1, samples_per_prompt=2)
    data = LiveTMaxRolloutData.Config(
        working_dir="/unused",
        records_per_update=2,
        num_samples_per_prompt=2,
        runner=runner,
    ).make()
    step = SimpleNamespace(pad_token_id=0, global_step=0)
    data.bind_step(cast("TrainStepProtocol", step))
    events: list[str] = []

    def start() -> None:
        data._next_step = 0
        data._first_missing_step = 1
        data._synced_model_step = 0
        data._runtime_started = True

    monkeypatch.setattr(data, "_ensure_runtime", start)
    monkeypatch.setattr(
        data,
        "_records_for_next_step",
        lambda: events.append(f"records{data._next_step}") or _records(data._next_step),
    )
    monkeypatch.setattr(
        data,
        "_sync_latest_policy",
        lambda: events.append(f"sync{data._next_step}"),
    )

    loader = data.train_dataloader()
    next(loader)
    assert events == ["records0"]
    step.global_step = 1
    next(loader)
    assert events == ["records0", "sync1", "records1"]


def test_live_record_payload_round_trips_masks() -> None:
    # Broadcast payloads rebuild records with tuple fields and masks intact.
    records = _records_from_payload(
        [
            {
                "step": 3,
                "sample_idx": 0,
                "prompt_idx": 0,
                "prompt_tokens": [1],
                "response_tokens": [2],
                "logprobs": [-0.1],
                "reward": 1.0,
                "finish_reason": "stop",
                "tool_mask": [1],
            },
        ],
    )
    assert records[0].tool_mask == (1,)
    assert records[0].response_tokens == (2,)


def test_live_resume_rejects_a_cursor_that_disagrees_with_the_learner() -> None:
    # Resume requires the rollout cursor and the learner step to agree.
    config = LiveTMaxRolloutData.Config(
        working_dir="/unused",
        records_per_update=2,
        num_samples_per_prompt=2,
        runner=RunnerConfig(num_updates=1, num_prompts=1, samples_per_prompt=2),
    )
    data = config.make()
    data.bind_step(
        cast(
            "TrainStepProtocol",
            SimpleNamespace(pad_token_id=0, global_step=2),
        ),
    )
    data.load_state_dict({"next_step": 3})
    with pytest.raises(ValueError, match="cursor does not match"):
        data._ensure_runtime()


def test_live_runtime_rejects_a_different_prompt_group_size() -> None:
    # Equal update totals are insufficient: advantages center within prompt groups.
    config = LiveTMaxRolloutData.Config(
        working_dir="/unused",
        records_per_update=4,
        num_samples_per_prompt=2,
        runner=RunnerConfig(num_updates=1, num_prompts=1, samples_per_prompt=4),
    )
    data = config.make()
    data.bind_step(
        cast(
            "TrainStepProtocol",
            SimpleNamespace(pad_token_id=0, global_step=0),
        ),
    )
    with pytest.raises(ValueError, match="num_samples_per_prompt"):
        data._ensure_runtime()


def test_live_runtime_primes_prompts_after_the_weight_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engines must hold the learner's weights before any prompt is queued.

    Upstream joins the trainer-to-engine update group and runs the initial
    weight sync BEFORE it starts ``DataPreparationActor``, which is what first
    puts prompts in the queue -- on a fresh run as well as a resumed one
    (``grpo_fast``'s ``initialize_weight_sync`` fires before
    ``_data_prep_actor.start``). Priming first would let the engines' prefetch
    workers generate from the base checkpoint while the trainer is still
    joining the group, so the first update could train on rollouts the learner
    never produced. This fresh run (``global_step=0``) must therefore sync
    before it primes, exactly like a resumed one.
    """
    events: list[str] = []
    captured: dict[str, object] = {}

    def load_api(root: object, policy: object) -> object:
        del root, policy
        return SimpleNamespace(
            ray=SimpleNamespace(
                _private=SimpleNamespace(
                    services=SimpleNamespace(
                        get_node_ip_address=lambda: "127.0.0.1",
                    ),
                ),
            ),
            utils=SimpleNamespace(find_free_port=lambda: 29_500),
        )

    def build(runner: object, api: object) -> object:
        del runner, api
        events.append("build")

        def load_state_dict(state: Mapping[str, object]) -> None:
            events.append(f"restore-loader-{state['position']}")

        return SimpleNamespace(
            iter_dataloader=SimpleNamespace(
                load_state_dict=load_state_dict,
            ),
        )

    class _WeightSync:
        def __init__(
            self,
            session: object,
            *,
            tensor_parallel_size: int,
            name_mapper: object = None,
        ) -> None:
            del session
            captured["tensor_parallel_size"] = tensor_parallel_size
            captured["name_mapper"] = name_mapper
            events.append("sync-construct")

        def initialize(self, *, node_ip: str, port: int) -> None:
            captured["node_ip"] = node_ip
            captured["port"] = port
            events.append("sync-initialize")

    class _Writer:
        def __init__(self, directory: object, run_name: str) -> None:
            del directory, run_name

        def committed_live_steps(self, records: int) -> set[int]:
            del records
            return {0}

        def live_cursor_after(self, step: int, records: int) -> dict[str, int]:
            del records
            assert step == 0
            return {"position": 10}

        def recover_uncommitted_live_tail(
            self,
            first_missing_step: int,
            records: int,
        ) -> set[int]:
            del records
            assert first_missing_step == 1
            return set()

    def prime(runner: object, session: object) -> int:
        del runner, session
        events.append("prime")
        return 1

    def sync_initial() -> None:
        events.append("initial-sync")

    monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)
    monkeypatch.setattr(live_module, "load_upstream_api", load_api)
    monkeypatch.setattr(live_module, "build_session", build)
    monkeypatch.setattr(live_module, "WeightSync", _WeightSync)
    monkeypatch.setattr(live_module, "RolloutWriter", _Writer)
    monkeypatch.setattr(live_module, "prime_prompts", prime)

    runner = RunnerConfig(num_updates=1, num_prompts=1, samples_per_prompt=2)
    config = LiveTMaxRolloutData.Config(
        working_dir="/unused",
        records_per_update=2,
        num_samples_per_prompt=2,
        runner=runner,
    )
    data = config.make()
    data.bind_step(
        cast(
            "TrainStepProtocol",
            SimpleNamespace(pad_token_id=0, global_step=0),
        ),
    )
    data.load_state_dict({"next_step": 0, "prompt_loader": {"position": 9}})
    monkeypatch.setattr(data, "_sync_latest_policy", sync_initial)
    data._ensure_runtime()

    assert events == [
        "build",
        "sync-construct",
        "sync-initialize",
        "restore-loader-10",
        "initial-sync",
    ]
    assert data._first_missing_step == 1
    assert captured["tensor_parallel_size"] == runner.tensor_parallel_size
    # Qwen3.5's checkpoint needs the vLLM ``language_model.`` namespace prefix.
    mapper = captured["name_mapper"]
    assert callable(mapper)
    assert mapper("model.embed_tokens.weight") == (
        "language_model.model.embed_tokens.weight"
    )


def test_live_checkpoint_state_is_side_effect_free() -> None:
    # Checkpointing replays captured state and never queries the live loader.
    calls: list[str] = []
    data = LiveTMaxRolloutData.Config(
        working_dir="/unused",
        records_per_update=2,
        num_samples_per_prompt=2,
        runner=RunnerConfig(num_updates=1, num_prompts=1, samples_per_prompt=2),
    ).make()
    data._session = cast(
        CollectionSession,
        cast(
            object,
            SimpleNamespace(
                iter_dataloader=SimpleNamespace(
                    state_dict=lambda: calls.append("queried"),
                ),
            ),
        ),
    )
    data._prompt_loader_state = {"position": 12}
    assert data.state_dict()["prompt_loader"] == {"position": 12}
    assert calls == []


def test_live_replays_committed_step_without_priming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Replaying a committed step reads the shard; no prompts are queued.
    data = LiveTMaxRolloutData.Config(
        working_dir="/unused",
        records_per_update=2,
        num_samples_per_prompt=2,
        runner=RunnerConfig(num_updates=2, num_prompts=1, samples_per_prompt=2),
    ).make()
    data._next_step = 0
    data._first_missing_step = 1
    data._session = cast(CollectionSession, cast(object, SimpleNamespace()))
    replayed = [cast("dict[str, object]", asdict(record)) for record in _records(0)]

    def committed_live_steps(count: int) -> set[int]:
        del count
        return {0}

    def records_for_step(step: int, count: int) -> list[dict[str, object]]:
        del step, count
        return replayed

    data._writer = cast(
        RolloutWriter,
        cast(
            object,
            SimpleNamespace(
                committed_live_steps=committed_live_steps,
                records_for_step=records_for_step,
            ),
        ),
    )
    monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)

    def fail_prime(*_args: object) -> None:
        pytest.fail("replay must not prime")

    monkeypatch.setattr(
        live_module,
        "prime_prompts",
        fail_prime,
    )
    assert data._records_for_next_step() == _records(0)


def test_live_primes_once_and_commits_captured_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prompts prime once, and each update commits the loader cursor it ended on.
    data = LiveTMaxRolloutData.Config(
        working_dir="/unused",
        records_per_update=2,
        num_samples_per_prompt=2,
        runner=RunnerConfig(num_updates=2, num_prompts=1, samples_per_prompt=2),
    ).make()
    positions = iter(({"position": 1}, {"position": 2}))
    data._session = cast(
        CollectionSession,
        cast(
            object,
            SimpleNamespace(
                iter_dataloader=SimpleNamespace(state_dict=lambda: next(positions)),
            ),
        ),
    )
    commits: list[tuple[int, object]] = []

    def committed_live_steps(count: int) -> set[int]:
        del count
        return set()

    def write_live(
        step: int,
        records: Sequence[Mapping[str, object]],
        prompt_loader_state_after: Mapping[str, object],
    ) -> None:
        del records
        commits.append((step, prompt_loader_state_after))

    data._writer = cast(
        RolloutWriter,
        cast(
            object,
            SimpleNamespace(
                committed_live_steps=committed_live_steps,
                write_live=write_live,
            ),
        ),
    )
    data._first_missing_step = 0
    primes: list[int] = []
    monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)

    def prime(*_args: object) -> None:
        primes.append(1)

    monkeypatch.setattr(
        live_module,
        "prime_prompts",
        prime,
    )

    def records_for_update(
        session: CollectionSession,
        runner: RunnerConfig,
        step: int,
        *,
        max_result_age_steps: int | None,
    ) -> list[dict[str, object]]:
        del session
        assert max_result_age_steps == runner.async_steps
        return [cast("dict[str, object]", asdict(record)) for record in _records(step)]

    monkeypatch.setattr(
        live_module,
        "records_for_update",
        records_for_update,
    )

    data._records_for_next_step()
    data._next_step = 1
    data._records_for_next_step()

    assert primes == [1]
    assert commits == [(0, {"position": 1}), (1, {"position": 2})]
    assert data.state_dict()["prompt_loader"] == {"position": 2}


def _live_data(*, num_updates: int = 2, start_step: int = 0) -> LiveTMaxRolloutData:
    """Build a two-sample live dataset whose runtime is never started."""
    data = LiveTMaxRolloutData.Config(
        working_dir="/unused",
        records_per_update=2,
        num_samples_per_prompt=2,
        runner=RunnerConfig(
            num_updates=num_updates,
            start_step=start_step,
            num_prompts=1,
            samples_per_prompt=2,
        ),
    ).make()
    assert isinstance(data, LiveTMaxRolloutData)
    return data


class _ExportingStep:
    """A bound train step that records every HF-state gather."""

    def __init__(self, global_step: int = 0) -> None:
        self.pad_token_id = 0
        self.global_step = global_step
        self.gathers = 0

    def hf_state_dict(self) -> dict[str, torch.Tensor]:
        self.gathers += 1
        return {"w": torch.zeros(1)}


class _RecordingSync:
    """Stands in for ``WeightSync``; records the steps it pushed."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def __call__(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        model_step: int,
    ) -> None:
        assert [name for name, _ in weights] == ["w"]
        self.steps.append(model_step)


def _bound(data: LiveTMaxRolloutData, step: object) -> None:
    data.bind_step(cast("TrainStepProtocol", step))


def test_live_loader_requires_a_bound_step() -> None:
    with pytest.raises(RuntimeError, match="bound to a train step"):
        next(iter(_live_data().train_dataloader()))


def test_runtime_start_requires_a_bound_step() -> None:
    with pytest.raises(RuntimeError, match="bound to a train step"):
        _live_data()._ensure_runtime()


def test_runtime_start_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _live_data()
    data._runtime_started = True
    monkeypatch.setattr(live_module, "load_upstream_api", _never)
    data._ensure_runtime()


def test_runtime_rejects_a_resume_before_start_step() -> None:
    data = _live_data(start_step=3)
    _bound(data, _ExportingStep(global_step=1))
    with pytest.raises(ValueError, match="before the configured rollout start_step"):
        data._ensure_runtime()


def test_runtime_rejects_a_different_update_size() -> None:
    data = _live_data()
    data.config.records_per_update = 4
    _bound(data, _ExportingStep())
    with pytest.raises(ValueError, match="records_per_update"):
        data._ensure_runtime()


def _never(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("must not be called")


class _GapWriter:
    """A writer whose commit markers skip step 1."""

    def __init__(self, directory: object, run_name: str) -> None:
        del directory, run_name

    def committed_live_steps(self, records: int) -> set[int]:
        del records
        return {0, 2}

    def live_cursor_after(self, step: int, records: int) -> dict[str, int]:
        del step, records
        return {"position": 1}


class _ReplayWriter:
    """A writer holding fixed committed steps and their replayed records."""

    def __init__(
        self,
        *,
        committed: set[int],
        replayed: list[dict[str, object]] | None = None,
    ) -> None:
        self.committed = committed
        self.replayed = replayed or []

    def committed_live_steps(self, records_per_update: int) -> set[int]:
        del records_per_update
        return self.committed

    def records_for_step(
        self,
        step: int,
        records_per_update: int,
    ) -> list[dict[str, object]]:
        del step, records_per_update
        return self.replayed


class _StubSync:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def initialize(self, *, node_ip: str, port: int) -> None:
        del node_ip, port


def test_runtime_rejects_non_consecutive_commit_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _live_data()
    _bound(data, _ExportingStep())
    api = SimpleNamespace(
        ray=SimpleNamespace(
            _private=SimpleNamespace(
                services=SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1"),
            ),
        ),
        utils=SimpleNamespace(find_free_port=lambda: 29_500),
    )

    def load_api(*_args: object) -> object:
        return api

    def build(*_args: object) -> object:
        return SimpleNamespace()

    monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)
    monkeypatch.setattr(live_module, "load_upstream_api", load_api)
    monkeypatch.setattr(live_module, "build_session", build)
    monkeypatch.setattr(live_module, "WeightSync", _StubSync)
    monkeypatch.setattr(live_module, "RolloutWriter", _GapWriter)
    with pytest.raises(ValueError, match=r"first gap is 1.*\[2\]"):
        data._ensure_runtime()


def test_non_zero_rank_adopts_rank_zeros_replay_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _live_data()
    _bound(data, _ExportingStep())

    def broadcast(payload: list[object], *, src: int) -> None:
        assert src == 0
        payload[0] = 3

    monkeypatch.setattr(live_module, "is_rank_zero", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    monkeypatch.setattr(data, "_sync_latest_policy", lambda: None)
    data._ensure_runtime()
    assert data._first_missing_step == 3
    assert data._runtime_started


def test_non_zero_rank_receives_the_broadcast_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _live_data()
    sent = [asdict(record) for record in _records(4)]

    def broadcast(payload: list[object], *, src: int) -> None:
        assert src == 0
        payload[0] = sent

    monkeypatch.setattr(live_module, "is_rank_zero", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    assert data._records_for_next_step() == _records(4)


def test_rank_zero_collection_requires_a_started_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)
    with pytest.raises(RuntimeError, match="not initialized"):
        _live_data()._records_for_next_step()


def test_fresh_collection_must_be_the_first_missing_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _live_data()
    data._session = cast(CollectionSession, cast(object, SimpleNamespace()))
    data._writer = cast(RolloutWriter, cast(object, _ReplayWriter(committed=set())))
    data._next_step = 1
    data._first_missing_step = 0
    monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)
    with pytest.raises(ValueError, match="Expected first missing live rollout step 0"):
        data._records_for_next_step()


def test_rank_zero_broadcasts_the_records_it_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _live_data()
    data._session = cast(CollectionSession, cast(object, SimpleNamespace()))
    replayed = [asdict(record) for record in _records(0)]
    data._writer = cast(
        RolloutWriter,
        cast(object, _ReplayWriter(committed={0}, replayed=replayed)),
    )
    broadcasts: list[tuple[int, object]] = []

    def broadcast(payload: list[object], *, src: int) -> None:
        broadcasts.append((src, payload[0]))

    monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    assert data._records_for_next_step() == _records(0)
    assert broadcasts == [(0, replayed)]


def test_policy_sync_pushes_the_gathered_state_from_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _live_data()
    step = _ExportingStep(global_step=7)
    _bound(data, step)
    sync = _RecordingSync()
    data._weight_sync = cast(WeightSync, cast(object, sync))
    monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)
    data._sync_latest_policy()
    assert step.gathers == 1
    assert sync.steps == [7]
    assert data._synced_model_step == 7


def test_policy_sync_gathers_on_every_rank_but_sends_only_from_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _live_data()
    step = _ExportingStep(global_step=2)
    _bound(data, step)
    monkeypatch.setattr(live_module, "is_rank_zero", lambda: False)
    data._sync_latest_policy()
    assert step.gathers == 1
    assert data._synced_model_step == 2


def test_policy_sync_requires_an_exporting_step() -> None:
    data = _live_data()
    _bound(data, SimpleNamespace(pad_token_id=0, global_step=0))
    with pytest.raises(TypeError, match="exporting HF weights"):
        data._sync_latest_policy()


def test_policy_sync_requires_an_initialized_weight_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _live_data()
    _bound(data, _ExportingStep())
    monkeypatch.setattr(live_module, "is_rank_zero", lambda: True)
    with pytest.raises(RuntimeError, match="weight sync is not initialized"):
        data._sync_latest_policy()


@pytest.mark.parametrize(
    ("started_ray", "initialized", "shutdowns"),
    [(True, True, 1), (True, False, 0), (False, True, 0)],
)
def test_close_stops_only_a_ray_runtime_it_started(
    *,
    started_ray: bool,
    initialized: bool,
    shutdowns: int,
) -> None:
    data = _live_data()
    events: list[str] = []
    ray = SimpleNamespace(
        is_initialized=lambda: initialized,
        shutdown=lambda: events.append("shutdown"),
    )
    data._session = cast(
        CollectionSession,
        cast(
            object,
            SimpleNamespace(started_ray=started_ray, api=SimpleNamespace(ray=ray)),
        ),
    )
    data.close()
    assert events == ["shutdown"] * shutdowns
    assert data._session is None


def test_close_without_a_session_is_a_no_op() -> None:
    _live_data().close()


def test_load_state_dict_accepts_a_checkpoint_without_a_prompt_cursor() -> None:
    data = _live_data()
    data._prompt_loader_state = {"position": 1}
    data.load_state_dict({"next_step": 3})
    assert data._next_step == 3
    assert data._prompt_loader_state is None


def test_live_config_rejects_a_non_live_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _live_data()
    monkeypatch.setattr(data, "config", TMaxRolloutData.Config(working_dir="/unused"))
    with pytest.raises(TypeError, match="non-live config"):
        _ = data.live_config


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
