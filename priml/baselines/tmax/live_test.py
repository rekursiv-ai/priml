"""Tests for the live rollout-to-learner seam."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from priml.baselines.tmax import live as live_module
from priml.baselines.tmax.live import LiveTMaxRolloutData, _records_from_payload
from priml.baselines.tmax.rollouts import RolloutRecord
from priml.baselines.tmax.scripts.runner import (
    CollectionSession,
    RolloutWriter,
    RunnerConfig,
)


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    data = cast(LiveTMaxRolloutData, config.make())
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
    data = cast(
        LiveTMaxRolloutData,
        LiveTMaxRolloutData.Config(
            working_dir="/unused",
            records_per_update=2,
            num_samples_per_prompt=2,
            runner=runner,
        ).make(),
    )
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
    data = cast(LiveTMaxRolloutData, config.make())
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
    data = cast(LiveTMaxRolloutData, config.make())
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

    def load_api(_root: object, _policy: object) -> object:
        return SimpleNamespace(
            ray=SimpleNamespace(
                _private=SimpleNamespace(
                    services=SimpleNamespace(
                        get_node_ip_address=lambda: "127.0.0.1",
                    ),
                ),
            ),
            utils=SimpleNamespace(find_free_port=lambda: 29500),
        )

    def build(_runner: object, _api: object) -> object:
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
            _session: object,
            *,
            tensor_parallel_size: int,
            name_mapper: object = None,
        ) -> None:
            captured["tensor_parallel_size"] = tensor_parallel_size
            captured["name_mapper"] = name_mapper
            events.append("sync-construct")

        def initialize(self, *, node_ip: str, port: int) -> None:
            captured["node_ip"] = node_ip
            captured["port"] = port
            events.append("sync-initialize")

    class _Writer:
        def __init__(self, _directory: object, _run_name: str) -> None:
            pass

        def committed_live_steps(self, _records: int) -> set[int]:
            return {0}

        def live_cursor_after(self, step: int, _records: int) -> dict[str, int]:
            assert step == 0
            return {"position": 10}

        def recover_uncommitted_live_tail(
            self,
            first_missing_step: int,
            _records: int,
        ) -> set[int]:
            assert first_missing_step == 1
            return set()

    def prime(_runner: object, _session: object) -> int:
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
    data = cast(LiveTMaxRolloutData, config.make())
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
    data = cast(
        LiveTMaxRolloutData,
        LiveTMaxRolloutData.Config(
            working_dir="/unused",
            records_per_update=2,
            num_samples_per_prompt=2,
            runner=RunnerConfig(num_updates=1, num_prompts=1, samples_per_prompt=2),
        ).make(),
    )
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
    data = cast(
        LiveTMaxRolloutData,
        LiveTMaxRolloutData.Config(
            working_dir="/unused",
            records_per_update=2,
            num_samples_per_prompt=2,
            runner=RunnerConfig(num_updates=2, num_prompts=1, samples_per_prompt=2),
        ).make(),
    )
    data._next_step = 0
    data._first_missing_step = 1
    data._session = cast(CollectionSession, cast(object, SimpleNamespace()))
    replayed = [cast("dict[str, object]", asdict(record)) for record in _records(0)]

    def committed_live_steps(_count: int) -> set[int]:
        return {0}

    def records_for_step(_step: int, _count: int) -> list[dict[str, object]]:
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
    data = cast(
        LiveTMaxRolloutData,
        LiveTMaxRolloutData.Config(
            working_dir="/unused",
            records_per_update=2,
            num_samples_per_prompt=2,
            runner=RunnerConfig(num_updates=2, num_prompts=1, samples_per_prompt=2),
        ).make(),
    )
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

    def committed_live_steps(_count: int) -> set[int]:
        return set()

    def write_live(
        step: int,
        _records: Sequence[Mapping[str, object]],
        prompt_loader_state_after: Mapping[str, object],
    ) -> None:
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
        _session: CollectionSession,
        _runner: RunnerConfig,
        step: int,
        *,
        max_result_age_steps: int | None,
    ) -> list[dict[str, object]]:
        assert max_result_age_steps == _runner.async_steps
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
