"""Interleave TMax rollout collection with the PriML learner.

The live dataset owns only the seam between the two systems. TMax still
creates prompts, runs vLLM, executes the terminal environment, and computes
rewards. PriML receives one accepted update, packs it with the same code used
for recorded shards, trains it, and then sends the newly trained Hugging Face
weights back to vLLM before asking for the next update.
"""

from __future__ import annotations

from dataclasses import asdict, field
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

from configgle import Makes
from treekle import from_plain

from priml.baselines.tmax.data import TMaxRolloutData
from priml.baselines.tmax.rollouts import RolloutRecord
from priml.baselines.tmax.scripts.rollout_writer import RolloutWriter
from priml.baselines.tmax.scripts.runner import (
    CollectionSession,
    RunnerConfig,
    build_session,
    load_upstream_api,
    prime_prompts,
    records_for_update,
    vllm_weight_name_mapper,
)
from priml.baselines.tmax.scripts.weight_sync import WeightSync
from priml.runtime import is_rank_zero


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from torch import Tensor

    import torch

    from priml.train.custom_types import TrainStepProtocol
else:
    from wrapt import lazy_import

    # ~1050 ms; only the rank-zero broadcasts reach it.
    torch = lazy_import("torch")


@runtime_checkable
class ExportsHFState(Protocol):
    """A train step that gathers its policy in Hugging Face names."""

    def hf_state_dict(self) -> dict[str, Tensor]:
        """Return full policy tensors keyed by Hugging Face name."""
        ...


class LiveTMaxRolloutData(TMaxRolloutData):
    """Collect one TMax update immediately before each learner update."""

    class Config(
        Makes["LiveTMaxRolloutData"],
        TMaxRolloutData.Config,
    ):
        """Recorded-data settings plus the pinned live TMax runtime."""

        runner: RunnerConfig = field(default_factory=RunnerConfig)
        """Released TMax rollout settings for this experiment."""

    def __init__(self, config: Config) -> None:
        """Keep runtime startup out of config construction and checkpoint load."""
        config.runner.make()
        super().__init__(config)
        self.config = config
        self._step: TrainStepProtocol | None = None
        self._session: CollectionSession | None = None
        self._weight_sync: WeightSync | None = None
        self._writer: RolloutWriter | None = None
        self._next_step = 0
        self._prompt_loader_state: Mapping[str, object] | None = None
        self._first_missing_step = 0
        self._prompts_primed = False
        self._runtime_started = False
        self._synced_model_step: int | None = None

    @property
    def live_config(self) -> Config:
        """Narrow the inherited config to the live-only settings."""
        if not isinstance(self.config, LiveTMaxRolloutData.Config):
            raise TypeError("Live TMax data received a non-live config.")
        return self.config

    @override
    def bind_step(self, step: TrainStepProtocol) -> None:
        """Bind the pad id and the learner used for post-update sync."""
        super().bind_step(step)
        self._step = step

    @override
    def train_dataloader(self) -> Iterator[dict[str, object]]:
        """Yield exactly one packed live update at a time.

        The generator resumes after the learner has completed the previous
        update. That resume point is the only place the next weight sync runs,
        so no rollout can be collected from the old policy after an optimizer
        step.
        """
        if self._step is None:
            raise RuntimeError("Live TMax data must be bound to a train step.")
        self._ensure_runtime()
        runner = self.live_config.runner
        while self._next_step < runner.start_step + runner.num_updates:
            if (
                self._next_step >= self._first_missing_step
                and self._synced_model_step != self._step.global_step
            ):
                self._sync_latest_policy()
            records = self._records_for_next_step()
            self._next_step += 1
            yield self.pack_update(records)
        return

    def _ensure_runtime(self) -> None:
        """Start Ray/vLLM only after TrainLoop has restored its checkpoint."""
        if self._runtime_started:
            return
        if self._step is None:
            raise RuntimeError("Live TMax data must be bound to a train step.")
        restored_cursor = self._next_step
        self._next_step = self._step.global_step
        if restored_cursor not in (0, self._next_step):
            raise ValueError(
                "Live rollout cursor does not match the learner checkpoint: "
                f"dataset next_step={restored_cursor}, learner global_step="
                f"{self._next_step}.",
            )
        runner = self.live_config.runner
        if self._next_step < runner.start_step:
            raise ValueError(
                f"Learner resumed at step {self._next_step}, before the "
                f"configured rollout start_step {runner.start_step}.",
            )
        if self.config.records_per_update != runner.records_per_update:
            raise ValueError(
                "Live TMax records_per_update must equal RunnerConfig's "
                "num_prompts * samples_per_prompt.",
            )
        if self.config.num_samples_per_prompt != runner.samples_per_prompt:
            raise ValueError(
                "Live TMax num_samples_per_prompt must equal RunnerConfig's "
                "samples_per_prompt.",
            )
        session: CollectionSession | None = None
        if is_rank_zero():
            api = load_upstream_api(runner.upstream_root, runner.ray_runtime)
            session = build_session(runner, api)
            self._session = session
            self._weight_sync = WeightSync(
                session,
                tensor_parallel_size=runner.tensor_parallel_size,
                name_mapper=vllm_weight_name_mapper(str(runner.checkpoint)),
            )
            self._weight_sync.initialize(
                node_ip=api.ray._private.services.get_node_ip_address(),  # noqa: SLF001 -- Upstream's Ray node-IP API.
                port=api.utils.find_free_port(),
            )
            self._writer = RolloutWriter(
                runner.output_dir,
                runner.run_name,
            )
            committed = self._writer.committed_live_steps(runner.records_per_update)
            cursor = self._next_step
            while cursor in committed:
                self._prompt_loader_state = self._writer.live_cursor_after(
                    cursor,
                    runner.records_per_update,
                )
                cursor += 1
            later = sorted(step for step in committed if step > cursor)
            if later:
                raise ValueError(
                    "Live rollout commit markers are not consecutive from "
                    f"learner step {self._next_step}; first gap is {cursor}, "
                    f"but later committed steps exist: {later}.",
                )
            self._first_missing_step = cursor
            self._writer.recover_uncommitted_live_tail(
                self._first_missing_step,
                runner.records_per_update,
            )
            if self._prompt_loader_state is not None:
                session.iter_dataloader.load_state_dict(self._prompt_loader_state)
        replay_frontier: list[object] = [
            self._first_missing_step if is_rank_zero() else None,
        ]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(replay_frontier, src=0)
        self._first_missing_step = from_plain(replay_frontier[0], int)
        # Upstream triggers the initial weight sync before it starts
        # ``DataPreparationActor`` -- on a fresh run as well as a resumed one
        # (``grpo_fast``'s ``initialize_weight_sync`` fires, then
        # ``_data_prep_actor.start``) -- so the engines hold the trainer's
        # weights before any prompt is served. Mirroring that costs one
        # collective gather here: ``hf_state_dict`` is collective under FSDP,
        # so every rank participates.
        self._sync_latest_policy()
        self._runtime_started = True

    def _records_for_next_step(self) -> list[RolloutRecord]:
        """Replay a complete staged step or collect and persist a fresh one."""
        if not is_rank_zero():
            received: list[object] = [None]
            if torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(received, src=0)
            return _records_from_payload(
                from_plain(received[0], list[dict[str, object]]),
            )

        if self._writer is None or self._session is None:
            raise RuntimeError("Live TMax runtime is not initialized.")
        runner = self.live_config.runner
        complete = self._writer.committed_live_steps(
            runner.records_per_update,
        )
        if self._next_step in complete:
            records = _records_from_payload(
                self._writer.records_for_step(
                    self._next_step,
                    runner.records_per_update,
                ),
            )
        else:
            if self._next_step != self._first_missing_step:
                raise ValueError(
                    f"Expected first missing live rollout step "
                    f"{self._first_missing_step}, got {self._next_step}.",
                )
            if not self._prompts_primed:
                prime_prompts(runner, self._session)
                self._prompts_primed = True
            raw_records = records_for_update(
                self._session,
                runner,
                self._next_step,
                max_result_age_steps=runner.async_steps,
            )
            records = _records_from_payload(raw_records)
            loader_state = self._session.iter_dataloader.state_dict()
            self._writer.write_live(
                self._next_step,
                raw_records,
                prompt_loader_state_after=loader_state,
            )
            self._prompt_loader_state = loader_state
            self._first_missing_step += 1
        payload: list[object] = [[asdict(record) for record in records]]
        if torch.distributed.is_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)
        return records

    def _sync_latest_policy(self) -> None:
        """Gather the trained policy and let rank zero update vLLM."""
        if not isinstance(self._step, ExportsHFState):
            raise TypeError("Live TMax data needs a bound step exporting HF weights.")
        # This is collective for FSDP, even though only rank zero owns Ray.
        state = self._step.hf_state_dict()
        if is_rank_zero():
            if self._weight_sync is None:
                raise RuntimeError("Live TMax weight sync is not initialized.")
            self._weight_sync(
                state.items(),
                model_step=self._step.global_step,
            )
        self._synced_model_step = self._step.global_step
        del state

    @override
    def state_dict(self) -> dict[str, object]:
        """Return captured cursors without advancing or querying the loader."""
        state = super().state_dict()
        state["next_step"] = self._next_step
        if self._prompt_loader_state is not None:
            state["prompt_loader"] = self._prompt_loader_state
        return state

    @override
    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the cursor after the learner checkpoint is loaded."""
        super().load_state_dict(state_dict)
        self._next_step = from_plain(state_dict.get("next_step"), int, default=0)
        prompt_loader = state_dict.get("prompt_loader")
        self._prompt_loader_state = (
            None
            if prompt_loader is None
            else from_plain(prompt_loader, dict[str, object])
        )

    def close(self) -> None:
        """Stop the rank-zero Ray runtime, if live collection started it."""
        if self._session is not None and self._session.started_ray:
            ray = self._session.api.ray
            if ray.is_initialized():
                ray.shutdown()
        self._session = None
        self._weight_sync = None
        self._writer = None


# Extra serialized fields (``dataset``, ``ground_truth``, ``request_info``) belong to
# TMax and are dropped; every learner field is type-checked.
def _records_from_payload(
    payload: Sequence[Mapping[str, object]],
) -> list[RolloutRecord]:
    """Narrow a broadcast or serialized record payload."""
    names = RolloutRecord.__dataclass_fields__.keys()
    return [
        from_plain(
            {name: value for name, value in item.items() if name in names},
            RolloutRecord,
        )
        for item in payload
    ]
