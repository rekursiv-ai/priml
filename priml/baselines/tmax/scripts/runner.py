#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Drive the pinned TMax rollout runtime from PriML.

This is a driver, not a second terminal-agent implementation. TMax owns the
environment pools, the vLLM engines, the tool parser, the reward functions,
and the accumulation semantics. PriML supplies the released configuration,
starts those components, keeps the one field the released writer omits, and
pushes freshly trained weights back into the engines.

Three pieces live here:

* :func:`build_session` constructs TMax's runtime under the released 4B
  settings, and :func:`prime_prompts` fills its prompt queue exactly as
  upstream's ``DataPreparationActor`` does -- called only AFTER the
  trainer-to-engine weight-transfer group exists.
* :func:`collect_update` takes one update's rollouts through TMax's own
  accumulation and reward path, and :class:`RolloutWriter` persists them in
  TMax's shard layout plus ``tool_mask``.
* :class:`WeightSync` pushes the learner's weights into the engines between
  updates, through vLLM's own weight-transfer engine, under the same
  ``language_model.`` namespace mapping upstream applies.

The rollout records carry every field of TMax's ``RolloutRecord`` plus
``tool_mask``. The extra field is load-bearing: TMax's released writer
(``rl_utils.save_rollouts_to_disk``) drops ``GenerationResult.masks``, and the
PriML learner refuses to fit the policy to sandbox-written tokens without it.

Running this module as a script collects a FIXED corpus and trains nothing,
which makes every update after the first off-policy. It is a corpus tool, not
the published recipe; ``experiments.exp000`` interleaves collection with
training through :mod:`priml.baselines.tmax.live`.
'''
# fmt: on


from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, Self, cast, override

import argparse
import hashlib
import importlib
import json
import os
import sys

from configgle import Fig
from treekle import ReadError, from_plain

from priml.baselines.tmax.scripts import upstream
from priml.baselines.tmax.scripts.custom_types import (
    ActorManagerHandle,
    ActorManagerModule,
    Batch,
    DataLoaderModule,
    DatasetTransformationModule,
    Engine,
    EnvironmentsUtilsModule,
    GenerationResult,
    GroundTruthUtilsModule,
    GrpoFastModule,
    GrpoUtilsModule,
    HasGetRuntimeEnv,
    HasRuntimeEnv,
    ModelUtilsModule,
    PromptLoader,
    Ray,
    RayQueue,
    RlUtilsModule,
    UtilsModule,
    VllmUtilsModule,
    WeightTransferBase,
    WeightTransferNccl,
)
from priml.baselines.tmax.scripts.rollout_writer import RolloutWriter
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
else:
    from wrapt import lazy_import

    # ~90 ms; only ``collect_update`` reaches it.
    np = lazy_import("numpy")


QWEN3_5_VLLM_PREFIX: Final = "language_model."
"""Namespace vLLM gives Qwen3.5/3.6 parameters that Hugging Face does not."""

SYSTEM_PROMPT: Final = Path(
    "training/open-instruct/scripts/train/debug/envs"
    "/swerl_vanillux_sandbox_system_prompt.txt",
)
"""The released Vanillux sandbox system prompt, inside the pinned checkout."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RayRuntimePolicy:
    """Environment explicitly propagated to Ray workers."""

    swerl_reset_failure_zero_reward: str = "1"
    vllm_use_v1: str = "1"
    vllm_allow_insecure_serialization: str = "1"
    vllm_disable_compile_cache: str = "1"
    nccl_cumem_enable: str = "0"

    def env_vars(self) -> dict[str, str]:
        """Return only correctness-critical, non-secret variables.

        Returns:
          env_vars: Variable name to value, applied to the driver and every worker.

        """
        return {
            "SWERL_RESET_FAILURE_ZERO_REWARD": self.swerl_reset_failure_zero_reward,
            "VLLM_USE_V1": self.vllm_use_v1,
            "VLLM_ALLOW_INSECURE_SERIALIZATION": (
                self.vllm_allow_insecure_serialization
            ),
            "VLLM_DISABLE_COMPILE_CACHE": self.vllm_disable_compile_cache,
            "NCCL_CUMEM_ENABLE": self.nccl_cumem_enable,
        }


class RunnerConfig(Fig["TMaxRolloutRunner"]):
    """The released TMax 4B rollout settings this driver reproduces.

    Every default is read off ``scripts/tmax/RL/qwen35_4b.sh`` at the pinned
    commit, or off the upstream parser default where the launcher is silent.
    """

    base_dir: Path | str | None = None
    """Root supplied by the experiment."""

    working_dir: Path | str = "/runs/tmax/exp000"
    """Run directory used to place generated rollout files."""

    checkpoint: Path = Path("/models/Qwen3.5-4B")
    output_dir: Path = Path("/rollouts")
    upstream_root: Path = Path("/datasets/tmax/upstream/tmax")
    run_name: str = "tmax"
    dataset_repo: str = "allenai/tmax-15k-open-instruct"
    """Published dataset identity recorded in manifests and documentation."""

    dataset_path: Path = Path(
        "/datasets/tmax/tmax-15k-open-instruct/data/train-00000-of-00001.parquet",
    )
    """Pinned local parquet consumed by upstream's existing data mixer.

    The public launcher names :attr:`dataset_repo`, but upstream hard-codes its
    revision to ``main``. ``prepare_data.py`` downloads that repository at the
    pinned revision and this local-file seam lets upstream load the exact rows
    without changing or copying its transformation code.
    """

    task_data_dir: Path = Path(
        "/datasets/tmax/tmax-15k-open-instruct/task-data.tar.gz.extracted",
    )
    """Pinned extracted terminal tasks used by TMax's sandbox environment."""

    dataset_split: str = "train"
    num_updates: int = 500
    start_step: int = 0
    num_prompts: int = 8
    samples_per_prompt: int = 32
    max_prompt_tokens: int = 2_048
    response_length: int = 65_536
    per_turn_max_tokens: int = 16_384
    max_steps: int = 64
    seed: int = 42
    temperature: float = 1.0
    top_p: float = 1.0
    num_engines: int = 16
    tensor_parallel_size: int = 1
    pool_size: int = 512
    active_sampling: bool = True
    filter_zero_std_samples: bool = True
    async_steps: int = 4
    """Updates of rollouts kept in flight ahead of the learner.

    The released ``--async_steps 4``. It sets three things at once upstream:
    the prompt queue is primed with this many updates' prompts, the queues are
    sized ``(async_steps + 1) * num_prompts``, and a result generated more
    than this many updates behind the current policy is dropped as stale.
    """

    inflight_updates: bool = True
    """Let engines take new weights without draining in-flight requests.

    The released ``--inflight_updates true``. Inside the engines it skips the
    drain in ``LLMRayActor.sleep`` and ``update_weights``, which is what keeps
    a 65k-token rollout from stalling every weight sync behind it.
    """

    enable_prefix_caching: bool = True
    """The released ``--vllm_enable_prefix_caching``."""

    gdn_prefill_backend: str | None = "triton"
    """The released ``--vllm_gdn_prefill_backend triton``.

    Qwen 3.5's gated delta-net prefill kernel; passed through to the vLLM
    engine arguments, where ``None`` leaves the engine default.
    """

    lm_head_fp32: bool = True
    """The released ``--lm_head_fp32 true``, patched into the engines too.

    The engines get upstream's vLLM fp32 head patch. The learner selects its
    explicit fp32 scoring capability, which casts hidden states and weight to
    fp32 BEFORE the matmul -- the same arithmetic without mutating its module.
    """

    gpu_memory_utilization: float = 0.9
    """Upstream's parser default; the released launcher does not override it."""

    enforce_eager: bool = False
    """Upstream's parser default; the released launcher does not override it."""

    system_prompt_override_file: Path | None = None
    """The agent's system prompt; ``None`` takes the released prompt file.

    The released launcher passes a path INSIDE the checkout
    (:data:`SYSTEM_PROMPT`), so the pinned commit carries the prompt and PriML
    does not keep a copy that could drift from it.
    """

    verification_reward: float = 1.0
    """The released ``--verification_reward 1.0``."""

    ray_runtime: RayRuntimePolicy = RayRuntimePolicy()
    """Allowlisted environment applied to the driver and every Ray worker."""

    @override
    def finalize(self) -> Self:
        self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
        self.checkpoint = resolve_working_dir(self.base_dir, self.checkpoint)
        self.output_dir = resolve_working_dir(self.working_dir, self.output_dir)
        self.upstream_root = resolve_working_dir(self.base_dir, self.upstream_root)
        self.dataset_path = resolve_working_dir(self.base_dir, self.dataset_path)
        self.task_data_dir = resolve_working_dir(self.base_dir, self.task_data_dir)
        if self.system_prompt_override_file is not None:
            self.system_prompt_override_file = resolve_working_dir(
                self.base_dir,
                self.system_prompt_override_file,
            )
        return super().finalize()

    @property
    def records_per_update(self) -> int:
        """Return the released 8-prompt by 32-sample batch size."""
        return self.num_prompts * self.samples_per_prompt

    @property
    def queue_size(self) -> int:
        """Return upstream's prompt and result queue bound.

        ``(async_steps + 1) * num_unique_prompts_rollout`` in ``grpo_fast``'s
        ``main``: the prompts in flight plus the update being consumed. There
        is no eval term because this driver runs no in-training evaluation.
        """
        return (self.async_steps + 1) * self.num_prompts

    @property
    def max_model_len(self) -> int:
        """Return upstream's vLLM context bound for these lengths."""
        return self.max_prompt_tokens + self.response_length

    def system_prompt(self) -> Path:
        """Return the agent system prompt this run uses."""
        if self.system_prompt_override_file is not None:
            return self.system_prompt_override_file
        return self.upstream_root / SYSTEM_PROMPT


class TMaxRolloutRunner:
    """Own the finalized, validated rollout configuration."""

    Config = RunnerConfig

    def __init__(self, config: RunnerConfig) -> None:
        """Keep Configgle's finalized copy and reject invalid prompt groups."""
        self.config = config
        if config.num_updates <= 0:
            raise ValueError("num_updates must be positive.")
        if config.start_step < 0:
            raise ValueError("start_step must be non-negative.")
        if config.num_prompts <= 0:
            raise ValueError("num_prompts must be positive.")
        if config.samples_per_prompt <= 1:
            raise ValueError("samples_per_prompt must be greater than one.")
        if config.max_prompt_tokens <= 0 or config.response_length <= 0:
            raise ValueError("Prompt and response lengths must be positive.")
        if config.per_turn_max_tokens <= 0 or config.max_steps <= 0:
            raise ValueError("per-turn length and max_steps must be positive.")
        if config.num_engines <= 0 or config.tensor_parallel_size <= 0:
            raise ValueError("Engine counts must be positive.")
        if config.pool_size <= 0:
            raise ValueError("pool_size must be positive.")
        if not config.dataset_repo:
            raise ValueError("dataset_repo must name the published dataset.")
        # Upstream's own two composition rules
        # (``StreamingDataLoaderConfig.__post_init__``): active sampling
        # refills slots the zero-spread filter empties, and it can only refill
        # them from rollouts already in flight.
        if config.active_sampling and not config.filter_zero_std_samples:
            raise ValueError(
                "active_sampling needs filter_zero_std_samples, matching TMax.",
            )
        if config.active_sampling and config.async_steps <= 1:
            raise ValueError(
                "active_sampling needs async_steps > 1, matching TMax: a "
                "refilled slot comes from the rollouts already in flight.",
            )
        if config.async_steps <= 0:
            raise ValueError("async_steps must be positive.")


def vllm_weight_name_mapper(model_name: str) -> Callable[[str], str] | None:
    """Return the HF-to-vLLM name mapper for a checkpoint, or ``None``.

    Upstream's ``_build_vlm_name_mapper``: Qwen3.5 and Qwen3.6 checkpoints are
    served by vLLM under a ``language_model.`` prefix that Hugging Face does not
    use, and the engines are built with ``language_model_only=True``. Both the
    metadata list and the tensors sent over NCCL must carry the mapped name, or
    vLLM cannot match a single parameter.

    Args:
      model_name: Checkpoint path or repo id the engines were built from.

    Returns:
      mapper: The prefixing mapper, or ``None`` for architectures whose vLLM
        and Hugging Face names already agree.

    """
    lowered = model_name.lower()
    if any(version in lowered for version in ("qwen3.5", "qwen3.6")):
        return lambda name: f"{QWEN3_5_VLLM_PREFIX}{name}"
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class UpstreamApi:
    """The small set of TMax callables used by the driver.

    Keeping these bindings in one object makes the orchestration testable
    without importing Ray, vLLM, Docker, or the upstream dependency graph.
    """

    ray: Ray
    ray_queue: RayQueue
    data_loader: DataLoaderModule
    dataset_transformation: DatasetTransformationModule
    environments_utils: EnvironmentsUtilsModule
    grpo_fast: GrpoFastModule
    grpo_utils: GrpoUtilsModule
    ground_truth_utils: GroundTruthUtilsModule
    model_utils: ModelUtilsModule
    rl_utils: RlUtilsModule
    utils: UtilsModule
    vllm_utils: VllmUtilsModule
    actor_manager: ActorManagerModule | None = None
    weight_transfer: tuple[WeightTransferBase, WeightTransferNccl] | None = None
    """vLLM's ``distributed.weight_transfer`` entry points, or ``None``.

    The transfer engine is vLLM's rather than TMax's; TMax only calls it. It
    is absent on a checkout whose vLLM predates the API, which makes weight
    synchronization unavailable rather than silently wrong.
    """


@dataclass(slots=True, kw_only=True)
class CollectionSession:
    """Live TMax objects shared across rollout updates."""

    api: UpstreamApi
    tokenizer: object
    dataset: object
    iter_dataloader: PromptLoader
    inference_queue: object
    prompt_queue: object
    generation_config: object
    base_env_config: object
    model_dims: object
    engines: list[Engine]
    pools: dict[str, object]
    started_ray: bool
    actor_manager: ActorManagerHandle


def load_upstream_api(
    root: Path | str | None = None,
    runtime_policy: RayRuntimePolicy | None = None,
) -> UpstreamApi:
    """Import the pinned TMax modules without making them PriML dependencies.

    Args:
      root: Checkout root; ``None`` takes the staged default. The commit is
        verified here, so no unpinned tree can produce rollouts.
      runtime_policy: Explicit environment propagated to imported runtimes.

    Returns:
      api: The bound upstream callables.

    """
    checkout = upstream.verify_checkout(root)
    policy = (runtime_policy or RayRuntimePolicy()).env_vars()
    os.environ.update(policy)
    path = str(upstream.open_instruct(checkout))
    if path not in sys.path:
        sys.path.insert(0, path)

    modules = {
        name: importlib.import_module(f"open_instruct.{name}")
        for name in (
            "data_loader",
            "dataset_transformation",
            "environments.tools.utils",
            "grpo_fast",
            "grpo_utils",
            "ground_truth_utils",
            "model_utils",
            "rl_utils",
            "utils",
            "vllm_utils",
            "actor_manager",
        )
    }
    ray = importlib.import_module("ray")
    ray_queue = importlib.import_module("ray.util.queue")
    try:
        base = importlib.import_module("vllm.distributed.weight_transfer.base")
        nccl = importlib.import_module("vllm.distributed.weight_transfer.nccl_engine")
    except ImportError:
        weight_transfer = None
    else:
        weight_transfer = (
            cast("WeightTransferBase", base),
            cast("WeightTransferNccl", nccl),
        )

    # ``importlib`` returns ``ModuleType``; each cast names the members the
    # pinned commit exposes, and the driver's tests exercise every one.
    return UpstreamApi(
        ray=cast("Ray", ray),
        ray_queue=cast("RayQueue", ray_queue),
        data_loader=cast("DataLoaderModule", modules["data_loader"]),
        dataset_transformation=cast(
            "DatasetTransformationModule",
            modules["dataset_transformation"],
        ),
        environments_utils=cast(
            "EnvironmentsUtilsModule",
            modules["environments.tools.utils"],
        ),
        grpo_fast=cast("GrpoFastModule", modules["grpo_fast"]),
        grpo_utils=cast("GrpoUtilsModule", modules["grpo_utils"]),
        ground_truth_utils=cast(
            "GroundTruthUtilsModule",
            modules["ground_truth_utils"],
        ),
        model_utils=cast("ModelUtilsModule", modules["model_utils"]),
        rl_utils=cast("RlUtilsModule", modules["rl_utils"]),
        utils=cast("UtilsModule", modules["utils"]),
        vllm_utils=cast("VllmUtilsModule", modules["vllm_utils"]),
        actor_manager=cast("ActorManagerModule", modules["actor_manager"]),
        weight_transfer=weight_transfer,
    )


def build_session(config: RunnerConfig, api: UpstreamApi) -> CollectionSession:
    """Construct TMax's pools, dataset, queues, and vLLM engines.

    The prompt queue is left EMPTY. Upstream creates the trainer-to-vLLM
    weight-transfer group and runs the initial weight sync BEFORE it starts
    ``DataPreparationActor``, which is what first puts prompts in the queue
    (``grpo_fast``'s ``initialize_weight_sync`` then
    ``_data_prep_actor.start.remote``). Priming here instead would let the
    prefetch workers -- started with the engines -- begin generating from the
    base checkpoint while the trainer is still joining the update group, so the
    first update could train on rollouts the learner never produced. Call
    :func:`prime_prompts` only after :meth:`WeightSync.initialize` has run.

    Args:
      config: The released rollout settings.
      api: The bound upstream modules.

    Returns:
      session: The live runtime with an empty prompt queue.

    Raises:
      FileNotFoundError: The pinned dataset rows or task data are missing.

    """
    config = config.make().config
    if not config.dataset_path.is_file():
        raise FileNotFoundError(
            f"Pinned TMax training rows are missing: {config.dataset_path}. "
            "Run scripts/prepare_data.py before launching exp000.",
        )

    if not config.task_data_dir.is_dir():
        raise FileNotFoundError(
            f"Pinned TMax task data are missing: {config.task_data_dir}. "
            "Run scripts/prepare_data.py before launching exp000.",
        )

    ray_was_initialized = api.ray.is_initialized()
    try:
        started_ray = _initialize_ray(config, api.ray)
        _require_checkpoint_on_ray_nodes(config.checkpoint, api.ray)
        return _build_initialized_session(config, api, started_ray=started_ray)
    except BaseException:
        if not ray_was_initialized and api.ray.is_initialized():
            api.ray.shutdown()
        raise


def prime_prompts(config: RunnerConfig, session: CollectionSession) -> int:
    """Fill the prompt queue with ``async_steps`` updates' prompts.

    This is the ONLY bulk push, exactly as upstream's
    ``DataPreparationActor`` does before entering its step loop. Each later
    update replenishes what it consumes, so a driver that also pushed per
    update would grow the queue without bound and keep handing the learner
    staler and staler rollouts. It runs after the initial weight sync, so no
    prompt can be served by an engine still holding the base checkpoint.

    Args:
      config: The released rollout settings.
      session: A built session whose queue is still empty.

    Returns:
      count: Prompts pushed, ``async_steps * num_prompts`` as upstream primes.

    """
    api = session.api
    count = config.async_steps * config.num_prompts
    for _ in range(count):
        api.data_loader.add_prompt_to_generator(
            next(session.iter_dataloader),
            session.iter_dataloader._epoch,  # noqa: SLF001 -- Upstream's own prompt producer reads it.
            session.prompt_queue,
            session.generation_config,
            is_eval=False,
            base_env_config=session.base_env_config,
        )
    return count


def collect_update(
    session: CollectionSession,
    config: RunnerConfig,
    step: int,
    *,
    max_result_age_steps: int | None,
) -> tuple[Batch, GenerationResult, Sequence[float]]:
    """Take one update's rollouts through TMax's accumulation and reward path.

    Args:
      session: The live TMax runtime.
      config: The released rollout settings.
      step: The learner step these rollouts will train, which upstream uses to
        decide which results are too stale to keep.
      max_result_age_steps: Maximum policy-version lag to retain, or ``None``
        when the policy is fixed and version lag does not represent staleness.

    Returns:
      batch: TMax's batch of retained rollouts.
      result: TMax's generation result, including ``masks``.
      advantages: Centered group advantages over the batch's rewards.

    Raises:
      RuntimeError: TMax retained no rollouts for this step.

    """
    api = session.api
    result, batch, _, _ = api.data_loader.accumulate_inference_batches(
        session.inference_queue,
        session.generation_config,
        num_prompts=config.num_prompts,
        model_dims=session.model_dims,
        tokenizer=session.tokenizer,
        dataset=session.dataset,
        base_env_config=session.base_env_config,
        actor_manager=session.actor_manager,
        active_sampling=config.active_sampling,
        filter_zero_std_samples=config.filter_zero_std_samples,
        # Always on, as upstream's preparation loop has it: one prompt is
        # pushed per result consumed, which is what holds the in-flight
        # population at the primed level instead of draining it.
        replenish_prompts=True,
        iter_dataloader=session.iter_dataloader,
        param_prompt_Q=session.prompt_queue,
        training_step=step,
        verbose=False,
        max_possible_score=config.verification_reward,
        requeue_on_timeout=True,
        max_result_age_steps=max_result_age_steps,
    )
    if result is None or batch is None:
        raise RuntimeError(
            f"TMax returned no retained rollouts at training step {step}.",
        )
    scores = np.asarray(batch.scores, dtype=np.float32)
    advantages = api.data_loader.compute_group_advantages(
        scores=scores,
        num_samples_per_prompt=config.samples_per_prompt,
        advantage_normalization_type="centered",
    )
    return batch, result, advantages.tolist()


def serialize_rollout_records(
    batch: Batch,
    result: GenerationResult,
    *,
    step: int,
    advantages: Sequence[float],
    num_samples_per_prompt: int,
    request_info_for_sample: Callable[[object, int], Mapping[str, object] | None],
) -> list[dict[str, object]]:
    """Add TMax's native records plus ``GenerationResult.masks``.

    Args:
      batch: TMax's batch of retained rollouts.
      result: TMax's generation result.
      step: The learner step these rollouts train.
      advantages: One advantage per sample.
      num_samples_per_prompt: Prompt-group size.
      request_info_for_sample: Upstream's per-sample request-info reader.

    Returns:
      records: One JSON-ready record per sample, with ``tool_mask``.

    Raises:
      ValueError: Scores or logprobs are missing, or fields are misaligned.

    """
    queries: list[Sequence[int]] = list(batch.queries)
    responses: list[Sequence[int]] = list(result.responses)
    finishes: list[str] = list(result.finish_reasons)
    masks: list[Sequence[int]] = list(result.masks)
    if batch.scores is None:
        raise ValueError("TMax rollout collection returned no reward scores.")
    scores: list[float] = list(batch.scores)
    logprobs = result.logprobs
    if logprobs is None:
        raise ValueError(
            "TMax rollout collection returned no logprobs; the DPPO learner "
            "requires use_vllm_logprobs.",
        )
    if not (
        len(queries)
        == len(responses)
        == len(finishes)
        == len(masks)
        == len(scores)
        == len(advantages)
        == len(logprobs)
    ):
        raise ValueError(
            "TMax batch/result fields have different sample counts; refusing "
            "to write a misaligned rollout shard.",
        )
    records: list[dict[str, object]] = []
    for index, query in enumerate(queries):
        response, mask = responses[index], masks[index]
        sample_logprobs = logprobs[index]
        if len(response) != len(mask) or len(response) != len(sample_logprobs):
            raise ValueError(
                f"sample_idx={index}: response, tool_mask, and logprobs must "
                "have equal lengths.",
            )
        records.append(
            {
                "step": step,
                "sample_idx": index,
                "prompt_idx": index // num_samples_per_prompt,
                "prompt_tokens": list(query),
                "response_tokens": list(response),
                "reward": float(scores[index]),
                "advantage": float(advantages[index]),
                "finish_reason": finishes[index],
                "dataset": batch.datasets[index],
                "ground_truth": batch.ground_truths[index],
                "request_info": request_info_for_sample(result.request_info, index),
                "logprobs": list(sample_logprobs),
                "tool_mask": list(mask),
            },
        )
    return records


def records_for_update(
    session: CollectionSession,
    config: RunnerConfig,
    step: int,
    *,
    max_result_age_steps: int | None,
    collect: Callable[
        [CollectionSession, RunnerConfig, int],
        tuple[Batch, GenerationResult, Sequence[float]],
    ]
    | None = None,
) -> list[dict[str, object]]:
    """Collect one update and return its serialized rollout records.

    Args:
      session: The live TMax runtime.
      config: The released rollout settings.
      step: The learner step these rollouts train.
      max_result_age_steps: Maximum policy lag to retain, or ``None``.
      collect: Replaces :func:`collect_update`; injected by tests.

    Returns:
      records: The update's serialized records.

    """
    if collect is None:
        batch, result, advantages = collect_update(
            session,
            config,
            step,
            max_result_age_steps=max_result_age_steps,
        )
    else:
        batch, result, advantages = collect(session, config, step)
    return serialize_rollout_records(
        batch,
        result,
        step=step,
        advantages=advantages,
        num_samples_per_prompt=config.samples_per_prompt,
        request_info_for_sample=session.api.rl_utils._get_request_info_for_sample,  # noqa: SLF001 -- The released writer's own per-sample reader.
    )


def run_updates(
    config: RunnerConfig,
    session: CollectionSession,
    *,
    collect_update: Callable[
        [CollectionSession, RunnerConfig, int],
        tuple[Batch, GenerationResult, Sequence[float]],
    ]
    | None = None,
) -> list[Path]:
    """Collect a FIXED corpus of updates; train nothing.

    Every update here is drawn from the same unchanging policy, so only the
    first is on-policy. ``experiments.exp000`` does not use this path.
    On restart, an incomplete physical suffix is removed before its update is
    collected again; complete earlier updates remain untouched.

    Args:
      config: The released rollout settings.
      session: The live TMax runtime.
      collect_update: Replaces :func:`collect_update`; injected by tests.

    Returns:
      paths: The shard written for each newly collected update.

    """
    writer = RolloutWriter(config.output_dir, config.run_name)
    writer.recover_incomplete_fixed_tail(config.records_per_update)
    done = writer.completed_steps(config.records_per_update)
    paths: list[Path] = []
    for step in range(config.start_step, config.start_step + config.num_updates):
        if step in done:
            continue
        records = records_for_update(
            session,
            config,
            step,
            max_result_age_steps=None,
            collect=collect_update,
        )
        paths.append(writer.write(step, records))
    return paths


def run(config: RunnerConfig, api: UpstreamApi | None = None) -> list[Path]:
    """Build the TMax runtime and collect a fixed rollout corpus.

    Args:
      config: The released rollout settings.
      api: Bound upstream modules; ``None`` imports the pinned checkout.

    Returns:
      paths: The shard written for each newly collected update.

    """
    config = config.make().config
    runtime = api or load_upstream_api(config.upstream_root, config.ray_runtime)
    session = build_session(config, runtime)
    try:
        # This path holds the weights fixed at the base checkpoint and never
        # syncs, so priming right after construction is safe.
        prime_prompts(config, session)
        return run_updates(config, session)
    finally:
        if session.started_ray and runtime.ray.is_initialized():
            runtime.ray.shutdown()


def main() -> int:
    """Parse the launcher's small reproducibility surface and collect traces.

    Returns:
      exit_code: Zero on success.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("_Flags", parser.parse_args())
    config = RunnerConfig(
        base_dir=flags.base_dir,
        working_dir=flags.working_dir,
        checkpoint=flags.checkpoint,
        output_dir=flags.output_dir,
        upstream_root=flags.upstream_root,
        run_name=flags.run_name,
        dataset_repo=flags.dataset_repo,
        dataset_path=flags.dataset_path,
        task_data_dir=flags.task_data_dir,
        dataset_split=flags.dataset_split,
        num_updates=flags.num_updates,
        start_step=flags.start_step,
        num_prompts=flags.num_prompts,
        samples_per_prompt=flags.samples_per_prompt,
        seed=flags.seed,
        system_prompt_override_file=flags.system_prompt_override_file,
    ).finalize()
    for path in run(config):
        print(path)
    return 0


class _Flags(Protocol):
    """Parsed command-line flags."""

    base_dir: Path
    working_dir: Path
    checkpoint: Path
    output_dir: Path
    upstream_root: Path
    run_name: str
    dataset_repo: str
    dataset_path: Path
    task_data_dir: Path
    dataset_split: str
    num_updates: int
    start_step: int
    num_prompts: int
    samples_per_prompt: int
    seed: int
    system_prompt_override_file: Path | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the runner's flags on ``parser``."""
    defaults = RunnerConfig()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/opt/scratch"),
        help="Root for input assets and generated run files.",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=defaults.working_dir,
        help="Logical run directory below base-dir.",
    )
    parser.add_argument("--checkpoint", type=Path, default=defaults.checkpoint)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=defaults.upstream_root,
        help=(
            f"Pinned TMax checkout ({upstream.UPSTREAM_REPO} at "
            f"{upstream.UPSTREAM_SHORT}); staged by prepare_data.py."
        ),
    )
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--dataset-repo", default=defaults.dataset_repo)
    parser.add_argument("--dataset-path", type=Path, default=defaults.dataset_path)
    parser.add_argument("--task-data-dir", type=Path, default=defaults.task_data_dir)
    parser.add_argument("--dataset-split", default=defaults.dataset_split)
    parser.add_argument("--num-updates", type=int, default=defaults.num_updates)
    parser.add_argument("--start-step", type=int, default=defaults.start_step)
    parser.add_argument("--num-prompts", type=int, default=defaults.num_prompts)
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=defaults.samples_per_prompt,
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--system-prompt-override-file",
        type=Path,
        default=None,
    )


def _initialized_runtime_env(ray: Ray) -> dict[str, object]:
    """Read an initialized Ray driver's runtime environment across Ray APIs."""
    context = ray.get_runtime_context()
    if isinstance(context, HasGetRuntimeEnv):
        runtime_env = context.get_runtime_env()
    elif isinstance(context, HasRuntimeEnv):
        runtime_env = context.runtime_env
    else:
        return {}
    try:
        return from_plain(runtime_env, dict[str, object])
    except ReadError:
        return {}


def _initialize_ray(config: RunnerConfig, ray: Ray) -> bool:
    """Start Ray with the allowlist, or verify a compatible existing runtime."""
    env_vars = config.ray_runtime.env_vars()
    os.environ.update(env_vars)
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": env_vars})
        return True
    existing = _initialized_runtime_env(ray).get("env_vars")
    if not isinstance(existing, Mapping) or any(
        cast("Mapping[object, object]", existing).get(name) != value
        for name, value in env_vars.items()
    ):
        raise RuntimeError(
            "Ray is already initialized without the required TMax runtime "
            "environment. Shut it down before constructing the live runtime.",
        )
    return False


def _checkpoint_inventory(
    checkpoint: str,
) -> tuple[tuple[str, int, str], ...] | None:
    """Return checkpoint paths, sizes, and SHA-256 digests visible here."""
    root = Path(checkpoint)
    if not root.is_dir():
        return None
    inventory: list[tuple[str, int, str]] = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory.append(
            (
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                digest.hexdigest(),
            ),
        )
    return tuple(inventory)


def _require_checkpoint_on_ray_nodes(checkpoint: Path, ray: Ray) -> None:
    """Fail before engine creation unless every live Ray node sees the checkpoint."""
    expected = _checkpoint_inventory(str(checkpoint))
    if expected is None or not expected:
        raise FileNotFoundError(
            f"TMax checkpoint is missing or empty on the driver: {checkpoint}.",
        )
    nodes = [node for node in ray.nodes() if node.get("Alive") is True]
    if not nodes:
        raise RuntimeError("The TMax Ray cluster has no live inference nodes.")
    probe = ray.remote(_checkpoint_inventory)
    refs: list[object] = []
    labels: list[str] = []
    for node in nodes:
        address = str(node.get("NodeManagerAddress", ""))
        if not address:
            raise RuntimeError("A live Ray node has no NodeManagerAddress.")
        labels.append(address)
        refs.append(
            probe.options(
                num_cpus=0,
                resources={f"node:{address}": 0.001},
            ).remote(str(checkpoint)),
        )
    observed = ray.get(refs)
    unavailable = [
        label
        for label, inventory in zip(labels, observed, strict=True)
        if inventory != expected
    ]
    if unavailable:
        raise RuntimeError(
            f"TMax checkpoint {checkpoint} is missing or differs on Ray node(s): "
            f"{', '.join(unavailable)}. Mount shared storage at the same path or "
            "stage the complete checkpoint on every inference node.",
        )


def _build_initialized_session(
    config: RunnerConfig,
    api: UpstreamApi,
    *,
    started_ray: bool,
) -> CollectionSession:
    """Build TMax actors after Ray and the shared checkpoint are ready."""
    model_config = api.model_utils.ModelConfig(
        model_name_or_path=str(config.checkpoint),
    )
    tokenizer_config = api.dataset_transformation.TokenizerConfig(
        tokenizer_name_or_path=str(config.checkpoint),
    )
    tokenizer = api.grpo_fast.make_tokenizer(tokenizer_config, model_config)

    tools_config = api.environments_utils.EnvsConfig(
        tools=["swerl_vanillux_sandbox"],
        tool_configs=[
            json.dumps(
                {
                    "task_data_dir": str(config.task_data_dir),
                    "test_timeout": 120,
                    "image": "python:3.12-slim",
                },
            ),
        ],
        tool_parser_type="vllm_qwen3_xml",
        max_steps=config.max_steps,
        per_turn_max_tokens=config.per_turn_max_tokens,
        pool_size=config.pool_size,
    )
    pools, tool_definitions, tool_stop_sequences = (
        api.grpo_fast.initialize_tools_and_envs(
            tools_config,
            tokenizer,
            pool_size=config.pool_size,
            dataset_mixer_list=[str(config.dataset_path), "1.0"],
            dataset_mixer_list_splits=[config.dataset_split],
        )
    )

    args = api.grpo_utils.GRPOExperimentConfig()
    args.seed = config.seed
    args.hf_entity = None
    args.model_name_or_path = str(config.checkpoint)
    args.verification_reward = config.verification_reward
    args.lm_head_fp32 = config.lm_head_fp32

    streaming_config = api.data_loader.StreamingDataLoaderConfig()
    streaming_config.dataset_mixer_list = [str(config.dataset_path), "1.0"]
    streaming_config.dataset_mixer_list_splits = [config.dataset_split]
    streaming_config.dataset_mixer_eval_list = []
    streaming_config.dataset_mixer_eval_list_splits = []
    streaming_config.max_prompt_token_length = config.max_prompt_tokens
    streaming_config.response_length = config.response_length
    streaming_config.num_unique_prompts_rollout = config.num_prompts
    streaming_config.num_samples_per_prompt_rollout = config.samples_per_prompt
    streaming_config.temperature = config.temperature
    streaming_config.active_sampling = config.active_sampling
    streaming_config.filter_zero_std_samples = config.filter_zero_std_samples
    streaming_config.async_steps = config.async_steps
    streaming_config.inflight_updates = config.inflight_updates
    streaming_config.mask_tool_use = True
    streaming_config.system_prompt_override_file = str(config.system_prompt())
    streaming_config.stop_strings = list(streaming_config.stop_strings or [])
    streaming_config.stop_strings.extend(tool_stop_sequences)

    train_dataset, _ = api.grpo_fast.setup_datasets(
        args,
        tokenizer_config,
        tokenizer,
        streaming_config,
        tool_definitions,
        tools_config.pass_tools_to_chat_template,
        configured_tool_call_names=tools_config.tool_call_names,
    )
    base_env_config = api.grpo_fast.build_base_env_config(tools_config, pools)
    reward_config = api.ground_truth_utils.RewardConfig(
        apply_verifiable_reward=True,
        verification_reward=config.verification_reward,
        verifier_functions=api.ground_truth_utils.build_all_verifiers(
            args,
            streaming_config,
        ),
    )
    generation_config = api.vllm_utils.SamplingConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.response_length,
        n=config.samples_per_prompt,
        stop=streaming_config.stop_strings,
        seed=config.seed,
        logprobs=1,
    )

    inference_queue = api.ray_queue.Queue(maxsize=config.queue_size)
    prompt_queue = api.ray_queue.Queue(maxsize=config.queue_size)
    args.enable_queue_dashboard = False
    vllm_config = api.data_loader.VLLMConfig()
    vllm_config.vllm_num_engines = config.num_engines
    vllm_config.vllm_tensor_parallel_size = config.tensor_parallel_size
    if api.actor_manager is None:
        raise RuntimeError("The upstream API does not expose ActorManager.")
    actor_manager = api.ray.remote(api.actor_manager.ActorManager).remote(
        {
            "Inference Results Queue": inference_queue,
            "Prompt Queue": prompt_queue,
        },
        args,
        streaming_config,
        vllm_config,
    )
    engines = api.vllm_utils.create_vllm_engines(
        config.num_engines,
        config.tensor_parallel_size,
        config.enforce_eager,
        str(config.checkpoint),
        str(config.checkpoint),
        None,
        config.seed,
        config.enable_prefix_caching,
        config.max_model_len,
        config.gpu_memory_utilization,
        tool_parser_type=tools_config.tool_parser_type,
        tool_definitions=tool_definitions,
        tool_stop_sequences=tool_stop_sequences,
        max_steps=config.max_steps,
        per_turn_max_tokens=config.per_turn_max_tokens,
        mask_tool_use=True,
        pools=pools,
        prompt_queue=prompt_queue,
        results_queue=inference_queue,
        actor_manager=actor_manager,
        reward_config=reward_config,
        train_dataset=train_dataset,
        eval_dataset=None,
        inflight_updates=config.inflight_updates,
        vllm_gdn_prefill_backend=config.gdn_prefill_backend,
        lm_head_fp32=config.lm_head_fp32,
    )
    model_dims = api.utils.ModelDims.from_hf_config(str(config.checkpoint))
    iter_dataloader = api.data_loader.HFDataLoader(
        dataset=train_dataset,
        batch_size=1,
        seed=config.seed,
        dp_rank=0,
        dp_world_size=1,
        work_dir=str(config.output_dir / "data-loader"),
        automatic_reshuffle=True,
        collator=api.data_loader.single_example_collator,
    )
    return CollectionSession(
        api=api,
        tokenizer=tokenizer,
        dataset=train_dataset,
        iter_dataloader=iter_dataloader,
        inference_queue=inference_queue,
        prompt_queue=prompt_queue,
        generation_config=generation_config,
        base_env_config=base_env_config,
        model_dims=model_dims,
        engines=engines,
        pools=pools,
        # A Ray actor handle exposes the class's methods as remote methods.
        actor_manager=cast("ActorManagerHandle", actor_manager),
        started_ray=started_ray,
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
