"""Drive the pinned TMax rollout runtime from PriML.

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
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final, Protocol, Self, cast, override


# The pinned upstream modules are optional runtime dependencies. They are
# imported only by ``load_upstream_api`` on the cluster, never by unit tests or
# by the offline learner.
# pyright: reportAny=false, reportExplicitAny=false
import argparse
import hashlib
import importlib
import json
import os
import sys

from configgle import Fig

import numpy as np
import torch

from priml.baselines.tmax.scripts import upstream
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


class _BatchLike(Protocol):
    """The upstream model-utils batch fields written by this adapter."""

    queries: Sequence[Sequence[int]]
    scores: Sequence[float] | None
    datasets: Sequence[object]
    ground_truths: Sequence[object]


class _ResultLike(Protocol):
    """The upstream GenerationResult fields written by this adapter."""

    responses: Sequence[Sequence[int]]
    finish_reasons: Sequence[str]
    masks: Sequence[Sequence[int]]
    logprobs: Sequence[Sequence[float]] | None
    request_info: object


ROLLOUT_SHARD_SIZE: Final = 10_000
"""TMax's released writer chooses a new shard per 10,000 prior samples."""

QWEN3_5_VLLM_PREFIX: Final = "language_model."
"""Namespace vLLM gives Qwen3.5/3.6 parameters that Hugging Face does not."""

SYSTEM_PROMPT: Final = Path(
    "training/open-instruct/scripts/train/debug/envs"
    "/swerl_vanillux_sandbox_system_prompt.txt",
)
"""The released Vanillux sandbox system prompt, inside the pinned checkout."""


@dataclass(frozen=True, slots=True)
class RayRuntimePolicy:
    """Environment explicitly propagated to Ray workers."""

    swerl_reset_failure_zero_reward: str = "1"
    vllm_use_v1: str = "1"
    vllm_allow_insecure_serialization: str = "1"
    vllm_disable_compile_cache: str = "1"
    nccl_cumem_enable: str = "0"

    def env_vars(self) -> dict[str, str]:
        """Return only correctness-critical, non-secret variables."""
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
        if not str(config.dataset_repo):
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


@dataclass(frozen=True, slots=True)
class UpstreamApi:
    """The small set of TMax callables used by the driver.

    Keeping these bindings in one object makes the orchestration testable
    without importing Ray, vLLM, Docker, or the upstream dependency graph.
    """

    ray: Any
    ray_queue: Any
    data_loader: ModuleType
    dataset_transformation: ModuleType
    environments_utils: ModuleType
    grpo_fast: ModuleType
    grpo_utils: ModuleType
    ground_truth_utils: ModuleType
    model_utils: ModuleType
    rl_utils: ModuleType
    utils: ModuleType
    vllm_utils: ModuleType
    actor_manager: ModuleType | None = None
    weight_transfer: Any = None
    """vLLM's ``distributed.weight_transfer`` entry points, or ``None``.

    The transfer engine is vLLM's rather than TMax's; TMax only calls it. It
    is absent on a checkout whose vLLM predates the API, which makes weight
    synchronization unavailable rather than silently wrong.
    """


@dataclass(slots=True)
class CollectionSession:
    """Live TMax objects shared across rollout updates."""

    api: UpstreamApi
    tokenizer: Any
    dataset: Any
    iter_dataloader: Any
    inference_queue: Any
    prompt_queue: Any
    generation_config: Any
    base_env_config: Any
    model_dims: Any
    engines: list[Any]
    pools: dict[str, Any]
    started_ray: bool
    actor_manager: Any = None


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
        weight_transfer = (base, nccl)

    return UpstreamApi(
        ray=ray,
        ray_queue=ray_queue,
        data_loader=modules["data_loader"],
        dataset_transformation=modules["dataset_transformation"],
        environments_utils=modules["environments.tools.utils"],
        grpo_fast=modules["grpo_fast"],
        grpo_utils=modules["grpo_utils"],
        ground_truth_utils=modules["ground_truth_utils"],
        model_utils=modules["model_utils"],
        rl_utils=modules["rl_utils"],
        utils=modules["utils"],
        vllm_utils=modules["vllm_utils"],
        actor_manager=modules["actor_manager"],
        weight_transfer=weight_transfer,
    )


def _initialized_runtime_env(ray: object) -> Mapping[str, object] | None:
    """Read an initialized Ray driver's runtime environment across Ray APIs."""
    runtime = cast(Any, ray)
    context = runtime.get_runtime_context()
    getter = getattr(context, "get_runtime_env", None)
    runtime_env = (
        getter() if callable(getter) else getattr(context, "runtime_env", None)
    )
    if not isinstance(runtime_env, Mapping):
        return None
    untyped = cast("Mapping[object, object]", runtime_env)
    if not all(isinstance(key, str) for key in untyped):
        return None
    return cast("Mapping[str, object]", runtime_env)


def _initialize_ray(config: RunnerConfig, ray: object) -> bool:
    """Start Ray with the allowlist, or verify a compatible existing runtime."""
    runtime = cast(Any, ray)
    env_vars = config.ray_runtime.env_vars()
    os.environ.update(env_vars)
    if not runtime.is_initialized():
        runtime.init(runtime_env={"env_vars": env_vars})
        return True
    runtime_env = _initialized_runtime_env(runtime)
    existing = runtime_env.get("env_vars") if runtime_env is not None else None
    if not isinstance(existing, Mapping) or any(
        existing.get(name) != value for name, value in env_vars.items()
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


def _require_checkpoint_on_ray_nodes(checkpoint: Path, ray: object) -> None:
    """Fail before engine creation unless every live Ray node sees the checkpoint."""
    expected = _checkpoint_inventory(str(checkpoint))
    if not expected:
        raise FileNotFoundError(
            f"TMax checkpoint is missing or empty on the driver: {checkpoint}.",
        )
    runtime = cast(Any, ray)
    nodes = [node for node in runtime.nodes() if node.get("Alive") is True]
    if not nodes:
        raise RuntimeError("The TMax Ray cluster has no live inference nodes.")
    probe = runtime.remote(_checkpoint_inventory)
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
    observed = runtime.get(refs)
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
    actor_manager_type = vars(api.actor_manager)["ActorManager"]
    actor_manager = api.ray.remote(actor_manager_type).remote(
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
        actor_manager=actor_manager,
        started_ray=started_ray,
    )


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
            getattr(session.iter_dataloader, "_epoch", 0),
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
) -> tuple[Any, Any, Sequence[float]]:
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


def _records_digest(records: Sequence[Mapping[str, object]]) -> str:
    """Hash records by sample index using the writer's JSON conversion."""
    normalized = [
        json.loads(json.dumps(record, default=_json_default)) for record in records
    ]
    normalized.sort(key=lambda record: int(record["sample_idx"]))
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RolloutWriter:
    """Persist rollout updates in TMax's shard layout, exactly once each.

    TMax's released writer appends, which is right for a single forward run
    and wrong for a resume: re-running a step would append a second copy of
    its records, and the learner would train the update twice. This writer
    reads what is already on disk, reports which steps are complete so a
    resumed campaign can skip collecting them, and refuses to write a
    ``(step, sample_idx)`` it already holds.
    """

    def __init__(self, directory: Path, run_name: str) -> None:
        """Index the shards already staged in ``directory``."""
        self.directory = Path(directory)
        self.run_name = run_name
        self._reset_index()
        self._load_index()

    def _reset_index(self) -> None:
        """Clear the in-memory view before loading or after tail recovery."""
        self._written: set[tuple[int, int]] = set()
        self._indices: dict[int, set[int]] = {}
        self._records: dict[int, list[dict[str, object]]] = {}
        self._locations: list[tuple[Path, int, int]] = []
        self._boundaries: dict[Path, set[int]] = {}
        self._markers: dict[int, dict[str, object]] = {}
        self._pending: dict[int, dict[str, object]] = {}
        self._torn_tail: tuple[Path, int] | None = None
        self._samples = 0

    def _load_index(self) -> None:
        """Load complete records and retain a recoverable final torn write."""
        shards = sorted(
            self.directory.glob(f"{self.run_name}_rollouts_*.jsonl"),
        )
        for shard_index, shard in enumerate(shards):
            payload = shard.read_bytes()
            lines = payload.splitlines(keepends=True)
            self._boundaries[shard] = {0}
            offset = 0
            for line_index, encoded in enumerate(lines):
                end = offset + len(encoded)
                if not encoded.strip():
                    self._boundaries[shard].add(end)
                    offset = end
                    continue
                is_final_line = (
                    shard_index == len(shards) - 1 and line_index == len(lines) - 1
                )
                try:
                    record = cast(
                        dict[str, object],
                        json.loads(encoded.decode("utf-8")),
                    )
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    if is_final_line:
                        self._torn_tail = (shard, offset)
                        break
                    raise ValueError(
                        f"Invalid rollout JSON before the recoverable tail in {shard}.",
                    ) from error
                if is_final_line and not encoded.endswith((b"\n", b"\r")):
                    self._torn_tail = (shard, offset)
                    break
                key = (
                    int(cast(int, record["step"])),
                    int(cast(int, record["sample_idx"])),
                )
                if key in self._written:
                    raise ValueError(
                        "Duplicate staged rollout record at step "
                        f"{key[0]} sample {key[1]} in {self.directory}.",
                    )
                self._written.add(key)
                self._indices.setdefault(key[0], set()).add(key[1])
                self._records.setdefault(key[0], []).append(record)
                self._locations.append((shard, offset, key[0]))
                self._boundaries[shard].add(end)
                self._samples += 1
                offset = end
        for marker_path in sorted(
            self.directory.glob(f"{self.run_name}_live_step_*.json"),
        ):
            if marker_path.name.endswith(".pending.json"):
                continue
            try:
                marker = cast(dict[str, object], json.loads(marker_path.read_text()))
                step = int(cast(int, marker["step"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid live rollout commit marker {marker_path}.",
                ) from error
            expected_path = (
                self.directory / f"{self.run_name}_live_step_{step:08d}.json"
            )
            if marker.get("version") != 1 or marker_path != expected_path:
                raise ValueError(
                    f"Invalid live rollout commit marker {marker_path}: "
                    "unsupported version or filename/step mismatch.",
                )
            if step in self._markers:
                raise ValueError(f"Duplicate live rollout marker for step {step}.")
            self._markers[step] = marker
        for pending_path in sorted(
            self.directory.glob(f"{self.run_name}_live_step_*.pending.json"),
        ):
            try:
                pending = cast(
                    dict[str, object],
                    json.loads(pending_path.read_text(encoding="utf-8")),
                )
                step = int(cast(int, pending["step"]))
                shard_name = str(pending["shard"])
                offset = int(cast(int, pending["byte_offset"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid live rollout write intent {pending_path}.",
                ) from error
            expected_path = (
                self.directory / f"{self.run_name}_live_step_{step:08d}.pending.json"
            )
            shard_path = self.directory / shard_name
            if (
                pending.get("version") != 1
                or pending_path != expected_path
                or shard_path.parent != self.directory
                or not shard_path.name.startswith(f"{self.run_name}_rollouts_")
                or shard_path.suffix != ".jsonl"
                or offset < 0
                or (shard_path.exists() and offset > shard_path.stat().st_size)
                or (
                    shard_path.exists()
                    and offset not in self._boundaries.get(shard_path, set())
                )
                or (not shard_path.exists() and offset != 0)
            ):
                raise ValueError(
                    f"Invalid live rollout write intent {pending_path}.",
                )
            if step in self._pending:
                raise ValueError(
                    f"Duplicate live rollout write intent for step {step}.",
                )
            self._pending[step] = pending

    def completed_steps(self, records_per_update: int) -> set[int]:
        """Return steps already holding a full update.

        Args:
          records_per_update: Records one complete update writes.

        Returns:
          steps: Steps a resumed campaign must not collect again.

        """
        self._require_complete_tail()
        expected = set(range(records_per_update))
        return {step for step, indices in self._indices.items() if indices == expected}

    def incomplete_steps(self, records_per_update: int) -> set[int]:
        """Return steps with records but fewer than one complete update."""
        self._require_complete_tail()
        expected = set(range(records_per_update))
        return {step for step, indices in self._indices.items() if indices != expected}

    def _require_complete_tail(self) -> None:
        """Reject non-live use of a shard whose final append was interrupted."""
        if self._torn_tail is not None:
            path, offset = self._torn_tail
            raise ValueError(
                f"Rollout shard {path} has an interrupted final record at byte "
                f"{offset}; only live resume can recover that speculative tail.",
            )

    def committed_live_steps(self, records_per_update: int) -> set[int]:
        """Return marker-backed steps whose record contract validates."""
        committed: set[int] = set()
        expected = list(range(records_per_update))
        for step, marker in self._markers.items():
            marker_count = marker.get("records_per_update")
            marker_indices = marker.get("sample_indices")
            cursor = marker.get("prompt_loader_state_after")
            if (
                marker_count != records_per_update
                or marker_indices != expected
                or not isinstance(cursor, Mapping)
            ):
                raise ValueError(
                    f"Live rollout marker for step {step} has an invalid "
                    "record or prompt-cursor contract.",
                )
            actual = sorted(self._indices.get(step, set()))
            if actual != expected:
                raise ValueError(
                    f"Live rollout marker for step {step} claims a complete "
                    f"update, but staged sample indices are {actual}.",
                )
            actual_digest = _records_digest(self._records.get(step, []))
            if marker.get("records_sha256") != actual_digest:
                raise ValueError(
                    f"Live rollout marker for step {step} does not match the "
                    "staged record contents.",
                )
            committed.add(step)
        return committed

    def live_cursor_after(
        self,
        step: int,
        records_per_update: int,
    ) -> Mapping[str, object]:
        """Return the validated prompt-loader cursor committed after ``step``."""
        if step not in self.committed_live_steps(records_per_update):
            raise ValueError(f"Live rollout step {step} is not committed.")
        return cast(
            Mapping[str, object],
            self._markers[step]["prompt_loader_state_after"],
        )

    def records_for_step(
        self,
        step: int,
        records_per_update: int,
    ) -> list[dict[str, object]]:
        """Return one complete indexed step without reparsing rollout shards."""
        expected = list(range(records_per_update))
        records = self._records.get(step, [])
        indices = sorted(int(cast(int, record["sample_idx"])) for record in records)
        if indices != expected:
            raise ValueError(
                f"Staged live rollout step {step} must contain each sample index "
                f"0..{records_per_update - 1} exactly once.",
            )
        return sorted(
            (dict(record) for record in records),
            key=lambda record: int(cast(int, record["sample_idx"])),
        )

    def recover_uncommitted_live_tail(
        self,
        first_missing_step: int,
        records_per_update: int,
    ) -> set[int]:
        """Roll back only the speculative physical suffix after the frontier.

        A live commit marker is the durable boundary. A write intent records
        the exact shard offset before a new update is appended; older runs
        without intents are recovered from the first indexed uncommitted
        record. Recovery refuses any layout where committed or earlier-step
        data appears after that boundary.
        """
        committed = self.committed_live_steps(records_per_update)
        uncommitted = set(self._indices).difference(committed)
        invalid = sorted(step for step in uncommitted if step < first_missing_step)
        if invalid:
            raise ValueError(
                "Cannot recover uncommitted rollout records before the durable "
                f"frontier {first_missing_step}: {invalid}.",
            )
        pending_uncommitted = {step for step in self._pending if step not in committed}
        invalid_pending = sorted(
            step for step in pending_uncommitted if step < first_missing_step
        )
        if invalid_pending:
            raise ValueError(
                "Cannot recover rollout write intents before the durable "
                f"frontier {first_missing_step}: {invalid_pending}.",
            )

        shards = sorted(
            self.directory.glob(f"{self.run_name}_rollouts_*.jsonl"),
        )
        order = {path: index for index, path in enumerate(shards)}
        candidates = [
            (order[path], offset, path)
            for path, offset, step in self._locations
            if step in uncommitted
        ]
        if self._torn_tail is not None:
            path, offset = self._torn_tail
            candidates.append((order[path], offset, path))
        for step in pending_uncommitted:
            pending = self._pending[step]
            path = self.directory / str(pending["shard"])
            if path.exists():
                candidates.append(
                    (order[path], int(cast(int, pending["byte_offset"])), path),
                )

        recovered = uncommitted | pending_uncommitted
        if candidates:
            shard_index, byte_offset, boundary = min(candidates)
            committed_after_boundary = sorted(
                step
                for path, offset, step in self._locations
                if step in committed
                and (order[path], offset) >= (shard_index, byte_offset)
            )
            if committed_after_boundary:
                raise ValueError(
                    "Cannot recover a non-suffix rollout tail because committed "
                    f"steps follow it: {committed_after_boundary}.",
                )
            with boundary.open("r+b") as handle:
                handle.truncate(byte_offset)
                handle.flush()
                os.fsync(handle.fileno())
            for shard in shards[shard_index + 1 :]:
                shard.unlink()

        removed_metadata = False
        for step in tuple(self._pending):
            pending_path = (
                self.directory / f"{self.run_name}_live_step_{step:08d}.pending.json"
            )
            if step in committed or step in pending_uncommitted:
                pending_path.unlink(missing_ok=True)
                self._pending.pop(step)
                removed_metadata = True
        for temporary in self.directory.glob(
            f"{self.run_name}_live_step_*.json.tmp",
        ):
            temporary.unlink()
            removed_metadata = True
        if candidates or recovered or removed_metadata:
            _fsync_directory(self.directory)
        if candidates or recovered:
            self._reset_index()
            self._load_index()
        return recovered

    def recover_incomplete_fixed_tail(self, records_per_update: int) -> set[int]:
        """Remove incomplete fixed-corpus updates from the physical suffix.

        The fixed writer appends one whole update at a time. After interruption,
        every incomplete update must therefore be in the physical suffix.
        Recovery refuses to truncate if any complete update follows it.
        """
        expected = set(range(records_per_update))
        incomplete = {
            step for step, indices in self._indices.items() if indices != expected
        }
        shards = sorted(
            self.directory.glob(f"{self.run_name}_rollouts_*.jsonl"),
        )
        order = {path: index for index, path in enumerate(shards)}
        candidates = [
            (order[path], offset, path)
            for path, offset, step in self._locations
            if step in incomplete
        ]
        if self._torn_tail is not None:
            path, offset = self._torn_tail
            candidates.append((order[path], offset, path))
        if not candidates:
            return set()

        shard_index, byte_offset, boundary = min(candidates)
        complete_after_boundary = sorted(
            step
            for path, offset, step in self._locations
            if self._indices[step] == expected
            and (order[path], offset) >= (shard_index, byte_offset)
        )
        if complete_after_boundary:
            raise ValueError(
                "Cannot recover a non-suffix fixed rollout update because "
                f"complete steps follow it: {complete_after_boundary}.",
            )
        with boundary.open("r+b") as handle:
            handle.truncate(byte_offset)
            handle.flush()
            os.fsync(handle.fileno())
        for shard in shards[shard_index + 1 :]:
            shard.unlink()
        _fsync_directory(self.directory)
        self._reset_index()
        self._load_index()
        return incomplete

    def write(self, step: int, records: Sequence[Mapping[str, object]]) -> Path:
        """Append one update's records under TMax's shard name.

        Args:
          step: The learner step these records belong to.
          records: The update's rollout records.

        Returns:
          path: The shard written.

        Raises:
          ValueError: A record duplicates one already written, which would
            train its update twice.

        """
        self._require_complete_tail()
        keys = [
            (int(cast(int, record["step"])), int(cast(int, record["sample_idx"])))
            for record in records
        ]
        if any(record_step != step for record_step, _ in keys):
            raise ValueError(
                f"Step {step} received records belonging to another step.",
            )
        duplicates = sorted(self._written.intersection(keys))
        if duplicates:
            raise ValueError(
                f"Refusing to write {len(duplicates)} rollout record(s) already "
                f"staged under {self.directory}, starting at step "
                f"{duplicates[0][0]} sample {duplicates[0][1]}: appending them "
                "would train one update twice.",
            )
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"Step {step} produced duplicate (step, sample_idx) records.",
            )
        shard_idx = self._samples // ROLLOUT_SHARD_SIZE
        path = self.directory / f"{self.run_name}_rollouts_{shard_idx:06d}.jsonl"
        self.directory.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, default=_json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        for key in keys:
            self._written.add(key)
            self._indices.setdefault(key[0], set()).add(key[1])
        for record in records:
            normalized = cast(
                dict[str, object],
                json.loads(json.dumps(record, default=_json_default)),
            )
            self._records.setdefault(step, []).append(normalized)
        self._samples += len(keys)
        return path

    def write_live(
        self,
        step: int,
        records: Sequence[Mapping[str, object]],
        *,
        prompt_loader_state_after: Mapping[str, object],
    ) -> Path:
        """Append records, then atomically publish their durable commit marker."""
        indices = sorted(int(cast(int, record["sample_idx"])) for record in records)
        if indices != list(range(len(records))):
            raise ValueError(
                f"Live rollout step {step} must contain sample indices "
                f"0..{len(records) - 1} exactly once.",
            )
        shard_idx = self._samples // ROLLOUT_SHARD_SIZE
        path = self.directory / f"{self.run_name}_rollouts_{shard_idx:06d}.jsonl"
        self.directory.mkdir(parents=True, exist_ok=True)
        pending_path = (
            self.directory / f"{self.run_name}_live_step_{step:08d}.pending.json"
        )
        pending_temporary = pending_path.with_suffix(pending_path.suffix + ".tmp")
        pending: dict[str, object] = {
            "version": 1,
            "step": step,
            "shard": path.name,
            "byte_offset": path.stat().st_size if path.exists() else 0,
        }
        with pending_temporary.open("w", encoding="utf-8") as handle:
            json.dump(pending, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        pending_temporary.replace(pending_path)
        _fsync_directory(self.directory)
        self._pending[step] = pending

        path = self.write(step, records)
        marker: dict[str, object] = {
            "version": 1,
            "step": step,
            "records_per_update": len(records),
            "sample_indices": indices,
            "records_sha256": _records_digest(self._records[step]),
            "prompt_loader_state_after": prompt_loader_state_after,
        }
        marker_path = self.directory / f"{self.run_name}_live_step_{step:08d}.json"
        temporary = marker_path.with_suffix(marker_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(marker, handle, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker_path)  # noqa: PTH105 -- Atomic publication.
        _fsync_directory(self.directory)
        self._markers[step] = marker
        pending_path.unlink()
        self._pending.pop(step)
        _fsync_directory(self.directory)
        return path


def _fsync_directory(directory: Path) -> None:
    """Make file creation, replacement, and removal durable on Linux."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class WeightSync:
    """Push the learner's weights into TMax's vLLM engines between updates.

    This mirrors ``grpo_fast``'s weight-sync thread and its
    ``setup_model_update_group``, over vLLM's own weight-transfer engine: the
    learner is NCCL rank zero and engine ``i`` occupies
    ``i * tensor_parallel_size + 1``. Engines sleep for the transfer, take the
    new weights, wake, and record which policy step they now serve, so TMax's
    staleness bound counts from a real number.

    What differs from upstream is only the SOURCE of the tensors. Upstream
    sends ``named_parameters()`` straight through a name mapper, which works
    because its trainer holds the Hugging Face module. PriML trains the native
    Qwen 3.5 model, whose parameter LAYOUT differs (the fused gate/up
    projection, among others), and a name mapper cannot express a layout
    change. So the tensors come from the same conversion the checkpoint export
    uses, which the weight-parity tests pin bit for bit -- names, dtypes, and
    shapes are read off that converted state rather than off the live module.
    The HF-to-vLLM NAMESPACE mapping upstream applies is still required and is
    applied here, to the metadata list and the transmitted tensors alike.

    A failed transfer is not recoverable in place: some engines may already
    hold new tensors. The engines are woken for cleanup, but admissions remain
    closed and this object rejects every later operation until the whole
    session and transfer group are rebuilt.
    """

    def __init__(
        self,
        session: CollectionSession,
        *,
        tensor_parallel_size: int,
        name_mapper: Callable[[str], str] | None = None,
    ) -> None:
        """Bind the engines this sync serves and the HF-to-vLLM name mapping."""
        self.session = session
        self.tensor_parallel_size = tensor_parallel_size
        self.name_mapper = name_mapper
        self._group: object | None = None
        self._initialized = False
        self._failed = False

    def initialize(self, *, node_ip: str, port: int) -> None:
        """Join the engines' weight-transfer group as NCCL rank zero.

        Args:
          node_ip: Address the engines dial back; the learner's node.
          port: A free port on that node.

        Raises:
          RuntimeError: The checkout's vLLM has no weight-transfer engine, so
            trained weights could never reach the engines.

        """
        if self._failed:
            raise RuntimeError(
                "WeightSync is unusable after a failed transfer; rebuild the "
                "collection session and transfer group.",
            )
        api = self.session.api
        if api.weight_transfer is None:
            raise RuntimeError(
                "This TMax checkout's vLLM exposes no weight-transfer engine "
                "(vllm.distributed.weight_transfer), so the learner's weights "
                "cannot reach the rollout engines and every update after the "
                "first would be off-policy.",
            )
        base, nccl = api.weight_transfer
        world_size = len(self.session.engines) * self.tensor_parallel_size + 1
        init_info = {
            "master_address": node_ip,
            "master_port": port,
            "world_size": world_size,
        }
        refs = [
            engine.init_weight_transfer_engine.remote(
                base.WeightTransferInitRequest(
                    init_info=init_info
                    | {"rank_offset": index * self.tensor_parallel_size + 1},
                ),
            )
            for index, engine in enumerate(self.session.engines)
        ]
        self._group = nccl.NCCLWeightTransferEngine.trainer_init(init_info)
        api.ray.get(refs)
        api.ray.get(
            [engine.set_model_step.remote(0) for engine in self.session.engines],
        )
        self._initialized = True

    def __call__(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        model_step: int,
    ) -> None:
        """Send one policy's weights to every engine.

        Args:
          weights: Hugging Face ``(name, tensor)`` pairs, in a stable order.
          model_step: The learner step these weights come from; the engines
            record it so stale rollouts can be identified.

        Raises:
          RuntimeError: :meth:`initialize` has not run, or an earlier transfer
            failed and the session must be rebuilt.

        """
        if self._failed:
            raise RuntimeError(
                "WeightSync is unusable after a failed transfer; rebuild the "
                "collection session and transfer group.",
            )
        if not self._initialized:
            raise RuntimeError("WeightSync.initialize must run before a sync.")
        api = self.session.api
        _, nccl = api.weight_transfer
        mapper = self.name_mapper
        pairs = [
            (mapper(name) if mapper is not None else name, tensor)
            for name, tensor in weights
        ]
        names = [name for name, _ in pairs]
        dtype_names = [str(tensor.dtype).split(".")[-1] for _, tensor in pairs]
        shapes = [list(tensor.shape) for _, tensor in pairs]
        engines = self.session.engines
        manager = self.session.actor_manager
        engines_awake = True
        try:
            api.ray.get(manager.set_should_stop.remote(True))
            engines_awake = False
            api.ray.get([engine.sleep.remote() for engine in engines])
            refs = [
                engine.update_weights.remote(
                    names,
                    dtype_names,
                    shapes,
                    packed=True,
                )
                for engine in engines
            ]
            nccl.NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(pairs),
                trainer_args=nccl.NCCLTrainerSendWeightsArgs(
                    group=self._group,
                    packed=True,
                ),
            )
            api.ray.get(refs)
            api.ray.get([engine.wake_up.remote() for engine in engines])
            engines_awake = True
            api.ray.get(
                [engine.set_model_step.remote(model_step) for engine in engines],
            )
        except BaseException:
            self._failed = True
            self._initialized = False
            if not engines_awake:
                with suppress(Exception):
                    api.ray.get([engine.wake_up.remote() for engine in engines])
            raise
        else:
            api.ray.get(manager.set_should_stop.remote(False))


def serialize_rollout_records(
    batch: _BatchLike,
    result: _ResultLike,
    *,
    step: int,
    advantages: Sequence[float],
    num_samples_per_prompt: int,
    request_info_for_sample: Callable[[Any, int], Mapping[str, object] | None],
) -> list[dict[str, object]]:
    """Add TMax's native records plus ``GenerationResult.masks``."""
    queries = list(batch.queries)
    responses = list(result.responses)
    finishes = list(result.finish_reasons)
    masks = list(result.masks)
    if batch.scores is None:
        raise ValueError("TMax rollout collection returned no reward scores.")
    scores = list(batch.scores)
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
    for index, (
        query,
        response,
        finish,
        mask,
        score,
        advantage,
        sample_logprobs,
    ) in enumerate(
        zip(
            queries,
            responses,
            finishes,
            masks,
            scores,
            advantages,
            logprobs,
            strict=True,
        ),
    ):
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
                "reward": float(score),
                "advantage": float(advantage),
                "finish_reason": str(finish),
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
        tuple[Any, Any, Sequence[float]],
    ]
    | None = None,
) -> list[dict[str, object]]:
    """Collect one update and return its serialized rollout records."""
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
        request_info_for_sample=cast(
            "Callable[[object, int], Mapping[str, object] | None]",
            vars(session.api.rl_utils)["_get_request_info_for_sample"],
        ),
    )


def run_updates(
    config: RunnerConfig,
    session: CollectionSession,
    *,
    collect_update: Callable[
        [CollectionSession, RunnerConfig, int],
        tuple[Any, Any, Sequence[float]],
    ]
    | None = None,
) -> list[Path]:
    """Collect a FIXED corpus of updates; train nothing.

    Every update here is drawn from the same unchanging policy, so only the
    first is on-policy. ``experiments.exp000`` does not use this path.
    On restart, an incomplete physical suffix is removed before its update is
    collected again; complete earlier updates remain untouched.
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
    """Build the TMax runtime and collect a fixed rollout corpus."""
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


def _json_default(value: object) -> object:
    """Match the released TMax writer's JSON conversion rules."""
    if is_dataclass(value):
        return asdict(cast(Any, value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "model_dump"):
        return cast(Any, value).model_dump()
    return str(value)


def main() -> int:
    """Parse the launcher's small reproducibility surface and collect traces."""
    defaults = RunnerConfig()
    parser = argparse.ArgumentParser(description=__doc__)
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
    flags = parser.parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
