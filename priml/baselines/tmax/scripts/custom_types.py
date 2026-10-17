"""Typed views of the pinned TMax, Ray, and vLLM surfaces the driver calls.

None of these packages is a PriML dependency: ``runner.load_upstream_api``
imports them from the pinned checkout at run time. Each Protocol names only
the members the driver reads, so the checker verifies every call against the
shape the pinned commit exposes instead of treating the runtime as ``Any``.
Values the driver only passes through (tokenizers, queues, generation
configs) stay ``object``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from torch import Tensor


class RemoteMethod(Protocol):
    """A Ray actor method or task, invoked through ``.remote``."""

    def remote(self, *args: object, **kwargs: object) -> object:
        """Submit the call; return its object reference."""
        ...


class RemoteFunction(RemoteMethod, Protocol):
    """A Ray remote function that can be pinned to a node."""

    def options(self, **kwargs: object) -> RemoteMethod:
        """Return the function with scheduling options applied."""
        ...


class Engine(Protocol):
    """The vLLM engine actor methods the weight sync drives."""

    @property
    def init_weight_transfer_engine(self) -> RemoteMethod:
        """Join the trainer-to-engine NCCL group."""
        ...

    @property
    def sleep(self) -> RemoteMethod:
        """Release the engine's GPU memory for a transfer."""
        ...

    @property
    def wake_up(self) -> RemoteMethod:
        """Reacquire GPU memory after a transfer."""
        ...

    @property
    def update_weights(self) -> RemoteMethod:
        """Receive one packed weight broadcast."""
        ...

    @property
    def set_model_step(self) -> RemoteMethod:
        """Record the policy step the engine now serves."""
        ...


class ActorManagerHandle(Protocol):
    """TMax's actor manager, which gates new rollout admissions."""

    @property
    def set_should_stop(self) -> RemoteMethod:
        """Open or close admissions."""
        ...


class RayPrivateServices(Protocol):
    """``ray._private.services``."""

    def get_node_ip_address(self) -> str:
        """Return this node's address."""
        ...


class RayPrivate(Protocol):
    """``ray._private``."""

    @property
    def services(self) -> RayPrivateServices:
        """Return Ray's service helpers."""
        ...


class Ray(Protocol):
    """The ``ray`` module members the driver calls."""

    @property
    def _private(self) -> RayPrivate:
        """Return Ray's private helpers; upstream reads the node IP there."""
        ...

    def is_initialized(self) -> bool:
        """Return whether a Ray driver is connected."""
        ...

    def init(self, *, runtime_env: Mapping[str, object]) -> object:
        """Connect a Ray driver."""
        ...

    def shutdown(self) -> None:
        """Disconnect the Ray driver."""
        ...

    def nodes(self) -> list[Mapping[str, object]]:
        """Return the cluster's node records."""
        ...

    def get_runtime_context(self) -> object:
        """Return the driver's runtime context."""
        ...

    def remote(self, target: object, /) -> RemoteFunction:
        """Wrap a function or class as a Ray task or actor."""
        ...

    @overload
    def get(self, refs: list[object], /) -> list[object]: ...
    @overload
    def get(self, refs: object, /) -> object: ...
    def get(self, refs: object, /) -> object:
        """Block on one reference or a list of references."""
        ...


class RayQueue(Protocol):
    """``ray.util.queue``."""

    Queue: Callable[..., object]


@runtime_checkable
class HasGetRuntimeEnv(Protocol):
    """A Ray runtime context exposing ``get_runtime_env``."""

    def get_runtime_env(self) -> object:
        """Return the runtime environment."""
        ...


@runtime_checkable
class HasRuntimeEnv(Protocol):
    """A Ray runtime context exposing ``runtime_env``."""

    @property
    def runtime_env(self) -> object:
        """Return the runtime environment."""
        ...


class PromptLoader(Protocol):
    """TMax's resumable prompt iterator."""

    @property
    def _epoch(self) -> int:
        """Return the pass number; upstream tags each queued prompt with it."""
        ...

    def __next__(self) -> object:
        """Return the next prompt example."""
        ...

    def state_dict(self) -> Mapping[str, object]:
        """Capture the loader position."""
        ...

    def load_state_dict(self, state: Mapping[str, object], /) -> None:
        """Restore a captured loader position."""
        ...


class Batch(Protocol):
    """The upstream batch fields the serializer reads."""

    @property
    def queries(self) -> Sequence[Sequence[int]]:
        """Prompt tokens per sample."""
        ...

    @property
    def scores(self) -> Sequence[float] | None:
        """Verifier reward per sample."""
        ...

    @property
    def datasets(self) -> Sequence[object]:
        """Dataset name per sample."""
        ...

    @property
    def ground_truths(self) -> Sequence[object]:
        """Verifier target per sample."""
        ...


class GenerationResult(Protocol):
    """The upstream generation-result fields the serializer reads."""

    @property
    def responses(self) -> Sequence[Sequence[int]]:
        """Response tokens per sample."""
        ...

    @property
    def finish_reasons(self) -> Sequence[str]:
        """Stop reason per sample."""
        ...

    @property
    def masks(self) -> Sequence[Sequence[int]]:
        """Per-token flag: truthy where the policy sampled the token."""
        ...

    @property
    def logprobs(self) -> Sequence[Sequence[float]] | None:
        """Behavior logprob per response token."""
        ...

    @property
    def request_info(self) -> object:
        """Opaque per-request metadata."""
        ...


class FloatArray(Protocol):
    """An array of advantages, as upstream's numpy result exposes it."""

    def tolist(self) -> list[float]:
        """Return the values as Python floats."""
        ...


class GRPOArgs(Protocol):
    """``grpo_utils.GRPOExperimentConfig`` fields the driver sets."""

    seed: int
    hf_entity: str | None
    model_name_or_path: str
    verification_reward: float
    lm_head_fp32: bool
    enable_queue_dashboard: bool


class StreamingConfig(Protocol):
    """``data_loader.StreamingDataLoaderConfig`` fields the driver sets."""

    dataset_mixer_list: list[str]
    dataset_mixer_list_splits: list[str]
    dataset_mixer_eval_list: list[str]
    dataset_mixer_eval_list_splits: list[str]
    max_prompt_token_length: int
    response_length: int
    num_unique_prompts_rollout: int
    num_samples_per_prompt_rollout: int
    temperature: float
    active_sampling: bool
    filter_zero_std_samples: bool
    async_steps: int
    inflight_updates: bool
    mask_tool_use: bool
    system_prompt_override_file: str | None
    stop_strings: list[str] | None


class VLLMConfig(Protocol):
    """``data_loader.VLLMConfig`` fields the driver sets."""

    vllm_num_engines: int
    vllm_tensor_parallel_size: int


class EnvsConfig(Protocol):
    """``environments.tools.utils.EnvsConfig`` fields the driver reads."""

    pass_tools_to_chat_template: bool
    tool_call_names: object
    tool_parser_type: str


class DataLoaderModule(Protocol):
    """``open_instruct.data_loader``."""

    StreamingDataLoaderConfig: Callable[[], StreamingConfig]
    VLLMConfig: Callable[[], VLLMConfig]
    HFDataLoader: Callable[..., PromptLoader]
    single_example_collator: object
    add_prompt_to_generator: Callable[..., object]
    accumulate_inference_batches: Callable[
        ...,
        tuple[GenerationResult | None, Batch | None, object, object],
    ]
    compute_group_advantages: Callable[..., FloatArray]


class DatasetTransformationModule(Protocol):
    """``open_instruct.dataset_transformation``."""

    TokenizerConfig: Callable[..., object]


class EnvironmentsUtilsModule(Protocol):
    """``open_instruct.environments.tools.utils``."""

    EnvsConfig: Callable[..., EnvsConfig]


class GrpoFastModule(Protocol):
    """``open_instruct.grpo_fast``."""

    make_tokenizer: Callable[[object, object], object]
    initialize_tools_and_envs: Callable[
        ...,
        tuple[dict[str, object], object, list[str]],
    ]
    setup_datasets: Callable[..., tuple[object, object]]
    build_base_env_config: Callable[[object, object], object]


class GrpoUtilsModule(Protocol):
    """``open_instruct.grpo_utils``."""

    GRPOExperimentConfig: Callable[[], GRPOArgs]


class GroundTruthUtilsModule(Protocol):
    """``open_instruct.ground_truth_utils``."""

    RewardConfig: Callable[..., object]
    build_all_verifiers: Callable[[object, object], object]


class ModelUtilsModule(Protocol):
    """``open_instruct.model_utils``."""

    ModelConfig: Callable[..., object]


class RlUtilsModule(Protocol):
    """``open_instruct.rl_utils``."""

    def _get_request_info_for_sample(
        self,
        request_info: object,
        index: int,
        /,
    ) -> Mapping[str, object] | None:
        """Return one sample's request metadata, as upstream's writer reads it."""
        ...


class ActorManagerModule(Protocol):
    """``open_instruct.actor_manager``."""

    @property
    def ActorManager(self) -> object:  # noqa: N802 -- Upstream's class name.
        """Return the actor class Ray wraps."""
        ...


class ModelDimsReader(Protocol):
    """``open_instruct.utils.ModelDims``."""

    def from_hf_config(self, model_name_or_path: str, /) -> object:
        """Read model dimensions from a checkpoint config."""
        ...


class UtilsModule(Protocol):
    """``open_instruct.utils``."""

    @property
    def ModelDims(self) -> ModelDimsReader:  # noqa: N802 -- Upstream's class name.
        """Return the model-dimension reader."""
        ...

    def find_free_port(self) -> int:
        """Return a free TCP port on this node."""
        ...


class VllmUtilsModule(Protocol):
    """``open_instruct.vllm_utils``."""

    SamplingConfig: Callable[..., object]
    create_vllm_engines: Callable[..., list[Engine]]


class WeightTransferBase(Protocol):
    """``vllm.distributed.weight_transfer.base``."""

    WeightTransferInitRequest: Callable[..., object]


class NcclTransferEngine(Protocol):
    """``NCCLWeightTransferEngine``'s trainer-side entry points."""

    def trainer_init(self, init_info: Mapping[str, object], /) -> object:
        """Join the transfer group as the trainer rank."""
        ...

    def trainer_send_weights(
        self,
        *,
        iterator: Iterator[tuple[str, Tensor]],
        trainer_args: object,
    ) -> None:
        """Broadcast one policy's weights."""
        ...


class WeightTransferNccl(Protocol):
    """``vllm.distributed.weight_transfer.nccl_engine``."""

    @property
    def NCCLWeightTransferEngine(self) -> NcclTransferEngine:  # noqa: N802 -- Upstream's class name.
        """Return the trainer-side transfer engine."""
        ...

    @property
    def NCCLTrainerSendWeightsArgs(self) -> Callable[..., object]:  # noqa: N802 -- Upstream's class name.
        """Return the send-arguments constructor."""
        ...
