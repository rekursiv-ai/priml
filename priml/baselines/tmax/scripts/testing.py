"""Test support shared by the TMax driver's tests: Ray, a loader and a session.

The runner and weight-sync tests both drive ``UpstreamApi`` and
``CollectionSession`` without importing Ray, vLLM or open-instruct. They build
those objects from the stand-ins here rather than from each other's modules.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import NoReturn, cast, overload

from priml.baselines.tmax.scripts import custom_types
from priml.baselines.tmax.scripts.runner import CollectionSession, UpstreamApi


def unused(*_args: object, **_kwargs: object) -> NoReturn:
    """Fail the test: it reached an upstream call it claims not to make."""
    raise AssertionError("This test does not reach this upstream call.")


@dataclass(kw_only=True, slots=True)
class FakeRay:
    """A Ray driver whose cluster state each test sets directly.

    ``get_runtime_context`` returns ``context`` when given, else a context
    carrying ``runtime_env``. ``remote`` wraps its target with ``tasks``, and
    ``get`` maps each reference through ``resolve``.
    """

    initialized: bool = True
    runtime_env: Mapping[str, object] = field(default_factory=dict[str, object])
    live_nodes: list[Mapping[str, object]] = field(
        default_factory=list[Mapping[str, object]],
    )
    tasks: Callable[[object], custom_types.RemoteFunction] = unused
    resolve: Callable[[object], object] | None = None
    context: object | None = None
    init_calls: list[Mapping[str, object]] = field(
        default_factory=list[Mapping[str, object]],
    )
    events: list[str] = field(default_factory=list[str])

    @property
    def _private(self) -> custom_types.RayPrivate:
        return unused()

    def is_initialized(self) -> bool:
        """Return whether the driver is up."""
        return self.initialized

    def init(self, *, runtime_env: Mapping[str, object]) -> None:
        """Record the runtime env and come up."""
        self.init_calls.append(runtime_env)
        self.initialized = True

    def shutdown(self) -> None:
        """Record the shutdown and go down."""
        self.events.append("shutdown")
        self.initialized = False

    def nodes(self) -> list[Mapping[str, object]]:
        """Return the cluster's nodes."""
        return self.live_nodes

    def get_runtime_context(self) -> object:
        """Return the given context, else one carrying ``runtime_env``."""
        if self.context is not None:
            return self.context
        return _RuntimeContext(runtime_env=self.runtime_env)

    def remote(self, target: object, /) -> custom_types.RemoteFunction:
        """Wrap ``target`` as a task."""
        return self.tasks(target)

    @overload
    def get(self, refs: list[object], /) -> list[object]: ...
    @overload
    def get(self, refs: object, /) -> object: ...
    def get(self, refs: object, /) -> object:
        """Resolve one reference, or each of a list; unchanged without ``resolve``.

        Args:
          refs: One reference or a list of them.

        Returns:
          values: The resolved value, or a list of them.

        """
        resolve = self.resolve
        if resolve is None:
            return refs
        if isinstance(refs, list):
            return [resolve(ref) for ref in cast("list[object]", refs)]
        return resolve(refs)


class FakeLoader:
    """A prompt loader over a fixed list of examples."""

    def __init__(self, examples: Iterable[object] = (), *, epoch: int = 0) -> None:
        self._examples: Iterator[object] = iter(examples)
        self._epoch = epoch

    def __next__(self) -> object:
        """Return the next example."""
        return next(self._examples)

    def state_dict(self) -> Mapping[str, object]:
        """Return no state: the examples are fixed."""
        return {}

    def load_state_dict(self, state: Mapping[str, object], /) -> None:
        """Ignore ``state``: the examples are fixed."""
        del state


def upstream_api(
    modules: object,
    *,
    ray: custom_types.Ray | None = None,
    rl_utils: custom_types.RlUtilsModule | None = None,
    actor_manager: custom_types.ActorManagerModule | None = None,
    weight_transfer: tuple[
        custom_types.WeightTransferBase,
        custom_types.WeightTransferNccl,
    ]
    | None = None,
) -> UpstreamApi:
    """Bind ``modules`` to every open-instruct slot, with the given overrides.

    Args:
      modules: One stand-in for every open-instruct module the test reaches.
      ray: The Ray driver; a default ``FakeRay`` when ``None``.
      rl_utils: The ``rl_utils`` module; ``modules`` when ``None``.
      actor_manager: The actor-manager module, or ``None``.
      weight_transfer: vLLM's weight-transfer modules, or ``None``.

    Returns:
      api: The upstream surface.

    """
    return UpstreamApi(
        ray=ray or FakeRay(),
        ray_queue=cast("custom_types.RayQueue", modules),
        data_loader=cast("custom_types.DataLoaderModule", modules),
        dataset_transformation=cast(
            "custom_types.DatasetTransformationModule",
            modules,
        ),
        environments_utils=cast("custom_types.EnvironmentsUtilsModule", modules),
        grpo_fast=cast("custom_types.GrpoFastModule", modules),
        grpo_utils=cast("custom_types.GrpoUtilsModule", modules),
        ground_truth_utils=cast("custom_types.GroundTruthUtilsModule", modules),
        model_utils=cast("custom_types.ModelUtilsModule", modules),
        rl_utils=rl_utils or cast("custom_types.RlUtilsModule", modules),
        utils=cast("custom_types.UtilsModule", modules),
        vllm_utils=cast("custom_types.VllmUtilsModule", modules),
        actor_manager=actor_manager,
        weight_transfer=weight_transfer,
    )


def session(
    api: UpstreamApi,
    *,
    engines: Sequence[custom_types.Engine] = (),
    loader: custom_types.PromptLoader | None = None,
    actor_manager: custom_types.ActorManagerHandle | None = None,
) -> CollectionSession:
    """Return a session over ``api`` with nothing built yet.

    Args:
      api: The upstream surface.
      engines: The vLLM engines; copied into the session.
      loader: The prompt loader; an empty ``FakeLoader`` when ``None``.
      actor_manager: The actor-manager handle; an inert object when ``None``.

    Returns:
      session: The session.

    """
    return CollectionSession(
        api=api,
        tokenizer=None,
        dataset=None,
        iter_dataloader=loader or FakeLoader(),
        inference_queue=None,
        prompt_queue=None,
        generation_config=None,
        base_env_config=None,
        model_dims=None,
        engines=list(engines),
        pools={},
        started_ray=False,
        actor_manager=actor_manager
        or cast("custom_types.ActorManagerHandle", object()),
    )


@dataclass(kw_only=True, slots=True)
class _RuntimeContext:
    runtime_env: Mapping[str, object]
