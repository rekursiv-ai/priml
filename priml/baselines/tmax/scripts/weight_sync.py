"""Push the learner's trained weights into TMax's vLLM engines."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor

    from priml.baselines.tmax.scripts.custom_types import WeightTransferNccl
    from priml.baselines.tmax.scripts.runner import CollectionSession


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
        self._nccl: WeightTransferNccl | None = None
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
        self._nccl = nccl

    def __call__(
        self,
        weights: Iterable[tuple[str, Tensor]],
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
        nccl = self._nccl
        if nccl is None:
            raise RuntimeError("WeightSync.initialize must run before a sync.")
        api = self.session.api
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
            self._nccl = None
            if not engines_awake:
                with suppress(Exception):
                    api.ray.get([engine.wake_up.remote() for engine in engines])
            raise
        else:
            api.ray.get(manager.set_should_stop.remote(False))
