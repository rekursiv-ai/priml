"""Distributed helpers for rank-zero-only dataset builds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist


if TYPE_CHECKING:
    from collections.abc import Callable


def run_rank_zero_build(*, name: str, build: Callable[[], None]) -> None:
    """Run ``build`` on rank 0 and propagate failures to every rank.

    Rank-zero dataset builders must not raise before other ranks leave their
    collectives. This helper lets rank 0 capture the build error, publish a
    success flag via ``all_reduce``, and broadcast the error summary before
    anyone raises. Nonzero ranks then fail immediately with rank 0's error
    summary instead of waiting for a distributed barrier timeout.

    Args:
      name: Human-readable build name for nonzero-rank error messages.
      build: Rank-zero build action.

    Raises:
      Exception: The original rank-zero exception on rank 0.
      RuntimeError: A rank-zero failure summary on nonzero ranks.

    """
    if not dist.is_available() or not dist.is_initialized():
        build()
        return

    error: Exception | None = None
    message: str | None = None
    if dist.get_rank() == 0:
        try:
            build()
        except Exception as exc:  # noqa: BLE001 -- The build boundary records arbitrary worker failures for diagnosis.
            error = exc
            message = f"{type(exc).__name__}: {exc}"

    # NCCL all_reduce requires a CUDA tensor; only gloo reduces CPU tensors
    # (cf. seed.py, train_loop.py). A CPU tensor here would raise an NCCL error
    # on the GPU cluster -- defeating this helper's whole fail-fast purpose.
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    success = torch.tensor(
        [1 if message is None else 0],
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if int(success.item()) == 1:
        return

    # Rank 0 already holds the real error; raise it before the broadcast so a
    # broadcast failure (e.g. a process-group abort) can never swallow it. Only
    # the nonzero ranks depend on the broadcast to learn what rank 0 hit.
    if error is not None:
        try:
            dist.broadcast_object_list([message], src=0)
        finally:
            raise error
    messages: list[str | None] = [None]
    dist.broadcast_object_list(messages, src=0)
    detail = messages[0] or "unknown rank-zero error"
    raise RuntimeError(f"{name} failed on rank 0: {detail}")
