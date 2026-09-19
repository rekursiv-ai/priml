"""Weighted sum loss combiner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    traffic,
)
from priml.loss.custom_types import LossOutput


if TYPE_CHECKING:
    from collections.abc import Callable

    LossFn = Callable[..., LossOutput | Tensor]


class WeightedSum(nn.Module):
    """Weighted sum of multiple loss functions."""

    class Config(Fig["WeightedSum"]):
        fns: Sequence[Makeable[LossFn]] = field(
            default_factory=list["Makeable[LossFn]"],
        )
        """Loss functions to combine; ``nn.Module`` or plain callable."""

        weights: Sequence[float] = field(default_factory=list[float])
        """Weight for each loss function."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Sum exact child costs and this combiner's whole-tensor work.

            Each child receives the caller's concrete geometry. The stack,
            weighting, reduction, and adjoint broadcast run for every output
            element in the complete batch.

            Args:
              seq_len: Output elements per sample.
              batch_size: Samples per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete batch.

            """
            children = sum(
                (
                    cost(
                        fn,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    for fn in self.fns
                ),
                Cost(),
            )
            count = len(self.fns)
            rows = seq_len * batch_size
            dt = dtype
            return children + (
                traffic(
                    "primal",
                    "elementwise",
                    elements=4 * count,
                    flops=2 * count,
                    dtype=dt,
                )
                + traffic("primal", "reduction", elements=count + 1, dtype=dt)
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=2 * count,
                    flops=2 * count,
                    dtype=dt,
                )
                + traffic("adjoint", "reduction", elements=count + 1, dtype=dt)
            ).tile(rows)

    def __init__(self, config: Config):
        super().__init__()
        # Losses may be plain callables (SimpleLoss, AdversarialLoss) or
        # nn.Modules. Keep the callables for ``forward`` and separately
        # register only the nn.Module ones so their parameters are tracked.
        self.fns: list[LossFn] = [f.make() for f in config.fns]
        self._modules_list = nn.ModuleList(
            [fn for fn in self.fns if isinstance(fn, nn.Module)],
        )
        self.weights = list(config.weights)

    @override
    def forward(self, *args: object, **kwargs: object) -> LossOutput:
        """Compute weighted sum of losses.

        Returns:
          loss: Dict with 'loss' key (weighted sum) and individual losses.

        """
        # Collect individual losses (each returns LossOutput dict)
        individual_results: list[LossOutput | Tensor] = [
            fn(*args, **kwargs) for fn in self.fns
        ]

        # Extract loss tensors and merge extra keys.
        result: dict[str, Tensor] = {}
        loss_tensors: list[Tensor] = []

        for i, individual in enumerate(individual_results):
            # Handle both LossOutput dict and plain Tensor returns.
            if isinstance(individual, dict):
                # Cast to dict[str, Tensor] for proper type handling.
                individual_dict = cast(dict[str, Tensor], individual)
                loss_tensors.append(individual_dict["loss"])
                # Add unscaled loss with index.
                result[f"loss_{i}"] = individual_dict["loss"]
                # Merge other keys.
                for key, value in individual_dict.items():
                    if key != "loss":
                        result[f"{key}_{i}"] = value
            else:
                # Plain tensor return (legacy support)
                assert isinstance(individual, Tensor)
                loss_tensors.append(individual)
                result[f"loss_{i}"] = individual

        # Compute weighted sum.
        if len(self.weights) != len(loss_tensors):
            msg = f"Weights ({len(self.weights)}) and losses ({len(loss_tensors)}) count mismatch"
            raise ValueError(msg)

        weighted: list[Tensor] = [
            w * loss for w, loss in zip(self.weights, loss_tensors, strict=True)
        ]
        total: Tensor = torch.stack(weighted).sum(dim=0)

        result["loss"] = total
        return cast(LossOutput, result)
