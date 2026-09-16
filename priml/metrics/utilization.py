"""Hardware utilization: the model's analytical cost against measured throughput.

The model config prices one token (:mod:`priml.model.cost`); the train
loop times each step and puts ``step_sec`` on the metric bus. This metric sums
tokens and seconds between ``reset`` calls and reports each kernel silo's
achieved fraction of its datasheet ceiling; the ``matmul`` silo is MFU.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import override

from configgle import Fig, LateBound
from torch import Tensor

from priml.model.cost import Cost, HasCost, KernelStats, cost


class Utilization(LateBound):
    """Tokens per second and per-silo utilization, averaged since the last reset.

    Belongs in ``TrainLoop.Config.metrics_train``: it reads the wall time the
    train step puts on the bus. Binds to the model config through the built
    loop, so no caller wires it.
    """

    class Config(Fig["Utilization"]):
        peak_flops_per_sec: float = 989.5e12
        """Dense bf16 tensor-core peak of ONE device; the H100 SXM default.

        The ``matmul`` ceiling, so ``mfu`` is the conventional figure."""

        peak_vector_flops_per_sec: float = 66.9e12
        """FP32 CUDA-core peak of ONE device; the H100 SXM default.

        The ceiling for every silo that is not a matmul: elementwise,
        reduction, selection, and sort all run on the vector units."""

        tokens_key: str = "input_ids"
        """Batch tensor whose element count is the token count."""

    def __init__(self, config: Config) -> None:
        vector = config.peak_vector_flops_per_sec
        self.peak = KernelStats(
            matmul=config.peak_flops_per_sec,
            elementwise=vector,
            reduction=vector,
            selection=vector,
            sort=vector,
        )
        self.tokens_key = config.tokens_key
        self._model_config: HasCost | None = None
        self._cost_by_seq_len: dict[int, Cost] = {}
        self.reset()

    @override
    def bind(self, root: object) -> None:
        """Adopt the built loop's model config, which is what prices a token.

        Args:
          root: The outermost built object; a ``TrainLoop`` exposes the model
            config at ``step.config.model``.

        Raises:
          TypeError: ``root`` carries no model config there, or the config
            cannot price itself; a silent zero would report every run as idle.

        """
        # ``runtime_checkable`` tests attribute NAMES one level deep, so the
        # path is walked by hand: a root whose ``step`` lacks a config would
        # otherwise pass the protocol and fail on the read.
        model_config: object = root
        for name in ("step", "config", "model"):
            model_config = getattr(model_config, name, None)
            if model_config is None:
                raise TypeError(
                    f"{type(root).__qualname__} exposes no step.config.model; a "
                    "utilization metric needs the model config to price a token.",
                )
        if not isinstance(model_config, HasCost):
            raise TypeError(
                f"{type(model_config).__qualname__} has no cost(); a utilization "
                "metric needs a model config implementing HasCost.",
            )
        self._model_config = model_config

    def reset(self) -> None:
        """Zero the token and second sums."""
        self.tokens = 0
        self.seconds = 0.0
        self.flops = KernelStats()

    def update(self, logits: Tensor, **batch: object) -> None:
        """Accumulate one train step.

        Args:
          logits: Unread; the model's output is not what utilization measures.
          **batch: The step's batch plus ``step_sec``, the wall seconds the step
            took; ``tokens_key`` selects the token tensor.

        Raises:
          RuntimeError: ``bind`` has not run.
          TypeError: ``step_sec`` is absent (the caller is not a timed step),
            or the ``tokens_key`` field is not a tensor.

        """
        del logits
        if self._model_config is None:
            raise RuntimeError("bind() must precede update(); build through make().")
        step_sec = batch.get("step_sec")
        if not isinstance(step_sec, float):
            raise TypeError(
                "update() needs step_sec on the bus; only a timed train step "
                "supplies it, so this metric belongs in metrics_train.",
            )
        tokens = batch[self.tokens_key]
        if not isinstance(tokens, Tensor):
            raise TypeError(
                f"batch[{self.tokens_key!r}] is {type(tokens).__qualname__}, "
                "not a Tensor; tokens_key must select the token tensor.",
            )
        seq_len = tokens.shape[-1]
        per_token = self._cost_by_seq_len.get(seq_len)
        if per_token is None:
            per_token = self._cost_by_seq_len[seq_len] = cost(
                self._model_config,
                seq_len=seq_len,
            )
        self.tokens += tokens.numel()
        self.seconds += step_sec
        self.flops = self.flops + per_token.training.flops * tokens.numel()

    def compute(self) -> dict[str, float]:
        """Report throughput and utilization over the updates since ``reset``.

        Returns:
          metrics: ``tokens_per_sec``, ``mfu`` (the matmul silo's fraction of
            peak), and ``utilization_<silo>`` for the other four; empty when
            nothing was timed, since a zero would read as an idle run.

        """
        if not self.seconds:
            return {}
        # Summed FLOPs over summed seconds: :func:`utilization` for a window,
        # with each step weighted by its tokens rather than counted once.
        achieved = self.flops / self.seconds / self.peak
        return {
            "tokens_per_sec": self.tokens / self.seconds,
            "mfu": achieved.matmul,
            "utilization_elementwise": achieved.elementwise,
            "utilization_reduction": achieved.reduction,
            "utilization_selection": achieved.selection,
            "utilization_sort": achieved.sort,
        }

    def state_dict(self) -> Mapping[str, object]:
        """Return nothing: a throughput window does not survive a restart.

        Returns:
          state: Empty.

        """
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Accept and ignore a saved state; see :meth:`state_dict`.

        Args:
          state_dict: Ignored.

        """
        del state_dict
