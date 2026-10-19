"""Gradient compressors: what a gradient becomes after a simulated round trip.

A compressor maps one parameter's gradient to the gradient the optimizer
steps with: compressed, communicated and decompressed. The task simulates data
parallel training on one process, so only the reconstruction matters. Each
function here is stateless; :class:`ErrorFeedback` wraps one and carries what
it dropped into the next step, per parameter.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor

import torch


if TYPE_CHECKING:
    from collections.abc import Mapping


class Compressor(Protocol):
    """Maps one parameter's gradient to its reconstruction after compression."""

    def __call__(self, grad: Tensor, /, *, name: str) -> Tensor:
        """Return the reconstructed gradient, shaped as ``grad``; ``name`` keys state."""
        raise NotImplementedError


def identity(grad: Tensor, /, *, name: str) -> Tensor:
    """Return the gradient unchanged: no compression."""
    del name
    return grad


def top_k(grad: Tensor, /, *, name: str, ratio: float) -> Tensor:
    """Keep the ``ratio`` fraction of entries largest in magnitude, at least one.

    Args:
      grad: The gradient, any shape.
      name: Unused; the function keeps no state.
      ratio: Fraction of entries kept, rounded down.

    Returns:
      grad: The kept entries in place and zeros elsewhere.

    References:
      https://arxiv.org/abs/1704.05021
        Aji and Heafield 2017, "Sparse Communication for Distributed Gradient
        Descent."

    """
    del name
    flat = grad.flatten()
    k = max(1, int(flat.numel() * ratio))
    _, indices = torch.topk(flat.abs(), k, sorted=False)
    return torch.zeros_like(flat).scatter_(0, indices, flat[indices]).view(grad.shape)


def scaled_sign(grad: Tensor, /, *, name: str) -> Tensor:
    """Replace each entry by its sign, zero counting as positive, times the mean magnitude.

    Args:
      grad: The gradient, any shape.
      name: Unused; the function keeps no state.

    Returns:
      grad: ``+-mean(|grad|)`` at every entry.

    References:
      https://arxiv.org/abs/1802.04434
        Bernstein et al. 2018, "signSGD: Compressed Optimisation for
        Non-Convex Problems."

    """
    del name
    flat = grad.flatten()
    return ((flat >= 0).float() * 2 - 1).mul(flat.abs().mean()).view(grad.shape)


def qsgd(grad: Tensor, /, *, name: str, levels: int, clip_norm: float) -> Tensor:
    """Clip to ``clip_norm``, then round each magnitude stochastically to ``levels`` steps.

    Each entry becomes ``norm / levels * sign * l`` with ``l`` the level below
    or above ``levels * |g| / norm``, the one above with probability equal to
    the remainder, so the result is unbiased. One uniform per entry is drawn
    from the global generator, unless the gradient is zero.

    Args:
      grad: The gradient, any shape.
      name: Unused; the function keeps no state.
      levels: Quantization steps between zero and the norm.
      clip_norm: Largest norm kept; a larger gradient is scaled down to it.

    Returns:
      grad: The quantized gradient.

    References:
      https://arxiv.org/abs/1610.02132
        Alistarh et al. 2017, "QSGD: Communication-Efficient SGD via Gradient
        Quantization and Encoding."

    """
    del name
    flat = grad.flatten()
    norm = flat.norm()
    # A zero gradient returns before the draw, as the reference does; drawing
    # anyway would move every later uniform of the run.
    if norm == 0:
        return (norm / levels * flat.to(torch.int16).float()).view(grad.shape)
    scale = clip_norm / (norm + 1e-6)
    if scale < 1.0:
        flat = flat * scale
        norm = flat.norm()
    level = levels / norm * flat.abs()
    below = level.floor()
    up = (torch.rand_like(flat) < (level - below)).float()
    quantized = ((below + up) * flat.sign()).to(torch.int16)
    return (norm / levels * quantized.float()).view(grad.shape)


class ErrorFeedback:
    """Adds what the last step's compression dropped before compressing again.

    References:
      https://arxiv.org/abs/1809.07599
        Stich, Cordonnier and Jaggi 2018, "Sparsified SGD with Memory."

    """

    class Config(Fig["ErrorFeedback"]):
        """The compressor wrapped and the weight of the carried residual."""

        compressor: Makeable[Compressor] = field(
            default_factory=lambda: PartialConfig(identity),
        )
        """Compresses the gradient plus the carried residual."""

        beta: float = 1.0
        """Multiplies the residual before it is added."""

    def __init__(self, config: Config) -> None:
        self.compressor: Compressor = config.compressor.make()
        self.beta = config.beta
        self.residuals: dict[str, Tensor] = {}
        """Per parameter name, the input of the last compression minus its output."""

    def __call__(self, grad: Tensor, /, *, name: str) -> Tensor:
        """Compress ``grad`` plus the residual of ``name``, and keep the new residual.

        Args:
          grad: The gradient of parameter ``name``.
          name: The parameter, which keys its residual.

        Returns:
          grad: The compressed sum.

        """
        if name in self.residuals:
            grad = grad + self.beta * self.residuals[name]
        compressed = self.compressor(grad, name=name)
        self.residuals[name] = grad - compressed
        return compressed

    class StateDict(TypedDict):
        """The residual of every parameter compressed so far."""

        residuals: dict[str, Tensor]

    def state_dict(self) -> StateDict:
        """Return the residuals, so a resumed run carries them on.

        Returns:
          state: ``residuals`` keyed by parameter name.

        """
        return {"residuals": dict(self.residuals)}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the residuals.

        Args:
          state_dict: State as returned by :meth:`state_dict`.

        """
        self.residuals = dict(cast(ErrorFeedback.StateDict, state_dict)["residuals"])
