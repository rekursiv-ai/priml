"""Activations of the MLS-Bench ``dl-activation-function`` baselines.

Each fills
:attr:`~priml.baselines.mlsbench.dl.vision.model.ResNet.Config.activation`,
the slot the task's editable ``CustomActivation`` becomes. ReLU and SiLU need
nothing here: :func:`priml.model.swiglu.relu` and :func:`priml.model.swiglu.silu`
call the same torch functions as the reference's ``F.relu`` and ``F.silu``.

Every expression keeps the reference's operations: a run that must reproduce
the reference bit for bit follows its arithmetic, so an algebraically equal
rewrite, a fused kernel included, is not a refactor.
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional

import torch

from priml.cost import map_cost, set_cost


# ``x / 2 * (1 + erf(x / sqrt(2)))``: scale, erf, add, halve, multiply. Back,
# ``g * (cdf + x * pdf)``: the cdf's four, the pdf's square, scale, exp and
# normalizer, then the product, the sum and ``g``.
@set_cost(map_cost(primal=5, adjoint=11))
def gelu(x: Tensor) -> Tensor:
    """Apply ``x * Phi(x)``, with ``Phi`` the exact standard normal CDF.

    Args:
      x: Pre-activation values, any shape.

    Returns:
      activated: ``x * Phi(x)``, elementwise; not the tanh approximation.

    References:
      https://arxiv.org/abs/1606.08415
        Hendrycks and Gimpel 2016, "Gaussian Error Linear Units (GELUs)."

    """
    return functional.gelu(x)


# Three kernels: softplus (exp, log1p and its threshold compare), tanh and the
# product. Back, the product's two gradients, tanh's ``1 - t**2`` and
# softplus's sigmoid, each scaled by the incoming gradient.
@set_cost(
    map_cost(
        primal=5,
        adjoint=10,
        inputs=4,
        outputs=3,
        adjoint_inputs=7,
        adjoint_outputs=4,
    ),
)
def mish(x: Tensor) -> Tensor:
    """Apply ``x * tanh(softplus(x))``.

    Args:
      x: Pre-activation values, any shape.

    Returns:
      activated: ``x * tanh(log(1 + exp(x)))``, elementwise, with softplus
        returning ``x`` itself above torch's threshold of 20.

    References:
      https://arxiv.org/abs/1908.08681
        Misra 2020, "Mish: A Self Regularized Non-Monotonic Activation
        Function."

    """
    return x * torch.tanh(functional.softplus(x))
