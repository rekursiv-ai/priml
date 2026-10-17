"""Initializers of the MLS-Bench ``dl-weight-initialization`` baselines.

The task's editable ``initialize_weights`` walks the network once. Its per-tensor
part fills :attr:`~priml.baselines.mlsbench.pytorch_vision.model.ResNet.Config.init_conv`
and ``init_head``; a rule that needs the network's structure, as Fixup's does,
fills ``ResNet.Config.init``, which runs after them.

Each draws exactly what the task's function draws, in the same order: the
generator's position after initialization is part of what a run reproduces.
"""

from __future__ import annotations

from typing import cast

from torch import Tensor, nn

from priml.baselines.mlsbench.pytorch_vision.model import BasicBlock


def small_normal(w: Tensor) -> None:
    """Draw from a normal with mean 0 and standard deviation 0.01: the pristine rule.

    Args:
      w: Weight to overwrite in place.

    """
    nn.init.normal_(w, 0, 0.01)


def orthogonal_relu(w: Tensor) -> None:
    """Draw a (semi-)orthogonal matrix scaled by ReLU's gain, ``sqrt(2)``.

    Args:
      w: Weight to overwrite in place; dimensions after the first are flattened.

    References:
      https://arxiv.org/abs/1312.6120
        Saxe et al. 2014, "Exact solutions to the nonlinear dynamics of
        learning in deep linear neural networks."

    """
    nn.init.orthogonal_(w, gain=nn.init.calculate_gain("relu"))


def fixup(model: nn.Module) -> None:
    """Shrink every residual branch: scale its last convolution, zero its last norm.

    Each block's second convolution is multiplied by ``L ** -0.5`` for ``L``
    blocks, and the scale of the normalization after it is set to zero, so
    every residual branch starts at its bias and a block passes on its
    activated shortcut. Draws nothing.

    Args:
      model: A built network; its :class:`BasicBlock` s are changed in place.

    References:
      https://arxiv.org/abs/1901.09321
        Zhang et al. 2019, "Fixup Initialization: Residual Learning Without
        Normalization."
      https://arxiv.org/abs/1706.02677
        Goyal et al. 2017, "Accurate, Large Minibatch SGD: Training ImageNet in
        1 Hour" (zero-initialized last normalization of each block).

    """
    blocks = [m for m in model.modules() if isinstance(m, BasicBlock)]
    if not blocks:
        raise ValueError(f"fixup needs BasicBlocks; {type(model).__name__} has none.")
    scale = len(blocks) ** (-0.5)
    for block in blocks:
        block.conv2.weight.data.mul_(scale)
        nn.init.constant_(cast(Tensor, block.norm2.weight), 0)
