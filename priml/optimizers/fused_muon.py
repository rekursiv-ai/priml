"""Muon over fp32 master weights, with a global-norm clip and fused CUDA kernels.

:class:`~priml.optimizers.muon.Muon` updates each parameter in its own
dtype. :class:`FusedMuon` keeps an fp32 master behind every parameter, steps
the master, and writes the parameter as the master rounded to its dtype. Its
other departures each change the bits, which is why it exists as its own
optimizer instead of a flag:

1. **A global-norm clip** over every gradient (:meth:`FusedMuon.norm`, as
   ``clip_grad_norm_`` reduces it), before momentum. The clip is
   ``min(max_grad_norm / (norm + 1e-6), 1)``.
2. **Nesterov momentum in fp32**, with the update rounded to the gradient's
   dtype (:func:`nesterov`).
3. **Per-step Newton-Schulz coefficients.** Each matrix is normalized by
   ``max(norm, eps)``, then takes one ``(a, b, c)`` triple per step as GEMMs in
   its dtype (:func:`newton_schulz`), then the scale
   ``sqrt(max(1, rows / cols))``. A parameter of more than two dimensions is
   the matrix of its rows, ``[shape[0], -1]``, as priml's
   :class:`~priml.optimizers.muon.Muon` takes it: a conv weight's ``[out,
   in, height, width]`` is ``[out, in * height * width]``. A vector skips all
   three and takes its Nesterov update as it is.
4. **Fused kernels.** On CUDA with bf16 parameters, :meth:`FusedMuon.step`
   launches Triton kernels: one for the clip and Nesterov step, one per matrix
   for its normalization, and one for the aspect scale, the master update and
   the cast. The stage functions below are the torch reference those kernels
   are held to, and what runs on any other device or dtype. A subclass may
   change the norms' reduction order, the kernels' device functions and their
   launch options (:attr:`FusedMuon.helpers`, :attr:`FusedMuon.launch_options`)
   to reproduce another implementation's bits.

The step is then ``master -= lr * update`` in fp32, with the product rounded.

References:
  https://github.com/PufferAI/PufferLib
    Suarez. PufferLib (MIT license), ``muon_step`` in ``src/algo.cu``, pin
    ``6ffa5b10``.
  https://kellerjordan.github.io/posts/muon/
    Jordan et al. Muon: An optimizer for hidden layers in neural networks.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import TYPE_CHECKING, ClassVar, cast, overload, override

import math
import struct

from configgle import Fig
from torch import Tensor
from torch.optim import Optimizer

import torch

from priml.kernel import jit_kernel
from priml.lib.custom_json import FloatCodec


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from torch.optim.optimizer import StateDict
    from triton import language

    import triton
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")
    language = lazy_import("triton.language")


__all__ = [
    "FusedMuon",
    "apply_update",
    "aspect_scale",
    "clip_coefficient",
    "nesterov",
    "newton_schulz",
    "normalize",
]


class FusedMuon(Optimizer):
    """Muon over fp32 masters, applied to parameters of any dtype.

    State per parameter: ``master_weight`` and ``momentum_buffer``, both fp32,
    created on the first step from the parameter itself unless loaded first
    through :meth:`load_state_dict` or written through :attr:`master_weights`
    and :attr:`momentum_buffers`.

    Args:
      params: Parameters or groups, in the order the global norm reduces them.
      lr: Step size on the fp32 masters.
      momentum: Nesterov momentum coefficient.
      max_grad_norm: Global L2 clip over every gradient, before momentum.
      ns_coefficients: One ``(a, b, c)`` per Newton-Schulz step.
      eps: Floor on a matrix update's norm before its Newton-Schulz steps.

    Raises:
      ValueError: A scalar hyperparameter is negative or not finite.

    """

    class Config(Fig["Callable[..., FusedMuon]"]):
        """Hyperparameters; see :class:`FusedMuon` for what each one does.

        The defaults are PufferLib's ``config/default.ini``; a recipe sets its
        own. ``make()`` yields a constructor, not an optimizer: a config tree
        has no parameters to hand one. Call the result with them, in the order
        the global norm should reduce them::

            optimizer = FusedMuon.Config().make()(parameters)
        """

        lr: float = 0.015
        """Step size on the fp32 masters; a schedule may rewrite it per group."""

        momentum: float = 0.95
        """Nesterov momentum coefficient."""

        max_grad_norm: float = 1.5
        """Global L2 clip over every gradient, before momentum: one value for
        every parameter group, since the norm spans them all."""

        ns_coefficients: tuple[tuple[float, float, float], ...] = (
            (4.0848, -6.8946, 2.9270),
            (3.9505, -6.3029, 2.6377),
            (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046),
            (2.8366, -3.0525, 1.2012),
        )
        """One ``(a, b, c)`` per Newton-Schulz step: ``X <- aX + (bG + cG²)X``."""

        eps: float = 1e-7
        """Floor on a matrix update's norm before its Newton-Schulz steps."""

        @override
        def make(self) -> Callable[..., FusedMuon]:
            """Return a constructor awaiting the parameters to optimize."""
            final = self.finalized()
            return partial(
                FusedMuon,
                lr=final.lr,
                momentum=final.momentum,
                max_grad_norm=final.max_grad_norm,
                ns_coefficients=final.ns_coefficients,
                eps=final.eps,
            )

    helpers: ClassVar[dict[str, Callable[..., object]]] = {}
    """Device functions the fused kernels call, by name, in place of this
    module's own; empty for Triton's standard arithmetic."""

    launch_options: ClassVar[dict[str, bool]] = {}
    """Keywords every fused launch adds to Triton's defaults."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, object]],
        *,
        lr: float,
        momentum: float,
        max_grad_norm: float,
        ns_coefficients: tuple[tuple[float, float, float], ...],
        eps: float,
    ) -> None:
        for name, value in (
            ("lr", lr),
            ("momentum", momentum),
            ("max_grad_norm", max_grad_norm),
            ("eps", eps),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"FusedMuon {name} must be finite and nonnegative.")
        super().__init__(params, {"lr": lr, "momentum": momentum, "eps": eps})
        # The clip and the polynomial are the run's code, not its state, so they
        # stay off the groups a checkpoint serializes and are rebuilt from config
        # on resume; the clip's norm spans every group, so a group cannot hold it.
        self.max_grad_norm = max_grad_norm
        self.ns_coefficients = ns_coefficients

    @property
    def master_weights(self) -> list[Tensor]:
        """The fp32 masters, one per parameter in group order."""
        return [
            self._state(parameter)["master_weight"] for parameter in self._parameters()
        ]

    @property
    def momentum_buffers(self) -> list[Tensor]:
        """The fp32 momentum, one per parameter in group order."""
        return [
            self._state(parameter)["momentum_buffer"]
            for parameter in self._parameters()
        ]

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], Tensor | float]) -> Tensor | float: ...

    @torch.no_grad()
    @override
    def step(
        self,
        closure: Callable[[], Tensor | float] | None = None,
    ) -> Tensor | float | None:
        """Clip, run Nesterov and Newton-Schulz, step the masters, cast them back.

        Args:
          closure: Optional loss computation, run with gradients enabled.

        Returns:
          loss: The closure's result, or None without one.

        Raises:
          ValueError: A parameter has no gradient; the global norm needs them all.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        parameters = self._parameters()
        gradients = [parameter.grad for parameter in parameters]
        present = [gradient for gradient in gradients if gradient is not None]
        if len(present) != len(gradients):
            raise ValueError("FusedMuon updates every parameter; one has no gradient.")
        coefficient = clip_coefficient(self.norm(present), self.max_grad_norm)
        # The kernels index flat, so every tensor they touch is contiguous.
        fused = all(
            gradient.is_cuda
            and gradient.dtype == parameter.dtype == torch.bfloat16
            and gradient.is_contiguous()
            and parameter.is_contiguous()
            for gradient, parameter in zip(present, parameters, strict=True)
        )
        index = 0
        for group in self.param_groups:
            lr, momentum, eps = _group_scalars(group)
            if fused and not isinstance(lr, Tensor):
                lr = torch.full((), lr, dtype=torch.float32, device=coefficient.device)
            for parameter in cast("list[Tensor]", group["params"]):
                state = self._state(parameter)
                update = (self._nesterov_cuda if fused else nesterov)(
                    present[index],
                    state["momentum_buffer"],
                    coefficient=coefficient,
                    momentum=momentum,
                )
                index += 1
                scale = 1.0
                if update.ndim >= 2:
                    matrix = update.reshape(update.shape[0], -1)
                    matrix = newton_schulz(
                        (self._normalize_cuda if fused else normalize)(
                            matrix,
                            norm=self.norm([matrix]),
                            eps=eps,
                        ),
                        self.ns_coefficients,
                    )
                    scale = aspect_scale(matrix.shape)
                    update = matrix.reshape(update.shape)
                (self._apply_cuda if fused else apply_update)(
                    parameter,
                    state["master_weight"],
                    update,
                    lr=lr,
                    scale=scale,
                )
        return loss

    @override
    def load_state_dict(self, state_dict: StateDict) -> None:
        """Restore the fp32 masters and momentum as saved.

        Torch's load rounds floating state to the parameter's dtype, which
        would put bf16 masters under bf16 parameters; the saved tensors are
        copied back over that afterwards. The state is new tensors either way,
        so a CUDA graph captured over the old ones must be captured again.

        Args:
          state_dict: What :meth:`state_dict` returned.

        """
        super().load_state_dict(state_dict)
        saved = cast("dict[int, dict[str, Tensor]]", state_dict["state"])
        groups = cast("list[dict[str, list[int]]]", state_dict["param_groups"])
        for saved_group, group in zip(groups, self.param_groups, strict=True):
            members = cast("list[Tensor]", group["params"])
            for saved_id, parameter in zip(saved_group["params"], members, strict=True):
                for key, value in saved.get(saved_id, {}).items():
                    self.state[parameter][key] = value.to(
                        device=parameter.device,
                        dtype=torch.float32,
                        copy=True,
                    )

    def norm(self, tensors: Sequence[Tensor]) -> Tensor:
        """Return the L2 norm of the tensors taken as one vector.

        The step calls this for the global clip and for each matrix. Several
        tensors reduce as ``clip_grad_norm_`` reduces a model's gradients, with
        ``get_total_norm``'s foreach kernels, in their own dtype; one tensor
        accumulates in fp32.

        Args:
          tensors: Any shapes and float dtypes, on one device.

        Returns:
          norm: A 0-dim fp32 tensor on their device.

        """
        # One multi-tensor launch: a norm per tensor then a stack took 53 us for
        # exp000's seven gradients on an H200, against 14 us here.
        if len(tensors) == 1:
            return torch.linalg.vector_norm(tensors[0], dtype=torch.float32)
        return torch.nn.utils.get_total_norm(tensors).float()

    def _parameters(self) -> list[Tensor]:
        return [
            parameter
            for group in self.param_groups
            for parameter in cast("list[Tensor]", group["params"])
        ]

    def _state(self, parameter: Tensor) -> dict[str, Tensor]:
        state = cast("dict[str, Tensor]", self.state[parameter])
        if not state:
            state["master_weight"] = parameter.detach().float().clone()
            state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
        return state

    def _nesterov_cuda(
        self,
        gradient: Tensor,
        buffer: Tensor,
        *,
        coefficient: Tensor,
        momentum: float,
    ) -> Tensor:
        """Do what :func:`nesterov` does in one launch (bf16 gradient, CUDA)."""
        update = torch.empty_like(gradient)
        count = gradient.numel()
        block = 1024
        _kernels(**self.helpers).nesterov[(-(-count // block),)](
            gradient,
            buffer,
            update,
            coefficient,
            momentum,
            count,
            block=block,
            num_warps=4,
            **self.launch_options,
        )
        return update

    def _normalize_cuda(self, matrix: Tensor, *, norm: Tensor, eps: float) -> Tensor:
        """Do what :func:`normalize` does in one launch (bf16 matrix, CUDA)."""
        normalized = torch.empty_like(matrix)
        count = matrix.numel()
        block = 1024
        _kernels(**self.helpers).normalize[(-(-count // block),)](
            matrix,
            normalized,
            norm,
            eps,
            count,
            block=block,
            num_warps=4,
            **self.launch_options,
        )
        return normalized

    def _apply_cuda(
        self,
        parameter: Tensor,
        master: Tensor,
        update: Tensor,
        *,
        lr: Tensor | float,
        scale: float,
    ) -> None:
        """Do what :func:`apply_update` does in one launch (bf16, CUDA, tensor lr)."""
        if not isinstance(lr, Tensor):
            raise TypeError("the fused update reads its rate from a device tensor")
        count = update.numel()
        block = 1024
        _kernels(**self.helpers).apply[(-(-count // block),)](
            update,
            master,
            parameter,
            lr,
            scale,
            count,
            scaled=scale != 1.0,
            block=block,
            num_warps=4,
            **self.launch_options,
        )


def clip_coefficient(norm: Tensor, max_grad_norm: float) -> Tensor:
    """Return the global clip, ``min(max_grad_norm / (norm + 1e-6), 1)``.

    Args:
      norm: The gradients' global L2 norm, a 0-dim fp32 tensor.
      max_grad_norm: The clip's limit, rounded to fp32.

    Returns:
      coefficient: A 0-dim fp32 tensor on the norm's device.

    """
    return torch.fmin(
        torch.full_like(norm, max_grad_norm) / (norm + 1e-6),
        torch.ones_like(norm),
    )


def nesterov(
    gradient: Tensor,
    buffer: Tensor,
    *,
    coefficient: Tensor,
    momentum: float,
) -> Tensor:
    """Advance the fp32 momentum in place and return the Nesterov update.

    ``m = fma(g, clip, mu * m)`` with the momentum product rounded, then the
    update ``fma(g, clip, mu * m)`` likewise, rounded to the gradient's dtype.

    Args:
      gradient: One parameter's gradient.
      buffer: Its fp32 momentum, updated in place.
      coefficient: The global clip, a 0-dim fp32 tensor.
      momentum: The coefficient ``mu``, an fp32 value.

    Returns:
      update: The clipped Nesterov gradient in the gradient's dtype.

    """
    value = gradient.float()
    buffer.mul_(momentum).addcmul_(value, coefficient)
    return torch.addcmul(buffer * momentum, value, coefficient).to(gradient.dtype)


def normalize(matrix: Tensor, *, norm: Tensor, eps: float) -> Tensor:
    """Scale a matrix by the reciprocal of its norm floored at ``eps``.

    One fp32 reciprocal of ``max(norm, eps)``, then a product rounded to the
    matrix's dtype.

    Args:
      matrix: The Newton-Schulz input.
      norm: Its L2 norm, a 0-dim fp32 tensor (:meth:`FusedMuon.norm`).
      eps: The floor.

    Returns:
      normalized: The same shape and dtype.

    """
    norm = torch.fmax(norm, torch.full_like(norm, eps))
    return (matrix.float() * (torch.ones_like(norm) / norm)).to(matrix.dtype)


def newton_schulz(
    matrix: Tensor,
    coefficients: Sequence[tuple[float, float, float]],
) -> Tensor:
    """Run one Newton-Schulz step per ``(a, b, c)``, each as three GEMMs.

    Each step forms the gram ``G`` (``XᵀX`` when the matrix is tall, ``XXᵀ``
    when wide), the polynomial ``cG² + bG`` and then ``aX + XP`` or
    ``aX + PX``: every product a GEMM in the matrix's dtype with fp32
    accumulation, in the ``alpha``/``beta`` forms cuBLAS takes.

    Args:
      matrix: ``[rows, cols]``, already normalized.
      coefficients: ``(a, b, c)`` per step.

    Returns:
      result: The orthogonalized matrix, before the aspect scale.

    """
    tall = matrix.shape[0] > matrix.shape[1]
    for linear, cubic, quintic in coefficients:
        gram = matrix.mT @ matrix if tall else matrix @ matrix.mT
        polynomial = torch.addmm(gram, gram, gram, beta=cubic, alpha=quintic)
        matrix = (
            torch.addmm(matrix, matrix, polynomial, beta=linear)
            if tall
            else torch.addmm(matrix, polynomial, matrix, beta=linear)
        )
    return matrix


def aspect_scale(shape: Sequence[int]) -> float:
    """Return ``sqrt(max(1, rows / cols))``; the update rounds it to fp32.

    Args:
      shape: ``(rows, cols)``.

    Returns:
      scale: The value in double.

    """
    rows, cols = shape
    return max(1.0, rows / cols) ** 0.5


def apply_update(
    parameter: Tensor,
    master: Tensor,
    update: Tensor,
    *,
    lr: Tensor | float,
    scale: float,
) -> None:
    """Scale the update, step the fp32 master, and round it into the parameter.

    The scaled update is rounded to the update's dtype (skipped at scale 1,
    where it is the identity), then ``master - lr * update`` with the product
    rounded, then the parameter becomes the master in its own dtype.

    Args:
      parameter: Written with the new master in its dtype.
      master: The fp32 master, updated in place.
      update: The Newton-Schulz result, or the Nesterov update of a vector.
      lr: The step size: a 0-dim fp32 tensor or an fp32 value.
      scale: :func:`aspect_scale` of a matrix; 1 for a vector.

    """
    if scale != 1.0:
        update = (update.float() * scale).to(update.dtype)
    master.sub_(update.float() * lr)
    parameter.copy_(master)


def _group_scalars(group: dict[str, object]) -> tuple[Tensor | float, float, float]:
    """Return a group's rate, its fp32 momentum and its eps."""
    # A 0-dim fp32 device tensor, torch's ``capturable`` convention, is read each
    # time a captured step replays; a Python float would be baked into the graph
    # at capture, so a tensor rate passes through unrounded and unconverted.
    rate = group["lr"]
    return (
        rate if isinstance(rate, Tensor) else _fp32(FloatCodec.coerce(rate, None)),
        _fp32(FloatCodec.coerce(group["momentum"], None)),
        FloatCodec.coerce(group["eps"], None),
    )


def _fp32(value: float) -> float:
    """Round to the nearest fp32; on fp32 operands this is fp32 arithmetic."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Kernels:
    nesterov: triton.JITFunction[..., object]
    normalize: triton.JITFunction[..., object]
    apply: triton.JITFunction[..., object]


@lru_cache(maxsize=2)
def _kernels(**helpers: Callable[..., object]) -> _Kernels:
    """Jit the kernels once per helper set, on first use; importing needs no Triton."""
    reciprocal = helpers.get("_reciprocal_triton", _reciprocal_triton)
    return _Kernels(
        nesterov=jit_kernel(_nesterov_triton),
        normalize=jit_kernel(
            _normalize_triton,
            _reciprocal_triton=jit_kernel(reciprocal),
        ),
        apply=jit_kernel(_apply_triton),
    )


# Each value is rounded where :func:`nesterov` rounds it; the products inside the
# ``fma`` calls are separately rounded multiplies under any launch options.
def _nesterov_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    gradient_ptr: language.tensor,
    buffer_ptr: language.tensor,
    update_ptr: language.tensor,
    coefficient_ptr: language.tensor,
    momentum: float,
    count: int,
    block: language.constexpr,
) -> None:
    """Momentum, then update, each ``fma(g, clip, mu * m)``."""
    index = language.program_id(0) * block + language.arange(0, block)
    live = index < count
    clip = language.load(coefficient_ptr)
    value = language.load(gradient_ptr + index, mask=live, other=0.0).to(
        language.float32,
    )
    buffer = language.load(buffer_ptr + index, mask=live, other=0.0)
    buffer = language.fma(value, clip, buffer * momentum)
    language.store(buffer_ptr + index, buffer, mask=live)
    update = language.fma(value, clip, buffer * momentum)
    language.store(update_ptr + index, update.to(language.bfloat16), mask=live)


def _normalize_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    matrix_ptr: language.tensor,
    output_ptr: language.tensor,
    norm_ptr: language.tensor,
    eps: float,
    count: int,
    block: language.constexpr,
) -> None:
    """``x * (1 / max(norm, eps))``, rounded to bf16."""
    index = language.program_id(0) * block + language.arange(0, block)
    live = index < count
    inverse = _reciprocal_triton(language.maximum(language.load(norm_ptr), eps))
    value = language.load(matrix_ptr + index, mask=live, other=0.0).to(
        language.float32,
    )
    language.store(
        output_ptr + index,
        (value * inverse).to(language.bfloat16),
        mask=live,
    )


def _apply_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    update_ptr: language.tensor,
    master_ptr: language.tensor,
    parameter_ptr: language.tensor,
    lr_ptr: language.tensor,
    scale: float,
    count: int,
    scaled: language.constexpr,
    block: language.constexpr,
) -> None:
    """Scale the update, step the master and round it into the bf16 parameter."""
    index = language.program_id(0) * block + language.arange(0, block)
    live = index < count
    update = language.load(update_ptr + index, mask=live, other=0.0).to(
        language.float32,
    )
    if scaled:
        update = (update * scale).to(language.bfloat16).to(language.float32)
    master = language.load(master_ptr + index, mask=live, other=0.0)
    master -= update * language.load(lr_ptr)
    language.store(master_ptr + index, master, mask=live)
    language.store(parameter_ptr + index, master.to(language.bfloat16), mask=live)


# :attr:`FusedMuon.helpers` replaces this by name: a subclass holding another
# implementation's bits divides with the IEEE ``div_rn`` instead.
def _reciprocal_triton(x: language.tensor) -> language.tensor:
    """Return ``1 / x`` with Triton's default fp32 division."""
    return 1.0 / x
