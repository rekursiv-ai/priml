"""Minimal gated recurrent units for batch-first sequences, in two numerical variants.

Each layer carries a hidden vector ``h`` through time:

    z = sigma(gate)              candidate = hidden + 0.5 if hidden >= 0 else sigma(hidden)
    h = lerp(h, candidate, z)    out = s * h + (1 - s) * x,  s = sigma(highway)

where ``hidden``, ``gate`` and ``highway`` are the three slices of one
projection of the layer's input ``x``, and a reset zeroes the carry before its
step. The two variants compute it differently, so their bits differ and
neither stands in for the other:

- :class:`MinGRU` is a stack of layers in fp32, differentiated by autograd.
  Each layer's carry is solved for every step at once by a parallel scan.
- :class:`MinGRUBlock` is one layer between low-precision boundaries. Its
  :class:`Scan` walks time sequentially in fp32 and rounds the carry to the
  state dtype -- the initial carry's -- before each next step; the projection,
  the input and the output stay in their own dtype. A bf16 carry rounds away
  every update smaller than half its ulp, which freezes the slowest units; an
  fp32 carry keeps them. Autograd differentiates the block, but through the
  scan it runs the scan's own backward (:meth:`Scan.backward`): that
  recomputes the gates from the saved carries, keeps its own fp32 carry
  gradient across steps and cuts it at a reset. Autograd through the scan's
  ops would sum and round in another order, which is enough to move the bits.

:class:`MinGRU` is the one to reach for. :class:`MinGRUBlock` exists to
reproduce PufferLib's recurrent policy.

:class:`TorchScan` is the sequential scan op by op in torch, on any device.
:class:`TritonScan` runs the same recurrence as Triton kernels on bf16 or fp32
CUDA tensors, each store rounding to its tensor's dtype, and the reference for
other dtypes: one program per block of ``(batch, width)`` elements, sequential
in time. The kernels use Triton's
``sigmoid``, ``lerp`` as ``a + w * (b - a)`` and its default contraction of a
multiply into the add that follows, so they agree with the reference to a few
ulp; a subclass replaces those device functions and the launch options
(:attr:`TritonScan.helpers`, :attr:`TritonScan.launch_options`) to reproduce
another implementation's rounding. :class:`TritonScan` refuses CPU tensors
rather than falling back, so one config gives one set of bits on every host; a
CPU run selects :class:`TorchScan`.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``src/algo.cu``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, elementwise_cost
from priml.kernel import jit_kernel, require_power_of_two
from priml.model.linear import Linear


if TYPE_CHECKING:
    from collections.abc import Callable

    from triton import language

    import triton
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")
    language = lazy_import("triton.language")


class MinGRU(nn.Module):
    """Apply a gated highway recurrence to ``[batch,time,channels]`` inputs.

    Args:
      input_channels: Width of the first recurrent layer's inputs.
      channels: Width of every recurrent state.
      layers: Number of stacked recurrent layers.

    """

    def __init__(self, input_channels: int, channels: int, layers: int = 1) -> None:
        super().__init__()
        if input_channels <= 0 or channels <= 0 or layers <= 0:
            raise ValueError("MinGRU widths and layer count must be positive.")
        self.input_channels = input_channels
        self.channels = channels
        self.input_projection = (
            nn.Identity()
            if input_channels == channels
            else nn.Linear(input_channels, channels, bias=False)
        )
        self.layers = nn.ModuleList(
            nn.Linear(channels, 3 * channels, bias=False) for _ in range(layers)
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return zero carry with shape ``[layers,batch,channels]``.

        Args:
          batch_size: Number of independent recurrent sequences.
          device: Optional destination device.

        Returns:
          state: Zero FP32 carry with shape ``[layers,batch,channels]``.

        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        parameter = next(self.parameters())
        return torch.zeros(
            len(self.layers),
            batch_size,
            self.channels,
            device=parameter.device if device is None else device,
            dtype=torch.float32,
        )

    @override
    def forward(
        self,
        inputs: Tensor,
        *,
        state: Tensor | None = None,
        reset: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return recurrent outputs and final carry for a batch-first sequence.

        Every layer's carry is a linear recurrence ``h_t = a_t*h_(t-1)+b_t``
        (``a_t`` zeroed by a reset, matching the highway gate ordering below),
        solved for every ``t`` at once by ``_scan_affine`` instead of a Python
        loop over time -- an inclusive parallel scan, not a sequential
        unroll. This changes only how the sum is associated, not its value: a
        one-step call (``time == 1``) skips the scan's doubling loop entirely
        and reduces to the same single affine update the loop body used to
        perform, at matching cost.

        Args:
          inputs: Sequence values with shape ``[batch,time,input_channels]``.
          state: Optional initial carry with shape ``[layers,batch,channels]``.
          reset: Boolean resets applied before each observation.

        Returns:
          outputs: Recurrent sequence with shape ``[batch,time,channels]``.
          state: Final carry with shape ``[layers,batch,channels]``.

        """
        if inputs.ndim != 3:
            raise ValueError("MinGRU inputs must have shape [batch,time,channels].")
        batch, time, width = inputs.shape
        if time <= 0:
            raise ValueError("MinGRU inputs must have a positive time dimension.")
        if width != self.input_channels:
            raise ValueError("MinGRU input width does not match input_channels.")
        if reset is None:
            reset = torch.zeros(batch, time, dtype=torch.bool, device=inputs.device)
        if reset.shape != (batch, time):
            raise ValueError("MinGRU reset must have shape [batch,time].")
        if state is None:
            state = self.initial_state(batch, device=inputs.device)
        if state.shape != (len(self.layers), batch, self.channels):
            raise ValueError("MinGRU state must have shape [layers,batch,channels].")

        # Every projection and recurrent step below must run at true FP32
        # regardless of an ambient autocast the caller (ArcPolicy) may have
        # enabled around the surrounding encoder for throughput: autocast
        # would otherwise downcast these matmuls to its compute dtype despite
        # the `.float()` calls, silently reintroducing the instability this
        # module's FP32 carry exists to avoid.
        device_type = "cuda" if inputs.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            value = self.input_projection(inputs.float())
            reset_mask = reset[..., None]
            carries: list[Tensor] = []
            for layer_index, layer in enumerate(self.layers):
                combined = layer(value).float()
                hidden, gate, highway = combined.chunk(3, dim=-1)
                candidate = torch.where(
                    hidden >= 0,
                    hidden + 0.5,
                    torch.sigmoid(hidden),
                )
                update = torch.sigmoid(gate)
                decay = torch.where(reset_mask, torch.zeros_like(update), 1 - update)
                carry = _scan_affine(
                    decay,
                    innovation=update * candidate,
                    initial=state[layer_index].float(),
                )
                strength = torch.sigmoid(highway)
                value = (1 - strength) * value + strength * carry
                carries.append(carry[:, -1])
            return value, torch.stack(carries)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScanForward:
    """What one layer's forward produces, and what its backward reads back.

    Attributes:
      outputs: The highway outputs, ``[batch, time, width]``, in the input
        dtype.
      final: The carry after the last step, ``[batch, width]``, in the state
        dtype.
      states: The carry each step read AFTER its reset, ``[batch, time,
        width]``, in the state dtype.

    """

    outputs: Tensor
    final: Tensor
    states: Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class ScanBackward:
    """The gradients one layer's backward produces.

    Attributes:
      grad_combined: Gradient of the projected gates, ``[batch, time, 3 *
        width]``, in the projection dtype.
      grad_inputs: Gradient reaching the layer input through the highway,
        ``[batch, time, width]``; the gradient through the projection is the
        caller's GEMM.
      grad_initial: Gradient reaching the initial carry, ``[batch, width]``.

    """

    grad_combined: Tensor
    grad_inputs: Tensor
    grad_initial: Tensor


class Scan(Protocol):
    """One layer's recurrence over a ``[batch, time]`` window, and its adjoint."""

    def __call__(
        self,
        combined: Tensor,
        inputs: Tensor,
        initial: Tensor,
        terminals: Tensor,
    ) -> ScanForward:
        """Run the recurrence forward.

        Args:
          combined: Projected gates, ``[batch, time, 3 * width]``: the
            ``hidden``, ``gate`` and ``highway`` slices in that order.
          inputs: The layer input, ``[batch, time, width]``.
          initial: The carry before the first step, ``[batch, width]``.
          terminals: Nonzero where the carry is reset BEFORE that step,
            ``[batch, time]``.

        Returns:
          result: Outputs, final carry and the per-step carries.

        """
        ...

    def step(
        self,
        combined: Tensor,
        inputs: Tensor,
        state: Tensor,
        *,
        carry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run one time step of the recurrence, without its reset or per-step stores.

        What a rollout runs once per environment step; any reset is the
        caller's, applied to ``state`` first.

        Args:
          combined: Projected gates, ``[batch, 3 * width]``.
          inputs: The layer input, ``[batch, width]``.
          state: The carry, ``[batch, width]``.
          carry: Where the next carry is written; a new tensor if None. It
            may be ``state`` itself, which is then advanced in place.

        Returns:
          outputs: ``[batch, width]`` in the input dtype.
          state: The next carry, ``[batch, width]``; ``carry`` if given.

        """
        ...

    def backward(
        self,
        combined: Tensor,
        inputs: Tensor,
        states: Tensor,
        terminals: Tensor,
        grad_outputs: Tensor,
    ) -> ScanBackward:
        """Run the recurrence backward from the gradient of ``outputs``.

        Args:
          combined: As in the forward.
          inputs: As in the forward.
          states: The forward's ``states``.
          terminals: As in the forward.
          grad_outputs: Gradient of the forward's ``outputs``, same shape and
            dtype.

        Returns:
          result: Gradients of the gates, the input and the initial carry.

        """
        ...


class TorchScan:
    """The reference scan: the recurrence's arithmetic, one torch op at a time."""

    class Config(Fig["TorchScan"]):
        """Build the reference scan; it has nothing to configure."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(
        self,
        combined: Tensor,
        inputs: Tensor,
        initial: Tensor,
        terminals: Tensor,
    ) -> ScanForward:
        """Run the recurrence forward, one step at a time.

        Args:
          combined: Projected gates, ``[batch, time, 3 * width]``.
          inputs: The layer input, ``[batch, time, width]``.
          initial: The carry before the first step, ``[batch, width]``.
          terminals: Nonzero where the carry resets before that step.

        Returns:
          result: Outputs, final carry and the per-step carries.

        """
        hidden, gate, highway = combined.float().chunk(3, dim=-1)
        resets = terminals != 0
        state = initial.float()
        states: list[Tensor] = []
        outputs: list[Tensor] = []
        for time in range(inputs.shape[1]):
            state = torch.where(resets[:, time, None], 0.0, state)
            states.append(state.to(initial.dtype))
            carry = torch.lerp(
                state,
                _candidate(hidden[:, time]),
                torch.sigmoid(gate[:, time]),
            )
            strength = torch.sigmoid(highway[:, time])
            outputs.append(
                (strength * carry + (1 - strength) * inputs[:, time].float()).to(
                    inputs.dtype,
                ),
            )
            state = carry.to(initial.dtype).float()
        return ScanForward(
            outputs=torch.stack(outputs, dim=1),
            final=state.to(initial.dtype),
            states=torch.stack(states, dim=1),
        )

    def step(
        self,
        combined: Tensor,
        inputs: Tensor,
        state: Tensor,
        *,
        carry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run one time step of the recurrence.

        Args:
          combined: Projected gates, ``[batch, 3 * width]``.
          inputs: The layer input, ``[batch, width]``.
          state: The carry, ``[batch, width]``.
          carry: Where the next carry is written, as :meth:`Scan.step` takes it.

        Returns:
          outputs: ``[batch, width]`` in the input dtype.
          state: The next carry; ``carry`` if given.

        """
        hidden, gate, highway = combined.float().chunk(3, dim=-1)
        updated = torch.lerp(state.float(), _candidate(hidden), torch.sigmoid(gate))
        strength = torch.sigmoid(highway)
        outputs = strength * updated + (1 - strength) * inputs.float()
        if carry is None:
            return outputs.to(inputs.dtype), updated.to(state.dtype)
        return outputs.to(inputs.dtype), carry.copy_(updated)

    def backward(
        self,
        combined: Tensor,
        inputs: Tensor,
        states: Tensor,
        terminals: Tensor,
        grad_outputs: Tensor,
    ) -> ScanBackward:
        """Run the recurrence backward, one step at a time from the last.

        Args:
          combined: As in the forward.
          inputs: As in the forward.
          states: The forward's ``states``.
          terminals: As in the forward.
          grad_outputs: Gradient of the forward's ``outputs``.

        Returns:
          result: Gradients of the gates, the input and the initial carry.

        """
        hidden, gate, highway = combined.float().chunk(3, dim=-1)
        resets = terminals != 0
        grad_combined = torch.empty_like(combined)
        grad_inputs = torch.empty_like(inputs)
        dh = torch.zeros_like(states[:, 0], dtype=torch.float32)
        for time in range(inputs.shape[1] - 1, -1, -1):
            previous = states[:, time].float()
            z = torch.sigmoid(gate[:, time])
            candidate = _candidate(hidden[:, time])
            carry = torch.lerp(previous, candidate, z)
            strength = torch.sigmoid(highway[:, time])
            current = grad_outputs[:, time].float()
            grad_highway = (
                current * (carry - inputs[:, time].float()) * strength * (1 - strength)
            )
            grad_inputs[:, time] = (current * (1 - strength)).to(inputs.dtype)
            total = dh + current * strength
            grad_candidate = total * z
            grad_gate = total * (candidate - previous) * z * (1 - z)
            grad_hidden = torch.where(
                hidden[:, time] >= 0,
                grad_candidate,
                grad_candidate * candidate * (1 - candidate),
            )
            grad_combined[:, time] = torch.cat(
                (grad_hidden, grad_gate, grad_highway),
                dim=-1,
            ).to(combined.dtype)
            dh = torch.where(resets[:, time, None], 0.0, total * (1 - z))
        return ScanBackward(
            grad_combined=grad_combined,
            grad_inputs=grad_inputs,
            grad_initial=dh.to(states.dtype),
        )


class TritonScan:
    """The scan as Triton kernels on a CUDA device; :class:`TorchScan` for other dtypes.

    The kernels take bf16 or fp32 gates and inputs with a bf16 or fp32 carry.
    """

    class Config(Fig["TritonScan"]):
        """Launch geometry; the recurrence is :class:`TorchScan`'s."""

        block: int = 256
        """``(batch, width)`` elements per program."""

        num_warps: int = 8
        """Warps per program: one element per thread at the default block."""

    helpers: ClassVar[dict[str, Callable[..., object]]] = {}
    """Device functions the kernels call, by name, in place of this module's
    own (``_sigmoid_triton``, ``_lerp_triton``, ``_highway_triton``); empty for
    Triton's standard arithmetic."""

    launch_options: ClassVar[dict[str, bool]] = {}
    """Keywords every launch adds to Triton's defaults."""

    def __init__(self, config: Config) -> None:
        """Keep the launch geometry and the reference for other tensors.

        Args:
          config: The geometry.

        """
        require_power_of_two(block=config.block, num_warps=config.num_warps)
        self.block = config.block
        self.num_warps = config.num_warps
        self.reference = TorchScan(TorchScan.Config())
        """The scan for CUDA tensors the kernels do not take: any neither bf16
        nor fp32."""

    def __call__(
        self,
        combined: Tensor,
        inputs: Tensor,
        initial: Tensor,
        terminals: Tensor,
    ) -> ScanForward:
        """Run the recurrence forward, one launch.

        Args:
          combined: Projected gates, ``[batch, time, 3 * width]``.
          inputs: The layer input, ``[batch, time, width]``.
          initial: The carry before the first step, ``[batch, width]``.
          terminals: Nonzero where the carry resets before that step.

        Returns:
          result: Outputs, final carry and the per-step carries.

        """
        _require_cuda(combined)
        if not _runs_triton(combined, inputs, initial):
            return self.reference(combined, inputs, initial, terminals)
        _check_layout(combined, inputs, initial, terminals)
        batch, time, width = inputs.shape
        outputs = torch.empty_like(inputs)
        final = torch.empty_like(initial)
        states = torch.empty_like(inputs, dtype=initial.dtype)
        count = batch * width
        _kernels(**self.helpers).forward[(triton.cdiv(count, self.block),)](
            combined,
            inputs,
            initial,
            _terminals_view(terminals),
            outputs,
            final,
            states,
            count,
            time,
            width,
            block=self.block,
            num_warps=self.num_warps,
            **self.launch_options,
        )
        return ScanForward(outputs=outputs, final=final, states=states)

    def step(
        self,
        combined: Tensor,
        inputs: Tensor,
        state: Tensor,
        *,
        carry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run one time step of the recurrence, one launch.

        Each lane reads its carry element before writing it, so ``carry`` may
        be ``state``.

        Args:
          combined: Projected gates, ``[batch, 3 * width]``.
          inputs: The layer input, ``[batch, width]``.
          state: The carry, ``[batch, width]``.
          carry: Where the next carry is written, as :meth:`Scan.step` takes it.

        Returns:
          outputs: ``[batch, width]`` in the input dtype.
          state: The next carry; ``carry`` if given.

        """
        written = () if carry is None else (carry,)
        _require_cuda(combined)
        if not _runs_triton(combined, inputs, state, *written):
            return self.reference.step(combined, inputs, state, carry=carry)
        _check_layout(combined, inputs, state, *written)
        outputs = torch.empty_like(inputs)
        next_state = torch.empty_like(state) if carry is None else carry
        count = inputs.numel()
        _kernels(**self.helpers).step[(triton.cdiv(count, self.block),)](
            combined,
            inputs,
            state,
            outputs,
            next_state,
            count,
            inputs.shape[-1],
            block=self.block,
            num_warps=self.num_warps,
            **self.launch_options,
        )
        return outputs, next_state

    def backward(
        self,
        combined: Tensor,
        inputs: Tensor,
        states: Tensor,
        terminals: Tensor,
        grad_outputs: Tensor,
    ) -> ScanBackward:
        """Run the recurrence backward, one launch.

        Args:
          combined: As in the forward.
          inputs: As in the forward.
          states: The forward's ``states``.
          terminals: As in the forward.
          grad_outputs: Gradient of the forward's ``outputs``.

        Returns:
          result: Gradients of the gates, the input and the initial carry.

        """
        _require_cuda(combined)
        if not _runs_triton(combined, inputs, states, grad_outputs):
            return self.reference.backward(
                combined,
                inputs,
                states,
                terminals,
                grad_outputs,
            )
        _check_layout(combined, inputs, states, terminals, grad_outputs)
        batch, time, width = inputs.shape
        grad_combined = torch.empty_like(combined)
        grad_inputs = torch.empty_like(inputs)
        grad_initial = torch.empty_like(states[:, 0])
        count = batch * width
        _kernels(**self.helpers).backward[(triton.cdiv(count, self.block),)](
            combined,
            inputs,
            states,
            _terminals_view(terminals),
            grad_outputs,
            grad_combined,
            grad_inputs,
            grad_initial,
            count,
            time,
            width,
            block=self.block,
            num_warps=self.num_warps,
            **self.launch_options,
        )
        return ScanBackward(
            grad_combined=grad_combined,
            grad_inputs=grad_inputs,
            grad_initial=grad_initial,
        )


class MinGRUBlock(nn.Module):
    """One gate projection and one recurrence over its output."""

    class Config(Fig["MinGRUBlock"]):
        """Configure one block; a parent may fill the width and dtype."""

        SCAN_PRIMAL_OPS: ClassVar[int] = 23
        """Recurrence operations per carry channel per step: the reset select 1,
        the candidate 7 (compare, add, sigmoid, select), the gate and highway
        sigmoids 4 each, the lerp 3 and the highway mix 4. A count of the rule,
        not a knob."""

        SCAN_ADJOINT_OPS: ClassVar[int] = 46
        """The backward's operations per carry channel per step: about twice
        :attr:`SCAN_PRIMAL_OPS`."""

        channels_hidden: int = -1
        """Width of the input, the carry and the output."""

        proj_gates: Linear.Config = field(default_factory=Linear.Config)
        """The ``width -> 3 * width`` projection into hidden, gate, highway."""

        scan: Makeable[Scan] = field(default_factory=TorchScan.Config)
        """The recurrence: the torch reference, or the Triton kernels."""

        dtype: torch.dtype | None = None
        """Parameter dtype, pushed into ``proj_gates`` where it has none."""

        @override
        def finalize(self) -> Self:
            self.proj_gates.channels_in = self.channels_hidden
            self.proj_gates.channels_out = 3 * self.channels_hidden
            if self.proj_gates.dtype is None:
                self.proj_gates.dtype = self.dtype
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the gate projection and the recurrence over every step.

            The recurrence is elementwise per carry channel: the reset select,
            the candidate, two sigmoids, the lerp and the highway mix,
            :attr:`SCAN_PRIMAL_OPS` operations, and :attr:`SCAN_ADJOINT_OPS` in
            its backward.
            It reads the three gate rows and the input and writes the output.

            Args:
              seq_len: Steps per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to the projection.

            Returns:
              cost: Integer FLOPs and logical bytes for the complete invocation.

            """
            rows = seq_len * batch_size
            elements = rows * self.channels_hidden
            scan = elementwise_cost(
                primal=self.SCAN_PRIMAL_OPS * elements,
                adjoint=self.SCAN_ADJOINT_OPS * elements,
                channels=self.channels_hidden,
                rows=rows,
                dtype=dtype,
                inputs=4,
                outputs=1,
                adjoint_inputs=5,
                adjoint_outputs=4,
            )
            projection = self.proj_gates.cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            return projection + scan

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.proj_gates = config.proj_gates.make()
        self.scan = config.scan.make()

    @override
    def forward(
        self,
        inputs: Tensor,
        initial: Tensor,
        terminals: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Project the gates and run the recurrence.

        Args:
          inputs: ``[batch, time, width]``.
          initial: The carry before the first step, ``[batch, width]``.
          terminals: Nonzero where the carry resets before that step,
            ``[batch, time]``.

        Returns:
          outputs: ``[batch, time, width]`` in the input dtype; autograd
            differentiates them through :meth:`Scan.backward`.
          final: The carry after the last step, ``[batch, width]``; not
            differentiated.

        """
        return _Recurrence.apply(
            self.scan,
            self.proj_gates(inputs),
            inputs,
            initial,
            terminals,
        )

    def step(
        self,
        inputs: Tensor,
        state: Tensor,
        *,
        carry: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run one time step: the projection, then :meth:`Scan.step`.

        Args:
          inputs: ``[batch, width]``.
          state: The carry, ``[batch, width]``, already reset where a
            sequence starts.
          carry: Where the next carry goes, as :meth:`Scan.step` takes it.

        Returns:
          outputs: ``[batch, width]``.
          state: The next carry.

        """
        return self.scan.step(self.proj_gates(inputs), inputs, state, carry=carry)


__all__ = [
    "MinGRU",
    "MinGRUBlock",
    "Scan",
    "ScanBackward",
    "ScanForward",
    "TorchScan",
    "TritonScan",
]


class _RecurrenceContext(Protocol):
    """What :class:`_Recurrence` keeps between its forward and its backward."""

    scan: Scan
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...

    def mark_non_differentiable(self, *tensors: Tensor) -> None: ...

    def set_materialize_grads(self, value: bool) -> None: ...


def _recurrence_backward(
    ctx: _RecurrenceContext,
    /,
    *grad_outputs: Tensor | None,
) -> tuple[None, Tensor, Tensor, Tensor, None]:
    """Run the scan's own backward from the outputs' gradient."""
    grad, _ = grad_outputs
    assert isinstance(grad, Tensor)
    combined, inputs, states, terminals = ctx.saved_tensors
    # Autograd may hand over a strided gradient (an expanded ``sum``'s); the
    # kernels address contiguous memory.
    result = ctx.scan.backward(combined, inputs, states, terminals, grad.contiguous())
    return None, result.grad_combined, result.grad_inputs, result.grad_initial, None


# The block input reaches the loss through the projection and through the highway;
# autograd adds the two gradients in one bf16 add (fp32 opmath, rounded once), which
# is the sum PufferLib's backward forms. The final carry's gradient is never seeded
# into the walk, so the final carry is marked non-differentiable rather than
# silently dropped.
class _Recurrence(torch.autograd.Function):
    """One layer's scan as an autograd node whose backward is the scan's own."""

    @classmethod
    @override
    def forward(
        cls,
        ctx: _RecurrenceContext,
        /,
        scan: Scan,
        combined: Tensor,
        inputs: Tensor,
        initial: Tensor,
        terminals: Tensor,
    ) -> tuple[Tensor, Tensor]:
        result = scan(combined, inputs, initial, terminals)
        ctx.scan = scan
        ctx.save_for_backward(combined, inputs, result.states, terminals)
        ctx.mark_non_differentiable(result.final)
        # An unseeded final carry then arrives as None, not as a zeroed buffer.
        ctx.set_materialize_grads(False)
        return result.outputs, result.final

    # Torch declares ``backward`` a staticmethod, and an override must stay one.
    backward = staticmethod(_recurrence_backward)


# An inclusive Hillis-Steele parallel scan over the associative affine-map composition
# ``combine((A,B),(a,b)) = (a*A, a*B+b)``: after ``ceil(log2(time))`` doubling rounds,
# ``(decay,innovation)`` at position ``t`` holds the composed map from the sequence
# start through ``t``. A ``decay`` of exactly zero (a reset) composes to zero for every
# later position, so it discards `initial` and every prior step with no special casing
# -- the same reason a log-space cumulative-sum scan is not used here, since ``log(0)``
# would need segment-aware handling to recover this property.
def _scan_affine(decay: Tensor, *, innovation: Tensor, initial: Tensor) -> Tensor:
    """Solve ``h_t = decay_t*h_(t-1)+innovation_t`` for every ``t`` at once."""
    time = decay.shape[1]
    coefficient, offset = decay, innovation
    stride = 1
    while stride < time:
        shifted_coefficient = functional.pad(
            coefficient,
            (0, 0, stride, 0),
            value=1.0,
        )[:, :time]
        shifted_offset = functional.pad(offset, (0, 0, stride, 0), value=0.0)[:, :time]
        offset = coefficient * shifted_offset + offset
        coefficient = coefficient * shifted_coefficient
        stride *= 2
    return coefficient * initial[:, None, :] + offset


@dataclass(frozen=True, slots=True, kw_only=True)
class _ScanKernels:
    forward: triton.JITFunction[..., object]
    backward: triton.JITFunction[..., object]
    step: triton.JITFunction[..., object]


@lru_cache(maxsize=2)
def _kernels(**helpers: Callable[..., object]) -> _ScanKernels:
    """Jit the kernels once per helper set, on first use; importing needs no Triton."""
    device = {
        "_sigmoid_triton": _sigmoid_triton,
        "_lerp_triton": _lerp_triton,
        "_highway_triton": _highway_triton,
    } | helpers
    sigmoid = jit_kernel(device["_sigmoid_triton"])
    bound = {
        "_sigmoid_triton": sigmoid,
        "_candidate_triton": jit_kernel(_candidate_triton, _sigmoid_triton=sigmoid),
        "_lerp_triton": jit_kernel(device["_lerp_triton"]),
        "_highway_triton": jit_kernel(device["_highway_triton"]),
    }
    forward_update = jit_kernel(_forward_update_triton, **bound)
    backward_update = jit_kernel(_backward_update_triton, **bound)
    return _ScanKernels(
        forward=jit_kernel(
            _scan_forward_triton,
            _forward_update_triton=forward_update,
            _forward_operands_triton=jit_kernel(_forward_operands_triton),
            _forward_store_triton=jit_kernel(
                _forward_store_triton,
                _forward_update_triton=forward_update,
            ),
        ),
        backward=jit_kernel(
            _scan_backward_triton,
            _backward_update_triton=backward_update,
            _backward_operands_triton=jit_kernel(_backward_operands_triton),
        ),
        step=jit_kernel(_step_triton, **bound),
    )


# Every floating tensor of a call is passed, so no argument's dtype escapes the rule;
# the terminals, which may be of any dtype, are not.
def _runs_triton(*values: Tensor) -> bool:
    """Whether the kernels take these: every value bf16 or fp32, on a CUDA device."""
    return values[0].is_cuda and all(
        value.dtype in {torch.bfloat16, torch.float32} for value in values
    )


def _require_cuda(tensor: Tensor) -> None:
    """Refuse a tensor off a CUDA device: the reference there has other bits."""
    if not tensor.is_cuda:
        msg = (
            f"TritonScan runs on a CUDA device, not {tensor.device}; select "
            "TorchScan to run the scan elsewhere"
        )
        raise ValueError(msg)


def _check_layout(*tensors: Tensor) -> None:
    """Refuse what the kernels do not address: strided tensors, or two devices."""
    for tensor in tensors:
        if not tensor.is_contiguous() or tensor.device != tensors[0].device:
            raise ValueError("TritonScan needs contiguous tensors on one device.")


def _terminals_view(terminals: Tensor) -> Tensor:
    """View a bool tensor as bytes; the kernels compare against zero either way."""
    return terminals.view(torch.uint8) if terminals.dtype == torch.bool else terminals


# :attr:`TritonScan.helpers` replaces the sigmoid, the lerp and the highway output by
# these names, for another implementation's rounding of each.
def _sigmoid_triton(x: language.tensor) -> language.tensor:
    return language.sigmoid(x)


def _candidate_triton(hidden: language.tensor) -> language.tensor:
    return language.where(hidden >= 0, hidden + 0.5, _sigmoid_triton(hidden))


def _lerp_triton(
    a: language.tensor,
    b: language.tensor,
    w: language.tensor,
) -> language.tensor:
    return a + w * (b - a)


def _highway_triton(
    s: language.tensor,
    h: language.tensor,
    x: language.tensor,
) -> language.tensor:
    """Return the highway output ``s * h + (1 - s) * x``."""
    return s * h + (1.0 - s) * x


# Four steps' operands are loaded before any of them is walked: a load per step, used at
# once behind the previous step's stores, left every step waiting on memory. Every store
# in these kernels rounds to its tensor's dtype, since ``language.store`` casts to the
# pointer's element type: the carry's for the states, the input's for the outputs.
def _scan_forward_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    combined_ptr: language.tensor,
    inputs_ptr: language.tensor,
    initial_ptr: language.tensor,
    terminals_ptr: language.tensor,
    outputs_ptr: language.tensor,
    final_ptr: language.tensor,
    states_ptr: language.tensor,
    count: int,
    time: int,
    width: int,
    block: language.constexpr,
) -> None:
    """Walk the recurrence forward, one ``(batch, width)`` element per lane."""
    idx = language.program_id(0) * block + language.arange(0, block)
    mask = idx < count
    b = idx // width
    h = idx % width
    initial = language.load(initial_ptr + idx, mask=mask, other=0.0)
    carry = initial.to(language.float32)
    full = time - time % 4
    for start in range(0, full, 4):
        row = b * time + start
        # Every load of the four steps is issued before the first store, which
        # the loads could not otherwise pass.
        r0, h0, g0, p0, x0 = _forward_operands_triton(
            combined_ptr,
            inputs_ptr,
            terminals_ptr,
            row,
            h,
            width,
            mask,
        )
        r1, h1, g1, p1, x1 = _forward_operands_triton(
            combined_ptr,
            inputs_ptr,
            terminals_ptr,
            row + 1,
            h,
            width,
            mask,
        )
        r2, h2, g2, p2, x2 = _forward_operands_triton(
            combined_ptr,
            inputs_ptr,
            terminals_ptr,
            row + 2,
            h,
            width,
            mask,
        )
        r3, h3, g3, p3, x3 = _forward_operands_triton(
            combined_ptr,
            inputs_ptr,
            terminals_ptr,
            row + 3,
            h,
            width,
            mask,
        )
        carry = _forward_store_triton(
            outputs_ptr,
            states_ptr,
            row,
            h,
            width,
            mask,
            carry,
            r0,
            h0,
            g0,
            p0,
            x0,
            initial.dtype,
        )
        carry = _forward_store_triton(
            outputs_ptr,
            states_ptr,
            row + 1,
            h,
            width,
            mask,
            carry,
            r1,
            h1,
            g1,
            p1,
            x1,
            initial.dtype,
        )
        carry = _forward_store_triton(
            outputs_ptr,
            states_ptr,
            row + 2,
            h,
            width,
            mask,
            carry,
            r2,
            h2,
            g2,
            p2,
            x2,
            initial.dtype,
        )
        carry = _forward_store_triton(
            outputs_ptr,
            states_ptr,
            row + 3,
            h,
            width,
            mask,
            carry,
            r3,
            h3,
            g3,
            p3,
            x3,
            initial.dtype,
        )
    for t in range(full, time):
        row = b * time + t
        cell = row * width + h
        gates = row * (3 * width) + h
        state, carry, out = _forward_update_triton(
            carry,
            language.load(terminals_ptr + row, mask=mask, other=0),
            language.load(combined_ptr + gates, mask=mask, other=0.0),
            language.load(combined_ptr + gates + width, mask=mask, other=0.0),
            language.load(combined_ptr + gates + 2 * width, mask=mask, other=0.0),
            language.load(inputs_ptr + cell, mask=mask, other=0.0),
            initial.dtype,
        )
        language.store(states_ptr + cell, state, mask=mask)
        language.store(outputs_ptr + cell, out, mask=mask)
    language.store(final_ptr + idx, carry, mask=mask)


def _forward_operands_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    combined_ptr: language.tensor,
    inputs_ptr: language.tensor,
    terminals_ptr: language.tensor,
    row: language.tensor,
    h: language.tensor,
    width: int,
    mask: language.tensor,
) -> tuple[language.tensor, ...]:
    """Load one step's reset, gates and input."""
    gates = row * (3 * width) + h
    return (
        language.load(terminals_ptr + row, mask=mask, other=0),
        language.load(combined_ptr + gates, mask=mask, other=0.0),
        language.load(combined_ptr + gates + width, mask=mask, other=0.0),
        language.load(combined_ptr + gates + 2 * width, mask=mask, other=0.0),
        language.load(inputs_ptr + row * width + h, mask=mask, other=0.0),
    )


def _forward_store_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    outputs_ptr: language.tensor,
    states_ptr: language.tensor,
    row: language.tensor,
    h: language.tensor,
    width: int,
    mask: language.tensor,
    carry: language.tensor,
    reset: language.tensor,
    hidden: language.tensor,
    gate: language.tensor,
    proj: language.tensor,
    x: language.tensor,
    state_dtype: language.dtype,
) -> language.tensor:
    """Walk one step from loaded operands, store its state and output, return the carry."""
    state, carry, out = _forward_update_triton(
        carry,
        reset,
        hidden,
        gate,
        proj,
        x,
        state_dtype,
    )
    cell = row * width + h
    language.store(states_ptr + cell, state, mask=mask)
    language.store(outputs_ptr + cell, out, mask=mask)
    return carry


def _forward_update_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    carry: language.tensor,
    reset: language.tensor,
    hidden: language.tensor,
    gate: language.tensor,
    proj: language.tensor,
    x: language.tensor,
    state_dtype: language.dtype,
) -> tuple[language.tensor, language.tensor, language.tensor]:
    """One step: the reset carry, the next rounded to ``state_dtype``, the output."""
    state = language.where(reset != 0, 0.0, carry)
    z = _sigmoid_triton(gate.to(language.float32))
    updated = _lerp_triton(state, _candidate_triton(hidden.to(language.float32)), z)
    s = _sigmoid_triton(proj.to(language.float32))
    out = _highway_triton(s, updated, x.to(language.float32))
    return state, updated.to(state_dtype).to(language.float32), out


def _step_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    combined_ptr: language.tensor,
    inputs_ptr: language.tensor,
    state_ptr: language.tensor,
    outputs_ptr: language.tensor,
    next_ptr: language.tensor,
    count: int,
    width: int,
    block: language.constexpr,
) -> None:
    """Run one time step of the recurrence, one element per lane."""
    idx = language.program_id(0) * block + language.arange(0, block)
    mask = idx < count
    gates = (idx // width) * (3 * width) + idx % width
    hidden = language.load(combined_ptr + gates, mask=mask, other=0.0)
    gate = language.load(combined_ptr + gates + width, mask=mask, other=0.0)
    proj = language.load(combined_ptr + gates + 2 * width, mask=mask, other=0.0)
    x = language.load(inputs_ptr + idx, mask=mask, other=0.0).to(language.float32)
    state = language.load(state_ptr + idx, mask=mask, other=0.0).to(language.float32)
    z = _sigmoid_triton(gate.to(language.float32))
    carry = _lerp_triton(state, _candidate_triton(hidden.to(language.float32)), z)
    s = _sigmoid_triton(proj.to(language.float32))
    out = _highway_triton(s, carry, x)
    language.store(outputs_ptr + idx, out, mask=mask)
    language.store(next_ptr + idx, carry, mask=mask)


# As the forward, four steps' operands are loaded before any is walked.
def _scan_backward_triton(  # noqa: PLR0917 -- The signature is the kernel's positional ABI.
    combined_ptr: language.tensor,
    inputs_ptr: language.tensor,
    states_ptr: language.tensor,
    terminals_ptr: language.tensor,
    grad_outputs_ptr: language.tensor,
    grad_combined_ptr: language.tensor,
    grad_inputs_ptr: language.tensor,
    grad_initial_ptr: language.tensor,
    count: int,
    time: int,
    width: int,
    block: language.constexpr,
) -> None:
    """Walk the recurrence backward, one ``(batch, width)`` element per lane."""
    idx = language.program_id(0) * block + language.arange(0, block)
    mask = idx < count
    b = idx // width
    h = idx % width
    dh = language.zeros([block], dtype=language.float32)
    full = time - time % 4
    for step in range(0, full, 4):
        # The walk runs backwards: ``row`` is the newest of the four steps, and
        # every load of the four is issued before the first store.
        row = b * time + (time - 1 - step)
        s0, h0, g0, p0, x0, o0, r0 = _backward_operands_triton(
            combined_ptr,
            inputs_ptr,
            states_ptr,
            terminals_ptr,
            grad_outputs_ptr,
            row,
            h,
            width,
            mask,
        )
        s1, h1, g1, p1, x1, o1, r1 = _backward_operands_triton(
            combined_ptr,
            inputs_ptr,
            states_ptr,
            terminals_ptr,
            grad_outputs_ptr,
            row - 1,
            h,
            width,
            mask,
        )
        s2, h2, g2, p2, x2, o2, r2 = _backward_operands_triton(
            combined_ptr,
            inputs_ptr,
            states_ptr,
            terminals_ptr,
            grad_outputs_ptr,
            row - 2,
            h,
            width,
            mask,
        )
        s3, h3, g3, p3, x3, o3, r3 = _backward_operands_triton(
            combined_ptr,
            inputs_ptr,
            states_ptr,
            terminals_ptr,
            grad_outputs_ptr,
            row - 3,
            h,
            width,
            mask,
        )
        dh = _backward_update_triton(
            grad_combined_ptr,
            grad_inputs_ptr,
            row,
            h,
            width,
            mask,
            dh,
            s0,
            h0,
            g0,
            p0,
            x0,
            o0,
            r0,
        )
        dh = _backward_update_triton(
            grad_combined_ptr,
            grad_inputs_ptr,
            row - 1,
            h,
            width,
            mask,
            dh,
            s1,
            h1,
            g1,
            p1,
            x1,
            o1,
            r1,
        )
        dh = _backward_update_triton(
            grad_combined_ptr,
            grad_inputs_ptr,
            row - 2,
            h,
            width,
            mask,
            dh,
            s2,
            h2,
            g2,
            p2,
            x2,
            o2,
            r2,
        )
        dh = _backward_update_triton(
            grad_combined_ptr,
            grad_inputs_ptr,
            row - 3,
            h,
            width,
            mask,
            dh,
            s3,
            h3,
            g3,
            p3,
            x3,
            o3,
            r3,
        )
    for step in range(full, time):
        row = b * time + (time - 1 - step)
        cell = row * width + h
        gates = row * (3 * width) + h
        dh = _backward_update_triton(
            grad_combined_ptr,
            grad_inputs_ptr,
            row,
            h,
            width,
            mask,
            dh,
            language.load(states_ptr + cell, mask=mask, other=0.0),
            language.load(combined_ptr + gates, mask=mask, other=0.0),
            language.load(combined_ptr + gates + width, mask=mask, other=0.0),
            language.load(combined_ptr + gates + 2 * width, mask=mask, other=0.0),
            language.load(inputs_ptr + cell, mask=mask, other=0.0),
            language.load(grad_outputs_ptr + cell, mask=mask, other=0.0),
            language.load(terminals_ptr + row, mask=mask, other=0),
        )
    language.store(grad_initial_ptr + idx, dh, mask=mask)


def _backward_operands_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    combined_ptr: language.tensor,
    inputs_ptr: language.tensor,
    states_ptr: language.tensor,
    terminals_ptr: language.tensor,
    grad_outputs_ptr: language.tensor,
    row: language.tensor,
    h: language.tensor,
    width: int,
    mask: language.tensor,
) -> tuple[language.tensor, ...]:
    """Load one step's carry, gates, input, output gradient and reset."""
    cell = row * width + h
    gates = row * (3 * width) + h
    return (
        language.load(states_ptr + cell, mask=mask, other=0.0),
        language.load(combined_ptr + gates, mask=mask, other=0.0),
        language.load(combined_ptr + gates + width, mask=mask, other=0.0),
        language.load(combined_ptr + gates + 2 * width, mask=mask, other=0.0),
        language.load(inputs_ptr + cell, mask=mask, other=0.0),
        language.load(grad_outputs_ptr + cell, mask=mask, other=0.0),
        language.load(terminals_ptr + row, mask=mask, other=0),
    )


def _backward_update_triton(  # noqa: PLR0917 -- A device function's positional ABI.
    grad_combined_ptr: language.tensor,
    grad_inputs_ptr: language.tensor,
    row: language.tensor,
    h: language.tensor,
    width: int,
    mask: language.tensor,
    dh: language.tensor,
    previous: language.tensor,
    hidden: language.tensor,
    gate: language.tensor,
    proj: language.tensor,
    x: language.tensor,
    g: language.tensor,
    reset: language.tensor,
) -> language.tensor:
    """One step of the backward walk: store its gradients, return the next ``dh``."""
    cell = row * width + h
    gates = row * (3 * width) + h
    previous = previous.to(language.float32)
    hidden = hidden.to(language.float32)
    x = x.to(language.float32)
    g = g.to(language.float32)
    z = _sigmoid_triton(gate.to(language.float32))
    candidate = _candidate_triton(hidden)
    carry = _lerp_triton(previous, candidate, z)
    s = _sigmoid_triton(proj.to(language.float32))
    grad_proj = g * (carry - x) * s * (1.0 - s)
    language.store(grad_inputs_ptr + cell, g * (1.0 - s), mask=mask)
    language.store(grad_combined_ptr + gates + 2 * width, grad_proj, mask=mask)
    # Written as fma so it fuses under launch options that disable contraction too.
    total = language.fma(g, s, dh)
    grad_candidate = total * z
    grad_gate = total * (candidate - previous) * z * (1.0 - z)
    grad_hidden = language.where(
        hidden >= 0,
        grad_candidate,
        grad_candidate * candidate * (1.0 - candidate),
    )
    language.store(grad_combined_ptr + gates, grad_hidden, mask=mask)
    language.store(grad_combined_ptr + gates + width, grad_gate, mask=mask)
    return language.where(reset != 0, 0.0, total * (1.0 - z))


def _candidate(hidden: Tensor) -> Tensor:
    """Return the candidate: a shifted identity above zero, a sigmoid below."""
    return torch.where(hidden >= 0, hidden + 0.5, torch.sigmoid(hidden))
