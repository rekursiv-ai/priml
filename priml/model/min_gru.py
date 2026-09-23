"""FP32 minimal gated recurrent unit for batch-first sequences."""

from __future__ import annotations

from typing import override

from torch import Tensor, nn
from torch.nn import functional

import torch


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


__all__ = ["MinGRU"]


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
