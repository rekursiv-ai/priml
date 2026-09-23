"""FP32 minimal gated recurrent unit for batch-first sequences."""

from __future__ import annotations

from typing import override

from torch import Tensor, nn

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

        value = self.input_projection(inputs.float())
        carries: list[Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            combined = layer(value).float()
            hidden, gate, highway = combined.chunk(3, dim=-1)
            carry = state[layer_index].float()
            outputs: list[Tensor] = []
            for step in range(time):
                carry = torch.where(reset[:, step, None], 0.0, carry)
                candidate = torch.where(
                    hidden[:, step] >= 0,
                    hidden[:, step] + 0.5,
                    torch.sigmoid(hidden[:, step]),
                )
                carry = torch.lerp(carry, candidate, torch.sigmoid(gate[:, step]))
                strength = torch.sigmoid(highway[:, step])
                outputs.append((1 - strength) * value[:, step] + strength * carry)
            value = torch.stack(outputs, dim=1)
            carries.append(carry)
        return value, torch.stack(carries)


__all__ = ["MinGRU"]
