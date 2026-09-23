"""Tests for the FP32 minimal gated recurrent unit."""

import pytest
import torch

from priml.model.min_gru import MinGRU


def test_min_gru_matches_hand_forward_and_returns_layer_state() -> None:
    model = MinGRU(input_channels=2, channels=2, layers=1)
    with torch.no_grad():
        model.layers[0].weight.zero_()
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    state = torch.zeros(1, 1, 2)
    outputs, final = model(inputs, state=state, reset=torch.tensor([[True, False]]))
    expected = torch.tensor([[[0.625, 1.125], [1.6875, 2.1875]]])
    assert outputs.shape == inputs.shape
    assert final.shape == state.shape
    torch.testing.assert_close(outputs, expected)


def test_min_gru_repeated_step_equals_sequence() -> None:
    torch.manual_seed(0)
    model = MinGRU(input_channels=3, channels=4, layers=2)
    inputs = torch.randn(2, 5, 3)
    reset = torch.zeros(2, 5, dtype=torch.bool)
    sequence, state = model(inputs, reset=reset)
    carry = model.initial_state(2)
    pieces: list[torch.Tensor] = []
    for index in range(inputs.shape[1]):
        output, carry = model(
            inputs[:, index : index + 1],
            state=carry,
            reset=reset[:, index : index + 1],
        )
        pieces.append(output)
    torch.testing.assert_close(sequence, torch.cat(pieces, dim=1))
    torch.testing.assert_close(state, carry)


def test_min_gru_reset_discards_previous_state() -> None:
    model = MinGRU(input_channels=2, channels=2, layers=1)
    inputs = torch.randn(1, 2, 2)
    reset = torch.tensor([[False, True]])
    outputs, _ = model(inputs, reset=reset)
    fresh, _ = model(
        inputs[:, 1:],
        state=model.initial_state(1),
        reset=torch.ones(1, 1, dtype=torch.bool),
    )
    torch.testing.assert_close(outputs[:, 1:], fresh)


def test_min_gru_rejects_an_empty_time_dimension() -> None:
    """Reject T=0 at the public boundary before stacked outputs fail incidentally."""
    model = MinGRU(input_channels=2, channels=2)

    with pytest.raises(ValueError, match="time"):
        model(torch.zeros(1, 0, 2))


def test_min_gru_has_gradient_through_inputs_and_parameters() -> None:
    model = MinGRU(input_channels=2, channels=3, layers=2)
    inputs = torch.randn(2, 4, 2, requires_grad=True)
    outputs, _ = model(inputs)
    outputs.square().mean().backward()
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("batch", "time", "channels", "layers"),
    [
        (1, 1, 3, 1),
        (2, 5, 4, 2),
        (3, 17, 6, 3),
        (4, 64, 8, 2),
    ],
)
def test_min_gru_vectorized_scan_matches_sequential_reference(
    batch: int,
    time: int,
    channels: int,
    layers: int,
) -> None:
    """The doubling scan must reproduce the loop it replaced, resets included."""
    torch.manual_seed(0)
    model = MinGRU(input_channels=channels, channels=channels, layers=layers)
    inputs = torch.randn(batch, time, channels)
    reset = torch.rand(batch, time) < 0.3
    state = torch.randn(layers, batch, channels)

    outputs, final = model(inputs, state=state, reset=reset)
    expected_outputs, expected_final = _sequential_min_gru(
        model,
        inputs=inputs,
        state=state,
        reset=reset,
    )

    torch.testing.assert_close(outputs, expected_outputs, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final, expected_final, atol=1e-5, rtol=1e-5)


def test_min_gru_forced_fp32_ignores_an_ambient_bf16_autocast() -> None:
    """MinGRU's carry must not degrade under a caller's bf16 autocast region."""
    torch.manual_seed(1)
    model = MinGRU(input_channels=4, channels=4, layers=2)
    inputs = torch.randn(2, 6, 4)
    reset = torch.rand(2, 6) < 0.3

    plain_outputs, plain_final = model(inputs, reset=reset)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_outputs, autocast_final = model(inputs, reset=reset)

    assert autocast_outputs.dtype == torch.float32
    assert autocast_final.dtype == torch.float32
    torch.testing.assert_close(autocast_outputs, plain_outputs, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(autocast_final, plain_final, atol=1e-6, rtol=1e-6)


def _sequential_min_gru(
    model: MinGRU,
    *,
    inputs: torch.Tensor,
    state: torch.Tensor,
    reset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reimplement the pre-scan per-step loop as an independent oracle."""
    time = inputs.shape[1]
    value = model.input_projection(inputs.float())
    carries: list[torch.Tensor] = []
    for layer_index, layer in enumerate(model.layers):
        combined = layer(value).float()
        hidden, gate, highway = combined.chunk(3, dim=-1)
        carry = state[layer_index].float()
        outputs: list[torch.Tensor] = []
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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
