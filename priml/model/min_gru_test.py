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


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
