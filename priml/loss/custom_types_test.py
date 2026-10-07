"""Tests for loss type contracts."""

from inspect import Parameter, signature
from typing import get_args, get_type_hints

from torch import Tensor

from priml.loss.custom_types import LossOutput, SimpleLossFn


def test_loss_output_requires_loss_and_allows_tensor_values() -> None:
    assert LossOutput.__required_keys__ == {"loss"}
    assert get_type_hints(LossOutput)["loss"] is Tensor
    assert type.__getattribute__(LossOutput, "__extra_items__") is Tensor


def test_simple_loss_fn_signature() -> None:
    call = SimpleLossFn.__call__
    parameters = signature(call).parameters
    hints = get_type_hints(call)
    assert list(parameters) == ["self", "input", "target", "reduction"]
    assert parameters["reduction"].kind is Parameter.KEYWORD_ONLY
    assert hints["input"] is Tensor
    assert hints["target"] is Tensor
    assert get_args(hints["reduction"]) == ("none", "mean", "sum")
    assert hints["return"] is Tensor


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
