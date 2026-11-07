"""Exercise eager dispatch without importing compiler internals."""

from __future__ import annotations

from typing import TYPE_CHECKING, override
from unittest.mock import Mock

import torch

from priml.testing.eager_dispatch import EagerDispatchMode


if TYPE_CHECKING:
    from torch._ops import OpOverload

    import pytest


class _RecordingMode(EagerDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[
            tuple[OpOverload[..., object], tuple[object, ...], dict[str, object]]
        ] = []

    @override
    def _dispatch(
        self,
        func: OpOverload[..., object],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        self.calls.append((func, args, kwargs))
        return func(*args, **kwargs)


def test_subclasses_skip_torchs_automatic_dynamo_wrapper() -> None:
    assert not _RecordingMode._should_skip_dynamo()
    assert "__wrapped__" not in vars(_RecordingMode.__torch_dispatch__)


def test_eager_dispatch_keeps_arguments_and_never_disables_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disable = Mock(side_effect=AssertionError("eager dispatch imported the compiler"))
    monkeypatch.setattr(torch.compiler, "disable", disable)
    left, right = torch.tensor([1, 2]), torch.tensor([3, 4])
    with _RecordingMode() as mode:
        actual = torch.add(left, right, alpha=2)
    assert torch.equal(actual, torch.tensor([7, 10]))
    assert len(mode.calls) == 1
    func, args, kwargs = mode.calls[0]
    assert func is torch.ops.aten.add.Tensor
    assert args[0] is left
    assert args[1] is right
    assert kwargs == {"alpha": 2}
    disable.assert_not_called()


def test_dispatch_normalizes_absent_keyword_arguments() -> None:
    value = torch.tensor([1, 2])
    mode = _RecordingMode()
    actual = mode.__torch_dispatch__(torch.ops.aten.neg.default, (), (value,))
    assert isinstance(actual, torch.Tensor)
    assert torch.equal(actual, torch.tensor([-1, -2]))
    assert mode.calls[0][2] == {}


def test_compiling_dispatch_disables_only_the_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def passthrough[F](function: F) -> F:
        return function

    disable = Mock(side_effect=passthrough)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(torch.compiler, "disable", disable)
    value = torch.tensor([1, 2])
    mode = _RecordingMode()
    actual = mode.__torch_dispatch__(torch.ops.aten.neg.default, (), (value,))
    assert isinstance(actual, torch.Tensor)
    assert torch.equal(actual, torch.tensor([-1, -2]))
    disable.assert_called_once_with(mode._dispatch)
    assert mode.calls[0][2] == {}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
