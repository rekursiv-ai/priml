"""Tests for attribute-passthrough mixins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

from torch import nn

import pytest
import torch

from priml.model.passthrough import (
    ReadPassthroughMixin,
    ReadWritePassthroughMixin,
)
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


@dataclass(kw_only=True, slots=True)
class _Target:
    value: int = 1


class _ReadOnly(ReadPassthroughMixin, passthrough="inner"):
    own: int

    def __init__(self) -> None:
        self.inner = _Target()
        self.own = 0


class _ReadWrite(ReadWritePassthroughMixin, passthrough="inner"):
    def __init__(self) -> None:
        self.inner = _Target()


class _TargetModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))
        self.factor = 1.0

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.weight


class _ReadModule(ReadPassthroughMixin, nn.Module, passthrough="inner"):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _TargetModule()

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * cast(torch.Tensor, self.weight)


class _ReadWriteModule(ReadWritePassthroughMixin, nn.Module, passthrough="inner"):
    def __init__(self) -> None:
        super().__init__()
        self.inner = _TargetModule()
        self.factor = 3.0

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.inner(input) * self.factor


class _PassthroughModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.read = _ReadModule()
        self.read_write = _ReadWriteModule()

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.read(input), self.read_write(input)))


def test_passthrough_text(request: pytest.FixtureRequest) -> None:
    assert_text_golden(
        request,
        test_file=__file__,
        name="passthrough",
        rendered=str(_PassthroughModule()),
    )


def test_passthrough_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="passthrough",
        build_module=_PassthroughModule,
        build_input=lambda: torch.tensor([1.0, -2.0]),
        seed=0,
    )


def test_read_passthrough_delegates_missing_attributes() -> None:
    wrapper = _ReadOnly()

    assert wrapper.value == 1
    wrapper.own = 2
    assert wrapper.own == 2


def test_read_write_passthrough_delegates_existing_attributes() -> None:
    wrapper = _ReadWrite()

    wrapper.value = 3

    assert wrapper.inner.value == 3
    with pytest.raises(AttributeError, match="typo"):
        wrapper.typo = 4


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
