"""Tests for linear module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from torch.distributed.tensor import Replicate, Shard

import pytest
import torch

from priml.model import linear
from priml.model.linear import EnsembleLinear, Linear
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def test_linear_config_pprint(request: pytest.FixtureRequest) -> None:
    config = Linear.Config(4, 3)
    assert_text_golden(
        request,
        test_file=__file__,
        name="linear",
        rendered=config.pformat(hide_default_values=False),
    )


def test_linear_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="linear",
        build_module=lambda: Linear.Config(4, 3).make(),
        build_input=lambda: torch.randn(1, 2, 4),
        seed=0,
    )


def test_ensemble_linear_config_pprint(request: pytest.FixtureRequest) -> None:
    config = EnsembleLinear.Config(4, 3, num_ensemble=2)
    assert_text_golden(
        request,
        test_file=__file__,
        name="ensemble_linear",
        rendered=config.pformat(hide_default_values=False),
    )


def test_ensemble_linear_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="ensemble_linear",
        build_module=lambda: EnsembleLinear.Config(4, 3, num_ensemble=2).make(),
        build_input=lambda: torch.randn(1, 2, 4),
        seed=0,
    )


def test_linear():
    m = Linear.Config(32, 64).make()
    x = torch.randn(2, 8, 32)
    assert m(x).shape == (2, 8, 64)
    assert m.bias is None


def test_linear_bias():
    m = Linear.Config(32, 64, bias=True).make()
    assert m.bias is not None


def test_linear_channels_infer_from_out():
    m = Linear.Config(channels_out=64).make()
    assert m.in_features == 64


def test_linear_channels_infer_from_in():
    m = Linear.Config(channels_in=64).make()
    assert m.out_features == 64


def test_linear_forward_accepts_messages_and_rejects_positional_extras():
    m = Linear.Config(32, 32).make()
    x = torch.randn(2, 8, 32)
    assert m(x, key="val").shape == (2, 8, 32)
    with pytest.raises(TypeError):
        m(x, "extra")


def test_linear_reset_parameters():
    m = Linear.Config(32, 32).make()
    m.reset_parameters()


def test_linear_arbitrary_batch_dims():
    """Linear supports arbitrary leading batch dims."""
    m = Linear.Config(16, 32).make()
    x = torch.randn(2, 3, 4, 8, 16)
    assert m(x).shape == (2, 3, 4, 8, 32)


def test_ensemble_linear_infers_either_channel_boundary() -> None:
    assert EnsembleLinear.Config(channels_in=64).finalize().channels_out == 64
    assert EnsembleLinear.Config(channels_out=32).finalize().channels_in == 32


def test_ensemble_linear():
    m = EnsembleLinear.Config(channels_in=64, channels_out=16, num_ensemble=4).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 4, 16)


def test_ensemble_linear_with_bias():
    m = EnsembleLinear.Config(
        channels_in=64,
        channels_out=16,
        num_ensemble=4,
        bias=True,
    ).make()
    assert m.bias is not None
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 4, 16)


def test_ensemble_linear_reset():
    m = EnsembleLinear.Config(channels_in=64, channels_out=16, num_ensemble=4).make()
    m.reset_parameters()


def test_ensemble_parallel_style_shards_parameters_and_installs_input_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = EnsembleLinear.Config(4, 3, num_ensemble=2, bias=True).make()
    mesh = cast("DeviceMesh", object())
    distributed: list[tuple[torch.Tensor, object]] = []
    hooked_inputs: list[tuple[object, ...]] = []

    def distribute(
        tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: object,
    ) -> torch.Tensor:
        assert device_mesh is mesh
        distributed.append((tensor, placements))
        return tensor

    def replicate(
        device_mesh: DeviceMesh,
        hooked_module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> tuple[object, ...]:
        assert device_mesh is mesh
        assert hooked_module is module
        hooked_inputs.append(inputs)
        return inputs

    monkeypatch.setattr(linear, "distribute_tensor", distribute)
    monkeypatch.setattr(linear, "_replicate_input", replicate)

    style = module.tensor_parallel_style()
    assert style._apply(module, mesh) is module
    module(torch.randn(1, 4))

    assert [placements for _, placements in distributed] == [
        [Shard(0)],
        [Shard(0)],
    ]
    assert len(hooked_inputs) == 1


def test_ensemble_parallel_input_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = cast("DeviceMesh", object())

    class FakeDTensor:
        def __init__(self, local: object) -> None:
            self.local = local
            self.redistributions: list[tuple[DeviceMesh, object]] = []

        @classmethod
        def from_local(
            cls,
            local: object,
            device_mesh: DeviceMesh,
            placements: object,
            *,
            run_check: bool,
        ) -> FakeDTensor:
            assert device_mesh is mesh
            assert placements == [Replicate()]
            assert run_check is False
            return cls(local)

        def redistribute(
            self,
            device_mesh: DeviceMesh,
            placements: object,
        ) -> FakeDTensor:
            self.redistributions.append((device_mesh, placements))
            return self

    monkeypatch.setattr(linear, "DTensor", FakeDTensor)
    local = object()

    replicated, extra = linear._replicate_input(
        mesh,
        torch.nn.Identity(),
        (local, "extra"),
    )
    assert isinstance(replicated, FakeDTensor)
    assert replicated.local is local
    assert extra == "extra"

    (redistributed,) = linear._replicate_input(
        mesh,
        torch.nn.Identity(),
        (replicated,),
    )
    assert redistributed is replicated
    assert replicated.redistributions == [(mesh, [Replicate()])]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
