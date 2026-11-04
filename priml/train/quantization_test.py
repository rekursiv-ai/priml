"""Tests for quantization strategies."""

from __future__ import annotations

from enum import Enum
from importlib import util
from importlib.machinery import ModuleSpec
from typing import TYPE_CHECKING

import dataclasses
import logging
import sys

from torch import Tensor, nn
from torch._library import opaque_object
from torch.utils import _pytree

import pytest
import torch

from priml.train import quantization
from priml.train.quantization import (
    Float8ModelQuantization,
    NoModelQuantization,
    _check_float8_available,
)


if TYPE_CHECKING:
    from torchao.float8.float8_linear import Float8Linear
else:
    from wrapt import lazy_import

    Float8Linear = lazy_import("torchao.float8.float8_linear", "Float8Linear")


def test_no_quantization_is_identity() -> None:
    model = nn.Linear(16, 16)
    out = NoModelQuantization.Config().make()(model)
    assert out is model


def test_disabled_quantization_does_not_import_torchao(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as isolated:
        for name in tuple(sys.modules):
            if name == "torchao" or name.startswith("torchao."):
                isolated.setitem(sys.modules, name, None)
        isolated.setitem(sys.modules, "torchao", None)
        spec = util.spec_from_file_location(
            f"{quantization.__name__}_isolated",
            quantization.__file__,
        )
        assert spec is not None
        assert spec.loader is not None
        module = util.module_from_spec(spec)
        isolated.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        model = nn.Linear(2, 3)
        assert NoModelQuantization.Config().make()(model) is model
        strategy = Float8ModelQuantization.Config().make()
        assert strategy.enabled is False
        assert strategy(model) is model
        assert _check_float8_available() == (False, "torchao not installed")


class _Preference(Enum):
    AUTO = "auto"
    MANUAL = "manual"


@pytest.mark.parametrize("native", [False, True])
def test_torchao_registration_preserves_native_enum_support(
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
) -> None:
    calls: list[type[object]] = []

    def is_custom_class(cls: type[object]) -> bool:
        return native and issubclass(cls, Enum)

    monkeypatch.setattr(
        opaque_object,
        "is_custom_class",
        is_custom_class,
        raising=False,
    )
    monkeypatch.setattr(_pytree, "register_constant", calls.append)
    quantization._prepare_torchao()
    prepared = _pytree.register_constant
    quantization._prepare_torchao()
    assert _pytree.register_constant is prepared
    _pytree.register_constant(_Preference)
    _pytree.register_constant(str)
    assert calls == ([str] if native else [_Preference, str])


def test_torchao_registration_supports_older_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[type[object]] = []
    monkeypatch.delattr(opaque_object, "is_custom_class", raising=False)
    monkeypatch.setattr(_pytree, "register_constant", calls.append)
    quantization._prepare_torchao()
    _pytree.register_constant(_Preference)
    assert calls == [_Preference]


def test_native_enum_registration_is_quiet_and_compilable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    quantization._prepare_torchao()
    _pytree.register_constant(_Preference)
    assert not any("register_constant()" in record.message for record in caplog.records)

    def apply(value: Tensor, preference: _Preference) -> Tensor:
        return value + 1 if preference is _Preference.AUTO else value - 1

    value = torch.arange(2, dtype=torch.float32)
    compiled = torch.compile(apply, backend="eager", fullgraph=True)
    assert torch.equal(compiled(value, _Preference.AUTO), value + 1)
    assert torch.equal(compiled(value, _Preference.MANUAL), value - 1)


def _force_enabled() -> Float8ModelQuantization:
    """Build a Float8 strategy, skipping if float8 isn't available here."""
    available, reason = _check_float8_available()
    if not available:
        pytest.skip(f"float8 unavailable: {reason}")
    return Float8ModelQuantization.Config().make()


def test_default_filter_is_pure() -> None:
    """T-026: the module filter predicate must not mutate conversion state."""
    quant = _force_enabled()
    quant.converted_count = 0
    module = nn.Linear(16, 16)

    # Calling the predicate repeatedly (as torchao may) must not inflate count.
    quant._default_module_filter(module, "linear")
    quant._default_module_filter(module, "linear")

    assert quant.converted_count == 0, (
        "filter predicate mutated converted_count (counts filter calls, not conversions)"
    )


def test_disabled_instance_has_consistent_attributes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-027: a disabled instance must not leave attributes unset."""
    monkeypatch.setattr(
        "priml.train.quantization._check_float8_available",
        lambda: (False, "forced-disabled"),
    )
    config = Float8ModelQuantization.Config()
    quant = config.make()

    assert quant.config == config.finalized()
    assert quant.enabled is False
    assert quant.converted_count == 0
    assert quant.module_filter is None
    assert caplog.records[-1].getMessage() == "Float8 disabled: forced-disabled"
    # And a disabled strategy is an identity transform.
    model = nn.Linear(16, 16)
    assert quant(model) is model


@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise", "rowwise_gw_hp"])
def test_enabled_quantization_preserves_recipe_and_initial_state(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    recipe: str,
) -> None:
    monkeypatch.setattr(quantization, "_check_float8_available", lambda: (True, ""))
    caplog.set_level(logging.INFO, logger=quantization.__name__)
    config = Float8ModelQuantization.Config(recipe=recipe)
    strategy = config.make()
    recipes = {
        "tensorwise": quantization.Float8LinearRecipeName.TENSORWISE,
        "rowwise": quantization.Float8LinearRecipeName.ROWWISE,
        "rowwise_gw_hp": quantization.Float8LinearRecipeName.ROWWISE_WITH_GW_HP,
    }
    expected = quantization.Float8LinearConfig.from_recipe_name(recipes[recipe])
    assert strategy.config == config.finalized()
    assert strategy.enabled is True
    assert strategy.converted_count == 0
    assert strategy.float8_config == expected
    assert caplog.records[-1].getMessage() == (
        f"Float8Linear enabled: recipe={recipe}, fsdp_all_gather=False"
    )


def test_fsdp_all_gather_uses_dataclass_replace(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-028: enabling FSDP all-gather must produce a valid Float8 config."""
    monkeypatch.setattr(quantization, "_check_float8_available", lambda: (True, ""))
    caplog.set_level(logging.INFO, logger=quantization.__name__)

    quant = Float8ModelQuantization.Config(
        recipe="tensorwise",
        enable_fsdp_float8_all_gather=True,
    ).make()

    assert dataclasses.is_dataclass(quant.float8_config)
    assert quant.float8_config.enable_fsdp_float8_all_gather is True
    assert caplog.records[-1].getMessage() == (
        "Float8Linear enabled: recipe=tensorwise, fsdp_all_gather=True"
    )


def test_unknown_recipe_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quantization, "_check_float8_available", lambda: (True, ""))
    with pytest.raises(ValueError, match="Invalid recipe 'blockwise'"):
        Float8ModelQuantization.Config(recipe="blockwise").make()


def test_fsdp_all_gather_requires_tensorwise_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quantization, "_check_float8_available", lambda: (True, ""))
    with pytest.raises(
        ValueError,
        match=r"^enable_fsdp_float8_all_gather only works with recipe='tensorwise'$",
    ):
        Float8ModelQuantization.Config(
            recipe="rowwise",
            enable_fsdp_float8_all_gather=True,
        ).make()


def test_availability_reports_missing_torchao(monkeypatch: pytest.MonkeyPatch) -> None:
    def find_spec(name: str) -> ModuleSpec | None:
        return None if name == "torchao" else ModuleSpec(name, None)

    monkeypatch.setattr(util, "find_spec", find_spec)
    assert _check_float8_available() == (False, "torchao not installed")


def test_availability_reports_missing_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _check_float8_available() == (False, "CUDA not available")


def test_availability_reports_old_compute_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    available, reason = _check_float8_available()
    assert available is False
    assert reason == "Float8 requires SM89+ (H100, 5090). Found SM80"


@pytest.mark.parametrize("capability", [(8, 9), (9, 0)])
def test_availability_accepts_supported_devices(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    assert _check_float8_available() == (True, "")


def test_float8_conversion_receives_filter_and_logs_exact_count(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(quantization, "_check_float8_available", lambda: (True, ""))
    calls: list[tuple[nn.Module, object, object]] = []

    def convert(
        model: nn.Module,
        *,
        config: object,
        module_filter_fn: object,
    ) -> nn.Module:
        calls.append((model, config, module_filter_fn))
        return model

    monkeypatch.setattr(quantization, "convert_to_float8_training", convert)
    caplog.set_level(logging.INFO, logger=quantization.__name__)
    filter_names: list[str] = []

    def select_first(module: nn.Module, name: str) -> bool:
        filter_names.append(name)
        return isinstance(module, nn.Linear) and name == "0"

    quant = Float8ModelQuantization.Config(module_filter=select_first).make()
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(8, 16))

    assert quant(model) is model
    assert filter_names == ["", "0", "1"]
    assert quant.converted_count == 1
    assert len(calls) == 1
    called_model, config, module_filter_fn = calls[0]
    assert called_model is model
    assert config == quant.float8_config
    assert callable(module_filter_fn)
    assert caplog.records[-1].getMessage() == ("Converted 1 nn.Linear → Float8Linear")


def test_float8_linear_forward_keeps_autocast_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The surrounding autocast must reach ``Float8Linear.forward`` unchanged.

    torchao's ``Float8Linear.forward`` reads ``torch.is_autocast_enabled()`` and
    casts its input to the autocast dtype before the fp8 matmul; quantization
    must NOT disable autocast around it (an earlier wrapper did, which left the
    input fp32 and crashed the fp8 matmul with an unsupported ``aten.to.dtype``
    on the Float8 tensor subclass).
    """
    monkeypatch.setattr(quantization, "_check_float8_available", lambda: (True, ""))

    observed: list[bool] = []

    def _spy_forward(self: Float8Linear, input: Tensor) -> Tensor:
        del self
        observed.append(torch.is_autocast_enabled("cpu"))
        return input

    strategy = Float8ModelQuantization.Config().make()
    monkeypatch.setattr(Float8Linear, "forward", _spy_forward)
    model = nn.Sequential(nn.Linear(16, 16, bias=False))
    quantized = strategy(model)

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        quantized(torch.randn(2, 16))

    assert observed == [True]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
