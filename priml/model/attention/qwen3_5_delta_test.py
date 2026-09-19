"""Qwen3.5 delta-layer parity, cache continuation, and serialization."""

from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeGuard, cast

import inspect

from torch import nn

import pytest
import torch

from priml.model.attention.gated_delta_net import GatedDeltaNet
from priml.model.attention.qwen3_5_delta import (
    Qwen35GatedDeltaNet,
    Qwen35RMSNormGated,
)
from priml.model.norm import CenteredRMSNorm, RMSNorm
from priml.testing.bfb import portable_half_precision
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.qwen3_5 import hf_tensor, torch_reference


if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache, LinearAttentionCacheLayerMixin
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
else:
    from wrapt import lazy_import

    # Avoids the measured 3.3--3.5s fresh-process Qwen reference import.
    DynamicCache = lazy_import("transformers.cache_utils", "DynamicCache")
    LinearAttentionCacheLayerMixin = lazy_import(
        "transformers.cache_utils",
        "LinearAttentionCacheLayerMixin",
    )
    Qwen3_5TextConfig = lazy_import(
        "transformers.models.qwen3_5.configuration_qwen3_5",
        "Qwen3_5TextConfig",
    )
    Qwen3_5GatedDeltaNet = lazy_import(
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "Qwen3_5GatedDeltaNet",
    )


def _reference() -> nn.Module:
    pytest.importorskip("transformers")
    config = Qwen3_5TextConfig(
        hidden_size=8,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=4,
    )
    raw_reference: object = Qwen3_5GatedDeltaNet(config, layer_idx=0)
    assert isinstance(raw_reference, nn.Module)
    return torch_reference(raw_reference)


def _native() -> Qwen35GatedDeltaNet:
    config = Qwen35GatedDeltaNet.Config()
    config.channels_in = 8
    config.num_heads_k = 1
    config.num_heads_v = 2
    config.channels_k_head = 4
    config.channels_v_head = 3
    return config.make()


# Native ``proj_*`` attributes against Hugging Face's parameter names.
_HF_NAMES: Final = {
    "proj_qkv.weight": "in_proj_qkv.weight",
    "proj_z.weight": "in_proj_z.weight",
    "proj_b.weight": "in_proj_b.weight",
    "proj_a.weight": "in_proj_a.weight",
    "proj_out.weight": "out_proj.weight",
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_delta_reference_forward_and_gradients(dtype: torch.dtype) -> None:
    reference = _reference().to(dtype=dtype)
    native = _native().to(dtype=dtype)
    native.load_state_dict(reference.state_dict(), strict=True)
    input_ref = torch.randn(1, 5, 8, dtype=dtype, requires_grad=True)
    input_native = input_ref.detach().clone().requires_grad_()
    with portable_half_precision():
        expected_output = cast(object, reference(input_ref))
        expected = hf_tensor(expected_output)
        actual = native(input_native)
        assert torch.equal(actual, expected)
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
    assert input_native.grad is not None
    assert input_ref.grad is not None
    assert torch.equal(input_native.grad, input_ref.grad)
    for name, parameter in native.named_parameters():
        expected_parameter = reference.get_parameter(_HF_NAMES.get(name, name))
        assert parameter.grad is not None
        assert expected_parameter.grad is not None
        assert torch.equal(parameter.grad, expected_parameter.grad), name


def test_delta_cache_continuation_and_serialization(tmp_path: Path) -> None:
    reference = _reference()
    native = _native()
    native.load_state_dict(reference.state_dict(), strict=True)
    config = Qwen3_5TextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    reference_cache = DynamicCache(config=config)
    cache = native.alloc_kv_cache(batch=1, max_seq=9)
    assert not cache
    reference_layer = reference_cache.layers[0]
    assert isinstance(reference_layer, LinearAttentionCacheLayerMixin)
    for length in (2, 1, 3, 1):
        x = torch.randn(1, length, 8)
        expected_output = cast(object, reference(x, cache_params=reference_cache))
        expected = hf_tensor(expected_output)
        actual, returned = native.forward_cached(x, cache=cache)
        assert returned is cache
        assert torch.equal(actual, expected)
        assert reference_layer.conv_states[0] is not None
        assert reference_layer.recurrent_states[0] is not None
        assert torch.equal(cache["conv_state"], reference_layer.conv_states[0])
        assert torch.equal(
            cache["recurrent_state"],
            reference_layer.recurrent_states[0],
        )
    path = tmp_path / "delta.pt"
    torch.save(cache, path)
    reloaded = cast(object, torch.load(path, weights_only=True))
    assert _is_tensor_cache(reloaded)
    x = torch.randn(1, 1, 8)
    expected, _ = native.forward_cached(x, cache=cache)
    actual, _ = native.forward_cached(x, cache=reloaded)
    assert torch.equal(actual, expected)


def test_delta_masking_and_arbitrary_batch_shape() -> None:
    reference = _reference()
    native = _native()
    native.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(2, 3, 4, 8)
    mask = torch.tensor([[[0, 0, 1, 1]]]).expand(2, 3, 4)
    expected_output = cast(
        object,
        reference(x.reshape(6, 4, 8), attention_mask=mask.reshape(6, 4)),
    )
    expected = hf_tensor(expected_output)
    actual = native(x, attention_mask=mask)
    assert torch.equal(actual.reshape(6, 4, 8), expected)


@pytest.mark.parametrize("attention_mask", [object(), "padding"])
def test_delta_rejects_non_tensor_attention_mask(attention_mask: object) -> None:
    native = _native()
    with pytest.raises(TypeError, match="attention_mask must be a Tensor or None"):
        native(torch.randn(1, 1, 8), attention_mask=attention_mask)


@pytest.mark.parametrize("state_name", ["conv_state", "recurrent_state"])
def test_delta_rejects_partial_cache_without_mutation(state_name: str) -> None:
    native = _native()
    cache = {state_name: torch.randn(1, 2, 4, 3)}
    original = cache[state_name].clone()
    with pytest.raises(TypeError, match="cache must be empty or contain"):
        native.forward_cached(torch.randn(1, 1, 8), cache=cache)
    assert cache.keys() == {state_name}
    assert torch.equal(cache[state_name], original)


@pytest.mark.parametrize(
    ("state_name", "axis", "dtype_or_layout", "device"),
    [
        ("conv_state", 0, None, None),
        ("conv_state", 1, None, None),
        ("conv_state", 2, None, None),
        ("recurrent_state", 0, None, None),
        ("recurrent_state", 1, None, None),
        ("recurrent_state", 2, None, None),
        ("recurrent_state", 3, None, None),
        ("conv_state", None, torch.bfloat16, None),
        ("recurrent_state", None, torch.bfloat16, None),
        ("conv_state", None, None, torch.device("meta")),
        ("recurrent_state", None, None, torch.device("meta")),
        ("conv_state", None, torch.sparse_coo, None),
        ("recurrent_state", None, torch.sparse_coo, None),
    ],
    ids=(
        "conv-batch",
        "conv-channels",
        "conv-kernel",
        "recurrent-batch",
        "recurrent-heads",
        "recurrent-key-width",
        "recurrent-value-width",
        "conv-dtype",
        "recurrent-dtype",
        "conv-device",
        "recurrent-device",
        "conv-layout",
        "recurrent-layout",
    ),
)
def test_delta_rejects_incompatible_cache_state_without_mutation(
    state_name: str,
    axis: int | None,
    dtype_or_layout: torch.dtype | torch.layout | None,
    device: torch.device | None,
) -> None:
    native = _native()
    cache = native.alloc_kv_cache(batch=2, max_seq=2)
    native.forward_cached(torch.randn(2, 1, 8), cache=cache)
    state = cache[state_name]
    if axis is not None:
        cache[state_name] = state.narrow(axis, start=0, length=1).clone()
    elif dtype_or_layout is torch.sparse_coo:
        cache[state_name] = state.to_sparse()
    elif dtype_or_layout is not None:
        assert isinstance(dtype_or_layout, torch.dtype)
        cache[state_name] = state.to(dtype=dtype_or_layout)
    else:
        assert device is not None
        cache[state_name] = state.to(device=device)
    original = dict(cache)
    original_values = {
        name: tensor.clone()
        for name, tensor in cache.items()
        if tensor.device.type != "meta"
    }
    with pytest.raises(ValueError, match="cache state is incompatible with input"):
        native.forward_cached(torch.randn(2, 1, 8), cache=cache)
    assert cache.keys() == original.keys()
    for name, tensor in cache.items():
        assert tensor is original[name]
        if name in original_values:
            assert torch.equal(
                tensor.to_dense(),
                other=original_values[name].to_dense(),
            )


@pytest.mark.parametrize("autocast", [False, True])
def test_delta_cache_matches_projection_and_recurrence_dtypes(autocast: bool) -> None:
    native = _native()
    cache = native.alloc_kv_cache(batch=1, max_seq=2)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        native.forward_cached(torch.randn(1, 1, 8), cache=cache)
        native.forward_cached(torch.randn(1, 1, 8), cache=cache)
    expected_conv_dtype = torch.bfloat16 if autocast else torch.float32
    assert cache["conv_state"].dtype == expected_conv_dtype
    assert cache["recurrent_state"].dtype == torch.float32


@pytest.mark.parametrize(
    "eps",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_delta_norm_rejects_nonpositive_or_nonfinite_eps(eps: float) -> None:
    config = Qwen35RMSNormGated.Config()
    config.channels_in = 3
    config.eps = eps
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        config.make()


def test_delta_norm_slot_defaults_to_gated_transform_and_accepts_ordinary_norm() -> (
    None
):
    """The injected norm replaces the complete post-delta transform."""
    x = torch.ones(1, 3)
    gate = torch.ones_like(x)
    gated_config = Qwen35RMSNormGated.Config()
    gated_config.channels_in = x.shape[-1]
    gated = gated_config.make()
    with pytest.raises(TypeError, match="gate"):
        inspect.signature(gated.forward).bind(x)
    assert not torch.equal(gated(x, gate=torch.zeros_like(gate)), gated(x, gate=gate))

    ordinary_config = Qwen35GatedDeltaNet.Config()
    ordinary_config.channels_in = 8
    ordinary_config.num_heads_k = 1
    ordinary_config.num_heads_v = 2
    ordinary_config.channels_k_head = 4
    ordinary_config.channels_v_head = 3
    ordinary_config.norm = CenteredRMSNorm.Config()
    ordinary = ordinary_config.make()
    assert isinstance(ordinary.norm, CenteredRMSNorm)
    assert torch.equal(ordinary.norm(x), ordinary.norm(x, gate=gate))


def test_delta_ordinary_norm_omits_gate_projection() -> None:
    gated_config = Qwen35GatedDeltaNet.Config()
    gated_config.channels_in = 8
    gated_config.num_heads_k = 1
    gated_config.num_heads_v = 2
    gated_config.channels_k_head = 4
    gated_config.channels_v_head = 3
    ordinary_config = gated_config.copy_tree()
    ordinary_config.norm = CenteredRMSNorm.Config()
    gated = gated_config.make()
    ordinary = ordinary_config.make()

    assert gated.proj_z is not None
    assert ordinary.proj_z is None
    assert "proj_z.weight" in gated.state_dict()
    assert "proj_z.weight" not in ordinary.state_dict()
    projection_params = 8 * 2 * 3
    assert (
        sum(parameter.numel() for parameter in gated.parameters())
        - sum(parameter.numel() for parameter in ordinary.parameters())
        == projection_params
    )
    gated_cost = (
        gated_config.copy_tree()
        .finalize()
        .cost(
            seq_len=2,
            batch_size=1,
            dtype=None,
        )
    )
    ordinary_cost = (
        ordinary_config.copy_tree()
        .finalize()
        .cost(
            seq_len=2,
            batch_size=1,
            dtype=None,
        )
    )
    projection_products = 2 * projection_params
    assert gated_cost.params - ordinary_cost.params == projection_params
    assert (
        gated_cost["flops", "primal", "matmul"].sum()
        - ordinary_cost["flops", "primal", "matmul"].sum()
        == 2 * projection_products
    )
    assert (
        gated_cost["flops", "adjoint", "matmul"].sum()
        - ordinary_cost["flops", "adjoint", "matmul"].sum()
        == 4 * projection_products
    )
    assert ordinary(torch.randn(1, 2, 8)).shape == (1, 2, 8)


def test_delta_norm_cost_is_an_affine_rms_norm_and_a_silu_gate() -> None:
    """Torch counts no matmul; the per-channel scale is the only parameter.

    Per row of width ``w``: square, mean, epsilon and rsqrt, scale, the
    learned scale, SiLU on the gate, and the gating product are ``9w + 3``
    scalar operations around one ``w - 1`` sum. The adjoint pulls back
    through the affine norm (``6w + 4``), the product (``2w``), and SiLU's
    saved-sigmoid derivative (``5w``); the scale's gradient sums over rows.
    """
    config = Qwen35RMSNormGated.Config()
    config.channels_in = 8
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: (torch.randn(3, 8, requires_grad=True), torch.randn(3, 8)),
        seq_len=3,
        batch_size=1,
        dtype=None,
        run=lambda module, inputs: cast(Qwen35RMSNormGated, module)(
            inputs[0],
            gate=inputs[1],
        ),
    )
    assert model_cost.params == 8
    assert model_cost["flops", "matmul"].sum() == 0
    assert model_cost["flops", "primal", "elementwise"].sum() == 3 * (9 * 8 + 3)
    assert model_cost["flops", "adjoint", "elementwise"].sum() == 3 * (13 * 8 + 4)
    assert model_cost["flops", "primal", "reduction"].sum() == 3 * (8 - 1)
    assert model_cost["flops", "adjoint", "reduction"].sum() == (
        3 * (8 - 1) + 8 * (3 - 1)
    )


def test_delta_norm_traffic_counts_row_reductions() -> None:
    config = Qwen35RMSNormGated.Config()
    config.channels_in = 8
    small = config.cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    large = config.cost(seq_len=4, batch_size=1, dtype=None)
    assert small["bytes", "primal", "reduction"].sum() == 2 * (4 * 8 + 4)
    assert small["bytes", "adjoint", "reduction"].sum() == 2 * (4 * 8 + 4 + 4 * 8 + 8)
    assert (
        large["bytes", torch.float32].sum() == small["bytes", torch.bfloat16].sum() * 2
    )
    norm = RMSNorm.Config(8)
    norm.elementwise_affine = True
    base = norm.cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    assert (
        small["bytes", "primal", "elementwise"].sum()
        - base["bytes", "primal", "elementwise"].sum()
        == 2 * 5 * 8 * 4
    )
    assert (
        small["bytes", "adjoint", "elementwise"].sum()
        - base["bytes", "adjoint", "elementwise"].sum()
        == 2 * 9 * 8 * 4
    )


def test_qwen_delta_cost_does_not_repeat_the_norm_owned_output_gate() -> None:
    config = Qwen35GatedDeltaNet.Config()
    config.channels_in = 8
    config.num_heads_k = 1
    config.num_heads_v = 2
    config.channels_k_head = 4
    config.channels_v_head = 3
    parent = GatedDeltaNet.Config().update(config, skip_missing=True)
    reference = (
        parent.copy_tree()
        .finalize()
        .cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    )
    actual = (
        config.copy_tree()
        .finalize()
        .cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    )
    assert (
        reference["flops", "primal", "elementwise"].sum()
        - actual["flops", "primal", "elementwise"].sum()
        == 6 * 6 * 4
    )
    assert (
        reference["flops", "adjoint", "elementwise"].sum()
        - actual["flops", "adjoint", "elementwise"].sum()
        == 7 * 6 * 4
    )
    assert reference.params == actual.params
    assert reference["flops", "matmul"].sum() == actual["flops", "matmul"].sum()


def _is_tensor_cache(value: object) -> TypeGuard[dict[str, torch.Tensor]]:
    """Return whether a restored delta cache has native tensor entries."""
    if not isinstance(value, dict):
        return False
    cache = cast(dict[object, object], value)
    return all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in cache.items()
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
