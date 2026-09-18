"""Tests for model channel-attribute Protocols and propagation helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

import ast

from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.self_attention import SelfAttention
from priml.model.custom_types import (
    AttentionKernel,
    CachedAttention,
    ChannelsIn,
    ChannelsOut,
    HasForwardCached,
    HasResetParameters,
    LatentAttentionKernel,
    LookupTable,
    RotaryFactors,
    TensorModule,
    flatten_depth_index,
    has_forward_cached,
    has_weight,
    infer_same_width,
    is_cached_attention,
    propagate_attr,
)
from priml.model.moe import SoftmaxRouter
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.golden import assert_text_golden


_CWD: Final = Path(__file__).resolve().parent


def test_custom_types_public_contract(request: pytest.FixtureRequest) -> None:
    rendered = "\n".join(
        [
            "flatten_depth_index:",
            f"  unspecified: {flatten_depth_index(())}",
            f"  one level ((3, 12),): {flatten_depth_index(((3, 12),))}",
            f"  nested ((1, 4), (3, 12)): {flatten_depth_index(((1, 4), (3, 12)))}",
        ],
    )
    assert_text_golden(
        request,
        test_file=__file__,
        name="custom_types",
        rendered=rendered,
    )


def test_custom_types_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="custom_types",
        build_module=_FlattenDepthIndex,
        build_input=lambda: torch.tensor(
            [
                [[0, 2], [1, 3]],
                [[1, 2], [2, 3]],
            ],
        ),
        seed=0,
    )


@pytest.mark.parametrize("weighted", [False, True])
def test_has_weight_finds_registered_parameters(weighted: bool) -> None:
    module = nn.Embedding(8, 4) if weighted else RMSNorm.Config(4).make()
    assert has_weight(module) is weighted
    assert not has_weight(None)


def test_head_capabilities_are_direct_attributes() -> None:
    @dataclass(slots=True, kw_only=True)
    class Attention:
        num_heads: int
        channels_head: int

    attention = Attention(num_heads=4, channels_head=32)

    assert attention.num_heads * attention.channels_head == 128


def test_flatten_depth_index_uses_global_to_local_mixed_radix() -> None:
    assert flatten_depth_index(()) == -1
    assert flatten_depth_index(((3, 12),)) == 3
    assert flatten_depth_index(((1, 4), (3, 12))) == 15


@pytest.mark.parametrize("depth_index", [((-1, 4),), ((4, 4),), ((0, 0),)])
def test_flatten_depth_index_rejects_invalid_levels(
    depth_index: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(ValueError, match="depth_index"):
        flatten_depth_index(depth_index)


def test_propagate_settable_field():
    cfg = SwiGLU.Config()
    propagate_attr(cfg, "channels_in", 128, protocol=ChannelsIn)
    assert cfg.channels_in == 128


def test_propagate_width_preserving_field_is_mutable():
    cfg = SelfAttention.Config(channels_in=64)
    propagate_attr(cfg, "channels_out", 999)
    assert cfg.channels_out == 999
    with pytest.raises(ValueError, match="channels_in=64 must equal channels_out=999"):
        cfg.make()


def test_propagate_non_participant_skipped():
    """A child not implementing the gating Protocol opts out silently.

    ``Router`` is the stand-in because it genuinely emits no width: it returns
    ``(weights, indices, logits)``, so it declares no ``channels_out`` for a
    parent to push into. A norm was used here once and stopped being a
    non-participant the moment norms gained the derived property.
    """
    cfg = SoftmaxRouter.Config(channels_in=32)
    assert not isinstance(cfg, ChannelsOut)
    propagate_attr(cfg, "channels_out", 999, protocol=ChannelsOut)
    propagate_attr(cfg, "depth_index", ((0, 1),))
    assert not hasattr(cfg, "channels_out")


def test_propagate_norm_width_is_mutable_but_checked_at_construction():
    cfg = RMSNorm.Config(channels_in=32)
    propagate_attr(cfg, "channels_out", 999, protocol=ChannelsOut)
    assert cfg.channels_out == 999
    with pytest.raises(ValueError, match="channels_in=32 must equal channels_out=999"):
        cfg.make()


def test_propagate_missing_attr_raises():
    """A typo / missing attribute under a Protocol must raise."""

    @dataclass(slots=True, kw_only=True)
    class NoChannels:
        channels_in: int = -1

    with pytest.raises(AttributeError, match="channels_out"):
        propagate_attr(NoChannels(), "channels_out", 64, protocol=ChannelsIn)


@dataclass(slots=True, kw_only=True)
class _Widths:
    channels_in: int = -1
    channels_out: int = -1


@pytest.mark.parametrize(
    ("channels_in", "channels_out"),
    [(-1, 8), (8, -1), (8, 8)],
)
def test_infer_same_width_fills_the_missing_side(
    channels_in: int,
    channels_out: int,
) -> None:
    widths = _Widths(channels_in=channels_in, channels_out=channels_out)
    infer_same_width(widths)
    assert widths.channels_in == 8
    assert widths.channels_out == 8


def test_infer_same_width_rejects_two_different_widths() -> None:
    with pytest.raises(ValueError, match="channels_in=4 must equal channels_out=8"):
        infer_same_width(_Widths(channels_in=4, channels_out=8))


class _CachedStub:
    """Borrows every cache-protocol stub body, which must be inert."""

    forward_cached = HasForwardCached[object].forward_cached
    alloc_kv_cache = CachedAttention[object].alloc_kv_cache


class _KernelStub:
    __call__ = AttentionKernel.__call__
    reset_parameters = HasResetParameters.reset_parameters


class _LatentKernelStub:
    __call__ = LatentAttentionKernel.__call__


class _TensorModuleStub:
    __call__ = TensorModule.__call__
    reset_parameters = HasResetParameters.reset_parameters


class _RotaryStub:
    __call__ = RotaryFactors.__call__


class _LookupStub:
    weight = torch.zeros(1)
    __call__ = TensorModule.__call__
    reset_parameters = HasResetParameters.reset_parameters
    to = LookupTable.to


def test_cache_guards_name_the_erased_cache_type() -> None:
    cached = _CachedStub()
    assert has_forward_cached(cached)
    assert is_cached_attention(cached)
    assert not has_forward_cached(object())
    assert not is_cached_attention(object())
    assert cached.forward_cached(torch.zeros(1), cache=object()) is None
    assert cached.alloc_kv_cache(batch=1, max_seq=2) is None


def test_protocol_stub_bodies_are_inert() -> None:
    """A stub hides no behavior: every default body is a no-op returning None."""
    x = torch.zeros(1)
    kernel: AttentionKernel[...] = _KernelStub()
    assert isinstance(kernel, AttentionKernel)
    assert kernel(x, x, x) is None
    latent: LatentAttentionKernel[...] = _LatentKernelStub()
    assert isinstance(latent, LatentAttentionKernel)
    assert latent(x, x, x, x) is None
    module: TensorModule = _TensorModuleStub()
    assert isinstance(module, HasResetParameters)
    assert module(x) is None
    assert module.reset_parameters() is None
    rotary: RotaryFactors = _RotaryStub()
    assert isinstance(rotary, RotaryFactors)
    assert rotary(x) is None
    assert has_weight(_LookupStub())
    assert _LookupStub().to(dtype=torch.float32) is None


@pytest.mark.compute_large_fixture
def test_channel_config_fields_are_uniform() -> None:
    root = _CWD.parent
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.name.endswith("_test.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name != "Config":
                continue
            fields = [
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign)
                and isinstance(child.target, ast.Name)
            ]
            properties = {
                child.name
                for child in node.body
                if isinstance(child, ast.FunctionDef)
                and any(
                    isinstance(decorator, ast.Name) and decorator.id == "property"
                    for decorator in child.decorator_list
                )
            }
            channel_properties = properties & {"channels_in", "channels_out"}
            if channel_properties:
                violations.append(
                    f"{path.relative_to(root)}:{node.lineno}: channel properties "
                    f"{sorted(channel_properties)}",
                )
            if "channels_in" in fields and "channels_out" in fields:
                if fields[:3] != ["channels_in", "channels_out", "_"]:
                    violations.append(
                        f"{path.relative_to(root)}:{node.lineno}: fields begin {fields[:3]}",
                    )
                channel_fields = {
                    child.target.id: child
                    for child in node.body
                    if isinstance(child, ast.AnnAssign)
                    and isinstance(child.target, ast.Name)
                    and child.target.id in {"channels_in", "channels_out"}
                }
                for name, field in channel_fields.items():
                    if field.value is None or ast.unparse(field.value) != "-1":
                        violations.append(
                            f"{path.relative_to(root)}:{field.lineno}: "
                            f"{name} default is not -1",
                        )
    assert not violations, "\n".join(violations)


class _FlattenDepthIndex(nn.Module):
    @override
    def forward(self, depth_indices: Tensor) -> Tensor:
        return torch.tensor(
            [
                flatten_depth_index(
                    tuple(
                        (int(levels[i, 0].item()), int(levels[i, 1].item()))
                        for i in range(levels.shape[0])
                    ),
                )
                for levels in depth_indices
            ],
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
