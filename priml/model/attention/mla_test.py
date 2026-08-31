"""Tests for ``priml.model.attention.mla``.

The Hugging Face formula-reference tests run offline from local tensors.

Regenerate the bit-for-bit golden after an intentional numeric change::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/attention/mla_test.py

Run regeneration through ``pytest``: the priml ``conftest.py`` sets
``MKL_CBWR`` and caps math threads before torch imports. Minting from a bare
``python`` process skips that setup and pins the golden to the mint host.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import functools
import tempfile

from configgle import Makeable, PartialConfig
from configgle.testing import assert_pprint_golden
from torch import Tensor, nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as f

import pytest
import torch

from priml import runtime
from priml.model.attention.kernel import SdpaFused, SdpaNaive
from priml.model.attention.mla import LatentAttention, MultiHeadLatentAttention
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE, RoPEMixed
from priml.model.custom_types import AttentionKernel
from priml.model.linear import Linear
from priml.testing.bfb import (
    assert_bfb_against_golden,
    bfb_devices,
    first_tensor,
    host_agnostic_numerics,
    move_to_device,
)
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.train.tensor_parallel import apply_tensor_parallel


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


@pytest.mark.parametrize("name", ["latent_attention", "multi_head_latent_attention"])
def test_mla_config_pprint(name: str) -> None:
    config, latent = _mla_config()
    assert_pprint_golden(
        test_file=__file__,
        name=name,
        config=latent if name == "latent_attention" else config,
    )


def _tiny(
    q_lora_rank: int | None = None,
    *,
    absorb: bool = True,
    kernel: Makeable[AttentionKernel] | None = None,
) -> MultiHeadLatentAttention:
    if kernel is None:
        kernel = SdpaNaive.Config()
    return MultiHeadLatentAttention.Config(
        channels_in=128,
        num_heads=4,
        channels_qk_nope_head=16,
        channels_qk_rope_head=8,
        channels_v_head=16,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=32,
        rope=RoPE.Config(
            channels_head=8, frequencies=HuggingFaceFrequencies.Config(base=50_000)
        ),
        attn_kernel=LatentAttention.Config(absorb=absorb, attn_kernel=kernel),
    ).make()


class _ResettableLatentKernel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


def test_mla_reset_parameters_resets_injected_children() -> None:
    module = _tiny()
    module.rope = RoPEMixed.Config(
        channels_head=8,
        num_heads=4,
        learnable=True,
    ).make()
    module.attn_kernel = _ResettableLatentKernel()
    for parameter in (*module.rope.parameters(), *module.attn_kernel.parameters()):
        nn.init.constant_(parameter, float("nan"))

    module.reset_parameters()

    assert all(
        torch.isfinite(parameter).all() for parameter in module.rope.parameters()
    )
    assert all(
        torch.isfinite(parameter).all() for parameter in module.attn_kernel.parameters()
    )


def test_projection_slot_keeps_caller_set_fields() -> None:
    """A deliberately-configured projection survives width propagation.

    ``_size_projections`` fills derived widths; overwriting the rest builds a
    model that differs from the configured one.
    """
    config = MultiHeadLatentAttention.Config(
        channels_in=128,
        num_heads=4,
        channels_qk_nope_head=16,
        channels_qk_rope_head=8,
        channels_v_head=16,
        kv_lora_rank=32,
        proj_out=Linear.Config(bias=True),
    ).finalize()

    assert isinstance(config.proj_out, Linear.Config)
    assert config.proj_out.bias is True
    assert config.proj_out.channels_out == 128


def test_forward_shape():
    m = _tiny()
    x = torch.randn(2, 6, 128)
    out = m(x)
    assert out.shape == (2, 6, 128)


def test_forward_with_q_lora():
    m = _tiny(q_lora_rank=64)
    x = torch.randn(1, 5, 128)
    out = m(x)
    assert out.shape == (1, 5, 128)
    # Sanity: q_proj is disabled, LoRA path is active.
    assert m.q_proj is None
    assert m.q_a_proj is not None
    assert m.q_a_layernorm is not None
    assert m.q_b_proj is not None


def test_explicit_zero_softmax_scale_is_honored() -> None:
    """``0.0`` is a value, not an absence.

    Truthiness defaulting silently substitutes the computed scale, so the
    model attends with a different temperature than the one configured.
    """
    module = MultiHeadLatentAttention.Config(
        channels_in=128,
        num_heads=4,
        channels_qk_nope_head=16,
        channels_qk_rope_head=8,
        channels_v_head=16,
        kv_lora_rank=32,
        softmax_scale=0.0,
    ).make()

    assert module.softmax_scale == 0.0


def test_prealloc_cache_decode():
    m = _tiny()
    cache = m.alloc_kv_cache(batch=2, max_seq=16)
    # Latent cache shapes: [B, 1, max_seq, feat].
    assert cache.k.shape == (2, 1, 16, 32)  # c_kv, kv_lora_rank=32
    assert cache.v.shape == (2, 1, 16, 8)  # k_pe, qk_rope=8
    prompt = torch.randn(2, 5, 128)
    out, cache = m.forward_cached(prompt, cache=cache)
    assert out.shape == (2, 5, 128)
    assert cache.length == 5
    for _ in range(3):
        step = torch.randn(2, 1, 128)
        out, cache = m.forward_cached(step, cache=cache)
        assert out.shape == (2, 1, 128)
    assert cache.length == 8


def test_decode_equivalent_to_full_reforward():
    """Cached incremental decode must match a from-scratch forward."""
    torch.manual_seed(0)
    m = _tiny()
    m.eval()
    prompt = torch.randn(1, 4, 128)
    steps = [torch.randn(1, 1, 128) for _ in range(3)]

    # Path A: full forward over [prompt + steps].
    full_input = torch.cat([prompt, *steps], dim=1)
    with torch.no_grad():
        full_out = m(full_input)

    # Path B: prefill + 3 decode steps via cache.
    cache = m.alloc_kv_cache(batch=1, max_seq=16)
    with torch.no_grad():
        _, cache = m.forward_cached(prompt, cache=cache)
        decode_outs: list[Tensor] = []
        for step in steps:
            out, cache = m.forward_cached(step, cache=cache)
            decode_outs.append(out)
    cached_tail = torch.cat(decode_outs, dim=1)
    assert torch.allclose(full_out[:, 4:], cached_tail, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("absorb", [False, True], ids=["reexpand", "absorb"])
@pytest.mark.parametrize("kernel", [SdpaFused.Config(), SdpaNaive.Config()])
@pytest.mark.parametrize("chunk_size", [1, 2])
def test_mla_cached_chunk_matches_full_causal_forward(
    absorb: bool,
    kernel: Makeable[AttentionKernel],
    chunk_size: int,
) -> None:
    torch.manual_seed(0)
    module = _tiny(absorb=absorb, kernel=kernel)
    module.eval()
    x = torch.randn(2, 4, 128)
    split = x.shape[1] - chunk_size
    with torch.no_grad():
        full = module(x)
        cache = module.alloc_kv_cache(batch=2, max_seq=8)
        _, cache = module.forward_cached(x[:, :split], cache=cache)
        chunk, _ = module.forward_cached(x[:, split:], cache=cache)
    torch.testing.assert_close(chunk, full[:, split:], atol=1e-5, rtol=1e-4)


def test_is_causal_false_overrides_causal_config() -> None:
    torch.manual_seed(0)
    module = _tiny()
    module.eval()
    x = torch.randn(2, 4, 128)
    cache = module.alloc_kv_cache(batch=2, max_seq=8)

    with torch.no_grad():
        _, cache = module.forward_cached(x[:, :2], cache=cache)
        overridden, _ = module.forward_cached(
            x[:, 2:],
            cache=cache,
            is_causal=False,
        )
        module.causal = False
        reference_cache = module.alloc_kv_cache(batch=2, max_seq=8)
        _, reference_cache = module.forward_cached(
            x[:, :2],
            cache=reference_cache,
        )
        configured, _ = module.forward_cached(x[:, 2:], cache=reference_cache)

    assert torch.equal(overridden, configured)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dropout", -0.1),
        ("dropout", 1.1),
    ],
)
def test_mla_rejects_invalid_geometry(field: str, value: float) -> None:
    """Only ``dropout`` is checked here: it is a probability this layer owns.

    A nonpositive width is torch's to reject when it builds the tensor
    (STYLE.md "Let the leaf complain").
    """
    config = MultiHeadLatentAttention.Config(
        channels_in=128,
        num_heads=4,
        kv_lora_rank=32,
    )
    setattr(config, field, value)

    with pytest.raises(ValueError, match=field):
        config.make()


def test_softmax_scale_override():
    m = MultiHeadLatentAttention.Config(
        channels_in=64,
        num_heads=2,
        channels_qk_nope_head=8,
        channels_qk_rope_head=8,
        channels_v_head=8,
        kv_lora_rank=16,
        softmax_scale=0.25,
    ).make()
    assert m.softmax_scale == 0.25


def test_mla_arbitrary_leading_dims():
    m = _tiny()
    x = torch.randn(2, 3, 5, 128)
    out = m(x)
    assert out.shape == (2, 3, 5, 128)


def test_mla_reset_parameters() -> None:
    module = _tiny(q_lora_rank=64)
    module.reset_parameters()
    assert module.tensor_parallel_style() is not None


def test_mla_config_reports_derived_boundary_geometry() -> None:
    config, _ = _mla_config()
    config.finalize()

    assert config.channels_out == config.channels_in
    assert config.channels_head == (
        config.channels_qk_nope_head + config.channels_qk_rope_head
    )


@pytest.mark.parametrize("inner", [SdpaFused.Config, SdpaNaive.Config])
def test_absorb_math_matches_the_reexpand_form_it_replaces(
    inner: Callable[[], Makeable[AttentionKernel]],
) -> None:
    """The equivalence ``mla.py:29-30`` claims, measured rather than asserted.

    Absorbing the latent projections into the query and output is only sound
    if it computes the same attention as applying them to the latent first --
    an associativity claim, and the reason MLA may cache 576 dims per token
    instead of 16384.

    Run against BOTH inner kernels: absorb hands the kernel a broadcast view
    where re-expand hands it a materialized tensor, and a kernel that copies
    its inputs must still agree with one that does not.

    Both paths run inside ``host_agnostic_numerics`` so the comparison is the
    MATH, not the host's float32 reduction order.
    """
    absorb, absorb_kernel = _mla_config()
    absorb_kernel.attn_kernel = inner()
    reexpand, reexpand_kernel = _mla_config()
    reexpand_kernel.absorb = False
    reexpand_kernel.attn_kernel = inner()

    with host_agnostic_numerics(), torch.no_grad():
        torch.manual_seed(0)
        x = torch.randn(2, 5, absorb.channels_in)
        torch.manual_seed(0)
        absorbed = absorb.copy_tree().make().to(torch.float32).eval()(x)
        torch.manual_seed(0)
        reference = reexpand.copy_tree().make().to(torch.float32).eval()(x)
    torch.testing.assert_close(absorbed, reference, rtol=0, atol=1e-6)


def test_sharding_refuses_a_fused_kernel_it_cannot_shard() -> None:
    """A fused inner kernel must be REFUSED, not die deep in the dispatcher.

    ``F.scaled_dot_product_attention`` dispatches to a flash kernel with no
    DTensor sharding strategy. ``SelfAttention`` has always refused it by name
    (``attention.py:339``); MLA could not, having no kernel to interrogate,
    so the same misconfiguration surfaced as a dispatcher stack trace.
    """
    config, kernel = _mla_config()
    kernel.attn_kernel = SdpaFused.Config()

    with pytest.raises(ValueError, match="DTensor-compatible"):
        config.copy_tree().make().assert_shardable_over(2)


def test_sharding_accepts_the_naive_kernel() -> None:
    """The DTensor-safe kernel passes the gate a fused one fails."""
    config, _ = _mla_config()
    config.copy_tree().make().assert_shardable_over(2)


def test_mla_forwards_the_open_message_bus_through_both_kernels() -> None:
    messages: list[object] = []

    def kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        message: object,
        **kwargs: object,
    ) -> Tensor:
        del q, k, kwargs
        messages.append(message)
        return v

    config, latent = _mla_config()
    latent.attn_kernel = PartialConfig(kernel)
    message = object()

    config.make()(torch.randn(1, 4, config.channels_in), message=message)

    assert messages == [message]


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_latent_attention_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="latent_attention",
        build_module=lambda: LatentAttention.Config().make().to(device),
        build_input=lambda: {
            "q_nope": torch.randn(1, 3, 2, 4),
            "q_pe": torch.randn(1, 3, 2, 2),
            "c_kv": torch.randn(1, 3, 5),
            "k_pe": torch.randn(1, 3, 2),
            "w_kr": torch.randn(2, 4, 5),
            "w_uv": torch.randn(2, 6, 5),
        },
        seed=0,
    )


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_mla_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="multi_head_latent_attention",
        build_module=lambda: (
            MultiHeadLatentAttention.Config(
                channels_in=16,
                num_heads=2,
                channels_qk_nope_head=8,
                channels_qk_rope_head=4,
                channels_v_head=8,
                kv_lora_rank=8,
                rope=RoPE.Config(
                    channels_head=4,
                ),
            )
            .make()
            .to(device)
        ),
        build_input=lambda: move_to_device(torch.randn(2, 4, 16), device),
        seed=0,
        run=lambda m, x: first_tensor(m(x)),
    )


@pytest.mark.parametrize("q_lora_rank", [None, 32])
def test_mla_matches_reference(q_lora_rank: int | None) -> None:
    torch.manual_seed(0)
    module = MultiHeadLatentAttention.Config(
        channels_in=128,
        num_heads=4,
        channels_qk_nope_head=16,
        channels_qk_rope_head=8,
        channels_v_head=16,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=24,
        rope=RoPE.Config(
            channels_head=8,
            frequencies=HuggingFaceFrequencies.Config(base=50_000),
        ),
    ).make()
    module.eval()
    x = torch.randn(2, 6, 128)
    with torch.no_grad():
        fused = module(x)
        reference = _reference_mla_forward(module, x)
    diff = (fused - reference).abs().max().item()
    assert torch.allclose(fused, reference, atol=5e-5, rtol=1e-4), (
        f"max abs diff: {diff:.3e}"
    )


def test_mla_decode_matches_reference() -> None:
    """Match cached decoding against a full-sequence formula reference."""
    torch.manual_seed(0)
    module = MultiHeadLatentAttention.Config(
        channels_in=64,
        num_heads=4,
        channels_qk_nope_head=8,
        channels_qk_rope_head=8,
        channels_v_head=8,
        kv_lora_rank=16,
        rope=RoPE.Config(
            channels_head=8,
            frequencies=HuggingFaceFrequencies.Config(base=50_000),
        ),
    ).make()
    module.eval()
    prompt = torch.randn(1, 4, 64)
    steps = [torch.randn(1, 1, 64) for _ in range(3)]
    full_input = torch.cat([prompt, *steps], dim=1)

    with torch.no_grad():
        reference_full = _reference_mla_forward(module, full_input)

        cache = module.alloc_kv_cache(batch=1, max_seq=16)
        _, cache = module.forward_cached(prompt, cache=cache)
        decoded: list[Tensor] = []
        for step in steps:
            out, cache = module.forward_cached(step, cache=cache)
            decoded.append(out)
    cached_tail = torch.cat(decoded, dim=1)
    diff = (reference_full[:, 4:] - cached_tail).abs().max().item()
    assert torch.allclose(
        reference_full[:, 4:],
        cached_tail,
        atol=5e-5,
        rtol=1e-4,
    ), f"max abs diff: {diff:.3e}"


@pytest.mark.compute_distributed
def test_mla_sharded_equals_dense_tp2(warm_pools: WarmPoolGetter) -> None:
    """#307: the custom MLA style makes sharded output equal dense at tp=2."""
    pool = warm_pools({"dp": 1, "tp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_mla_tp_worker, tmp))
        results = {
            path.name: path.read_text()
            for path in Path(tmp).iterdir()
            if path.is_file()
        }
    expected = {
        f"{case}_rank{rank}": "ok"
        for case in ("no_lora", "q_lora", "indivisible")
        for rank in (0, 1)
    }
    assert results == expected, results


def _reference_mla_forward(
    module: MultiHeadLatentAttention,
    x: Tensor,
) -> Tensor:
    """Compute MLA directly from the published Hugging Face formulas."""
    sequence_length = x.shape[-2]
    num_heads = module.num_heads
    channels_qk_nope = module.channels_qk_nope_head
    channels_qk_rope = module.channels_qk_rope_head
    channels_v = module.channels_v_head
    channels_qk = module.channels_qk_head

    if module.q_proj is not None:
        q = module.q_proj(x)
    else:
        assert module.q_a_proj is not None
        assert module.q_a_layernorm is not None
        assert module.q_b_proj is not None
        q = module.q_b_proj(module.q_a_layernorm(module.q_a_proj(x)))
    q = q.view(*q.shape[:-1], num_heads, channels_qk)
    q_nope, q_pe = q[..., :channels_qk_nope], q[..., channels_qk_nope:]

    compressed = module.kv_a_proj(x)
    c_kv_raw = compressed[..., : module.kv_lora_rank]
    k_pe_raw = compressed[..., module.kv_lora_rank :]
    c_kv = module.kv_a_layernorm(c_kv_raw)
    kv = module.kv_b_proj(c_kv).view(
        *x.shape[:-1],
        num_heads,
        channels_qk_nope + channels_v,
    )
    k_nope, v = kv[..., :channels_qk_nope], kv[..., channels_qk_nope:]
    k_pe = k_pe_raw.unsqueeze(-2)

    assert module.rope is not None
    positions = torch.arange(sequence_length, device=x.device)
    cos, sin = module.rope(positions)
    # DeepSeek-V3 and Kimi-K2 pair interleaved dimensions before rotation.
    q_pe, k_pe = RoPE.rotate(q_pe, k_pe, cos, sin, interleave=True)
    k_pe = k_pe.expand(*k_nope.shape[:-1], channels_qk_rope)

    q_full = torch.cat([q_nope, q_pe], dim=-1).movedim(-3, -2)
    k_full = torch.cat([k_nope, k_pe], dim=-1).movedim(-3, -2)
    v = v.movedim(-3, -2)
    out = f.scaled_dot_product_attention(
        q_full,
        k_full,
        v,
        is_causal=True,
        scale=module.softmax_scale,
    )
    return module.o_proj(out.movedim(-3, -2).flatten(-2))


def _tensor_parallel_mla(
    *,
    q_lora_rank: int | None = None,
) -> tuple[nn.Module, Tensor]:
    """Build a small MLA exercising head-parallel and latent-absorb paths."""
    module = MultiHeadLatentAttention.Config(
        channels_in=16,
        num_heads=2,
        channels_qk_nope_head=4,
        channels_qk_rope_head=4,
        channels_v_head=4,
        kv_lora_rank=8,
        q_lora_rank=q_lora_rank,
        rope=RoPE.Config(channels_head=4),
        shard="colwise",
    ).make()
    return module, torch.randn(1, 3, 16)


def _record_case(
    result_dir: Path,
    case: str,
    rank: int,
    q_lora_rank: int | None,
    mesh: DeviceMesh,
) -> None:
    """Assert one MLA variant's sharded forward equals dense; record outcome."""
    target = result_dir / f"{case}_rank{rank}"
    try:
        torch.manual_seed(0)
        model, x = _tensor_parallel_mla(q_lora_rank=q_lora_rank)
        dense = first_tensor(model(x))
        sharded = apply_tensor_parallel(model, mesh)
        # A sharded q-path guards against a replicated no-op passing trivially.
        q = sharded.q_proj if sharded.q_proj is not None else sharded.q_b_proj
        if not isinstance(q.weight, DTensor):
            target.write_text("FAIL:q-path-not-sharded")
            return
        out = first_tensor(sharded(x))
        full = out.full_tensor() if isinstance(out, DTensor) else out
        if torch.allclose(full, dense, rtol=1e-4, atol=1e-5):
            target.write_text("ok")
        else:
            target.write_text(f"FAIL:max={float((full - dense).abs().max()):.2e}")
    except Exception as error:  # noqa: BLE001 -- surface worker errors to parent
        target.write_text(f"FAIL:{error!r}")


def _record_indivisible_guard(
    result_dir: Path,
    rank: int,
    mesh: DeviceMesh,
) -> None:
    """Record whether tensor parallelism rejects indivisible head counts."""
    target = result_dir / f"indivisible_rank{rank}"
    try:
        torch.manual_seed(0)
        module = MultiHeadLatentAttention.Config(
            channels_in=16,
            num_heads=3,
            channels_qk_nope_head=4,
            channels_qk_rope_head=4,
            channels_v_head=4,
            kv_lora_rank=8,
            shard="colwise",
        ).make()
        try:
            apply_tensor_parallel(module, mesh)
        except ValueError as error:
            valid = "num_heads" in str(error) and "divis" in str(error).lower()
            target.write_text("ok" if valid else f"FAIL:{error!r}")
        else:
            target.write_text("FAIL:no-raise")
    except Exception as error:  # noqa: BLE001 -- surface worker errors to parent
        target.write_text(f"FAIL:{error!r}")


def _mla_tp_worker(result_dir_str: str, mesh: DeviceMesh) -> None:
    """Run MLA tensor-parallel cases and record each rank's outcomes."""
    result_dir = Path(result_dir_str)
    rank = mesh.get_rank()
    runtime._device_mesh = mesh
    try:
        _record_case(result_dir, "no_lora", rank, None, mesh)
        _record_case(result_dir, "q_lora", rank, 8, mesh)
        _record_indivisible_guard(result_dir, rank, mesh)
    finally:
        runtime._device_mesh = None


def _mla_config() -> tuple[MultiHeadLatentAttention.Config, LatentAttention.Config]:
    """A tiny MLA whose every width differs, so a swapped axis cannot pass.

    Returns the kernel config too: the slot is typed by the PROTOCOL, so
    reaching a concrete field through it needs narrowing, and handing back the
    value already narrowed keeps that out of every caller.

    Returns:
      config: The MLA config.
      kernel: Its latent attention kernel, narrowed to the concrete class.

    """
    config = MultiHeadLatentAttention.Config(
        channels_in=32,
        num_heads=4,
        channels_qk_nope_head=8,
        channels_qk_rope_head=4,
        channels_v_head=16,
        kv_lora_rank=12,
        bias=False,
        causal=True,
    )
    kernel = LatentAttention.Config()
    config.attn_kernel = kernel
    return config, kernel


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
