"""Tests for the KV-cache decode throughput harness."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, override

import logging

from torch import Tensor, nn

import pytest
import torch

from priml.baselines.kvcache.decode import (
    DecodeBenchmark,
    DecodeReport,
    _synchronize,
    greedy_decode,
)
from priml.cost import Cost, cost
from priml.model.attention.attention import Attention
from priml.model.attention.kvcache import KVCache, alloc_layer_cache
from priml.model.attention.rope import RoPE
from priml.model.embedding import Embedding
from priml.model.generate import generate
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.golden import joined


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.model.attention.kvcache import HasModules
    from priml.model.custom_types import DepthIndex


_CWD: Final = Path(__file__).resolve().parent
VOCAB: Final = 7
WIDTH: Final = 12
NUM_HEADS: Final = 6
CHANNELS_HEAD: Final = WIDTH // NUM_HEADS
NUM_HEADS_KV: Final = 3
NUM_LAYERS: Final = 2
PROMPT_LEN: Final = 5
NUM_DECODE_STEPS: Final = 3
# Byte-level prompts; every byte is a token below VOCAB.
PROMPTS: Final = (
    "\x01\x02\x03\x04\x05\x06",
    "\x06\x05\x04\x03\x02\x01",
    "\x02\x04\x06\x01\x03\x05",
    "\x03\x01\x04\x01\x05\x02",
)


def test_kv_bytes_equal_the_caches_the_model_allocates() -> None:
    bench = _tiny().make()
    report = bench.measure()

    caches = alloc_layer_cache(
        bench.model,
        batch=len(PROMPTS),
        max_seq=PROMPT_LEN + NUM_DECODE_STEPS,
    )
    allocated = 0
    for state in caches.values():
        assert isinstance(state, KVCache)
        allocated += state.k.nbytes + state.v.nbytes
    itemsize = torch.float32.itemsize
    assert report.kv_bytes_per_token == (
        NUM_LAYERS * 2 * NUM_HEADS_KV * CHANNELS_HEAD * itemsize
    )
    assert (
        report.kv_bytes_per_token
        == (
            cost(bench.config.model, seq_len=1, batch_size=1, dtype=torch.float32)
        ).bytes_state
    )
    assert report.kv_bytes == allocated


def test_tokens_are_greedy_generation_of_the_frozen_prompts() -> None:
    bench = _tiny().make()
    report = bench.measure()

    expected = generate(
        bench.model,
        _prompt(),
        max_new_tokens=NUM_DECODE_STEPS + 1,
        temperature=0.0,
        max_seq_len=PROMPT_LEN + NUM_DECODE_STEPS + 1,
    )
    assert torch.equal(report.tokens, expected)


def test_an_injected_cache_changes_the_cache_dtype_alone() -> None:
    cfg = _tiny()
    full = cfg.make().measure()
    cfg.allocate_cache = _half_caches
    half = cfg.make().measure()

    assert half.kv_bytes * 2 == full.kv_bytes
    assert half.kv_bytes_per_token * 2 == full.kv_bytes_per_token
    assert half.tokens.shape == full.tokens.shape


def test_the_seed_fixes_the_weights() -> None:
    cfg = _tiny()
    first = cfg.make().measure()
    again = cfg.make().measure()
    cfg.seed = 1
    other = cfg.make().measure()

    assert (first.seed, other.seed) == (0, 1)
    assert torch.equal(first.tokens, again.tokens)
    assert not torch.equal(first.tokens, other.tokens)


def test_every_decode_step_and_prefill_is_timed_once_per_repeat() -> None:
    cfg = _tiny()
    cfg.num_warmup = 2
    cfg.num_repeats = 3
    report = cfg.make().measure()

    assert len(report.prefill_sec) == cfg.num_repeats
    assert len(report.step_sec) == cfg.num_repeats * NUM_DECODE_STEPS
    assert all(t > 0 for t in (*report.prefill_sec, *report.step_sec))
    assert report.batch_size == len(PROMPTS)


def test_decode_cost_prices_each_step_at_its_context() -> None:
    bench = _tiny().make()
    report = bench.measure()

    expected = sum(
        (
            cost(
                bench.config.model,
                seq_len=1,
                batch_size=len(PROMPTS),
                dtype=torch.float32,
                context_len=PROMPT_LEN + step,
            )
            for step in range(NUM_DECODE_STEPS)
        ),
        Cost(),
    ).only("primal")
    assert report.decode_cost == expected
    assert report.decode_cost["adjoint"].sum() == 0


def test_report_rates_count_decode_steps_only() -> None:
    # Three repeats of 200 steps; nearest-rank p99 of 600 is the 594th smallest.
    step_sec = [index / 1024 for index in reversed(range(600))]
    report = DecodeReport(
        batch_size=5,
        prefill_sec=[8.0, 2.0, 4.0],
        step_sec=step_sec,
        kv_bytes_per_token=2,
        kv_bytes=6,
        decode_cost=Cost(cells={("bytes", "primal", "matmul", torch.float32): 7}),
        tokens=torch.zeros(5, 2, dtype=torch.long),
        seed=0,
    )

    assert report.tokens_per_sec == 5 * 600 / sum(step_sec)
    assert report.step_ms_p99 == 593 / 1024 * 1000
    assert report.step_ms_max == 599 / 1024 * 1000
    assert report.prefill_ms_median == 4.0 * 1000
    assert report.decode_bytes_per_sec == 3 * 7 / sum(step_sec)


@pytest.mark.parametrize(
    ("unmeasurable", "match"),
    [
        (
            partial(DecodeBenchmark.Config.update, num_decode_steps=0),
            "num_decode_steps",
        ),
        (partial(DecodeBenchmark.Config.update, num_repeats=0), "num_repeats"),
        (partial(DecodeBenchmark.Config.update, num_warmup=-1), "num_warmup"),
        (
            partial(DecodeBenchmark.Config.update, prompt_len=7),
            "Prompt 0 has 6 bytes, shorter than prompt_len=7",
        ),
        (partial(DecodeBenchmark.Config.update, prompts=[]), "prompts"),
        (partial(DecodeBenchmark.Config.update, prompt_len=0), "prompt_len"),
        (partial(DecodeBenchmark.Config.update, prompt_len=-1), "prompt_len"),
    ],
)
def test_rejects_an_unmeasurable_config(
    unmeasurable: Callable[[DecodeBenchmark.Config], DecodeBenchmark.Config],
    match: str,
) -> None:
    cfg = unmeasurable(_tiny())

    with pytest.raises(ValueError, match=match):
        cfg.make()


def test_synchronize_waits_on_an_accelerator_and_skips_the_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synchronized: list[torch.device] = []
    monkeypatch.setattr(torch.accelerator, "synchronize", synchronized.append)

    _synchronize(torch.device("cpu"))
    _synchronize(torch.device("cuda", 1))

    assert synchronized == [torch.device("cuda", 1)]


def test_run_logs_the_rates(caplog: pytest.LogCaptureFixture) -> None:
    cfg = _tiny()
    cfg.roofline_device = "rtx5090"

    with caplog.at_level(logging.INFO, logger="priml.baselines.kvcache.decode"):
        cfg.make().run("--passthrough")

    messages = [record.message for record in caplog.records]
    assert any(m.startswith("decode tokens/sec: ") for m in messages)
    assert any(m.startswith("decode step ms: p99 ") for m in messages)
    assert any(m.startswith("kv cache bytes: ") for m in messages)
    assert any(m.startswith("rtx5090 roofline utilization:") for m in messages)


def test_greedy_decode_bfb() -> None:
    cfg = _tiny()
    # One layer: replay cost is per op under host_agnostic_numerics, and a
    # second layer repeats the first's code path.
    cfg.model.num_layers = 1
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="decode",
        build_module=cfg.model.make,
        build_input=_prompt,
        run=_decode_tokens,
    )


def _decode_tokens(model: nn.Module, prompt: Tensor) -> Tensor:
    """Decode, then return the filled KV cache; it implies every fed token."""
    assert isinstance(model, Transformer)
    cache = alloc_layer_cache(
        model,
        batch=len(PROMPTS),
        max_seq=PROMPT_LEN + NUM_DECODE_STEPS,
    )
    greedy_decode(model, prompt, num_steps=NUM_DECODE_STEPS, cache=cache)
    states = [state for state in cache.values() if isinstance(state, KVCache)]
    return joined(t for state in states for t in (state.k, state.v))


def _tiny() -> DecodeBenchmark.Config:
    cfg = DecodeBenchmark.Config()
    cfg.prompts = list(PROMPTS)
    cfg.prompt_len = PROMPT_LEN
    cfg.num_decode_steps = NUM_DECODE_STEPS
    cfg.num_warmup = 0
    cfg.num_repeats = 1
    cfg.dtype = torch.float32
    cfg.seed = 0
    cfg.model.channels_in = WIDTH
    cfg.model.channels_out = VOCAB
    assert isinstance(cfg.model.proj_in, Embedding.Config)
    cfg.model.proj_in.channels_in = VOCAB
    cfg.model.num_layers = NUM_LAYERS
    block = cfg.model.block
    assert isinstance(block, TransformerBlock.Config)
    attn = block.attn
    assert isinstance(attn, Attention.Config)
    attn.num_heads = NUM_HEADS
    attn.channels_head = CHANNELS_HEAD
    attn.num_heads_kv = NUM_HEADS_KV
    attn.rope = RoPE.Config(channels_head=CHANNELS_HEAD)
    ffn = block.ffn
    assert isinstance(ffn, SwiGLU.Config)
    ffn.round_to = 8
    return cfg


def _prompt() -> Tensor:
    return torch.tensor([list(p.encode()[:PROMPT_LEN]) for p in PROMPTS])


class _HalfKVCache(KVCache):
    """Stores keys and values in float16; attends in the model's dtype."""

    @override
    def update(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        k_all, v_all = super().update(k, v)
        return k_all.to(k.dtype), v_all.to(v.dtype)


def _half_caches(
    model: HasModules,
    *,
    batch: int,
    max_seq: int,
    device: torch.device,
) -> dict[DepthIndex, object]:
    caches: dict[DepthIndex, object] = {}
    for key, state in alloc_layer_cache(
        model,
        batch=batch,
        max_seq=max_seq,
        device=device,
        dtype=torch.float16,
    ).items():
        assert isinstance(state, KVCache)
        caches[key] = _HalfKVCache(state.k, state.v, length=0)
    return caches


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
