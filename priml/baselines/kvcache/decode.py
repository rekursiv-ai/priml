"""KV-cache decode throughput: tokens per second as the cache grows.

:class:`DecodeBenchmark` prefills a batch of fixed prompts, then times each
greedy decode step against a cache that grows to the full context. A decode
step at long context is bound by reading the cache, so the cache's storage is
what a KV-cache experiment changes; ``allocate_cache`` is that slot. The model,
its dtype, and the prompts stay put, so a fork moves cache bytes alone.

Prefill and decode are timed apart. Prefill is one dense forward over the
prompt; folding it into tokens per second credits a cache scheme with work
the cache never sees. Each step's time is read with a device synchronization
on both sides, so asynchronous kernel launches cannot hide in the next step.

The report keeps every step's time. The headline is decode tokens per
second; the worst-case view is the p99 step, since a cache scheme that
dequantizes in bursts stalls some steps and not others.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import logging
import math
import statistics
import time

from configgle import Fig
from torch import Tensor

import torch

from priml.cost import Cost, cost, utilization
from priml.math.seed import set_seed_local
from priml.model.attention.attention import Attention
from priml.model.attention.kvcache import KVCache, alloc_layer_cache
from priml.model.attention.rope import RoPE
from priml.model.custom_types import DepthIndex
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer


if TYPE_CHECKING:
    from priml.model.custom_types import LayerCache


logger = logging.getLogger(__name__)


__all__ = [
    "CacheAllocator",
    "DecodeBenchmark",
    "DecodeReport",
    "decoder",
    "greedy_decode",
]


type CacheAllocator = Callable[..., dict[DepthIndex, object]]
"""Builds an empty per-layer decode cache.

Called as ``allocate(model, batch=, max_seq=, device=)``; returns one empty
state per cached attention, keyed by depth index.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeReport:
    """What one :meth:`DecodeBenchmark.measure` call observed."""

    batch_size: int
    """Sequences decoded together: one per prompt."""

    prefill_sec: list[float]
    """Seconds to prefill the batch, one per timed repeat."""

    step_sec: list[float]
    """Seconds per decode step, every step of every timed repeat."""

    kv_bytes_per_token: int
    """Cache bytes one token of one sequence occupies, every layer."""

    kv_bytes: int
    """Cache bytes allocated for the batch at full context."""

    decode_cost: Cost
    """Analytical forward cost of one repeat's decode steps, each at its context."""

    tokens: Tensor
    """``(batch, prompt_len + num_decode_steps + 1)`` greedy tokens."""

    seed: int
    """Seed the weights were drawn under."""

    @property
    def tokens_per_sec(self) -> float:
        """Decode tokens per second over every timed step; prefill excluded."""
        return self.batch_size * len(self.step_sec) / sum(self.step_sec)

    @property
    def step_ms_p99(self) -> float:
        """Nearest-rank 99th-percentile decode step, in milliseconds."""
        ranked = sorted(self.step_sec)
        return ranked[math.ceil(0.99 * len(ranked)) - 1] * 1000

    @property
    def step_ms_max(self) -> float:
        """Slowest decode step, in milliseconds."""
        return max(self.step_sec) * 1000

    @property
    def prefill_ms_median(self) -> float:
        """Median prefill over the timed repeats, in milliseconds."""
        return statistics.median(self.prefill_sec) * 1000

    @property
    def decode_bytes_per_sec(self) -> float:
        """Analytical decode traffic over measured decode time."""
        repeats = len(self.prefill_sec)
        return repeats * self.decode_cost["bytes"].sum() / sum(self.step_sec)


def decoder() -> Transformer.Config:
    """Return a byte-level decoder-only transformer with grouped-query attention.

    Returns:
      config: 256-token vocabulary, width 512, 8 layers, 8 query heads sharing
        2 key/value heads, rotary positions, an RMS-normed linear head.

    """
    cfg = Transformer.Config()
    cfg.channels_in = 512
    cfg.channels_out = 256
    cfg.num_layers = 8
    cfg.proj_in = Embedding.Config()
    cfg.proj_in.channels_in = cfg.channels_out
    cfg.proj_out = Sequential.Config()
    cfg.proj_out.elements = [RMSNorm.Config(), Linear.Config()]
    block = cfg.block
    assert isinstance(block, TransformerBlock.Config)
    attn = block.attn
    assert isinstance(attn, Attention.Config)
    attn.channels_head = 64
    attn.num_heads_kv = 2
    attn.causal = True
    attn.rope = RoPE.Config(channels_head=attn.channels_head)
    return cfg


def greedy_decode(
    model: Transformer,
    prompt: Tensor,
    *,
    num_steps: int,
    cache: LayerCache,
) -> tuple[Tensor, list[float]]:
    """Prefill ``prompt``, then take ``num_steps`` timed greedy decode steps.

    Each step feeds the previous token and reads every cached position. The
    timer brackets the step with a device synchronization on both sides.

    Args:
      model: Decoder whose blocks read and update ``cache``.
      prompt: ``(batch, prompt_len)`` token ids, on the model's device.
      num_steps: Decode steps to take after the prefill.
      cache: Empty per-layer cache sized for ``prompt_len + num_steps``.

    Returns:
      tokens: ``(batch, prompt_len + num_steps + 1)``: the prompt, the token
        the prefill predicts, and one token per step.
      seconds: Prefill first, then one entry per decode step.

    """
    seconds: list[float] = []
    with torch.inference_mode():
        tokens = [prompt]
        for _ in range(num_steps + 1):
            _synchronize(prompt.device)
            started = time.perf_counter()
            logits = model(tokens[-1], cache=cache)
            tokens.append(logits[:, -1:].argmax(dim=-1))
            _synchronize(prompt.device)
            seconds.append(time.perf_counter() - started)
    return torch.cat(tokens, dim=-1), seconds


class DecodeBenchmark:
    """Time greedy decode over a growing KV cache."""

    class Config(Fig["DecodeBenchmark"]):
        model: Transformer.Config = field(default_factory=decoder)
        """The decoder; frozen across a KV-cache chain."""

        allocate_cache: CacheAllocator = alloc_layer_cache
        """Builds the empty per-layer cache: the slot a cache experiment fills.

        Every layer's state is a ``KVCache`` whose ``k`` and ``v`` hold all
        the bytes it stores, so a quantized cache packs its scales there too;
        ``kv_bytes`` counts nothing else."""

        prompts: list[str] = field(default_factory=list[str])
        """Prompts decoded as one batch; each is read as UTF-8 bytes."""

        prompt_len: int = 24
        """Bytes kept from each prompt, so every row starts at one context."""

        num_decode_steps: int = 8168
        """Timed decode steps; the cache ends at ``prompt_len + num_decode_steps``."""

        num_warmup: int = 1
        """Untimed full passes before the timed ones."""

        num_repeats: int = 3
        """Timed full passes; the report keeps every step of each."""

        dtype: torch.dtype = torch.bfloat16
        """Weight and activation dtype."""

        device: str = "cpu"
        """Device the model and cache live on."""

        seed: int | None = None
        """Weight seed; ``None`` draws one and logs it."""

        roofline_device: str = ""
        """A ``priml.cost.peak`` device name to report utilization against."""

    def __init__(self, config: Config) -> None:
        if config.num_decode_steps < 1:
            raise ValueError(
                f"num_decode_steps must be positive; got {config.num_decode_steps}.",
            )
        if config.num_repeats < 1:
            raise ValueError(f"num_repeats must be positive; got {config.num_repeats}.")
        if config.num_warmup < 0:
            raise ValueError(
                f"num_warmup must be non-negative; got {config.num_warmup}.",
            )
        if config.prompt_len < 1:
            raise ValueError(f"prompt_len must be positive; got {config.prompt_len}.")
        if not config.prompts:
            raise ValueError("prompts must name at least one prompt.")
        rows = [list(p.encode()) for p in config.prompts]
        for index, row in enumerate(rows):
            if len(row) < config.prompt_len:
                raise ValueError(
                    f"Prompt {index} has {len(row)} bytes, shorter than "
                    f"prompt_len={config.prompt_len}.",
                )
        self.config = config
        self.seed = set_seed_local(config.seed)
        self.model = config.model.make().to(device=config.device, dtype=config.dtype)
        self.model.eval()
        self.prompt = torch.tensor(
            [row[: config.prompt_len] for row in rows],
            device=config.device,
        )

    def measure(self) -> DecodeReport:
        """Run the warmup and timed passes.

        Returns:
          report: Every timed step, the cache's size, and the decoded tokens.

        """
        cfg = self.config
        max_seq = cfg.prompt_len + cfg.num_decode_steps
        batch = len(cfg.prompts)
        prefill_sec: list[float] = []
        step_sec: list[float] = []
        kv_bytes = 0
        tokens = self.prompt
        for repeat in range(cfg.num_warmup + cfg.num_repeats):
            cache = cfg.allocate_cache(
                self.model,
                batch=batch,
                max_seq=max_seq,
                device=self.prompt.device,
            )
            kv_bytes = _cache_bytes(cache)
            tokens, seconds = greedy_decode(
                self.model,
                self.prompt,
                num_steps=cfg.num_decode_steps,
                cache=cache,
            )
            if repeat >= cfg.num_warmup:
                prefill_sec.append(seconds[0])
                step_sec += seconds[1:]
        return DecodeReport(
            batch_size=batch,
            prefill_sec=prefill_sec,
            step_sec=step_sec,
            kv_bytes_per_token=kv_bytes // (batch * max_seq),
            kv_bytes=kv_bytes,
            decode_cost=sum(
                (
                    cost(
                        cfg.model,
                        seq_len=1,
                        batch_size=batch,
                        dtype=cfg.dtype,
                        context_len=cfg.prompt_len + step,
                    )
                    for step in range(cfg.num_decode_steps)
                ),
                Cost(),
            ).only("primal"),
            tokens=tokens.cpu(),
            seed=self.seed,
        )

    def run(self, *args: str) -> None:
        """Measure and log the decode rates.

        Args:
          *args: Ignored; accepted so launcher passthrough arguments never raise.

        """
        if args:
            logger.info("Ignoring launcher passthrough args: %r.", args)
        cfg = self.config
        report = self.measure()
        logger.info(
            "decode tokens/sec: %.1f over %d steps x %d repeats at batch %d.",
            report.tokens_per_sec,
            cfg.num_decode_steps,
            cfg.num_repeats,
            report.batch_size,
        )
        logger.info(
            "decode step ms: p99 %.3f, max %.3f; prefill ms median %.3f.",
            report.step_ms_p99,
            report.step_ms_max,
            report.prefill_ms_median,
        )
        logger.info(
            "kv cache bytes: %d at full context, %d per token; "
            "analytical decode traffic %.3g B/s.",
            report.kv_bytes,
            report.kv_bytes_per_token,
            report.decode_bytes_per_sec,
        )
        if cfg.roofline_device:
            logger.info(
                "%s roofline utilization:\n%r",
                cfg.roofline_device,
                utilization(
                    report.decode_cost.tile(cfg.num_repeats),
                    device=cfg.roofline_device,
                    duration_sec=sum(report.step_sec),
                ),
            )


def _synchronize(device: torch.device) -> None:
    """Wait for ``device``'s queued kernels; the CPU has none."""
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def _cache_bytes(cache: dict[DepthIndex, object]) -> int:
    """Sum the key and value storage every layer's cache holds."""
    total = 0
    for state in cache.values():
        assert isinstance(state, KVCache)
        total += state.k.nbytes + state.v.nbytes
    return total
