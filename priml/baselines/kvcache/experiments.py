"""KV-cache decode experiments.

``exp000`` times greedy decode of a byte-level GQA decoder over a cache that
grows to a 4096-token context, with the cache stored at the model's own
dtype. Every later experiment forks a named parent and changes ONE thing:
``allocate_cache``, the storage the cache uses.

    exp000  bf16 weights, bf16 KV cache, 4096-token context
      +-- exp001  int8 codes, one float32 scale per token, packed in k/v
      +-- exp002  e2m1 (fp4) codes, one float32 scale per token, packed in k/v
      +-- exp003  exp002's rows read by a fused Triton attention kernel

Launch::

    uv --quiet run --frozen python -m priml priml.baselines.kvcache.experiments.exp000

That command draws a fresh seed. For paired measurements at the reported
seed, including generated-token agreement, follow the adjacent ``README.md``.
The Results sections retain the author's original measurements. The scale-range
fixes have not been benchmarked on that A100; raw timing artifacts are not included.

Objective:
  Raise decode tokens per second, or shrink cache bytes at equal tokens per
  second, by changing how the cache stores keys and values.

Score:
  Decode tokens per second over every timed step, prefill excluded, with the
  p99 step beside it: a scheme that stalls some steps shows there first.

A fork may change:
  ``allocate_cache``: the cache's dtype, layout, quantization, and the
  arithmetic that reads it back -- including, for a fused read path, the
  attention kernel that consumes the cache's served rows.

A fork may not change:
  The model, its weights' dtype, the prompts, the context length, or the
  harness and its timing.
"""

from __future__ import annotations

from priml.baselines.kvcache.decode import DecodeBenchmark
from priml.baselines.kvcache.fp4 import fp4_layer_cache
from priml.baselines.kvcache.fp4_fused import (
    Fp4TritonAttention,
    fp4_fused_layer_cache,
)
from priml.baselines.kvcache.int8 import int8_layer_cache
from priml.model.attention.attention import Attention
from priml.model.transformer.block import TransformerBlock


def exp000() -> DecodeBenchmark.Config:
    """Time bf16 greedy decode to a 4096-token context on a CUDA device.

    The control every KV-cache experiment forks. Frozen: improvements belong
    in a fork, never in an edit here.

    Hypothesis:
      At a 4096-token context a decode step of this model reads more cache
      than weights, so decode tokens per second is bound by cache bytes, and
      a cache stored in fewer bytes per value decodes faster at equal model
      quality.

    References:
      https://arxiv.org/abs/2305.13245
        Ainslie et al. 2023. GQA: Training generalized multi-query
        transformer models from multi-head checkpoints.
      https://arxiv.org/abs/2402.02750
        Liu et al. 2024. KIVI: A tuning-free asymmetric 2bit quantization for
        KV cache.

    Results:
      2026-10-21, 1x A100-SXM4-80GB (MIG 2g.20gb), torch 2.11.0+cu128,
      seed 20261021: 1082.1-1083.8 decode tokens/sec over two runs (3
      repeats each, batch 8, 4072 steps to the 4096-token context), p99
      step 9.1-9.2 ms, kv cache 134217728 bytes at full context, 4096
      per token.

    Returns:
      config: The frozen decode benchmark.

    """
    cfg = DecodeBenchmark.Config()
    cfg.prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that",
        "The mitochondria is the powerhouse of the cell",
        "The speed of light is approximately 299792458 meters per second",
        "Photosynthesis is the process by which plants convert",
        "Quantum entanglement is a phenomenon where two particles",
        "To be or not to be, that is the question",
        "Water boils at 100 degrees Celsius at sea level",
    ]
    cfg.num_decode_steps = 4096 - cfg.prompt_len
    cfg.device = "cuda"
    return cfg


def exp001() -> DecodeBenchmark.Config:
    """Store int8 KV rows in the frozen exp000 decode benchmark.

    ``allocate_cache`` becomes :func:`int8_layer_cache`: each token's key
    row and value row quantize symmetrically to signed int8 against one
    scale, and the float32 scale's bits pack into the row's trailing
    columns, so measured cache bytes count codes and scales together.
    Attention reads the dequantized slice in the model's dtype, so the
    model, its dtype, the prompts, the context length, and the harness are
    exp000's.

    Hypothesis:
      At a 4096-token context a decode step reads more cache than weights,
      so decode tokens per second scale with the bytes each cached value
      occupies. int8 codes with one scale per token move 68/128 of exp000's
      bf16 bytes per value, so decode runs faster while the greedy tokens
      stay within the quantization error of the control.

    References:
      https://arxiv.org/abs/2402.02750
        Liu et al. 2024. KIVI: A tuning-free asymmetric 2bit quantization
        for KV cache. KIVI's per-token value granularity motivates this
        fork's scale; KIVI quantizes keys per channel at 2 bits, while this
        fork keeps both halves per-token at 8 bits as the first rung.

    Results:
      2026-10-21, 1x A100-SXM4-80GB (MIG 2g.20gb), torch 2.11.0+cu128,
      seed 20261021: 730.1-738.6 decode tokens/sec, 33% below exp000's
      1082.1-1083.8 under the same seed; p99 step 13.2-13.6 ms; kv cache
      71303168 bytes at full context, 2176 per token against exp000's
      4096. The hypothesis's speed half fails on this harness: update
      returns the dequantized valid slice every step, so each decode step
      re-reads the codes, rewrites the whole cache in the model's dtype,
      and attention reads that copy -- more traffic than bf16's single
      read, whatever the codes occupy. Token agreement with exp000 is
      0.567 over the generated tail, first divergence at position 35.
      Near-tied logits from untrained weights could contribute, but logit
      margins were not measured. These observations do not establish why
      the streams diverged or support an equal-quality claim.

    Returns:
      config: exp000 with the int8 cache allocator.

    """
    cfg = exp000()
    cfg.allocate_cache = int8_layer_cache
    return cfg


def exp002() -> DecodeBenchmark.Config:
    """Store fp4 KV rows in the frozen exp000 decode benchmark.

    ``allocate_cache`` becomes :func:`fp4_layer_cache`: each token's key row
    and value row quantize to the e2m1 grid against one scale, two codes
    ride in one byte, and the float32 scale's bits pack into the row's
    trailing columns, so measured cache bytes count codes and scales
    together. Attention reads the dequantized slice in the model's dtype,
    so the model, its dtype, the prompts, the context length, and the
    harness are exp000's.

    Hypothesis:
      At a 4096-token context a decode step reads more cache than weights,
      so decode tokens per second scale with the bytes each cached value
      occupies. fp4 codes with one scale per token move 36/128 of exp000's
      bf16 bytes per value at 64 channels, so decode runs faster while the
      greedy tokens stay within the quantization error of the control.

    References:
      https://arxiv.org/abs/2402.02750
        Liu et al. 2024. KIVI: A tuning-free asymmetric 2bit quantization
        for KV cache. KIVI's per-token value granularity motivates this
        fork's scale; KIVI quantizes keys per channel at 2 bits, while this
        fork keeps both halves per-token at 4 bits as the next rung.
      https://github.com/opencompute/microscaling-formats
        OCP Microscaling Formats (MX) working group. The e2m1 grid this
        fork snaps to: {0, 0.5, 1, 1.5, 2, 3, 4, 6} and its negatives.

    Results:
      2026-10-21, 1x A100-SXM4-80GB (MIG 2g.20gb), torch 2.11.0+cu128,
      seed 20261021. The nibble unpack as first written -- a shift and a
      mask, two int64 casts, two gathers, a stack -- ran 545.6-546.6 decode
      tokens/sec, 49% below exp000's 1082.1-1083.8 under the same seed,
      p99 step 19.1 ms. Replacing it with a 256-entry table that maps a
      whole packed byte to its (high, low) e2m1 pair in one int64 gather
      lifts the run to 643.8-649.2 tokens/sec against that session's
      exp000 1099.6 and exp001 746.3-746.6, p99 step 14.7-15.2 ms: 41%
      below exp000 instead of 49%, with dequantized output bitwise equal
      (torch.equal holds against the shifts on a quantized store), so the
      generated tokens do not move. kv cache stays 37748736 bytes at full
      context, 1152 per token against exp000's 4096 and exp001's 2176.
      The byte half of the ladder holds; the speed half still fails as
      exp001's did -- the dequantized slice is materialized every step, so
      each step rewrites the valid cache in the model's dtype and
      attention reads that copy instead of the codes, and no unpack
      micro-optimization pays back a pass the read path does not need.
      Attention over packed codes with dequantization in registers is a
      possible next experiment; its speedup has not been measured.
      Token agreement with exp000 is
      0.158 over the generated tail, first divergence at position 4,
      unchanged across the unpack swap because the output is bitwise
      equal. Near-tied logits from untrained weights could contribute, but
      logit margins were not measured. The cause of divergence and model
      quality remain unmeasured; a trained checkpoint or logit-level
      comparison is needed before claiming equal quality.

    Returns:
      config: exp000 with the fp4 cache allocator.

    """
    cfg = exp000()
    cfg.allocate_cache = fp4_layer_cache
    return cfg


def exp003() -> DecodeBenchmark.Config:
    """Serve exp002's packed fp4 rows straight to a fused attention kernel.

    ``allocate_cache`` becomes :func:`fp4_fused_layer_cache`: the same packed
    e2m1 rows and in-row float32 scales as exp002, but the cache serves the
    packed slices themselves. ``attn_kernel`` becomes
    :class:`Fp4TritonAttention`, whose Triton kernel unpacks each byte
    through a 16-entry table in registers, scales by the row's float32 scale,
    and attends -- the dequantized cache-sized copy exp001 and exp002
    materialize every step is never written. The model, its dtype, the
    prompts, the context length, and the harness are exp000's. Cache storage
    matches exp002: 1152 bytes per token, versus exp000's 4096. The allocator
    and the kernel consuming its packed rows differ from exp002.

    Hypothesis:
      exp002's −41% is the dequantized copy, not the codes: with the
      materialization pass gone, decode reads packed rows (~0.56 B per
      value including the scale's share) instead of a bf16 rewrite (2 B),
      so exp003 beats exp000's tokens per second at exp002's cache bytes.
      Frozen before measuring, on the same seed and session: SUCCESS is
      exp003 >= exp000 (the ladder's speed sign flips); FAILURE is exp003
      <= exp001 (fusion did not pay for its unpack); between the two is
      reported as a partial result with no sign-flip claim.

    References:
      https://arxiv.org/abs/2402.02750
        Liu et al. 2024. KIVI: A tuning-free asymmetric 2bit quantization
        for KV cache. The motivation this rung tests end to end: packed
        codes only win when attention reads them directly.
      priml.baselines.nanochat.exp016
        The house precedent for a Triton kernel entering a ladder as an
        experiment rung rather than a core-model change.

    Results:
      AUTHOR-REPORTED / UNVERIFIED: 2026-10-21, same A100 box, seed 20261021,
      two runs: exp003 690.6-692.1
      tokens/sec, p99 13.9 ms, kv 37748736 bytes -- against that session's
      exp000 1068.8-1070.9 and exp002 634.8-638.6. FAILURE by the frozen
      rule: fusion beats exp002 by 8.6% and cuts p99, but lands below
      exp001 and recovers only ~13% of the gap to exp000. The author also
      reported max abs kernel-output difference 0.00195 against SDPA over
      _dequantized rows under the ladder's masks.
      A follow-up probe changed the head counts
      (``num_heads_kv = 8``, ``kv_groups = 1``, same seed): 703.4-705.9
      tokens/sec -- +1.9% over exp003, still ~36% under exp000's same-
      session 1106-1109. Raw timing artifacts and a profiler trace are not
      included, and these results have not been reproduced after the fixes.
      The timings do not isolate a dominant cost or establish a CPU/launch
      bottleneck. Packed storage is 1152 * 4096 * 8 = 37748736 bytes (36 MiB)
      for the batch at full context, before GQA expansion and scale clones;
      allocated bytes are not measured memory traffic. The current path
      still expands packed GQA heads and clones scales into float32 tensors.
      Read traffic, quantized writes, and launch overhead remain hypotheses
      requiring profiling and controlled comparisons.

    Returns:
      config: exp000 with the packed-served fp4 cache and the fused Triton
        attention kernel.

    """
    cfg = exp000()
    cfg.allocate_cache = fp4_fused_layer_cache
    block = cfg.model.block
    assert isinstance(block, TransformerBlock.Config)
    attn = block.attn
    assert isinstance(attn, Attention.Config)
    attn.attn_kernel = Fp4TritonAttention.Config()
    return cfg
