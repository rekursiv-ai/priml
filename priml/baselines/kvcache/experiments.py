"""KV-cache decode experiments.

``exp000`` times greedy decode of a byte-level GQA decoder over a cache that
grows to a 4096-token context, with the cache stored at the model's own
dtype. Every later experiment forks a named parent and changes ONE thing:
``allocate_cache``, the storage the cache uses.

    exp000  bf16 weights, bf16 KV cache, 4096-token context

Launch::

    uv --quiet run --frozen python -m priml priml.baselines.kvcache.experiments.exp000

Objective:
  Raise decode tokens per second, or shrink cache bytes at equal tokens per
  second, by changing how the cache stores keys and values.

Score:
  Decode tokens per second over every timed step, prefill excluded, with the
  p99 step beside it: a scheme that stalls some steps shows there first.

A fork may change:
  ``allocate_cache``: the cache's dtype, layout, quantization, and the
  arithmetic that reads it back.

A fork may not change:
  The model, its weights' dtype, the prompts, the context length, or the
  harness and its timing.
"""

from __future__ import annotations

from priml.baselines.kvcache.decode import DecodeBenchmark


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
      TBD.

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
