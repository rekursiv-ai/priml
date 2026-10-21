# KV-cache decode comparison

`exp000` stores bf16 keys and values. `exp001` swaps in INT8 storage;
`exp002` swaps in packed E2M1 codes. Both compressed caches materialize
the valid cache in the model's dtype on every update. Their storage savings
do not imply faster attention.

The experiment docstrings retain the author's reported A100 results:
approximately 1099, 746, and 646 decode tokens/sec, with 4096, 2176, and
1152 cache bytes per token respectively. Both compressed variants were
slower. Raw timing artifacts for those measurements are not included.
The scale-range fixes have not been benchmarked on that A100.

For a paired run, save the following as `paired.py` in your development
checkout. Every factory keeps its default `seed=None`; this comparison
explicitly selects the reported seed, 20261021, for all three runs.
It reports timing, allocated bytes, and agreement over generated positions
only. The model has untrained weights, so agreement does not measure
trained-model quality.

```python
from priml.baselines.kvcache.decode import DecodeReport, token_agreement
from priml.baselines.kvcache.experiments import exp000, exp001, exp002


reports: dict[str, DecodeReport] = {}
for factory in (exp000, exp001, exp002):
    cfg = factory()
    cfg.seed = 20261021
    report = cfg.make().measure()
    reports[factory.__name__] = report
    control = reports["exp000"].tokens[:, cfg.prompt_len :]
    candidate = report.tokens[:, cfg.prompt_len :]
    print(
        factory.__name__,
        {
            "seed": report.seed,
            "tokens_per_sec": report.tokens_per_sec,
            "step_ms_p99": report.step_ms_p99,
            "step_ms_max": report.step_ms_max,
            "prefill_ms_median": report.prefill_ms_median,
            "kv_bytes": report.kv_bytes,
            "kv_bytes_per_token": report.kv_bytes_per_token,
            "generated_token_agreement": token_agreement(control, candidate),
        },
    )
    print("prefill_sec", report.prefill_sec)
    print("step_sec", report.step_sec)
```

Run on the target CUDA device with the environment's normal device
selection. Keep the complete timing output outside the checkout:

```bash
mkdir -p /opt/scratch/artifacts/kvcache/paired
uv --quiet run --frozen python paired.py > /opt/scratch/artifacts/kvcache/paired/timings.txt
```

`fp4_layer_cache` and `int8_layer_cache` support ordinary `Attention`
(including GQA) and `MultiHeadLatentAttention`. They reject other cached
attention families at allocation, including `GatedAttention`, multi-stream
attention, and recurrent delta state. Direct cache updates use a four-byte
float32 scale per row; the module docstrings state its finite range and
saturation behavior. FP4 midpoint ties choose the lower magnitude; this is
an E2M1-grid quantizer rather than a complete MX-compliant converter.
