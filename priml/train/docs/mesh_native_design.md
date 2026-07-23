# Mesh-Native `priml`: DeviceMesh as a First-Class Citizen

Status: DESIGN (awaiting owner review)
Mother issues: #298 (tensor parallelism), #299 (EMA consolidation)
Author: campaign coordinator
Date: 2026-06-03

## Thesis

`priml` should be **mesh-native**: a `DeviceMesh` is plumbed through the
library so that *any* mesh topology (`tp=1`, `tp=4`, `dp×tp`, `dp×pp×tp`)
"Just Works" without per-topology branching in user code. The deep-review
campaign (#273) found the library is mesh-naive everywhere except
`runtime`/`seed`/`parallelism`/`train_loop`; this design closes that gap.

The single rule readers rely on:

> **A component declares HOW it shards (in its config); the mesh decides
> WHETHER and ACROSS WHAT.**

The sharding plan is the configgle config tree — not a separate registry.

## Why now (campaign context)

- TP was a dead stub: `TensorParallel.__call__` required
  `model.apply_tensor_parallel_plan(mesh)`; zero models implemented it
  (verified by grep).
- The coordinator initially recommended staging TP to `experimental/`; the
  owner corrected: **fix TP, make models mesh-native.** This doc reverses the
  staging.
- Mesh dim `tp` is already plumbed in `runtime.py` (`mesh_topology`); the
  infrastructure is ready, only the consumers are missing.

## Decided design (B): shard style on the building-block Config

Rejected alternative (A): a convention registry in the applier keyed by
submodule name. Rejected because it couples sharding to *names* (a hidden
global contract), silently mis-shards a model whose submodules are named
differently, and adds a mechanism *beside* configgle rather than using it.

Chosen (B): the shard style is a configgle field on each building-block
Config. This is the configgle idiom (`Literal` enum field, documented,
basedpyright-checked, dispatchable) and makes the **config tree the sharding
plan** — composed hierarchically via `finalize()` sentinel-propagation,
exactly like `channels_in`/`channels_out` inference already works.

```python
class Linear:
    class Config(Fig["Linear"]):
        ...
        shard: Literal["none", "colwise", "rowwise"] = "none"
        """Tensor-parallel sharding style over the mesh tp dim; none = replicated."""
```

### Mechanism: parents wire `shard=` into inline child configs (not Makeable slots)

The loop building blocks construct their children **inline** in `__init__`
from non-slot configs (`self.proj_qkv = EnsembleLinear.Config(...).make()`),
not via `Makeable` config slots. Every child parameter (`channels_in`,
`depth`, `init_weight`) is already passed inline there. So the shard style is
wired the same way — the parent sets the child's `shard` at the inline
`.Config(...)` call, fixed by the child's role:

```python
class SelfAttention:
    def __init__(self, config: Config) -> None:
        self.proj_qkv = EnsembleLinear.Config(
            ...,
            shard="colwise",  # heads are the parallel axis
        ).make()
        self.proj_out = Linear.Config(
            ...,
            shard="rowwise",  # reduce-scatter back to residual
        ).make()
```

This honors "the config tree IS the plan" using the existing inline-wiring
pattern; it does NOT introduce `Makeable` child slots (which these blocks
deliberately don't use). The contract is asserted on the **built module**
(`attn.proj_qkv.shard == "colwise"`), not on a config slot.

A model composed of standard blocks shards automatically; no per-model TP
code, no naming dependency.

## The generic applier

One function in `lib/train` (un-staged from `experimental/parallelism/`):

```python
def apply_tensor_parallel(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """Shard each submodule per its declared shard style over mesh['tp'].

    tp size 1 -> structural no-op (forward bit-for-bit unchanged).
    """
```

It walks the module tree, reads each module's shard style (stored on the
runtime module at construction from `config.shard`), and applies the matching
`ParallelStyle` via `parallelize_module`. The `ParallelStrategyProtocol`
(from the #273 W1 redesign) owns when this runs in the device→shard→
materialize→reset lifecycle.

## The one exception: custom (non-`nn.Linear`) layers

Spike result (verified, 2026-06-03):

- `Linear` + `ColwiseParallel` via `parallelize_module`: **works**, output
  matches dense.
- `EnsembleLinear` (custom einsum `"...c,edc->...ed"`) over a `Shard(0)`
  weight: **fails** — `aten.bmm got mixed torch.Tensor and DTensor`. The
  einsum needs *both* operands as DTensors.

Rule the design names:

> **Standard `nn.Linear`-based blocks use the built-in Colwise/Rowwise
> styles. Custom layers (`EnsembleLinear`, anything with a hand-written
> forward) MUST provide their own `ParallelStyle`** that redistributes
> input/output (e.g. `Replicate` the input, `Shard` the ensemble dim, define
> the output layout) so the forward sees consistent DTensor operands.

`EnsembleLinear` (attention's `proj_qkv`) gets a custom style sharding the
ensemble (head) dim — heads are the natural parallel axis.

### Split-alignment rule (fused-output layers)

Verified hazard (cpu:gloo tp=2 spike): a layer whose forward **splits or
chunks a projection output** along a sharded dim breaks under naive
Colwise/Rowwise, because each rank's local slice straddles the split boundary
and the post-projection `chunk`/`split` mis-pairs the segments.

- Fused-gate `SwiGLU`: `up_proj` outputs `2*c_hidden`, forward does
  `up_proj(x).chunk(2, dim=-1)` → `(gate, x)`. Naive colwise gives err 0.22
  (sharded != dense).

Rule:

> **A layer that splits/chunks a projection output along a sharded dim must
> provide a `ParallelStyle` that shards each split-segment consistently**, so
> the post-projection split stays aligned with shard boundaries.

Fused `SwiGLU` gets a custom style sharding `up_proj` so each rank holds
`[gate_shard_i | up_shard_i]` (the two halves sharded identically) — local
`chunk(2)` stays paired (err returns to ~3e-8). The `proj_qkv` ensemble split
is covered because it shards the ensemble axis directly.

### MLA partial-shard (latent is head-shared)

`MultiHeadLatentAttention` cannot be fully sharded by head:

- q-path and output are per-head `[B, S, n, ...]` — cleanly head-parallel.
- The latent (`c_kv` / `k_pe`, dims `kv_lora_rank` / `qk_rope`) is
  **head-shared** (no `num_heads` dim) — sharding it over the `tp` (head) axis
  is a **correctness bug**, not just wasteful.
- Absorb-math reshapes `kv_b_proj.weight` in Python — a DTensor weight breaks
  the hand-written reshape.

Rule for MLA: shard only `q_proj`/`q_b_proj` (colwise, head dim) and `o_proj`
(rowwise); leave `q_a`/`kv_a`/`kv_b`/lora **replicated** (`shard="none"`).
A code comment in `mla.py` records why the latent path stays replicated.

These custom styles are per-LAYER-TYPE (SwiGLU, EnsembleLinear, MLA q/o),
reusable across models. "No per-model TP code" holds — models compose styled
blocks; the styles live with the layer types.

### Attention kernel must be DTensor-compatible under TP

Verified (cpu:gloo tp=2): the fused flash kernel
(`F.scaled_dot_product_attention` →
`aten._scaled_dot_product_flash_attention_for_cpu`) has **no DTensor sharding
strategy registered** — sharded attention raises `NotImplementedError`. The
manual `SdpaNaive` kernel (matmul + softmax) is DTensor-decomposable and
composes correctly (sharded == dense, err 6e-8).

Because `attn_kernel` is a configgle `Makeable` slot, this needs no
special-casing: **TP models set `attn_kernel=SdpaNaive`.** The applier raises a
clear error if it shards a `SelfAttention` still using the fused flash kernel,
rather than letting the cryptic deep-aten `NotImplementedError` surface.
`tp=1` is unaffected (no DTensor, fused flash runs as normal — goldens green).

## Invariants

1. **`tp=1` is bit-for-bit.** Single-device forward must equal pre-change
   forward. W0 model goldens (transformer/mla/gated_delta_net/moe/mmdit)
   stay green untouched.
2. **Sharded == dense.** Multirank cpu:gloo `tp=2` test: sharded model output
   `== ` dense model output (within fp reassociation tol), proving the plan
   is correct, not just non-crashing.
3. **No per-model TP code.** Models declare shard styles on blocks via config
   finalize; the applier is generic.

## Extension to the rest of the library (same contract)

The "declare how, mesh decides whether" rule plumbs every mesh-naive
subsystem the campaign found:

- **data** (`data/pipeline`, `data/sources`): sources declare they shard on
  `dp`; the loader uses mesh `dp` rank/world instead of bare DataLoader
  worker slicing. (Today: worker-slice only, mesh-unaware.)
- **optimizers** (`optimizers/*`): operate on `param.to_local()` for DTensor
  params, or are DTensor-aware. Muon's Newton-Schulz and Newton's Hessian
  must be validated on sharded params. (Today: no DTensor handling.)
- **EMA** (`train/ema.py`, #299): param-dict name-keyed shadow over local
  shards (the sudoku pattern, already proven across sudoku/arcagi1/maze),
  replacing the deepcopy-clone shadow that breaks under `fully_shard`.
- **checkpointing** (`train/checkpointing.py`): DCP
  (`torch.distributed.checkpoint`) saves/loads DTensor state by mesh, instead
  of the hand-rolled per-rank shard files. (Today: hand-rolled, not
  mesh-native.)

Each is a separate child; TP (#298) proves the pattern first.

## Child issues

Under #298 (TP):
1. `ShardStyle` field + generic `apply_tensor_parallel` applier; un-stage
   `TensorParallel` to `lib`; wire into `ParallelStrategyProtocol`.
2. Custom `ParallelStyle` for `EnsembleLinear` (ensemble-dim shard) + spike
   hardened into a test.
3. Shard styles on `Linear`/`SwiGLU`/`MoE`/`SelfAttention`/`MLA`/`embedding`
   block Configs via `finalize()`.
4. Multirank cpu:gloo `tp=2` correctness test (sharded == dense) + `tp=1`
   golden-green regression.

Under #299 (EMA): promote sudoku param-dict EMA to canonical `lib`; migrate
sudoku/arcagi1/maze (they share `TRMTrainStep`) + cifar/nanogpt.

Follow-on mothers (after TP proves the pattern): data-mesh-sharding,
optimizer-DTensor, checkpoint-DCP.

## Open questions

None blocking. Confirm: (a) custom-layer-provides-own-style rule acceptable;
(b) child partition above; (c) whether to extract the cross-imported
`TRMTrainStep` from `experimental/sudoku` to `lib` (separate decision).

## Estimate

- TP core (#298 children 1–4): ~1–2 focused sessions; CPU-gloo testable.
- EMA (#299): ~1 session (pattern exists, mostly migration).
- Data/optim/checkpoint mesh-native: larger; scope each as its own mother
  after TP lands.
