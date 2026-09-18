# Speeding Up Slow Torch Tests

## Contents

- Check dtype before shape (the CPU bf16 cliff)
- Shrink levers, by payoff
- Never shrink these
- Legitimate blockers -- revert, do not force

Profile before changing anything: `cProfile`, read the `cumulative`
column. The cost is `run_backward` or Dynamo tracing, essentially never
the forward pass.

## Check dtype before shape

bf16 autocast is a win on CUDA and catastrophic on CPU. Autograd's
weight-gradient matmul multiplies a TRANSPOSED view of `grad_output`, and
ATen has no bf16 kernel for that layout on a host lacking `avx512_bf16` /
`amx_bf16`, so it falls back to a scalar loop:

| `[512,388] @ [388,16]` | fp32 | bf16 |
|---|---|---|
| transposed lhs (autograd layout) | 0.048 ms | **59.7 ms** |
| same operands, contiguous | 0.045 ms | 0.050 ms |
| CUDA, transposed lhs | 0.0076 ms | 0.0055 ms |

~120x per CPU train step. Confirm on a new host before assuming:

```python
a = torch.randn(388, 512)
b = torch.randn(388, 16)
x = a.to(dt).t()  # transposed lhs is the autograd layout
```

Fix on the TEST CONFIG. Locate the test's configuration and its dtype
fields; the field names below are examples:

```python
_CPU_TEST_AUTOCAST = None
"""fp32 (autocast off), not the bf16 recipe -- CPU kernel cliff."""

cfg.device = "cpu"
cfg.dtype_autocast = _CPU_TEST_AUTOCAST
cfg.model.dtype = _CPU_TEST_AUTOCAST
```

**Never gate this inside the trainer on `device.type == "cpu"`.** A
bit-parity suite compares two ports that autocast independently; a
one-sided gate makes them compute in different dtypes and parity fails for
a reason that is not a real divergence (`oss=3.104384
internal=3.113914`).

Related one-time cost, not worth chasing: the first backward in any
process costs ~2s regardless of shape (autograd starting its thread
pool); later ones are ~0ms. Warming it in a session fixture only relabels
the cost -- measured 29.85s with it against 26.80s without.

## Shrink levers, by payoff

| lever | scaling | notes |
|---|---|---|
| sequence length / token count | **quadratic** | attention. A 30x30 board is 900 tokens; 5x5 is 25. |
| graph size under `torch.compile` | ~linear in depth | tracing dominates: depth 2 traces 0.50s, depth 1 traces 0.13s |
| sequential loop counts | linear | cycles, ACT/rollout steps, train steps, MoE layers -- each re-runs ONE code path |
| trajectory length | linear | bit-parity diverges on the FIRST differing step; length buys confidence, not coverage |
| hidden width / heads | ~linear | usually already minimal |

Derive shapes from ONE named constant per file:

```python
_GRID_SIDE = 5
_GRID_CELLS = _GRID_SIDE * _GRID_SIDE
```

Two levers that look immovable and often are not:

- A harness that "must run the real thing" frequently accepts shape
  parameters. Read its signature.
- When a test pins production widths, the cost is often a per-*layer*
  multiplier. Cut layers, keep widths.

## Never shrink these

- **What the test exists to pin.** Verifying finite gradients at
  production widths 1536/1920 means those widths are the subject.
- **A value another assertion is anchored to.** `vocab_size=512` stays
  when the loss assertion reads `ln(512)`.
- **One side of a parity pair.** Both ports get identical shrinks.
- **A domain constant.** `ARC_GRID_SIZE = 30` is baked into positional
  embeddings; shrinking it needs a production change, so stop and report.

**Distrust comments claiming a test cannot shrink.** One read "asserts
recipe-fidelity parity, so it cannot shrink the model." Wrong -- parity is
about both ports issuing the same primitives in the same order, true at
any width. Shrinking took it 113s -> 1.3s. Test the claim.

## Legitimate blockers -- revert, do not force

- An RNG range needs a minimum (`random_ expects 'from' < 'to'` at
  `max_act_steps=1`).
- A checkpoint the test loads must exist (`save_every` vs `keep_every`
  divisibility).
- The cost IS the subject: a test whose contract is "compile, then reset
  dynamo" cannot avoid paying for compilation.
- Production code asserts the old shape. Relax the CHECK **only** if the
  real requirement is weaker and the guard stays as strong: `if tokens !=
  900: raise` where the math needs only a square becomes `side =
  math.isqrt(tokens); if side * side != tokens: raise`. Never relax a
  check merely to admit a smaller fixture.

## Non-torch slowness

Before reaching for shape levers, check whether the test is simply
*waiting*. A profile with no Python hotspot means a blocked syscall.

Real case: `socket.close()` does not wake a thread blocked in `accept()`
on Linux, so a teardown `join(timeout=5.0)` burned the full 5s on every
test. `shutdown(SHUT_RDWR)` before `close()` took it to 0.000s.
