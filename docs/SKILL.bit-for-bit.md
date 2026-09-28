---
name: bit-for-bit
description: ALWAYS invoke this skill when the user suspects a numerical regression, wants two codepaths verified identical, asks to validate a refactor did not change numerics, or says "bit-for-bit". Do not compare numerics ad hoc -- invoke first.
---

# Bit-for-Bit Verification

Verifying bit-for-bit equivalence across code changes.

## The Rule

**NEVER EDIT SOURCE FILES. ONLY MEASURE VALUES.**

Find divergence first. Understand cause second. Fix third.

Numerical invariants (hold everywhere, not just here):

- **Never loosen test tolerances to fix a test.** Tolerances are
  evidence; a change means something regressed.
- **Never swap custom numerics for stdlib without verifying tails.**
  Custom code usually exists for precision. If unsure, ask.

## Porting against a reference implementation

Use this when a port (e.g. torch) must reproduce a reference (e.g. JAX).
"Bit-for-bit" means every function, including every random one, produces
identical bits given identical inputs and identical random draws. It does
not mean matching statistics. Do these steps in order. Do not start a
whole-program rollout until step 5 is green.

1. **Inventory.** List every function in the reference and pair each with
   its port function in one table. An unpaired function is a gap. So is a
   port function that restructures the reference, e.g. a permutation in
   place of sequential weighted choice. Rewrite it to mirror the
   reference's arithmetic and draw order.
2. **Pin randomness on both sides.** Patch the lowest random primitives,
   e.g. JAX's `jax._src.random.core` uniform and randint, not only the
   public names. Let the reference's own sampling code (`choice`, etc.)
   run on top of them. Emulate that exact formula on the port side; for
   JAX's weighted choice that is `searchsorted(cumsum(p), p_total*(1-u))`.
   Pass the pinned value as a traced argument, not a closure, or jit
   caches will replay stale values. Any unpatched primitive must raise.
3. **Compare by bisection.** Split the system into its few major
   components (in ML: the model forward, the dataloader, a few train
   steps; in an env: world gen, the step's sub-phases) and compare each
   exactly. A component that matches, with full coverage (step 4), is
   done. Split only the mismatching ones into their sub-functions and
   recurse until each mismatch is pinned to one function's own code.
   For each pair, run many random
   inputs times several pinned values, with the reference under the same
   compilation and backend it uses in production (jit, GPU). Compare every
   output field with exact equality. Collect ALL mismatches into one
   report (function, field, count, first example). Do not stop at the
   first.
4. **Prove coverage.** Count pinned calls per call site (file:line) on
   both sides. Count at trace time for jitted code. Assert every random
   call site in both codebases was reached, and that inputs drive every
   branch (each side of every `where`/`select`/`cond`) at least once. An
   unreached site or branch is untested.
5. **Integrate.** Only then run whole-step or rollout lockstep with pins,
   comparing full state after every step.
6. **Fix every divergence in the port, then repeat from step 3.** No
   separate approval is needed; the task is not done while any mismatch
   remains. For each bug, add exactly one reference-free native test:
   hardcode the reference's captured values as literals. Prove it red
   with only that fix reverted.

Writing the harness, adapters, and tests is not "modifying code" in the
sense of the debugging rules below; start them immediately.

Statistical or distributional tests are not bit-for-bit. They miss any
bug that preserves rates. Keep them only as a supplement.

Reference semantics a JAX port must mirror (each was a real bug):

- `choice(p=x / x.sum())` with `x` all zero gives NaN weights and returns
  index 0. Do not fall back to uniform.
- `dynamic_slice` / `dynamic_update_slice` wrap a negative start by the
  axis length, then clamp to `[0, dim - size]`. Do not clamp negatives
  to 0.
- XLA lowers a float32 divide by a *traced* value on GPU to
  `x * rcp(y)`, with `rcp` rounded toward zero. Division by a Python
  constant lowers differently. Find the lowering by measuring a
  rounding-mode table, not by guessing.
- Eager and jitted XLA round differently, and so do CPU and GPU. Compare
  against the mode production uses. If the reference itself differs
  across backends, pin to the production backend. Do not add a
  tolerance.
- An unconditional reference write (e.g. `type_id` into an empty slot, a
  dead projectile's position) must be unconditional in the port too, even
  when the value looks unused.

## Host-dependent float divergence (cross-implementation parity)

When a test asserts bit-for-bit equality between **two different
computation paths** (e.g. our module vs a reference/third-party model,
a rewrite vs the original), the divergence may be neither codepath's
"bug" -- it can be the host. A float32 reduction (matmul, softmax,
sum, any transcendental) lands on a different last mantissa bit
depending on the CPU's vector width (AVX2 vs AVX-512), thread count,
and vendor (AMD vs Intel), because the kernel accumulates in a
different order. Two paths that reduce in different orders then agree
on the machine where the golden was minted and diverge (~1e-7) on
another.

Principle: **for bit-for-bit across implementations, the two paths must
issue the same primitive ops in the same order** -- not merely compute
the same value.

Diagnose before fixing:

- Confirm it is host/order, not a real regression: run the *same*
  codepath twice (reproducible?) and, if possible, the *same* comparison
  on a second CPU class or thread count. Identical drift under a torch
  upgrade but differing across hosts points to reduction order, not code.
- A test that only fails on a different machine than CI is the signature. An
  integration-gated parity test can pass for months on one host and never run
  on the host where it fails.
- Sweep one axis at a time. Threads, vector ISA, and vendor are three
  variables: rerun at 1/2/4/8/64 threads, then under each ISA cap. Identical
  bits rules that axis out. What is left is vendor, which no env var fixes.
- A null result on the wrong host proves nothing. `MKL_CBWR` measured inert on
  AMD because MKL takes a generic path there regardless. Ask which host a knob
  was written for, and test on that one.

Report ULPs, not absolute difference. 1 ULP means the hosts round differently;
a large count means the computation changed. The absolute value says neither:
one float32 ULP is 1e-7 near one and 1e-45 near zero. A differencing helper
that downcasts to report will print `0.000e+00` for a float64 miss below
float32 resolution.

Two portable fixes (never loosen `torch.equal` to `allclose` to mask it):

1. Paths you control: run both inside
   `host_agnostic_numerics()` (search for its definition in `bfb.py`), which upcasts
   every float32 arithmetic op to float64 and downcasts.
2. One path is third-party with a **fused** kernel the dispatch-mode
   upcast cannot reach (e.g. HuggingFace's default fused SDPA attention):
   force both sides onto the same **unfused** kernel so they issue
   identical primitives. Example: `qwen3_hf_test.py` builds HF with
   `attn_implementation="eager"` and uses `SdpaNaive` for the other path, restoring
   exact `torch.equal`.

### The downcast is the mechanism

Fix 1 works because of the round back, not because float64 agrees. Measured on
torch 2.11 against `mpmath` at 60 digits, `sigmoid` over 4096 inputs: native
float32 is wrong 1277 times, float64 is wrong 1316 times (by 2 ULP), and
float64 rounded once to float32 is wrong 0 times.

Float64 is an approximation too -- each vendor's libm picks a different
polynomial. Two float64 ULP is ~2**-28 of one float32 ULP, so the rounding
absorbs it. Two consequences:

- A golden's comparand must be float32. A runner returning the float64 scratch
  keeps that host's libm error. One did, off by 1 ULP between an Intel laptop
  and an AMD server, passing on the mint host for months.
  `_assert_portable_output_dtype` refuses anything but float32.
- Masking low mantissa bits does not substitute. Two values 1 ULP apart can
  straddle a bucket boundary, so masking never reaches zero: 15.3% residual at
  1 bit, 0.012% at 12, while each bit halves sensitivity to the regression you
  are looking for. A test that fails 1 run in 10,000 gets marked flaky and
  deleted.

### What the upcast does not fix

- BLAS kernel choice. A float64 GEMM's reduction order varies with the kernel
  MKL selects, and a float64 difference crosses a float32 boundary
  occasionally. `MKL_CBWR=COMPATIBLE` (priml conftest) pins it. Removing that
  pin measured inert on AMD and broke goldens on Intel.
- Non-torch stacks. JAX/XLA has no dispatch-mode upcast, so none of this
  applies there.

### When to key the golden on the host instead

Some divergence cannot be normalized and has to be named. For a JAX/XLA
golden, measured: capping the vector ISA changes nothing (`AVX2`, `AVX512`,
and unpinned hash identically on an AVX-512 host), while Intel and AMD differ
by one float32 ULP at any cap. Integer and PRNG arrays still match exactly --
that asymmetry means codegen, not logic.

So the golden filename keys on `(system, machine, cpu_vendor, backend)`, and a
host with no archive skips. A missing golden means nobody minted one for that
machine. Do not invent a tolerance, and do not pin an env var that measurement
shows is inert.

## Goldens: small, scoped, conventionally named

A golden proves the code path, not kernel throughput. Every checked-in `.pt`
stays at most 28,000 bytes, and most are far smaller (`golden_test.py`
enforces the ceiling). SIMD width is not what the test pins; touching every line of Python
is.

### Store only what the check needs

Before writing a golden -- and before shrinking one -- classify every stored
item. Do this first; trimming what is already stored is how 192 KB goldens
survive.

1. **Rebuilt by the test? Do not store it.** An input made by deterministic
   code (`arange`, literals, fixed masks) is rebuilt on every run; storing it
   re-checks `torch.arange`, not the model. arcagi3 `canonical_exp000_training`
   stored its 96 KB batch this way.
2. **An input the replay cannot rebuild? Store it whole.** Randomized initial
   weights are this. Never regenerate them from a seed: RNG bits differ across
   architectures and torch versions, so a seed is not a spec.
3. **Only compared for equality? Store a digest.** A post-run state, a
   gradient, or an output checked with `torch.equal` needs a SHA-256 of its
   dtype, shape, and bytes (`bfb.state_digest`), not a copy. Keep its first
   few elements beside the digest so a failure still reports its size in ULPs.
   Raw 32-byte digests, never hex strings keyed by name.
4. **Implied by something already stored? Drop it.** Intermediate gradients
   and step-1 optimizer state are implied by the final optimizer state; a
   resumed run checked equal in place needs no stored copy; a full model output
   is implied by the losses and final state computed from it.
5. **Diagnostic only?** Keep it only if it is a few scalars (per-step losses
   say which step broke).

Whatever remains is the golden. If it is still over the ceiling, shrink the
model by size only (next section) -- never drop an item from class 2 or 3.

### Scope and names

- Golden only the nexus experiments. A derivative recipe gets no golden.
- Test a component in the baseline that owns it. A sudoku mechanism is unit
  tested in `sudoku/`, not replayed through an ARC recipe.
- Name goldens by priml convention: `testdata/<expNNN>[_<precision>].pt`, or
  `<module>.pt`. Never `legacy_`, `oss_`, `source_`, `_digest`, and never json.
- Mint on the SOURCE side. The minting test sits beside the old code (a
  `priml_golden_test.py` in the source package) and is deleted with it; the
  priml test only replays. A priml test importing the stack it replaces is a
  bug.
- Mint under the priml conftest environment (`OMP_NUM_THREADS=1`, ...,
  `MKL_CBWR=COMPATIBLE`), i.e. through `pytest`, never a bare `python`: a
  two-rank golden minted without it failed replay.

### Shrink levers

Apply in order; re-mint and confirm zero source mismatches after each.

| Lever | Before -> after (measured) | Why it stays a valid test |
|---|---|---|
| Shrink by size only: width 16->8, heads 2, 1 layer, batch 2, 3-5 steps, 9-cell grid, `round_to` 8 | arcagi1 `exp004` 216 KB -> 38 KB (with the rows below) | Every numerical choice (init, norms, optimizer, loss, ACT) still runs. |
| Compared-only state as a digest plus first elements (`bfb.state_digest`; the bfb harness does this for the post-run state) | nanochat `exp022` 69 KB -> 42 KB; speedrundit `reg_source` 40 KB -> 16.6 KB (hex -> raw bytes) | SHA-256 over dtype, shape, and bytes: any one-bit change fails; the first elements keep the ULP report. |
| Drop stored inputs the test rebuilds, and outputs implied by stored state | arcagi3 `canonical_exp000_training` 192 KB -> 11 KB | The rebuilt input and the implied output are still computed and still reach the compared state. |
| RNG as its next 8 draws (`golden.rng_fingerprint`), not the 5 KB Mersenne state | 5,056 B -> 64 B per record | Same position pinned; any extra draw changes it. |
| One key per quantity: stack per-step values on a step axis (`golden.put_steps`); join per-parameter heads into one tensor | arcagi1 110 -> 57 keys; `exp004` 25 KB -> 13 KB | Same values; per-key pickle overhead (~70-100 B) dominated. |
| Write with `golden.write_tensors`: one flat tensor per dtype plus a zlib index | arcagi2 `metric_distributed` 300 keys: 25 KB -> 5.3 KB | File size stops growing with key count; still a plain `torch.load`. |
| Narrow whole small numbers to uint8 (`golden.stored`) | token grids, counters: 8x smaller | Exact; a value that stops being whole changes the dtype, which is reported. |
| Record carried state once, at the end (ACT latents, final params) | arcagi2 `distributed` 33 KB -> 18.6 KB | Final carried state depends on every step. |
| Never lzma or other whole-file compression | -- | `bfb_test` `torch.load`s every `.pt`; a compressed one broke it. |

Prove each shrunk golden still bites: perturb one weight by one ULP, or one
constant, and require a reported mismatch.

### Order is numerics

Parameter REGISTRATION order is part of the computation. A body norm sums
parameters in that order; under `torch.compile` the float32 reduction landed
1 ULP differently when a prefix was registered after the blocks instead of
before. Eager goldens under `host_agnostic_numerics` cannot see this; only the
compiled comparison did. Match the reference's registration order and pin it
with a unit test.

## Protocol

1. Create an isolated checkout of the known-good commit. Read the repository's
   Git policy first and obtain any required approval for the exact command.
2. Write a comparison script that runs **both codepaths unmodified**.
3. Save checkpoints outside the checkout, under the repository's configured
   artifact directory. Discover its location from the project instructions.
4. Find the first divergence point. Only **then** read code to
   understand why.

## Checkpoints (in order)

1. End of `__init__`: model weights, RNG state.
2. After data loading: first batch.
3. After augmentation: inputs, labels.
4. Model forward: inputs and outputs.
5. Loss values.
6. Gradients before optimizer step.
7. Weight deltas after optimizer step.

If checkpoint N matches but N+1 diverges, the bug is between them.
Add more checkpoints and repeat.

## Script structure

Use functions, not embedded strings. Subclass to override
settings -- never edit source files:

```python
from pathlib import Path

# Replace with the artifact directory discovered above.
checkpoint_dir = Path("/path/to/artifacts/bit-for-bit/my-comparison")
checkpoint_dir.mkdir(parents=True, exist_ok=True)


def run_good():
    import torch
    from package.experiment import Experiment

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    class TestExp(Experiment):
        compile_model: bool = False  # Override via subclass, NOT source edits

    exp = TestExp()
    torch.save({"weights": exp.model.state_dict()}, checkpoint_dir / "good_ckpt.pt")


def run_suspect():
    # Same pattern pointed at the suspect codebase
    ...


def compare():
    import torch

    good = torch.load(checkpoint_dir / "good_ckpt.pt", weights_only=True)
    sus = torch.load(checkpoint_dir / "sus_ckpt.pt", weights_only=True)
    for k in good["weights"]:
        if not torch.equal(good["weights"][k], sus["weights"][k]):
            diff = (good["weights"][k] - sus["weights"][k]).abs()
            print(f"DIVERGENCE: {k} max_diff={diff.max().item():.6e}")
```

## Pitfalls

1. **Don't hypothesize from code.** Only measured values matter.
   Code can differ and produce identical results; identical-looking
   code can diverge.
2. **Don't modify code before finding divergence.**
3. **Don't blame `torch.compile` without evidence.** Disable in
   *both* via subclass, not source edits.
4. **Verify training data is identical.**
5. Check inherited defaults -- print actual runtime values.
6. **RNG discipline:** any `torch.randn()` call advances state,
   even if multiplied by zero.
7. **Sign conventions:** if losses differ by sign, check whether
   the formulas are equivalent. Compare gradients and weight
   deltas, not just losses.
8. Run comparisons multiple times -- GPU cache can cause spurious
   divergence.
9. **Set explicit seeds in BOTH subprocesses.**
10. **CUDA RNG:** batched `torch.randn([K, D])` ≠ stacked loop of
    `torch.randn(D)`.
11. `dtype` params affect RNG -- explicit dtype can consume extra
    RNG calls.
12. Device placement order matters: CPU-then-CUDA vs direct-CUDA
    differ in RNG consumption.
13. **A numerics docstring is a claim; re-measure it or delete it.** Three
    here were false when checked: "float64 GEMM is vector-width- and
    thread-count-invariant", an ISA pin "for portable goldens" that changed
    no bits, and a skip function cited by name that did not exist. Source is
    truth; in this area a stale doc also decides what the next person measures.
