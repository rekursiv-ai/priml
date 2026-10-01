# ConvexTok

A PyTorch port of ConvexTok (Tempus et al., 2026), the tokenizer that chooses
its vocabulary by solving a linear-programming relaxation of the token-count
objective. From the same text it fits the same vocabulary as the authors'
implementation, and runs about 3.8 times faster end to end on one GPU.

Upstream is [`JanTempus/tokenisation_lp`][upstream] at
`72bc70055a65a313f2bd2e667655588e372d9fdc`, standard training mode, nanochat
pretokenizer. Its solver is NVIDIA cuOpt; this port replaces cuOpt and its C
presolver with torch, Triton and Numba.

## Contents

- [The problem](#the-problem)
- [Upstream vs. this port](#upstream-vs-this-port)
- [Equivalence](#equivalence)
- [Performance](#performance)
- [Layout](#layout)
- [Training nanochat on it](#training-nanochat-on-it)
- [Out of scope](#out-of-scope)
- [References](#references)

## The problem

1. **Pretokens.** Each document is split by the nanochat regular expression and
   mapped to the GPT-2 ByteLevel alphabet. Unique pretokens are kept with their
   frequencies, in first-occurrence order.
2. **Candidates.** Every substring of two or more bytes of every pretoken. A
   candidate's count is its occurrences weighted by pretoken frequency;
   candidates seen more than once are kept, in Python string order.
3. **Linear program.** Each pretoken of `n` bytes becomes a path graph of
   `n + 1` vertices with one unit of flow from the first vertex to the last.
   Byte edges join neighbours, and a token edge spans every candidate
   occurrence. Variables are token edges `f`, byte edges `g`, and candidate
   indicators `t`:

   ```
   minimize    w_f . f + w_g . g           (w = pretoken frequency)
   subject to  A f + B g = b               (flow conservation)
               f_e - t_{token(e)} <= 0      (edge needs its token)
               sum(t) <= K                 (vocabulary budget)
               0 <= f, g, t <= 1
   ```

   `K = V - 256 - 10`: the vocabulary size less the byte alphabet and ten
   special tokens. The flow blocks are disjoint; only `t` and the budget row
   couple them.
4. **Presolve and solve.** PSLP presolve, then PDLP (cuOpt's default Stable3
   mode: Ruiz and Pock-Chambolle scaling, a constant step from power
   iteration, reflected Halpern updates, restarts, PID primal-weight control).
5. **Rounding.** `det` keeps the `K` largest positive `t` (stable, so ties keep
   candidate order) plus the 256 bytes. `bias` ranks by `t / len`; `all_ones`
   keeps `t >= 0.99`.
6. **Export.** A Unigram model with every score equal, so encoding picks the
   segmentation with the fewest tokens.

At paper scale -- the first seven ClimbMix shards, 593,920 documents -- the
program has 1,996,113 pretokens, 8,825,612 candidates, 99,168,445 rows,
105,997,943 columns and 361.3M nonzeros.

## Upstream vs. this port

| Stage | Upstream | This port |
|---|---|---|
| Pretokenize | Python, HF `datasets.map`, 16 processes | Python with HF `tokenizers` pre-tokenizer, process pool |
| Candidates | Python | Python, process pool |
| LP arrays | Python, SciPy sparse | torch, vectorized CSR assembly |
| Model build | cuOpt Python API, one `addConstraint` call per row | None: the CSR arrays go to the solver directly |
| Presolve | PSLP 0.0.8 (C), inside cuOpt | Numba kernels for the sequential reductions; torch on the GPU for the transpose and parallel-row sorts |
| Scaling, step size | cuOpt (C++/CUDA) | torch |
| PDLP | cuOpt (C++/CUDA), cuSPARSE products | torch; Triton CSR products; `torch.compile`d fused updates |
| Postsolve | PSLP (C) | Numba, primal only |
| Rounding, export | Python, HF `tokenizers` | torch, HF `tokenizers` |
| Compiled code | cuOpt (C++/CUDA), which bundles PSLP (C) | None of its own: Numba and Triton compile the kernels at run time |
| Solver control | Concurrent race (PDLP, barrier, dual simplex); presolve capped at 60 s wall clock | PDLP only; presolve uncapped |

Two upstream behaviours were measured before the port dropped them. PDLP wins
the concurrent race on this problem family, and its result equals PDLP alone on
the presolved program. PSLP's 60 s cap does not change its result at paper
scale; the port drops the cap because PSLP reads the clock only between rounds,
so a capped result would depend on the machine.

## Equivalence

Three upstream properties bound what any port can match:

- **Exported ids are not reproducible.** Upstream orders the vocabulary by
  `list(set(...))`, which depends on the per-process string hash seed. With
  equal Unigram scores the ids do not affect segmentation, so token strings,
  not ids, are compared.
- **The published tokenizers predate the pinned revision.** Its 16K `det`
  vocabulary differs from the published one in 45 of 16,374 pieces, so the
  reference is upstream at the pinned revision, run with cuOpt 26.8.
- **Summation order moves the last bits.** Permuting the program's rows and
  columns moves cuOpt's `t` by at most 4.7e-8 at paper scale and changes no
  piece. Presolve is different: dropping it moves 129 of 16,118 `det` pieces,
  so it is part of the algorithm.

| Output | Criterion | Result at paper scale |
|---|---|---|
| Pretokens, candidates, LP arrays | Bit-identical, order included | Bit-identical |
| Presolved program, postsolve | Bit-identical to PSLP 0.0.8 | Bit-identical |
| PDLP iterates | Same stop iteration; within the noise floor | 5,600 iterations, as the paper reports; max `\|dt\|` 1.6e-12 |
| Vocabulary | Identical per rounding scheme | Identical for `det`, `bias`, `all_ones`; 16,118 `det` pieces |
| Held-out encoding | Identical token strings | 10,000 unseen documents (6.64M tokens), no difference |

The presolver repeats PSLP's floating-point operations in PSLP's order, so it
is exact for any linear program, not only ConvexTok's. It was checked step by
step against an instrumented PSLP build, and end to end on 3,000 small
ConvexTok programs, 3,000 general random programs (infinite bounds,
inequalities, parallel rows and columns, infeasible and unbounded cases) and 74
targeted edge cases. `testdata/presolve_fuzz/` keeps the 32 programs that, with
the fixture, execute every kernel line and branch the full set does.

## Performance

Paper scale, one H100, each stage's output identical to upstream's:

| Stage | Upstream | This port |
|---|---:|---:|
| Pretokenize | 207 s | 84 s |
| Candidates | 128 s | 51 s |
| LP arrays | 600-915 s | 70-104 s |
| Presolve | 53-85 s | 26.5-26.9 s |
| PDLP, 5,600 iterations | 75 s (13.4 ms each) | 105 s (13.8 ms each) |
| End to end, raw shards to `tokenizer.json` | about 1,300 s | 346 s |

Per PDLP iteration, the Triton products take 3.8 ms for `A` (cuSPARSE:
3.8 ms) and 5.4 ms for `A^T`, and the fused updates 4.5 ms. The products sum
each row in a fixed order, never by atomics, so repeated solves return the same
bits; torch's own CSR product does not. The rest of the solve's 105 s is setup:
scaling, 579 power iterations, and compiling the fused kernels once per
process.

Peak memory, paper scale, one H200, 16 workers on both sides:

| Run | Host | GPU | Wall |
|---|---:|---:|---:|
| This port | 58.4 GiB | 62.4 GiB | 296 s |
| Upstream, default path | 365.4 GiB | 115.4 GiB | 2,534 s |
| Upstream, PDLP called directly (no per-row model build) | 78.3 GiB | 75.1 GiB | 1,300 s |

On an 80 GB H100 cuOpt finishes the solve but cannot return the solution
("Memory allocation failed"). The port solves there: torch's allocation peaks
at 56.9 GiB.

## Layout

| Module | Input -> output |
|---|---|
| `pretokens.py` | texts -> unique pretokens and frequencies |
| `candidates.py` | pretokens -> kept candidates and counts |
| `program.py` | pretokens, candidates -> LP in CSR form |
| `presolver/` | LP -> reduced LP and postsolve (`core.py` Numba kernels, `bulk.py` torch sorts, `presolve.py` main loop) |
| `scaling.py` | reduced LP -> scaled LP |
| `step_size.py` | scaled matrix -> initial step (power iteration) |
| `spmv.py` | deterministic CSR products (Triton on CUDA) |
| `pdlp.py` | scaled LP -> solution (`Pdlp.Config` holds the solver constants) |
| `rounding.py` | `t`, candidates -> vocabulary (`det`, `bias`, `all_ones`) |
| `export.py` | vocabulary -> `tokenizer.json` |
| `prepare.py` | `ConvexTokPreparation`: every stage above, as one config |

`ConvexTokPreparation.Config` holds its choices as slots, not mode strings:
`presolve` (`None` solves the program as built), `solver`, and `rounding`, where
the three schemes are three functions. `solver` takes any config whose made
object maps a `LinearProgram` to a result with a `primal` tensor (one value per
column); PDLP is the default, and a different method -- an interior-point
solver, say -- drops in without touching the preparation.

Tests are CPU-only, against goldens minted from upstream on a synthetic
eight-document corpus. They cover pretokens, candidates, the LP, PSLP's reduced
programs, cuOpt's solution, the rounded vocabularies and held-out token
strings. cuOpt's PDLP iterates and initial step are minted on a smaller LP
(`pdlp_program.npz`: the corpus's first 20 pretokens and the candidates in
them), a third of the columns, which keeps every check's iterates under the
repository's 32 KB testdata limit while both solves still restart at the 200
check. GPU kernels carry a CPU reference; GPU-only checks are marked `cuda`.

The presolver's Numba kernels run as plain Python in tests (`conftest.py`'s
`kernels` fixture), so each presolver test takes about 30 ms instead of paying
a JIT compile of about 45 s per process. Only what needs machine code is
compiled, in the slow tier: one test runs every golden program through the
compiled kernels, which proves each kernel compiles and still matches PSLP, and
one checks that a kernel keeps a single specialization. The interpreted run is
also how the kernels' line coverage is measured, since coverage cannot trace
compiled code.

## Training nanochat on it

The nanochat preparation takes any tokenizer fitting in its tokenizer slot.
`prepare.donor_convextok16k` is nanochat's Unigram recipe with
`ConvexTokPreparation` in that slot, fitted on raw shards 0-6 as upstream fits,
with ten reserved IDs (16,384 in all). nanochat's `exp023` forks `exp022`,
changing only the prepared inputs.

```bash
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data priml.baselines.convextok.prepare.donor_convextok16k --directory /opt/scratch/datasets/nanochat-convextok --stage all
uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_data priml.baselines.convextok.prepare.donor_convextok16k --directory /opt/scratch/datasets/nanochat-convextok --stage train --experiment exp023 --seed 42 --run-directory /opt/scratch/runs/nanochat-exp023-42
```

H200, seeds 42-44, `exp022` controls run on the same node and code:

| Budget | `exp022` BPB | `exp023` BPB | Difference |
|---|---:|---:|---:|
| 525 s | 0.888754 | 0.890578 | +0.001824 |
| 300 s | 0.923618 | 0.925462 | +0.001844 |

ConvexTok is worse in 5 of 6 seed pairs, but 3 seeds do not settle it (paired
t of 1.4 and 2.3). It needs 0.7% fewer tokens for the reference bytes, and
both arms take the same number of steps.

## Out of scope

- Upstream's document, boundless and super training modes.
- The morphology and vocabulary-utilisation objectives. Both change only the
  edge costs, so they can follow as values.
- Multi-GPU vocabulary sweeps.

## References

- Tempus, Whittington, Schmidt, Komm, Pimentel. Tokenisation via Convex
  Relaxations. <https://arxiv.org/abs/2605.22821>
- NVIDIA cuOpt. <https://github.com/NVIDIA/cuopt>
- Lu, Peng, and Yang. cuPDLPx: A further enhanced GPU-based first-order solver
  for linear programming. <https://arxiv.org/abs/2507.14051>
- Schmidt et al. Tokenization is more than compression (PathPiece).
  <https://arxiv.org/abs/2402.18376>

[upstream]: https://github.com/JanTempus/tokenisation_lp
