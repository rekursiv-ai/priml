# Oscillator transfer benchmark 0.1

This release asks whether a forecasting recipe can use limited accurate data to
correct missing simulator physics, and whether that correction still helps on
larger displacements. It evaluates fixed recipes on new systems rather than
training one model across all systems. The task is deliberately small enough to
inspect and reproduce on a CPU.

## Evaluation

The [protocol](PROTOCOL.md) defines the task, information available to a method,
and scoring rule. `exp040` is its complete configuration. The audit has four
independently sampled systems in each of four families: combined cubic
restoration and quadratic drag, saturating restoration, sine restoration, and
position-dependent drag. The combined family holds out a composition of known
terms. The other families introduce force shapes absent from development.

Each method receives 64 observed position/velocity pairs and predicts the next
24 pairs. Each familiar and shifted test set contains 128 trajectories. The
shift changes the initial displacement range; it does not guarantee that every
subsequent observation is outside the training range. Adaptation budgets count
both fitting and validation windows: 64 + 32 or 256 + 128. Test histories are
prediction inputs, not additional adaptation targets.

The nine recipes include DLinear and a two-block history network. Patch attention
and minGRU provide sequence-model controls. Ridge and a radial kernel correct the
cheap simulator's output, while a fixed sparse force library corrects its
equations. Chronos-Bolt Tiny and Chronos-2 are zero-shot anchors, reported with
their target access made explicit. They neither fit nor select checkpoints using
the adaptation targets. Their pretraining cost is not matched to local training.

Neural fits receive 2,400 updates and select checkpoints using validation data.
This allowance followed a development-only comparison with 600 updates: longer
training lowered validation error for all four neural recipes on each of the
three development systems at the larger budget. There were 216 development fits
including that check. Optimizers may still be unconverged; this is a fixed-budget
comparison, not a claim of optimal hyperparameters for every model.

The audit plan contains 544 fits: neural recipes use three model seeds per
system and deterministic or zero-shot recipes use one. Seeds are averaged within
a system before systems enter the summary. Families receive equal weight.
Paired 95% bootstrap intervals resample systems within each fixed family; model
seeds and individual forecast cells are not counted as independent systems.

## Local audit results

The following results and verification counts are author-reported. This record
does not link an immutable raw run bundle. Independent review has not verified
the audit scores, replay matches, or release-machine validation logs against
that evidence.

The author reports that the local audit completed all 544 planned fits without
a model failure. It ran
on an Apple M4 with 16 GiB memory, using CPU execution and one Torch thread.
Python was 3.12.3, Torch 2.11.0,
and Chronos forecasting 2.3.2.
These are local results, not measurements on PRIML reference hardware.

Ridge correction had the lowest mean error at both budgets on both test splits.
At the larger budget, its shifted scaled MSE was 0.000264474, compared with
0.0135742 for DLinear, a 98.05% reduction in this cohort's mean error.
Increasing its adaptation budget reduced that error by 48.15%. These are
error reductions, not percentages of correct forecasts. The force library nearly
recovered the compositional family but was less accurate on missing force shapes.

A simple correction already handles this cohort well. This result supports using
the release as a controlled adaptation benchmark; it provides no basis for
claiming that the task defeats current models. Establishing greater difficulty
requires a new, separately evaluated cohort, not tuning this audit after seeing
the results.

The primary metric is scaled MSE, with lower values better. Brackets contain
95% system-bootstrap intervals. All four families have equal weight. The
zero-shot rows are anchors with different training access and pretraining cost,
not matched-training comparisons. Intervals are unadjusted for the number of
comparisons and should be read descriptively.

| Method | 96, familiar | 96, shifted | 384, familiar | 384, shifted |
| --- | ---: | ---: | ---: | ---: |
| dlinear | 0.0001205 [0.0001041, 0.0001344] | 0.01387 [0.008409, 0.02091] | 0.0001147 [0.0001042, 0.0001249] | 0.01357 [0.009171, 0.01897] |
| ridge | 4.698e-07 [3.995e-07, 5.456e-07] | 0.00051 [0.0003558, 0.0006867] | 1.431e-07 [1.215e-07, 1.676e-07] | 0.0002645 [0.0001832, 0.0003538] |
| rbf | 9.985e-06 [4.853e-06, 1.578e-05] | 0.0399 [0.01878, 0.06504] | 1.154e-06 [4.362e-07, 1.964e-06] | 0.02361 [0.01137, 0.037] |
| force_library | 0.0004306 [0.0003382, 0.000522] | 0.002191 [0.001686, 0.002742] | 0.0004145 [0.0003486, 0.0004834] | 0.00222 [0.001731, 0.002771] |
| history_net | 0.001053 [0.0007564, 0.001367] | 1.118 [0.5373, 1.811] | 0.0002521 [0.0002118, 0.0002924] | 0.7103 [0.4434, 1.014] |
| patch_attention | 3.204e-05 [2.535e-05, 4.263e-05] | 0.01011 [0.008498, 0.01233] | 1.424e-05 [1.326e-05, 1.521e-05] | 0.009344 [0.008318, 0.0105] |
| patch_mingru | 2.047e-05 [1.854e-05, 2.228e-05] | 0.006356 [0.005147, 0.007652] | 1.766e-05 [1.61e-05, 1.929e-05] | 0.005915 [0.004556, 0.007295] |
| chronos_bolt_tiny (zero-shot) | 0.178 [0.1453, 0.2132] | 0.5313 [0.4301, 0.6422] | 0.178 [0.1453, 0.2132] | 0.5313 [0.4301, 0.6422] |
| chronos2 (zero-shot) | 0.03374 [0.03066, 0.03714] | 0.2013 [0.1763, 0.2262] | 0.03374 [0.03066, 0.03714] | 0.2013 [0.1763, 0.2262] |

The following paired contrasts use the 384-trajectory budget and shifted
initial conditions. A negative difference favors the listed method over
DLinear; an interval spanning zero does not resolve the direction on this
cohort. The intervals condition on these four families.

| Method | Difference from DLinear | 95% paired interval |
| --- | ---: | ---: |
| ridge | -0.01331 | [-0.01864, -0.008906] |
| rbf | 0.01003 | [-0.003552, 0.02465] |
| force_library | -0.01135 | [-0.01664, -0.006867] |
| history_net | 0.6967 | [0.432, 0.9955] |
| patch_attention | -0.004231 | [-0.008969, -0.0001509] |
| patch_mingru | -0.007659 | [-0.01284, -0.003091] |
| chronos_bolt_tiny | 0.5177 | [0.4147, 0.6301] |
| chronos2 | 0.1877 | [0.1603, 0.2146] |

Family means below use the larger budget and shifted split. They make
differences between mechanisms visible rather than hiding them in one mean.

| Method | combined | coupled_drag | saturating | sine | Worst sampled system |
| --- | ---: | ---: | ---: | ---: | ---: |
| dlinear | 0.02923 | 0.005111 | 0.009383 | 0.01058 | 0.06182 |
| ridge | 0.000161 | 6.973e-06 | 0.000398 | 0.0004919 | 0.0008268 |
| rbf | 0.001001 | 0.0001151 | 0.02688 | 0.06642 | 0.1361 |
| force_library | 6.364e-16 | 0.001406 | 0.004617 | 0.002856 | 0.007173 |
| history_net | 0.6459 | 0.135 | 0.9439 | 1.116 | 2.583 |
| patch_attention | 0.002135 | 0.001672 | 0.01358 | 0.01999 | 0.02529 |
| patch_mingru | 0.001494 | 0.0004339 | 0.004495 | 0.01724 | 0.02408 |
| chronos_bolt_tiny | 0.3798 | 0.4104 | 0.5834 | 0.7517 | 1.215 |
| chronos2 | 0.156 | 0.1004 | 0.2476 | 0.3011 | 0.3451 |

All 32 system/budget diagnostics passed. Reference continuation
matched the stored targets exactly. The largest discrepancy against a
four-times-finer integration grid was 5.416e-05 scaled MSE, below
the declared 0.001 threshold. This checks numerical resolution; it does not
establish that a discrete simulator is a real physical system.

A separate reduction of the raw records reproduced all 36 reported point
estimates. Nine fits chosen by a fixed rule, the first system at the smaller
budget with model seed 2021, were rerun. Their complete returned study records
matched the original records exactly. Source and asset locks were verified
again afterward. These replays are verification, not additional independent
systems in the reported intervals.

## Reproduction and evidence

The [README](README.md#version-01-transfer-benchmark) gives staging and execution
commands. Pretrained weights are pinned to exact public revisions and checked
against their local manifests. Additional package versions and distribution
hashes are isolated in `pretrained-requirements.txt`; the project's dependency
files are unchanged. Ordinary tests need neither pretrained weights nor network
access.

The current reproduction path launches `exp040` through PRIML's standard
experiment entry point. The job writes the declared template configuration,
each attempt's actual finalized configuration, integration diagnostics, and
paired summaries. Completed attempts are
saved atomically. The Python API permits resuming with matching configuration,
default tensor dtype, and supplied provenance. A failed method has no finite
aggregate score; the native job writes the report before raising on failures.

The source and asset lock checks reported above describe the author's historical
release workflow. The bespoke freeze-and-run executable has been removed; the
current experiment does not enforce a source lock or collect installed versions.

Full run records and generated artifacts stay outside Git. The author reports
that the release evidence contains the audit lock, per-run configurations, and
every attempted fit; no immutable location for that bundle is linked here.
The benchmark's test suite checks budget nesting, training-only score scales,
target isolation, and pairing across methods. It also rejects incomplete results
and inconsistent control scores. Existing oscillator and ETTh1 numerical
regression records are retained.

## Limits and next validation

These are noiseless, fully observed, two-variable systems with known linear
physics. With Torch's default dtype, initial states, generated trajectories, and
neural models use float32. Calibration losses and direct solves use float64;
the simulator retains the dtype of its input states. The force library has a
useful prior on the compositional holdout and
shares the evaluator's integration scheme. A small integration-grid discrepancy
does not make the discrete targets exact continuous-time physics.

The equations and seeds are public. The evaluator separates target access
through its API; it is not a security boundary against an untrusted program.
Unseen mechanisms mean mechanisms excluded from local development. Similar
waveforms may have appeared in a pretrained model's training data. Architecture,
precision, and optimization differ across methods, so their comparison does not
isolate the causal effect of adding a physics prior.
There are only four sampled systems per audit family, so the intervals describe
this cohort, not all dynamical laws. Zero-shot anchors do not establish anything
about LLM reasoning, and poor forecasts here do not imply universal model failure.

Physics-assisted forecasting and evaluation under changing dynamics already
exist. This release contributes a controlled PRIML implementation and an explicit
adaptation protocol. It makes no claim that the research area is new. Useful next
validation would be independent replication and an externally chosen family of
systems, followed by a separately versioned test on real observations. Audit
results from this release must not be used to tune a recipe and claim a fresh
evaluation on the same cohort.

## Validation on the release machine

The oscillator, ETTh1, and model-cost checks passed: 537 tests, one skip, and
four deselections for other execution tiers. All three ETTh1 executable smoke
checks also passed, as did the new release executable when invoked from another
directory. Scoped pre-commit checks passed,
including the changed-source 98% coverage gate. Repository-wide ty and basedpyright
checks passed without diagnostics, and the package wheel built successfully.

These are the author's historical validation counts. The release executable
mentioned here has since been removed in favor of the native experiment launch.

A repository-wide test attempt did not pass on this laptop. All 58 observed
failures reproduced on unchanged base revision `631a0c11`, as did four collection
errors from unavailable Triton modules. The broad run was interrupted after
9,582 passes; it is not a complete passing CI run. These baseline issues were
kept separate from the benchmark work. Linux CI and independent review remain
necessary before publication.
