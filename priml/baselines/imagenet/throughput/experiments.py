r"""ImageNet input pipeline throughput experiments.

``exp000`` times the ffcv-imagenet training pipeline that the training
``exp000`` (``imagenet/experiments.py``) feeds its model, and pins its
batches. Every later experiment forks a named parent and applies ONE change
to ``pipeline``.

    exp000  ffcv-imagenet training pipeline, as the training exp000 runs it

Stage the fixed image set once, then launch::

    uv --quiet run --frozen python -m priml.baselines.imagenet.throughput.scripts.prepare_throughput_data
    uv --quiet run --frozen python -m priml priml.baselines.imagenet.throughput.experiments.exp000

Path fields are logical: the image set resolves beneath ``base_dir``
(``/opt/scratch``).

Objective:
  Make ``pipeline`` produce the batches ``exp000`` produces on less CPU.

Score:
  Images per CPU-second, the median over ``num_repeats`` passes. Each pass
  runs in a fresh spawned process and is charged for the CPU of its whole
  process tree, from before the pipeline's config is unpickled to the last
  batch, construction and the first batch included (``throughput.py`` says
  exactly what is and is not counted). Wall images/sec and first-batch
  latency are logged beside it and not scored: thread count alone moves wall
  time, and on a shared machine it is the noisier number.

Correctness:
  Exact (the default): every batch's ``image`` and ``label`` tensors equal
  the ones ``reference`` makes from the same image set under the same seed,
  bit for bit; a mismatch names the batch and field, how many values moved,
  the largest difference, and the first index. The tensor golden in
  ``throughput_test.py`` pins what ``reference`` itself computes. A
  ``PixelTolerance`` admits bounded drift in ``image`` only, labels staying
  exact; a run under one says so in every log line, and its number is not
  comparable with an exact one.

A fork may change:
  Anything inside ``pipeline``: decoders, processor order, threads,
  processes, buffers, batching internals, as long as every batch arrives in
  order, holds the same bytes, and stays valid once yielded.

A fork may not change:
  The output bytes beyond its tier, the image set, the batch order,
  ``reference`` or the code it runs, the seed, or the harness and its timing.
  Nor may it keep state outside its pass (files written in one pass and read
  in a later one, work done ahead of the run), leave processes running after
  its pass, or read anything prepared offline from the image set.

Out of scope:
  GPU decode (nvjpeg, DALI). It moves work off the CPU, so this score would
  reward it for free, and whether it costs the train step GPU time is
  something only a run of the train step can show. It belongs in a chain
  scored on train step time with the real model, not in this one.
"""

from __future__ import annotations

from priml.baselines.imagenet.throughput.throughput import LoaderThroughput
from priml.data.pipeline.dataset import DataPipeline
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource


def exp000() -> LoaderThroughput.Config:
    """Time the ffcv-imagenet training pipeline over a fixed image set.

    The control every pipeline experiment forks. Frozen: improvements belong
    in a fork, never in an edit here.

    Hypothesis:
      The training exp000's pipeline, unchanged (512-image batches of
      random-resized 160x160 crops, flipped, decoded on two threads with the
      fast IDCT), is the throughput every faster pipeline must beat while
      producing identical batches.

    References:
      https://github.com/libffcv/ffcv-imagenet
      Leclerc et al. 2022. FFCV: Accelerating training by removing data
      bottlenecks.

    Results:
      TBD.

    """
    cfg = LoaderThroughput.Config()
    for pipeline in (cfg.pipeline, cfg.reference):
        assert isinstance(pipeline, DataPipeline.Config)
        assert isinstance(pipeline.source, ExtractedImageNetSource.Config)
        # Where ``prepare_throughput_data`` stages the timed image set.
        pipeline.source.working_dir = "/datasets/imagenet_throughput"
    return cfg
