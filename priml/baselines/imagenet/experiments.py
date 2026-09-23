r"""ImageNet experiments.

``exp000`` is the ffcv-imagenet ResNet-50 recipe at its 16-epoch budget, on a
single device: the strongest published recipe using nothing exotic. Every
later experiment forks a named parent and applies ONE change.

    exp000  ffcv-imagenet ResNet-50, 16 epochs

Prepare the data once, then launch::

    uv --quiet run --frozen python -m priml.baselines.imagenet.scripts.prepare_data
    uv --quiet run --frozen python -m priml priml.baselines.imagenet.experiments.exp000

Path fields are logical: ``dataset.working_dir`` resolves beneath the run's
``base_dir`` (``/opt/scratch``).
"""

from __future__ import annotations

from dataclasses import field

from configgle import Makes, PartialConfig

from priml.baselines.imagenet.data import NUM_TRAIN_SAMPLES, ImageNetData
from priml.baselines.imagenet.train_step import ImageNetTrainStep
from priml.data.pipeline.batching import Batcher
from priml.data.pipeline.dataset import DataPipeline
from priml.math.schedules import cyclic
from priml.metrics.topk import TopK
from priml.runtime import SingleProcess
from priml.train.train_loop import TrainLoop


class ImageNetTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[ImageNetTrainStep.Config, ImageNetData.Config],
):
    """A training loop with the ImageNet step and dataset already in place."""

    step: ImageNetTrainStep.Config = field(default_factory=ImageNetTrainStep.Config)
    """Model, optimization, schedule, and resolution policy."""

    dataset: ImageNetData.Config = field(default_factory=ImageNetData.Config)
    """The ffcv-imagenet input pipelines over the extracted archive."""


def exp000() -> ImageNetTrainLoop:
    """ffcv-imagenet ResNet-50 at 16 epochs, on one device.

    The baseline every other experiment forks. Frozen: improvements belong in
    a fork, never in an edit here.

    Hypothesis:
      SGD with momentum, label smoothing, a cyclic rate, progressive resizing
      160->192 and FixRes testing at 256 is the strongest ResNet-50 recipe
      that uses nothing exotic; ffcv-imagenet measured 73.8% top-1 at this
      budget on 8xA100. Reproduced here bit-for-bit against that script.

    Returns:
      cfg: ImageNet training loop configured with the baseline recipe.

    References:
      https://github.com/libffcv/ffcv-imagenet
      Leclerc et al. 2022. FFCV: Accelerating training by removing data
      bottlenecks.

    Results:
      TBD.

    """
    cfg = ImageNetTrainLoop()
    cfg.study_name = "imagenet"
    cfg.experiment_name = "exp000"

    epochs = 16
    train = cfg.dataset.train_data_pipeline
    assert isinstance(train, DataPipeline.Config)
    (batcher,) = (p for p in train.processors if isinstance(p, Batcher.Config))
    # Floors: ffcv drops the last batch.
    steps_per_epoch = NUM_TRAIN_SAMPLES // batcher.size
    cfg.num_steps_eval = steps_per_epoch
    cfg.max_steps = cfg.step.train_budget_steps = epochs * steps_per_epoch
    cfg.step.schedule = PartialConfig(cyclic)
    cfg.step.schedule.peak = 2 / epochs
    cfg.step.resize_start = 11 / epochs
    cfg.step.resize_end = 13 / epochs

    topk = cfg.metrics_eval["accuracy"] = TopK.Config()
    topk.k_values = [1, 5]

    cfg.runtime = SingleProcess.Config()
    return cfg


def exp_smoke() -> ImageNetTrainLoop:
    """exp000 cut to a few steps: is the data prepared and does the loop run.

    Returns:
      cfg: exp000 with a 4-step budget, evaluated once.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.max_steps = cfg.step.train_budget_steps = 4
    cfg.num_steps_eval = 4
    cfg.max_eval_time = 60.0
    cfg.eval_stop_on_time_limit = True
    cfg.checkpointer = None
    for pipeline in (cfg.dataset.train_data_pipeline, cfg.dataset.eval_data_pipeline):
        assert isinstance(pipeline, DataPipeline.Config)
        (batcher,) = (p for p in pipeline.processors if isinstance(p, Batcher.Config))
        batcher.size = 16
    return cfg
