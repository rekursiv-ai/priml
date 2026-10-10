"""Tests for the throughput experiment configs.

``exp000`` is pinned whole by a pprint golden. It must time exactly the
pipeline the training ``exp000`` feeds its model, and compare it against
itself, or the number it reports belongs to some other recipe.
"""

from __future__ import annotations

from configgle.pprinting import pformat

from priml.baselines.imagenet import experiments
from priml.baselines.imagenet.throughput.experiments import exp000
from priml.data.pipeline.dataset import DataPipeline
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.testing.golden import assert_pprint_golden


def test_exp000_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="exp000", config=exp000())


def test_exp000_times_its_own_reference() -> None:
    cfg = exp000()
    assert pformat(cfg.pipeline) == pformat(cfg.reference)


def test_exp000_times_the_training_pipeline_on_its_own_image_set() -> None:
    timed = exp000().pipeline
    trained = experiments.exp000().dataset.train_data_pipeline
    assert isinstance(timed, DataPipeline.Config)
    assert isinstance(trained, DataPipeline.Config)
    assert isinstance(timed.source, ExtractedImageNetSource.Config)
    assert isinstance(trained.source, ExtractedImageNetSource.Config)
    assert timed.source.working_dir == "/datasets/imagenet_throughput"
    timed.source.working_dir = trained.source.working_dir
    assert pformat(timed) == pformat(trained)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
