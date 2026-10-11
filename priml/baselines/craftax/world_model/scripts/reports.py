"""The JSON each evaluation step writes, typed once for ``report.py`` to read.

``report.py`` reads each with ``json_loads(path.read_text(), Schema)``, so a
field is named and typed in one place rather than re-narrowed from
``dict[str, object]`` at each reader. Fields a step writes that ``report.py``
never reads are left undeclared; ``json_loads`` drops them.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict


class BaselinesReport(TypedDict):
    """``baselines-NAME.json``: ``baselines.report`` of the validation batches."""

    beats: dict[str, bool]
    model_nll: dict[str, float]
    empirical_nll: dict[str, float]
    cell_accuracy: dict[str, float]


class EngineChecks(TypedDict):
    """``engine-checks-NAME.json``: what ``report.py`` reads of the checks."""

    all_passed: bool


class DreamReport(TypedDict):
    """``dreams-NAME/report.json``: what ``report.py`` reads of a dream run."""

    distance: dict[str, float]
    run: dict[str, object]


class TilesRun(TypedDict):
    """How ``report.py`` scored a checkpoint's tiles."""

    checkpoint: str
    experiment: str
    overrides: list[str]
    corpus: str
    t_g: int
    seed: int
    tiles_available: int
    seconds: float


class TilesReport(TypedDict):
    """``data-eval-NAME.json``: ``data_eval.score`` of the common tiles."""

    metric: dict[str, float]
    decisions: int
    run: NotRequired[TilesRun]


class Trend(TypedDict):
    """One modality's validation bits per byte over a training run."""

    decreased: bool
    monotone: bool


class Timing(TypedDict, total=False):
    """What ``report.py`` reads of ``curves.timing``."""

    dt_median_seconds: float | None
    eval_total_seconds: float | None
    runtime_seconds: float | None


class TrainingRun(TypedDict):
    """One run of ``wandb-report.json``: ``curves.training_report`` and its id."""

    run: str
    last_step: int | None
    eval_steps: list[int]
    loss_finite: bool
    grad_norm_finite: bool
    unrecovered_spikes: int
    spikes: list[object]
    logged: dict[str, int]
    grad_norm_first_last_max: list[float] | None
    clip_fraction_last: float | None
    max_attention_logit_max: float | None
    modalities: dict[str, Trend]
    timing: NotRequired[Timing]


class WandbReport(TypedDict):
    """``wandb-report.json``: each checkpoint's training run."""

    runs: list[TrainingRun]


class Settings(TypedDict):
    """``settings.json``: the flags a run was given, as much as ``--summarize`` reads."""

    names: list[str]
    sampler_seeds: str
    windows: list[int]
