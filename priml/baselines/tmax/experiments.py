"""TMax experiment configurations.

``exp000`` trains Qwen3.5-4B from live TMax rollouts. ``exp_smoke`` runs one
small CPU update from a saved fixture.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import Self, cast, override

from configgle import Makes

import torch

from priml.baselines.tmax.data import TMaxRolloutData
from priml.baselines.tmax.live import LiveTMaxRolloutData
from priml.baselines.tmax.train_step import (
    TMaxDPPOTrainStep,
    qwen35_4b_config,
    tiny_qwen35_config,
)
from priml.model.transformer.block import TransformerBlock
from priml.paths import resolve_working_dir
from priml.runtime import MultiProcess, SingleProcess
from priml.train.activation import (
    DefaultActivationStorage,
    SelectiveActivationCheckpointing,
)
from priml.train.checkpointer import Checkpointer
from priml.train.parallelism import FullySharded, NoParallel
from priml.train.train_loop import TrainLoop


class TMaxTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[TMaxDPPOTrainStep.Config, TMaxRolloutData.Config],
):
    """Configure PriML training with TMax rollout data."""

    step: TMaxDPPOTrainStep.Config = field(
        default_factory=TMaxDPPOTrainStep.Config,
    )
    """Qwen3.5 model and DPPO objective."""

    dataset: TMaxRolloutData.Config = field(
        default_factory=TMaxRolloutData.Config,
    )
    """TMax rollouts and packed training rows."""

    @override
    def finalize(self) -> Self:
        """Share run settings with the live rollout runner."""
        if isinstance(self.working_dir, str):
            run_working_dir: Path | str = self.working_dir.format(
                study_name=self.study_name,
                experiment_name=self.experiment_name,
            )
        else:
            run_working_dir = self.working_dir
        run_dir = resolve_working_dir(self.base_dir, run_working_dir)
        model_path = self.step.model_path
        if model_path is not None:
            self.step.model_path = resolve_working_dir(self.base_dir, model_path)
        if self.step.pad_token_id is not None:
            if self.dataset.pad_token_id not in (None, self.step.pad_token_id):
                raise ValueError(
                    "dataset.pad_token_id must match step.pad_token_id.",
                )
            self.dataset.pad_token_id = self.step.pad_token_id
        if isinstance(self.dataset, LiveTMaxRolloutData.Config):
            runner = self.dataset.runner
            if model_path is None:
                raise ValueError("Live TMax requires step.model_path.")
            if self.dataset.base_dir not in (None, run_dir):
                raise ValueError(
                    "Live TMax dataset.base_dir must inherit the experiment "
                    "working directory.",
                )
            self.dataset.base_dir = run_dir
            if runner.base_dir is None:
                runner.base_dir = self.base_dir
            runner.working_dir = run_working_dir
            runner.checkpoint = Path(model_path)
            runner.output_dir = Path(self.dataset.working_dir)
            if not isinstance(self.step.train_budget_steps, int):
                raise TypeError(
                    "Live TMax requires an integer step.train_budget_steps.",
                )
            runner.num_updates = self.step.train_budget_steps
            if self.seed is None:
                raise ValueError("Live TMax requires an explicit seed.")
            runner.seed = self.seed
            runner.temperature = self.step.temperature
            runner.lm_head_fp32 = self.step.fp32_head
            self.dataset.records_per_update = runner.records_per_update
            self.dataset.num_samples_per_prompt = runner.samples_per_prompt
            if self.dataset.active_sampling != runner.active_sampling:
                raise ValueError(
                    "Live TMax dataset.active_sampling must match "
                    "dataset.runner.active_sampling.",
                )
            if self.dataset.filter_zero_std_samples != runner.filter_zero_std_samples:
                raise ValueError(
                    "Live TMax dataset.filter_zero_std_samples must match "
                    "dataset.runner.filter_zero_std_samples.",
                )
        return super().finalize()


def exp000() -> TMaxTrainLoop:
    """Configure the published TMax Qwen3.5-4B DPPO recipe.

    Hypothesis:
      PriML's DPPO implementation matches TMax closely enough to provide a
      trusted baseline for later experiments.

    Returns:
      config: Training with live TMax rollouts.

    References:
      https://arxiv.org/abs/2606.23321 (the TMax paper)
      https://github.com/hamishivi/tmax/tree/6d3d606
        training/open-instruct/scripts/tmax/RL/qwen35_4b.sh
      https://arxiv.org/abs/2602.04879 (the DPPO objective)

    Results:
      TBD. Source parity covers packed data, logits, loss, gradients, and five
      updates. The full training and evaluation run has not been completed.

    """
    config = TMaxTrainLoop()
    config.study_name = "tmax"
    config.experiment_name = "exp000"
    # Match the released recipe.
    config.seed = 42
    config.step.model_path = "/models/Qwen3.5-4B"
    config.step.model = qwen35_4b_config()
    config.step.pad_token_id = 248_044
    config.step.parallelism = FullySharded.Config()
    config.step.parallelism.mp_param_dtype = torch.bfloat16
    # Limit memory use for 67k-token packed rows.
    config.step.activation_memoization = SelectiveActivationCheckpointing.Config(
        module_types=(TransformerBlock,),
    )
    config.step.head_chunk_size = 2_048
    config.step.fp32_head = True
    config.step.export_dir = "/export"
    # Keep fresh 4B rollouts separate from the published 9B fixture.
    live_data = LiveTMaxRolloutData.Config()
    live_data.working_dir = "/rollouts"
    config.dataset = live_data
    # One epoch is 128,000 / (8 prompts * 32 samples) = 500 updates.
    config.max_epochs = 1
    config.step.train_budget_steps = 500
    # Save every 10 updates, retaining the latest 3 and every 50th.
    if config.checkpointer is None:
        raise AssertionError("TMax exp000 requires checkpointing.")
    checkpointer = cast(Checkpointer.Config, config.checkpointer)
    checkpointer.save_every = 10
    checkpointer.keep_last_n = 3
    checkpointer.keep_every = 50
    config.num_steps_eval = 0
    config.eval_every_epoch = False
    config.runtime = MultiProcess.Config(process_group_timeout_sec=72_000)
    return config


def exp_smoke() -> TMaxTrainLoop:
    """Run one small CPU update to check that TMax training works.

    The fixture contains two short responses with different rewards. Their
    logprobs were recomputed with the seeded tiny model, giving an initial mean
    importance ratio near 1 and a nonzero gradient.

    Not a result.
    """
    config = exp000()
    config.experiment_name = "exp_smoke"
    config.runtime = SingleProcess.Config()
    # The fixture logprobs were computed with this seeded tiny model.
    config.seed = 0
    config.step.parallelism = NoParallel.Config()
    config.step.activation_memoization = DefaultActivationStorage.Config()
    config.step.head_chunk_size = None
    config.step.fp32_head = False
    config.step.model_path = None
    config.step.model = tiny_qwen35_config(vocab_size=248_576)
    config.step.pad_token_id = 0
    offline_data = TMaxRolloutData.Config()
    fixture = Path(__file__).resolve().parent / "testdata" / "smoke_rollout.jsonl"
    offline_data.base_dir = fixture.parent
    offline_data.working_dir = fixture.name
    config.dataset = offline_data
    # Two rewards are needed for a nonzero centered advantage.
    config.dataset.records_per_update = 2
    config.dataset.num_samples_per_prompt = 2
    config.dataset.mask_tool_use = False
    config.max_steps = config.step.train_budget_steps = 1
    config.checkpointer = None
    # The smoke run does not save checkpoints or export a model.
    config.step.export_dir = None
    return config
