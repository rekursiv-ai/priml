"""openai/math experiment configurations.

``exp000`` trains Qwen3.5-4B from live openai/math rollouts. ``exp_smoke`` runs
one small CPU update from a saved fixture.

The learner is the TMax baseline's, imported rather than re-derived: DPPO's
objective is a function of tokens, masks, behavior logprobs, and advantages, and
none of those know what task produced them. What this substrate changes is the
data and the rollout loop, which live in this baseline's own modules.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

from configgle import Makes

from priml.baselines.openaimath.data import OpenMathRolloutData
from priml.baselines.openaimath.live import LiveOpenMathRolloutData
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


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    # ~1050 ms; only launching an experiment resolves it.
    torch = lazy_import("torch")


class OpenMathTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[TMaxDPPOTrainStep.Config, OpenMathRolloutData.Config],
):
    """Configure PriML training with openai/math rollout data."""

    step: TMaxDPPOTrainStep.Config = field(
        default_factory=TMaxDPPOTrainStep.Config,
    )
    """Qwen3.5 model and DPPO objective, shared with the TMax baseline."""

    dataset: OpenMathRolloutData.Config = field(
        default_factory=OpenMathRolloutData.Config,
    )
    """openai/math rollouts and packed training rows."""

    @override
    def finalize(self) -> Self:
        """Fill the data and rollout settings this loop owns, where unset.

        Pure propagation: an explicit value on the dataset or runner wins, and a
        value that cannot work is refused by the component that uses it (the
        dataset's ``bind_step`` checks the pad id and temperature; the runner's
        constructor refuses any per-run setting still unset).
        """
        if isinstance(self.working_dir, str):
            run_working_dir: Path | str = self.working_dir.format(
                study_name=self.study_name,
                experiment_name=self.experiment_name,
            )
        else:
            run_working_dir = self.working_dir
        # The runner receives logical paths and resolves them against its own
        # base_dir, so it gets the model path as configured, not as resolved.
        model_path = self.step.model_path
        if model_path is not None:
            self.step.model_path = resolve_working_dir(self.base_dir, model_path)
        if self.dataset.pad_token_id is None:
            self.dataset.pad_token_id = self.step.pad_token_id
        if isinstance(self.dataset, LiveOpenMathRolloutData.Config):
            runner = self.dataset.runner
            if self.dataset.base_dir is None:
                self.dataset.base_dir = resolve_working_dir(
                    self.base_dir,
                    run_working_dir,
                )
            if runner.base_dir is None:
                runner.base_dir = self.base_dir
            if runner.working_dir is None:
                runner.working_dir = run_working_dir
            if runner.output_dir is None:
                runner.output_dir = self.dataset.working_dir
            if runner.checkpoint is None:
                runner.checkpoint = model_path
            if runner.num_updates is None and isinstance(
                self.step.train_budget_steps,
                int,
            ):
                runner.num_updates = self.step.train_budget_steps
            if runner.seed is None:
                runner.seed = self.seed
        return super().finalize()


def exp000() -> OpenMathTrainLoop:
    """Train Qwen3.5-4B with TMax's DPPO recipe on openai/math lemma tasks.

    trax experiment 4

    The campaign's root and frozen control. The recipe is the TMax 4B one,
    changed only where the substrate forces it: a packed row holds one Lean
    episode, and rollouts are Harbor trials graded by the dataset's own
    Comparator grader. Later experiments fork this one, one change each.

    Hypothesis:
      Lemma tasks give a 4B policy a reward it can earn. A whole challenge's
      reference proof spans 67 modules at the median, so training on the
      challenges themselves scores zero everywhere, while a lemma is one step
      of such a proof, graded the same way. DPPO on lemmas from the training
      families' modules should raise pass@1 on lemmas from the held-out
      families' modules, which training never sees (``lemma_tasks`` defines
      the split). Whether any of that reaches the held-out challenges is open.

      Measured with ``scripts/eval.py`` on saved checkpoints: the progress
      metric is pass@1 over a fixed 256-task sample of ``lemmas/test``; the
      protected score is pass@k over the held-out families' published
      challenges (``official_test_tasks`` in ``lemmas/split.json``), run at the
      start and end only, never to choose between forks. The budget is 500
      updates of 8 prompts times 32 attempts, at least 128,000 graded episodes.
      Every graded attempt, dropped groups included, gets a line in the attempt
      log beside the rollout shards with its ``sandbox_seconds`` and
      ``grade_seconds``, so the Lean cost is summed from that log beside the
      GPU hours.

    Returns:
      config: Training with live openai/math rollouts.

    References:
      https://huggingface.co/datasets/FineEnvs/openai-math (tasks and grader)
      https://github.com/openai/math (statements and the lemma library)
      https://arxiv.org/abs/2602.04879 (the DPPO objective)

    Results:
      TBD.

    """
    config = OpenMathTrainLoop()
    config.study_name = "openaimath"
    config.experiment_name = "exp000"
    config.step.model_path = "/models/Qwen3.5-4B"
    config.step.model = qwen35_4b_config()
    config.step.pad_token_id = 248_044
    config.step.parallelism = FullySharded.Config()
    config.step.parallelism.mp_param_dtype = torch.bfloat16
    # Limit memory use for 82k-token packed rows.
    config.step.activation_memoization = SelectiveActivationCheckpointing.Config(
        module_types=(TransformerBlock,),
    )
    config.step.head_chunk_size = 2_048
    config.step.fp32_head = True
    config.step.export_dir = "/export"
    live_data = LiveOpenMathRolloutData.Config()
    live_data.working_dir = "/rollouts"
    # One 8-GPU node, split in half: four learner processes hold GPUs 0-3
    # (launch with --nproc_per_node=4), and four single-GPU vLLM engines hold
    # 4-7. The learner's weights and optimizer state stay resident while the
    # server samples, so the two cannot share a device; a 4B policy fits one
    # GPU, so the server's half goes to data parallelism, not tensor sharding.
    live_data.runner.rollout_gpus = (4, 5, 6, 7)
    live_data.runner.data_parallel_size = 4
    config.dataset = live_data
    config.max_epochs = 1
    config.step.train_budget_steps = 500
    # Save every 10 updates, retaining the latest 3 and every 50th. One update
    # costs a full round of sandboxes and a rollout-server restart, so losing
    # more than a few to an interruption would cost hours.
    if config.checkpointer is None:
        raise ValueError("openai/math exp000 requires checkpointing.")
    checkpointer = cast(Checkpointer.Config, config.checkpointer)
    checkpointer.save_every = 10
    checkpointer.keep_last_n = 3
    checkpointer.keep_every = 50
    config.num_steps_eval = 0
    config.eval_every_epoch = False
    config.runtime = MultiProcess.Config(process_group_timeout_sec=72_000)
    return config


def exp_smoke() -> OpenMathTrainLoop:
    """Run one small CPU update to check that openai/math training works.

    The fixture holds one prompt group of two Lean-shaped episodes: one accepted
    proof and one rejected, so the centered advantage is nonzero and the update
    produces a gradient. Its logprobs are the tiny seeded model's own, written by
    ``scripts/regenerate_smoke_logprobs.py``, so the first update's importance
    ratio starts near 1.

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
    offline_data = OpenMathRolloutData.Config()
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
