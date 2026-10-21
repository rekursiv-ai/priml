"""Native experiments."""

from __future__ import annotations

from pathlib import Path

from configgle import PartialConfig

import torch

from priml.baselines.casmi2026.data import TrainingPreparation
from priml.baselines.casmi2026.dreams import (
    BankCheckpoint,
    MainBank,
    keep_count,
    modal_adduct,
)
from priml.baselines.casmi2026.fusion import SimulatorFallback
from priml.baselines.casmi2026.metric import MoleculeMRR
from priml.baselines.casmi2026.model import DreamsFP
from priml.baselines.casmi2026.pubchem import PubChemAssets, first_modal_adduct
from priml.baselines.casmi2026.recipe import (
    MainAssets,
    MainResource,
    PubChemResource,
    PublicRecipe,
    SecondResource,
    SimulatorStage,
)
from priml.baselines.casmi2026.second_engine import SecondAssets
from priml.baselines.casmi2026.simulators import (
    GlacierBackend,
    GlacierScheduling,
    IceBackend,
)
from priml.baselines.casmi2026.train_step import CasmiRun, casmi_loop
from priml.model.ae.checkpoint import LocalFile
from priml.runtime import SingleProcess
from priml.train.checkpointer import Checkpointer
from priml.train.tracker import FileTracker


def exp000() -> PublicRecipe.Config:
    """Preserve the complete public-best inference recipe and released checkpoints.

    Hypothesis:
      The independently native public ensemble preserves every selected stage,
      released checkpoint, candidate budget and full-test publication policy.

    References:
      https://www.kaggle.com/code/antriksharya/casmi26-lb-0-443-v7c-engine-2-rank-2-insert

    Results:
      Historical source public score: 0.443. Native qualification uses tiny
      records minted in-tree; a native full-GPU score is unmeasured.
      Released checkpoint-training recipes were not published.

    Returns:
      config: Complete native released-checkpoint recipe with CUDA inference.

    """
    cfg = PublicRecipe.Config()
    cfg.study_name = "casmi2026"
    cfg.experiment_name = "exp000"
    cfg.device = "cuda"
    cfg.dreams = "inputs/datasets/hengck23/hengck23-dreams-enveda-casmi26"

    # Fusion: reciprocal-rank votes of the legacy list and both identity policies.
    policy = cfg.policy
    policy.legacy.alpha = 0.6
    policy.legacy.offset = 3.0
    policy.legacy.window = 40
    policy.control.alpha = 0.6
    policy.control.offset = 3.0
    policy.control.window = 40
    policy.final.alpha = 0.6
    policy.final.offset = 3.0
    policy.final.window = 40
    fallback = policy.fallback = SimulatorFallback.Config()
    fallback.gap_weight = 0.5

    # Reports cite the source experiments, notebook versions and digests.
    cfg.provenance.control_experiment = "E004"
    cfg.provenance.final_experiment = "E006"
    cfg.provenance.recipe_sha256 = (
        "81ebe7b6972bdebdf3d1227d77565e35a71b4b3418b155e3035ccaaff9d53005"
    )
    cfg.provenance.notebook_sha256 = (
        "1f0f9b3026a93ddb5ad6fa6993c8ac0a4bfaddc54594eb1dd6dda7a20d8ef0d6"
    )
    cfg.provenance.source_code_sha256 = (
        "eee496a371fce179b4aeda1c5082a58212c724eaf1ca4ba0c943bbe8d7196ef2"
    )
    cfg.provenance.helper_source_sha256 = (
        "a027a69ddda2c2e14639e3ec5196d5f1eb2ab57cba10ef929723148f31fc8f34"
    )
    cfg.provenance.metric_source_sha256 = (
        "c61d163f1bb768313fd0863822952f035f343924f37e3f5e4b5883ccef7e6201"
    )
    cfg.provenance.source_version_id = 356_740_494
    cfg.provenance.source_version = 80
    cfg.provenance.ice_policy_source_version_id = 356_785_434
    cfg.provenance.ice_policy_source = (
        "samanyu1808/casmi26-iceberg-where-glacier-is-blind-lb-0-438/1"
    )

    # Main arm: the Full1 bank, then the separate A/B/D views bank.
    main = cfg.main = MainResource.Config()
    main_assets = main.assets = MainAssets.Config()
    full1 = main_assets.bank = MainBank.Config()
    full1.merged_adduct, full1.single_count = modal_adduct, keep_count
    checkpoint = BankCheckpoint.Config()
    checkpoint.path = Path(
        "inputs/datasets/ahmedberatozer/casmi26-fpnet-full1/fpnet_full1.pt",
    )
    checkpoint.sha256 = (
        "d13a3611e499d7783c4a6fce5b04363eb297dcf70f2d7d00bb66f9ff1f547123"
    )
    checkpoint.dtype_autocast = torch.bfloat16
    full1.checkpoints = [checkpoint]
    views = main_assets.views_bank = MainBank.Config()
    views.checkpoints = [BankCheckpoint.Config() for _ in range(3)]
    for checkpoint, name, sha256 in zip(
        views.checkpoints,
        ("fe_A.pt", "fe_B.pt", "fe_D.pt"),
        (
            "edccf479a2a2e53461a06bc5e1aee4341b22abb32302c4f6d2c45281000d742a",
            "6f60a22b3c3b21aa48d47328bb9523bdf07cc004bfb72966e0f9713507c24f2e",
            "8ae5bee2c978f4f01ea7cfe29b514cd3e9112006ab51aa8b13f20e26e9365362",
        ),
        strict=True,
    ):
        checkpoint.path = (
            Path("inputs/datasets/ahmedberatozer/casmi26-v4b-models") / name
        )
        checkpoint.sha256 = sha256
        checkpoint.dtype_autocast = torch.bfloat16
    views.checkpoints[2].model = DreamsFP.Config()

    # Second arm: the released single and merged checkpoints and twelve rankers.
    second = cfg.second = SecondResource.Config()
    second_assets = second.assets = SecondAssets.Config()
    merged = LocalFile.Config()
    merged.path = "inputs/datasets/prvsiyan/casmi26-fp-models-v2/fp_merged_m1.pt"
    merged.sha256 = "261529c823825cbe2028684bbf7214a95a7e631ca9f656336d577c507cce8294"
    single = LocalFile.Config()
    single.path = "inputs/datasets/prvsiyan/casmi26-fp-models-v2/fp_single_s2.pt"
    single.sha256 = "c19aa29016ed4d7a67bac38105b8fa5076c696c29eb66a504558c9b2e52b614a"
    second_assets.bank.merged = [merged]
    second_assets.bank.single = [single]
    rankers = second_assets.rankers
    rankers.own.priors, rankers.own.seeds = [0.35, 0.55], [0, 1]
    rankers.pv.priors, rankers.pv.seeds = [0.30, 0.60], [0, 1, 2, 3]
    rankers.weight_fingerprint = 0.88

    # PubChem: the two V3 checkpoints with first-occurrence merged-adduct ties.
    pubchem = cfg.pubchem = PubChemResource.Config()
    pubchem_assets = pubchem.assets = PubChemAssets.Config()
    pubchem_assets.bank.merged_adduct = first_modal_adduct
    pubchem_assets.bank.single_count = keep_count
    pubchem_assets.bank.checkpoints = [BankCheckpoint.Config() for _ in range(2)]
    for checkpoint, name, sha256 in zip(
        pubchem_assets.bank.checkpoints,
        ("fpnet_0.pt", "fpnet_1.pt"),
        (
            "edccf479a2a2e53461a06bc5e1aee4341b22abb32302c4f6d2c45281000d742a",
            "6f60a22b3c3b21aa48d47328bb9523bdf07cc004bfb72966e0f9713507c24f2e",
        ),
        strict=True,
    ):
        checkpoint.path = (
            Path("inputs/datasets/ahmedberatozer/casmi26-v3-models") / name
        )
        checkpoint.sha256 = sha256
        checkpoint.dtype_autocast = torch.bfloat16

    # Simulators: ICE keeps the stage defaults; GLACIER has its own schedule.
    ice = cfg.ice = SimulatorStage.Config()
    ice_backend = ice.backend = IceBackend.Config()
    ice.asset_seals = {
        ice_backend.generator_checkpoint: (
            "deef0451c1870d2125653bb212c9d95c5b5d9bf80156dfa9b22fedda7db49bea"
        ),
        ice_backend.intensity_checkpoint: (
            "c49e330e049d3a8b0095026ed694fcf70045c24305354853732aaa5f89d55068"
        ),
    }
    glacier = cfg.glacier = SimulatorStage.Config()
    glacier_backend = glacier.backend = GlacierBackend.Config()
    glacier.scheduling = GlacierScheduling.Config()
    glacier.budget.seconds = 4000.0
    glacier.batch_size = 32
    glacier.asset_seals = {
        glacier_backend.checkpoint: (
            "9192e4c2eeafe83317d5a2c2641e1fc4d68934488757a4faafeb66f2ab397a12"
        ),
    }
    return cfg


def train_exp000() -> CasmiRun:
    """Train a new small FPNet under a five-minute post-download wall budget.

    Hypothesis:
      A width128, two-block FPNet can learn molecule retrieval within five minutes.
      This new optimization protocol does not reconstruct released training.

    References:
      https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models

    Results:
      TBD.

    Returns:
      config: Printable dataset, train-step and real Priml loop with final evaluation.

    """
    config = CasmiRun()
    config.budget.seconds = 300.0
    loop = casmi_loop(config)
    loop.study_name = "casmi2026-training"
    loop.experiment_name = "train_exp000"
    loop.step.model.channels_hidden = 128
    loop.step.model.num_layers = 2
    loop.step.optimizer = PartialConfig(
        torch.optim.AdamW,
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    loop.step.compile = None
    loop.max_time = loop.step.train_budget_sec = 240.0
    loop.max_eval_time = 45.0
    loop.num_steps_eval = -1
    loop.eval_every_epoch = False
    loop.metrics_eval["retrieval"] = MoleculeMRR.Config()
    runtime = loop.runtime = SingleProcess.Config()
    runtime.device = "cuda"
    loop.tracker = FileTracker.Config()
    checkpointer = loop.checkpointer = Checkpointer.Config()
    checkpointer.resume = False
    return config


def train_exp001() -> CasmiRun:
    """Fork train_exp000 with a shorter peak sequence for faster optimizer updates.

    Hypothesis:
      Retaining 128 rather than 160 peaks improves five-minute retrieval by
      exchanging low-intensity detail for more complete training updates.

    References:
      https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models

    Results:
      TBD.

    Returns:
      config: train_exp000 with only its peak cap changed; run identity is bookkeeping.

    """
    config = train_exp000()
    config.loop.experiment_name = "train_exp001"
    casmi_loop(config).dataset.max_peaks = 128
    return config


def train_exp002() -> CasmiRun:
    """Fork train_exp000 and charge cold cache preparation to the shared wall budget.

    Hypothesis:
      The five-minute wall budget is honest only when cache verification and
      construction run inside it. Training and evaluation targets stay at their
      train_exp000 values, so preparation consumes the remaining post-download time.

    References:
      https://www.kaggle.com/datasets/ahmedberatozer/casmi26-v4b-models

    Results:
      The historical first cold H200 attempt exhausted 300.76 seconds during
      identity indexing, before training, MRR@25 or checkpoint export.

    Returns:
      config: train_exp000 with the cache build declared as its staging job.

    """
    config = train_exp000()
    config.loop.experiment_name = "train_exp002"
    config.preparation = TrainingPreparation.Config()
    return config
