"""Test support shared by the evaluation runners' tests: fixtures and tiny configs.

The runner goldens in ``testdata/`` replay frozen evaluation trajectories.
Every case runs one real runner end to end at tiny size on CPU. The pipeline
cases (exp008, exp012) train their generators and verifiers inside the case,
segmented, screened, and triggered as the recipes do. The evaluation cases
read those same trained fixtures from ``testdata/fixtures.pt``: training is
the pipelines' code path, so re-running it per evaluation case covers no new
line and costs most of the case's time. Each records every array its runner
writes, its numeric metrics, and the trained weights, compared with
``torch.equal``.

Only goldens covering a code path no other test reaches are kept: the
pipelines (``exp008``, ``exp012``), the staircase sieve (exp011), HPS (exp007),
the agreement lock, and the denoising fit. Most were recorded from the
reference implementation this baseline was ported from; ``staircase`` and
``denoising_fit`` are snapshots of this port, minted after the pieces they
compose matched the reference bit for bit.

A test that needs the dataset fixture, a tiny generator, or a runner configured
for it imports it from here rather than from another test module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import json

from torch import Tensor

import numpy as np
import torch

from priml.baselines.sudoku import experiments
from priml.baselines.sudoku.harvest import Harvest, write_npz
from priml.baselines.sudoku.hypotheses import HypothesisRecursion, WinnerRecompute
from priml.baselines.sudoku.latent_recursion import LatentRecursion, recipe_block
from priml.baselines.sudoku.runner import file_sha256, load_eval_weights
from priml.baselines.sudoku.search import HpsSearch
from priml.baselines.sudoku.verifier import SudokuVerifier
from priml.baselines.sudoku.verifier_fit import VerifierFit, VerifierLoss
from priml.model.swiglu import SwiGLU
from priml.paths import resolve_working_dir
from priml.runtime import SingleProcess
from priml.testing import regenerate
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import read_tensors, stored


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.lib.npyio import NpzFile

    from priml.baselines.sudoku.train_loop import SudokuTrainLoop


_CWD: Final = Path(__file__).resolve().parent

FIXTURES: Final = _CWD / "testdata" / "fixtures.pt"
"""The evaluation goldens' trained generators, verifiers, and harvest shard."""


def write_dataset(root: Path) -> Path:
    """Write a solvable-shaped fixture: 3 train groups x 3 views, 2 test rows.

    Inputs keep ~35% of the solution as givens and blank the rest, and cell 0 is
    always blank, so harvest unstuck starts and search pins have cells to act
    on.

    Args:
      root: Dataset root; ``train/`` and ``test/`` are written below.

    Returns:
      root: The dataset root.

    """
    rng = np.random.default_rng(0)
    for split, rows, groups in (
        ("train", 9, np.arange(0, 10, 3)),
        ("test", 2, np.arange(3)),
    ):
        labels = rng.integers(2, 11, (rows, 81))
        inputs = np.where(rng.random((rows, 81)) < 0.35, labels, 1)
        inputs[:, 0] = 1
        directory = root / split
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "all__inputs.npy", inputs.astype(np.uint8))
        np.save(directory / "all__labels.npy", labels.astype(np.uint8))
        np.save(directory / "all__group_indices.npy", groups.astype(np.int32))
        (directory / "dataset.json").write_text('{"vocab_size": 11, "seq_len": 81}')
    return root


def replay(
    record: Callable[[Harness], dict[str, Tensor]],
    scratch: Path,
    *,
    train: bool = True,
) -> dict[str, Tensor]:
    """Build the fixture, run ``record`` under host-agnostic numerics, store it.

    Args:
      record: One golden's recorder.
      scratch: Empty directory owned by this call.
      train: Train the fixtures ``record`` asks for; otherwise read the stored ones.

    Returns:
      trajectory: Flat name-to-tensor record.

    """
    write_dataset(scratch / "data")
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(0)
        harness = Harness(scratch, train=train)
        return {k: stored(v) for k, v in record(harness).items()}


def mint_fixtures(
    recorders: Mapping[str, Callable[[Harness], dict[str, Tensor]]],
    scratch: Path,
) -> dict[str, Tensor]:
    """Train every fixture ``recorders`` reads and return them, as stored.

    Args:
      recorders: The goldens that read :data:`FIXTURES`.
      scratch: Empty directory owned by this call.

    Returns:
      fixtures: The trained weights and harvest shard, keyed as stored.

    """
    trained: dict[str, Tensor] = {}
    for case, record in recorders.items():
        harness_dir = scratch / case
        write_dataset(harness_dir / "data")
        with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
            torch.manual_seed(0)
            # A fixture an earlier case trained is read back, not retrained:
            # every case trains it from the same seed, so the bits agree.
            harness = Harness(harness_dir, train=True, known=trained)
            record(harness)
        trained |= harness.trained
    # The shard is replayed as written, dtypes and all, so it is never narrowed.
    return {
        k: v.detach().clone() if k.startswith("harvest/") else stored(v)
        for k, v in trained.items()
    }


def load_golden(case: str) -> dict[str, Tensor]:
    """Load the frozen golden ``testdata/<case>.pt``."""
    return read_tensors(_CWD / "testdata" / f"{case}.pt")


def weight_tensors(prefix: str, state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Return every trained tensor under ``prefix``, whole."""
    return {f"{prefix}/{k}": v.detach().clone() for k, v in state.items()}


def metric_tensors(values: Mapping[str, object]) -> dict[str, Tensor]:
    """Numeric metrics as float64 tensors; timings and labels are dropped."""
    return {
        f"metrics/{key}": torch.tensor(float(value), dtype=torch.float64)
        for key, value in values.items()
        if isinstance(value, int | float) and "seconds" not in key
    }


def tiny_recursion_config() -> LatentRecursion.Config:
    """Return the smallest TRM architecture the eval runners accept.

    Returns:
      config: Width 4, one layer, one slow and one fast cycle, no compile.

    """
    cfg = LatentRecursion.Config()
    cfg.vocab_size = 11
    cfg.puzzle_grid_shape = (81,)
    cfg.pos2d_grid_shape = (9, 9)
    cfg.pos2d_box_shape = (3, 3)
    cfg.channels_in = 4
    cfg.num_heads = 2
    cfg.num_layers = 1
    cfg.slow_cycles = 1
    cfg.fast_cycles = 1
    cfg.num_puzzle_identifiers = 0
    cfg.compile = False
    cfg.dtype = None
    cfg.block = recipe_block()
    assert isinstance(cfg.block.ffn, SwiGLU.Config)
    cfg.block.ffn.expansion = 1
    cfg.block.ffn.round_to = 1
    return cfg


def tiny_recursion() -> LatentRecursion:
    """Build a real CPU TRM at the smallest size the eval runners accept."""
    return tiny_recursion_config().make().eval()


def sudoku_solution() -> Tensor:
    """Return one valid 9x9 solution as 81 digit tokens (2-10)."""
    return torch.tensor(
        [((row * 3 + row // 3 + col) % 9) + 2 for row in range(9) for col in range(9)],
    )


class Harness:
    """Configurations of the runners, rooted at ``scratch``.

    The fixture methods (:meth:`generator`, :meth:`hypothesis_generator`,
    :meth:`harvest`, :meth:`verifier`) train when ``train`` is set and
    otherwise write the weights or shard :data:`FIXTURES` stores for that call,
    to the paths training would have written. Either way the caller reads the
    same files; ``--regenerate-b4b`` re-records them from a training run.
    Under ``train``, a fixture already in ``known`` is written from there instead.
    """

    def __init__(
        self,
        scratch: Path,
        *,
        train: bool = True,
        known: Mapping[str, Tensor] | None = None,
    ) -> None:
        self.scratch = scratch
        self.data = scratch / "data"
        self.train = train or regenerate.b4b()
        self.fixtures: Mapping[str, Tensor] = (
            (known or {}) if self.train else read_tensors(FIXTURES)
        )
        self.trained: dict[str, Tensor] = {}

    def model(self) -> LatentRecursion.Config:
        """Return the generator architecture.

        Returns:
          config: The tiny TRM with a two-token puzzle prefix.

        """
        cfg = LatentRecursion.Config()
        cfg.vocab_size = 11
        cfg.puzzle_grid_shape = (81,)
        cfg.pos2d_grid_shape = (9, 9)
        cfg.pos2d_box_shape = (3, 3)
        cfg.channels_in = 4
        cfg.num_heads = 2
        cfg.num_layers = 1
        cfg.slow_cycles = 1
        cfg.fast_cycles = 1
        cfg.puzzle_emb_len = 2
        cfg.compile = False
        cfg.dtype = None
        cfg.block = recipe_block()
        assert isinstance(cfg.block.ffn, SwiGLU.Config)
        cfg.block.ffn.expansion = 1
        cfg.block.ffn.round_to = 1
        return cfg

    def hypothesis_model(self) -> HypothesisRecursion.Config:
        """Return the staircase generator architecture.

        Width 12 with an FFN of hidden width 4: hypothesis alignment projects
        on the FFN's top-10 subspace, which is the whole space at width 10 or
        less and divides by a zero perpendicular norm there.

        Returns:
          config: The three-hypothesis staircase generator.

        """
        cfg = HypothesisRecursion.Config().update(self.model())
        cfg.channels_in = 12
        cfg.num_hypotheses = 3
        if cfg.block is None:
            raise ValueError("Expected cfg.block is not None.")
        assert isinstance(cfg.block.ffn, SwiGLU.Config)
        cfg.block.ffn.round_to = 4
        cfg.block.ffn.channels_hidden = 4
        return cfg

    def hypothesis_generator(self, name: str, seed: int) -> str:
        """Train a three-step recomputed-winner generator; return its checkpoint.

        Args:
          name: Run identity.
          seed: Training seed.

        Returns:
          checkpoint: Logical path of the final checkpoint.

        """
        cfg = self._recipe(name, seed=seed)
        cfg.step.model = self.hypothesis_model()
        cfg.step.routing = WinnerRecompute.Config()
        cfg.dataset.batch_size = 3
        logical = f"/runs/{cfg.study_name}/{name}/checkpoints/step_00000003.pt"
        self._checkpoint(name, self.scratch / logical.removeprefix("/"), recipe=cfg)
        return logical

    def search(self) -> HpsSearch.Config:
        """Return the search policy.

        Returns:
          config: A one-pin, two-candidate search on two ACT steps.

        """
        cfg = HpsSearch.Config()
        cfg.max_act_steps = 2
        cfg.acceptance_checkpoints = (1,)
        cfg.dtype_autocast = None
        cfg.search_depth = 1
        cfg.search_candidates = 2
        cfg.search_cell_attempts = 1
        cfg.search_budget = 2
        cfg.search_max_rows = 16
        return cfg

    def verifier_model(self) -> SudokuVerifier.Config:
        """Return the verifier architecture."""
        cfg = SudokuVerifier.Config()
        cfg.channels_hidden = 4
        cfg.num_layers = 1
        cfg.heads = 2
        return cfg

    def generator(self, name: str, seed: int) -> str:
        """Train a three-step generator; return its logical checkpoint path.

        Args:
          name: Run identity.
          seed: Training seed.

        Returns:
          checkpoint: Logical path of the final checkpoint.

        """
        cfg = self._recipe(name, seed=seed)
        cfg.step.model = self.model()
        cfg.dataset.batch_size = 2
        logical = f"/runs/{cfg.study_name}/{name}/checkpoints/step_00000003.pt"
        self._checkpoint(name, self.scratch / logical.removeprefix("/"), recipe=cfg)
        return logical

    def harvest(self, checkpoint: str) -> Path:
        """Harvest a tiny corpus from ``checkpoint``; return its directory.

        Args:
          checkpoint: Logical path of the source generator.

        Returns:
          harvest_dir: The shard and manifest directory.

        """
        cfg = Harvest.Config()
        cfg.base_dir = self.scratch
        cfg.harvest_source_checkpoint = checkpoint
        cfg.model = self.model()
        cfg.working_dir = "/data"
        cfg.group_count = 3
        cfg.views_per_group = 1
        cfg.max_act_steps = 2
        cfg.checkpoints = (1,)
        cfg.search_depth = 1
        cfg.search_candidates = 2
        cfg.search_cell_attempts = 1
        cfg.search_budget = 2
        cfg.random_corruption_strengths = (1,)
        cfg.random_starts_per_strength = 1
        cfg.dtype_autocast = None
        cfg.device = "cpu"
        if self.train and not self._stored("harvest/"):
            out = cfg.make().run()
            with cast("NpzFile", np.load(out / "shard-00000.npz")) as shard:
                self.trained |= {
                    f"harvest/{key}": torch.from_numpy(np.array(shard[key]))
                    for key in shard.files
                }
            return out
        return self._stored_harvest(cfg)

    def verifier(
        self,
        seed: int,
        *,
        model: SudokuVerifier.Config | None = None,
        loss: VerifierLoss.Config | None = None,
    ) -> Path:
        """Fit one tiny committee member on the harvest; return its checkpoint.

        Args:
          seed: Member seed.
          model: Verifier architecture; None is :meth:`verifier_model`.
          loss: Verifier objective; None keeps the fit's default.

        Returns:
          checkpoint: The final checkpoint path.

        """
        cfg = VerifierFit.Config()
        cfg.experiment_name = f"verifier_s{seed}"
        cfg.seed = seed
        cfg.base_dir = self.scratch
        cfg.model = model or self.verifier_model()
        if loss is not None:
            cfg.loss = loss
        cfg.dataset.working_dir = "/data"
        cfg.dataset.harvest_dir = "/runs/harvest/harvest"
        cfg.dataset.batch_size = 2
        cfg.dataset.steps_per_epoch = 1
        cfg.dataset.train_group_end = 1
        cfg.device = "cpu"
        cfg.dtype_autocast = None
        cfg.max_steps = 3
        if loss is not None:
            # A custom objective is the fit under test, so it always trains.
            return cfg.make().run()
        path = self.scratch / "runs" / cfg.experiment_name / "checkpoints"
        path /= "step_00000003.pt"
        self._checkpoint(cfg.experiment_name, path, recipe=cfg)
        return path

    def dumps(self, name: str) -> dict[str, Tensor]:
        """Every array in every npz under a run.

        A packed search ``rows`` array is split at its grid columns: the
        leading score columns stay float32 while the token grids narrow to
        bytes, which is most of a golden's size.

        Args:
          name: Run identity.

        Returns:
          arrays: Each array keyed by its file and field.

        """
        out: dict[str, Tensor] = {}
        root = self.scratch / "runs" / name
        for path in sorted(root.rglob("*.npz")):
            with cast("NpzFile", np.load(path)) as archive:
                for key in archive.files:
                    array = torch.from_numpy(np.array(archive[key]))
                    prefix = f"{path.relative_to(root)}/{key}"
                    if key == "rows":
                        grids = 2 * 81
                        out[f"{prefix}/scores"] = array[:, :-grids]
                        out[f"{prefix}/grids"] = array[:, -grids:]
                    else:
                        out[prefix] = array
        return out

    # The generators these goldens were recorded with trained the recipe on
    # plain data: unaugmented, shuffled from the ambient stream, scored on the
    # whole fixture, at full precision.
    def _recipe(self, name: str, *, seed: int) -> SudokuTrainLoop:
        """Return ``exp006`` shrunk to two ACT steps and three train steps."""
        cfg = experiments.exp006()
        cfg.experiment_name = name
        cfg.base_dir = self.scratch
        cfg.seed = seed
        runtime = cfg.runtime = SingleProcess.Config()
        runtime.device = "cpu"
        cfg.step.dtype_autocast = None
        cfg.step.pool.max_steps = 2
        cfg.dataset.working_dir = self.data
        cfg.dataset.seed = None
        cfg.dataset.eval_num_instances = None
        cfg.dataset.augment = False
        cfg.dataset.augment_digits_only = False
        cfg.max_steps = 3
        cfg.num_steps_eval = float("inf")
        cfg.eval_warmup_batches = 0
        return cfg

    # Stored is each weight the eval-side loader ends up with, once: the EMA shadow
    # where the run kept one, which overlays the live parameter on load, and the live
    # value elsewhere. Nothing else in a checkpoint is read by an evaluation.
    def _checkpoint(
        self,
        name: str,
        path: Path,
        *,
        recipe: SudokuTrainLoop | VerifierFit.Config,
    ) -> None:
        """Train ``recipe``, or write its stored weights, to the file ``path``."""
        if self.train and not self._stored(f"{name}/"):
            recipe.make().run()
            weights = load_eval_weights(path)
            self.trained |= {f"{name}/{k}": v for k, v in weights.items()}
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self._stored(f"{name}/")}, path)

    def _stored_harvest(self, config: Harvest.Config) -> Path:
        """Write the stored shard and the manifest :meth:`Harvest.run` would."""
        out = Path(
            resolve_working_dir(
                self.scratch,
                str(config.run_dir).format(experiment_name=config.experiment_name),
            ),
        )
        out.mkdir(parents=True, exist_ok=True)
        arrays = {key: value.numpy() for key, value in self._stored("harvest/").items()}
        shard = out / "shard-00000.npz"
        write_npz(shard, arrays)
        rows = len(arrays["original"])
        manifest = {
            "schema_version": 1,
            "source_checkpoint": str(
                resolve_working_dir(
                    self.scratch,
                    str(config.harvest_source_checkpoint),
                ),
            ),
            "groups_per_shard": 8,
            "row_count": rows,
            "shard_count": 1,
            "shards": [
                {"file": shard.name, "rows": rows, "sha256": file_sha256(shard)},
            ],
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return out

    def _stored(self, prefix: str) -> dict[str, Tensor]:
        """Return the stored tensors under ``prefix``, the prefix removed."""
        return {
            key.removeprefix(prefix): value
            for key, value in self.fixtures.items()
            if key.startswith(prefix)
        }
