"""Compare unmodified DLinear model/data/recipe code, then mint source goldens.

Reference-only dependencies belong in a separate validation environment:
pandas, scikit-learn, and matplotlib. Native tests never import this source.
"""
# ruff: noqa: S603, S607 -- Trusted Git and pytest commands for source verification.

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import hashlib
import importlib
import json
import subprocess
import sys

from torch import Tensor, nn

import numpy as np
import torch

from priml.baselines.etth1.experiments import exp000
from priml.baselines.etth1.scripts.data import DATASET_SHA256
from priml.baselines.etth1.testing import (
    model_record,
    tiny_batches,
    tiny_config,
    training_record,
    update_record,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


_CWD: Final = Path(__file__).resolve().parent
SOURCE_COMMIT: Final = "da9e67442b95af76488b8e4e1806cc3185723dd9"
_SOURCE_FILES: Final = (
    "models/DLinear.py",
    "data_provider/data_loader.py",
    "data_provider/data_factory.py",
    "exp/exp_main.py",
    "exp/exp_basic.py",
    "utils/tools.py",
    "utils/timefeatures.py",
)


class _Recipe(Protocol):
    model: nn.Module

    def _select_optimizer(self) -> torch.optim.Optimizer: ...
    def _select_criterion(self) -> nn.Module: ...


def reference_args(directory: Path, *, tiny: bool) -> SimpleNamespace:
    """Declare the source recipe, changing only dimensions for the tiny probe.

    Args:
      directory: Directory containing the prepared ETTh1 CSV.
      tiny: Whether to use the smaller geometry for the deterministic probe.

    Returns:
      args: Arguments consumed by the pinned upstream experiment and loader.

    """
    return SimpleNamespace(
        seq_len=5 if tiny else 336,
        pred_len=3 if tiny else 96,
        enc_in=4 if tiny else 7,
        individual=False,
        model="DLinear",
        use_multi_gpu=False,
        use_gpu=False,
        learning_rate=1e-4,
        lradj="type1",
        data="ETTh1",
        embed="timeF",
        freq="h",
        root_path=str(directory),
        data_path="ETTh1.csv",
        label_len=48,
        features="M",
        target="OT",
        num_workers=0,
        batch_size=8,
    )


def verify(reference: Path, directory: Path, *, mint: bool) -> dict[str, object]:
    """Require tiny and canonical source parity without writing goldens.

    Args:
      reference: DLinear checkout at SOURCE_COMMIT.
      directory: Prepared canonical ETTh1 dataset.
      mint: Must be False; minting runs through the pytest entry point.

    Returns:
      report: Source identity and exact-comparison evidence.

    """
    if mint:
        raise ValueError(
            "Mint through scripts.verify_reference --mint, which runs pytest.",
        )
    report, _, _ = capture_reference(reference, directory=directory)
    return report


def capture_reference(
    reference: Path,
    directory: Path,
) -> tuple[dict[str, object], dict[str, Tensor], dict[str, Tensor]]:
    """Compare both implementations and return records captured from the source.

    Args:
      reference: DLinear checkout at SOURCE_COMMIT.
      directory: Prepared canonical ETTh1 dataset.

    Returns:
      report: Source identity and exact-comparison evidence.
      model: Portable tiny-model record captured from the source.
      training: Portable three-update training record captured from the source.

    Raises:
      ValueError: The dataset, checkout revision, or source files are not pinned.

    """
    dataset_hash = hashlib.sha256((directory / "ETTh1.csv").read_bytes()).hexdigest()
    if dataset_hash != DATASET_SHA256:
        raise ValueError("Canonical source verification requires the pinned ETTh1 CSV.")
    revision = subprocess.check_output(
        ["git", "-C", str(reference), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != SOURCE_COMMIT:
        raise ValueError(f"Expected DLinear {SOURCE_COMMIT}, found {revision}.")
    hashes: dict[str, str] = {}
    compatibility_edits: list[str] = []
    for name in _SOURCE_FILES:
        payload = (reference / name).read_bytes()
        committed = subprocess.check_output(
            ["git", "-C", str(reference), "show", f"{SOURCE_COMMIT}:{name}"],
        )
        if name == "utils/tools.py" and payload == committed.replace(
            b"np.Inf",
            b"np.inf",
        ):
            compatibility_edits.append(
                "utils/tools.py: np.Inf -> np.inf (NumPy 2 spelling)",
            )
        elif payload != committed:
            raise ValueError(f"Reference source was modified: {name}.")
        hashes[name] = hashlib.sha256(payload).hexdigest()
    sys.path.insert(0, str(reference))
    experiments = cast(_Experiments, importlib.import_module("exp.exp_main"))
    data = cast(_Data, importlib.import_module("data_provider.data_factory"))
    source_tools = cast(_Tools, importlib.import_module("utils.tools"))
    torch.set_num_threads(1)
    portable_model: dict[str, Tensor] = {}
    portable_training: dict[str, Tensor] = {}
    for portable in (False, True):
        with host_agnostic_numerics() if portable else nullcontext():
            torch.manual_seed(2021)
            args = reference_args(directory, tiny=True)
            recipe = experiments.Exp_Main(args)
            optimizer = recipe._select_optimizer()  # noqa: SLF001 -- Exercise the reference's own recipe.
            criterion = recipe._select_criterion()  # noqa: SLF001 -- Exercise the reference's own recipe.
            source_model = _canonical_names(model_record(recipe.model))
            source_training = dict(source_model)
            for index, batch in enumerate(tiny_batches()):
                optimizer.zero_grad()
                prediction = cast(Tensor, recipe.model(batch["media"]))
                loss = cast(Tensor, criterion(prediction, target=batch["label"]))
                loss.backward()
                optimizer.step()
                source_training.update(
                    {
                        f"step{index + 1}/{key}": value
                        for key, value in _canonical_names(
                            update_record(
                                recipe.model,
                                optimizer=optimizer,
                                output=prediction,
                                loss=loss,
                            ),
                        ).items()
                    },
                )
                source_tools.adjust_learning_rate(optimizer, epoch=index + 1, args=args)
            candidate = tiny_config().make()
            _require_equal(source_model, actual=model_record(candidate.model))
            _require_equal(source_training, actual=training_record(candidate))
            if portable:
                portable_model, portable_training = source_model, source_training

    # The real-data check uses the normal float32 path.
    torch.manual_seed(2021)
    args = reference_args(directory, tiny=False)
    recipe = experiments.Exp_Main(args)
    source_initial = _canonical_names(
        {
            key: value.detach().clone()
            for key, value in recipe.model.state_dict().items()
        },
    )
    source_rng = torch.get_rng_state()
    _, loader = data.data_provider(args, flag="train")
    source_iterator = iter(loader)
    source_batches: list[dict[str, Tensor]] = []
    for _ in range(3):
        x, y, _, _ = next(source_iterator)
        source_batches.append({"media": x.float(), "label": y[:, -96:, :].float()})
    source_data_rng = torch.get_rng_state()
    optimizer = recipe._select_optimizer()  # noqa: SLF001 -- Exercise the reference's own recipe.
    criterion = recipe._select_criterion()  # noqa: SLF001 -- Exercise the reference's own recipe.
    cfg = exp000()
    cfg.dataset.working_dir = directory.resolve()
    cfg.dataset.base_dir = "/"
    cfg = cfg.copy_tree().finalize()
    candidate = cfg.step.make()
    _require_equal(source_initial, actual=candidate.model.state_dict())
    _require_equal({"rng": source_rng}, actual={"rng": torch.get_rng_state()})
    dataset = cfg.dataset.make()
    iterator = iter(dataset.train_dataloader())
    for batch in source_batches:
        candidate_batch = next(iterator)
        _require_equal(batch, actual=candidate_batch)
        optimizer.zero_grad()
        prediction = cast(Tensor, recipe.model(batch["media"]))
        loss = cast(Tensor, criterion(prediction, target=batch["label"]))
        loss.backward()
        optimizer.step()
        output = candidate.train_step(**candidate_batch)
        _require_equal(
            _canonical_names(
                update_record(
                    recipe.model,
                    optimizer=optimizer,
                    output=prediction,
                    loss=loss,
                ),
            ),
            actual=update_record(
                candidate.model,
                optimizer=candidate.optimizer,
                output=output["model"],
                loss=output["loss"],
            ),
        )
    _require_equal({"rng": source_data_rng}, actual={"rng": torch.get_rng_state()})
    report: dict[str, object] = {
        "source_commit": revision,
        "dataset_sha256": dataset_hash,
        "source_sha256": hashes,
        "source_compatibility_edits": compatibility_edits,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tiny_native_mismatches": 0,
        "tiny_portable_mismatches": 0,
        "canonical_three_step_mismatches": 0,
        "geometry": {
            "batch": 2,
            "history": 5,
            "horizon": 3,
            "channels": 4,
            "kernel": 25,
        },
        "training_steps": 3,
        "schedule_epochs": [0, 1, 2],
    }
    return report, portable_model, portable_training


def main() -> int:
    """Verify a pinned reference checkout and optionally mint portable goldens.

    Returns:
      code: 0 on success, or the minting pytest run's exit code.

    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    if flags.mint:
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                str(_CWD / "mint_reference_test.py"),
                "-o",
                "addopts=",
                "-p",
                "no:cacheprovider",
                "--etth1-reference",
                str(flags.reference.resolve()),
                "--etth1-directory",
                str(flags.directory.resolve()),
            ],
        )
    print(
        json.dumps(
            verify(flags.reference, directory=flags.directory, mint=False),
            indent=2,
        ),
    )
    return 0


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--mint",
        action="store_true",
        help="Mint source goldens in a fresh pytest process under priml/conftest.py.",
    )


def _canonical_names(record: Mapping[str, Tensor]) -> dict[str, Tensor]:
    names = {
        "Linear_Seasonal.": "seasonal.",
        "Linear_Trend.": "trend.",
        "Linear_Decoder.": "decoder.",
    }
    result: dict[str, Tensor] = {}
    for key, value in record.items():
        canonical = key
        for source, target in names.items():
            canonical = canonical.replace(source, target)
        result[canonical] = value
    return result


def _require_equal(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
) -> None:
    differences = mismatches(expected, actual=actual)
    if differences:
        raise AssertionError("\n".join(differences))


class _Experiments(Protocol):
    def Exp_Main(self, args: SimpleNamespace) -> _Recipe: ...  # noqa: N802 -- Upstream's class name.


class _Data(Protocol):
    def data_provider(
        self,
        args: SimpleNamespace,
        flag: str,
    ) -> tuple[object, Iterable[tuple[Tensor, Tensor, Tensor, Tensor]]]: ...


class _Tools(Protocol):
    def adjust_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        args: SimpleNamespace,
    ) -> None: ...


class _Flags(Protocol):
    reference: Path
    directory: Path
    mint: bool
