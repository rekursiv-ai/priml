"""Check that a trained model scores on the kernels and autocast it trained under."""

from contextlib import nullcontext
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch

from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.checkpoint_test import (
    smoke_on_flash4,
)
from priml.baselines.craftax.world_model.experiments import (
    WorldModelLoop,
    exp_smoke,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.scoring import (
    autocast,
    load_trained,
)
from priml.lib.codec import from_plain
from priml.model.attention.flash4 import Flash4UnavailableError, Flash4Varlen
from priml.model.attention.kernel import SdpaVarlen


_HERE = "priml.baselines.craftax.world_model.scoring_test"


def test_off_cuda_a_flash4_experiment_scores_in_float32_on_sdpa(
    tmp_path: Path,
) -> None:
    checkpoint = _saved(tmp_path)
    model, config = load_trained(
        f"{_HERE}.smoke_on_flash4_in_bfloat16",
        checkpoint,
        overrides=["dataset.sampler_seed=3"],
        device=torch.device("cpu"),
    )
    assert config.dataset.sampler_seed == 3
    assert not model.training
    assert _kernels(model) == {SdpaVarlen}
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    # Training's bfloat16 autocast is a CUDA one; off CUDA the model scores in
    # float32.
    assert config.step.dtype_autocast == torch.bfloat16
    assert isinstance(autocast(config, torch.device("cpu")), nullcontext)


def test_an_experiment_without_autocast_scores_without_it() -> None:
    config = exp_smoke()
    assert config.step.dtype_autocast is None
    assert isinstance(autocast(config, torch.device("cuda")), nullcontext)


@pytest.mark.filterwarnings("ignore:.*deprecated:DeprecationWarning")
@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
def test_on_cuda_the_experiments_flash4_kernels_and_autocast_are_kept(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Keeping FlashAttention 4 needs a CUDA device.")
    try:
        Flash4Varlen.Config().make()
    except Flash4UnavailableError as error:
        pytest.skip(str(error))
    checkpoint = _saved(tmp_path)
    device = torch.device("cuda")
    # Every build initializes its model once; the checkpoint needs only one.
    with patch.object(
        WorldModel,
        "reset_parameters",
        autospec=True,
        side_effect=WorldModel.reset_parameters,
    ) as initialized:
        model, config = load_trained(
            f"{_HERE}.smoke_on_flash4_in_bfloat16",
            checkpoint,
            device=device,
        )
    assert initialized.call_count == 1
    assert _kernels(model) == {Flash4Varlen}
    assert {p.device.type for p in model.parameters()} == {"cuda"}
    state = from_plain(
        cast("object", torch.load(checkpoint, weights_only=True)),
        dict[str, object],
    )
    step = from_plain(state["step"], dict[str, object])
    saved = from_plain(step["model"], dict[str, torch.Tensor])
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), saved[name]), name
    with autocast(config, device):
        assert torch.get_autocast_dtype("cuda") == torch.bfloat16
        assert torch.is_autocast_enabled("cuda")


def smoke_on_flash4_in_bfloat16() -> WorldModelLoop.Config:
    """Return exp_smoke on FlashAttention 4 under bfloat16 autocast, as GPU runs train."""
    config = smoke_on_flash4()
    config.step.dtype_autocast = torch.bfloat16
    return config


def _saved(root: Path) -> Path:
    """Save a fresh exp_smoke model as a checkpoint; return its path."""
    torch.manual_seed(1)
    path = root / "step.pt"
    torch.save({"step": {"model": exp_smoke().step.model.make().state_dict()}}, path)
    return path


def _kernels(model: torch.nn.Module) -> set[type[object]]:
    """Return the varlen attention kernel types of ``model``."""
    return {
        type(m.attn_kernel) for m in model.modules() if isinstance(m, VarlenAttention)
    }


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
