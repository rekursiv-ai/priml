"""Check that a trained world model is rebuilt with its checkpointed weights."""

from pathlib import Path

import pytest
import torch

from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.checkpoint import (
    load_world_model,
)
from priml.baselines.craftax.world_model.experiments import (
    WorldModelLoop,
    exp_smoke,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.model.attention.flash4 import Flash4Varlen
from priml.model.attention.kernel import SdpaVarlen
from priml.model.transformer.block import TransformerBlock


_HERE = "priml.baselines.craftax.world_model.checkpoint_test"


def test_load_world_model_restores_weights_and_applies_overrides(
    tmp_path: Path,
) -> None:
    torch.manual_seed(1)
    saved = exp_smoke().step.model.make()
    torch.save({"step": {"model": saved.state_dict()}, "rng": {}}, tmp_path / "s.pt")
    torch.manual_seed(2)
    model, config = load_world_model(
        "priml.baselines.craftax.world_model.experiments.exp_smoke",
        tmp_path / "s.pt",
        overrides=["dataset.sampler_seed=3"],
    )
    assert not model.training
    assert isinstance(config, WorldModelLoop.Config)
    assert config.dataset.sampler_seed == 3
    # Finalized: the corpus resolves beneath the archive root.
    assert str(config.dataset.corpus).endswith("smoke/corpora/smoke.json")
    for (name, ours), theirs in zip(
        model.state_dict().items(),
        saved.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(ours, theirs), name


def test_a_flash4_world_model_experiment_loads_on_the_cpu(tmp_path: Path) -> None:
    torch.manual_seed(1)
    saved = exp_smoke().step.model.make()
    torch.save({"step": {"model": saved.state_dict()}}, tmp_path / "s.pt")
    model, _ = load_world_model(f"{_HERE}.smoke_on_flash4", tmp_path / "s.pt")
    kernels = [m.attn_kernel for m in model.modules() if isinstance(m, VarlenAttention)]
    assert kernels
    assert all(isinstance(k, SdpaVarlen) for k in kernels)
    assert {p.device.type for p in model.parameters()} == {"cpu"}
    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_an_experiment_training_another_model_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="does not train a WorldModel"):
        load_world_model(
            "priml.baselines.craftax.experiments.exp_smoke",
            tmp_path / "absent.pt",
        )


def smoke_on_flash4() -> WorldModelLoop.Config:
    """Return exp_smoke on FlashAttention 4, as every GPU experiment trains."""
    config = exp_smoke()
    model = config.step.model
    assert isinstance(model, WorldModel.Config)
    block = model.transformer.block
    assert isinstance(block, TransformerBlock.Config)
    assert isinstance(block.attn, VarlenAttention.Config)
    block.attn.attn_kernel = Flash4Varlen.Config()
    return config


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
