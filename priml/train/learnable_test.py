"""Tests for Learnable (model + optimizer + scheduler + EMA wrapper)."""

from __future__ import annotations

from typing import Any, override

from configgle import Fig, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.train.ema import EMA, NoEMA
from priml.train.learnable import Learnable
from priml.train.parallelism import NoParallel


class _Tiny(nn.Module):
    """Linear model with a deterministic forward for eval comparison."""

    class Config(Fig["_Tiny"], make_with_kwargs=True):
        dim: int = -1

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)

    @override
    def forward(self, x: Tensor, **_kwargs: Any) -> Tensor:
        """Apply the linear layer."""
        return self.fc(x)


def _make(ema_config: Any) -> Learnable:
    return Learnable.Config(
        model=_Tiny.Config(dim=4),
        optimizer=PartialConfig(torch.optim.SGD, lr=0.0),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
        ema=ema_config,
    ).make()


def test_call_eval_param_dict_uses_shadow_not_live_weights() -> None:
    """#338: param_dict EMA call_eval must evaluate averaged (shadow) weights.

    param_dict mode keeps ``shadow_model is None`` by design (FSDP-safe), so
    the old ``self.ema.shadow_model or self.model`` fell back to LIVE weights.
    Drive the live weights away from the shadow and confirm ``call_eval``
    matches the shadow-swapped forward, not the live forward.
    """
    torch.manual_seed(0)
    learnable = _make(EMA.Config(decay=0.5, shadow_kind="param_dict"))

    x = torch.randn(2, 4)

    # Seed the shadow at the current weights, then mutate live weights so the
    # shadow (average) diverges from live.
    learnable.ema(learnable.model)
    with torch.no_grad():
        for p in learnable.model.parameters():
            p.add_(5.0)
    learnable.ema(learnable.model)  # shadow = 0.5*orig + 0.5*(orig+5)

    # Reference: forward with the shadow swapped in.
    with torch.inference_mode(), learnable.ema.apply_to(learnable.model):
        shadow_out = learnable.model(x).clone()
    live_out = learnable.model(x).clone()

    eval_out = learnable.call_eval(x)

    # The shadow and live forwards must differ (precondition), and call_eval
    # must match the shadow.
    assert not torch.allclose(shadow_out, live_out), "shadow == live; test inert"
    torch.testing.assert_close(eval_out, shadow_out)


def test_call_eval_module_shadow_uses_shadow() -> None:
    """#338: module-kind EMA call_eval still evaluates the shadow module."""
    torch.manual_seed(0)
    learnable = _make(EMA.Config(decay=0.5, shadow_kind="module"))

    x = torch.randn(2, 4)
    learnable.ema(learnable.model)
    with torch.no_grad():
        for p in learnable.model.parameters():
            p.add_(5.0)
    learnable.ema(learnable.model)

    with torch.inference_mode(), learnable.ema.apply_to(learnable.model):
        shadow_out = learnable.model(x).clone()
    torch.testing.assert_close(learnable.call_eval(x), shadow_out)


def test_call_eval_no_ema_uses_live() -> None:
    """#338: NoEMA call_eval evaluates live weights (apply_to is a no-op)."""
    torch.manual_seed(0)
    learnable = _make(NoEMA.Config())

    x = torch.randn(2, 4)
    with torch.inference_mode():
        live_out = learnable.model(x).clone()
    torch.testing.assert_close(learnable.call_eval(x), live_out)


def _learnable_with(**config_kwargs: Any) -> Learnable:
    return Learnable.Config(
        model=_Tiny.Config(dim=4),
        optimizer=PartialConfig(torch.optim.SGD, lr=0.0),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
        ema=NoEMA.Config(),
        **config_kwargs,
    ).make()


def test_load_strict_false_tolerates_missing_keys() -> None:
    """strict=False loads a checkpoint missing model keys w/o error.

    Strict load (the default) raises on the absent key; the finetuning policy
    lets the missing parameter keep its fresh initialization.
    """
    full = _learnable_with()
    state = full.state_dict()
    del state["model"]["fc.weight"]  # simulate a checkpoint lacking this param

    strict = _learnable_with()
    with pytest.raises(RuntimeError, match="Missing key"):
        strict.load_state_dict(state)  # default policy is strict

    lenient = _learnable_with()
    lenient.load_state_dict(state, strict=False)  # must not raise


def test_parameter_remap_transforms_before_load() -> None:
    """Remap kwarg rewrites the saved model dict before it is loaded."""
    source = _learnable_with()
    with torch.no_grad():
        for p in source.model.parameters():
            p.fill_(3.0)
    state = source.state_dict()
    # Rename fc.weight -> renamed.weight in the checkpoint; remap puts it back.
    state["model"] = {"renamed.weight": state["model"]["fc.weight"]}

    def _remap(sd: Any) -> Any:
        return {"fc.weight": sd["renamed.weight"]}

    target = _learnable_with()
    target.load_state_dict(state, remap=_remap)
    assert float(target.model.fc.weight.detach()[0, 0]) == 3.0


def _adam_learnable() -> Learnable:
    """A learnable whose optimizer (Adam) keeps per-parameter state."""
    return Learnable.Config(
        model=_Tiny.Config(dim=4),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.1),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
        ema=NoEMA.Config(),
    ).make()


def test_load_optimizer_false_skips_optimizer_restore() -> None:
    """Finetuning must not restore the old optimizer state.

    ``load_optimizer=False`` keeps the fresh optimizer, letting
    ``strict=False`` finetuning succeed without a mismatched optimizer.
    """
    source = _adam_learnable()
    source.model(torch.randn(2, 4)).sum().backward()
    source.optimizer.step()  # populates Adam state
    state = source.state_dict()
    del state["model"]["fc.weight"]  # architecture changed

    finetune = _adam_learnable()
    finetune.load_state_dict(state, strict=False, load_optimizer=False)
    assert finetune.optimizer.state_dict()["state"] == {}


def test_load_optimizer_true_restores_optimizer_by_default() -> None:
    """The default policy restores optimizer state (ordinary resume)."""
    source = _adam_learnable()
    source.model(torch.randn(2, 4)).sum().backward()
    source.optimizer.step()
    state = source.state_dict()

    target = _adam_learnable()
    target.load_state_dict(state)
    assert target.optimizer.state_dict()["state"], "optimizer state not restored"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
