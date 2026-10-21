"""Checks for the train step module."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, NotRequired, TypedDict, cast
from unittest.mock import patch

from configgle import PartialConfig
from torch import Tensor
from treekle import from_plain, json_loads

import pytest
import torch

from priml.baselines.casmi2026.data import TrainingPreparation
from priml.baselines.casmi2026.data_test import cache
from priml.baselines.casmi2026.model import FPNet, PeakAttention
from priml.baselines.casmi2026.model_test import inputs, tiny_model
from priml.baselines.casmi2026.train_step import (
    CasmiRun,
    FingerprintFineTune,
    FingerprintLoss,
    casmi_loop,
)
from priml.loss.simple_loss import SimpleLoss, mse
from priml.model.norm import LayerNorm
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU
from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.newton import Newton
from priml.runtime import SingleProcess
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import (
    assert_tensor_golden,
    mismatches,
    read_tensors,
    rng_fingerprint,
)
from priml.train.checkpointer import Checkpointer
from priml.train.ema import EMA
from priml.train.parallelism import NoParallel


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from priml.baselines.casmi2026.data import SpectrumData
    from priml.loss.custom_types import LossOutput


GOLDEN: Final = _CWD / "testdata" / "fpnet.pt"


def tiny_step() -> FingerprintFineTune.Config:
    """Keep AdamW, BCE, and dropout while shrinking only the source geometry."""
    config = FingerprintFineTune.Config()
    config.model = tiny_model()
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    return config


def batch() -> SpectrumData.Batch:
    """Rebuild the deterministic inputs instead of storing them."""
    mz, it, pad, prec, ad, ce, known, merged, mode = inputs()
    return {
        "mz": mz,
        "it": it,
        "pad": pad,
        "prec": prec,
        "ad": ad,
        "ce": ce,
        "ce_known": known,
        "merged": merged,
        "mode": mode,
        "label": torch.arange(21).reshape(3, 7).remainder(2).float(),
        "sid": torch.tensor([0, 0, 1]),
        "neutral_mass": torch.tensor([100.0, 100.0, 200.0]),
        "valid_count": torch.tensor(3),
    }


def test_three_steps_match_every_state_tensor() -> None:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        _ = torch.manual_seed(42)
        step = tiny_step().make()
        model = step.model
        assert isinstance(model, FPNet)
        actual = {
            f"initial/{name}": value.detach().clone()
            for name, value in model.state_dict().items()
        }
        actual["initial/rng"] = rng_fingerprint()
        actual["loss"] = torch.stack(
            [step.train_step(**batch())["loss"] for _ in range(3)],
        ).float()
        for name, value in model.state_dict().items():
            actual[f"final/{name}"] = value.detach().clone()
        optimizer = step.optimizer.state_dict()
        state = from_plain(optimizer["state"], dict[int, dict[str, Tensor]])
        names = [name for name, _ in model.named_parameters()]
        for index, values in state.items():
            for name, value in values.items():
                actual[f"optimizer/{names[index]}/{name}"] = (
                    value.detach().clone().float()
                )
        step.model.eval()
        logits, elements = model.forward(*inputs())
        actual["logits"] = logits.detach().float()
        actual["elements"] = elements.detach().float()
        with torch.inference_mode():
            inference = model.forward(*inputs())
        actual["inference/logits"] = inference[0].detach().float()
        actual["inference/elements"] = inference[1].detach().float()
        actual["final/rng"] = rng_fingerprint()
    assert_tensor_golden(GOLDEN, actual)


def test_native_eval_and_keyword_contract() -> None:
    step = tiny_step().make()
    assert step.eval_loss(**batch())["loss"].shape == (3,)
    output = from_plain(step.train_loss(**batch())["model"], tuple[Tensor, Tensor])
    assert output[0].shape == (3, 7)
    assert output[1].shape == (3, 6)


def test_pretrained_state_is_loaded_exactly(tmp_path: Path) -> None:
    model = tiny_model().make()
    checkpoint = tmp_path / "released.pt"
    torch.save({"model": model.released_state()}, checkpoint)
    config = tiny_step()
    config.pretrained = checkpoint
    with patch.object(torch, "load", wraps=torch.load) as load:
        step = config.make()
    load.assert_called_once_with(checkpoint, map_location="cpu", weights_only=True)
    assert all(
        torch.equal(tensor, step.model.state_dict()[name])
        for name, tensor in model.state_dict().items()
    )


def test_fingerprint_loss_averages_the_last_axis_of_the_bit_losses() -> None:
    def per_bit(prediction: Tensor, **batch: object) -> LossOutput:
        del batch
        return {"loss": prediction.reshape(2, 3, 4)}

    config = FingerprintLoss.Config(loss=PartialConfig(per_bit))
    prediction = torch.arange(24, dtype=torch.float32)
    result = config.make()((prediction, torch.zeros(2, 5)))
    assert torch.equal(result["loss"], prediction.reshape(2, 3, 4).mean(-1))


def test_injected_loss_and_ema_are_used() -> None:
    config = tiny_step()
    config.loss = FingerprintLoss.Config(loss=SimpleLoss.Config(loss_fn=mse))
    step = config.make()
    with patch.object(step.ema, "apply_to", return_value=nullcontext()) as apply:
        output = step.eval_loss(**batch())
    apply.assert_called_once_with(step.model)
    logits = from_plain(output["model"], tuple[Tensor, Tensor])[0]
    assert torch.equal(output["loss"], (logits - batch()["label"]).square().mean(-1))


def test_training_uses_native_forward_and_its_compile_slot() -> None:
    step = tiny_step().make()
    logits = torch.zeros(3, 7, requires_grad=True)
    with patch.object(
        step,
        "_train_forward",
        return_value=(logits, torch.zeros(3, 6)),
    ) as forward:
        output = step.train_step(**batch())
    assert forward.call_count == 1
    assert torch.equal(
        from_plain(output["model"], tuple[Tensor, Tensor])[0],
        logits.detach(),
    )


def test_unsupported_accumulation_is_rejected_before_model_creation() -> None:
    config = tiny_step()
    config.accumulate_grad_batches = 2
    with pytest.raises(
        ValueError,
        match=r"^Fingerprint source recipe requires accumulation=1\.$",
    ):
        _ = config.make()


@pytest.mark.parametrize("fails", [False, True])
def test_evaluation_restores_every_module_mode(fails: bool) -> None:
    step = tiny_step().make()
    step.model.train()
    step.model.get_submodule("blocks.0.ffn.0.middle").eval()
    expected = [module.training for module in step.model.modules()]
    if fails:
        malformed = batch()
        malformed["ad"] = torch.full((3,), 1000, dtype=torch.long)
        with pytest.raises(IndexError):
            _ = step.call_eval(**malformed)
    else:
        _ = step.call_eval(**batch())
    assert [module.training for module in step.model.modules()] == expected


def test_training_loss_keeps_gradients_dropout_and_live_weights() -> None:
    step = tiny_step().make()
    step.model.eval()
    with patch.object(step.ema, "apply_to", side_effect=AssertionError("EMA used")):
        result = step.train_loss(**batch())
    assert result["loss"].requires_grad
    assert all(module.training for module in step.model.modules())
    result["loss"].mean().backward()
    assert any(parameter.grad is not None for parameter in step.model.parameters())


@pytest.mark.parametrize("kind", ["newton", "lbfgs", "composite"])
def test_closure_optimizer_is_rejected_before_training(kind: str) -> None:
    config = tiny_step()
    config.optimizer = (
        Newton.Config()
        if kind == "newton"
        else PartialConfig(torch.optim.LBFGS)
        if kind == "lbfgs"
        else CompositeOptimizer.Config(optimizers=[Newton.Config()])
    )
    with pytest.raises(
        ValueError,
        match=r"^Fingerprint mean-backward recipe does not support closure optimizers\.$",
    ):
        _ = config.make()


def test_native_evaluation_context_is_explicit() -> None:
    expected = read_tensors(GOLDEN)
    with host_agnostic_numerics():
        step = tiny_step().make()
        model = step.model
        assert isinstance(model, FPNet)
        model.load_state_dict(
            {
                name.removeprefix("final/"): tensor
                for name, tensor in expected.items()
                if name.startswith("final/") and name != "final/rng"
            },
        )
        step.model.eval()
        with torch.inference_mode():
            fingerprints, elements = model.forward(*inputs())
        actual = {
            "inference/logits": from_plain(
                step.call_eval(**batch()),
                tuple[Tensor, Tensor],
            )[0],
            "inference/elements": elements,
        }
        assert torch.equal(fingerprints, actual["inference/logits"])
    assert (
        mismatches(
            {name: expected[name] for name in actual},
            actual,
        )
        == []
    )


class _Metadata(TypedDict):
    completed: bool
    steps: int
    cache_reused: NotRequired[bool]


class _Receipt(TypedDict):
    status: Literal["complete", "failed"]
    budget_seconds: float
    elapsed_seconds: float
    metadata: _Metadata
    error: NotRequired[str]


def tiny_job(root: Path) -> CasmiRun:
    """Keep the optimization recipe while shrinking cache, model, and step count."""
    data = root / "data"
    data.mkdir()
    config = CasmiRun()
    config.budget.seconds = 300.0
    casmi_loop(config).dataset = cache(data)
    casmi_loop(config).dataset.base_dir = None
    casmi_loop(config).step = tiny_step()
    config.loop.runtime = SingleProcess.Config(device="cpu")
    config.loop.checkpointer = None
    config.loop.tracker = None
    casmi_loop(config).metrics_eval = {}
    config.loop.max_steps = 3
    config.loop.working_dir = root / "native"
    config.loop.base_dir = None
    return config


def test_native_three_steps_export_and_budget_receipt(tmp_path: Path) -> None:
    config = tiny_job(tmp_path)
    config.loop.working_dir = tmp_path / "new" / "native"
    config.make().run()
    receipt = json_loads(
        (Path(config.loop.working_dir) / "budget.json").read_text(),
        _Receipt,
    )
    assert receipt["metadata"]["completed"] is True
    assert receipt["status"] == "complete"
    assert receipt["metadata"]["steps"] == 3
    assert receipt["elapsed_seconds"] > 0
    assert receipt["budget_seconds"] == 300.0
    assert "cache_reused" not in receipt["metadata"]
    assert (Path(config.loop.working_dir) / "fpnet.pt").is_file()
    assert (Path(config.loop.working_dir) / "config.txt").is_file()
    assert set(receipt) == {"status", "budget_seconds", "elapsed_seconds", "metadata"}


@pytest.mark.parametrize("warm", [False, True])
def test_cold_preparation_and_verified_warm_reuse_are_distinct(
    tmp_path: Path,
    warm: bool,
) -> None:
    config = tiny_job(tmp_path)
    config.preparation = TrainingPreparation.Config()
    config.loop.max_steps = 1
    with (
        patch.object(TrainingPreparation, "cache_ready", return_value=warm),
        patch.object(TrainingPreparation, "run") as prepare,
    ):
        config.make().run()
    assert prepare.call_count == int(not warm)
    receipt = json_loads(
        (Path(config.loop.working_dir) / "budget.json").read_text(),
        _Receipt,
    )
    assert receipt["metadata"].get("cache_reused") is warm
    assert receipt["status"] == "complete"


def test_export_follows_the_native_run_directory(tmp_path: Path) -> None:
    config = tiny_job(tmp_path)
    config.loop.max_steps = 1
    config.loop.working_dir = str(tmp_path / "{experiment_name}")
    config.loop.experiment_name = "exp001"
    config.make().run()
    original = tmp_path / "exp001/fpnet.pt"
    expected = original.read_bytes()
    config.loop.experiment_name = "exp002"
    config.make().run()
    assert (tmp_path / "exp002/fpnet.pt").is_file()
    assert original.read_bytes() == expected


def test_resumed_custom_loop_is_not_admitted(tmp_path: Path) -> None:
    config = tiny_job(tmp_path)
    native = config.loop.make()
    assert isinstance(native.step, FingerprintFineTune)
    _ = native.step.train_step(**batch())
    with (
        patch.object(type(config.loop), "make", return_value=native),
        pytest.raises(
            ValueError,
            match=r"^A budgeted run cannot resume an earlier training budget\.$",
        ),
    ):
        config.make().run()
    assert not (Path(config.loop.working_dir) / "fpnet.pt").exists()
    receipt = json_loads(
        (Path(config.loop.working_dir) / "budget.json").read_text(),
        _Receipt,
    )
    assert receipt["metadata"]["completed"] is False
    assert receipt["metadata"]["steps"] == 1


def test_relaunch_refuses_to_replace_completed_artifacts(tmp_path: Path) -> None:
    config = tiny_job(tmp_path)
    config.loop.checkpointer = Checkpointer.Config(resume=False)
    config.loop.max_steps = 1
    config.make().run()
    destination = Path(config.loop.working_dir)
    expected = {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        config.make().run()
    assert {
        path.relative_to(destination): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == expected


def test_export_reconstructs_nondefault_geometry_and_evaluated_ema(
    tmp_path: Path,
) -> None:
    config = tiny_job(tmp_path)
    block = casmi_loop(config).step.model.block
    assert isinstance(block.ffn, Sequential.Config)
    assert isinstance(block.ffn.elements, list)
    assert isinstance(block.ffn.elements[0], SwiGLU.Config)
    assert isinstance(block.norm1, LayerNorm.Config)
    block.ffn.elements[0].expansion = 2.0
    block.norm1.eps = 0.003
    casmi_loop(config).step.model.norm.eps = 0.02
    casmi_loop(config).step.ema = EMA.Config(decay=1.0)
    native = config.loop.make()
    assert isinstance(native.step, FingerprintFineTune)
    with patch.object(type(config.loop), "make", return_value=native):
        config.make().run()
    payload = from_plain(
        cast(
            object,
            torch.load(Path(config.loop.working_dir) / "fpnet.pt", weights_only=True),
        ),
        dict[str, object],
        allow_imports=True,
    )
    assert payload["step"] == 3
    assert payload["d"] == 8
    model = casmi_loop(config).step.model
    assert payload["nbits"] == model.channels_out
    assert payload["layers"] == model.num_layers
    assert isinstance(model.block.attn, PeakAttention.Config)
    assert payload["heads"] == model.block.attn.attention.num_heads
    assert payload["note"] == "New training recipe; not public-best training parity."
    assert from_plain(payload["model"], dict[str, torch.Tensor])
    restored_config = from_plain(payload["config"], FPNet.Config, allow_imports=True)
    assert restored_config.pformat(hide_default_values=False) == model.pformat(
        hide_default_values=False,
    )
    restored = restored_config.make().eval()
    _ = restored.load_state_dict(from_plain(payload["model"], dict[str, torch.Tensor]))
    expected = from_plain(native.step.call_eval(**batch()), tuple[Tensor, Tensor])[0]
    with torch.inference_mode():
        assert torch.equal(restored.forward(*inputs())[0], expected)
    assert any(
        not torch.equal(tensor, native.step.model.state_dict()[name])
        for name, tensor in restored.state_dict().items()
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
