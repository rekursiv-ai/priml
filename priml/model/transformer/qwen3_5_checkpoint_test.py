"""Resume native hybrid language-model training with registered LoRA weights."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeGuard, cast, override

import copy
import random
import shutil

from configgle import Fig, Makes, PartialConfig
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import parametrize

import pytest
import torch

from priml.loss.custom_types import LossOutput
from priml.math.seed import RngState, get_rng_state, numpy_rng
from priml.model.linear import Linear
from priml.model.transformer.qwen3_5 import Qwen35
from priml.optimizers.composite import CompositeOptimizer
from priml.runtime import SingleProcess
from priml.testing.qwen3_5 import hf_config
from priml.timer import CheckpointableStepTimer
from priml.train.checkpointing import Checkpointer
from priml.train.custom_types import OptimizerProtocol, TrainStepOutput
from priml.train.train_loop import TrainLoop
from priml.train.train_step import TrainStep


@pytest.mark.parametrize("source", ["cadence", "final"])
def test_lora_training_checkpoint_resumes_exactly(
    tmp_path: Path,
    source: Literal["cadence", "final"],
) -> None:
    """A rebuilt hybrid resumes the next microbatch and optimizer update exactly."""
    reference_config = _loop_config(tmp_path / "reference")
    reference = reference_config.make()
    assert isinstance(reference.step, _RecordingStep)
    initial = copy.deepcopy(reference.step.model.state_dict())
    _assert_trainable_membership(reference.step)
    reference.train()
    assert reference.checkpointing is not None
    assert reference.checkpointing.available_steps() == [1, 2]
    reference_rng = get_rng_state()

    source_dir = tmp_path / "reference" / "checkpoints"
    if source == "final":
        prefix_config = _loop_config(tmp_path / "prefix")
        prefix_config.max_steps = 1
        assert isinstance(prefix_config.checkpointing, Checkpointer.Config)
        prefix_config.checkpointing.save_every = 2
        prefix = prefix_config.make()
        prefix.train()
        assert prefix.checkpointing is not None
        assert prefix.checkpointing.available_steps() == [1]
        source_dir = tmp_path / "prefix" / "checkpoints"

    resumed_dir = tmp_path / "resumed" / "checkpoints"
    resumed_dir.mkdir(parents=True)
    shutil.copyfile(source_dir / "step_00000001.pt", resumed_dir / "step_00000001.pt")
    resumed_config = _loop_config(tmp_path / "resumed")
    # Construction must not accidentally recreate the checkpoint's initialization.
    resumed_config.seed = 91
    resumed = resumed_config.make()
    assert isinstance(resumed.step, _RecordingStep)
    _assert_trainable_membership(resumed.step)
    assert resumed.step.global_step == 1
    assert resumed.step.accumulation_steps == resumed.step.accumulated_samples == 0
    resumed.train()

    expected = reference.step.observations[2:]
    actual = resumed.step.observations
    assert len(actual) == len(expected) == 2
    for restored, continuous in zip(actual, expected, strict=True):
        assert torch.equal(restored.media, continuous.media)
        assert torch.equal(restored.label, continuous.label)
        _assert_rng_equal(restored.rng, continuous.rng)
        assert torch.equal(restored.loss, continuous.loss)
    _assert_rng_equal(get_rng_state(), reference_rng)
    torch.testing.assert_close(
        resumed.step.model.state_dict(),
        reference.step.model.state_dict(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        resumed.step.optimizer.state_dict(),
        reference.step.optimizer.state_dict(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        resumed.step.last_microbatch_grads,
        reference.step.last_microbatch_grads,
        rtol=0,
        atol=0,
    )
    assert resumed.step.global_step == 2
    for name, parameter in reference.step.model.named_parameters():
        if parameter.requires_grad:
            assert not torch.equal(initial[name], parameter), name
        else:
            assert torch.equal(initial[name], parameter), name


def test_lora_adapter_only_roundtrip(tmp_path: Path) -> None:
    """Adapter weights alone reproduce logits on the same frozen native base."""
    config = _loop_config(tmp_path)
    config.max_steps = 1
    config.checkpointing = None
    loop = config.make()
    assert isinstance(loop.step, _RecordingStep)
    assert isinstance(loop.step.model, Qwen35)
    initial = copy.deepcopy(loop.step.model.state_dict())
    tokens = torch.tensor([[1, 2]])
    before = loop.step.model.forward(tokens).detach()
    loop.train()
    expected = loop.step.model.forward(tokens).detach()
    assert not torch.equal(expected, before)
    adapters = {
        name: parameter.detach().clone()
        for name, parameter in loop.step.model.named_parameters()
        if parameter.requires_grad
    }
    path = tmp_path / "adapters.pt"
    torch.save(adapters, path)

    restored = config.copy_tree().finalize().step.make().model
    assert isinstance(restored, Qwen35)
    restored.load_state_dict(initial)
    serialized = cast(object, torch.load(path, weights_only=True))
    incompatible = restored.load_state_dict(
        _tensor_state_dict(serialized),
        strict=False,
    )
    assert incompatible.unexpected_keys == []
    assert set(incompatible.missing_keys) == set(initial) - set(adapters)
    assert torch.equal(restored.forward(tokens), expected)
    for name, parameter in restored.named_parameters():
        if not parameter.requires_grad:
            assert torch.equal(parameter, initial[name]), name


def _loop_config(
    directory: Path,
) -> TrainLoop.Config[_RecordingStep.Config, _TokenStream.Config]:
    config = TrainLoop.Config(
        step=_RecordingStep.Config(),
        dataset=_TokenStream.Config(),
    )
    config.base_dir = directory
    config.working_dir = "."
    config.seed = 42
    config.max_steps = 2
    config.num_steps_eval = 0
    config.eval_every_epoch = False
    config.early_train_log_steps = 0
    runtime = SingleProcess.Config()
    runtime.device = "cpu"
    config.runtime = runtime
    config.step.model = Qwen35.Config.from_hf(hf_config())
    config.step.compile = None
    config.step.accumulate_grad_batches = 2
    config.step.train_budget_steps = config.max_steps
    config.step.loss = PartialConfig(_token_loss)
    optimizer = CompositeOptimizer.Config()
    optimizer.optimizers = [PartialConfig(torch.optim.AdamW)]
    config.step.optimizer = optimizer
    assert isinstance(config.checkpointing, Checkpointer.Config)
    config.checkpointing.save_every = 1
    return config


class _LowRankWeight(nn.Module):
    """Represent a nonzero LoRA update to a frozen native weight."""

    def __init__(self, weight: Tensor) -> None:
        super().__init__()
        self.lora_a = nn.Parameter(torch.randn(2, weight.shape[1]) * 0.05)
        self.lora_b = nn.Parameter(torch.randn(weight.shape[0], 2) * 0.05)

    @override
    def forward(self, weight: Tensor) -> Tensor:
        return weight + self.lora_b @ self.lora_a


@dataclass(frozen=True, slots=True, kw_only=True)
class _Observation:
    media: Tensor
    label: Tensor
    rng: RngState
    loss: Tensor


class _RecordingStep(TrainStep):
    """Adapt batch arguments and observe the real supervised TrainStep."""

    class Config(Makes["_RecordingStep"], TrainStep.Config[Qwen35.Config]):
        """Native model and real optimizer, with observations outside checkpoint state."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.observations: list[_Observation] = []

    @override
    def build_optimizer(self, model: nn.Module) -> OptimizerProtocol:
        model.requires_grad_(False)
        for module in tuple(model.modules()):
            if isinstance(module, Linear):
                parametrize.register_parametrization(
                    module,
                    "weight",
                    _LowRankWeight(module.weight),
                )
        return super().build_optimizer(model)

    @override
    def __call__(self, *args: object, **kwargs: object) -> Tensor:
        del args
        media = kwargs["media"]
        assert isinstance(media, Tensor)
        result: object = super().__call__(media)
        assert isinstance(result, Tensor)
        return result

    @override
    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        media, label = preprocessed_batch["media"], preprocessed_batch["label"]
        assert isinstance(media, Tensor)
        assert isinstance(label, Tensor)
        rng = get_rng_state()
        result = super().train_step(**preprocessed_batch)
        self.observations.append(
            _Observation(
                media=media.clone(),
                label=label.clone(),
                rng=rng,
                loss=result["loss"].detach().clone(),
            )
        )
        return result


class _TokenStream:
    """Checkpoint a cursor in a tiny stream whose batches consume all CPU RNGs."""

    class Config(Fig["_TokenStream"]):
        """Fixed tiny stream for checkpoint integration tests."""

    def __init__(self, config: Config) -> None:
        del config
        self.cursor = 0
        self.timer_epoch = CheckpointableStepTimer()

    def train_dataloader(self) -> Iterator[dict[str, Tensor]]:
        while True:
            media = torch.randint(0, 32, (1, 2))
            media[0, 0] = self.cursor % 32
            offsets = list(range(1, 8))
            random.shuffle(offsets)
            label = (media + offsets[0] + int(numpy_rng.integers(1, 8))) % 32
            self.cursor += 1
            yield {"media": media, "label": label}

    def eval_dataloader(self) -> tuple[()]:
        return ()

    def state_dict(self) -> dict[str, object]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        cursor = state_dict["cursor"]
        assert isinstance(cursor, int)
        self.cursor = cursor


def _token_loss(prediction: Tensor, **batch: object) -> LossOutput:
    label = batch["label"]
    assert isinstance(label, Tensor)
    return {
        "loss": functional.cross_entropy(
            prediction.flatten(0, 1),
            label.flatten(),
            reduction="none",
        )
    }


def _assert_rng_equal(actual: RngState, expected: RngState) -> None:
    assert torch.equal(actual["torch"], expected["torch"])
    assert actual["python"] == expected["python"]
    assert "numpy" in actual
    assert "numpy" in expected
    assert actual["numpy"] == expected["numpy"]


def _tensor_state_dict(value: object) -> dict[str, Tensor]:
    assert _is_tensor_state_dict(value)
    return value


def _is_tensor_state_dict(value: object) -> TypeGuard[dict[str, Tensor]]:
    """Return whether an adapter state dictionary contains named tensors."""
    if not isinstance(value, dict):
        return False
    state_dict = cast(dict[object, object], value)
    return all(
        isinstance(name, str) and isinstance(parameter, Tensor)
        for name, parameter in state_dict.items()
    )


def _parameters_in(group: Mapping[str, object]) -> list[Tensor]:
    return _tensor_list(group["params"])


def _tensor_list(value: object) -> list[Tensor]:
    assert _is_tensor_list(value)
    return value


def _is_tensor_list(value: object) -> TypeGuard[list[Tensor]]:
    """Return whether a list contains only tensors."""
    if not isinstance(value, list):
        return False
    items = cast(list[object], value)
    return all(isinstance(item, Tensor) for item in items)


def _assert_trainable_membership(step: TrainStep) -> None:
    assert isinstance(step.model, Qwen35)
    trainable = {
        name: parameter
        for name, parameter in step.model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert set(trainable) == {
        name
        for name, _ in step.model.named_parameters()
        if name.endswith((".lora_a", ".lora_b"))
    }
    for module in step.model.modules():
        if isinstance(module, _LowRankWeight):
            assert torch.count_nonzero(module.lora_b @ module.lora_a) > 0
    assert {id(parameter) for parameter in trainable.values()} == {
        id(parameter)
        for group in step.optimizer.param_groups
        for parameter in _parameters_in(group)
    }


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
