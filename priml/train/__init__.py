"""Training infrastructure: loop, step, optimizers, checkpointing, etc."""

from __future__ import annotations

from priml.optimizers.newton import Newton
from priml.train.activation import (
    DefaultActivationStorage,
    LayerActivationCheckpointing,
    QuantizedActivationStorage,
    QuantizedModuleActivationStorage,
    SelectiveActivationCheckpointing,
)
from priml.train.checkpointing import Checkpointer
from priml.train.custom_types import (
    ModelQuantizationProtocol,
    ParallelStrategyProtocol,
)
from priml.train.ema import EMA, NoEMA
from priml.train.parallelism import (
    DataParallel,
    FullySharded,
    HybridSharded,
    NoParallel,
    RecursiveSharded,
)
from priml.train.profiling import TorchProfiling
from priml.train.quantization import (
    Float8ModelQuantization,
    NoModelQuantization,
)
from priml.train.tensor_parallel import (
    TensorParallel,
    apply_tensor_parallel,
)
from priml.train.tracker import TensorBoardTracker, WandbTracker
from priml.train.train_loop import TrainLoop
from priml.train.train_step import TrainStep
from priml.train.train_step_gan import GANTrainStep


__all__ = [
    "EMA",
    "Checkpointer",
    "DataParallel",
    "DefaultActivationStorage",
    "Float8ModelQuantization",
    "FullySharded",
    "GANTrainStep",
    "HybridSharded",
    "LayerActivationCheckpointing",
    "ModelQuantizationProtocol",
    "Newton",
    "NoEMA",
    "NoModelQuantization",
    "NoParallel",
    "ParallelStrategyProtocol",
    "QuantizedActivationStorage",
    "QuantizedModuleActivationStorage",
    "RecursiveSharded",
    "SelectiveActivationCheckpointing",
    "TensorBoardTracker",
    "TensorParallel",
    "TorchProfiling",
    "TrainLoop",
    "TrainStep",
    "WandbTracker",
    "apply_tensor_parallel",
]
