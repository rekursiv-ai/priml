from abc import ABC, abstractmethod
from typing import Any

import torch

class Quantizer(ABC):
    @abstractmethod
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...

class TwoStepQuantizer:
    @abstractmethod
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
    @abstractmethod
    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module: ...
