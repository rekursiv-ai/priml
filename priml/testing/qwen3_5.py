"""Shared Qwen3.5 test fixtures and reference helpers."""

from __future__ import annotations

from functools import partial
from types import CellType, FunctionType, MethodType
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import inspect

from torch import Tensor, nn

import torch

from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3_5 import Qwen35


if TYPE_CHECKING:
    from collections.abc import Iterable

    from transformers.configuration_utils import PretrainedConfig


@runtime_checkable
class HfReference(Protocol):
    """The Qwen reference operations exercised by parity tests."""

    @property
    def config(self) -> PretrainedConfig:
        """Return the reference configuration."""
        ...

    def __call__(self, x: Tensor, /, **kwargs: object) -> object:
        """Run a Qwen reference model."""
        ...

    def named_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        """Return named reference parameters."""
        ...

    def state_dict(self) -> dict[str, Tensor]:
        """Return reference parameters."""
        ...

    def get_parameter(self, target: str) -> nn.Parameter:
        """Return one named reference parameter."""
        ...


@runtime_checkable
class HfCausalLMOutput(Protocol):
    """The causal-LM output field compared by parity tests."""

    logits: Tensor


def hf_config() -> dict[str, object]:
    """Return the CPU-tiny Qwen3.5 text configuration used by tests.

    Returns:
      config: Configuration values for the CPU-tiny Qwen3.5 text model.

    """
    return {
        "model_type": "qwen3_5_text",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10_000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
    }


def native_qwen35_config() -> Qwen35.Config:
    """Return the native test config with separate SwiGLU projection kernels.

    Returns:
      config: Native Qwen3.5 configuration for HF-reference parity checks.

    """
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    for block in config.block:
        assert isinstance(block, TransformerBlock.Config)
        assert isinstance(block.ffn, SwiGLU.Config)
        block.ffn.split_gate_projection = True
    return config


def hf_reference(value: object, *, dtype: torch.dtype | None = None) -> HfReference:
    """Narrow a lazy Hugging Face Qwen model at the exercised test boundary.

    Args:
      value: Lazily imported Hugging Face Qwen model.
      dtype: Optional reference computation dtype.

    Returns:
      reference: Runtime-checked Qwen interface used by parity tests.

    """
    assert isinstance(value, nn.Module)
    if dtype is not None:
        value = value.to(dtype=dtype)
    for name, module in value.named_modules():
        if name.endswith("linear_attn"):
            torch_reference(module)
    assert isinstance(value, HfReference)
    return value


def hf_logits(value: object) -> Tensor:
    """Return the logits tensor from one Hugging Face causal-LM result."""
    assert isinstance(value, HfCausalLMOutput)
    return value.logits


def hf_tensor(value: object) -> Tensor:
    """Return a tensor result from one Hugging Face reference call."""
    assert isinstance(value, Tensor)
    return value


# Hub and installed FLA kernels are bypassed only for this reference instance.
def torch_reference(reference: nn.Module) -> nn.Module:
    """Bind the installed HF bytecode to its PyTorch kernel fallbacks.

    Every function of the forward's decorator chain is copied, each reaching the
    next copy, so a decorator still runs: HF's ``deprecate_kwarg`` reads
    ``cache_params`` as ``cache``, and without it the keyword is dropped.

    Args:
      reference: Hugging Face Qwen3.5 delta-attention module to adapt.

    Returns:
      reference: The same module with PyTorch fallback functions bound.

    Raises:
      TypeError: A function of the chain is no plain function, or a kernel the
        innermost names is not callable.

    """
    bound = cast(object, reference.forward)
    chain = [bound.__func__ if isinstance(bound, MethodType) else bound]
    while (inner := cast(object, getattr(chain[-1], "__wrapped__", None))) is not None:
        chain.append(inner)
    functions = [f for f in chain if isinstance(f, FunctionType)]
    if len(functions) != len(chain):
        raise TypeError("Expected the reference forward to be a plain function.")
    kernels = {
        name: _fallback(cast(object, functions[-1].__globals__[name]))
        for name in (
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
            "causal_conv1d_update",
        )
    }
    # The exact upstream code runs with private function bindings; installed FLA
    # otherwise selects CUDA kernels even for this CPU reference.
    copies: dict[int, FunctionType] = {}
    for function in reversed(functions):
        copies[id(function)] = _copied(function, kernels=kernels, copies=copies)
    reference.forward = partial(copies[id(functions[0])], reference)
    return reference


def _fallback(kernel: object) -> object:
    """Return the PyTorch function an FLA kernel, or its module wrapper, falls back to."""
    if isinstance(kernel, nn.Module):
        kernel = cast(object, inspect.getclosurevars(kernel.forward).nonlocals["func"])
    if not callable(kernel):
        raise TypeError("Expected callable(kernel).")
    return cast(object, inspect.unwrap(kernel))


# Globals, cells, and a function's ``__dict__`` hold anything: typeshed types them
# ``Any``, shed here as ``object``.
def _copied(
    function: FunctionType,
    *,
    kernels: dict[str, object],
    copies: dict[int, FunctionType],
) -> FunctionType:
    """Return ``function`` with the fallbacks bound, reaching each copy made so far."""
    names = {
        name: copies.get(id(value), kernels.get(name, value))
        for name, value in cast(dict[str, object], function.__globals__).items()
    }
    cells = tuple(
        _cell(cast(object, cell.cell_contents), cell=cell, copies=copies)
        for cell in function.__closure__ or ()
    )
    copy = FunctionType(
        function.__code__,
        names,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=cells or None,
    )
    copy.__kwdefaults__ = function.__kwdefaults__
    for key, value in cast(dict[str, object], function.__dict__).items():
        copy.__dict__[key] = copies.get(id(value), value)
    return copy


def _cell(
    contents: object,
    *,
    cell: CellType,
    copies: dict[int, FunctionType],
) -> CellType:
    """Return a cell holding the copy of ``contents``, else ``cell`` itself."""
    return CellType(copies[id(contents)]) if id(contents) in copies else cell
