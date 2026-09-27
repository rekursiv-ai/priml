"""Tests of the Triton binding for modules that import Triton lazily."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pytest
import torch

from priml.kernel import jit_kernel, require_power_of_two


if TYPE_CHECKING:
    from triton import language
else:
    from wrapt import lazy_import

    language = lazy_import("triton.language")


@runtime_checkable
class _Hashed(Protocol):
    """The cache key Triton hashes a kernel's source and globals into."""

    @property
    def cache_key(self) -> str: ...


def test_a_bound_kernel_hashes_its_helper_and_the_modules() -> None:
    # The hash walks the kernel's globals, which cannot copy a lazy proxy; the
    # binding is what puts the real modules there.
    pytest.importorskip("triton")
    kernel = jit_kernel(_double_triton, _twice_triton=jit_kernel(_twice_triton))
    assert isinstance(kernel, _Hashed)
    assert isinstance(kernel.cache_key, str)


@pytest.mark.gpu_triton
def test_a_bound_kernel_calls_its_helper() -> None:
    if not torch.cuda.is_available():
        pytest.skip("the kernel needs a CUDA device")
    kernel = jit_kernel(_double_triton, _twice_triton=jit_kernel(_twice_triton))
    values = torch.arange(8, dtype=torch.float32, device="cuda")
    kernel[(1,)](values, block=8)
    assert torch.equal(values.cpu(), torch.arange(8, dtype=torch.float32) * 2)


def test_a_launch_size_must_be_a_positive_power_of_two() -> None:
    require_power_of_two(block=1, num_warps=4)
    for size in (0, -2, 6):
        with pytest.raises(ValueError, match=f"num_warps .* not {size}"):
            require_power_of_two(block=8, num_warps=size)


def _twice_triton(x: language.tensor) -> language.tensor:
    return x + x


def _double_triton(values_ptr: language.tensor, block: language.constexpr) -> None:
    index = language.arange(0, block)
    doubled = _twice_triton(language.load(values_ptr + index))
    language.store(values_ptr + index, doubled)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
