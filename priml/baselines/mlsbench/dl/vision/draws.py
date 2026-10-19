"""Count the random draws of a block of code by API and by calling function.

A bit-for-bit comparison is only as strong as the draws it reaches: a random
call site that no input drives is compared against nothing. Counting the calls
of each torch sampler, keyed by the function that made them, shows which sites
a run reached.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest import mock

import functools
import inspect
import traceback


if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


@contextmanager
def recorded_draws() -> Generator[dict[str, int]]:
    """Count the draws torch's samplers make inside the block.

    Draws made in other processes, such as the workers of a data loader, are
    not seen, nor are draws from numpy's or Python's generators.

    Yields:
      counts: Maps ``"<api>@<file>:<function>"``, the sampler and the last two
        components of its caller's path and its name, to the number of calls.

    """
    counts: dict[str, int] = {}
    with ExitStack() as stack:
        for api in (
            "randint",
            "rand",
            "randperm",
            "randn",
            "normal",
            "bernoulli",
            "multinomial",
            "rand_like",
            "randn_like",
            "randint_like",
        ):
            counted = _Counted(
                cast("Callable[..., object]", getattr(torch, api)),
                api,
                counts,
            )
            stack.enter_context(mock.patch.object(torch, api, counted))
        for api in (
            "uniform_",
            "normal_",
            "random_",
            "bernoulli_",
            "exponential_",
            "geometric_",
            "log_normal_",
            "cauchy_",
        ):
            counted = _Counted(
                cast("Callable[..., object]", getattr(torch.Tensor, api)),
                api,
                counts,
            )
            stack.enter_context(mock.patch.object(torch.Tensor, api, counted))
        yield counts


class _Counted:
    """Calls ``original`` and tallies the function that called this."""

    def __init__(
        self,
        original: Callable[..., object],
        api: str,
        counts: dict[str, int],
    ) -> None:
        self.original = original
        self.api = api
        self.counts = counts

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Count the call, then make it."""
        # The caller is the nearest frame outside this file and outside any
        # trampoline a mutation tester wraps around ``__call__``. Walking the
        # live frames reads no source, unlike ``traceback.extract_stack``. The
        # walk starts at this frame because ``walk_stack(None)`` starts at a
        # different one per version: its caller on 3.14, but four frames up on
        # 3.12, the export's floor, where every key named the caller's caller.
        frame = next(
            frame
            for frame, _ in traceback.walk_stack(inspect.currentframe())
            if frame.f_code.co_filename != __file__
            and Path(frame.f_code.co_filename).name != "trampoline.py"
        )
        file = "/".join(Path(frame.f_code.co_filename).parts[-2:])
        key = f"{self.api}@{file}:{frame.f_code.co_name}"
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.original(*args, **kwargs)

    def __get__(
        self,
        instance: object,
        owner: type | None = None,
    ) -> Callable[..., object]:
        """Bind to a tensor the way the method it replaces does."""
        del owner
        return self if instance is None else functools.partial(self, instance)
