"""Declared structure of the presolver's kernel buffer types."""

from __future__ import annotations

from collections.abc import Callable
from inspect import getattr_static
from typing import TypeVar, cast
from typing_extensions import get_protocol_members

import numpy as np

from priml.baselines.convextok.presolver.custom_typings import (
    Buf,
    FBuf,
    IBuf,
)


def test_buf_protocol_members() -> None:
    assert get_protocol_members(Buf) == {
        "size",
        "__getitem__",
        "__setitem__",
        "__len__",
        "__array__",
        "copy",
        "astype",
    }
    (param,) = Buf.__type_params__
    assert isinstance(param, TypeVar)
    assert param.__constraints__ == (int, float)


def test_buf_default_bodies_return_none() -> None:
    buf = cast(Buf[int], object())
    size = cast(property, getattr_static(Buf, "size"))
    assert size.fget is not None
    assert cast(Callable[[object], object], size.fget)(buf) is None
    copy = cast(Callable[[Buf[int]], object], getattr_static(Buf, "copy"))
    astype = cast(
        Callable[[Buf[int], type[np.float64]], object],
        getattr_static(Buf, "astype"),
    )
    assert copy(buf) is None
    assert astype(buf, np.float64) is None


def test_buffer_aliases() -> None:
    assert cast(object, FBuf.__value__) == Buf[float]
    assert cast(object, IBuf.__value__) == Buf[int]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
