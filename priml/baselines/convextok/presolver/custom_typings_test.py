"""Declared structure of the presolver's kernel buffer types."""

from __future__ import annotations

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
    assert Buf.size.fget is not None
    assert Buf.size.fget(buf) is None
    assert Buf.copy(buf) is None
    assert Buf.astype(buf, np.float64) is None


def test_buffer_aliases() -> None:
    assert FBuf.__value__ == Buf[float]
    assert IBuf.__value__ == Buf[int]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
