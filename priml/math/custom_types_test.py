"""Tests for the math type aliases and the jaxtyping entry point."""

from __future__ import annotations

from typing import get_args

from jaxtyping import jaxtyped
from torch import Tensor
from typeguard import typechecked

import numpy as np

from priml.math.custom_types import Numeric, Tensorable, jaxtypechecked


def test_jaxtypechecked_binds_typeguard_as_the_checker() -> None:
    assert jaxtypechecked.func is jaxtyped  # pyrefly: ignore[missing-attribute] -- pyrefly types a `functools.partial` of an overload as the overload, dropping `partial`'s attributes.
    assert jaxtypechecked.args == ()  # pyrefly: ignore[missing-attribute] -- pyrefly types a `functools.partial` of an overload as the overload, dropping `partial`'s attributes.
    assert jaxtypechecked.keywords == {"typechecker": typechecked}  # pyrefly: ignore[missing-attribute] -- pyrefly types a `functools.partial` of an overload as the overload, dropping `partial`'s attributes.


def test_numeric_admits_python_and_numpy_scalars() -> None:
    assert {bool, int, float, complex, np.number, np.bool_} == set(get_args(Numeric))


def test_tensorable_admits_tensors_and_numerics() -> None:
    members = get_args(Tensorable)
    assert Tensor in members
    assert set(get_args(Numeric)) <= set(members)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
