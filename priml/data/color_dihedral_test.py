"""Shared color permutation and square symmetry for a puzzle's grids."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import re

import numpy as np
import pytest
import torch

from priml.data import color_dihedral
from priml.data.color_dihedral import ColorDihedral, dihedral_transform


def test_token_identity_without_colors() -> None:
    config = ColorDihedral.Config()
    config.colors = ()
    config.transforms = (0,)
    grid = torch.arange(4).repeat(3, 1)
    inputs, labels = config.make().augment_tokens(
        grid,
        grid.flip(1),
        vocab_size=7,
        token_offset=2,
        generator=torch.Generator().manual_seed(42),
    )
    assert torch.equal(inputs, grid)
    assert torch.equal(labels, grid.flip(1))


def test_unfilled_identifier_separator_rejected() -> None:
    policy = ColorDihedral.Config().make()
    with pytest.raises(
        ValueError,
        match=re.escape("identifier separator must be filled before sampling."),
    ):
        policy.sample("task", rng=np.random.default_rng(0))
    with pytest.raises(
        ValueError,
        match=re.escape("identifier separator must be filled before decoding."),
    ):
        policy.inverse("task")


def test_dihedral_transforms_have_exact_orientation() -> None:
    grid = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    expected = (
        grid,
        np.array([[3, 6], [2, 5], [1, 4]], dtype=np.uint8),
        np.array([[6, 5, 4], [3, 2, 1]], dtype=np.uint8),
        np.array([[4, 1], [5, 2], [6, 3]], dtype=np.uint8),
        np.array([[3, 2, 1], [6, 5, 4]], dtype=np.uint8),
        np.array([[4, 5, 6], [1, 2, 3]], dtype=np.uint8),
        np.array([[1, 4], [2, 5], [3, 6]], dtype=np.uint8),
        np.array([[6, 3], [5, 2], [4, 1]], dtype=np.uint8),
    )
    for tid, transformed in enumerate(expected):
        assert np.array_equal(dihedral_transform(grid, tid=tid), transformed)
    with pytest.raises(ValueError, match=r"Invalid dihedral tid=8; must be in 0\.\.7"):
        dihedral_transform(grid, tid=8)


def test_token_augmentation_passes_input_device_to_symmetry_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[torch.device] = []
    original = color_dihedral._token_symmetries

    def spy(
        side: int,
        transforms: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        calls.append(device)
        return original(side, transforms, device)

    monkeypatch.setattr(color_dihedral, "_token_symmetries", spy)
    policy = ColorDihedral.Config(transforms=(1,), colors=()).make()
    inputs = torch.arange(4).repeat(2, 1)
    policy.augment_tokens(
        inputs,
        inputs,
        vocab_size=4,
        token_offset=0,
    )
    assert calls == [inputs.device]


def test_token_augmentation_pins_values_dtypes_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, list[tuple[tuple[object, ...], dict[str, object]]]] = {}
    for name in ("tensor", "arange", "rand", "randint"):
        original = cast(Callable[..., object], getattr(torch, name))

        def spy(
            *args: object,
            name: str = name,
            original: Callable[..., object] = original,
            **kwargs: object,
        ) -> object:
            calls.setdefault(name, []).append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(torch, name, spy)
    policy = ColorDihedral.Config(transforms=(0,), colors=(1, 3)).make()
    inputs = torch.tensor([[0, 3, 2, 5], [5, 2, 3, 0]], dtype=torch.int16)
    labels = torch.tensor([[5, 3, 2, 0], [0, 2, 3, 5]], dtype=torch.int32)
    generator = torch.Generator().manual_seed(4)
    expected_generator = torch.Generator().manual_seed(4)
    # Token augmentation draws one score per row and selected color symbol.
    expected_draws = torch.rand((2, 2), generator=expected_generator)
    expected_choice = torch.randint(0, 1, (2,), generator=expected_generator)
    assert torch.equal(expected_choice, torch.zeros(2, dtype=torch.long))
    calls.clear()
    augmented_inputs, augmented_labels = policy.augment_tokens(
        inputs,
        labels,
        vocab_size=7,
        token_offset=2,
        generator=generator,
    )
    assert calls["tensor"][0] == (
        ((1, 3),),
        {"device": inputs.device, "dtype": torch.long},
    )
    assert calls["arange"][0] == (
        (7,),
        {"device": inputs.device, "dtype": torch.long},
    )
    assert calls["rand"][0] == (
        (2, 2),
        {"device": inputs.device, "generator": generator},
    )
    assert calls["randint"][0] == (
        (0, 1, (2,)),
        {"device": inputs.device, "generator": generator},
    )
    symbols = torch.tensor([3, 5])
    permutations = torch.arange(7).expand(2, -1).clone()
    permutations[:, symbols] = symbols[expected_draws.argsort(dim=1)]
    assert torch.equal(augmented_inputs, permutations.gather(1, inputs.long()))
    assert torch.equal(augmented_labels, permutations.gather(1, labels.long()))
    assert augmented_inputs.dtype == torch.int16
    assert augmented_labels.dtype == torch.int32
    assert torch.equal(
        torch.rand(3, generator=generator),
        torch.rand(3, generator=expected_generator),
    )
    with pytest.raises(ValueError, match="square grids"):
        policy.augment_tokens(
            inputs[:, :3],
            labels[:, :3],
            vocab_size=5,
            token_offset=0,
        )


def test_color_dihedral_inverse_covers_each_dihedral_transform() -> None:
    grid = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    for tid in range(8):
        config = ColorDihedral.Config(
            separator="|||",
            transforms=(tid,),
            colors=(),
        ).make()
        name, forward = config.sample("task", rng=np.random.default_rng(0))
        augmented = forward(grid)
        original, inverse = config.inverse(name)
        assert original == "task"
        assert np.array_equal(inverse(augmented), grid)
        assert not np.shares_memory(inverse(augmented), augmented)


def test_color_dihedral_exact_encoded_view_and_inverse_prefix() -> None:
    config = ColorDihedral.Config(separator="::", transforms=(1,), colors=(1, 2, 3))
    name, transform = config.make().sample("task::part", rng=np.random.default_rng(0))
    assert name == "task::part::t1::0312456789"
    grid = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    augmented = transform(grid)
    assert np.array_equal(augmented, [[2, 6], [1, 5], [3, 4]])
    assert augmented.dtype == np.uint8
    original, inverse = config.make().inverse(name)
    assert original == "task"
    restored = inverse(augmented)
    assert np.array_equal(restored, grid)
    assert restored.dtype == np.uint8
    with pytest.raises(ValueError, match="expected a permutation of '0123456789'"):
        config.make().inverse("x::t0::0123456788")
    with pytest.raises(ValueError, match=r"invalid literal for int\(\)"):
        config.make().inverse("x::bad::0123456789")


def test_color_dihedral_rejects_invalid_config_and_keeps_unlisted_colors() -> None:
    for transforms, colors, message in (
        ((), (1,), "transforms must contain identifiers in 0..7."),
        ((8,), (1,), "transforms must contain identifiers in 0..7."),
        ((0,), (-1,), "colors must contain distinct identifiers in 0..9."),
        ((0,), (1, 1), "colors must contain distinct identifiers in 0..9."),
        ((0,), (10,), "colors must contain distinct identifiers in 0..9."),
    ):
        config = ColorDihedral.Config()
        config.transforms = transforms
        config.colors = colors
        with pytest.raises(ValueError, match=re.escape(message)):
            config.make()
    zero_color = ColorDihedral.Config(
        separator="|||",
        transforms=(0,),
        colors=(0,),
    ).make()
    zero_name, zero_transform = zero_color.sample("zero", rng=np.random.default_rng(1))
    assert zero_name == "zero|||t0|||0123456789"
    assert np.array_equal(zero_transform(np.array([[0, 1]], dtype=np.uint8)), [[0, 1]])
    config = ColorDihedral.Config(separator="|||", transforms=(0,), colors=(1, 3))
    name, transform = config.make().sample("q", rng=np.random.default_rng(3))
    assert name.startswith("q|||t0|||")
    grid = np.array([[0, 1, 2, 3, 9]], dtype=np.uint8)
    augmented = transform(grid)
    assert augmented[0, 0] == 0
    assert augmented[0, 2] == 2
    assert augmented[0, 4] == 9
    decoded, inverse = config.make().inverse(name)
    assert decoded == "q"
    assert np.array_equal(inverse(augmented), grid)


def test_color_dihedral_errors_match_exact_messages() -> None:
    invalid_calls = (
        (
            lambda: ColorDihedral.Config(transforms=()).make(),
            "transforms must contain identifiers in 0..7.",
        ),
        (
            lambda: ColorDihedral.Config(colors=(10,)).make(),
            "colors must contain distinct identifiers in 0..9.",
        ),
        (
            lambda: (
                ColorDihedral.Config()
                .make()
                .augment_tokens(
                    torch.zeros((2, 3)),
                    torch.zeros((2, 3)),
                    vocab_size=4,
                    token_offset=0,
                )
            ),
            "Token rows must represent square grids.",
        ),
        (
            lambda: (
                ColorDihedral.Config(separator="|||")
                .make()
                .inverse("x|||t0|||0123456788")
            ),
            (
                "invalid color-permutation suffix '0123456788' in identifier "
                "'x|||t0|||0123456788'; expected a permutation of '0123456789'."
            ),
        ),
        (
            lambda: (
                ColorDihedral.Config(separator="|||")
                .make()
                .inverse("x|||t8|||0123456789")
            ),
            "Encoded transform must be in 0..7.",
        ),
    )
    for call, expected in invalid_calls:
        with pytest.raises(ValueError, match=r".+") as error:
            call()
        assert str(error.value) == expected
    for call, expected in (
        (
            lambda: (
                ColorDihedral.Config()
                .make()
                .sample("task", rng=np.random.default_rng(0))
            ),
            "identifier separator must be filled before sampling.",
        ),
        (
            lambda: ColorDihedral.Config().make().inverse("task"),
            "identifier separator must be filled before decoding.",
        ),
    ):
        with pytest.raises(ValueError, match=r".+") as error:
            call()
        assert str(error.value) == expected


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
