"""Checks for the products module."""

from __future__ import annotations

from priml.baselines.casmi2026.scripts.products import unique_products


def _name(product: str) -> str:
    if product == "invalid":
        raise ValueError("Rejected product.")
    return product.lower()


def test_each_group_contributes_its_first_product_once_per_canonical_name() -> None:
    groups = [("A", "ignored"), ("a",), ("B",), ("invalid",), ("skip",), ("C",)]
    assert unique_products(
        groups,
        canonical=_name,
        limit=10,
        keep=lambda product: product != "skip",
    ) == [("A", "a"), ("B", "b"), ("C", "c")]


def test_the_cap_counts_accepted_products_and_zero_still_admits_the_first() -> None:
    groups = [("invalid",), ("A",), ("B",), ("C",)]
    assert unique_products(groups, canonical=_name, limit=2) == [
        ("A", "a"),
        ("B", "b"),
    ]
    assert unique_products(groups, canonical=_name, limit=0) == [("A", "a")]
    assert unique_products([], canonical=_name, limit=3) == []


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
