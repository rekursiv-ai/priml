"""Select unique sanitized reaction products in the scientific library's order."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


def unique_products[Product](
    groups: Iterable[Sequence[Product]],
    *,
    canonical: Callable[[Product], str],
    limit: int,
    keep: Callable[[Product], bool] | None = None,
) -> list[tuple[Product, str]]:
    """Keep the first product of each group, once per canonical string.

    Args:
      groups: Product tuples of every reaction outcome, in library order.
      canonical: Sanitizes a product and names it; raising rejects the product.
      limit: Unique-product cap; zero still admits the first product.
      keep: Pre-filter applied to a product before it is sanitized; None keeps all.

    Returns:
      products: Accepted products with their canonical strings, up to the cap.

    """
    seen: set[str] = set()
    result: list[tuple[Product, str]] = []
    for group in groups:
        product = group[0]
        if keep is not None and not keep(product):
            continue
        try:
            name: str | None = canonical(product)
        except Exception:  # noqa: BLE001 -- The scientific extension rejects invalid products individually.
            name = None
        if name is None or name in seen:
            continue
        seen.add(name)
        result.append((product, name))
        if len(result) >= limit:
            break
    return result
