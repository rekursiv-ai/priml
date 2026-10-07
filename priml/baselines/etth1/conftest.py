"""Options for explicitly minting ETTh1 goldens from the pinned source."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the source paths used only when minting goldens."""
    parser.addoption(
        "--etth1-reference",
        help="Pinned DLinear source checkout for minting.",
    )
    parser.addoption(
        "--etth1-directory",
        help="Prepared ETTh1 data for source verification.",
    )
