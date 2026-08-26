"""Readable golden-file assertions for Priml tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import pytest


def assert_text_golden(
    request: pytest.FixtureRequest,
    *,
    test_file: str,
    name: str,
    rendered: str,
) -> None:
    """Assert rendered text matches its test-local golden.

    Args:
      request: Active pytest request carrying ``--golden-overwrite``.
      test_file: ``__file__`` of the owning test module.
      name: Golden filename without its extension.
      rendered: Snapshot text without a trailing newline.

    """
    golden = Path(test_file).resolve().parent / "testdata" / f"{name}.txt"
    missing = not golden.exists()
    if missing or request.config.getoption("--golden-overwrite", default=False):
        golden.parent.mkdir(parents=True, exist_ok=True)
        _ = golden.write_text(rendered + "\n", encoding="utf-8")
    if missing:
        raise AssertionError(
            f"Missing golden regenerated at {golden}; inspect it, then rerun the test."
        )
    assert golden.read_text(encoding="utf-8") == rendered + "\n", (
        f"{name} changed; read the diff, then rerun with --golden-overwrite "
        "if the change is intended."
    )
