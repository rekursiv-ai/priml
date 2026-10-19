"""Pin the strings pandas reads as missing values in a CSV."""

from __future__ import annotations

from priml.data.na_values import PANDAS_NA_VALUES


def test_na_values_are_the_nineteen_tokens_the_baselines_meet() -> None:
    assert all(isinstance(token, str) for token in PANDAS_NA_VALUES)
    assert {"", "NA", "N/A", "NaN", "nan", "null", "NULL", "None"} <= PANDAS_NA_VALUES
    assert len(PANDAS_NA_VALUES) == 19
    assert "0" not in PANDAS_NA_VALUES
    assert "NAN" not in PANDAS_NA_VALUES


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
