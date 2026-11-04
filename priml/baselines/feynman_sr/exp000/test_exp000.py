"""Test script for exp000 baseline - verifies bit-for-bit reproducibility."""

import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    ),
)

from priml.baselines.feynman_sr.exp000.experiments import exp000


def test_exp000_reproducibility() -> None:
    """Run exp000 twice and verify identical results."""
    # Create smoke test config: smaller population, fewer generations
    cfg1 = exp000()
    cfg1.step.model.ga_config.population_size = 20
    cfg1.step.model.ga_config.max_generations = 5
    cfg1.step.n_samples = 100
    result1 = cfg1.run()

    cfg2 = exp000()
    cfg2.step.model.ga_config.population_size = 20
    cfg2.step.model.ga_config.max_generations = 5
    cfg2.step.n_samples = 100
    result2 = cfg2.run()

    # Check bit-for-bit reproducibility
    assert result1["recovery_rate"] == result2["recovery_rate"], (
        f"Recovery rate differs: {result1['recovery_rate']} vs {result2['recovery_rate']}"
    )
    assert result1["exact_recoveries"] == result2["exact_recoveries"], (
        f"Exact recoveries differs: {result1['exact_recoveries']} vs {result2['exact_recoveries']}"
    )

    # Check individual results match
    for r1, r2 in zip(result1["results"], result2["results"], strict=False):
        assert r1["nmse"] == r2["nmse"], f"NMSE differs for {r1['eq_id']}"
        assert r1["expression"] == r2["expression"], (
            f"Expression differs for {r1['eq_id']}"
        )
        assert r1["exact"] == r2["exact"], f"Exact flag differs for {r1['eq_id']}"

    valid_results = [r for r in result1["results"] if r["nmse"] != float("inf")]
    if valid_results:
        sum(r["nmse"] for r in valid_results) / len(valid_results)

    # Print per-equation results
    for r in result1["results"]:
        "✓" if r["exact"] else "✗"


if __name__ == "__main__":
    test_exp000_reproducibility()
