"""Tests for the openai/math Harbor agent adapter."""

from __future__ import annotations

import pytest

from priml.baselines.openaimath.scripts.runner import AGENT_IMPORT_PATH


pytest.importorskip(
    "harbor",
    reason="Harbor runs the rollout trials; it is not a PriML dependency.",
)

from priml.baselines.openaimath.scripts.agent import OpenMathLeanAgent


def test_the_agent_names_itself() -> None:
    """Harbor selects the agent by this name."""
    assert OpenMathLeanAgent.name() == "openmath-lean"


def test_the_runner_names_this_agent() -> None:
    """The rollout runner launches exactly this agent class."""
    assert AGENT_IMPORT_PATH == (
        "priml.baselines.openaimath.scripts.agent:OpenMathLeanAgent"
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
