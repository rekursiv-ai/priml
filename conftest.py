"""Repo-root conftest - applies to every pytest process including xdist workers.

Holds the private-repo-only setup: git-variable stripping, developer secrets,
PGlite limits, and the CI mark policy. Everything priml also needs -- the
native-thread caps and the ``warm_pools`` fixture -- is defined in priml's own
package conftest and imported here, so the two cannot drift and the public
export carries the same test behavior.
"""

from __future__ import annotations

import os
import pathlib

import pytest

# Importing priml's conftest caps the native math threads as a side effect of
# module execution, which must happen before torch loads -- and does, since
# conftest is imported before any test module. ``warm_pools`` is re-exported
# because tests OUTSIDE priml (experimental/sudoku) see only this root conftest,
# and pytest collects fixtures by module attribute.
from priml.conftest import warm_pools


__all__ = ["warm_pools"]


# Git hooks export local-repo variables. If pytest inherits them, subprocess
# git commands in temp dirs can still mutate the caller repo's `.git/config`.
for _var in (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
):
    os.environ.pop(_var, None)


# PGlite integration tests boot many short-lived Node-backed engines under
# xdist. Four cold-start slots kept the full integration gate below 60s locally
# without reintroducing readiness failures; production keeps substrate defaults.
os.environ.setdefault("LOOP_PGLITE_MAX_BOOTS", "4")


# Live integration tests read service URLs / API keys from the environment
# (e.g. SEARXNG_URL). Those secrets live in the developer's ``~/.secrets``,
# sourced by an interactive ``.bashrc`` -- so a NONINTERACTIVE harness (pytest,
# spawned subagents, an xdist worker) never inherits them and the dependent
# tests skip even though the service works. Load ``~/.secrets`` here so every
# pytest process sees the same values regardless of the launching shell. Only
# ``export NAME=VALUE`` lines are honored, and an already-set variable always
# wins (an explicit ``SEARXNG_URL=... pytest`` override is never clobbered).
def _load_developer_secrets() -> None:
    """Populate the environment from ``~/.secrets`` for noninteractive runs."""
    secrets_path = pathlib.Path.home() / ".secrets"
    try:
        lines = secrets_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        line = line.removeprefix("export ")
        name, sep, value = line.partition("=")
        name = name.strip()
        if not sep or not name.isidentifier() or name in os.environ:
            continue
        os.environ[name] = value.strip().strip("'\"")


_load_developer_secrets()


# Deterministic CLI text under test: with FORCE_COLOR exported (some dev
# shells set FORCE_COLOR=3), Python 3.14's argparse colorizes --help/usage
# and every exact-text assertion (e.g. jobber's `usage: jobber` checks)
# fails on invisible ANSI noise. PYTHON_COLORS=0 outranks FORCE_COLOR for
# Python's own colorizer and propagates to subprocess launcher tests. Set
# unconditionally: inside pytest, deterministic bytes beat pretty colors.
os.environ["PYTHON_COLORS"] = "0"


# Marks whose tests need live external credentials, services, real GPUs, or
# timing-sensitive perf runs. Hosted CI has none of those, so skip them there
# instead of failing. Set RUN_INTEGRATION=1 in a CI job that has provisioned the
# secrets/devices to opt back in. This gates at collection time so it holds no
# matter how pytest is invoked -- including ``bin/tests``, which otherwise
# overrides the default marker deselection.
_CI_SKIPPED_MARKS = (
    "cuda",
    "integration",
    "performance",
)  # config-globals: ignore -- CI mark policy.


# Live model-CLI tests can touch OS credential stores even before they fail or
# skip on auth. Keep them out of noninteractive gates unless explicitly enabled.
_REAL_LLM_ENV_VAR = "RUN_REAL_LLM"  # config-globals: ignore -- pytest opt-in env var.


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip environment-dependent marks unless the job provisions dependencies."""
    del config
    for item in items:
        if item.get_closest_marker("real_llm") is not None and not os.environ.get(
            _REAL_LLM_ENV_VAR
        ):
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        "real_llm test skipped"
                        f" (set {_REAL_LLM_ENV_VAR}=1 to run live model CLIs)"
                    )
                )
            )
            continue
        if os.environ.get("CI") and not os.environ.get("RUN_INTEGRATION"):
            for mark in _CI_SKIPPED_MARKS:
                if item.get_closest_marker(mark) is not None:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=(
                                f"{mark} test skipped in CI"
                                " (no live credentials/services/devices;"
                                " set RUN_INTEGRATION=1 to opt in)"
                            )
                        )
                    )
                    break
