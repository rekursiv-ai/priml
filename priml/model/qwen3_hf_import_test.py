"""Import-hermeticity guard for the Qwen3 parity test."""

from __future__ import annotations

import importlib.util

import torch

from priml.model import qwen3_hf_test


def test_importing_parity_module_does_not_enable_global_determinism() -> None:
    # qwen3_hf_test must not flip process-global torch determinism state at
    # import/collection time: doing so silently changes numerics for every
    # other test collected in the same worker, even when this MPS-skipped
    # integration test never runs. Execute the module body in a fresh module
    # object and confirm it leaves determinism untouched.
    torch.use_deterministic_algorithms(False)
    spec = importlib.util.spec_from_file_location(
        "_qwen3_hf_import_probe", qwen3_hf_test.__file__
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert not torch.are_deterministic_algorithms_enabled()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
