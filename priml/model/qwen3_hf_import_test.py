"""Hermeticity guards for the Qwen3 parity test."""

from __future__ import annotations

from unittest.mock import Mock

import importlib.util
import os

import pytest
import torch

from priml.model import qwen3_hf_test


@pytest.mark.parametrize(
    ("algorithms_enabled", "warn_only_enabled"),
    [(False, False), (True, True)],
)
def test_importing_parity_module_preserves_global_determinism(
    algorithms_enabled: bool,
    warn_only_enabled: bool,
) -> None:
    # qwen3_hf_test must not flip process-global torch determinism state at
    # import/collection time: doing so silently changes numerics for every
    # other test collected in the same worker, even when this MPS-skipped
    # integration test never runs. Execute the module body in a fresh module
    # object and confirm it leaves determinism untouched.
    original_algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )
        spec = importlib.util.spec_from_file_location(
            "_qwen3_hf_import_probe", qwen3_hf_test.__file__
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert torch.are_deterministic_algorithms_enabled() == algorithms_enabled
        assert (
            torch.is_deterministic_algorithms_warn_only_enabled() == warn_only_enabled
        )
    finally:
        torch.use_deterministic_algorithms(
            original_algorithms_enabled,
            warn_only=original_warn_only_enabled,
        )


def test_parity_test_restores_process_state_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    flash_sdp_enabled = torch.backends.cuda.flash_sdp_enabled()
    memory_efficient_sdp_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
    rng_state = torch.get_rng_state()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(983)
        torch.set_rng_state(generator.get_state())
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        cudnn_available = Mock(return_value=True)
        cuda_available = Mock(return_value=True)
        enable_flash_sdp = Mock(wraps=torch.backends.cuda.enable_flash_sdp)
        enable_mem_efficient_sdp = Mock(
            wraps=torch.backends.cuda.enable_mem_efficient_sdp
        )
        monkeypatch.setattr(torch.backends.cudnn, "is_available", cudnn_available)
        monkeypatch.setattr(torch.cuda, "is_available", cuda_available)
        monkeypatch.setattr(
            torch.backends.cuda,
            "enable_flash_sdp",
            enable_flash_sdp,
        )
        monkeypatch.setattr(
            torch.backends.cuda,
            "enable_mem_efficient_sdp",
            enable_mem_efficient_sdp,
        )
        expected_rng_state = torch.get_rng_state()
        monkeypatch.setattr(
            qwen3_hf_test,
            "_build_hf_model",
            Mock(side_effect=RuntimeError("setup failed")),
        )

        with pytest.raises(RuntimeError, match="setup failed"):
            qwen3_hf_test.test_qwen3_matches_hf(False)

        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.benchmark
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cuda.flash_sdp_enabled()
        assert torch.backends.cuda.mem_efficient_sdp_enabled()
        assert torch.equal(torch.get_rng_state(), expected_rng_state)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
        cudnn_available.assert_not_called()
        cuda_available.assert_not_called()
        enable_flash_sdp.assert_not_called()
        enable_mem_efficient_sdp.assert_not_called()
    finally:
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.enable_flash_sdp(flash_sdp_enabled)
        torch.backends.cuda.enable_mem_efficient_sdp(memory_efficient_sdp_enabled)
        torch.set_rng_state(rng_state)


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
