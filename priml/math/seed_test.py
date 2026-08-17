from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast
from unittest.mock import MagicMock

import inspect
import os
import random

import numpy as np
import pytest
import torch

from priml.math import seed
from priml.math.seed import (
    RngState,
    dataloader_worker_init_fn,
    enable_determinism,
    get_rng_state,
    make_seed,
    numpy_rng,
    salt,
    set_rng_state,
    set_seed_distributed,
    set_seed_local,
)


def test_make_seed_returns_nonzero():
    seed = make_seed()
    assert seed > 0
    assert isinstance(seed, int)
    assert seed < 2**63  # PyTorch / numpy accept 64-bit seeds


def test_make_seed_distinct_consecutive_calls():
    """Successive calls must differ. OS entropy makes this overwhelmingly
    likely; the failure probability is ~2^-63.
    """
    seeds = {make_seed() for _ in range(100)}
    assert len(seeds) == 100


def test_set_seed_with_explicit_value():
    seed = set_seed_local(seed=42)
    assert seed == 42
    assert torch.initial_seed() == salt("torch", 42)


def test_set_seed_generates_when_none():
    seed = set_seed_local(seed=None)
    assert seed > 0
    assert torch.initial_seed() == salt("torch", seed)


def test_set_seed_reproducibility():
    set_seed_local(seed=123)
    rand1 = torch.rand(5)
    np_rand1 = numpy_rng.random(5)
    py_rand1 = random.random()  # noqa: S311

    set_seed_local(seed=123)
    rand2 = torch.rand(5)
    np_rand2 = numpy_rng.random(5)
    py_rand2 = random.random()  # noqa: S311

    torch.testing.assert_close(rand1, rand2)
    assert (np_rand1 == np_rand2).all()
    assert py_rand1 == py_rand2


@pytest.mark.cuda
def test_set_seed_reproducibility_on_cuda() -> None:
    """The CPU reproducibility check above misses the case readers
    actually rely on -- CUDA RNG reproducibility. Cover at least the
    single-device path so a regression in the per-device seeding loop
    fires here, not only in the ``device_count() >= 2`` test that
    skips on most CI.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    # Force initialization so set_seed_local runs the per-device loop.
    torch.zeros(1, device="cuda")

    set_seed_local(seed=123)
    a = torch.rand(5, device="cuda")
    set_seed_local(seed=123)
    b = torch.rand(5, device="cuda")
    torch.testing.assert_close(a, b)


def _mock_broadcast_fill(tensor: torch.Tensor, **_kwargs: object) -> None:
    tensor.fill_(100)


def _capture_broadcast(
    sink: list[torch.Tensor],
) -> Callable[..., None]:
    def capture(tensor: torch.Tensor, *_args: object, **_kwargs: object) -> None:
        sink.append(tensor.detach().clone())

    return capture


def _patch_dist(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rank: int,
    initialized: bool = True,
    backend: str = "gloo",
    broadcast: Callable[..., None] | None = None,
) -> list[torch.Tensor]:
    """Patch ``torch.distributed`` symbols and return the broadcast sink."""
    sink: list[torch.Tensor] = []

    def _is_initialized() -> bool:
        return initialized

    def _get_rank(*_a: object, **_k: object) -> int:
        return rank

    def _get_backend(*_a: object, **_k: object) -> str:
        return backend

    monkeypatch.setattr(torch.distributed, "is_initialized", _is_initialized)
    monkeypatch.setattr(torch.distributed, "get_rank", _get_rank)
    monkeypatch.setattr(torch.distributed, "get_backend", _get_backend)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        broadcast if broadcast is not None else _capture_broadcast(sink),
    )
    return sink


def _mesh_local_rank(local_rank: int) -> MagicMock:
    mesh = MagicMock()
    mesh.get_local_rank.return_value = local_rank
    return mesh


@pytest.mark.parametrize(
    ("rank", "broadcast", "mesh", "salt_by_rank", "expected_local_salt"),
    [
        pytest.param(
            0,
            None,
            None,
            False,
            None,
            id="rank0_no_salt",
        ),
        pytest.param(
            0,
            None,
            _mesh_local_rank(2),
            True,
            ("rank", 2, 100),
            id="rank0_mesh_salt",
        ),
        pytest.param(
            1,
            _mock_broadcast_fill,
            None,
            True,
            ("rank", 1, 100),
            id="rank1_global_salt",
        ),
        pytest.param(
            1,
            _mock_broadcast_fill,
            None,
            False,
            None,
            id="rank1_no_salt",
        ),
    ],
)
def test_set_seed_distributed_salting_combinations(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    broadcast: Callable[..., None] | None,
    mesh: MagicMock | None,
    salt_by_rank: bool,
    *,
    expected_local_salt: tuple[str, int, int] | None,
) -> None:
    """Salting matrix: rank × broadcast-fill × mesh × salt_by_rank.

    Asserts that every backend (torch CPU, Python random, numpy_rng)
    receives a per-component-salted seed derived from the broadcast
    base. The previous five tests pinned ``torch.initial_seed`` only;
    a regression that dropped Python or NumPy seeding would have stayed
    green. Compare exact expected states so the test is independent of
    whatever seed a previous test left behind.
    """
    _patch_dist(monkeypatch, rank=rank, broadcast=broadcast)
    base_seed, local_seed = set_seed_distributed(
        seed=100,
        mesh=mesh,
        salt_by_rank=salt_by_rank,
    )
    assert base_seed == 100

    if expected_local_salt is not None:
        expected_local_seed = salt(*expected_local_salt)
    else:
        # salt_by_rank=False means the local seed equals the base.
        expected_local_seed = 100
    assert local_seed == expected_local_seed
    assert torch.initial_seed() == salt("torch", expected_local_seed)
    assert (
        random.getstate()
        == random.Random(  # noqa: S311
            salt("python", expected_local_seed),
        ).getstate()
    )
    assert (
        numpy_rng.bit_generator.state
        == np.random.PCG64(
            salt("numpy", expected_local_seed),
        ).state
    )


def test_set_seed_distributed_returns_base_then_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return order: ``(base_seed, local_seed)``.

    The base seed (what came off the wire) is the primary value
    callers store; the locally-salted seed is derived metadata.
    Earlier convention had the opposite order; flipped because every
    call site discarded the salted value first.
    """
    _patch_dist(monkeypatch, rank=0)
    base, local = set_seed_distributed(
        seed=100,
        mesh=None,
        salt_by_rank=False,
    )
    assert base == 100
    assert local == 100  # salt_by_rank=False means equal here


def test_set_seed_distributed_rank0_generates_when_seed_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``seed=None``, rank 0 draws from OS entropy and broadcasts."""
    sink = _patch_dist(monkeypatch, rank=0)
    base_seed, local_seed = set_seed_distributed(
        seed=None,
        mesh=None,
        salt_by_rank=True,
    )
    assert base_seed > 0
    assert local_seed > 0
    assert len(sink) == 1


def test_set_seed_distributed_broadcast_carries_user_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The broadcast tensor must actually contain the seed rank 0 chose."""
    sink = _patch_dist(monkeypatch, rank=0)
    base_seed, _local_seed = set_seed_distributed(
        seed=999,
        mesh=None,
        salt_by_rank=True,
    )
    assert base_seed == 999
    assert int(sink[0].item()) == 999


def test_set_seed_distributed_broadcast_tensor_shape_matches_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rank 0 and rank>0 must produce a broadcast tensor of identical
    shape, dtype, and device. NCCL rejects shape/device mismatch; this
    is the bug pattern reviewers caught -- mocks accepted any tensor,
    so a 0-d vs 1-d mismatch shipped.
    """
    sink0 = _patch_dist(monkeypatch, rank=0)
    set_seed_distributed(seed=42, mesh=None, salt_by_rank=False)
    rank0_tensor = sink0[0]
    sink1 = _patch_dist(monkeypatch, rank=1)
    set_seed_distributed(seed=42, mesh=None, salt_by_rank=False)
    rank1_tensor = sink1[0]
    assert rank0_tensor.shape == rank1_tensor.shape, (
        f"rank 0 shape {rank0_tensor.shape} != rank 1 shape {rank1_tensor.shape}",
    )
    assert rank0_tensor.dtype == rank1_tensor.dtype
    assert rank0_tensor.device == rank1_tensor.device


def test_set_seed_distributed_raises_on_nccl_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NCCL backend requires CUDA. If a caller declares ``backend=nccl``
    on a host with no CUDA devices, fail explicitly instead of letting
    ``torch.cuda.current_device()`` raise the opaque "No CUDA GPUs are
    available" error.
    """
    _patch_dist(monkeypatch, rank=0, backend="nccl")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises((AssertionError, RuntimeError), match=r"(?i)nccl.*cuda"):
        set_seed_distributed(seed=42, mesh=None, salt_by_rank=False)


@pytest.mark.cuda
def test_set_seed_distributed_uses_cuda_tensor_on_nccl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NCCL requires CUDA tensors. ``dist.broadcast(<cpu>)`` on a NCCL
    process group raises 'Tensors must be CUDA and dense'. The
    construction must consult ``dist.get_backend()`` and place the
    tensor on the current CUDA device when backend is nccl.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    sink = _patch_dist(monkeypatch, rank=0, backend="nccl")
    set_seed_distributed(seed=42, mesh=None, salt_by_rank=False)
    assert sink[0].device.type == "cuda"


def test_set_seed_distributed_uses_cpu_tensor_on_gloo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gloo is happy with CPU tensors and avoids a CUDA context."""
    sink = _patch_dist(monkeypatch, rank=0, backend="gloo")
    set_seed_distributed(seed=42, mesh=None, salt_by_rank=False)
    assert sink[0].device.type == "cpu"


def test_set_seed_distributed_asserts_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling without an initialized process group is a programmer
    error. ``dist.get_rank()`` would raise an opaque RuntimeError mid-
    function; surface the precondition at the boundary instead.
    """
    _patch_dist(monkeypatch, rank=0, initialized=False)
    with pytest.raises(
        (AssertionError, RuntimeError),
        match=r"(?i)initialized|process group",
    ):
        set_seed_distributed(seed=42, mesh=None, salt_by_rank=False)


def test_set_seed_distributed_rank_n_does_not_use_user_seed_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On rank>0 the broadcast result wins; the user-supplied seed
    argument is informational only. Pin that contract so the renamed
    ``base_seed`` (vs the original shadowed ``seed``) stays correct.
    """
    _patch_dist(monkeypatch, rank=1, broadcast=_mock_broadcast_fill)
    base_seed, local_seed = set_seed_distributed(
        seed=999,
        mesh=None,
        salt_by_rank=False,
    )
    # _mock_broadcast_fill writes 100, not 999.
    assert base_seed == 100
    assert local_seed == 100


@pytest.mark.cuda
def test_set_seed_local_seeds_each_visible_cuda_device() -> None:
    """``set_seed_local`` must seed device i with ``salt("cuda", i, seed)``.

    ``torch.cuda.manual_seed`` seeds the *current* device, not the device
    whose index is passed; a naive ``for i in range(device_count)`` loop
    overwrites the active device's seed once per iteration and leaves the
    other devices unseeded by us. The contract is per-device unique
    streams, so check each device's ``initial_seed`` reflects its own
    salted seed.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires >= 2 visible CUDA devices")

    # ``set_seed_local`` skips the per-device seeding loop when CUDA is
    # not yet initialized (deferred to lazy init). Force initialization
    # so this test exercises the explicit path.
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.zeros(1, device="cuda")

    set_seed_local(seed=42)
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            assert torch.cuda.initial_seed() == salt("cuda", i, 42), (
                f"device {i}: expected salt('cuda', {i}, 42), "
                f"got {torch.cuda.initial_seed()}"
            )


def _baseline_state_without_cuda() -> RngState:
    return {
        "python": random.getstate(),
        "numpy": numpy_rng.bit_generator.state,
        "torch": torch.get_rng_state(),
    }


def _record_set_rng_state_all(
    sink: list[list[torch.Tensor]],
) -> Callable[[Iterable[torch.Tensor]], None]:
    def capture(states: Iterable[torch.Tensor]) -> None:
        sink.append(list(states))

    return capture


def test_set_rng_state_raises_when_cuda_states_exceed_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict policy: ``len(cuda_states) != n_devices`` is an error.

    Multi-GPU checkpoints restored on fewer GPUs is the common silent-
    truncation footgun. The caller must explicitly slice the state list
    if they want to opt out; we will not paper over the mismatch.
    """
    set_all_calls: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        _record_set_rng_state_all(set_all_calls),
    )
    state = _baseline_state_without_cuda()
    state["cuda"] = [
        torch.zeros(16, dtype=torch.uint8),
        torch.ones(16, dtype=torch.uint8),
    ]

    with pytest.raises(ValueError, match="CUDA RNG state mismatch"):
        set_rng_state(state)
    assert set_all_calls == []


def test_set_rng_state_raises_when_cuda_states_below_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric to the over-count case: silent partial restore is the
    deeper bug, not just truncation. A short list leaves the tail
    devices on whatever startup state they had, which is the same kind
    of silent footgun.
    """
    set_all_calls: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        _record_set_rng_state_all(set_all_calls),
    )
    state = _baseline_state_without_cuda()
    state["cuda"] = [torch.zeros(16, dtype=torch.uint8)]

    with pytest.raises(ValueError, match="CUDA RNG state mismatch"):
        set_rng_state(state)
    assert set_all_calls == []


def test_set_rng_state_raises_when_cuda_states_present_but_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CUDA checkpoint restored on a CPU-only host is also a mismatch.

    Silent drop hides env mis-config (container missing the GPU). Force
    the caller to acknowledge the topology change.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    state = _baseline_state_without_cuda()
    state["cuda"] = [torch.zeros(16, dtype=torch.uint8)]

    with pytest.raises(ValueError, match="CUDA RNG state mismatch"):
        set_rng_state(state)


def test_set_rng_state_restores_when_count_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: equal-count restore calls ``set_rng_state_all`` once."""
    set_all_calls: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        _record_set_rng_state_all(set_all_calls),
    )
    state = _baseline_state_without_cuda()
    state["cuda"] = [
        torch.zeros(16, dtype=torch.uint8),
        torch.ones(16, dtype=torch.uint8),
    ]

    set_rng_state(state)

    assert len(set_all_calls) == 1
    assert len(set_all_calls[0]) == 2


def _fake_device_identity(index: int) -> str:
    """Stand in for the CUDA device probe on a host without CUDA initialized."""
    return f"cuda:{index}"


def test_cuda_entries_are_droppable_by_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CPU run on a GPU host narrows the capture itself, without an API knob.

    Keeping CUDA state a CPU run never advances makes resume equality
    host-dependent, so the caller pops it; ``set_rng_state`` then takes the
    no-CUDA path.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [torch.zeros(8, dtype=torch.uint8)],
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(seed, "_cuda_device_identity", _fake_device_identity)

    state = get_rng_state()
    assert "cuda" in state

    state.pop("cuda", None)
    state.pop("cuda_uuids", None)

    assert {"python", "numpy", "torch"} <= set(state)
    set_rng_state(state)


def test_set_rng_state_without_numpy_key_restores_the_rest() -> None:
    """Checkpoints predating numpy tracking must still resume, not KeyError."""
    legacy: RngState = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    pre_numpy = numpy_rng.bit_generator.state

    set_rng_state(legacy)

    # numpy is untouched rather than restored: the state simply is not there.
    assert numpy_rng.bit_generator.state == pre_numpy


def test_set_rng_state_no_cuda_key_does_not_touch_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A state without a ``cuda`` key must not call into ``torch.cuda``.

    This is the legitimate "saved on CPU, restored anywhere" case.
    """
    set_all_calls: list[list[torch.Tensor]] = []
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        _record_set_rng_state_all(set_all_calls),
    )

    set_rng_state(_baseline_state_without_cuda())

    assert set_all_calls == []


def test_enable_determinism_warns_on_cublas_env_overwrite(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the caller already set ``CUBLAS_WORKSPACE_CONFIG`` to a
    different value, ``enable_determinism`` must not silently clobber
    it. Log a warning so the divergence is visible.
    """
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()

    try:
        with caplog.at_level("WARNING", logger="priml.math.seed"):
            enable_determinism(cudnn=False, sdpa=False)
    finally:
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert any("CUBLAS_WORKSPACE_CONFIG" in r.message for r in caplog.records), (
        f"expected CUBLAS env warning; got: {[r.message for r in caplog.records]}"
    )


def test_enable_determinism_warns_when_called_after_cuda_init(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``CUBLAS_WORKSPACE_CONFIG`` must be set before CUDA context
    creation or it has no effect. Surface the precondition violation
    as a warning so the operator sees that determinism may not have
    taken effect.
    """
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()

    try:
        with caplog.at_level("WARNING", logger="priml.math.seed"):
            enable_determinism(cudnn=False, sdpa=False)
    finally:
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )
    assert any("after CUDA initialization" in r.message for r in caplog.records), (
        f"expected post-init warning; got: {[r.message for r in caplog.records]}"
    )


def test_enable_determinism_disables_sdpa_by_default() -> None:
    """The default must be deterministic. ``sdpa=False`` (i.e. don't
    disable the nondeterministic SDPA backends) contradicted every
    explicit caller in the codebase, so flip the default to ``True``.
    """
    sig = inspect.signature(enable_determinism)
    assert sig.parameters["sdpa"].default is True


def test_enable_determinism_sets_cublas_env_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the env var is unset, ``enable_determinism`` sets it to
    the deterministic value. No warning needed.
    """
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    algorithms_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        enable_determinism(cudnn=False, sdpa=False)
    finally:
        torch.use_deterministic_algorithms(
            algorithms_enabled,
            warn_only=warn_only_enabled,
        )
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_dataloader_worker_init_fn_decorrelates_numpy_streams() -> None:
    """Module-level ``numpy_rng`` (and ``random`` / legacy ``np.random``)
    inherits identical state across forked DataLoader workers. Provide
    a ``dataloader_worker_init_fn`` helper that reseeds per worker.
    """
    set_seed_local(seed=100)
    pristine_numpy = numpy_rng.bit_generator.state

    dataloader_worker_init_fn(0)
    state_worker0 = numpy_rng.bit_generator.state

    # Reset and run a different worker id.
    numpy_rng.bit_generator.state = pristine_numpy
    dataloader_worker_init_fn(1)
    state_worker1 = numpy_rng.bit_generator.state

    assert state_worker0 != state_worker1, (
        "dataloader_worker_init_fn(0) and (1) produced identical numpy "
        "state -- workers will draw correlated random sequences"
    )


def test_dataloader_worker_init_fn_reseeds_legacy_numpy_global() -> None:
    """Same fork hazard applies to ``np.random``."""
    set_seed_local(seed=100)
    dataloader_worker_init_fn(0)
    a = np.random.rand(3)  # noqa: NPY002 -- exercising the legacy reseed contract
    dataloader_worker_init_fn(0)
    b = np.random.rand(3)  # noqa: NPY002 -- exercising the legacy reseed contract
    assert (a == b).all(), "worker init must be deterministic per worker_id"
    dataloader_worker_init_fn(1)
    c = np.random.rand(3)  # noqa: NPY002 -- exercising the legacy reseed contract
    assert not (a == c).all(), "worker 0 and worker 1 must differ"


def test_get_rng_state_warns_when_mps_backend_active(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When MPS (Apple Silicon) is the active backend, MPS RNG state is
    NOT captured by ``get_rng_state``. Surface that as a warning so the
    operator can't silently rely on a checkpoint that doesn't actually
    round-trip MPS reproducibility.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    with caplog.at_level("WARNING", logger="priml.math.seed"):
        get_rng_state()
    assert any("mps" in r.message.lower() for r in caplog.records), (
        f"expected MPS warning; got: {[r.message for r in caplog.records]}"
    )


def test_get_rng_state_warns_when_xpu_backend_active(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same shape for XPU (Intel GPU)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    if not hasattr(torch, "xpu"):
        pytest.skip("xpu module not available in this torch build")
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)

    with caplog.at_level("WARNING", logger="priml.math.seed"):
        get_rng_state()
    assert any("xpu" in r.message.lower() for r in caplog.records), (
        f"expected XPU warning; got: {[r.message for r in caplog.records]}"
    )


def test_get_rng_state_no_warning_when_only_cpu(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CPU-only host: no accelerator warning."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    if hasattr(torch, "xpu"):
        monkeypatch.setattr(torch.xpu, "is_available", lambda: False)

    with caplog.at_level("WARNING", logger="priml.math.seed"):
        get_rng_state()
    assert not any(
        ("mps" in r.message.lower() or "xpu" in r.message.lower())
        for r in caplog.records
    )


def test_set_rng_state_raises_on_missing_required_key() -> None:
    """``RngState`` declares python/numpy/torch as required. A malformed
    state missing one of those keys must fail loudly, not silently
    skip restoration. KeyError is the correct failure shape -- the
    contract is "all keys must be present"; we previously guarded each
    one with ``if "x" in state:`` which weakened the TypedDict contract.
    """
    # Deliberately malformed: missing the required "torch" key. Cast
    # to bypass the TypedDict check; the runtime must catch what the
    # type system declares as required.
    malformed = cast(
        RngState,
        {
            "python": random.getstate(),
            "numpy": numpy_rng.bit_generator.state,
        },
    )
    with pytest.raises(KeyError, match="torch"):
        set_rng_state(malformed)


def test_set_rng_state_warns_on_cuda_device_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Count-match alone doesn't detect a CUDA_VISIBLE_DEVICES remap.
    When the checkpoint carries device UUIDs and they don't match the
    current devices, log a warning so the operator can confirm intent.
    Backward-compatible: checkpoints without ``cuda_uuids`` are accepted
    silently.
    """
    sink: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        _record_set_rng_state_all(sink),
    )

    class FakeProps:
        uuid = "GPU-CURRENT"

    def _fake_props(_i: int) -> FakeProps:
        return FakeProps()

    monkeypatch.setattr(torch.cuda, "get_device_properties", _fake_props)

    state: RngState = {
        "python": random.getstate(),
        "numpy": numpy_rng.bit_generator.state,
        "torch": torch.get_rng_state(),
        "cuda": [torch.zeros(16, dtype=torch.uint8)],
        "cuda_uuids": ["GPU-DIFFERENT"],
    }
    with caplog.at_level("WARNING", logger="priml.math.seed"):
        set_rng_state(state)

    assert any("device identit" in r.message.lower() for r in caplog.records), (
        f"expected identity-mismatch warning; "
        f"got: {[r.message for r in caplog.records]}"
    )
    assert len(sink) == 1


def test_set_rng_state_no_warning_when_cuda_uuids_absent(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Legacy checkpoints have no ``cuda_uuids`` key. Accept them
    silently; the warning only fires when we have something to compare.
    """
    sink: list[list[torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        _record_set_rng_state_all(sink),
    )

    state: RngState = {
        "python": random.getstate(),
        "numpy": numpy_rng.bit_generator.state,
        "torch": torch.get_rng_state(),
        "cuda": [torch.zeros(16, dtype=torch.uint8)],
    }
    with caplog.at_level("WARNING", logger="priml.math.seed"):
        set_rng_state(state)
    assert not any("identit" in r.message.lower() for r in caplog.records)


def test_make_seed_uses_os_entropy() -> None:
    """``make_seed`` should draw from OS entropy (``secrets.randbits``),
    not a timestamp/PID hash with hand-rolled "mixing." The folklore
    docstring is gone; the implementation now uses a 63-bit value the
    way torch and numpy both accept.
    """
    import inspect  # noqa: PLC0415

    src = inspect.getsource(make_seed)
    assert "secrets" in src, "make_seed must draw from OS entropy"
    assert "time" not in src, "make_seed must not depend on wall-clock"
    assert "pid" not in src.lower(), "make_seed must not depend on PID"


def test_make_seed_64_bit_range() -> None:
    """The 32-bit narrowing was unmotivated; torch/numpy accept 64-bit
    seeds. Verify the output fits the wider range.
    """
    seeds = {make_seed() for _ in range(1_000)}
    assert max(seeds) > 2**32, f"make_seed narrowed to 32-bit: max={max(seeds)}"


def test_set_seed_local_does_not_force_cuda_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``set_seed_local`` should not initialize CUDA as a side effect.
    A CPU-only run that imports any seeding caller should not allocate
    a CUDA context. Seed the per-device RNGs lazily instead -- torch's
    ``manual_seed`` already supports this via ``_lazy_call``.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    init_calls: list[None] = []

    def _track_init() -> None:
        init_calls.append(None)

    monkeypatch.setattr(torch.cuda, "init", _track_init)
    set_seed_local(seed=42)
    assert init_calls == [], (
        "set_seed_local must not call torch.cuda.init when CUDA is not yet initialized"
    )


def test_set_seed_local_seeds_legacy_numpy_global() -> None:
    """Many ML codepaths still call ``np.random.rand`` / ``randn`` etc.
    (the legacy module-level API). ``set_seed_local``'s docstring says
    it "sets the random seed for ... NumPy" -- enforce that this
    includes the legacy global, not just the module-level
    ``numpy_rng`` Generator.
    """
    import numpy as np  # noqa: PLC0415

    set_seed_local(seed=7)
    a = np.random.rand(3)  # noqa: NPY002 -- exercising the legacy reseed contract
    set_seed_local(seed=7)
    b = np.random.rand(3)  # noqa: NPY002 -- exercising the legacy reseed contract
    assert (a == b).all(), f"legacy np.random not reseeded: {a} vs {b}"


def test_salt_rejects_objects_with_default_repr() -> None:
    """``salt(*args)`` uses ``str(args)`` for hashing. Objects whose
    ``__repr__`` includes an address (default object repr) produce
    non-deterministic salts across runs. The function advertises
    determinism; enforce by rejecting anything that isn't a stable
    primitive.
    """

    class Opaque:
        pass

    with pytest.raises(TypeError, match=r"(?i)stable repr|primitive"):
        salt(Opaque(), 42)


def test_salt_accepts_stable_primitives() -> None:
    """``str``, ``int``, ``bytes``, ``bool``, ``None``, and ``float``
    have stable reprs across processes. They must remain accepted.
    """
    # No exception means the contract is preserved.
    salt("torch", 42)
    salt(b"bytes", 0)
    salt(True, None)
    salt(3.14, 0)


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
