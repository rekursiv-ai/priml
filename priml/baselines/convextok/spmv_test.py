"""A CSR product must equal the dense product and repeat bit for bit on every device."""

from types import FunctionType
from unittest.mock import Mock

from torch import Tensor

import pytest
import torch

from priml.baselines.convextok import spmv
from priml.baselines.convextok.spmv import CsrMatrix


def _jit_probe(value: int = 1) -> None:
    del value


def _always_cuda(tensor: torch.Tensor) -> bool:
    del tensor
    return True


def test_constructor_normalizes_offsets_and_records_row_count() -> None:
    crow = torch.tensor([0, 2, 2, 3], dtype=torch.int32)
    columns = torch.tensor([0, 1, 0], dtype=torch.int32)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

    matrix = CsrMatrix(crow, columns, values)

    assert matrix.crow_indices.dtype == torch.int64
    assert matrix.crow_indices.tolist() == [0, 2, 2, 3]
    assert matrix.col_indices is columns
    assert matrix.values is values
    assert matrix.num_rows == 3
    assert matrix._plan is None


def test_cuda_constructor_builds_a_plan_from_normalized_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = spmv._Plan(
        crow=torch.tensor([0, 1], dtype=torch.int64),
        buckets=[],
        chunk_start=torch.zeros(0, dtype=torch.int64),
        chunk_end=torch.zeros(0, dtype=torch.int64),
        partial_columns=torch.zeros(0, dtype=torch.int64),
        partial_plan=None,
    )
    plan_spy = Mock(return_value=plan)
    monkeypatch.setattr(spmv, "_plan", plan_spy)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(_always_cuda))
    crow = torch.tensor([0, 1], dtype=torch.int32)
    columns = torch.tensor([0], dtype=torch.int32)
    values = torch.tensor([2.0])

    matrix = CsrMatrix(crow, columns, values)

    assert matrix._plan is plan
    plan_spy.assert_called_once_with(matrix.crow_indices)
    assert matrix.crow_indices.dtype == torch.int64
    assert matrix.crow_indices.tolist() == [0, 1]


def test_cuda_matmul_allocates_and_dispatches_with_exact_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = spmv._Plan(
        crow=torch.tensor([0, 1], dtype=torch.int64),
        buckets=[],
        chunk_start=torch.zeros(0, dtype=torch.int64),
        chunk_end=torch.zeros(0, dtype=torch.int64),
        partial_columns=torch.zeros(0, dtype=torch.int64),
        partial_plan=None,
    )
    matrix = CsrMatrix(
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([2.0], dtype=torch.float64),
    )
    matrix._plan = plan
    vector = torch.empty(2, dtype=torch.float64, device="meta")
    run = Mock()
    monkeypatch.setattr(spmv, "_run", run)

    product = matrix @ vector

    assert product.shape == (1,)
    assert product.dtype == vector.dtype
    assert product.device == vector.device
    run.assert_called_once_with(
        plan,
        matrix.col_indices,
        matrix.values,
        vector,
        product,
    )


def test_plan_buckets_short_rows_and_chunks_long_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    crow = torch.tensor([0, 0, 1, 5, 260, 516, 773], dtype=torch.int64)
    repeat_spy = Mock(wraps=torch.repeat_interleave)
    monkeypatch.setattr(torch, "repeat_interleave", repeat_spy)

    plan = spmv._plan(crow)

    calls = repeat_spy.call_args_list
    assert len(calls) == 2
    assert len(calls[1].args) == 2
    assert isinstance(calls[1].args[0], Tensor)
    assert isinstance(calls[1].args[1], Tensor)
    assert torch.equal(calls[1].args[0], torch.tensor([0, 1, 2]))
    assert torch.equal(calls[1].args[1], torch.tensor([1, 1, 2]))
    assert torch.equal(plan.crow, crow)
    assert [
        (bucket.width, bucket.rows.tolist(), bucket.targets.tolist())
        for bucket in plan.buckets
    ] == [
        (1, [0, 1], [0, 1]),
        (4, [2], [2]),
    ]
    assert plan.chunk_start.tolist() == [5, 260, 516, 772]
    assert plan.chunk_end.tolist() == [260, 516, 772, 773]
    assert plan.partial_columns.tolist() == [0, 0, 0, 0]
    assert plan.partial_columns.dtype == torch.int64
    assert plan.partial_columns.device == crow.device
    assert plan.partial_plan is not None
    assert plan.partial_plan.crow.tolist() == [0, 1, 2, 4]
    assert plan.partial_plan.crow.dtype == torch.int64
    assert plan.partial_plan.crow.device == crow.device
    assert torch.equal(plan.partial_plan.buckets[0].rows, torch.tensor([0, 1]))
    assert torch.equal(plan.partial_plan.buckets[0].targets, torch.tensor([3, 4]))
    assert plan.partial_plan.buckets[0].width == 1
    assert torch.equal(plan.partial_plan.buckets[1].rows, torch.tensor([2]))
    assert torch.equal(plan.partial_plan.buckets[1].targets, torch.tensor([5]))
    assert plan.partial_plan.buckets[1].width == 2


def test_plan_for_only_short_rows_has_empty_chunk_tensors() -> None:
    crow = torch.tensor([0, 0, 1, 3, 35], dtype=torch.int64)

    plan = spmv._plan(crow)

    assert torch.equal(plan.crow, crow)
    assert [
        (bucket.width, bucket.rows.tolist(), bucket.targets.tolist())
        for bucket in plan.buckets
    ] == [
        (1, [0, 1], [0, 1]),
        (2, [2], [2]),
        (32, [3], [3]),
    ]
    assert plan.chunk_start.shape == (0,)
    assert plan.chunk_end.shape == (0,)
    assert plan.partial_columns.shape == (0,)
    for tensor in (plan.chunk_start, plan.chunk_end, plan.partial_columns):
        assert tensor.dtype == torch.int64
        assert tensor.device == crow.device
    assert plan.partial_plan is None


def test_plan_allocations_use_the_csr_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_devices: list[torch.device | None] = []
    original_arange = torch.arange
    original_zeros = torch.zeros

    def record_arange(end: int, *, device: torch.device) -> torch.Tensor:
        requested_devices.append(device)
        return original_arange(end, device=device)

    def record_zeros(
        size: int | tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        requested_devices.append(device)
        return original_zeros(size, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "arange", record_arange)
    monkeypatch.setattr(torch, "zeros", record_zeros)
    spmv._plan(torch.tensor([0, 0, 1, 5, 262], dtype=torch.int64))

    assert len(requested_devices) >= 4
    assert requested_devices == [torch.device("cpu")] * len(requested_devices)


def test_jit_binds_the_real_triton_language(monkeypatch: pytest.MonkeyPatch) -> None:
    language = object()
    jit_result = object()
    import_module = Mock(return_value=language)
    jit = Mock(return_value=jit_result)
    triton = Mock(jit=jit)
    monkeypatch.setattr(spmv, "import_module", import_module)
    monkeypatch.setattr(spmv, "triton", triton)

    result = spmv._jit(_jit_probe)

    assert result is jit_result
    import_module.assert_called_once_with("triton.language")
    jit.assert_called_once()
    bound = jit.call_args.args[0]
    assert isinstance(bound, FunctionType)
    assert bound is not _jit_probe
    assert bound.__name__ == _jit_probe.__name__
    assert bound.__defaults__ == _jit_probe.__defaults__
    assert bound.__globals__["language"] is language
    assert bound.__annotations__ == _jit_probe.__annotations__


class _KernelLaunch:
    def __init__(self, grid: tuple[int, ...]) -> None:
        self.grid = grid
        self.buffers: list[tuple[torch.Tensor, ...]] = []
        self.scalars: list[tuple[int, ...]] = []

    def __call__(
        self,
        buffers: tuple[torch.Tensor, ...],
        *scalars: int,
    ) -> None:
        self.buffers.append(buffers)
        self.scalars.append(scalars)


class _Kernel:
    def __init__(self) -> None:
        self.launches: list[_KernelLaunch] = []

    def __getitem__(self, grid: tuple[int, ...]) -> _KernelLaunch:
        launch = _KernelLaunch(grid)
        self.launches.append(launch)
        return launch


def test_run_dispatches_bucket_and_chunk_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket_kernel = _Kernel()
    chunk_kernel = _Kernel()
    triton = Mock()

    def cdiv(count: int, block: int) -> int:
        return (count + block - 1) // block

    triton.cdiv.side_effect = cdiv
    empty_spy = Mock(wraps=torch.empty)
    ones_spy = Mock(wraps=torch.ones)
    monkeypatch.setattr(torch, "empty", empty_spy)
    monkeypatch.setattr(torch, "ones", ones_spy)
    monkeypatch.setattr(spmv, "_bucket_kernel", lambda: bucket_kernel)
    monkeypatch.setattr(spmv, "_chunk_kernel", lambda: chunk_kernel)
    monkeypatch.setattr(spmv, "triton", triton)

    crow = torch.tensor([0, 1, 2], dtype=torch.int64)
    rows = torch.tensor([0, 1], dtype=torch.int64)
    targets = torch.tensor([0, 1], dtype=torch.int64)
    partial_plan = spmv._Plan(
        crow=torch.tensor([0, 1], dtype=torch.int64),
        buckets=[
            spmv._Bucket(
                width=1,
                rows=torch.tensor([0], dtype=torch.int64),
                targets=torch.tensor([1], dtype=torch.int64),
            ),
        ],
        chunk_start=torch.zeros(0, dtype=torch.int64),
        chunk_end=torch.zeros(0, dtype=torch.int64),
        partial_columns=torch.zeros(0, dtype=torch.int64),
        partial_plan=None,
    )
    chunk_start = torch.tensor([0, 256], dtype=torch.int64)
    chunk_end = torch.tensor([256, 300], dtype=torch.int64)
    partial_columns = torch.zeros(2, dtype=torch.int64)
    plan = spmv._Plan(
        crow=crow,
        buckets=[spmv._Bucket(width=4, rows=rows, targets=targets)],
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        partial_columns=partial_columns,
        partial_plan=partial_plan,
    )
    columns = torch.tensor([0, 0], dtype=torch.int64)
    values = torch.tensor([2.0, 3.0])
    vector = torch.tensor([5.0])
    out = torch.zeros(2)

    spmv._run(plan, columns, values, vector, out)

    assert [launch.grid for launch in bucket_kernel.launches] == [(1,), (1,)]
    first, second = bucket_kernel.launches
    assert first.scalars == [(2, 4, 256)]
    assert type(first.scalars[0][2]) is int
    assert len(first.buffers) == 1
    assert all(
        actual is expected
        for actual, expected in zip(
            first.buffers[0],
            (rows, targets, crow, columns, values, vector, out),
            strict=True,
        )
    )
    assert type(first.scalars[0][1]) is int

    assert [launch.grid for launch in chunk_kernel.launches] == [(2,)]
    (chunk,) = chunk_kernel.launches
    assert chunk.scalars == [(256,)]
    assert len(chunk.buffers) == 1
    chunk_buffers = chunk.buffers[0]
    assert chunk_buffers[0] is chunk_start
    assert chunk_buffers[1] is chunk_end
    assert chunk_buffers[2] is columns
    assert chunk_buffers[3] is values
    assert chunk_buffers[4] is vector
    assert chunk_buffers[5].shape == (2,)

    assert second.scalars == [(1, 1, 1024)]
    assert len(second.buffers) == 1
    partial_buffers = second.buffers[0]
    assert partial_buffers[0] is partial_plan.buckets[0].rows
    assert partial_buffers[1] is partial_plan.buckets[0].targets
    assert partial_buffers[2] is partial_plan.crow
    assert partial_buffers[3] is partial_columns
    assert partial_buffers[4] is chunk_buffers[5]
    assert partial_buffers[5].tolist() == [1.0]
    assert partial_buffers[6] is out
    empty_spy.assert_called_once_with(2, dtype=out.dtype, device=out.device)
    ones_spy.assert_called_once_with(1, dtype=out.dtype, device=out.device)


def test_matches_dense_product_with_an_empty_row() -> None:
    # [[1, 2], [0, 0], [3, 0]] @ [5, 7].
    matrix = CsrMatrix(
        torch.tensor([0, 2, 2, 3], dtype=torch.int32),
        torch.tensor([0, 1, 0], dtype=torch.int32),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    )
    assert (matrix @ torch.tensor([5.0, 7.0], dtype=torch.float64)).tolist() == [
        19.0,
        0.0,
        15.0,
    ]


def test_repeats_bit_for_bit() -> None:
    matrix, vector = _random(torch.device("cpu"))
    assert torch.equal(matrix @ vector, matrix @ vector)


@pytest.mark.gpu_torch_cuda
@pytest.mark.gpu_triton
def test_cuda_product_matches_the_cpu_reference_and_repeats() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA device.")
    matrix, vector = _random(torch.device("cuda"))
    reference = (
        CsrMatrix(
            matrix.crow_indices.cpu(),
            matrix.col_indices.cpu(),
            matrix.values.cpu(),
        )
        @ vector.cpu()
    )
    product = matrix @ vector
    assert torch.equal(product, matrix @ vector)
    torch.testing.assert_close(product.cpu(), reference, rtol=1e-12, atol=1e-12)


def _random(device: torch.device) -> tuple[CsrMatrix, torch.Tensor]:
    # Row lengths span empty rows, every bucket width, and rows long enough to need
    # two levels of chunked partial sums.
    generator = torch.Generator().manual_seed(0)
    lengths = torch.tensor([0, 1, 2, 3, 5, 9, 17, 32, 33, 255, 256, 257, 4000, 70_000])
    columns = 97
    crow = torch.zeros(len(lengths) + 1, dtype=torch.int64)
    crow[1:] = lengths.cumsum(0)
    nnz = int(crow[-1])
    col = torch.randint(0, columns, (nnz,), generator=generator, dtype=torch.int32)
    values = torch.randn(nnz, generator=generator, dtype=torch.float64)
    vector = torch.randn(columns, generator=generator, dtype=torch.float64)
    return CsrMatrix(crow.to(device), col.to(device), values.to(device)), vector.to(
        device,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
