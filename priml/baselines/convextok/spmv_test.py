"""A CSR product must equal the dense product and repeat bit for bit on every device."""

import pytest
import torch

from priml.baselines.convextok.spmv import CsrMatrix


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
