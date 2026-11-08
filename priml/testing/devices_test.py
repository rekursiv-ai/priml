from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.testing.devices import OnCuda


def test_an_op_on_a_cuda_value_stays_on_cuda() -> None:
    value = torch.arange(3.0).as_subclass(OnCuda)

    derived = value[1:] * 2

    assert value.device == torch.device("cuda", 0)
    assert isinstance(derived, OnCuda)
    assert derived.device == torch.device("cuda", 0)


def test_cpu_hands_back_a_plain_host_copy() -> None:
    value = torch.arange(6.0).view(2, 3).as_subclass(OnCuda)

    host = value.cpu()

    assert type(host) is Tensor
    assert host.device == torch.device("cpu")
    assert torch.equal(host, torch.arange(6.0).view(2, 3))
    assert host.data_ptr() != value.data_ptr()


def test_numpy_refuses_unless_forced_as_it_does_for_cuda_memory() -> None:
    value = torch.arange(3.0).as_subclass(OnCuda)

    with pytest.raises(TypeError, match=r"Use Tensor\.cpu"):
        value.numpy()
    assert value.cpu().numpy().tolist() == [0.0, 1.0, 2.0]
    assert value.numpy(force=True).tolist() == [0.0, 1.0, 2.0]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
