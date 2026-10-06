"""Tests for Muon optimizer and lr_scale."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import tempfile

from torch import Tensor, nn
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

import pytest
import torch

from priml import runtime
from priml.lib.custom_json import ReadError
from priml.optimizers import lr_scale
from priml.optimizers.muon import (
    Muon,
    _shape,
    adjust_lr_conv_heuristic,
    adjust_lr_match_rms_adamw,
    adjust_lr_original,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


class TestLrScale:
    def test_warmup_start(self):
        assert lr_scale(0, total_steps=100, warmup_steps=10) == 0.0

    def test_warmup_midpoint(self):
        assert lr_scale(5, total_steps=100, warmup_steps=10) == 0.5

    def test_warmup_end(self):
        assert lr_scale(10, total_steps=100, warmup_steps=10) == 1.0

    def test_cosine_end(self):
        result = lr_scale(100, total_steps=100, warmup_steps=0)
        assert abs(result) < 1e-7

    def test_cosine_midpoint(self):
        result = lr_scale(50, total_steps=100, warmup_steps=0)
        assert abs(result - 0.5) < 1e-7

    def test_min_ratio(self):
        result = lr_scale(100, total_steps=100, warmup_steps=0, min_ratio=0.1)
        assert abs(result - 0.1) < 1e-7

    def test_no_warmup(self):
        assert lr_scale(0, total_steps=100, warmup_steps=0) == 1.0

    def test_zero_total_steps(self):
        # Should not raise.
        lr_scale(0, total_steps=0, warmup_steps=0)

    def test_monotone_decay(self):
        values = [lr_scale(s, total_steps=100, warmup_steps=0) for s in range(101)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_warmup_then_decay(self):
        # Warmup phase should be monotonically increasing.
        warmup = [lr_scale(s, total_steps=100, warmup_steps=10) for s in range(11)]
        for i in range(len(warmup) - 1):
            assert warmup[i] <= warmup[i + 1]
        # Decay phase should be monotonically decreasing.
        decay = [lr_scale(s, total_steps=100, warmup_steps=10) for s in range(10, 101)]
        for i in range(len(decay) - 1):
            assert decay[i] >= decay[i + 1]


class TestMuon:
    def _make_model(self, in_features: int = 8, out_features: int = 4) -> nn.Linear:
        return nn.Linear(in_features, out_features, bias=False)

    def test_eligible_tensor_requires_matrix_parameters(self):
        matrix = nn.Parameter(torch.ones(2, 3))
        vector = nn.Parameter(torch.ones(2))

        assert Muon.eligible_tensor("weight", matrix)
        assert not Muon.eligible_tensor("bias", vector)

    def test_shape_folds_only_the_trailing_matrix_axes(self) -> None:
        parameter = torch.empty(2, 3, 4, 5)
        assert _shape(parameter, 0) == (2, 60)
        assert _shape(parameter, 1) == (3, 20)
        assert _shape(parameter, 2) == (4, 5)

    def test_adjust_lr_variants_match_their_shape_formulas(self) -> None:
        parameter = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
        lr = 0.25
        assert adjust_lr_match_rms_adamw(lr, parameter, ensemble_dims=1) == (
            lr * 0.2 * 5**0.5
        )
        assert adjust_lr_match_rms_adamw(lr, parameter) == lr * 0.2 * 15**0.5

        expected = lr * parameter.norm() / 3**0.5
        actual = adjust_lr_conv_heuristic(lr, parameter, ensemble_dims=1)
        assert actual.shape == ()
        assert torch.equal(actual, expected)
        default_expected = lr * parameter.norm() / 2**0.5
        assert torch.equal(adjust_lr_conv_heuristic(lr, parameter), default_expected)

    def test_adjust_lr_original_uses_both_fan_ratio_branches(self) -> None:
        parameter = nn.Parameter(torch.empty(2, 4))
        assert adjust_lr_original(0.2, parameter) == 0.2
        assert adjust_lr_original(0.2, parameter.mT) == 0.2 * 2**0.5

    def test_constructor_defaults_and_zero_boundaries(self) -> None:
        parameter = nn.Parameter(torch.ones(2, 3))
        defaults = Muon([parameter])
        assert defaults.param_groups[0] == {
            "params": [parameter],
            "lr": 1e-3,
            "weight_decay": 0.1,
            "momentum": 0.95,
            "nesterov": True,
            "ns_coefficients": (3.4445, -4.775, 2.0315),
            "eps": 1e-7,
            "ns_steps": 5,
            "reference_numerics": False,
            "ensemble_dims": 0,
        }
        assert defaults.adjust_lr_fn is adjust_lr_original

        optimizer = Muon([parameter], lr=0.0, weight_decay=0.0)
        assert optimizer.adjust_lr_fn is adjust_lr_original
        assert optimizer.param_groups[0] == {
            "params": [parameter],
            "lr": 0.0,
            "weight_decay": 0.0,
            "momentum": 0.95,
            "nesterov": True,
            "ns_coefficients": (3.4445, -4.775, 2.0315),
            "eps": 1e-7,
            "ns_steps": 5,
            "reference_numerics": False,
            "ensemble_dims": 0,
        }
        Muon([parameter], momentum=0.0, nesterov=False)
        with pytest.raises(
            ValueError,
            match="Nesterov momentum requires momentum",
        ) as error:
            Muon([parameter], momentum=0.0, nesterov=True)
        assert str(error.value) == "Nesterov momentum requires momentum > 0"

    def test_step_group_uses_configured_momentum_decay_and_newton_schulz(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parameter = nn.Parameter(
            torch.tensor([[2.0, -4.0, 6.0], [8.0, 10.0, -12.0]]),
        )

        def fixed_adjustment(lr: float, param: Tensor, ensemble_dims: int) -> float:
            del lr, param, ensemble_dims
            return 0.1

        optimizer = Muon(
            [parameter],
            lr=0.2,
            weight_decay=0.25,
            momentum=0.5,
            nesterov=True,
            ns_coefficients=(2.0, -3.0, 4.0),
            eps=0.02,
            ns_steps=3,
            reference_numerics=True,
            adjust_lr_fn=fixed_adjustment,
        )
        old_buffer = torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
        optimizer.state[parameter]["momentum_buffer"] = old_buffer.clone()
        gradient = torch.tensor([[1.0, 3.0, 5.0], [7.0, 9.0, 11.0]])
        parameter.grad = gradient.clone()
        projected = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])
        calls: list[tuple[Tensor, dict[str, object]]] = []

        def project(matrix: Tensor, **kwargs: object) -> Tensor:
            calls.append((matrix.clone(), kwargs))
            return projected

        monkeypatch.setattr(
            "priml.optimizers.muon.matrix_signum_via_newtonschulz",
            project,
        )
        before = parameter.detach().clone()
        expected_buffer = torch.lerp(old_buffer, gradient, 0.5)
        expected_gradient = torch.lerp(gradient, expected_buffer, 0.5)

        optimizer.step()

        buffer = cast(Tensor, optimizer.state[parameter]["momentum_buffer"])
        assert torch.equal(buffer, expected_buffer)
        assert len(calls) == 1
        # matrix_signum_via_newtonschulz receives a singleton ensemble axis.
        torch.testing.assert_close(calls[0][0], expected_gradient.reshape(1, 2, 3))
        assert calls[0][1] == {
            "coefficients": (2.0, -3.0, 4.0),
            "steps": 3,
            "eps": 0.02,
            "reference_numerics": True,
        }
        expected_parameter = before * (1 - 0.2 * 0.25) - projected * 0.1
        torch.testing.assert_close(parameter, expected_parameter)

    def test_step_group_applies_tensor_adjusted_rate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parameter = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        adjusted_rate = torch.tensor(0.125)

        def tensor_adjustment(lr: float, param: Tensor, ensemble_dims: int) -> Tensor:
            assert lr == 0.0
            assert param is parameter
            assert ensemble_dims == 0
            return adjusted_rate

        optimizer = Muon(
            [parameter],
            lr=0.0,
            weight_decay=None,
            momentum=0.0,
            nesterov=False,
            adjust_lr_fn=tensor_adjustment,
        )
        gradient = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        parameter.grad = gradient.clone()

        def project(matrix: Tensor, **kwargs: object) -> Tensor:
            assert kwargs == {
                "coefficients": (3.4445, -4.775, 2.0315),
                "steps": 5,
                "eps": 1e-7,
                "reference_numerics": False,
            }
            return matrix

        monkeypatch.setattr(
            "priml.optimizers.muon.matrix_signum_via_newtonschulz",
            project,
        )
        optimizer.step()
        assert torch.equal(parameter, torch.tensor([[0.75, 1.5], [2.25, 3.0]]))

    def test_step_group_reuses_momentum_between_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parameter = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        optimizer = Muon(
            [parameter],
            lr=0.0,
            weight_decay=None,
            momentum=0.5,
            nesterov=False,
        )
        calls: list[Tensor] = []

        def project(matrix: Tensor, **kwargs: object) -> Tensor:
            del kwargs
            calls.append(matrix.clone())
            return matrix

        monkeypatch.setattr(
            "priml.optimizers.muon.matrix_signum_via_newtonschulz",
            project,
        )
        gradients = (
            torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
            torch.tensor([[10.0, 12.0], [14.0, 16.0]]),
        )
        for gradient in gradients:
            parameter.grad = gradient.clone()
            optimizer.step()

        buffer = cast(Tensor, optimizer.state[parameter]["momentum_buffer"])
        assert torch.equal(buffer, torch.tensor([[5.5, 7.0], [8.5, 10.0]]))
        assert len(calls) == 2
        assert torch.equal(
            calls[0],
            # matrix_signum_via_newtonschulz receives a singleton ensemble axis.
            (gradients[0] * 0.5).reshape(1, 2, 2),
        )
        second_buffer = torch.lerp(gradients[0] * 0.5, gradients[1], 0.5)
        assert torch.equal(
            calls[1],
            # matrix_signum_via_newtonschulz receives a singleton ensemble axis.
            second_buffer.reshape(1, 2, 2),
        )

    @pytest.mark.parametrize(
        "field",
        [
            "lr",
            "momentum",
            "nesterov",
            "eps",
            "ns_steps",
            "reference_numerics",
            "ensemble_dims",
        ],
    )
    def test_step_group_rejects_none_for_required_group_scalars(
        self,
        field: str,
    ) -> None:
        parameter = nn.Parameter(torch.ones(2, 3))
        optimizer = Muon([parameter])
        parameter.grad = torch.ones_like(parameter)
        optimizer.param_groups[0][field] = None

        with pytest.raises(ReadError):
            optimizer.step()

    def test_step_group_rejects_malformed_non_none_weight_decay(self) -> None:
        parameter = nn.Parameter(torch.ones(2, 3))
        optimizer = Muon([parameter])
        parameter.grad = torch.ones_like(parameter)
        optimizer.param_groups[0]["weight_decay"] = True

        with pytest.raises(ReadError):
            optimizer.step()

    def test_step_group_rejects_nontriplet_coefficients(self) -> None:
        parameter = nn.Parameter(torch.ones(2, 3))
        optimizer = Muon([parameter])
        parameter.grad = torch.ones_like(parameter)
        optimizer.param_groups[0]["ns_coefficients"] = (1.0, 2.0)

        with pytest.raises(ReadError):
            optimizer.step()

    def test_step_group_accepts_list_coefficients_from_a_restored_group(
        self,
    ) -> None:
        """A JSON-restored group holds a list where the config held a tuple."""
        listed = nn.Parameter(torch.tensor([[1.0, -2.0, 3.0], [0.5, 4.0, -1.0]]))
        tupled = nn.Parameter(listed.detach().clone())
        with_list, with_tuple = Muon([listed]), Muon([tupled])
        with_list.param_groups[0]["ns_coefficients"] = [3.4445, -4.775, 2.0315]
        for parameter, optimizer in ((listed, with_list), (tupled, with_tuple)):
            parameter.grad = torch.tensor([[0.1, 0.2, -0.3], [0.4, -0.5, 0.6]])
            optimizer.step()
        assert torch.equal(listed.detach(), tupled.detach())

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (partial(Muon, lr=float("nan")), "Learning rate must be finite"),
            (partial(Muon, lr=float("inf")), "Learning rate must be finite"),
            (partial(Muon, momentum=float("nan")), "Momentum must be finite"),
            (partial(Muon, momentum=float("inf")), "Momentum must be finite"),
            (partial(Muon, weight_decay=float("nan")), "Weight decay must be finite"),
            (partial(Muon, weight_decay=float("inf")), "Weight decay must be finite"),
            (partial(Muon, eps=float("inf")), "Epsilon must be finite"),
            (partial(Muon, eps=float("nan")), "Epsilon must be finite"),
            (partial(Muon, momentum=1.0), "Momentum must be finite and lie in"),
            (partial(Muon, eps=0.0), "Epsilon must be finite and positive"),
            (partial(Muon, ns_steps=0), "Invalid ns_steps"),
            (partial(Muon, ensemble_dims=-1), "Invalid ensemble_dims"),
        ],
    )
    def test_invalid_scalar_hyperparameters_are_rejected(
        self,
        build: Callable[[list[nn.Parameter]], Muon],
        message: str,
    ) -> None:
        """Each bound would otherwise fail deep in a step, or poison it with NaN."""
        with pytest.raises(ValueError, match=message):
            build([nn.Parameter(torch.ones(2, 3))])

    def test_ensemble_axes_are_leading_and_need_a_matrix_behind_them(self) -> None:
        """``ensemble_dims`` peels LEADING axes; what remains must be a matrix."""
        stacked = nn.Parameter(torch.ones(2, 3, 4))
        stacked.grad = torch.ones_like(stacked)
        Muon([stacked], ensemble_dims=1).step()
        too_flat = nn.Parameter(torch.ones(2, 3))
        too_flat.grad = torch.ones_like(too_flat)
        with pytest.raises(ValueError, match=r"ndim >= 3 \(ensemble_dims=1\)"):
            Muon([too_flat], ensemble_dims=1).step()

    def test_step_group_continues_after_a_missing_gradient(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        missing = nn.Parameter(torch.ones(2, 3))
        present = nn.Parameter(torch.full((2, 3), 4.0))
        optimizer = Muon(
            [missing, present],
            lr=0.1,
            weight_decay=None,
            momentum=0.0,
            nesterov=False,
        )
        present.grad = torch.ones_like(present)

        def project(matrix: Tensor, **kwargs: object) -> Tensor:
            del kwargs
            return matrix

        monkeypatch.setattr(
            "priml.optimizers.muon.matrix_signum_via_newtonschulz",
            project,
        )

        optimizer.step()

        assert torch.equal(present, torch.full((2, 3), 3.9))
        assert not optimizer.state[missing]

    def test_basic_step(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_weight_changes(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02)
        w_before = model.weight.data.clone()

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        assert not torch.equal(model.weight.data, w_before)

    def test_multiple_steps(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02)

        for _ in range(5):
            x = torch.randn(2, 8)
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_weight_decay(self):
        model = self._make_model()
        opt_wd = Muon(model.parameters(), lr=0.02, weight_decay=0.1)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt_wd.step()

    def test_no_weight_decay(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02, weight_decay=None)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_nesterov_false(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02, nesterov=False)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_rejects_1d_params(self):
        param = nn.Parameter(torch.randn(8))
        opt = Muon([param], lr=0.02)
        param.grad = torch.randn(8)

        with pytest.raises(ValueError, match="ndim >= 2"):
            opt.step()

    def test_invalid_lr(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="Learning rate"):
            Muon(model.parameters(), lr=-1.0)

    def test_invalid_momentum(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="Momentum"):
            Muon(model.parameters(), lr=0.02, momentum=-0.1)

    def test_invalid_weight_decay(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="Weight decay"):
            Muon(model.parameters(), lr=0.02, weight_decay=-0.1)

    def test_nesterov_requires_momentum(self):
        model = self._make_model()
        with pytest.raises(ValueError, match="Nesterov"):
            Muon(model.parameters(), lr=0.02, momentum=0, nesterov=True)

    def test_adjust_lr_match_rms_adamw(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02, adjust_lr_fn=adjust_lr_match_rms_adamw)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_adjust_lr_conv_heuristic(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02, adjust_lr_fn=adjust_lr_conv_heuristic)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_state_dict_roundtrip(self):
        model = self._make_model()
        opt = Muon(model.parameters(), lr=0.02)

        x = torch.randn(2, 8)
        loss = model(x).sum()
        loss.backward()
        opt.step()

        state = opt.state_dict()
        opt2 = Muon(model.parameters(), lr=0.02)
        opt2.load_state_dict(state)


# Newton-Schulz is a *global* spectral op: ``g.norm`` and ``g @ g.T`` must see the whole
# matrix. Muon passes the DTensor grad straight into the NS kernel; torch's DTensor
# dispatch redistributes ``Shard(0) @ Shard(0).T`` into a collective, so the sharded
# update matches the single-device update (maxdiff ~0). This test pins that property: a
# regression to *naive per-shard* NS (independent NS on each 4-row block) would diverge
# by ~0.24. The worker writes ``ok`` or ``FAIL:<reason>`` (max-abs divergence) per rank.
def _muon_shard_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Worker: one Muon step on a row-sharded param vs the full-tensor update."""
    result_path = Path(result_dir)
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh
        torch.manual_seed(0)
        weight = torch.randn(4, 5)
        grad = torch.randn(4, 5)

        full_param = nn.Parameter(weight.clone())
        full_param.grad = grad.clone()
        Muon([full_param], lr=0.1, momentum=0.0, nesterov=False).step()

        weight_dt: DTensor = distribute_tensor(weight.clone(), mesh, [Shard(0)])
        sharded = nn.Parameter(weight_dt)
        sharded.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
        # Guard against a silent world_size=1 (no real shard): each rank must
        # hold exactly 2 of the 4 rows, else the test proves nothing.
        if tuple(weight_dt.to_local().shape) != (2, 5):
            (result_path / f"rank_{rank}").write_text(
                f"FAIL:not-sharded local={tuple(weight_dt.to_local().shape)}",
            )
            return
        Muon([sharded], lr=0.1, momentum=0.0, nesterov=False).step()

        got = cast(DTensor, sharded.data).full_tensor()
        max_abs = (got - full_param.detach()).abs().max().item()
        if max_abs > 1e-2:
            (result_path / f"rank_{rank}").write_text(f"FAIL:maxdiff={max_abs:.4f}")
        else:
            (result_path / f"rank_{rank}").write_text("ok")
    except Exception as e:  # noqa: BLE001 -- The worker must serialize every subprocess failure for the parent assertion.
        (result_path / f"rank_{rank}").write_text(f"FAIL:{e!r}")
    finally:
        runtime._device_mesh = None


@pytest.mark.cli_python_subprocess
def test_muon_sharded_matches_replicated_multirank(
    warm_pools: WarmPoolGetter,
) -> None:
    """Muon on a 2-way row-sharded param must equal the single-device update."""
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(partial(_muon_shard_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
