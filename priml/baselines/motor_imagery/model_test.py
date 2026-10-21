"""Construction, costing, and the init-and-forward golden of the EEG networks."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Self, override

from configgle import Fig, InlineConfig

import pytest
import torch

from priml.baselines.motor_imagery.model import (
    AttentionEEGNet,
    MultiScaleEEGNet,
    ShallowConvNet,
)
from priml.baselines.motor_imagery.testing import (
    compare_record,
    model_record,
)
from priml.cost import Cost, cost
from priml.model.conv import Conv2d
from priml.model.linear import Linear
from priml.model.norm import BatchNorm2d
from priml.model.pool import AvgPool2d
from priml.model.special import Dropout, Identity
from priml.model.squeeze_excitation import SqueezeExcitation
from priml.model.swiglu import elu, relu
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import (
    PinnedRandom,
    assert_tensor_golden_part,
    tensor_bits_equal,
)


_CWD: Final = Path(__file__).resolve().parent


def _as[T](config: object, kind: type[T]) -> T:
    """Narrow a slot to the config the test edits."""
    assert isinstance(config, kind)
    return config


def _small(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> MultiScaleEEGNet.Config | ShallowConvNet.Config | AttentionEEGNet.Config:
    cfg = model_type.Config(3, 2)
    cfg.num_samples = 100
    cfg.channels_hidden = 2
    if isinstance(cfg, (MultiScaleEEGNet.Config, AttentionEEGNet.Config)):
        cfg.channels_projection = 2
    return cfg


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
def test_state_saved_under_the_source_names_loads_strictly(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> None:
    cfg = _small(model_type)
    saved = cfg.make()
    source = {
        key.replace("norm", "bn").replace("proj_out", "classifier"): value
        for key, value in saved.state_dict().items()
    }
    assert source.keys() != saved.state_dict().keys()
    loaded = cfg.make()
    loaded.load_state_dict(source)
    compare_record(
        model_record(saved, prefix="model"),
        model_record(loaded, prefix="model"),
    )


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
def test_networks_compose_native_priml_parameter_layers(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> None:
    model = _small(model_type).make()
    leaves = [
        part
        for part in model.modules()
        if isinstance(part, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d))
    ]
    assert leaves
    assert all(isinstance(part, (Conv2d, Linear, BatchNorm2d)) for part in leaves)
    assert type(model.proj_out) is Linear
    for part in model.modules():
        if isinstance(part, torch.nn.Dropout):
            assert isinstance(part, Dropout)
        if isinstance(part, torch.nn.AvgPool2d):
            assert isinstance(part, AvgPool2d)
    if isinstance(model, AttentionEEGNet):
        assert isinstance(model.se1, SqueezeExcitation)
        assert isinstance(model.se2, SqueezeExcitation)


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
@pytest.mark.parametrize("initializer", ["zeros", "uniform"])
def test_projection_respects_an_explicit_child_initializer(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
    initializer: str,
) -> None:
    cfg = _small(model_type)
    init = torch.nn.init.zeros_ if initializer == "zeros" else torch.nn.init.uniform_
    _as(cfg.proj_out, Linear.Config).init_bias = init
    resolved = cfg.finalized()
    assert _as(resolved.proj_out, Linear.Config).init_bias is init
    bias = _as(resolved.make().proj_out, Linear).bias
    assert bias is not None
    if initializer == "zeros":
        assert torch.equal(bias, torch.zeros(2))
    else:
        assert ((bias >= 0) & (bias <= 1)).all()


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
@pytest.mark.parametrize("change", ["conv_bias", "normalization", "projection"])
def test_configured_child_parameter_ownership_is_costed(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
    change: str,
) -> None:
    cfg = _small(model_type)
    if change == "conv_bias":
        if isinstance(cfg, MultiScaleEEGNet.Config):
            _as(cfg.conv_t1, Conv2d.Config).bias = True
        else:
            assert isinstance(cfg, (ShallowConvNet.Config, AttentionEEGNet.Config))
            _as(cfg.conv_t, Conv2d.Config).bias = True
    elif change == "normalization":
        if isinstance(cfg, ShallowConvNet.Config):
            _as(cfg.norm, BatchNorm2d.Config).elementwise_affine = False
        else:
            assert isinstance(cfg, (MultiScaleEEGNet.Config, AttentionEEGNet.Config))
            _as(cfg.norm_s, BatchNorm2d.Config).elementwise_affine = False
    else:
        _as(cfg.proj_out, Linear.Config).bias = False
    resolved = cfg.finalized()
    model = resolved.make()
    expected = sum(parameter.numel() for parameter in model.parameters())
    assert cost(resolved, seq_len=100, batch_size=4, dtype=None).params == expected


def test_default_gate_cost_respects_both_child_linear_biases() -> None:
    cfg = _as(AttentionEEGNet.Config().gate, SqueezeExcitation.Config)
    cfg.channels_in = 5
    cfg.channels_hidden = 1
    _as(cfg.fc1, Linear.Config).bias = True
    _as(cfg.fc2, Linear.Config).bias = True
    resolved = cfg.finalized()
    counted = cost(resolved, seq_len=4, batch_size=3, dtype=None)
    assert counted.params == sum(
        parameter.numel() for parameter in resolved.make().parameters()
    )


class IdentityPool(torch.nn.Identity):
    """Declare identity geometry and cost for an injected spatial transform."""

    def reset_parameters(self) -> None:
        """Retain the stateless identity transform."""

    class Config(Fig["IdentityPool"]):
        """Construct a transform preserving both spatial axes."""

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Return the input grid unchanged."""
            return input_grid

        def cost(self, **kwargs: object) -> Cost:
            """Perform no work and own no parameters."""
            del kwargs
            return Cost()

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
@pytest.mark.parametrize("kind", ["identity", "average"])
def test_injected_pool_geometry_sizes_the_projection_and_cost(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
    kind: str,
) -> None:
    cfg = _small(model_type)
    if isinstance(cfg, ShallowConvNet.Config):
        cfg.pool = (
            IdentityPool.Config()
            if kind == "identity"
            else AvgPool2d.Config(kernel_size=(1, 20), stride=(1, 10))
        )
    else:
        assert isinstance(cfg, (MultiScaleEEGNet.Config, AttentionEEGNet.Config))
        cfg.pool1 = (
            IdentityPool.Config()
            if kind == "identity"
            else AvgPool2d.Config(kernel_size=(1, 2))
        )
        cfg.pool2 = (
            IdentityPool.Config()
            if kind == "identity"
            else AvgPool2d.Config(kernel_size=(1, 3))
        )
    resolved = cfg.finalized()
    model = resolved.make()
    # EEG convolutions require one input plane.
    media = torch.randn(4, 1, 3, 100)
    assert model(media).shape == (4, 2)
    assert_cost_matches_torch(
        resolved,
        build_input=media.requires_grad_,
        seq_len=100,
        batch_size=4,
        dtype=None,
    )


@pytest.mark.parametrize("kind", ["identity", "zero_probability"])
def test_disabling_dropout_removes_its_analytical_work(kind: str) -> None:
    cfg = ShallowConvNet.Config(3, 2)
    cfg.num_samples = 100
    cfg.channels_hidden = 2
    before = cost(cfg.finalized(), seq_len=100, batch_size=4, dtype=None)
    cfg.drop = Identity.Config() if kind == "identity" else Dropout.Config(p=0)
    after = cost(cfg.finalized(), seq_len=100, batch_size=4, dtype=None)
    assert after["flops", "elementwise"].sum() < before["flops", "elementwise"].sum()


@pytest.mark.parametrize("model_type", [MultiScaleEEGNet, AttentionEEGNet])
def test_activation_slot_sets_the_nonlinearity_and_its_cost(
    model_type: type[MultiScaleEEGNet | AttentionEEGNet],
) -> None:
    cfg = _small(model_type)
    assert isinstance(cfg, (MultiScaleEEGNet.Config, AttentionEEGNet.Config))
    assert cfg.activation is elu
    # MultiScaleEEGNet.__init__ fixes the Conv2d input plane to one.
    media = torch.randn(4, 1, 3, 100)
    with host_agnostic_numerics(), PinnedRandom():
        baseline_model = cfg.make().eval()
    baseline = cost(cfg.finalized(), seq_len=100, batch_size=4, dtype=None)
    cfg.activation = relu
    with host_agnostic_numerics(), PinnedRandom():
        swapped_model = cfg.make().eval()
    swapped = cost(cfg.finalized(), seq_len=100, batch_size=4, dtype=None)
    assert not torch.equal(baseline_model(media), swapped_model(media))
    assert (
        swapped["flops", "elementwise"].sum() < baseline["flops", "elementwise"].sum()
    )


def test_activation_slot_accepts_a_config_building_a_module() -> None:
    cfg = _small(MultiScaleEEGNet)
    # MultiScaleEEGNet.__init__ fixes the Conv2d input plane to one.
    media = torch.randn(4, 1, 3, 100)
    with host_agnostic_numerics(), PinnedRandom():
        baseline = cfg.make().eval()
    assert isinstance(cfg, MultiScaleEEGNet.Config)
    cfg.activation = InlineConfig(torch.nn.ELU)
    with host_agnostic_numerics(), PinnedRandom():
        built = cfg.make().eval()
    assert torch.equal(baseline(media), built(media))


class DerivedGate(torch.nn.Module):
    """Require width propagation before finalization, at each gate position."""

    class Config(Fig["DerivedGate"]):
        channels_in: int = 16
        """Input width supplied by the containing network."""

        derived_width: int = -1
        """Width resolved by child finalization."""

        @override
        def finalize(self) -> Self:
            self.derived_width = self.channels_in
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.width = config.derived_width

    @override
    def forward(self, media: torch.Tensor) -> torch.Tensor:
        assert media.shape[1] == self.width
        return media


@pytest.mark.parametrize("finalized_template", [False, True])
def test_injected_gates_derive_their_own_final_width_before_construction(
    finalized_template: bool,
) -> None:
    cfg = AttentionEEGNet.Config(8, 2)
    cfg.channels_hidden = 2
    cfg.channels_projection = 5
    cfg.num_samples = 100
    cfg.gate = (
        DerivedGate.Config().finalized() if finalized_template else DerivedGate.Config()
    )
    model = cfg.make()
    assert isinstance(model.se1, DerivedGate)
    assert isinstance(model.se2, DerivedGate)
    assert model.se1.width == 4
    assert model.se2.width == 5


@pytest.mark.parametrize("reduction", [1, 2, 4, 100])
def test_gate_bottleneck_is_the_integer_reduction_of_its_position_width(
    reduction: int,
) -> None:
    cfg = AttentionEEGNet.Config(8, 2)
    cfg.channels_hidden = 2
    cfg.channels_projection = 5
    cfg.num_samples = 100
    cfg.reduction = reduction
    first, second = cfg.finalized().gates
    assert _as(first, SqueezeExcitation.Config).channels_hidden == max(
        1,
        4 // reduction,
    )
    assert _as(second, SqueezeExcitation.Config).channels_hidden == max(
        1,
        5 // reduction,
    )


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
def test_constructor_dummy_updates_every_batch_norm_running_buffer(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> None:
    cfg = _small(model_type)
    with (
        host_agnostic_numerics(),
        PinnedRandom(),
    ):
        model = cfg.make()
    norms = [part for part in model.modules() if isinstance(part, torch.nn.BatchNorm2d)]
    assert norms
    for norm in norms:
        assert norm.momentum is not None
        assert norm.running_mean is not None
        assert norm.running_var is not None
        assert norm.num_batches_tracked is not None
        assert norm.num_batches_tracked == 1
        assert tensor_bits_equal(norm.running_mean, torch.zeros_like(norm.running_mean))
        assert tensor_bits_equal(
            norm.running_var,
            torch.full_like(norm.running_var, 1.0 - norm.momentum),
        )


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
def test_model_init_and_forward_match_golden(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> None:
    with (
        host_agnostic_numerics(),
        PinnedRandom(),
    ):
        model = _small(model_type).make()
        record = model_record(model, prefix="init")
        model.eval()
        # Every network's Conv2d requires a singleton input plane.
        media = torch.arange(4 * 3 * 100).reshape(4, 1, 3, 100).float() / 100
        with torch.no_grad():
            record["init/logits"] = model(media)
    assert_tensor_golden_part(
        _CWD / "testdata" / f"{model_type.__name__.lower()}.pt",
        prefix="init/",
        record=record,
    )


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
def test_complete_model_cost_matches_dispatched_products_and_parameters(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> None:
    cfg = model_type.Config(3, 8)
    cfg.num_samples = 100
    cfg.channels_hidden = 2
    if isinstance(cfg, (MultiScaleEEGNet.Config, AttentionEEGNet.Config)):
        cfg.channels_projection = 7
    if isinstance(cfg, AttentionEEGNet.Config):
        cfg.reduction = 2
    counted = assert_cost_matches_torch(
        cfg,
        # All three model __init__ methods fix one input plane; their _features
        # methods derive the pooling widths and depthwise channel ties.
        build_input=lambda: torch.randn(5, 1, 3, 100, requires_grad=True),
        seq_len=100,
        batch_size=5,
        dtype=None,
    )
    assert counted.params_active == counted.params
    for phase in ("primal", "adjoint"):
        for kernel in ("matmul", "elementwise", "reduction"):
            assert counted["flops", phase, kernel].sum() > 0
            assert counted["bytes", phase, kernel].sum() > 0


def test_default_gate_cost_matches_dispatched_products_and_parameters() -> None:
    cfg = _as(AttentionEEGNet.Config().gate, SqueezeExcitation.Config)
    cfg.channels_in = 5
    cfg.channels_hidden = 2
    counted = assert_cost_matches_torch(
        cfg,
        # SqueezeExcitation.forward requires collapsed spatial height one.
        build_input=lambda: torch.randn(3, 5, 1, 4, requires_grad=True),
        seq_len=4,
        batch_size=3,
        dtype=torch.float32,
    )
    assert counted.params == counted.params_active == 20
    assert counted["flops", "primal", "reduction"].sum() == 45
    assert counted["flops", "adjoint", "elementwise"].sum() == 291


def test_overlapping_pool_cost_counts_covered_positions_before_unused_tail() -> None:
    cfg = ShallowConvNet.Config(3, 4)
    cfg.channels_hidden = 2
    cfg.num_samples = 139
    counted = cost(cfg.finalized(), seq_len=139, batch_size=5, dtype=None)
    assert counted["flops", "adjoint", "reduction"].sum() == (
        2 * (5 * 2 * 115 - 2) + 4 * (5 * 115 - 1) + 5 * 2 * (3 * 75 - 105) + 4 * (5 - 1)
    )


def test_attention_cost_requires_the_injected_gate_to_price_itself() -> None:
    cfg = AttentionEEGNet.Config(8, 2)
    cfg.channels_in = 3
    cfg.channels_hidden = 2
    cfg.channels_projection = 3
    cfg.num_samples = 100
    cfg.gate = DerivedGate.Config()
    with pytest.raises(TypeError, match=r"DerivedGate\.Config has no cost"):
        cost(cfg.finalized(), seq_len=100, batch_size=2, dtype=None)


@pytest.mark.parametrize(
    "model_type",
    [MultiScaleEEGNet, ShallowConvNet, AttentionEEGNet],
)
def test_projection_cost_rejects_a_different_temporal_geometry(
    model_type: type[MultiScaleEEGNet | ShallowConvNet | AttentionEEGNet],
) -> None:
    cfg = model_type.Config(3, 4)
    cfg.num_samples = 500
    with pytest.raises(ValueError, match="sequence length must match num_samples"):
        cost(cfg.finalized(), seq_len=100, batch_size=2, dtype=None)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
