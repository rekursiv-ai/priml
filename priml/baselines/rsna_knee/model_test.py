"""Check every RSNA knee network against in-tree goldens and torch costs."""

from __future__ import annotations

from pathlib import Path
from typing import Final, cast

import hashlib
import math

from configgle import InlineConfig, Makeable, Makes
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.rsna_knee.model import (
    R7,
    FindingAttention,
    ImageNetEncoder,
    Raptor,
    ReleasedVNet,
    TinyConvNeXt,
    _checkpoint_name,
    _readout_cost,
    _softmax_cost,
    _vnet_name,
    checkpoint_state,
    tiny_drop_path,
)
from priml.baselines.rsna_knee.support_test import blocks_of
from priml.cost import cost, elementwise_cost, reduction_cost
from priml.math.custom_types import TensorFn
from priml.model.coatnet import CoAtNet, CoAtTransformer
from priml.model.convnext import (
    ConvNeXt,
    ConvNeXtBlock,
    ConvNeXtHead,
    ConvNeXtStage,
)
from priml.model.linear import Linear
from priml.model.resnet import ResNet
from priml.model.sequential import Sequential
from priml.model.special import Identity
from priml.model.swiglu import SwiGLU, gelu
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import (
    assert_tensor_golden,
    mismatches,
    read_tensors,
    rng_fingerprint,
)
from priml.train.parallelism import NoParallel
from priml.train.train_step import TrainStep


_CWD: Final = Path(__file__).resolve().parent


def _r7_config() -> R7.Config:
    config = R7.Config()
    config.encoder.channels_hidden = (2, 3, 4, 5)
    config.encoder.blocks_per_stage = (1, 1, 1, 1)
    assert isinstance(config.encoder.stage, ConvNeXtStage.Config)
    assert isinstance(config.encoder.stage.block, ConvNeXtBlock.Config)
    config.encoder.stage.block.conv_dw.kernel_size = 3
    config.encoder.stage.block.conv_dw.padding = 1
    config.channels_hidden = 2
    return config


def _batch() -> tuple[Tensor, Tensor, Tensor]:
    # R7 reads three adjacent slices per window and scores twelve findings.
    image = (torch.arange(2 * 3 * 3 * 64 * 96).reshape(2, 3, 3, 64, 96) % 256).to(
        torch.uint8,
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])
    meta = torch.tensor(
        [
            [
                [0, 0, -0.5, 1.5, 0.7, 1, 0, 0, 3],
                [2, 1, 0.5, 1.3, 0.8, 0, 1, 1, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 2, -0.5, 1.4, 0.6, 1, 0, 0, 5],
                [2, 3, 0.0, 1.2, 0.9, 1, 1, 1, 6],
                [4, 4, 0.5, 1.1, 0.5, 1, 2, 2, 7],
            ],
        ],
        dtype=torch.float32,
    )
    return image, meta, mask


def _literal_weights(model: nn.Module) -> dict[str, Tensor]:
    """Return the model state with every parameter set to exact values in [-1, 1)."""
    state = {name: value.clone() for name, value in model.state_dict().items()}
    for index, (name, parameter) in enumerate(model.named_parameters()):
        # A periodic ramp gives small tensors one sign, which kills narrow ReLU stacks.
        code = (
            torch.arange(parameter.numel()) * 2_654_435_761 + 40_503 * index
        ) % 8_191
        state[name] = (code.double() / 8_191 * 2 - 1).float().reshape(parameter.shape)
    return state


def test_r7_forward_and_state_readout_match_their_golden() -> None:
    records: list[dict[str, Tensor]] = []
    config = _r7_config()
    # Eight still pads the five real windows; deployment pads to 32.
    config.pad_multiple = 8
    with host_agnostic_numerics():
        model = config.make().eval()
        model.load_checkpoint(_literal_weights(model))
        image, meta, mask = _batch()
        for chunk in (0, 3):
            logits, states, binaries = model(image, meta, mask, chunk=chunk)
            assert model.area is not None
            assert model.syn_specific is not None
            records.append(
                {
                    "logits": logits,
                    "states": states,
                    "binaries": binaries,
                    "area": model.area,
                    "syn_specific": model.syn_specific,
                    "state_readout": model.state_readout(states, binaries),
                },
            )
    # Chunking only splits the encoder batch, so both passes agree bit for bit.
    assert mismatches(records[0], records[1]) == []
    assert_tensor_golden(_CWD / "testdata" / "r7_model.pt", records[0])


def test_r7_loads_released_head_norm_and_timm_encoder_names() -> None:
    model = _r7_config().make()
    state = _literal_weights(model)
    released = dict(state)
    for native, name in (
        ("norm_out.weight", "out_norm.weight"),
        ("norm_sp_head.bias", "sp_head_norm.bias"),
        ("enc.stem.conv.weight", "enc.stem.0.weight"),
        (
            "enc.stages.0.blocks.0.mlp.up_proj.weight",
            "enc.stages.0.blocks.0.mlp.fc1.weight",
        ),
    ):
        released[name] = released.pop(native)
    model.load_checkpoint(released)
    assert mismatches(state, model.state_dict()) == []


def test_synovitis_covariate_detaches_and_owns_its_saved_effusion_values() -> None:
    model = _r7_config().make()
    with torch.no_grad():
        model.syn_beta.fill_(0.7)
    values = torch.arange(2 * 12, dtype=torch.float32).reshape(2, 12).requires_grad_()
    result = model.decompose_syn(values)
    expected_beta_gradient = values[:, 7].detach().sum().clone()
    with torch.no_grad():
        values[:, 7].add_(3)
    result[:, 8].sum().backward()
    assert values.grad is not None
    assert torch.equal(values.grad[:, 7], torch.zeros(2))
    assert torch.equal(values.grad[:, 8], torch.ones(2))
    assert model.syn_beta.grad is not None
    assert torch.equal(model.syn_beta.grad, expected_beta_gradient)


def test_synovitis_selector_preserves_logit_metadata_under_other_factory_defaults() -> (
    None
):
    model = _r7_config().make()
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            actual = model.decompose_syn(logits)
    finally:
        torch.set_default_dtype(previous)
    assert actual.dtype == logits.dtype
    assert actual.device == logits.device
    assert torch.equal(actual, logits)


def test_r7_requires_a_complete_checkpoint_before_mutating_parameters() -> None:
    model = _r7_config().make()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="names"):
        model.load_checkpoint({})
    assert all(
        torch.equal(value, initial[name]) for name, value in model.state_dict().items()
    )
    for name, bad in (
        ("enc.stem.conv.weight", torch.zeros(1)),
        ("state_prior", initial["state_prior"].double()),
    ):
        state = dict(initial)
        state[name] = bad
        with pytest.raises(ValueError, match="shape or dtype"):
            model.load_checkpoint(state)
    collision = dict(initial)
    collision["enc.stem.0.weight"] = initial["enc.stem.conv.weight"]
    with pytest.raises(ValueError, match="collide"):
        model.load_checkpoint(collision)
    assert all(
        torch.equal(value, initial[name]) for name, value in model.state_dict().items()
    )


def test_r7_rejects_incompatible_finding_and_channel_layouts() -> None:
    config = _r7_config()
    config.state_counts = (2,)
    with pytest.raises(
        ValueError,
        match=r"^State counts must match the canonical finding order\.$",
    ):
        config.make()
    config = _r7_config()
    config.encoder.channels_in = 1
    with pytest.raises(
        ValueError,
        match=r"^The released R7 model consumes three adjacent slices\.$",
    ):
        config.make()


def test_r7_encoder_forward_exposes_the_spatial_compile_boundary() -> None:
    model = _r7_config().make().eval()
    features = model.enc(torch.zeros(2, 3, 64, 96))
    assert features.shape == (2, 5, 2, 3)


@pytest.mark.parametrize("series_count", [16, 20])
def test_complete_r7_cost_counts_padding_and_every_trainable_readout(
    series_count: int,
) -> None:
    config = _r7_config()
    config.sp_key.channels_out = 3
    assert_cost_matches_torch(
        config,
        build_input=lambda: _cost_batch(series_count),
        run=_cost_forward,
        input_grid=(32, 64),
        seq_len=5,
        real_windows=7,
        batch_size=2,
        dtype=torch.float32,
        num_series=series_count,
    )


def _cost_batch(series_count: int) -> tuple[Tensor, Tensor, Tensor]:
    image = torch.randint(0, 256, (2, 5, 3, 32, 64), dtype=torch.uint8)
    meta = torch.arange(90, dtype=torch.float32).reshape(2, 5, 9) / 40
    for column in (0, 1, 6, 7):
        meta[..., column] = torch.arange(5) % 3
    # An unused slot holds the largest series code, so the membership width is observable.
    meta[0, 1, 6] = series_count - 1
    mask = torch.tensor(
        [[True, False, True, False, True], [True, True, True, True, False]],
    )
    return image, meta, mask


def _cost_forward(module: torch.nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    assert isinstance(module, R7)
    assert len(inputs) == 3
    outputs = module(inputs[0], inputs[1], inputs[2])
    return torch.cat([output.flatten() for output in outputs])


def _head_model() -> R7:
    config = _r7_config()
    config.channels_hidden = 4
    config.encoder.channels_hidden = (2, 3, 4, 6)
    return config.make().eval()


def _head_inputs() -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(5)
    _, meta, mask = _batch()
    return torch.randn(5, 12, 6, generator=generator), meta, mask


def _randomize(model: R7, *, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))


def test_r7_starts_without_saved_outputs_and_with_its_declared_buffers() -> None:
    model = _r7_config().make()
    assert model.area is None
    assert model.syn_specific is None
    assert model.has_bn is False
    assert model.state_valid.sum(1).tolist() == list(R7.Config().state_counts)
    assert model.ext_mask.tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_r7_head_adds_each_readout_bias_to_its_logits() -> None:
    model = _head_model()
    with torch.no_grad():
        for weight in (model.cls_w, model.state_w, model.bin_w):
            weight.zero_()
        model.bias.copy_(torch.arange(12) / 8 + 0.5)
        model.state_b.copy_(torch.arange(12 * 8).reshape(12, 8) / 16 - 2)
        model.bin_b.copy_(torch.arange(13) / 4 + 1)
    logits, states, binaries = model.head(*_head_inputs())
    assert torch.equal(logits, model.bias.expand(2, -1))
    padded = torch.where(model.state_valid, model.state_b, -math.inf)
    assert torch.equal(states, padded.expand(2, -1, -1))
    assert torch.equal(binaries, model.bin_b.expand(2, -1))


def test_r7_head_clamps_slot_and_view_codes_to_the_embedding_tables() -> None:
    model = _head_model()
    _randomize(model, seed=11)
    tokens, meta, mask = _head_inputs()

    def run(slot: int, view: int) -> Tensor:
        changed = meta.clone()
        changed[..., 0], changed[..., 1] = slot, view
        return model.head(tokens, changed, mask)[0]

    last_slot, last_view = model.num_slots - 1, model.num_views - 1
    assert torch.equal(run(99, 0), run(last_slot, 0))
    assert not torch.equal(run(last_slot, 0), run(last_slot - 1, 0))
    assert torch.equal(run(-7, 0), run(0, 0))
    assert not torch.equal(run(0, 0), run(1, 0))
    assert torch.equal(run(0, 99), run(0, last_view))
    assert not torch.equal(run(0, last_view), run(0, last_view - 1))
    assert torch.equal(run(0, -7), run(0, 0))
    assert not torch.equal(run(0, 0), run(0, 1))


@pytest.mark.parametrize("prior", [0.0, 1.0])
def test_r7_head_clamps_the_implied_probability_before_its_logit(
    prior: float,
) -> None:
    model = _head_model()
    inputs = _head_inputs()
    with torch.no_grad():
        model.state_prior.fill_(prior)
        plain = model.head(*inputs)[0]
        model.state_gate.fill_(1)
        gated = model.head(*inputs)[0]
    edge = 1e-3 if prior == 0 else 1 - 1e-3
    shift = torch.full_like(gated, torch.logit(torch.tensor(edge)).item())
    torch.testing.assert_close(gated - plain, shift, rtol=1e-5, atol=1e-5)


def _extent_reference(
    model: R7,
    logits: Tensor,
    meta: Tensor,
    mask: Tensor,
) -> Tensor:
    # Loop the released rule: view means, softmax over present views, span, planes.
    findings = len(model.state_counts)
    area = torch.zeros(len(logits), findings, dtype=torch.float64)
    for study in range(len(logits)):
        real = [window for window in range(mask.shape[1]) if mask[study, window]]
        planes = {
            plane: torch.zeros(findings, dtype=torch.float64) for plane in range(3)
        }
        for series in sorted({int(meta[study, window, 6]) for window in real}):
            members = [w for w in real if int(meta[study, w, 6]) == series]
            by_view: dict[int, list[Tensor]] = {}
            for window in members:
                view = min(max(int(meta[study, window, 1]), 0), model.num_views - 1)
                by_view.setdefault(view, []).append(
                    torch.sigmoid(logits[study, window].double()),
                )
            mean = torch.zeros(findings, dtype=torch.float64)
            for finding in range(findings):
                present = sorted(by_view)
                raw = torch.stack(
                    [model.ext_view[finding, v].double() for v in present],
                )
                weights = torch.softmax(raw, 0)
                for weight, view in zip(weights, present, strict=True):
                    mean[finding] += (
                        weight * torch.stack(by_view[view])[:, finding].mean()
                    )
            span = torch.stack(
                [
                    torch.expm1(meta[study, w, 3].double()).clamp_min(0)
                    * meta[study, w, 8].double()
                    for w in members
                ],
            ).mean()
            plane = round(
                float(
                    torch.stack([meta[study, w, 7].double() for w in members]).mean(),
                ),
            )
            planes[plane] = torch.maximum(planes[plane], mean * span)
        sagittal, coronal, axial = planes[0], planes[1], planes[2]
        medial, lateral, patellofemoral = model.extent_classes
        for finding in (medial, lateral):
            area[study, finding] = sagittal[finding] * coronal[finding]
        area[study, patellofemoral] = axial[patellofemoral] * sagittal[patellofemoral]
    return area


def test_r7_extent_area_matches_a_looped_reference_for_awkward_metadata() -> None:
    model = _head_model()
    _randomize(model, seed=3)
    generator = torch.Generator().manual_seed(8)
    windows = 9
    logits = torch.randn(2, windows, 12, generator=generator)
    meta = torch.zeros(2, windows, 9)
    meta[..., 1] = torch.tensor([0, 5, 9, -2, 5, 0, 1, 5, 2.0])
    meta[..., 3] = torch.linspace(-0.4, 1.3, windows)
    meta[..., 6] = torch.tensor([3, 3, 17, 17, 17, 19, 3, 19, 19.0])
    meta[..., 7] = torch.tensor([0, 0, 1, 1, 1, 2, 0, 2, 2.0])
    meta[..., 8] = torch.linspace(0.5, 2.0, windows)
    meta[1] = meta[0].roll(2, 0)
    mask = torch.ones(2, windows, dtype=torch.bool)
    mask[0, 6] = mask[1, 2] = False
    actual = model.extent_area(logits, meta=meta, mask=mask)
    expected = _extent_reference(model, logits, meta, mask)
    assert expected.abs().max() > 0
    torch.testing.assert_close(actual.double(), expected, rtol=1e-4, atol=1e-6)


def _positions(height: int, width: int, frequencies: tuple[float, ...]) -> Tensor:
    rows: list[list[float]] = []
    for index in range(height * width):
        row, column = divmod(index, width)
        y = -1 + 2 * row / (height - 1)
        x = -1 + 2 * column / (width - 1)
        angles_y = [y * f * math.pi / 2 for f in frequencies]
        angles_x = [x * f * math.pi / 2 for f in frequencies]
        rows.append(
            [math.sin(a) for a in angles_y]
            + [math.cos(a) for a in angles_y]
            + [math.sin(a) for a in angles_x]
            + [math.cos(a) for a in angles_x],
        )
    return torch.tensor(rows)


def _spatial_model() -> R7:
    model = _head_model()
    _randomize(model, seed=2)
    return model


def test_r7_spatial_pool_attends_by_fourier_position_per_finding() -> None:
    model = _spatial_model()
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(4, 6, 2, 3, generator=generator)
    flat = features.flatten(2).transpose(1, 2)
    positions = _positions(2, 3, model.frequencies)
    keys = torch.tanh(
        model.sp_key(model.norm_spatial(flat)) + model.sp_pos(positions)[None],
    )
    weights = torch.softmax(keys @ model.sp_q.t(), dim=1)
    pooled = weights.transpose(1, 2) @ flat
    torch.testing.assert_close(
        model.spatial_pool(features),
        model.norm_sp_head(pooled),
        rtol=1e-5,
        atol=1e-5,
    )


def test_r7_spatial_pool_follows_the_input_device_and_dtype() -> None:
    model = _spatial_model()
    with torch.device("meta"):
        meta_model = _head_model()
    meta_output = meta_model.spatial_pool(torch.empty(4, 6, 2, 3, device="meta"))
    assert meta_output.device.type == "meta"
    model.sp_key.bfloat16()
    model.sp_pos.bfloat16()
    features = torch.randn(4, 6, 2, 3).bfloat16()
    assert model.spatial_pool(features).dtype == torch.bfloat16


def test_softmax_cost_counts_both_passes_and_two_forward_reductions() -> None:
    dtype = torch.bfloat16
    expected = (
        elementwise_cost(primal=18, adjoint=18, channels=6, dtype=dtype)
        + reduction_cost(input_elements=6, output_groups=3, dtype=dtype).tile(2)
        + reduction_cost(
            input_elements=6,
            output_groups=3,
            phase="adjoint",
            dtype=dtype,
        )
    )
    assert _softmax_cost(elements=6, groups=3, dtype=dtype) == expected


def test_readout_cost_counts_class_weights_and_their_reduction() -> None:
    dtype = torch.bfloat16
    classes, hidden, rows = 3, 4, 5
    elements = rows * classes * hidden
    expected = elementwise_cost(
        primal=elements + rows * classes,
        adjoint=2 * elements,
        channels=classes * hidden,
        params=classes * (hidden + 1),
        rows=rows,
        dtype=dtype,
    ) + reduction_cost(
        input_elements=elements,
        output_groups=rows * classes,
        dtype=dtype,
    )
    actual = _readout_cost(classes=classes, hidden=hidden, rows=rows, dtype=dtype)
    assert actual == expected


def test_attention_matches_explicit_finding_weighted_sum() -> None:
    config = FindingAttention.Config()
    config.channels_in = 4
    config.channels_out = 5
    config.channels_hidden = 6
    config.dropout.p = 0.0
    with host_agnostic_numerics():
        model = config.make()
        with torch.no_grad():
            for parameter in model.att.parameters():
                parameter.zero_()
        features = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4) / 11
        normalized = model.norm(features)
        pooled = normalized.mean(dim=1).unsqueeze(1)
        assert torch.equal(
            model(features).float(),
            ((pooled * model.clsW).sum(-1) + model.clsb).float(),
        )


def test_attention_reset_covers_injected_projection_state() -> None:
    config = FindingAttention.Config()
    config.channels_in, config.channels_out = 4, 5
    config.channels_hidden = 6
    model = config.make()
    with torch.no_grad():
        model.clsb.fill_(3)
    model.reset_parameters()
    assert torch.equal(model.clsb, torch.zeros(5))
    assert model(torch.zeros(2, 3, 4)).shape == (2, 5)


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_attention_cost_matches_complete_torch_invocation(dropout: float) -> None:
    config = FindingAttention.Config(channels_in=4, channels_out=5)
    config.channels_hidden = 6
    config.dropout.p = dropout
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 4, requires_grad=True),
        seq_len=3,
        batch_size=2,
        dtype=torch.float32,
    )


def test_injected_head_projections_retain_parameter_costs() -> None:
    config = FindingAttention.Config()
    config.channels_in, config.channels_out = 4, 5
    config.channels_hidden = 6
    for bias in (False, True):
        config.proj_in.bias = config.proj_out.bias = bias
        measured = cost(config, seq_len=3, batch_size=2, dtype=torch.float32)
        assert measured.params == sum(
            parameter.numel() for parameter in config.make().parameters()
        )


def _raptor_config() -> Raptor.Config:
    config = Raptor.Config()
    config.chunk_size = 16
    backbone = _backbone(config)
    backbone.image_size = (64, 96)
    backbone.stem.channels_hidden = (1, 2)
    backbone.channels_hidden = (2, 4, 2, 4)
    backbone.blocks_per_stage = (1, 1, 1, 1)
    for block in blocks_of(backbone, CoAtTransformer.Config):
        block.attention.channels_head = 2
        block.attention.relative.mlp.channels_hidden = 2
    config.head.channels_hidden = 2
    return config


def _backbone(config: Raptor.Config) -> CoAtNet.Config:
    assert isinstance(config.backbone, CoAtNet.Config)
    return config.backbone


def test_raptor_chunked_and_whole_bag_forwards_match_their_golden() -> None:
    config = _raptor_config()
    _backbone(config).image_size = (32, 64)
    with host_agnostic_numerics():
        model = config.make().eval()
        model.load_checkpoint(_literal_weights(model))
        # Two studies of five three-slice windows; chunks of four split the bag.
        image = torch.arange(2 * 5 * 3 * 32 * 64, dtype=torch.float32) % 251 / 127
        model.chunk_size = 4
        chunked = model(image.reshape(2, 5, 3, 32, 64))
        model.chunk_size = math.inf
        whole = model(image.reshape(2, 5, 3, 32, 64))
    # Chunking only splits the encoder batch, so both passes agree bit for bit.
    assert torch.equal(chunked, whole)
    assert_tensor_golden(_CWD / "testdata" / "raptor_model.pt", {"output": whole})


def test_raptor_loads_released_finding_head_names() -> None:
    model = _raptor_config().make()
    state = _literal_weights(model)
    released = {
        name.removeprefix("head."): value
        for name, value in state.items()
        if name.startswith(("head.clsW", "head.clsb", "head.norm.", "head.att."))
    }
    released |= {name: value for name, value in state.items() if name[:5] != "head."}
    model.load_checkpoint(released)
    assert mismatches(state, model.state_dict()) == []


def test_raptor_checkpoint_remapping_rejects_missing_and_colliding_names() -> None:
    model = _raptor_config().make()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    for bad in (
        dict[str, torch.Tensor](),
        {**state, "clsW": state["head.clsW"]},
        {**state, "head.clsb": torch.zeros(2)},
        {**state, "head.clsW": state["head.clsW"].double()},
    ):
        with pytest.raises(ValueError, match="Checkpoint"):
            model.load_checkpoint(bad)
        assert all(
            torch.equal(value, state[name])
            for name, value in model.state_dict().items()
        )


def test_raptor_head_sizes_derive_from_backbone_and_reset_all_modules() -> None:
    model = _raptor_config().make()
    assert model.head.clsW.shape == (12, 4)
    with torch.no_grad():
        model.head.clsb.fill_(3)
    model.reset_parameters()
    assert torch.equal(model.head.clsb, torch.zeros(12))
    config = _raptor_config()
    config.chunk_size = 0
    with pytest.raises(ValueError, match="positive"):
        config.make()


@pytest.mark.parametrize("chunk_size", [math.inf, 4])
def test_raptor_cost_counts_whole_bag_without_duplicating_parameters(
    chunk_size: float,
) -> None:
    config = _raptor_config()
    backbone = _backbone(config)
    backbone.image_size = (32, 64)
    backbone.blocks_per_stage = (1, 1, 1, 1)
    config.chunk_size = chunk_size
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 5, 3, 32, 64, requires_grad=True),
        seq_len=5,
        input_grid=(32, 64),
        batch_size=2,
        dtype=torch.float32,
    )


def test_raptor_accepts_a_single_window_chunk_and_rejects_less() -> None:
    config = _raptor_config()
    config.chunk_size = 1
    assert config.make().chunk_size == 1
    for chunk_size in (0, 0.5):
        config = _raptor_config()
        config.chunk_size = chunk_size
        with pytest.raises(
            ValueError,
            match=r"^Encoder chunk size must be positive\.$",
        ):
            config.make()


def _tiny_config() -> TinyConvNeXt.Config:
    config = TinyConvNeXt.Config()
    config.channels_hidden = (2, 3, 4, 5)
    config.blocks_per_stage = (1, 1, 1, 1)
    assert isinstance(config.head, ConvNeXtHead.Config)
    assert isinstance(config.head.proj_out, Linear.Config)
    config.head.proj_out.channels_out = 2
    return config


def _tiny_image() -> Tensor:
    return (
        torch.arange(4 * 3 * 32 * 64, dtype=torch.float32).reshape(4, 3, 32, 64) % 251
    ) / 127


def test_tiny_constructor_state_draws_and_features_match_their_golden() -> None:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics(), torch.no_grad():
        torch.manual_seed(19)
        model = _tiny_config().make().eval()
        record = {
            f"constructor/initial/{name}": value.clone()
            for name, value in model.state_dict().items()
        }
        record["constructor/rng"] = rng_fingerprint()
        rates = [
            module.drop_path
            for module in model.modules()
            if isinstance(module, ConvNeXtBlock)
        ]
        model.set_submodule("head.proj_out", Identity.Config(5).make(), strict=True)
        record["constructor/features"] = model(_tiny_image())
    assert rates == [0.1 * index / 3.0 for index in range(4)]
    assert_tensor_golden(_CWD / "testdata" / "tiny_model.pt", record)


def test_checkpoint_mapping_is_complete_atomic_and_checks_source_gamma_shape() -> None:
    model = _tiny_config().make()
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    without_gamma = {
        name: value
        for name, value in saved.items()
        if name != "stages.0.blocks.0.gamma"
    }
    for bad in (
        dict[str, torch.Tensor](),
        {**saved, "features.0.0.weight": saved["stem.conv.weight"]},
        {**without_gamma, "features.1.0.layer_scale": torch.zeros(2)},
        {**saved, "head.proj_out.weight": saved["head.proj_out.weight"].double()},
    ):
        with pytest.raises(ValueError, match="Checkpoint"):
            model.load_checkpoint(bad)
        assert all(
            torch.equal(value, saved[name])
            for name, value in model.state_dict().items()
        )
    model.load_checkpoint(saved)


@pytest.mark.parametrize(
    ("released", "native"),
    [
        ("features.0.0.weight", "stem.conv.weight"),
        ("features.0.1.bias", "stem.norm.bias"),
        ("features.1.0.layer_scale", "stages.0.blocks.0.gamma"),
        ("features.3.0.block.0.weight", "stages.1.blocks.0.conv_dw.weight"),
        ("features.5.2.block.2.bias", "stages.2.blocks.2.norm.bias"),
        ("features.7.1.block.3.weight", "stages.3.blocks.1.mlp.up_proj.weight"),
        ("features.1.0.block.5.bias", "stages.0.blocks.0.mlp.down_proj.bias"),
        ("features.6.1.weight", "stages.3.downsample.1.weight"),
        ("classifier.0.weight", "head.norm.weight"),
        ("classifier.2.bias", "head.proj_out.bias"),
        ("stages.0.blocks.0.gamma", "stages.0.blocks.0.gamma"),
    ],
)
def test_torchvision_initializer_names_map_onto_the_native_convnext(
    released: str,
    native: str,
) -> None:
    assert _checkpoint_name(released) == native


def test_reset_replays_constructor_draw_order_and_every_initial_tensor() -> None:
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics():
        torch.manual_seed(19)
        model = _tiny_config().make()
        constructed = {
            name: value.clone() for name, value in model.state_dict().items()
        }
        position = rng_fingerprint()
        torch.manual_seed(19)
        model.reset_parameters()
        assert mismatches(constructed, model.state_dict()) == []
        assert torch.equal(rng_fingerprint(), position)


def test_production_tiny_keeps_every_stage_and_all_initializer_classes() -> None:
    config = TinyConvNeXt.Config().copy_tree().finalize()
    assert config.channels_hidden == (96, 192, 384, 768)
    assert config.blocks_per_stage == (3, 3, 9, 3)
    assert config.drop_path == 0.1
    assert isinstance(config.head, ConvNeXtHead.Config)
    assert isinstance(config.head.proj_out, Linear.Config)
    assert config.head.proj_out.channels_out == 1000


def test_explicit_blocks_preserve_caller_drop_path_policies() -> None:
    config = TinyConvNeXt.Config()
    config.channels_hidden = (2, 3, 4, 5)
    assert isinstance(config.stage, ConvNeXtStage.Config)
    template = config.stage
    config.stage = [template.copy_tree() for _ in config.channels_hidden]
    counts = (2, 1, 1, 1)
    for stage, count in zip(config.stage, counts, strict=True):
        assert isinstance(stage.block, ConvNeXtBlock.Config)
        stage.block.drop_path = 0.375
        stage.block = [stage.block.copy_tree() for _ in range(count)]
    expected = tuple(
        config.drop_path * index / (sum(counts) - 1.0) for index in range(sum(counts))
    )
    assert (
        config.drop_path_schedule(config.drop_path, sum(config.block_counts()))
        == expected
    )
    resolved = config.copy_tree().finalize()
    assert isinstance(resolved.stage, list)
    rates: list[float] = []
    for stage in resolved.stage:
        assert isinstance(stage.block, list)
        rates.extend(block.drop_path for block in stage.block)
    assert tuple(rates) == (0.375,) * sum(counts)
    assert (
        resolved.drop_path_schedule(resolved.drop_path, sum(resolved.block_counts()))
        == expected
    )


def test_single_block_tiny_has_zero_stochastic_depth() -> None:
    config = TinyConvNeXt.Config()
    config.channels_hidden = (5,)
    config.blocks_per_stage = (1,)
    assert config.drop_path_schedule(config.drop_path, sum(config.block_counts())) == (
        0.0,
    )
    resolved = config.copy_tree().finalize()
    assert isinstance(resolved.stage, list)
    assert isinstance(resolved.stage[0].block, list)
    assert resolved.stage[0].block[0].drop_path == 0.0
    model = config.make()
    assert model(torch.randn(2, 3, 8, 12)).shape == (2, 1_000)


def test_tiny_mlp_cost_counts_both_owned_projections() -> None:
    assert_cost_matches_torch(
        SwiGLU.Config(channels_in=3, gate=False, round_to=1, act=gelu, bias=True),
        build_input=lambda: torch.randn(2, 5, 3, requires_grad=True),
        seq_len=5,
        batch_size=2,
        dtype=torch.float32,
    )


def test_tiny_classifier_cost_counts_pooling_normalization_and_projection() -> None:
    assert_cost_matches_torch(
        ConvNeXtHead.Config(
            channels_in=3,
            proj_out=Linear.Config(channels_out=5, bias=True),
        ),
        build_input=lambda: torch.randn(2, 3, 4, 6, requires_grad=True),
        input_grid=(4, 6),
        batch_size=2,
        dtype=torch.float32,
    )


def test_complete_tiny_cost_counts_all_independent_blocks_and_initializer_classes() -> (
    None
):
    config = _tiny_config()
    config.channels_hidden = (2, 3, 4, 5)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(4, 3, 32, 64, requires_grad=True),
        input_grid=(32, 64),
        batch_size=4,
        dtype=torch.float32,
    )


def test_composed_tiny_cost_advances_past_the_pooled_head() -> None:
    backbone = _tiny_config()
    backbone.channels_hidden = (2, 3, 4, 5)
    assert isinstance(backbone.head, ConvNeXtHead.Config)
    assert isinstance(backbone.head.proj_out, Linear.Config)
    backbone.head.proj_out.channels_out = 7
    config = Sequential.Config(elements=[backbone, Linear.Config(7, 11)])
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(4, 3, 32, 64, requires_grad=True),
        input_grid=(32, 64),
        seq_len=32 * 64,
        batch_size=4,
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    "source_name",
    [
        "features.1.0.not_a_block.0.weight",
        "features.1.0.blocks.0.weight",
        "features.01.0.block.0.weight",
        "features.1.0.block.0.weight.extra",
    ],
)
def test_checkpoint_source_block_grammar_is_validated_before_loading(
    source_name: str,
) -> None:
    config = TinyConvNeXt.Config()
    config.channels_hidden = (3,)
    config.blocks_per_stage = (2,)
    assert isinstance(config.head, ConvNeXtHead.Config)
    assert isinstance(config.head.proj_out, Linear.Config)
    config.head.proj_out.channels_out = 5
    model = config.make()
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    state = dict(saved)
    value = state.pop("stages.0.blocks.0.conv_dw.weight")
    state[source_name] = value
    with pytest.raises(ValueError, match="Checkpoint"):
        model.load_checkpoint(state)
    assert all(
        torch.equal(value, saved[name]) for name, value in model.state_dict().items()
    )


def test_tiny_drop_path_spreads_the_maximum_linearly_over_the_blocks() -> None:
    assert tiny_drop_path(0.1, 1) == (0.0,)
    assert tiny_drop_path(0.1, 2) == (0.0, 0.1)
    assert tiny_drop_path(0.3, 4) == tuple(0.3 * index / 3.0 for index in range(4))


@pytest.mark.parametrize(
    "source_name",
    [
        "classifier.1.weight",
        "classifier.0.running_mean",
        "features.1.0.weight",
        "features.0.2.weight",
        "features.0.0.gamma",
        "features.2.0.layer_scale",
        "features.1.x.layer_scale",
        "features.2.0.block.0.weight",
        "features.1.x.block.0.weight",
        "features.1.0.block.1.weight",
        "features.1.0.block.0.scale",
        "features.0.0.0",
    ],
)
def test_checkpoint_names_outside_the_source_grammar_are_rejected(
    source_name: str,
) -> None:
    with pytest.raises(ValueError, match=r"^Checkpoint source name is invalid: "):
        checkpoint_state({source_name: torch.zeros(2)})


def test_checkpoint_layer_scale_must_be_a_channel_vector_in_source_layout() -> None:
    # checkpoint_state requires torchvision's (channels, 1, 1) layer-scale layout.
    state = {"features.1.0.layer_scale": torch.arange(4.0).reshape(4, 1, 1)}
    assert checkpoint_state(state)["stages.0.blocks.0.gamma"].tolist() == [0, 1, 2, 3]
    for shape in ((4, 2, 2), (4, 1), (4, 1, 1, 1)):
        with pytest.raises(ValueError, match=r"^Checkpoint layer-scale shape differs"):
            checkpoint_state({"features.1.0.layer_scale": torch.zeros(shape)})


def test_encoder_checks_complete_checkpoint_and_strict_structure(
    tmp_path: Path,
) -> None:
    backbone = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 6))
    path = tmp_path / "initializer.pt"
    torch.save(backbone.state_dict(), path)
    config = ImageNetEncoder.Config()
    config.backbone = InlineConfig(
        nn.Sequential,
        nn.Linear(4, 5),
        nn.Linear(5, 6),
    )
    config.checkpoint.path = path
    config.checkpoint.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    config.classifier = "1"
    encoder = config.make()
    media = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    assert torch.equal(encoder(media), backbone[0](media))
    config.checkpoint.sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="hashes to"):
        config.make()


def test_study_model_keeps_study_and_window_axes_distinct() -> None:
    config = Raptor.Config()
    config.backbone = InlineConfig(nn.Identity)
    config.head.channels_in = 4
    config.head.channels_out = 5
    config.head.channels_hidden = 6
    config.head.dropout.p = 0.0
    model = config.make()
    media = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    assert torch.equal(model(media, label=torch.zeros(2, 5)), model.head(media))


@pytest.mark.parametrize("parameterized_encoder", [False, True])
def test_training_meta_placement_resets_every_owned_study_parameter(
    parameterized_encoder: bool,
) -> None:
    model = Raptor.Config()
    if parameterized_encoder:
        encoder = Linear.Config()
        encoder.channels_in, encoder.channels_out = 4, 7
        model.backbone = encoder
    else:
        model.backbone = InlineConfig(nn.Identity)
    model.head.channels_in = 7 if parameterized_encoder else 4
    model.head.channels_out, model.head.channels_hidden = 5, 6
    model.head.dropout.p = 0.0
    config = TrainStep.Config(model=model)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        eager = config.make().model
        config.device_init = "meta"
        torch.manual_seed(19)
        materialized = config.make().model
    assert isinstance(materialized, Raptor)
    assert all(
        torch.equal(value, eager.state_dict()[name])
        for name, value in materialized.state_dict().items()
    )
    media = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 17
    assert torch.equal(materialized(media), cast(TensorFn, eager)(media))


def test_meta_pretrained_encoder_reloads_complete_features_before_training(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "tiny_model.pt")
    path = tmp_path / "imagenet.pt"
    torch.save(
        {
            name.removeprefix("constructor/initial/"): value
            for name, value in expected.items()
            if name.startswith("constructor/initial/")
        },
        path,
    )
    encoder = ImageNetEncoder.Config()
    encoder.backbone = _tiny_config()
    encoder.checkpoint.path = path
    encoder.checkpoint.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    study = Raptor.Config()
    study.backbone = encoder
    study.head.channels_in, study.head.channels_out = 5, 6
    study.head.channels_hidden = 3
    config = TrainStep.Config(model=study)
    config.parallelism = NoParallel.Config(device="cpu")
    config.compile = None
    with host_agnostic_numerics():
        eager = config.make().model
        config.device_init = "meta"
        materialized = config.make().model
        assert isinstance(eager, Raptor)
        assert isinstance(materialized, Raptor)
        saved = {
            name: value.clone() for name, value in eager.backbone.state_dict().items()
        }
        assert saved.keys() == materialized.backbone.state_dict().keys()
        assert all(
            torch.equal(value, saved[name])
            for name, value in materialized.backbone.state_dict().items()
        )
        assert all(torch.isfinite(value).all() for value in materialized.parameters())
        with torch.no_grad():
            for parameter in materialized.parameters():
                parameter.fill_(torch.nan)
        materialized.reset_parameters()
        assert all(
            torch.equal(value, saved[name])
            for name, value in materialized.backbone.state_dict().items()
        )
        # Evaluation mode disables stochastic depth, as in the stored features.
        assert torch.equal(
            cast(TensorFn, materialized.backbone.eval())(_tiny_image()),
            expected["constructor/features"],
        )


def test_imagenet_encoder_loads_every_checkpoint_tensor_before_removing_classifier(
    tmp_path: Path,
) -> None:
    expected = read_tensors(_CWD / "testdata" / "tiny_model.pt")
    path = tmp_path / "imagenet.pt"
    torch.save(
        {
            name.removeprefix("constructor/initial/"): value
            for name, value in expected.items()
            if name.startswith("constructor/initial/")
        },
        path,
    )
    config = ImageNetEncoder.Config()
    config.backbone = _tiny_config()
    config.checkpoint.path = path
    config.checkpoint.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with torch.random.fork_rng(devices=[]), host_agnostic_numerics(), torch.no_grad():
        torch.manual_seed(19)
        encoder = config.make().eval()
        assert torch.equal(rng_fingerprint(), expected["constructor/rng"])
        assert torch.equal(encoder(_tiny_image()), expected["constructor/features"])
        assert "backbone.head.norm.weight" in encoder.state_dict()
        assert "backbone.head.proj_out.weight" not in encoder.state_dict()


def test_backbone_config_names_its_native_architecture() -> None:
    config = ImageNetEncoder.Config()
    rendered = config.pformat(hide_default_values=False)
    assert "TinyConvNeXt.Config" in rendered
    assert "LazyObjectProxy" not in rendered


def test_encoder_and_study_cost_match_complete_torch_architecture() -> None:
    # ConvNeXt's three downsamplers require both spatial extents to be >=32.
    # Meta tensors keep the narrow four-stage encoder free of weight allocation.
    encoder = _Architecture.Config()
    assert isinstance(encoder.backbone, ConvNeXt.Config)
    encoder.backbone.channels_hidden = (2, 3, 4, 5)
    encoder.backbone.blocks_per_stage = (1, 1, 1, 1)
    study = Raptor.Config()
    study.backbone = encoder
    with torch.device("meta"):
        assert_cost_matches_torch(
            encoder,
            build_input=lambda: torch.randn(2, 3, 32, 64, requires_grad=True),
            input_grid=(32, 64),
            batch_size=2,
            dtype=torch.float32,
        )
        assert_cost_matches_torch(
            study,
            build_input=lambda: torch.randn(2, 4, 3, 32, 64, requires_grad=True),
            input_grid=(32, 64),
            seq_len=4,
            batch_size=2,
            dtype=torch.float32,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_study_cost_owns_encoder_and_head_parameters(dtype: torch.dtype) -> None:
    config = Raptor.Config(backbone=ImageNetEncoder.Config())
    measured = cost(
        config,
        input_grid=(32, 64),
        seq_len=3,
        batch_size=2,
        dtype=dtype,
    )
    # Torchvision's 28,589,128 includes a 768-to-1000 classifier, removed here.
    encoder = 28_589_128 - (768 * 1_000 + 1_000)
    head = 2 * 768 + 768 * 256 + 256 + 256 * 12 + 12 + 12 * 768 + 12
    assert measured.params == encoder + head
    assert {key[-1] for key in measured.cells} == {dtype}


def test_encoder_cost_follows_custom_widths_depths_and_geometry() -> None:
    config = _Architecture.Config()
    backbone = TinyConvNeXt.Config()
    backbone.channels_hidden = (4, 5)
    backbone.blocks_per_stage = (1, 2)
    config.backbone = backbone
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 16, 24, requires_grad=True),
        input_grid=(16, 24),
        batch_size=2,
        dtype=torch.float32,
    )
    assert backbone.head == TinyConvNeXt.Config().head


def test_encoder_cost_accepts_the_shared_backbone_and_declared_feature_head() -> None:
    config = _Architecture.Config()
    backbone = ConvNeXt.Config()
    backbone.channels_hidden = (4, 5)
    backbone.blocks_per_stage = (1, 2)
    config.backbone = backbone
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(2, 3, 16, 24, requires_grad=True),
        input_grid=(16, 24),
        batch_size=2,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("injection", ["classifier", "function", "config", "head"])
def test_encoder_cost_rejects_an_unmodeled_architecture(injection: str) -> None:
    config = ImageNetEncoder.Config()
    if injection == "classifier":
        config.classifier = "features"
    elif injection == "function":
        config.backbone = InlineConfig(nn.Identity)
    elif injection == "head":
        backbone = TinyConvNeXt.Config()
        backbone.head = Identity.Config()
        config.backbone = backbone
    else:
        config.backbone = FindingAttention.Config(channels_in=4, channels_out=5)
    with pytest.raises(ValueError, match="cost model"):
        cost(config, input_grid=(32, 64), batch_size=2, dtype=torch.float32)


class _Architecture(ImageNetEncoder):
    class Config(Makes["_Architecture"], ImageNetEncoder.Config):
        pass

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        self.backbone = config.backbone.make()
        self.backbone.set_submodule(
            config.classifier,
            module=Identity.Config().make(),
            strict=True,
        )


def test_feature_width_requires_a_convnext_backbone() -> None:
    config = ImageNetEncoder.Config()
    config.backbone = cast("Makeable[nn.Module]", Linear.Config())
    with pytest.raises(TypeError, match="ConvNeXt"):
        config.feature_width()


def _sequence_encoder(path: Path) -> ImageNetEncoder:
    backbone = nn.Sequential(nn.Linear(4, 5), nn.Linear(5, 6))
    torch.save(backbone.state_dict(), path)
    config = ImageNetEncoder.Config()
    config.backbone = InlineConfig(nn.Sequential, nn.Linear(4, 5), nn.Linear(5, 6))
    config.checkpoint.path = path
    config.checkpoint.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    config.classifier = "1"
    return config.make()


def test_encoder_requires_its_classifier_path_to_exist(tmp_path: Path) -> None:
    path = tmp_path / "initializer.pt"
    _ = _sequence_encoder(path)
    config = ImageNetEncoder.Config()
    config.backbone = InlineConfig(nn.Sequential, nn.Linear(4, 5), nn.Linear(5, 6))
    config.checkpoint.path = path
    config.checkpoint.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    config.classifier = "9"
    with pytest.raises(AttributeError):
        config.make()


def test_encoder_reset_rejects_a_checkpoint_that_does_not_fit_the_backbone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = _sequence_encoder(tmp_path / "initializer.pt")
    state = {"0.weight": torch.zeros(5, 4), "0.bias": torch.zeros(5)}
    monkeypatch.setattr(
        encoder,
        "_read_checkpoint",
        lambda: {**state, "extra": state["0.bias"]},
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        encoder.reset_parameters()
    monkeypatch.setattr(
        encoder,
        "_read_checkpoint",
        lambda: {"0.weight": state["0.weight"]},
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        encoder.reset_parameters()


def test_encoder_reads_its_checkpoint_on_cpu_as_plain_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, object]] = []
    real = torch.load

    def spy(path: Path, *, map_location: str, weights_only: bool) -> object:
        seen.append({"map_location": map_location, "weights_only": weights_only})
        return cast(
            object,
            real(path, map_location=map_location, weights_only=weights_only),
        )

    monkeypatch.setattr(torch, "load", spy)
    _ = _sequence_encoder(tmp_path / "initializer.pt")
    assert seen == [{"map_location": "cpu", "weights_only": True}]


def _side_config() -> ResNet.Config:
    config = ResNet.Config(channels_in=1, channels_out=1)
    config.stem.channels_out = 2
    config.channels_hidden = (2, 2, 2, 2)
    return config


def test_resnet34_laterality_forward_matches_its_golden() -> None:
    with host_agnostic_numerics():
        model = _side_config().make().eval()
        model.load_torchvision_state_dict(_literal_weights(model))
        # The laterality model reads one MRI channel; ResNet.Config(channels_in=1).
        image = (
            torch.arange(3 * 1 * 32 * 64, dtype=torch.float32).reshape(3, 1, 32, 64)
            / 101
        )
        output = model(image)
    assert_tensor_golden(_CWD / "testdata" / "side_model.pt", {"output": output})


def _anatomy_config() -> ReleasedVNet.Config:
    config = ReleasedVNet.Config(channels_in=1, channels_out=4)
    config.channels_hidden = (2, 2, 2, 2, 2)
    return config


def test_anatomy_forward_matches_its_golden() -> None:
    with host_agnostic_numerics():
        model = _anatomy_config().make().eval()
        model.load_checkpoint(_literal_weights(model))
        # The released VNet reads one channel and halves both sides four times.
        x = torch.arange(2 * 1 * 16 * 32, dtype=torch.float32).reshape(2, 1, 16, 32)
        output = model(x / 101)
    assert_tensor_golden(_CWD / "testdata" / "anatomy_model.pt", {"output": output})


def test_anatomy_rejects_incomplete_checkpoint_before_mutating_parameters() -> None:
    model = _anatomy_config().make()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="names"):
        model.load_checkpoint({})
    assert all(
        torch.equal(value, before[name]) for name, value in model.state_dict().items()
    )


def test_anatomy_checkpoint_validates_prefix_shape_dtype_and_collisions() -> None:
    model = _anatomy_config().make()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_checkpoint({"model.net." + name: value for name, value in state.items()})
    wrong = dict(state)
    wrong["input.layer.conv.weight"] = state["input.layer.conv.weight"].double()
    with pytest.raises(ValueError, match="shape or dtype"):
        model.load_checkpoint(wrong)
    wrong = dict(state)
    wrong["in_tr.conv_block.conv.weight"] = state["input.layer.conv.weight"]
    with pytest.raises(ValueError, match="collide"):
        model.load_checkpoint(wrong)


@pytest.mark.parametrize(
    ("released", "native"),
    [
        ("model.net.in_tr.conv_block.conv.weight", "input.layer.conv.weight"),
        # Split, as in the production pattern: codespell rewrites the whole token.
        ("in_tr.conv_block.ad" + "n.N.running_var", "input.layer.norm.running_var"),
        ("down_tr128.ops.2.conv_block.ad" + "n.N.bias", "down.2.blocks.2.norm.bias"),
        ("up_tr32.ops.0.conv_block.ad" + "n.N.weight", "up.3.blocks.0.norm.weight"),
        ("down_tr32.down_conv.weight", "down.0.proj.conv.weight"),
        ("down_tr64.bn1.bias", "down.1.proj.norm.bias"),
        ("down_tr256.ops.0.conv_block.conv.bias", "down.3.blocks.0.conv.bias"),
        ("up_tr256.up_conv.weight", "up.0.proj.weight"),
        ("up_tr32.bn1.running_mean", "up.3.norm.running_mean"),
        ("up_tr64.ops.0.conv_block.conv.weight", "up.2.blocks.0.conv.weight"),
        ("out_tr.conv2.weight", "output.proj.weight"),
    ],
)
def test_released_anatomy_names_map_onto_the_native_vnet(
    released: str,
    native: str,
) -> None:
    assert _vnet_name(released) == native


def test_anatomy_production_configuration_retains_complete_model_state() -> None:
    with torch.device("meta"):
        model = ReleasedVNet.Config(channels_in=1, channels_out=4).make()
        control = _anatomy_config().make()
    assert model.state_dict().keys() == control.state_dict().keys()
    assert sum(value.numel() for value in model.state_dict().values()) == 9_368_112


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
