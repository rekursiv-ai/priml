"""INVAE: the published architecture, its checkpoint names, and its arithmetic.

``testdata/invae.pt`` freezes the round trip at golden size.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final

import json

from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from treekle import json_loads, to_plain

import pytest
import torch

from priml.cost import KERNELS, PHASES, cost
from priml.math.probability import weyl_randint
from priml.model.ae.checkpoint import HubFile, LocalFile
from priml.model.ae.distribution import DiagonalGaussian
from priml.model.ae.invae import INVAE, Decoder, Encoder, remap_hf_state_dict
from priml.model.ae.latent_norm import ScaleLatents
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Generator


_CWD: Final = Path(__file__).resolve().parent

_TESTDATA: Final = _CWD / "testdata"


def tiny() -> INVAE.Config:
    """Return the published layout at golden size: widths 2 and 4, 16px."""
    config = INVAE.Config()
    assert config.encoder is not None
    assert config.decoder is not None
    for half in (config.encoder, config.decoder):
        half.channels_hidden = (2, 4)
        half.attention = (False, True)
        half.channels_latent = 2
        half.num_groups = 2
        half.checkpoint = None
    config.encoder.blocks_per_stage = 1
    config.decoder.blocks_per_stage = 2
    config.image_size = 16
    return config


def _published() -> INVAE.Config:
    return INVAE.Config.from_hf(
        json_loads(
            (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
            dict[str, object],
        ),
    )


def _published_keys() -> dict[str, tuple[int, ...]]:
    """Every tensor in the published safetensors, by name, with its shape."""
    lines = (_TESTDATA / "e2e_invae_hf_keys.txt").read_text().splitlines()
    return {
        name: tuple(int(n) for n in shape.split("x"))
        for name, shape in (line.split() for line in lines)
    }


# The posterior draw reads the global generator, which the harness reseeds before
# building the module; an input drawn from it would shift that draw between mint and
# replay, since replay loads the stored input instead of rebuilding it.
def _image() -> Tensor:
    torch.Generator().manual_seed(0)
    return weyl_randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)


def _round_trip(module: nn.Module, image: Tensor) -> Tensor:
    assert isinstance(module, INVAE)
    latent = module.encode(image).sample()
    return torch.cat([latent.flatten(), module.decode(latent).flatten()])


def test_invae_round_trip_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="invae",
        build_module=lambda: tiny().make(),
        build_input=_image,
        run=_round_trip,
    )


def test_satisfies_the_autoencoder_protocol() -> None:
    model = tiny().make()
    assert isinstance(model.encode(_image()), DiagonalGaussian)


def test_from_hf_parses_the_published_architecture_into_the_defaults() -> None:
    parsed, default = _published(), INVAE.Config()
    assert parsed.latent_shape() == (32, 16, 16)
    assert default.encoder is not None
    assert default.decoder is not None
    default.encoder.checkpoint = default.decoder.checkpoint = None
    assert parsed == default


@pytest.mark.parametrize(
    "field",
    ["act_fn", "mid_block_add_attention", "use_quant_conv", "use_post_quant_conv"],
)
@pytest.mark.parametrize("null", [False, True], ids=["missing", "null"])
def test_from_hf_defaults_missing_or_null_supported_options(
    field: str,
    null: bool,
) -> None:
    config = json_loads(
        (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
        dict[str, object],
    )
    if null:
        config[field] = None
    else:
        del config[field]
    assert INVAE.Config.from_hf(config).finalized() == _published().finalized()


@pytest.mark.parametrize(
    "field",
    ["mid_block_add_attention", "use_quant_conv", "use_post_quant_conv"],
)
def test_from_hf_refuses_each_disabled_architecture_component(field: str) -> None:
    config = json_loads(
        (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
        dict[str, object],
    )
    config[field] = False
    with pytest.raises(ValueError, check=lambda error: error.__class__ is ValueError):
        _ = INVAE.Config.from_hf(config)


def test_published_halves_load_exactly_the_published_tensors() -> None:
    """Remapped, the diffusers names and shapes are each half's state, strictly."""
    published = _published_keys()
    with torch.device("meta"):
        model = _published().make()
        state = {name: torch.empty(shape) for name, shape in published.items()}
    for half, module in (("encoder", model.encoder), ("decoder", model.decoder)):
        assert module is not None
        expected = {k: tuple(v.shape) for k, v in module.state_dict().items()}
        remapped = remap_hf_state_dict(state, half=half)
        assert {k: tuple(v.shape) for k, v in remapped.items()} == expected
    assert len(published) == 342


def test_the_published_checkpoint_is_pinned_on_both_halves() -> None:
    config = INVAE.Config()
    assert config.encoder is not None
    assert config.decoder is not None
    assert isinstance(config.encoder.checkpoint, HubFile.Config)
    assert config.encoder.checkpoint == config.decoder.checkpoint


def test_each_half_loads_from_a_file_holding_only_its_own_tensors(
    tmp_path: Path,
) -> None:
    """A half reads its own prefixes: the other half's absence never matters."""
    source = tiny().make()
    assert source.encoder is not None
    assert source.decoder is not None
    halves = {
        "encoder": _to_diffusers(source.encoder.state_dict(), half="encoder"),
        "decoder": _to_diffusers(source.decoder.state_dict(), half="decoder"),
    }
    config = tiny()
    assert config.encoder is not None
    assert config.decoder is not None
    for half, state in halves.items():
        save_file(state, str(tmp_path / f"{half}.safetensors"))
    config.encoder.checkpoint = LocalFile.Config(path=tmp_path / "encoder.safetensors")
    config.decoder.checkpoint = LocalFile.Config(path=tmp_path / "decoder.safetensors")
    loaded = config.make()
    image = _image()
    latent = source.encode(image).mode()
    assert torch.equal(loaded.encode(image).mode(), latent)
    assert torch.equal(loaded.decode(latent), source.decode(latent))


@pytest.mark.parametrize(
    ("half", "prefixes"),
    [
        ("encoder", ("encoder.", "quant_conv.")),
        ("decoder", ("decoder.", "post_quant_conv.")),
    ],
)
def test_each_half_materializes_only_its_tensors_from_a_combined_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    half: str,
    prefixes: tuple[str, ...],
) -> None:
    source = tiny().make()
    assert source.encoder is not None
    assert source.decoder is not None
    state = {
        **_to_diffusers(source.encoder.state_dict(), half="encoder"),
        **_to_diffusers(source.decoder.state_dict(), half="decoder"),
    }
    path = tmp_path / "combined.safetensors"
    save_file(state, str(path))
    materialized: list[str] = []
    monkeypatch.setattr(
        "priml.model.ae.checkpoint.safetensors",
        SimpleNamespace(
            safe_open=partial(_recorded_safe_open, materialized=materialized),
        ),
    )
    config = tiny()
    selected = config.encoder if half == "encoder" else config.decoder
    expected = source.encoder if half == "encoder" else source.decoder
    assert selected is not None
    selected.checkpoint = LocalFile.Config(path=path)
    loaded = selected.make().state_dict()
    assert loaded.keys() == expected.state_dict().keys()
    for name, tensor in expected.state_dict().items():
        assert torch.equal(loaded[name], tensor), name
    assert sorted(materialized) == sorted(
        name for name in state if name.startswith(prefixes)
    )


@contextmanager
def _recorded_safe_open(
    path: Path,
    *,
    framework: str,
    materialized: list[str],
) -> Generator[SimpleNamespace]:
    with safe_open(path, framework=framework) as stream:
        yield SimpleNamespace(
            keys=stream.keys,
            get_tensor=partial(
                _record_tensor,
                stream=stream,
                materialized=materialized,
            ),
        )


def _record_tensor(name: str, *, stream: safe_open, materialized: list[str]) -> Tensor:
    materialized.append(name)
    return stream.get_tensor(name)


def test_either_half_can_be_left_out() -> None:
    encode_only, decode_only = tiny(), tiny()
    encode_only.decoder = None
    decode_only.encoder = None
    encoder = encode_only.make()
    decoder = decode_only.make()
    assert encoder.decoder is None
    assert decoder.encoder is None
    with pytest.raises(ValueError, match="without a decoder"):
        _ = encoder.decode(torch.zeros(3, 2, 8, 6))
    with pytest.raises(ValueError, match="without an encoder"):
        _ = decoder.encode(_image())


def test_leading_dimensions_pass_through() -> None:
    model = tiny().make()
    generator = torch.Generator().manual_seed(1)
    image = torch.randint(
        0,
        256,
        (2, 5, 3, 16, 12),
        generator=generator,
        dtype=torch.uint8,
    )
    flat = model.encode(image.flatten(0, 1)).mode()
    nested = model.encode(image).mode()
    assert nested.shape == (2, 5, *flat.shape[1:])
    assert torch.equal(nested.flatten(0, 1), flat)
    assert model.decode(nested).shape == (2, 5, 3, 16, 12)


def test_five_latent_channels_preserve_posterior_and_decoded_shapes() -> None:
    config = tiny()
    assert config.encoder is not None
    assert config.decoder is not None
    config.encoder.channels_latent = config.decoder.channels_latent = 5
    config.image_size = 6
    model = config.make()
    image = torch.zeros(2, 3, 6, 8, dtype=torch.uint8)
    posterior = model.encode(image)
    expected = (image.shape[0], 5, image.shape[-2] // 2, image.shape[-1] // 2)
    assert posterior.mean.shape == expected
    assert posterior.logvar.shape == expected
    assert model.decode(posterior.mode()).shape == image.shape


def test_mode_is_the_mean_and_a_draw_follows_its_generator() -> None:
    posterior = tiny().make().encode(_image())
    first = posterior.sample(generator=torch.Generator().manual_seed(3))
    second = posterior.sample(generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, second)
    assert not torch.equal(first, posterior.mode())
    assert torch.equal(posterior.mode(), posterior.mean)


def test_decode_returns_the_image_in_the_unit_interval() -> None:
    model = tiny().make()
    image = model.decode(model.encode(_image()).mode())
    assert image.shape == (2, 3, 16, 16)
    assert image.min() >= 0
    assert image.max() <= 1


def test_encode_refuses_a_float_image() -> None:
    with pytest.raises(TypeError, match="uint8"):
        _ = tiny().make().encode(torch.rand(2, 3, 16, 12))


def test_train_leaves_the_frozen_model_in_eval_mode() -> None:
    model = tiny().make()
    _ = model.train()
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_latent_norm_is_the_published_scale() -> None:
    norm = INVAE.Config().latent_norm
    assert isinstance(norm, ScaleLatents.Config)
    assert norm.scale == 0.3099


def test_rejects_an_image_size_the_stages_cannot_halve() -> None:
    config = tiny()
    config.image_size = 15
    with pytest.raises(ValueError, match="divisible by 2"):
        _ = config.make()


def test_rejects_halves_that_disagree_on_the_latent() -> None:
    config = tiny()
    assert config.decoder is not None
    config.decoder.channels_latent = 3
    with pytest.raises(ValueError, match="channels_latent"):
        _ = config.make()


def test_rejects_halves_that_disagree_on_the_grid() -> None:
    """A decoder with another stage count would return another image size."""
    config = tiny()
    assert config.decoder is not None
    config.decoder.channels_hidden = (2, 4, 4)
    config.decoder.attention = (False, True, True)
    with pytest.raises(ValueError, match="stages"):
        _ = config.make()


@pytest.mark.parametrize("half", ["encoder", "decoder"])
@pytest.mark.parametrize("attention", [(False,), (False, True, False)])
def test_each_half_refuses_attention_without_one_entry_per_stage(
    half: str,
    attention: tuple[bool, ...],
) -> None:
    config = tiny()
    selected = config.encoder if half == "encoder" else config.decoder
    assert selected is not None
    selected.attention = attention
    with pytest.raises(ValueError, check=lambda error: error.__class__ is ValueError):
        _ = selected.make()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("_class_name", "AutoencoderTiny", "AutoencoderKL"),
        ("act_fn", "gelu", "silu"),
        ("use_quant_conv", False, "use_quant_conv"),
        ("down_block_types", ["CrossAttnDownBlock2D"], "down_block_types"),
    ],
)
def test_from_hf_refuses_an_architecture_it_does_not_build(
    field: str,
    value: object,
    match: str,
) -> None:
    config = json_loads(
        (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
        dict[str, object],
    )
    config[field] = value
    with pytest.raises(ValueError, match=match):
        _ = INVAE.Config.from_hf(config)


def test_load_reads_a_local_diffusers_directory(tmp_path: Path) -> None:
    """``load`` parses ``config.json`` and gives both halves the weights file."""
    source = tiny().make()
    assert source.encoder is not None
    assert source.decoder is not None
    hf = json_loads(
        (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
        dict[str, object],
    )
    hf.update(
        block_out_channels=[2, 4],
        down_block_types=["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
        up_block_types=["AttnUpDecoderBlock2D", "UpDecoderBlock2D"],
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=2,
    )
    _ = (tmp_path / "config.json").write_text(json.dumps(hf))
    state = {
        **_to_diffusers(source.encoder.state_dict(), half="encoder"),
        **_to_diffusers(source.decoder.state_dict(), half="decoder"),
    }
    save_file(state, str(tmp_path / "diffusion_pytorch_model.safetensors"))
    loaded = INVAE.load(tmp_path)
    image = _image()
    assert torch.equal(loaded.encode(image).mode(), source.encode(image).mode())


def test_load_reads_both_halves_from_a_pinned_hub_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tiny().make()
    assert source.encoder is not None
    assert source.decoder is not None
    hf = json_loads(
        (_TESTDATA / "e2e_invae_hf_config.json").read_text(),
        dict[str, object],
    )
    hf.update(
        block_out_channels=[2, 4],
        down_block_types=["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
        up_block_types=["AttnUpDecoderBlock2D", "UpDecoderBlock2D"],
        layers_per_block=1,
        latent_channels=2,
        norm_num_groups=2,
    )
    _ = (tmp_path / "config.json").write_text(json.dumps(to_plain(hf)))
    state = {
        **_to_diffusers(source.encoder.state_dict(), half="encoder"),
        **_to_diffusers(source.decoder.state_dict(), half="decoder"),
    }
    save_file(state, str(tmp_path / "diffusion_pytorch_model.safetensors"))
    calls: list[tuple[str, str, str]] = []

    def download(*, repo_id: str, filename: str, revision: str) -> str:
        calls.append((repo_id, filename, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(
        "priml.model.ae.checkpoint.huggingface_hub",
        SimpleNamespace(hf_hub_download=download),
    )
    revision = "0123456789abcdef0123456789abcdef01234567"
    loaded = INVAE.load("owner/test-repo", revision=revision)
    assert loaded.encoder is not None
    assert loaded.decoder is not None
    for actual, expected in (
        (loaded.encoder, source.encoder),
        (loaded.decoder, source.decoder),
    ):
        assert actual.state_dict().keys() == expected.state_dict().keys()
        for name, tensor in expected.state_dict().items():
            assert torch.equal(actual.state_dict()[name], tensor), name
    assert calls == [
        ("owner/test-repo", "config.json", revision),
        ("owner/test-repo", "diffusion_pytorch_model.safetensors", revision),
        ("owner/test-repo", "diffusion_pytorch_model.safetensors", revision),
    ]


def _to_diffusers(state: dict[str, Tensor], *, half: str) -> dict[str, Tensor]:
    """Invert :func:`remap_hf_state_dict` for one half, as diffusers saves it."""
    out: dict[str, Tensor] = {}
    for key, tensor in state.items():
        if key.startswith(("quant_conv.", "post_quant_conv.")):
            out[key] = tensor.contiguous()
            continue
        name = f"{half}.{key}".replace(".to_out.", ".to_out.0.")
        projection = any(f".to_{p}.weight" in name for p in ("q", "k", "v", "out.0"))
        out[name] = (tensor.flatten(1) if projection else tensor).contiguous()
    return out


def test_cost_matches_torch_for_a_forward_round_trip() -> None:
    analytical = assert_cost_matches_torch(
        tiny(),
        build_input=_image,
        run=_round_trip_decode,
        batch_size=2,
        dtype=None,
    )
    assert analytical["adjoint"].sum() == 0
    assert analytical["flops", "primal", "elementwise"].sum() > 0


def test_cost_prices_the_pixel_maps_at_the_configured_channels() -> None:
    """Grayscale in: the uint8 read and the float maps cover one channel, not three."""
    rgb, gray = tiny(), tiny()
    assert gray.encoder is not None
    gray.encoder.channels_in = 1
    rgb_cost, gray_cost = (
        cost(config, batch_size=2, dtype=None) for config in (rgb, gray)
    )
    pixels = 2 * 16 * 16
    key = ("bytes", "primal", "elementwise", torch.uint8)
    assert rgb_cost[key] - gray_cost[key] == 2 * pixels


def _round_trip_decode(module: nn.Module, image: Tensor) -> Tensor:
    assert isinstance(module, INVAE)
    return module.decode(module.encode(image).sample())


def test_cost_params_are_every_parameter() -> None:
    config = tiny()
    model = config.make()
    analytical = cost(config.copy_tree().finalize(), batch_size=2, dtype=None)
    assert analytical.params == sum(p.numel() for p in model.parameters())


def test_published_cost_is_linear_in_batch_size() -> None:
    """The pinned FLOPs are torch's for one 256px image through the published build."""
    config = INVAE.Config().copy_tree().finalize()
    one = cost(config, batch_size=1, dtype=torch.bfloat16)
    two = cost(config, batch_size=2, dtype=torch.bfloat16)
    assert one["flops", "primal", "matmul", torch.bfloat16] == 390_131_613_696
    assert one.params == 69_836_067
    assert two["flops"] == one["flops"].tile(2)


def test_half_costs_preserve_dtype_in_every_silo() -> None:
    for half in (Encoder.Config(), Decoder.Config()):
        half.channels_hidden = (4, 6)
        half.attention = (False, True)
        half.num_groups = 2
        half.channels_latent = 5
        half.blocks_per_stage = 2
        half.checkpoint = None
        single, double = (
            cost(half, input_grid=6, batch_size=3, dtype=dtype)
            for dtype in (torch.float32, torch.float64)
        )
        assert single.params == double.params > 0
        assert single.params_active == double.params_active == single.params
        assert single.only(torch.float32).cells == single.cells
        assert double.only(torch.float64).cells == double.cells
        for phase in PHASES:
            for kernel in KERNELS:
                assert (
                    single["flops", phase, kernel, torch.float32]
                    == double["flops", phase, kernel, torch.float64]
                )
                assert (
                    2 * single["bytes", phase, kernel, torch.float32]
                    == double["bytes", phase, kernel, torch.float64]
                )


def test_half_cost_group_count_changes_normalization_statistics() -> None:
    config = Encoder.Config()
    config.channels_hidden = (4,)
    config.attention = (False,)
    config.blocks_per_stage = 1
    config.num_groups = 2
    before = cost(config, input_grid=6, batch_size=3, dtype=torch.float32)
    config.num_groups = 4
    after = cost(config, input_grid=6, batch_size=3, dtype=torch.float32)
    groups_added = 8 * 3 * (4 - 2)
    assert (
        after["flops", "primal", "elementwise", torch.float32]
        - before["flops", "primal", "elementwise", torch.float32]
    ) == 4 * groups_added
    assert (
        before["flops", "primal", "reduction", torch.float32]
        - after["flops", "primal", "reduction", torch.float32]
    ) == 2 * groups_added


def test_half_cost_extra_residual_has_binary_operand_traffic() -> None:
    config = Encoder.Config()
    config.channels_hidden = (4,)
    config.attention = (False,)
    config.num_groups = 2
    elements, groups, channels = 3 * 4 * 6 * 6, 3 * 2, 4
    for dtype in (torch.float32, torch.float64):
        config.blocks_per_stage = 1
        before = cost(config, input_grid=6, batch_size=3, dtype=dtype)
        config.blocks_per_stage = 2
        after = cost(config, input_grid=6, batch_size=3, dtype=dtype)
        assert (
            after["flops", "primal", "elementwise", dtype]
            - before["flops", "primal", "elementwise", dtype]
        ) == 23 * elements + 8 * groups
        assert (
            after["bytes", "primal", "elementwise", dtype]
            - before["bytes", "primal", "elementwise", dtype]
        ) == (33 * elements + 6 * channels + 20 * groups) * dtype.itemsize


def test_half_cost_attention_has_binary_operand_and_softmax_traffic() -> None:
    config = Encoder.Config()
    config.channels_hidden = (4,)
    config.blocks_per_stage = 1
    config.num_groups = 2
    batch, channels, positions, groups = 3, 4, 6 * 6, 3 * 2
    elements = batch * channels * positions
    for dtype in (torch.float32, torch.float64):
        config.attention = (False,)
        before = cost(config, input_grid=6, batch_size=batch, dtype=dtype)
        config.attention = (True,)
        after = cost(config, input_grid=6, batch_size=batch, dtype=dtype)
        assert (
            after["flops", "primal", "elementwise", dtype]
            - before["flops", "primal", "elementwise", dtype]
        ) == 10 * elements + 4 * groups + 4 * batch * positions * positions
        assert (
            after["bytes", "primal", "elementwise", dtype]
            - before["bytes", "primal", "elementwise", dtype]
        ) == (
            13 * elements
            + 6 * channels
            + 10 * groups
            + batch * positions * (8 * positions + 2)
        ) * dtype.itemsize


def test_two_stage_half_costs_include_padding_and_interpolation_traffic() -> None:
    batch, channels, coarse, fine, latent, output = 2, 4, 3, 6, 5, 3
    groups = batch * 2
    small, large = batch * channels * coarse**2, batch * channels * fine**2
    residual_small = 33 * small + 6 * channels + 20 * groups
    residual_large = 33 * large + 6 * channels + 20 * groups
    attention_small = (
        13 * small
        + 6 * channels
        + 10 * groups
        + batch * coarse**2 * (8 * coarse**2 + 2)
    )
    for half in (Encoder.Config(), Decoder.Config()):
        # Equal stage widths are supported by Encoder and Decoder constructors.
        half.channels_hidden = (channels, channels)
        half.attention = (False, False)
        half.blocks_per_stage = 1
        half.num_groups = 2
        half.channels_latent = latent
        if isinstance(half, Encoder.Config):
            side = fine
            expected = (
                2 * channels
                + residual_large
                + 3 * residual_small
                + attention_small
                + 10 * small
                + 2 * channels
                + 10 * groups
                + 5 * small
                + 4 * latent
                + batch * channels * (fine**2 + (fine + 1) ** 2)
            )
        else:
            side = coarse
            expected = (
                latent
                + 2 * channels
                + output
                + 3 * residual_small
                + attention_small
                + 5 * small
                + residual_large
                + 10 * large
                + 2 * channels
                + 10 * groups
                + 5 * large
            )
        for dtype in (torch.float32, torch.float64):
            analytical = cost(half, input_grid=side, batch_size=batch, dtype=dtype)
            assert analytical["bytes", "primal", "elementwise", dtype] == (
                expected * dtype.itemsize
            )


def test_round_trip_cost_adds_only_pixel_and_posterior_maps() -> None:
    config = tiny()
    assert config.encoder is not None
    assert config.decoder is not None
    config.image_size = 6
    config.encoder.channels_in = 4
    config.decoder.channels_out = 5
    config.encoder.channels_latent = config.decoder.channels_latent = 7
    batch = 2
    pixels_in = batch * 4 * 6 * 6
    pixels_out = batch * 5 * 6 * 6
    latents = batch * 7 * 3 * 3
    full = cost(config, batch_size=batch, dtype=torch.float32)
    halves = cost(
        config.encoder,
        input_grid=6,
        batch_size=batch,
        dtype=torch.float32,
    ) + cost(
        config.decoder,
        input_grid=3,
        batch_size=batch,
        dtype=torch.float32,
    )
    expected: dict[tuple[object, ...], int] = {
        ("flops", "primal", "elementwise", torch.float32): (
            2 * pixels_in + 6 * latents + 4 * pixels_out
        ),
        ("bytes", "primal", "elementwise", torch.float32): (
            4 * (5 * pixels_in + 13 * latents + 6 * pixels_out)
        ),
        ("bytes", "primal", "elementwise", torch.uint8): pixels_in,
    }
    primal = halves.only("primal")
    for key in full.cells.keys() | primal.cells.keys() | expected.keys():
        assert full.cells.get(key, 0) - primal.cells.get(key, 0) == expected.get(key, 0)
    assert full.params == halves.params
    assert full.params_active == halves.params_active
    assert full["adjoint"].sum() == 0


def test_invae_config_pprint() -> None:
    assert_pprint_golden(test_file=__file__, name="invae_config", config=INVAE.Config())


def test_config_needs_a_half_for_its_latent_shape() -> None:
    config = tiny()
    config.encoder = config.decoder = None
    with pytest.raises(ValueError, match="needs an encoder or a decoder"):
        _ = config.latent_shape()
    with pytest.raises(ValueError, match="needs an encoder or a decoder"):
        _ = config.make()


def test_cost_needs_both_halves() -> None:
    config = tiny()
    config.decoder = None
    with pytest.raises(ValueError, match="prices both halves"):
        _ = config.cost(batch_size=1, dtype=None)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
