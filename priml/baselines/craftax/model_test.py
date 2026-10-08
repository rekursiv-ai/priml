"""Unit tests for the MinGRU policy at tiny sizes on the CPU.

These pin the config propagation, the module contracts, the init
distributions and the gradients' agreement with float64 autograd through torch
ops alone; the multi-hot embedding's and the scan's own tests are priml's. The
goldens freeze the forward and backward bits at test size: on any CPU, and for
exp000's policy, its exact scan, on the GPU.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    cast,
)

from torch import Tensor, nn
from torch.nn import functional

import numpy as np
import pytest
import torch

from priml.baselines.craftax.game.mobs import FLOOR_MOB_TYPES
from priml.baselines.craftax.game.state import (
    NUM_BLOCK_TYPES,
    NUM_ITEM_TYPES,
    NUM_MOB_TYPES,
    OBS_SIZE,
    SYMBOLIC_OBS_SIZE,
    ProjectileType,
)
from priml.baselines.craftax.lib.arrays import ints
from priml.baselines.craftax.model import (
    DenseObservation,
    FeasibilityLoss,
    MinGRUPolicy,
    NoEncoder,
    packed_observation_embedding,
)
from priml.baselines.craftax.policies.encoder import MLP, BoardEncoder
from priml.baselines.craftax.testing import (
    assert_golden,
    digest,
    fill_portable,
    gpu_key,
    multi_hot_embedding,
    packed_observations,
    portable_uniform,
    require_golden,
    sole_feature_policy,
    tiny_board_policy,
    tiny_exp000_step,
    tiny_policy,
)
from priml.lib.codec import from_plain
from priml.model.embedding import MultiHotEmbedding
from priml.model.linear import Linear
from priml.testing.bfb import assert_bfb_against_golden, host_agnostic_numerics
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


if TYPE_CHECKING:
    from collections.abc import Callable


_CWD: Final = Path(__file__).resolve().parent


def test_the_default_config_is_pufferlibs_geometry() -> None:
    config = MinGRUPolicy.Config().copy_tree().finalize()
    assert multi_hot_embedding(config).observation_size == 843
    assert multi_hot_embedding(config).channels_concat == 1635
    assert config.proj_in.channels_in == 1635
    assert config.proj_in.channels_out == 1024
    assert config.block.proj_gates.channels_in == 1024
    assert config.block.proj_gates.channels_out == 3072
    assert config.proj_out.channels_in == 1024
    assert config.proj_out.channels_out == 44
    assert not config.proj_in.bias
    assert not config.proj_out.bias
    assert not config.block.proj_gates.bias


def test_the_forward_contract() -> None:
    config = tiny_policy()
    model = config.make()
    observations = packed_observations(config, batch=3)
    state = model.initial_state(3)
    assert state.shape == (2, 3, 8)
    assert state.dtype == torch.bfloat16
    logits, values, state = model(observations, state, torch.zeros(3, dtype=torch.bool))
    assert logits.shape == (3, 43)
    assert values.shape == (3,)
    assert state.shape == (2, 3, 8)
    assert logits.dtype == values.dtype == state.dtype == torch.bfloat16


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_a_sequence_is_the_step_applied_repeatedly(state_dtype: torch.dtype) -> None:
    config = tiny_policy()
    config.state_dtype = state_dtype
    model = config.make()
    observations = packed_observations(config, batch=3, time=4)
    starts = torch.zeros(3, 4, dtype=torch.bool)
    starts[1, 2] = True
    state = model.initial_state(3)
    decoded, final, _ = model.forward_sequence(observations, state, starts)
    for time in range(4):
        logits, values, state = model(observations[:, time], state, starts[:, time])
        assert torch.equal(logits, decoded[:, time, :-1])
        assert torch.equal(values, decoded[:, time, -1])
    assert torch.equal(state, final)
    assert final.dtype == state_dtype


def test_the_default_carry_is_fp32_over_bf16_weights_and_stays_unrounded() -> None:
    """The class default keeps the carry in fp32; a bf16 carry is its rounding."""
    assert MinGRUPolicy.Config().state_dtype == torch.float32
    config = tiny_policy()
    config.state_dtype = torch.float32
    model = config.make()
    assert all(weight.dtype == torch.bfloat16 for weight in model.parameters())
    observations = packed_observations(config, batch=3)
    state = model.initial_state(3)
    assert state.dtype == torch.float32
    _, state = model.forward_fused(observations, state, None)
    rounded = tiny_policy().make()
    rounded.load_state_dict(model.state_dict())
    _, bf16 = rounded.forward_fused(observations, rounded.initial_state(3), None)
    # From a zero carry the first step is the same fp32 arithmetic, rounded once.
    assert torch.equal(bf16, state.bfloat16())
    assert not torch.equal(state, bf16.float())


def test_the_default_decoder_sums_in_fp32_and_differentiates_in_bf16() -> None:
    """The class default keeps the decoder's fp32 sums; their gradient comes back in bf16.

    Its rows are the float64 products of the bf16 features and weights to 1e-6,
    2,000 times finer than a bf16 value's rounding near 0.5. The gradient is
    rounded once to bf16 on the way back, so the weight's is the float64
    product of that rounded gradient, rounded to its bf16.
    """
    assert MinGRUPolicy.Config().output_dtype == torch.float32
    config = tiny_policy()
    config.output_dtype = torch.float32
    _assert_the_decoder_is_wide(config.make(), device=torch.device("cpu"))


@pytest.mark.gpu_triton
def test_the_default_decoder_sums_in_fp32_and_differentiates_in_bf16_on_the_gpu() -> (
    None
):
    """The same on the GPU, where one cuBLAS GEMM writes its fp32 sums out."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    config = tiny_policy()
    config.output_dtype = torch.float32
    _assert_the_decoder_is_wide(config.make().cuda(), device=torch.device("cuda"))


def test_parameters_are_every_weight_in_pufferlibs_allocator_order() -> None:
    # FusedMuon's global norm reduces the gradients in this order, so exp000's
    # bits depend on it: proj_out comes before the blocks.
    model = MinGRUPolicy.Config().make()
    names = [name for name, _ in model.named_parameters()]
    assert names == [
        "embedding.weight",
        "proj_in.weight",
        "proj_out.weight",
        *(f"blocks.{index}.proj_gates.weight" for index in range(4)),
    ]
    shapes = [tuple(parameter.shape) for parameter in model.parameters()]
    assert shapes == [(154, 16), (1024, 1635), (44, 1024), *[(3072, 1024)] * 4]
    assert sum(parameter.numel() for parameter in model.parameters()) == 14_304_672
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())


def test_every_weight_is_pufferlibs_draw_rounded_once_to_bf16() -> None:
    """Each layer holds PufferLib's distribution, drawn in fp32 and rounded to bf16.

    PufferLib draws the table from N(0, 1) and every projection from
    U(+-1/sqrt(fan_in)), in fp32, then casts them to bf16. Each distribution
    depends only on its weight's fan-in, so a one-layer trunk of 64 suffices;
    the table is widened to 512 so that its draws resolve the values near its
    mode, where a draw made in bf16 shows (as does every projection's).
    """
    config = MinGRUPolicy.Config()
    multi_hot_embedding(config).channels_out = 512
    config.channels_hidden = 64
    config.num_layers = 1
    torch.manual_seed(0)
    model = config.make()
    for name, weight in model.named_parameters():
        assert weight.dtype == torch.bfloat16, name
        assert _fit_to_rounded_draw(weight, _pufferlib_cdf(name, weight)) > 1e-6, name


@pytest.mark.compute_large_fixture
def test_exp000s_init_draws_pufferlibs_distributions() -> None:
    """The same fit at exp000's geometry: 14.3M weights, the table at its 2,464."""
    torch.manual_seed(0)
    model = MinGRUPolicy.Config().make()
    for name, weight in model.named_parameters():
        assert _fit_to_rounded_draw(weight, _pufferlib_cdf(name, weight)) > 1e-6, name


def test_pufferlibs_own_seed_73_init_fits_the_reference() -> None:
    """PufferLib's own weights pass the fit the port's are held to.

    So the reference -- each distribution drawn in fp32 and rounded once --
    is PufferLib's algorithm, measured on its output rather than read off
    its source. The draws are PufferLib's seed-73 init (the oracle's
    ``trainer/init/seed73.bin``), one pool per distribution it draws from,
    in bf16, which holds them exactly: the table whole, N(0, 1); three rows
    of ``proj_in``, U(+-1635^-1/2); six rows across ``proj_out`` and the four
    blocks, U(+-1024^-1/2). Pooled, a draw made in bf16 instead fails the
    two uniforms at p = 0 (measured); the table's 2,464 are all it has.
    """
    pools = from_plain(
        cast(
            "object",
            torch.load(
                _CWD / "testdata" / "pufferlib_init_seed73_pools.pt",
                weights_only=True,
            ),
        ),
        dict[str, torch.Tensor],
    )
    assert {name: tuple(pool.shape) for name, pool in pools.items()} == {
        "embedding.weight": (154, 16),
        "proj_in.weight": (3, 1635),
        "fan_in_1024": (6, 1024),
    }
    for name, pool in pools.items():
        weight = pool.float()
        assert _fit_to_rounded_draw(weight, _pufferlib_cdf(name, weight)) > 1e-6, name


def _pufferlib_cdf(name: str, weight: Tensor) -> Callable[[Tensor], Tensor]:
    """Return the CDF PufferLib draws a weight from, by the weight's name."""
    # ``puf_normal_init`` draws the table from N(0, 1) and ``puf_kaiming_init``
    # every projection from U(+-gain/sqrt(fan_in)), gain 1, the fan-in the
    # weight's last axis (``ocean/craftax/craftax.cu``, ``src/algo.cu``).
    if name == "embedding.weight":
        return torch.special.ndtr
    bound = weight.shape[-1] ** -0.5
    return lambda x: ((x / bound + 1) / 2).clamp(0, 1)


def _fit_to_rounded_draw(weight: Tensor, cdf: Callable[[Tensor], Tensor]) -> float:
    """Return the chi-square p-value of ``weight`` as draws from ``cdf`` rounded to bf16."""
    # Each bf16 value is a cell holding the mass ``cdf`` puts between the
    # midpoints to its neighbours, where a draw rounds to it. Near the mode
    # the cells stay one value wide, which is what catches a draw made in
    # bf16: torch's CPU generator fills a bf16 uniform from 8 random bits (256
    # values, so some cells stay empty) and runs the normal's Box-Muller in
    # bf16.
    patterns = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    values = patterns.view(torch.bfloat16).double()
    # Ascending with -0 before +0, where a negative draw rounds; the stable
    # sort keeps the patterns' order, which has -0 first.
    finite = values.isfinite().nonzero().squeeze(-1)
    order = finite[torch.argsort(values[finite], stable=True)]
    atoms = values[order]
    edges = cdf((atoms[1:] + atoms[:-1]) / 2)
    mass = torch.cat((edges.new_zeros(1), edges, edges.new_ones(1))).diff()
    bits = weight.detach().bfloat16().flatten().view(torch.int16).int() + 2**15
    counts = torch.bincount(bits, minlength=2**16)[order].double()
    draws = counts.sum()
    # Runs of cells pool until the fullest expects five draws; every cell
    # that still expects fewer then pools into one.
    size = int(torch.ceil(5 / (draws * mass.max())))
    pad = -len(mass) % size
    expected = draws * functional.pad(mass, (0, pad)).reshape(-1, size).sum(-1)
    observed = functional.pad(counts, (0, pad)).reshape(-1, size).sum(-1)
    kept = expected >= 5
    expected = torch.cat((expected[kept], expected[~kept].sum(0, keepdim=True)))
    observed = torch.cat((observed[kept], observed[~kept].sum(0, keepdim=True)))
    statistic = ((observed - expected) ** 2 / expected).sum()
    # The chi-square survival function, Q(k / 2, statistic / 2) for k degrees
    # of freedom.
    half_freedom = torch.tensor((len(expected) - 1) / 2, dtype=torch.float64)
    return float(torch.special.gammaincc(half_freedom, statistic / 2))


# The table's lookup and the scan run their torch references directly, not as the
# autograd nodes whose backwards are their own kernels.
# A policy without ``proj_in`` reads ``features``' projection alone.
def _plain_autograd_decoded(
    model: MinGRUPolicy,
    observations: Tensor,
    starts: Tensor,
    features: Tensor | None = None,
) -> Tensor:
    """Run the window forward through torch ops alone, each differentiated by autograd."""
    batch, time = starts.shape
    if model.proj_in is None:
        assert model.proj_feature is not None
        assert features is not None
        value = model.proj_feature(features.flatten(0, 1)).reshape(batch, time, -1)
    else:
        embedding = model.embedding
        embedded = (
            embedding.forward_torch(observations)
            if isinstance(embedding, MultiHotEmbedding)
            else observations
        )
        value = model.proj_in(embedded.flatten(0, 1)).reshape(batch, time, -1)
    state = model.initial_state(batch)
    for index, block in enumerate(model.blocks):
        combined = block.proj_gates(value)
        value = block.scan(combined, value, state[index], starts).outputs
    return model.proj_out(value.flatten(0, 1)).reshape(batch, time, -1)


def _assert_gradients_agree_with_float64_autograd(
    config: MinGRUPolicy.Config,
    observations: Tensor,
    features: Tensor | None = None,
) -> None:
    """Check the fp32 policy's gradients against float64 autograd, same weights."""
    model = config.make()
    starts = torch.zeros(2, 3, dtype=torch.bool)
    starts[0, 1] = True
    generator = torch.Generator().manual_seed(3)
    grad = torch.randn(2, 3, 44, generator=generator)
    decoded, _, _ = model.forward_sequence(
        observations,
        model.initial_state(2),
        starts,
        features=features,
    )
    decoded.backward(grad)

    # A float64 policy and carry, so its observations round to float64 too.
    double = config.copy_tree()
    double.dtype = double.state_dtype = torch.float64
    reference = double.make()
    reference.load_state_dict(model.state_dict())
    _plain_autograd_decoded(
        reference,
        observations.double(),
        starts,
        None if features is None else features.double(),
    ).backward(grad.double())
    for (name, ours), expected in zip(
        model.named_parameters(),
        reference.parameters(),
        strict=True,
    ):
        assert ours.grad is not None, name
        assert expected.grad is not None, name
        torch.testing.assert_close(
            ours.grad.double(),
            expected.grad,
            rtol=1e-4,
            atol=1e-5,
        )


def test_gradients_agree_with_float64_autograd_in_fp32() -> None:
    config = tiny_policy(dtype=torch.float32)
    _assert_gradients_agree_with_float64_autograd(
        config,
        packed_observations(config, batch=2, time=3),
    )


def test_a_dense_observations_gradients_agree_with_float64_autograd_in_fp32() -> None:
    config = tiny_policy(dtype=torch.float32)
    config.embedding = DenseObservation.Config()
    generator = torch.Generator().manual_seed(4)
    _assert_gradients_agree_with_float64_autograd(
        config,
        torch.rand(2, 3, SYMBOLIC_OBS_SIZE, generator=generator),
    )


def test_the_embedding_is_an_injectable_slot() -> None:
    config = tiny_policy()
    embedding = config.embedding = packed_observation_embedding()
    embedding.channels_out = 3
    finalized = config.copy_tree().finalize()
    assert finalized.proj_in.channels_in == 99 * 3 + 51


def test_a_dense_observation_is_the_features() -> None:
    config = tiny_policy()
    config.embedding = DenseObservation.Config()
    model = config.make()
    shapes = [tuple(parameter.shape) for parameter in model.parameters()]
    assert shapes == [(8, SYMBOLIC_OBS_SIZE), (44, 8), (24, 8), (24, 8)]
    observations = torch.rand(3, 2, SYMBOLIC_OBS_SIZE).bfloat16()
    assert model.embedding(observations) is observations
    starts = torch.zeros(3, 2, dtype=torch.bool)
    decoded, _, _ = model.forward_sequence(observations, model.initial_state(3), starts)
    assert decoded.shape == (3, 2, 44)


def test_every_id_the_game_writes_is_inside_its_fields_rows() -> None:
    """The packed observation's ids never reach past their field's vocabulary.

    ``compute_observations_numba`` writes the block id, ``item + 1``, a 0/1 light
    flag and ``species + 1`` for each creature class
    (``game/observation.py:73-75, 381-386``); species come from
    ``FLOOR_MOB_TYPES`` and ``ProjectileType``. So every row the embedding
    reads or its backward adds to is its own field's, on every path.
    """
    embedding = packed_observation_embedding()
    widths = np.diff([*embedding.offsets, embedding.channels_in])
    assert widths[0] >= NUM_BLOCK_TYPES
    assert widths[1] >= NUM_ITEM_TYPES + 1
    assert widths[2] >= 2
    levels, classes = FLOOR_MOB_TYPES.shape
    species = [
        FLOOR_MOB_TYPES[level, c] for level in range(levels) for c in range(classes)
    ]
    assert min(species) >= 0
    assert max(*species, *ProjectileType) < NUM_MOB_TYPES
    assert all(width >= NUM_MOB_TYPES + 1 for width in ints(widths[3:]))


def test_the_dense_stage_defaults_to_the_symbolic_width() -> None:
    assert DenseObservation.Config().observation_size == SYMBOLIC_OBS_SIZE


def test_the_tiny_policy_matches_its_golden() -> None:
    """The bf16 policy at test size, frozen bit for bit on every host.

    Weights and inputs are portable draws and the arithmetic runs inside
    ``host_agnostic_numerics``, so every CPU computes the same bits. The
    golden holds the rollout's step four times (the decoded logits and value,
    and the carry advanced in place through resets), the learner's window
    forward, and autograd's backward from fixed decoder gradients.
    """
    config = tiny_policy()
    model = config.make()
    fill_portable(model, seed=0)
    observations = packed_observations(config, batch=3, time=4, seed=1)
    starts, grad_logits, grad_values = _backward_inputs(3, 4, scale=2**-6)
    with host_agnostic_numerics():
        lines = _policy_entries(
            model,
            observations,
            starts,
            grad_logits,
            grad_values,
            steps=4,
        )
    assert_golden(test_file=__file__, name="model_tiny", lines=lines)


@pytest.mark.gpu_triton
def test_exp000s_policy_matches_its_golden_on_the_gpu() -> None:
    """exp000's policy at test size -- bf16 with a bf16 carry, the exact scan -- frozen.

    From portable seed-73 weights, three rows of four steps: the rollout's
    step over all four, the window forward, and the backward from fixed
    gradients.
    """
    host = gpu_key()
    require_golden(test_file=__file__, name="model_exp000_tiny", host=host)
    config = tiny_exp000_step().model
    assert isinstance(config, MinGRUPolicy.Config)
    model = config.make()
    fill_portable(model, seed=73)
    model.cuda()
    observations = packed_observations(config, batch=3, time=4, seed=1)
    starts, grad_logits, grad_values = _backward_inputs(3, 4, scale=2**-6)
    lines = _policy_entries(
        model,
        observations.cuda(),
        starts.cuda(),
        grad_logits.cuda(),
        grad_values.cuda(),
        steps=4,
    )
    assert_golden(
        test_file=__file__,
        name="model_exp000_tiny",
        lines=lines,
        host=host,
    )


def test_without_the_new_slots_the_policy_builds_what_it_built() -> None:
    """Both slots default to None, which registers no module: exp000's tree exactly."""
    config = MinGRUPolicy.Config()
    assert config.injection is None
    assert config.auxiliary is None
    model = tiny_policy().make()
    assert model.injections is None
    assert model.auxiliary is None
    assert [name for name, _ in model.named_modules()] == [
        "",
        "embedding",
        "proj_in",
        "proj_out",
        "blocks",
        "blocks.0",
        "blocks.0.proj_gates",
        "blocks.1",
        "blocks.1.proj_gates",
    ]


def test_the_board_policy_holds_the_references_live_weights() -> None:
    """exp102's model: the reference's weights less D1's dead ones, new ones last."""
    config = MinGRUPolicy.Config()
    config.embedding = BoardEncoder.Config()
    injection = config.injection = MLP.Config()
    injection.channels_hidden = 128
    config.auxiliary = FeasibilityLoss.Config()
    model = config.make()
    encoder = [
        f"embedding.{name}"
        for name, _ in BoardEncoder.Config().make().named_parameters()
    ]
    shapes = [(name, tuple(weight.shape)) for name, weight in model.named_parameters()]
    assert [name for name, _ in shapes] == [
        *encoder,
        "proj_in.weight",
        "proj_out.weight",
        *(f"blocks.{index}.proj_gates.weight" for index in range(4)),
        *(
            f"injections.{index}.proj_{end}.weight"
            for index in range(4)
            for end in ("in", "out")
        ),
        "auxiliary.proj_out.weight",
    ]
    assert shapes[-3:] == [
        ("injections.3.proj_in.weight", (128, 1024)),
        ("injections.3.proj_out.weight", (1024, 128)),
        ("auxiliary.proj_out.weight", (43, 1024)),
    ]
    # 15,512,256 in the reference, less action_out (32,768) and the action
    # columns of the FiLM projection (256 x 32).
    assert sum(weight.numel() for weight in model.parameters()) == 15_471_296


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_a_board_policy_sequence_is_the_step_applied_repeatedly(
    state_dtype: torch.dtype,
) -> None:
    """The encoder and the injections run alike in the actor's step and the learner's window."""
    config = tiny_board_policy()
    config.state_dtype = state_dtype
    model = config.make()
    fill_portable(model, seed=4)
    observations = packed_observations(config, batch=3, time=4, seed=5)
    starts = torch.zeros(3, 4, dtype=torch.bool)
    starts[1, 2] = True
    state = model.initial_state(3)
    decoded, final, _ = model.forward_sequence(observations, state, starts)
    for time in range(4):
        step, state = model.forward_fused(observations[:, time], state, starts[:, time])
        assert torch.equal(step, decoded[:, time])
    assert torch.equal(state, final)


def test_a_zero_injection_leaves_the_trunk_and_a_filled_one_changes_it() -> None:
    config = tiny_board_policy()
    model = config.make()
    fill_portable(model, seed=6)
    plain_config = tiny_board_policy()
    plain_config.injection = None
    plain = plain_config.make()
    plain.load_state_dict(
        {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith("injections.")
        },
    )
    observations = packed_observations(config, batch=3, time=2, seed=7)
    starts = torch.zeros(3, 2, dtype=torch.bool)
    assert not _same_outputs(model, plain, observations, starts)
    assert model.injections is not None
    with torch.no_grad():
        for injection in model.injections:
            assert isinstance(injection, MLP)
            injection.proj_out.weight.zero_()
    assert _same_outputs(model, plain, observations, starts)


def test_each_block_builds_its_own_injection_of_the_trunks_width() -> None:
    model = tiny_board_policy().make()
    assert model.injections is not None
    assert len(model.injections) == 2
    shapes = [tuple(weight.shape) for weight in model.injections.parameters()]
    assert shapes == [(3, 8), (8, 3)] * 2
    assert len({id(weight) for weight in model.injections.parameters()}) == 4


def test_only_the_learners_window_with_actions_scores_the_auxiliary_loss() -> None:
    config = tiny_board_policy()
    model = config.make()
    auxiliary = model.auxiliary
    assert isinstance(auxiliary, FeasibilityLoss)
    calls: list[int] = []
    auxiliary.register_forward_hook(partial(_record_call, calls))
    observations = packed_observations(config, batch=3, time=2, seed=8)
    starts = torch.zeros(3, 2, dtype=torch.bool)
    actions = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    state = model.initial_state(3)
    model.forward_fused(observations[:, 0], state, starts[:, 0])
    model(observations[:, 0], state, starts[:, 0])
    _, _, skipped = model.forward_sequence(observations, state, starts)
    assert not calls
    assert skipped.shape == ()
    assert skipped.dtype == torch.float32
    assert float(skipped) == 0
    _, _, scored = model.forward_sequence(observations, state, starts, actions=actions)
    assert calls == [1]
    assert float(scored.detach()) > 0
    plain = tiny_policy().make()
    plain_observations = packed_observations(tiny_policy(), batch=3, time=2)
    _, _, none = plain.forward_sequence(
        plain_observations,
        plain.initial_state(3),
        starts,
        actions=actions,
    )
    assert float(none) == 0


def test_the_feasibility_target_is_a_change_in_the_compared_fields() -> None:
    """Three compared fields, then a previous action that changes every step."""
    loss = _feasibility_loss()
    observations = torch.zeros(2, 5, 4)
    observations[..., 3] = torch.arange(5.0)
    observations[0, 1, 0] = 1
    observations[0, 4, 2] = 2
    observations[1, 2:, 1] = 7
    starts = torch.zeros(2, 5)
    starts[0, 3] = starts[1, 0] = 1
    changed, valid = loss.targets(observations, starts)
    assert changed.tolist() == [
        [True, True, False, True, False],
        [False, True, False, False, False],
    ]
    assert valid.tolist() == [
        [True, True, False, True, False],
        [True, True, True, True, False],
    ]


def test_the_feasibility_loss_is_the_mean_bce_of_the_taken_actions_logits() -> None:
    loss = _feasibility_loss()
    generator = torch.Generator().manual_seed(9)
    features = torch.randn(2, 5, 6, generator=generator)
    observations = torch.randint(3, (2, 5, 4), generator=generator).float()
    starts = torch.zeros(2, 5)
    starts[1, 2] = 1
    actions = torch.randint(7, (2, 5), generator=generator).float()
    value = loss(features, observations, starts, actions)
    changed, valid = loss.targets(observations, starts)
    weight = loss.proj_out.weight.detach().double()
    logits = (features.double() @ weight.T).gather(-1, actions.long()[..., None])[
        ...,
        0,
    ]
    target, counted = changed.double(), valid.double()
    bce = logits.clamp(min=0) - logits * target + torch.log1p(torch.exp(-logits.abs()))
    expected = 0.25 * (bce * counted).sum() / counted.sum()
    torch.testing.assert_close(value.double(), expected, rtol=1e-6, atol=0)
    value.backward()
    grad = loss.proj_out.weight.grad
    assert grad is not None
    taken = set(actions[valid].long().tolist())
    for action in range(7):
        assert bool(grad[action].any()) == (action in taken), action


def test_a_window_of_one_step_scores_nothing() -> None:
    loss = _feasibility_loss()
    # A window of one step: the loss pairs step t with t + 1, so it has no pair.
    features = torch.randn(2, 1, 6, requires_grad=True)
    value = loss(features, torch.zeros(2, 1, 4), torch.zeros(2, 1), torch.zeros(2, 1))
    assert float(value.detach()) == 0
    value.backward()
    assert features.grad is not None
    assert not features.grad.any()


def test_the_tiny_board_policy_config_matches_its_golden() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="board_policy_tiny",
        config=_tiny_board_policy_at_defaults(),
    )


def test_the_tiny_board_policy_matches_its_bfb_golden() -> None:
    """The window, its loss, the trunk's gradients and an actor step, on every host."""
    config = _tiny_board_policy_at_defaults()
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="board_policy_tiny",
        build_module=config.make,
        build_input=lambda: _board_policy_inputs(config),
        run=_score_board_policy,
    )


def _record_call(
    calls: list[int],
    module: nn.Module,
    args: tuple[object, ...],
    output: object,
) -> None:
    """Count one forward of a module, as a forward hook."""
    del module, args, output
    calls.append(1)


def _feasibility_loss() -> FeasibilityLoss:
    """Return a loss over 3 compared fields, a head of 6 -> 7, weighing 0.25."""
    config = FeasibilityLoss.Config()
    config.channels_in = 6
    config.channels_out = 7
    config.observation_size = 3
    config.coefficient = 0.25
    return config.make()


def _same_outputs(
    model: MinGRUPolicy,
    other: MinGRUPolicy,
    observations: Tensor,
    starts: Tensor,
) -> bool:
    """Return whether two policies' windows and first steps are equal, bit for bit."""
    state = model.initial_state(starts.shape[0])
    window, _, _ = model.forward_sequence(observations, state, starts)
    other_window, _, _ = other.forward_sequence(observations, state, starts)
    step, _ = model.forward_fused(observations[:, 0], state, starts[:, 0])
    other_step, _ = other.forward_fused(observations[:, 0], state, starts[:, 0])
    return torch.equal(window, other_window) and torch.equal(step, other_step)


def _tiny_board_policy_at_defaults() -> MinGRUPolicy.Config:
    """Return the tiny board policy with the port's fp32 carry and output, as exp102."""
    config = tiny_board_policy()
    config.state_dtype = config.output_dtype = torch.float32
    return config


def _board_policy_inputs(config: MinGRUPolicy.Config) -> dict[str, Tensor]:
    """Draw a bf16-stored window of 2 rows and 3 steps, a start, and the actions taken."""
    starts = torch.zeros(2, 3, dtype=torch.bool)
    starts[1, 1] = True
    return {
        "observations": packed_observations(
            config,
            batch=2,
            time=3,
            seed=10,
        ).bfloat16(),
        "starts": starts,
        "actions": torch.tensor([[4.0, 0.0, 42.0], [7.0, 13.0, 1.0]]),
    }


# The encoder's gradients are its own golden's; the blocks', the decoder's, each
# injection's and the auxiliary head's are this one's.
def _score_board_policy(module: nn.Module, inputs: dict[str, Tensor]) -> Tensor:
    """Return the window, final carry, loss and actor step, then the trunk's gradients."""
    assert isinstance(module, MinGRUPolicy)
    observations, starts = inputs["observations"], inputs["starts"]
    state = module.initial_state(starts.shape[0])
    decoded, final, loss = module.forward_sequence(
        observations,
        state,
        starts,
        actions=inputs["actions"],
    )
    weights = torch.linspace(-1.0, 1.0, decoded.numel()).reshape(decoded.shape)
    ((decoded.float() * weights).sum() + loss).backward()
    with torch.no_grad():
        step, _ = module.forward_fused(observations[:, 0], state, starts[:, 0])
    gradients: list[Tensor] = []
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        if name.startswith(("blocks.", "proj_out.", "injections.", "auxiliary.")):
            gradients.append(parameter.grad.float().flatten())
    outputs = (decoded, final, loss[None], step)
    return torch.cat([output.float().flatten() for output in outputs] + gradients)


def _assert_the_decoder_is_wide(model: MinGRUPolicy, *, device: torch.device) -> None:
    """Check fp32 decoder rows and weight gradient against float64, from one step."""
    fill_portable(model, seed=5)
    model.to(device)
    config = tiny_policy()
    observations = packed_observations(config, batch=64, seed=6).to(device)
    state = model.initial_state(64, device=device)
    assert model.proj_in is not None
    features = model.proj_in(model.embedding(observations.to(model.dtype)))
    for index, block in enumerate(model.blocks):
        features, _ = block.step(features, state[index])
    features = features.detach().double()
    weight = model.proj_out.weight
    decoded, _ = model.forward_fused(observations, state, None)
    assert decoded.dtype == torch.float32
    expected = features @ weight.detach().double().T
    torch.testing.assert_close(decoded.double(), expected, rtol=0, atol=1e-6)
    assert not torch.equal(decoded, decoded.bfloat16().float())
    generator = torch.Generator().manual_seed(7)
    grad = portable_uniform(64, 44, bound=1.0, generator=generator).to(device)
    decoded.backward(grad)
    assert weight.grad is not None
    torch.testing.assert_close(
        weight.grad.double(),
        (grad.bfloat16().double().T @ features).bfloat16().double(),
        rtol=2**-7,
        atol=0,
    )


def _backward_inputs(
    batch: int,
    time: int,
    *,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Draw portable episode starts (one step in ten) and decoder gradients."""
    generator = torch.Generator().manual_seed(2)
    starts = torch.rand(batch, time, generator=generator) < 0.1
    grad_logits = portable_uniform(batch, time, 43, bound=scale, generator=generator)
    grad_values = portable_uniform(batch, time, bound=scale, generator=generator)
    return starts, grad_logits, grad_values


# The backward is seeded as the learning rule seeds it: the fp32 gradients of the logits
# and the value, which autograd rounds once to the policy's dtype. Seeding the bf16 rows
# directly would also make cuBLAS the first thing the backward thread runs, which warns
# that the thread has no current CUDA context.
def _policy_entries(  # noqa: PLR0917 -- The inputs, as the policy's methods take them.
    model: MinGRUPolicy,
    observations: Tensor,
    starts: Tensor,
    grad_logits: Tensor,
    grad_values: Tensor,
    steps: int,
) -> list[str]:
    """Run the rollout's step, the window forward and the backward; digest them."""
    batch = starts.shape[0]
    device = observations.device
    lines: list[str] = []
    state = model.initial_state(batch, device=device)
    with torch.no_grad():
        for time in range(steps):
            decoded, _ = model.forward_fused(
                observations[:, time],
                state,
                starts[:, time],
                carry=state,
            )
            lines += [f"step {time} decoded {digest(decoded)}"]
            lines += [f"step {time} carry {digest(state)}"]
    decoded, final, _ = model.forward_sequence(
        observations,
        model.initial_state(batch, device=device),
        starts,
    )
    lines += [f"sequence decoded {digest(decoded)}"]
    lines += [f"sequence final {digest(final)}"]
    decoded.float().backward(torch.cat((grad_logits, grad_values[..., None]), dim=-1))
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        lines += [f"gradient {name} {digest(parameter.grad)}"]
    return lines


FEATURE_WIDTH: Final = 5
"""The external feature's width at test size, distinct from every other."""


def test_a_zero_feature_projection_starts_the_policy_as_its_control() -> None:
    """Init parity: the control's weights, draws, outputs and gradients, bit for bit."""
    torch.manual_seed(7)
    control = tiny_board_policy().make()
    after_control = torch.get_rng_state()
    torch.manual_seed(7)
    treated = _feature_policy().make()
    assert torch.equal(torch.get_rng_state(), after_control)
    names = [name for name, _ in treated.named_parameters()]
    assert names == [*(n for n, _ in control.named_parameters()), "proj_feature.weight"]
    for (name, ours), theirs in zip(
        treated.named_parameters(),
        control.parameters(),
        strict=False,
    ):
        assert torch.equal(ours, theirs), name
    assert treated.proj_feature is not None
    assert not treated.proj_feature.weight.any()
    observations = packed_observations(tiny_board_policy(), batch=3, time=4, seed=8)
    features = torch.randn(3, 4, FEATURE_WIDTH)
    starts = torch.zeros(3, 4, dtype=torch.bool)
    actions = torch.randint(0, 43, (3, 4))
    window, final, loss = treated.forward_sequence(
        observations,
        treated.initial_state(3),
        starts,
        actions=actions,
        features=features,
    )
    plain, plain_final, plain_loss = control.forward_sequence(
        observations,
        control.initial_state(3),
        starts,
        actions=actions,
    )
    assert torch.equal(window, plain)
    assert torch.equal(final, plain_final)
    assert torch.equal(loss, plain_loss)
    (window.float().sum() + loss).backward()
    (plain.float().sum() + plain_loss).backward()
    for (name, ours), theirs in zip(
        treated.named_parameters(),
        control.parameters(),
        strict=False,
    ):
        assert ours.grad is not None, name
        assert theirs.grad is not None, name
        assert torch.equal(ours.grad, theirs.grad), name
    assert treated.proj_feature.weight.grad is not None
    assert treated.proj_feature.weight.grad.any()


def test_a_feature_policys_window_is_its_step_applied_repeatedly() -> None:
    config = _feature_policy()
    model = config.make()
    fill_portable(model, seed=9)
    assert model.proj_feature is not None
    with torch.no_grad():
        model.proj_feature.weight.normal_(generator=torch.Generator().manual_seed(10))
    observations = packed_observations(config, batch=3, time=4, seed=11)
    features = torch.randn(
        3,
        4,
        FEATURE_WIDTH,
        generator=torch.Generator().manual_seed(12),
    )
    starts = torch.zeros(3, 4, dtype=torch.bool)
    starts[2, 1] = True
    state = model.initial_state(3)
    decoded, final, _ = model.forward_sequence(
        observations,
        state,
        starts,
        features=features,
    )
    for time in range(4):
        step, state = model.forward_fused(
            observations[:, time],
            state,
            starts[:, time],
            features=features[:, time],
        )
        assert torch.equal(step, decoded[:, time])
    assert torch.equal(state, final)
    unread, _, _ = model.forward_sequence(
        observations,
        model.initial_state(3),
        starts,
        features=torch.zeros_like(features),
    )
    assert not torch.equal(unread, decoded)


def test_a_policy_refuses_a_feature_it_cannot_read_and_the_lack_of_one() -> None:
    observations = packed_observations(tiny_board_policy(), batch=3, seed=13)
    control = tiny_board_policy().make()
    with pytest.raises(ValueError, match="no proj_feature"):
        control.forward_fused(
            observations,
            control.initial_state(3),
            None,
            features=torch.zeros(3, FEATURE_WIDTH),
        )
    treated = _feature_policy().make()
    with pytest.raises(ValueError, match="none came"):
        treated.forward_fused(observations, treated.initial_state(3), None)


def test_no_encoder_states_the_observations_width_and_encodes_none() -> None:
    config = NoEncoder.Config()
    assert config.observation_size == OBS_SIZE
    assert config.channels_concat == 0
    stage = config.make()
    assert not list(stage.parameters())
    observations = packed_observations(tiny_policy(), batch=3, time=2)
    assert stage(observations).shape == (3, 2, 0)


def test_a_policy_without_an_encoder_holds_its_trunk_heads_and_projection_alone() -> (
    None
):
    """No ``proj_in`` and no injection: what the reference's sole-WM network holds."""
    config = sole_feature_policy(feature_width=FEATURE_WIDTH)
    finalized = config.copy_tree().finalize()
    assert finalized.observation_size == tiny_board_policy().observation_size == 844
    assert finalized.proj_feature is not None
    assert finalized.proj_feature.channels_out == 8
    model = config.make()
    assert model.proj_in is None
    shapes = [(name, tuple(weight.shape)) for name, weight in model.named_parameters()]
    assert shapes == [
        ("proj_out.weight", (44, 8)),
        ("blocks.0.proj_gates.weight", (24, 8)),
        ("blocks.1.proj_gates.weight", (24, 8)),
        ("auxiliary.proj_out.weight", (43, 8)),
        ("proj_feature.weight", (8, FEATURE_WIDTH)),
    ]
    # Drawn as every projection is, so the trunk does not start from zero input.
    assert model.proj_feature is not None
    assert model.proj_feature.weight.all()


def test_a_policy_without_an_encoder_is_its_control_with_the_encoder_removed() -> None:
    """The trunk reads the feature's projection alone, never the observation.

    Against the board policy at the same weights with ``proj_in`` zeroed (a
    zero bf16 GEMM is exact zeros) the windows match bit for bit. Other
    observations change only the auxiliary loss, whose targets read them:
    each random step differs from the last, so repeating step 0 turns its
    target.
    """
    model = sole_feature_policy(feature_width=FEATURE_WIDTH).make()
    _fill_feature_policy(model, seed=14)
    control_config = tiny_board_policy()
    control_config.injection = None
    control_config.proj_feature = sole_feature_policy(
        feature_width=FEATURE_WIDTH,
    ).proj_feature
    control = control_config.make()
    control.load_state_dict(model.state_dict(), strict=False)
    assert control.proj_in is not None
    with torch.no_grad():
        control.proj_in.weight.zero_()
    observations = packed_observations(tiny_board_policy(), batch=3, time=4, seed=15)
    other = observations.clone()
    other[:, 1] = other[:, 0]
    features = portable_uniform(
        3,
        4,
        FEATURE_WIDTH,
        bound=1.0,
        generator=torch.Generator().manual_seed(17),
    )
    starts = torch.zeros(3, 4, dtype=torch.bool)
    starts[1, 2] = True
    actions = torch.randint(0, 43, (3, 4), generator=torch.Generator().manual_seed(18))
    window = partial(_scored_window, starts=starts, actions=actions)
    decoded, final, loss = window(model, observations, features)
    assert all(
        torch.equal(ours, theirs)
        for ours, theirs in zip(
            (decoded, final, loss),
            window(control, observations, features),
            strict=True,
        )
    )
    unseen, unseen_final, unseen_loss = window(model, other, features)
    assert torch.equal(unseen, decoded)
    assert torch.equal(unseen_final, final)
    assert not torch.equal(unseen_loss, loss)
    moved, _, _ = window(model, observations, features.flip(0))
    assert not torch.equal(moved, decoded)


def test_a_policy_without_an_encoder_steps_as_its_window_reads() -> None:
    config = sole_feature_policy(feature_width=FEATURE_WIDTH)
    model = config.make()
    _fill_feature_policy(model, seed=19)
    observations = packed_observations(tiny_board_policy(), batch=3, time=4, seed=20)
    features = portable_uniform(
        3,
        4,
        FEATURE_WIDTH,
        bound=1.0,
        generator=torch.Generator().manual_seed(21),
    )
    starts = torch.zeros(3, 4, dtype=torch.bool)
    starts[0, 1] = starts[2, 3] = True
    state = model.initial_state(3)
    decoded, final, _ = model.forward_sequence(
        observations,
        state,
        starts,
        features=features,
    )
    for time in range(4):
        step, state = model.forward_fused(
            observations[:, time],
            state,
            starts[:, time],
            features=features[:, time],
        )
        assert torch.equal(step, decoded[:, time])
    assert torch.equal(state, final)


def test_a_policy_without_an_encoder_needs_its_feature() -> None:
    config = sole_feature_policy(feature_width=FEATURE_WIDTH)
    config.proj_feature = None
    with pytest.raises(ValueError, match="only proj_feature to read"):
        config.make()
    model = sole_feature_policy(feature_width=FEATURE_WIDTH).make()
    observations = packed_observations(tiny_board_policy(), batch=3, seed=22)
    with pytest.raises(ValueError, match="none came"):
        model.forward_fused(observations, model.initial_state(3), None)


def test_a_policy_without_an_encoders_weights_are_pufferlibs_draws() -> None:
    """The feature's projection draws U(+-1/sqrt(fan_in)) in fp32, rounded once to bf16.

    At the world model's width, 1,152, into a one-layer trunk of 64.
    """
    config = sole_feature_policy(feature_width=FEATURE_WIDTH)
    config.channels_hidden = 64
    config.num_layers = 1
    assert config.proj_feature is not None
    config.proj_feature.channels_in = 1_152
    torch.manual_seed(0)
    model = config.make()
    for name, weight in model.named_parameters():
        assert weight.dtype == torch.bfloat16, name
        assert _fit_to_rounded_draw(weight, _pufferlib_cdf(name, weight)) > 1e-6, name


def test_a_policy_without_an_encoders_gradients_agree_with_float64_autograd() -> None:
    config = sole_feature_policy(feature_width=FEATURE_WIDTH)
    config.dtype = config.state_dtype = config.output_dtype = torch.float32
    config.auxiliary = None
    _assert_gradients_agree_with_float64_autograd(
        config,
        packed_observations(tiny_board_policy(), batch=2, time=3, seed=23),
        torch.randn(2, 3, FEATURE_WIDTH, generator=torch.Generator().manual_seed(24)),
    )


def test_the_tiny_sole_feature_policy_config_matches_its_golden() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="sole_feature_policy_tiny",
        config=_sole_feature_policy_at_defaults(),
    )


def test_the_tiny_sole_feature_policy_matches_its_bfb_golden() -> None:
    """The window, its loss, every gradient and an actor step, on every host."""
    config = _sole_feature_policy_at_defaults()
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="sole_feature_policy_tiny",
        build_module=config.make,
        build_input=_sole_feature_policy_inputs,
        run=_score_sole_feature_policy,
    )


@pytest.mark.parametrize(
    "variant",
    ["packed", "dense", "board", "feature", "sole_feature"],
)
def test_a_policys_cost_prices_the_learners_window_as_torch_runs_it(
    variant: str,
) -> None:
    """Each first stage and each optional slot: every product torch dispatches.

    fp32 throughout: the decoder's wide fp32 output over bf16 weights runs
    another product on the CPU than on CUDA.
    """
    config = _policy_variant(variant)
    config.dtype = config.state_dtype = config.output_dtype = torch.float32
    assert_cost_matches_torch(
        config,
        build_input=partial(_window_inputs, config),
        run=_scored_window_sum,
        seq_len=4,
        batch_size=3,
        dtype=None,
    )


def _policy_variant(variant: str) -> MinGRUPolicy.Config:
    """Return the tiny policy of each first stage, and with each optional slot."""
    if variant == "dense":
        config = tiny_policy()
        config.embedding = DenseObservation.Config()
        return config
    if variant == "board":
        return tiny_board_policy()
    if variant == "feature":
        return _feature_policy()
    if variant == "sole_feature":
        return sole_feature_policy(feature_width=FEATURE_WIDTH)
    return tiny_policy()


def _window_inputs(config: MinGRUPolicy.Config) -> tuple[Tensor, ...]:
    """Draw 3 rows of 4 steps: the observations, the actions taken, and any feature."""
    if isinstance(config.embedding, DenseObservation.Config):
        observations = torch.rand(3, 4, SYMBOLIC_OBS_SIZE)
    else:
        packed = (
            tiny_board_policy()
            if isinstance(config.embedding, NoEncoder.Config)
            else config
        )
        observations = packed_observations(packed, batch=3, time=4)
    actions = torch.randint(43, (3, 4)).float()
    if config.proj_feature is None:
        return observations, actions
    return observations, actions, torch.randn(3, 4, FEATURE_WIDTH)


def _scored_window_sum(model: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Sum a zero-carry window's rows and its loss, so backward reaches every product."""
    assert isinstance(model, MinGRUPolicy)
    observations, actions, *feature = inputs
    rows, steps = actions.shape
    decoded, _, loss = model.forward_sequence(
        observations,
        model.initial_state(rows),
        torch.zeros(rows, steps),
        actions=actions,
        features=feature[0] if feature else None,
    )
    return decoded.sum() + loss


def _feature_policy() -> MinGRUPolicy.Config:
    """Return the tiny board policy reading a zero-initialized feature projection."""
    config = tiny_board_policy()
    proj = config.proj_feature = Linear.Config()
    proj.channels_in = FEATURE_WIDTH
    proj.init_weight = nn.init.zeros_
    return config


# No encoder and no injection; it is handed the board policy's observations, whose
# packed fields the loss compares.


def _sole_feature_policy_at_defaults() -> MinGRUPolicy.Config:
    """Return the tiny sole-feature policy with the port's fp32 carry and output."""
    config = sole_feature_policy(feature_width=FEATURE_WIDTH)
    config.state_dtype = config.output_dtype = torch.float32
    return config


def _scored_window(
    policy: MinGRUPolicy,
    observations: Tensor,
    features: Tensor,
    *,
    starts: Tensor,
    actions: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return a window's rows, final carry and auxiliary loss from a zero carry."""
    return policy.forward_sequence(
        observations,
        policy.initial_state(starts.shape[0]),
        starts,
        actions=actions,
        features=features,
    )


def _fill_feature_policy(model: MinGRUPolicy, *, seed: int) -> None:
    """Fill :func:`fill_portable`'s weights, then the feature's projection after them."""
    fill_portable(model, seed=seed)
    assert model.proj_feature is not None
    weight = model.proj_feature.weight
    with torch.no_grad():
        weight.copy_(
            portable_uniform(
                *weight.shape,
                bound=weight.shape[-1] ** -0.5,
                generator=torch.Generator().manual_seed(seed + 1),
            ),
        )


def _sole_feature_policy_inputs() -> dict[str, Tensor]:
    """Draw the board policy's window inputs and a feature for every step."""
    inputs = _board_policy_inputs(tiny_board_policy())
    inputs["features"] = portable_uniform(
        2,
        3,
        FEATURE_WIDTH,
        bound=1.0,
        generator=torch.Generator().manual_seed(25),
    ).bfloat16()
    return inputs


def _score_sole_feature_policy(module: nn.Module, inputs: dict[str, Tensor]) -> Tensor:
    """Return the window, final carry, loss and actor step, then every gradient."""
    assert isinstance(module, MinGRUPolicy)
    observations, starts = inputs["observations"], inputs["starts"]
    features = inputs["features"]
    state = module.initial_state(starts.shape[0])
    decoded, final, loss = module.forward_sequence(
        observations,
        state,
        starts,
        actions=inputs["actions"],
        features=features,
    )
    weights = torch.linspace(-1.0, 1.0, decoded.numel()).reshape(decoded.shape)
    ((decoded.float() * weights).sum() + loss).backward()
    with torch.no_grad():
        step, _ = module.forward_fused(
            observations[:, 0],
            state,
            starts[:, 0],
            features=features[:, 0],
        )
    gradients: list[Tensor] = []
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        gradients.append(parameter.grad.float().flatten())
    outputs = (decoded, final, loss[None], step)
    return torch.cat([output.float().flatten() for output in outputs] + gradients)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
