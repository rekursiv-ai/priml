"""Tests for one native Qwen3.5 DPPO update."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

import functools
import json
import tempfile

from torch import Tensor, nn

import pytest
import torch

from priml import hub, runtime
from priml.baselines.tmax import train_step as train_step_module
from priml.baselines.tmax.train_step import (
    TMaxDPPOTrainStep,
    _cluster_attention,
    _global_token_count,
    _hf_projection_order,
    _load_checkpoint_state,
    _merge_results,
    _row_metrics,
    _row_response_tokens,
    _rows,
    _update_token_count,
    tiny_qwen35_config,
)
from priml.model.attention.gated_self_attention import GatedSelfAttention
from priml.model.attention.kernel import SdpaFused, SdpaNaive
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3_5 import Qwen35
from priml.model.transformer.qwen3_5_weights import export_hf_state_dict
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.qwen3_5 import hf_config, native_qwen35_config
from priml.train.parallelism import FullySharded, NoParallel


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter
    from priml.train.custom_types import TrainStepOutput


_TESTDATA: Final = Path(__file__).resolve().parent / "testdata"


def _batch() -> dict[str, object]:
    """Build one small packed batch for train-step tests."""
    return {
        "query_responses": torch.tensor([1, 2, 3, 4, 5, 6]),
        "attention_mask": torch.ones(6, dtype=torch.long),
        "position_ids": torch.arange(6),
        "response_mask": torch.tensor([0, 0, 0, 1, 1, 1]),
        "prompt_mask": torch.tensor([1, 1, 1, 0, 0, 0]),
        "rollout_sample_ids": torch.zeros(6, dtype=torch.long),
        "model_steps": torch.zeros(6, dtype=torch.long),
        "vllm_logprobs": torch.tensor([float("nan")] * 3 + [-3.0] * 3),
        "advantages": torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0, 0.5]),
    }


def _row(*, prompt: int, response: int, logprob: float = -0.5) -> dict[str, object]:
    """One packed single-segment row with a uniform positive advantage."""
    total = prompt + response
    return {
        "query_responses": torch.arange(1, total + 1),
        "attention_mask": torch.ones(total, dtype=torch.long),
        "position_ids": torch.arange(total),
        "response_mask": torch.tensor([0] * prompt + [1] * response, dtype=torch.long),
        "prompt_mask": torch.tensor([1] * prompt + [0] * response, dtype=torch.long),
        "rollout_sample_ids": torch.zeros(total, dtype=torch.long),
        "model_steps": torch.zeros(total, dtype=torch.long),
        "vllm_logprobs": torch.tensor([float("nan")] * prompt + [logprob] * response),
        "advantages": torch.tensor([0.0] * prompt + [1.0] * response),
    }


def _step() -> TMaxDPPOTrainStep:
    """Build a deterministic CPU step over the tiny shared Qwen3.5 fixture."""
    torch.manual_seed(0)
    config = TMaxDPPOTrainStep.Config()
    config.model = tiny_qwen35_config()
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    config.gradient_clip_norm = float("inf")
    return config.make()


def test_native_qwen_dppo_update_changes_parameters() -> None:
    """Produce a finite loss, advance the step, and update model weights."""
    step = _step()
    before = next(step.model.parameters()).detach().clone()
    result = step.train_step(**step.preprocess_batch(_batch()))
    assert result["loss"].ndim == 0
    assert step.global_step == 1
    assert torch.isfinite(result["loss"])
    assert cast(Tensor, _row_metrics(result)["response_tokens"]) == 3.0
    assert not torch.equal(before, next(step.model.parameters()).detach())


def test_multi_row_loss_is_one_global_token_mean() -> None:
    """Average a multi-row loss over all response tokens in the update."""
    step = _step()
    first = _row(prompt=3, response=3)
    second = _row(prompt=2, response=2, logprob=-0.25)
    together = step.train_loss(
        rows=(first, second),
        response_token_count=5.0,
    )
    first_alone = step.train_loss(rows=(first,), response_token_count=5.0)
    second_alone = step.train_loss(rows=(second,), response_token_count=5.0)
    torch.testing.assert_close(
        together["loss"],
        first_alone["loss"] + second_alone["loss"],
        rtol=0.0,
        atol=0.0,
    )
    assert cast(Tensor, _row_metrics(together)["response_tokens"]) == 5.0
    # Every row reports its SHARE of the update's global token mean, so the
    # merge sums the shares -- averaging row means would shrink the number
    # (float32 sums in a different order, hence the tolerance).
    torch.testing.assert_close(
        cast(Tensor, _row_metrics(together)["ratio_mean"]),
        cast(Tensor, _row_metrics(first_alone)["ratio_mean"])
        + cast(Tensor, _row_metrics(second_alone)["ratio_mean"]),
        rtol=1e-6,
        atol=0.0,
    )


def test_per_row_backward_matches_one_summed_backward() -> None:
    """Match one batched update with per-row backward calls and one step."""
    first = _row(prompt=3, response=3)
    second = _row(prompt=2, response=2, logprob=-0.25)
    batched = _step()
    sequential = _step()
    batched.train_step(
        **batched.preprocess_batch(
            {"rows": (first, second), "response_token_count": 5.0},
        ),
    )
    for row in (first, second):
        alone = sequential.train_loss(
            **sequential.preprocess_batch(
                {"rows": (row,), "response_token_count": 5.0},
            ),
        )
        alone["loss"].backward()
    sequential.step()
    for batched_param, sequential_param in zip(
        batched.model.parameters(),
        sequential.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            batched_param.detach(),
            sequential_param.detach(),
            rtol=0.0,
            atol=0.0,
        )


def test_microbatch_accumulation_is_rejected() -> None:
    """Reject gradient accumulation because each batch is one full update."""
    config = TMaxDPPOTrainStep.Config()
    config.accumulate_grad_batches = 2
    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        config.make()


def test_full_checkpoint_uses_the_distributed_loader_for_dtensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load a full checkpoint through PyTorch's distributed state loader."""
    model = nn.Linear(2, 2)
    state = model.state_dict()
    captured: dict[str, object] = {}

    def load(
        target: nn.Module,
        values: dict[str, Tensor],
        *,
        options: object,
    ) -> None:
        """Record the distributed state-load call."""
        captured.update(target=target, values=values, options=options)

    monkeypatch.setattr(train_step_module, "DTensor", Tensor)
    monkeypatch.setattr(train_step_module, "set_model_state_dict", load)
    _load_checkpoint_state(model, state)

    assert captured["target"] is model
    assert captured["values"] is state
    options = vars(captured["options"])
    assert options["full_state_dict"] is True
    assert options["strict"] is True


def test_checkpoint_metadata_failures_are_loud(tmp_path: Path) -> None:
    """Report distinct errors for invalid checkpoint metadata."""
    config = TMaxDPPOTrainStep.Config()
    config.model_path = tmp_path
    with pytest.raises(FileNotFoundError, match="config not found"):
        config.make()

    (tmp_path / "config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="object"):
        config.make()

    (tmp_path / "config.json").write_text(json.dumps(hf_config()), encoding="utf-8")
    with pytest.raises(ValueError, match="pad_token_id"):
        config.make()


def test_pad_id_defaults_to_zero_without_a_checkpoint() -> None:
    """Use padding id zero for the checkpoint-free smoke model."""
    assert _step().pad_token_id == 0


def test_the_smoke_model_is_the_hf_parity_fixture() -> None:
    """The step builds exactly the architecture priml's HF parity tests pin.

    ``qwen3_5_hf_test`` proves the native class reproduces Hugging Face Qwen3.5
    -- which is the model the released trainer runs -- bit for bit on logits and
    gradients, but only for the config it builds there. Asserting the identity
    is what transfers that evidence to this training step, gate/up projection
    order included, instead of relying on the two merely looking similar.
    """
    assert tiny_qwen35_config().pprint() == native_qwen35_config().pprint()


def test_the_hf_projection_order_refuses_a_tree_it_cannot_reach() -> None:
    """A required guard, not a nicety: skipping a block trains the wrong order."""
    with pytest.raises(TypeError, match="explicit block list"):
        _hf_projection_order(Qwen35.Config())

    template = Qwen35.Config()
    template.block = [SwiGLU.Config()]
    with pytest.raises(TypeError, match="TransformerBlock configs"):
        _hf_projection_order(template)

    template = Qwen35.Config()
    block = TransformerBlock.Config()
    # Type-valid but not a SwiGLU: exactly the case the third guard exists for.
    block.ffn = RMSNorm.Config()
    template.block = [block]
    with pytest.raises(TypeError, match="SwiGLU feed-forward blocks"):
        _hf_projection_order(template)


def test_temperature_must_be_positive() -> None:
    """Reject a non-positive scoring temperature."""
    config = TMaxDPPOTrainStep.Config()
    config.temperature = 0.0
    with pytest.raises(ValueError, match="temperature"):
        config.make()


def test_the_cluster_recipe_fuses_full_attention() -> None:
    """A 67,584-token row cannot be scored with a dense score tensor.

    ``SdpaNaive`` materializes ``q @ k^T``, which for Qwen3.5-4B's 32 heads over
    a full-length packed row is ~145 GB in bfloat16 -- the released launcher
    therefore runs the fused flash backend its own ``detect_attn_implementation``
    selects. The cluster path must not inherit the eager-equivalent default that
    the CPU parity goldens pin.
    """
    config = _cluster_attention(Qwen35.Config.from_hf(hf_config()))
    assert isinstance(config.block, list)
    kernels = [
        block.attn.attn_kernel
        for block in config.block
        if isinstance(block, TransformerBlock.Config)
        and isinstance(block.attn, GatedSelfAttention.Config)
    ]
    assert kernels, "the fixture must carry a full-attention block"
    assert all(isinstance(kernel, SdpaFused.Config) for kernel in kernels)


def test_the_eager_parity_paths_keep_the_naive_kernel() -> None:
    """The HF parity evidence was generated against eager attention, so it stays."""
    config = tiny_qwen35_config()
    assert isinstance(config.block, list)
    for block in config.block:
        assert isinstance(block, TransformerBlock.Config)
        if isinstance(block.attn, GatedSelfAttention.Config):
            assert isinstance(block.attn.attn_kernel, SdpaNaive.Config)


def test_the_cluster_attention_refuses_a_tree_it_cannot_reach() -> None:
    """Reject attention fusion for a model without explicit blocks."""
    with pytest.raises(TypeError, match="explicit block list"):
        _cluster_attention(Qwen35.Config())

    template = Qwen35.Config()
    template.block = [SwiGLU.Config()]
    with pytest.raises(TypeError, match="TransformerBlock configs"):
        _cluster_attention(template)


def test_a_staged_checkpoint_scores_on_the_fused_kernel(
    tmp_path: Path,
) -> None:
    """The checkpoint defines the cluster recipe; the built policy is fused."""
    source = _staged_tiny_checkpoint(tmp_path / "source")
    config = TMaxDPPOTrainStep.Config()
    config.model_path = source
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    step = config.make()
    kernels = [
        module.attn_kernel
        for module in step.model.modules()
        if isinstance(module, GatedSelfAttention)
    ]
    assert kernels, "the fixture must build a full-attention block"
    assert all(isinstance(kernel, SdpaFused) for kernel in kernels)


def test_divergence_threshold_must_be_positive() -> None:
    """Reject a non-positive trust-region threshold."""
    config = TMaxDPPOTrainStep.Config()
    config.divergence_threshold = 0.0
    with pytest.raises(ValueError, match="divergence threshold"):
        config.make()


def test_a_negative_pad_token_id_is_refused() -> None:
    """Reject a negative padding token id."""
    config = TMaxDPPOTrainStep.Config()
    config.pad_token_id = -1
    with pytest.raises(ValueError, match="pad_token_id"):
        config.make()


def test_a_staged_checkpoint_owns_the_architecture_and_pad_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """exp000's checkpoint is the source of truth for the model and the pad id."""
    metadata = {**hf_config(), "pad_token_id": 7}
    (tmp_path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    architecture = Qwen35.Config.from_hf(metadata)

    def load(*_args: object, **_kwargs: object) -> Qwen35:
        """Return the small model used to test checkpoint adoption."""
        return architecture.make()

    monkeypatch.setattr("priml.baselines.tmax.train_step.Qwen35.load", load)
    config = TMaxDPPOTrainStep.Config()
    config.model_path = tmp_path
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    assert config.make().pad_token_id == 7


def test_a_declared_architecture_must_match_the_staged_checkpoint(
    tmp_path: Path,
) -> None:
    """Reject a declared model that differs from the staged checkpoint."""
    source = _staged_tiny_checkpoint(tmp_path / "source")
    config = TMaxDPPOTrainStep.Config()
    config.model_path = source
    config.model = tiny_qwen35_config(hidden_size=8)
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    with pytest.raises(ValueError, match=r"does not match step\.model"):
        config.make()


def _staged_tiny_checkpoint(root: Path) -> Path:
    """Stage a tiny source checkpoint: config, safetensors weights, tokenizer."""
    torch.manual_seed(0)
    metadata = {**hf_config(), "pad_token_id": 7}
    architecture = Qwen35.Config.from_hf(metadata)
    model = architecture.make()
    hub.save_hf_checkpoint(
        root,
        metadata,
        export_hf_state_dict(model.state_dict(), architecture),
    )
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


def _fsdp_checkpoint_worker(
    checkpoint: str,
    result_dir: str,
    mesh: DeviceMesh,
) -> None:
    """Load and verify a full checkpoint through a two-rank FSDP model."""
    rank = mesh.get_rank()
    try:
        runtime._device_mesh = mesh
        config = TMaxDPPOTrainStep.Config()
        config.model_path = checkpoint
        config.parallelism = FullySharded.Config()
        config.dtype_autocast = None
        actual = config.make().hf_state_dict()
        _, expected = hub.load_hf_checkpoint(checkpoint, dtype=None)
        matches = all(
            torch.equal(actual[name], value) for name, value in expected.items()
        )
        Path(result_dir, f"rank_{rank}").write_text("ok" if matches else "mismatch")
    except Exception as error:  # noqa: BLE001 -- Serialize worker failures.
        Path(result_dir, f"rank_{rank}").write_text(f"FAIL:{error!r}")
    finally:
        runtime._device_mesh = None


@pytest.mark.compute_distributed
def test_full_checkpoint_loads_into_fully_sharded_model(
    tmp_path: Path,
    warm_pools: WarmPoolGetter,
) -> None:
    """Rebuild a full checkpoint through both FSDP ranks."""
    checkpoint = _staged_tiny_checkpoint(tmp_path / "source")
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as result_dir:
        pool(
            functools.partial(
                _fsdp_checkpoint_worker,
                str(checkpoint),
                result_dir,
            ),
        )
        results = {
            path.name: path.read_text()
            for path in Path(result_dir).iterdir()
            if path.is_file()
        }
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


def test_export_round_trips_the_trained_weights(tmp_path: Path) -> None:
    """The exported checkpoint is the staged format with the trained weights."""
    source = _staged_tiny_checkpoint(tmp_path / "source")
    (source / "generation_config.json").write_text("{}", encoding="utf-8")
    (source / "README.md").write_text("not part of the export", encoding="utf-8")
    config = TMaxDPPOTrainStep.Config()
    config.model_path = source
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    step = config.make()

    # Move the policy off its initialization so the export is proven to read
    # the CURRENT weights, not the staged ones it was built from.
    parameter = next(step.model.parameters())
    with torch.no_grad():
        parameter.add_(1.0)
    step.export_checkpoint(tmp_path / "export")

    reloaded = Qwen35.load(tmp_path / "export")
    for name, value in step.model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[name], value)
    written = cast(
        "dict[str, object]",
        json.loads(
            (tmp_path / "export" / "config.json").read_text(encoding="utf-8"),
        ),
    )
    assert written == {**hf_config(), "pad_token_id": 7}
    assert (tmp_path / "export" / "tokenizer.json").is_file()
    assert (tmp_path / "export" / "generation_config.json").is_file()
    assert not (tmp_path / "export" / "README.md").exists()


def test_export_needs_the_staged_source(tmp_path: Path) -> None:
    """A checkpoint-free smoke model has no config or tokenizer to carry."""
    step = _step()
    with pytest.raises(ValueError, match="model_path"):
        step.export_checkpoint(tmp_path / "export")


def test_export_derives_the_text_only_config(tmp_path: Path) -> None:
    """The released multimodal source exports as the text model upstream saves.

    Upstream's trainer holds the text model ``AutoModelForCausalLM`` maps the
    conditional-generation config onto (``Qwen3_5ForCausalLM``, which drops
    the vision tower at load), and that model's own ``save_pretrained`` writes
    the extracted text config. The export carries that config rather than
    refusing the real source checkpoint after the run that produced it.
    """
    source = _staged_tiny_checkpoint(tmp_path / "source")
    config = TMaxDPPOTrainStep.Config()
    config.model_path = source
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    step = config.make()
    assert step._checkpoint_metadata is not None
    # The released form: a conditional-generation wrapper around a
    # text_config whose own tied-head flag differs from the wrapper's.
    step._checkpoint_metadata["text_config"] = {
        **hf_config(),
        "tie_word_embeddings": True,
    }
    step._checkpoint_metadata["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    step._checkpoint_metadata["model_type"] = "qwen3_5"
    step._checkpoint_metadata["tie_word_embeddings"] = False
    step.export_checkpoint(tmp_path / "export")

    written = cast(
        "dict[str, object]",
        json.loads(
            (tmp_path / "export" / "config.json").read_text(encoding="utf-8"),
        ),
    )
    assert "text_config" not in written
    assert written["architectures"] == ["Qwen3_5ForCausalLM"]
    assert written["model_type"] == "qwen3_5_text"
    # The wrapper's tied-head flag propagates over the nested one, exactly as
    # transformers' text-config extraction and Qwen35.Config.from_hf read it.
    assert written["tie_word_embeddings"] is False
    # The exported checkpoint reloads as the trained text model.
    reloaded = Qwen35.load(tmp_path / "export")
    for name, value in step.model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[name], value)


def test_the_pad_id_resolves_from_the_staged_tokenizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The released source's config carries no pad id; upstream uses the tokenizer's.

    Upstream's trainer takes ``tokenizer.pad_token_id`` off the tokenizer
    built from the staged checkpoint, and the released ``hamishivi/Qwen3.5-4B``
    ``config.json`` names none -- so a config-only reader has nothing to
    fall back on and must resolve the tokenizer's, not refuse to start.
    """
    metadata = {**hf_config()}
    assert "pad_token_id" not in metadata
    (tmp_path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "<|endoftext|>",
                "added_tokens_decoder": {
                    "248044": {"content": "<|endoftext|>"},
                    "248046": {"content": "<|im_end|>"},
                },
            },
        ),
        encoding="utf-8",
    )
    architecture = Qwen35.Config.from_hf(metadata)

    def load(*_args: object, **_kwargs: object) -> Qwen35:
        """Return the small model used to test checkpoint adoption."""
        return architecture.make()

    monkeypatch.setattr("priml.baselines.tmax.train_step.Qwen35.load", load)
    config = TMaxDPPOTrainStep.Config()
    config.model_path = tmp_path
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    assert config.make().pad_token_id == 248_044


def test_an_unresolvable_pad_id_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No config pad id and no staged tokenizer to resolve one: refuse, don't guess."""
    metadata = {**hf_config()}
    (tmp_path / "config.json").write_text(json.dumps(metadata), encoding="utf-8")
    architecture = Qwen35.Config.from_hf(metadata)

    def load(*_args: object, **_kwargs: object) -> Qwen35:
        """Return a checkpoint model without a usable padding id."""
        return architecture.make()

    monkeypatch.setattr("priml.baselines.tmax.train_step.Qwen35.load", load)
    config = TMaxDPPOTrainStep.Config()
    config.model_path = tmp_path
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    with pytest.raises(ValueError, match="pad_token_id"):
        config.make()


def test_a_staged_checkpoint_does_not_patch_its_head(tmp_path: Path) -> None:
    """Fp32 scoring is injected per call, leaving the registered head intact."""
    source = _staged_tiny_checkpoint(tmp_path / "source")
    staged = TMaxDPPOTrainStep.Config()
    staged.model_path = source
    staged.parallelism = NoParallel.Config(device="cpu")
    staged.dtype_autocast = torch.bfloat16
    staged_step = staged.make()
    tokens = torch.arange(1, 7).unsqueeze(0)
    positions = torch.arange(6).unsqueeze(0)
    staged_model = cast(Qwen35, staged_step.model)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        staged_logits = staged_model(tokens, positions=positions)
    assert staged_logits.dtype == torch.bfloat16
    assert "forward" not in vars(staged_model.proj_out)


def test_on_epoch_end_exports_the_configured_directory(tmp_path: Path) -> None:
    """Export trained weights to the configured directory at epoch end."""
    source = _staged_tiny_checkpoint(tmp_path / "source")
    config = TMaxDPPOTrainStep.Config()
    config.model_path = source
    config.export_dir = tmp_path / "export"
    config.parallelism = NoParallel.Config(device="cpu")
    config.dtype_autocast = None
    step = config.make()
    assert not (tmp_path / "export").exists()
    step.on_epoch_end()
    assert (tmp_path / "export" / "model.safetensors").is_file()


def test_on_epoch_end_is_a_pure_flush_by_default() -> None:
    """No export configured: the hook must neither write nor raise.

    The smoke model is checkpoint-free, so a default export_dir would
    make every epoch boundary fail loudly here.
    """
    step = _step()
    assert step.config.export_dir is None
    step.on_epoch_end()


def test_eval_loss_is_a_detached_diagnostic() -> None:
    """Return a finite evaluation loss without a gradient."""
    step = _step()
    result = step.eval_loss(
        **step.preprocess_batch({"rows": (_row(prompt=3, response=3),)}),
    )
    loss = result["loss"]
    assert not loss.requires_grad
    assert torch.isfinite(loss)


def test_a_row_that_is_not_one_dimensional_is_refused() -> None:
    """Require each packed row to be one-dimensional."""
    row = _row(prompt=2, response=2)
    row["query_responses"] = cast(Tensor, row["query_responses"]).unsqueeze(0)
    with pytest.raises(ValueError, match="one packed row"):
        _step().train_loss(rows=(row,), response_token_count=4.0)


def test_row_tensors_must_share_one_shape() -> None:
    """Require every per-token tensor in a row to share one shape."""
    row = _row(prompt=2, response=2)
    row["advantages"] = cast(Tensor, row["advantages"])[:-1]
    with pytest.raises(ValueError, match="same shape"):
        _step().train_loss(rows=(row,), response_token_count=4.0)


def test_a_non_tensor_row_field_is_refused() -> None:
    """Reject a row field that is not a tensor."""
    row = _row(prompt=2, response=2)
    row["advantages"] = [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(TypeError, match=r"advantages must be a torch\.Tensor"):
        _step().train_loss(rows=(row,), response_token_count=4.0)


def test_an_empty_row_batch_is_refused() -> None:
    """Require an update to contain at least one packed row."""
    with pytest.raises(ValueError, match="non-empty list or tuple"):
        _rows({"rows": []})


def test_a_row_batch_of_non_mappings_is_refused() -> None:
    """Require every row in a batch to be a mapping."""
    with pytest.raises(TypeError, match="mapping objects"):
        _rows({"rows": [1]})


def test_a_pinned_row_token_count_is_reused() -> None:
    """Reuse a row's stored response-token count."""
    row = {"row_response_tokens": 3, "response_mask": torch.zeros(5, dtype=torch.long)}
    assert _row_response_tokens(row) == 3


def test_a_scalar_tensor_token_count_is_read() -> None:
    """Read an update token count from a scalar tensor."""
    assert _global_token_count({"response_token_count": torch.tensor(5.0)}) == 5.0


def test_a_non_scalar_token_count_is_refused() -> None:
    """Reject a token count containing more than one value."""
    with pytest.raises(ValueError, match="must be scalar"):
        _global_token_count({"response_token_count": torch.tensor([1.0, 2.0])})


def test_a_non_numeric_token_count_is_refused() -> None:
    """Reject a token count that is not numeric."""
    with pytest.raises(TypeError, match="scalar number"):
        _global_token_count({"response_token_count": "many"})


def test_a_non_positive_token_count_is_refused() -> None:
    """Require the update token count to be positive and finite."""
    with pytest.raises(ValueError, match="positive and finite"):
        _global_token_count({"response_token_count": 0.0})


def test_an_update_without_response_tokens_is_refused() -> None:
    """Reject an update with no response tokens to score."""
    row = {"response_mask": torch.zeros(4, dtype=torch.long)}
    with pytest.raises(ValueError, match="at least one response token"):
        _update_token_count({}, (row,))


def test_a_row_result_without_metrics_is_refused() -> None:
    """Require row results to include metrics before merging."""
    with pytest.raises(ValueError, match="must carry its metrics"):
        _row_metrics(cast("TrainStepOutput", {"loss": torch.zeros(())}))


def test_merging_no_rows_is_refused() -> None:
    """Reject an empty set of row results."""
    with pytest.raises(ValueError, match="empty DPPO result"):
        _merge_results([], token_count=1.0)


class _TrainStepModule(nn.Module):
    """Adapts the DPPO step to the module interface the golden harness drives.

    Registering the step's model as a child puts its weights in this module's
    ``state_dict``, which is what the harness randomizes and compares.
    """

    def __init__(self) -> None:
        """Build the three-update module used by the compiled golden test."""
        super().__init__()
        torch.manual_seed(0)
        config = TMaxDPPOTrainStep.Config()
        # Half the shared fixture's width: the golden stores the whole state
        # twice (pre-run and post-run), and the repo's golden size limit is
        # 32 KiB.
        config.model = tiny_qwen35_config(hidden_size=8)
        config.parallelism = NoParallel.Config(device="cpu")
        config.dtype_autocast = None
        self.step = config.make()
        self.inner = self.step.model

    @override
    def forward(self, **row: Tensor) -> Tensor:
        """Run three training updates and return their losses."""
        losses = [self.step.train_step(**row)["loss"] for _ in range(3)]
        return torch.stack(losses)


# Wrapping the whole train step, not just a forward, is what makes the golden
# cover the recipe: the DPPO mask, the loss, and AdamW all affect the compared
# end state. Not compute_training: three tiny CPU updates, like imagenet's
# three-step golden.
def test_dppo_train_steps_bfb() -> None:
    """Pin three DPPO updates: losses and every resulting weight."""
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="dppo_three_steps",
        build_module=_TrainStepModule,
        build_input=_batch,
        seed=42,
    )
