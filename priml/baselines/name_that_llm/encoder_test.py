"""Pin the native Qwen forward, pooling arithmetic, padding, and saved-layer order."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

import json
import math

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torch import Tensor
from treekle import json_loads, to_plain

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm import encoder as encoder_module
from priml.baselines.name_that_llm.encoder import (
    FrozenQwen,
    QwenRMSNorm,
    mean_pool,
    selected_hidden_states,
)
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.rope import RoPE, rotate_activation_dtype
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3 import Qwen3, remap_hf_state_dict
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import assert_tensor_golden_part, mismatches


_CWD: Final = Path(__file__).resolve().parent

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from transformers.models.auto import tokenization_auto
    from transformers.tokenization_utils_base import (
        BatchEncoding,
        PreTrainedTokenizerBase,
    )
    from transformers.tokenization_utils_tokenizers import TokenizersBackend
else:
    from wrapt import lazy_import

    tokenization_auto = lazy_import("transformers.models.auto.tokenization_auto")
    BatchEncoding = lazy_import("transformers.tokenization_utils_base", "BatchEncoding")
    TokenizersBackend = lazy_import(
        "transformers.tokenization_utils_tokenizers",
        "TokenizersBackend",
    )


@pytest.fixture
def tiny_checkpoint(tmp_path: Path) -> FrozenQwen.Config:
    model_config = {
        "model_type": "qwen3",
        "hidden_size": 7,
        "intermediate_size": 11,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 6,
        "vocab_size": 17,
        "torch_dtype": "float32",
    }
    (tmp_path / "config.json").write_text(json.dumps(to_plain(model_config)))
    # ``pytorch_model.bin`` loads with torch alone; safetensors is the hub extra.
    torch.save(_checkpoint_state(), tmp_path / "pytorch_model.bin")
    config = FrozenQwen.Config()
    config.directory = tmp_path
    config.layers = [1, 2, 3, 4]
    config.channels_hidden = 7
    config.batch_size = 2
    config.max_length = 5
    config.dtype = torch.float16
    config.release_model = _no_collection
    return config


@pytest.fixture
def tiny_encoder(tiny_checkpoint: FrozenQwen.Config) -> FrozenQwen.Config:
    # Tokenizer files load through ``transformers``, priml's ``hub`` extra.
    pytest.importorskip("transformers")
    backend = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[EOS]": 1, "red": 2, "blue": 3, "green": 4},
            unk_token="[UNK]",  # noqa: S106 -- Literal unknown-token marker.
        ),
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="$A [EOS]",
        special_tokens=[("[EOS]", 1)],
    )
    tokenizer = TokenizersBackend(
        tokenizer_object=backend,
        eos_token="[EOS]",  # noqa: S106 -- Literal end-of-sequence marker.
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(tiny_checkpoint.directory)
    return tiny_checkpoint


def test_checkpoint_builds_a_frozen_priml_qwen(
    tiny_checkpoint: FrozenQwen.Config,
) -> None:
    model = tiny_checkpoint.make()._load_model(torch.float32)
    assert type(model) is Qwen3
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        assert type(block.attn) is Attention
        assert block.attn.rotate == rotate_activation_dtype


def test_checkpoint_recipe_edits_follow_the_encoder_slots(
    tiny_checkpoint: FrozenQwen.Config,
) -> None:
    tiny_checkpoint.norm = RMSNorm.Config()
    tiny_checkpoint.rotate = RoPE.rotate
    tiny_checkpoint.split_qkv_projection = False
    tiny_checkpoint.split_gate_projection = False
    model = tiny_checkpoint.make()._load_model(torch.float32)
    assert isinstance(model.proj_out, Sequential)
    assert type(model.proj_out[0]) is RMSNorm
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        assert isinstance(block.attn, Attention)
        assert isinstance(block.ffn, SwiGLU)
        assert type(block.norm1) is RMSNorm
        assert type(block.norm2) is RMSNorm
        assert type(block.attn.norm_q) is RMSNorm
        assert block.attn.rotate == RoPE.rotate
        assert not block.attn.split_qkv_projection
        assert not block.ffn.split_gate_projection


def test_hidden_states_require_the_token_embedding(
    tiny_checkpoint: FrozenQwen.Config,
) -> None:
    model = tiny_checkpoint.make()._load_model(torch.float32)
    model.proj_in = None
    with pytest.raises(ValueError, match="proj_in"):
        selected_hidden_states(
            model,
            input_ids=torch.zeros(2, 3, dtype=torch.long),
            attention_mask=torch.ones(2, 3, dtype=torch.long),
            layers=[1],
        )


def test_checkpoint_width_must_match_the_configured_features(
    tiny_checkpoint: FrozenQwen.Config,
) -> None:
    tiny_checkpoint.channels_hidden = 13
    with pytest.raises(ValueError, match="checkpoint width"):
        tiny_checkpoint.make()._load_model(torch.float32)


@pytest.mark.parametrize(
    "dtype",
    argvalues=[torch.float32, torch.float16, torch.bfloat16],
)
def test_native_qwen_replays_pooled_states_at_every_precision(
    tiny_checkpoint: FrozenQwen.Config,
    dtype: torch.dtype,
) -> None:
    tiny_checkpoint.layers = [0, 1, 2, 4]
    tiny_checkpoint.attention_kernel = SdpaNaive.Config()
    metadata = json_loads(
        (Path(tiny_checkpoint.directory) / "config.json").read_text(),
        dict[str, object],
    )
    with host_agnostic_numerics():
        model = tiny_checkpoint.make()._load_model(dtype)
    loaded = remap_hf_state_dict(_checkpoint_state(), Qwen3.Config.from_hf(metadata))
    expected = {name: value.to(dtype) for name, value in loaded.items()}
    assert mismatches(expected, actual=model.state_dict()) == []
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    pools: list[Tensor] = []
    with host_agnostic_numerics(), torch.inference_mode():
        for case in range(3):
            states = selected_hidden_states(
                model,
                input_ids=(torch.arange(15).reshape(3, 5) + case) % 17,
                attention_mask=mask,
                layers=tiny_checkpoint.layers,
            )
            pools.append(torch.stack([mean_pool(hidden, mask) for hidden in states]))
    prefix = f"{str(dtype).removeprefix('torch.')}/"
    assert_tensor_golden_part(
        _CWD / "testdata" / "encoder.pt",
        prefix=prefix,
        record={f"{prefix}pool": torch.stack(pools)},
    )


def test_unweighted_qwen_norm_keeps_dtype_and_dtype_default_epsilon() -> None:
    config = QwenRMSNorm.Config()
    config.channels_in = 5
    config.eps = None
    value = torch.ones(2, 3, 5, dtype=torch.float16)
    actual = config.make()(value)
    assert actual.dtype == torch.float16
    expected = torch.rsqrt(torch.tensor(1 + torch.finfo(torch.float16).eps)).half()
    assert torch.equal(actual, expected.expand_as(value))


@pytest.mark.parametrize("affine", argvalues=[False, True])
def test_qwen_norm_rejects_an_incorrect_normalized_width(affine: bool) -> None:
    config = QwenRMSNorm.Config()
    config.channels_in = 5
    config.elementwise_affine = affine
    with pytest.raises(RuntimeError, match="normalized_shape"):
        config.make()(torch.ones(2, 3, 7))


@pytest.mark.parametrize("shape", argvalues=[(7, 1, 5), (7, 1, 7)])
def test_qwen_rotation_preserves_partial_and_padded_geometry(
    shape: tuple[int, int, int],
) -> None:
    queries = torch.arange(2 * 7 * 5 * 14).reshape(2, 7, 5, 14).float()
    keys = torch.arange(2 * 7 * 6 * 14).reshape(2, 7, 6, 14).float()
    # RoPE._build_inv_freqs inserts the shared singleton head dimension.
    actual_q, actual_k = rotate_activation_dtype(
        queries,
        k=keys,
        cos=torch.ones(shape),
        sin=torch.zeros(shape),
    )
    assert torch.equal(actual_q, other=queries)
    assert torch.equal(actual_k, other=keys)


@pytest.mark.parametrize("layers", argvalues=[[], [-1, 2], [2, 5]])
def test_saved_state_indices_must_exist(
    tiny_checkpoint: FrozenQwen.Config,
    layers: list[int],
) -> None:
    tiny_checkpoint.layers = layers
    with pytest.raises(ValueError, match="hidden-state indices"):
        tiny_checkpoint.make()._load_model(torch.float32)


def test_mean_pool_excludes_padding_and_keeps_float32_accumulation() -> None:
    hidden = torch.arange(30, dtype=torch.float16).reshape(2, 3, 5)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    actual = mean_pool(hidden, attention_mask=mask)
    assert actual.dtype == torch.float32
    assert torch.equal(
        actual,
        torch.tensor(
            [
                [2.5, 3.5, 4.5, 5.5, 6.5],
                [20, 21, 22, 23, 24],
            ],
        ),
    )


@pytest.mark.parametrize("padding", argvalues=[False, True])
def test_tiny_qwen_preserves_rows_under_length_sorting(
    tiny_encoder: FrozenQwen.Config,
    padding: bool,
) -> None:
    if padding:
        tokenizer = tokenization_auto.AutoTokenizer.from_pretrained(
            tiny_encoder.directory,
            local_files_only=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(tiny_encoder.directory)
    encoder = tiny_encoder.make()
    texts = ["red blue", "green", "red green blue"]
    embeddings = encoder.encode(texts)
    assert embeddings.shape == (3, 4, 7)
    assert embeddings.dtype == np.float32
    assert np.array_equal(embeddings, a2=encoder.encode(texts[::-1])[::-1])


def test_equal_token_lengths_keep_their_input_order(
    tiny_encoder: FrozenQwen.Config,
) -> None:
    encoder = _RowsSeen(tiny_encoder)
    # Twenty rows: past sixteen, NumPy's default sort stops keeping ties in order.
    encoder.encode(["red", "red blue"] * 10)
    assert encoder.seen == [([*range(1, 20, 2), *range(0, 20, 2)], torch.float16)]


def test_retry_recomputes_nonfinite_rows(tiny_encoder: FrozenQwen.Config) -> None:
    encoder = _NaNFirstPass(tiny_encoder)
    result = encoder.encode(["red blue", "green", "red green blue"])
    assert np.isfinite(result).all()
    assert encoder.dtypes == [torch.float16, torch.bfloat16]


def test_extraction_preserves_rows_batches_tokens_precision_and_device(
    tiny_encoder: FrozenQwen.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = tokenization_auto.AutoTokenizer.from_pretrained(
        tiny_encoder.directory,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    calls: list[tuple[Tensor, Tensor, torch.dtype]] = []
    placements: list[str | torch.device] = []
    original_to = BatchEncoding.to

    def move(
        batch: BatchEncoding,
        device: str | torch.device,
        non_blocking: bool = False,
    ) -> BatchEncoding:
        placements.append(device)
        return original_to(batch, device=device, non_blocking=non_blocking)

    def observe(
        model: Qwen3,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        layers: list[int],
    ) -> tuple[Tensor, ...]:
        calls.append(
            (input_ids.clone(), attention_mask.clone(), next(model.parameters()).dtype),
        )
        return selected_hidden_states(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            layers=layers,
        )

    monkeypatch.setattr(BatchEncoding, "to", move)
    monkeypatch.setattr(encoder_module, "selected_hidden_states", observe)
    output = tiny_encoder.make()._encode_rows(
        np.array([2, 0, 1]),
        dtype=torch.float16,
        texts=["red blue green red blue green", "blue red", "green red blue green"],
        tokenizer=tokenizer,
    )
    assert output.shape == (3, 4, 7)
    assert output.dtype == np.float32
    assert placements == [torch.device("cpu"), torch.device("cpu")]
    assert len(calls) == 2
    assert [dtype for _, _, dtype in calls] == [torch.float16, torch.float16]
    assert torch.equal(calls[0][0], torch.tensor([[4, 2, 3, 4, 1], [2, 3, 4, 2, 3]]))
    assert torch.equal(calls[0][1], torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]))
    # _encode_rows emits one remaining row when three rows use batch_size=2.
    assert torch.equal(calls[1][0], torch.tensor([[3, 2]]))
    assert torch.equal(calls[1][1], torch.ones((1, 2), dtype=torch.int64))


def test_retry_cannot_accept_persistent_nonfinite_values(
    tiny_encoder: FrozenQwen.Config,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=r"^Qwen embeddings remain nonfinite after bf16 retry\.$",
    ):
        _AlwaysNaN(tiny_encoder).encode(["red blue", "green"])


def test_unprepared_weights_fail_explicitly(tmp_path: Path) -> None:
    config = FrozenQwen.Config()
    config.directory = tmp_path
    with pytest.raises(FileNotFoundError, match="snapshot missing"):
        config.make().encode(["red blue", "green"])


class _NaNFirstPass(FrozenQwen):
    def __init__(self, config: FrozenQwen.Config) -> None:
        super().__init__(config)
        self.dtypes: list[torch.dtype] = []

    @override
    def _encode_rows(
        self,
        rows: NDArray[np.int64],
        dtype: torch.dtype,
        *,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
    ) -> NDArray[np.float32]:
        self.dtypes.append(dtype)
        output = super()._encode_rows(
            rows,
            dtype=dtype,
            texts=texts,
            tokenizer=tokenizer,
        )
        if dtype == torch.float16:
            output[0] = np.nan
        return output


class _AlwaysNaN(FrozenQwen):
    @override
    def _encode_rows(
        self,
        rows: NDArray[np.int64],
        dtype: torch.dtype,
        *,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
    ) -> NDArray[np.float32]:
        del dtype, texts, tokenizer
        return np.full(
            (len(rows), len(self.config.layers), self.config.channels_hidden),
            fill_value=np.nan,
            dtype=np.float32,
        )


class _RowsSeen(FrozenQwen):
    def __init__(self, config: FrozenQwen.Config) -> None:
        super().__init__(config)
        self.seen: list[tuple[list[int], torch.dtype]] = []

    @override
    def _encode_rows(
        self,
        rows: NDArray[np.int64],
        dtype: torch.dtype,
        *,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
    ) -> NDArray[np.float32]:
        del texts, tokenizer
        self.seen.append((cast("list[int]", rows.tolist()), dtype))
        return np.zeros(
            (len(rows), len(self.config.layers), self.config.channels_hidden),
            dtype=np.float32,
        )


def _checkpoint_state() -> dict[str, Tensor]:
    """Return distinct weights in every tensor, exact in float32 on every host."""
    shapes = {
        "model.embed_tokens.weight": (17, 7),
        "model.norm.weight": (7,),
        "lm_head.weight": (17, 7),
    }
    for layer in range(4):
        prefix = f"model.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (7,),
                f"{prefix}.post_attention_layernorm.weight": (7,),
                f"{prefix}.self_attn.q_proj.weight": (24, 7),
                f"{prefix}.self_attn.k_proj.weight": (12, 7),
                f"{prefix}.self_attn.v_proj.weight": (12, 7),
                f"{prefix}.self_attn.o_proj.weight": (7, 24),
                f"{prefix}.self_attn.q_norm.weight": (6,),
                f"{prefix}.self_attn.k_norm.weight": (6,),
                f"{prefix}.mlp.gate_proj.weight": (11, 7),
                f"{prefix}.mlp.up_proj.weight": (11, 7),
                f"{prefix}.mlp.down_proj.weight": (7, 11),
            },
        )
    state: dict[str, Tensor] = {}
    offset = 0
    for name, shape in shapes.items():
        count = math.prod(shape)
        steps = torch.arange(offset, offset + count)
        offset += count
        wave = (steps * 37 % 101 - 50).double() / 101
        values = 1 + wave / 2 if "norm.weight" in name else wave
        state[name] = values.float().reshape(shape)
    return state


def _no_collection() -> None:
    """Keep interpreter-wide garbage collection outside unit-test arithmetic."""


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
