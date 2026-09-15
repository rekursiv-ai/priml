"""Strict text checkpoint mapping and local loading tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import json

import pytest
import torch

from priml.model.custom_types import has_weight
from priml.model.transformer.qwen3_5 import Qwen35
from priml.model.transformer.qwen3_5_weights import remap_hf_state_dict
from priml.testing.qwen3_5 import hf_config


pytest.importorskip("transformers")


if TYPE_CHECKING:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
else:
    from wrapt import lazy_import

    Qwen3_5TextConfig = lazy_import(
        "transformers.models.qwen3_5.configuration_qwen3_5", "Qwen3_5TextConfig"
    )
    Qwen3_5ForCausalLM = lazy_import(
        "transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5ForCausalLM"
    )


def test_reference_weights_load_strictly() -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    native = Qwen35.Config.from_hf(config)
    mapped = remap_hf_state_dict(reference.state_dict(), native)
    model = native.make()
    model.load_state_dict(mapped, strict=True)
    assert has_weight(model.out_proj)
    assert torch.equal(model.out_proj.weight, reference.lm_head.weight)


def test_missing_head_is_not_discarded() -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    state = reference.state_dict()
    del state["lm_head.weight"]
    with pytest.raises(ValueError, match="lm_head"):
        remap_hf_state_dict(state, Qwen35.Config.from_hf(config), non_text="discard")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "Missing"),
        ("extra", "Unexpected"),
        ("shape", "has shape"),
        ("integer", "floating-point"),
    ],
)
def test_rejects_incomplete_or_malformed_text_weights(
    mutation: str,
    match: str,
) -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    state = reference.state_dict()
    name = "model.layers.0.linear_attn.in_proj_qkv.weight"
    if mutation == "missing":
        del state[name]
    elif mutation == "extra":
        state["model.unexpected.weight"] = torch.zeros(1)
    elif mutation == "shape":
        state[name] = state[name][1:]
    else:
        state[name] = state[name].to(torch.int32)
    with pytest.raises(ValueError, match=match):
        remap_hf_state_dict(state, Qwen35.Config.from_hf(config))


def test_conditional_namespace_and_explicit_nontext_policy() -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    state = {
        name.replace("model.", "model.language_model.", 1): tensor
        for name, tensor in reference.state_dict().items()
    }
    state["model.visual.patch_embed.proj.weight"] = torch.zeros(2, 2)
    native_config = Qwen35.Config.from_hf(config)
    with pytest.raises(ValueError, match="Unexpected"):
        remap_hf_state_dict(state, native_config)
    mapped = remap_hf_state_dict(state, native_config, non_text="discard")
    assert torch.equal(mapped["out_proj.weight"], state["lm_head.weight"])
    state["model.language_model.extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="extra"):
        remap_hf_state_dict(state, native_config, non_text="discard")


def test_local_text_loading_preserves_parameters_and_logits(tmp_path: Path) -> None:
    config = hf_config()
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save(reference.state_dict(), tmp_path / "pytorch_model.bin")
    native_config = Qwen35.Config.from_hf(config)
    expected = native_config.make()
    expected.load_state_dict(remap_hf_state_dict(reference.state_dict(), native_config))
    loaded = Qwen35.load(tmp_path)
    torch.testing.assert_close(
        loaded.state_dict(), expected.state_dict(), rtol=0, atol=0
    )
    tokens = torch.tensor([[1, 2, 3]])
    assert torch.equal(loaded(tokens), expected(tokens))
    assert torch.equal(loaded.hidden_states(tokens), expected.hidden_states(tokens))


def test_tied_head_alias_must_match_the_embedding() -> None:
    config = hf_config()
    config["tie_word_embeddings"] = True
    reference = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**config))
    state = reference.state_dict()
    native_config = Qwen35.Config.from_hf(config)
    mapped = remap_hf_state_dict(state, native_config)
    native_config.make().load_state_dict(mapped, strict=True)
    assert "out_proj.weight" not in mapped
    state["lm_head.weight"] = state["lm_head.weight"] + 1
    with pytest.raises(ValueError, match="Tied"):
        remap_hf_state_dict(state, native_config)
    state["lm_head.weight"] = state["model.embed_tokens.weight"].double()
    with pytest.raises(ValueError, match="dtype"):
        remap_hf_state_dict(state, native_config)
    del state["lm_head.weight"]
    native_config.make().load_state_dict(remap_hf_state_dict(state, native_config))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
