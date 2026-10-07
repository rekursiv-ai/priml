"""PriML's DPPO update for TMax rollout data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

import json
import math

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor
from torch.distributed.checkpoint.state_dict import (  # ty: ignore[unresolved-import] -- PyTorch ships this runtime module without a matching ty stub.
    StateDictOptions,
    ValueType,
    set_model_state_dict,
)

import torch

from priml import hub
from priml.baselines.tmax.data import data_parallel_shape
from priml.baselines.tmax.objective import (
    DivergenceType,
    dppo_mask,
    dppo_token_loss,
    importance_ratio,
    mask_logprobs,
    masked_mean,
)
from priml.baselines.tmax.scoring import (
    LogitModel,
    response_logprobs,
)
from priml.model.attention.gated_self_attention import GatedSelfAttention
from priml.model.attention.kernel import SdpaFused
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3_5 import Qwen35
from priml.model.transformer.qwen3_5_weights import export_hf_state_dict
from priml.paths import resolve_working_dir
from priml.runtime import is_rank_zero
from priml.train.train_step import TrainStep


_HF_AUXILIARY_FILE_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
    "generation_config.json",
)
"""Text-model files a Tmax checkpoint may carry into an HF export."""


if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor

    from priml.train.custom_types import TrainStepOutput
else:
    from wrapt import lazy_import

    DTensor = lazy_import("torch.distributed.tensor", "DTensor")


def _hf_projection_order(config: Qwen35.Config) -> Qwen35.Config:
    """Score with the separate gate/up matmuls that Hugging Face Qwen3.5 uses.

    The released trainer runs ``transformers``' Qwen3.5, and PriML's HF parity
    tests pin that model's logits and gradients against the native one *only*
    when the gate and up projections are separate matmuls
    (``priml.testing.qwen3_5.native_qwen35_config``). The fused projection is
    PriML's native preference -- same values, one wider matmul -- but it is a
    different kernel order, which would put the training step's logits outside
    what the parity tests verify. The parameter layout is unchanged (the split
    re-reads the fused weight), so checkpoints and goldens load as before.
    """
    if not isinstance(config.block, list):
        raise TypeError("Qwen3.5 requires an explicit block list.")
    for block in config.block:
        if not isinstance(block, TransformerBlock.Config):
            raise TypeError("Qwen3.5 requires TransformerBlock configs.")
        if not isinstance(block.ffn, SwiGLU.Config):
            raise TypeError("Qwen3.5 requires SwiGLU feed-forward blocks.")
        block.ffn.split_gate_projection = True
    return config


def _cluster_attention(config: Qwen35.Config) -> Qwen35.Config:
    """Score the cluster recipe with fused attention, as the released trainer does.

    ``GatedSelfAttention.Config`` defaults to ``SdpaNaive`` because that is the
    only kernel PriML's Hugging Face parity tests can pin: they build the
    reference model with ``attn_implementation="eager"``, and only an unfused
    matmul+softmax produces the same primitive ops (see
    ``priml.testing.bfb``). The released 4B trainer never runs eager attention --
    ``model_utils.detect_attn_implementation()`` selects the fastest available
    backend, flash attention on the H100s the recipe names.

    That is not a stylistic difference at this context length. ``SdpaNaive``
    materializes ``q @ k^T`` before the softmax, so one full-attention layer over
    a 67,584-token packed row holds a ``[heads, 67584, 67584]`` score tensor --
    ~145 GB in bfloat16 for Qwen 3.5-4B's 32 query heads -- before any mask or
    softmax is applied. The row cannot be scored at all, which is why the
    released launcher passes ``--sequence_parallel_size 4`` and a fused kernel.
    ``SdpaFused`` dispatches to the same flash/memory-efficient backends and
    never materializes the scores.

    Applied only where the staged checkpoint supplies the architecture, so the
    CPU-sized smoke policy and every HF parity golden keep the eager-equivalent
    kernel their evidence was generated with.

    Args:
      config: The checkpoint-derived Qwen3.5 configuration, already in HF
        projection order.

    Returns:
      config: The same configuration with full-attention layers fused.

    Raises:
      TypeError: ``config`` does not carry an explicit block list, or a block
        is not a transformer block with gated self-attention.

    """
    if not isinstance(config.block, list):
        raise TypeError("Qwen3.5 requires an explicit block list.")
    for block in config.block:
        if not isinstance(block, TransformerBlock.Config):
            raise TypeError("Qwen3.5 requires TransformerBlock configs.")
        if isinstance(block.attn, GatedSelfAttention.Config):
            block.attn.attn_kernel = SdpaFused.Config()
    return config


def _text_only_config(metadata: Mapping[str, object]) -> dict[str, object]:
    """Derive the text-only ``config.json`` a text export carries.

    The released source checkpoint is the multimodal form --
    ``Qwen3_5ForConditionalGeneration`` wrapping a ``text_config`` -- but
    upstream's own export never carries it: its trainer holds the text model
    ``AutoModelForCausalLM`` maps that config onto (``Qwen3_5ForCausalLM``,
    whose ``_keys_to_ignore_on_load_unexpected`` drops the vision tower at
    load), and that model's ``save_pretrained`` writes the extracted text
    config. This derives the same config: the nested ``text_config`` fields,
    the wrapper's tied-head flag propagated the way transformers' text-config
    extraction and :func:`Qwen35.Config.from_hf` both read it, and the text
    architecture ``save_pretrained`` would record.

    Args:
      metadata: The staged checkpoint's ``config.json`` contents.

    Returns:
      config: The text-only ``config.json`` contents for the export.

    Raises:
      TypeError: The wrapper carries no ``text_config`` mapping, so no text
        architecture can be derived from it.

    """
    nested = metadata.get("text_config")
    if not isinstance(nested, Mapping):
        raise TypeError(
            "Checkpoint export found a conditional-generation config.json "
            "with no text_config to derive a text-only config from; set "
            "step.model_path to a text-only source.",
        )
    # JSON objects carry string keys, so the staged checkpoint's nested
    # mapping is the string-keyed config the export writes.
    result = dict(cast("Mapping[str, object]", nested))
    if "tie_word_embeddings" in metadata:
        result["tie_word_embeddings"] = metadata["tie_word_embeddings"]
    result["architectures"] = ["Qwen3_5ForCausalLM"]
    return result


def _tokenizer_pad_token_id(checkpoint: Path) -> int:
    """Resolve the staged tokenizer's pad id, as upstream's trainer does.

    Upstream takes ``tokenizer.pad_token_id`` from the tokenizer built off
    the staged checkpoint, and the released source's ``config.json`` carries
    no ``pad_token_id`` of its own -- so a config-only reader has nothing to
    fall back on. The staged ``tokenizer_config.json`` names the pad token
    and maps every special token to its id, which resolves the same id
    without building the tokenizer.

    Args:
      checkpoint: The staged checkpoint directory.

    Returns:
      pad_token_id: The id the staged tokenizer pads with.

    Raises:
      ValueError: The staged tokenizer cannot be read, names no pad token, or
        maps the pad token to no id.
      TypeError: The staged tokenizer config is not the object this reads.

    """
    path = checkpoint / "tokenizer_config.json"
    if not path.is_file():
        raise ValueError(
            f"{path} is missing, so the pad id cannot be resolved from the "
            "staged tokenizer; set step.pad_token_id explicitly.",
        )
    try:
        payload = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} could not be read: {error}.") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain the tokenizer config object.")
    tokenizer_config = cast("dict[str, object]", payload)
    token = tokenizer_config.get("pad_token")
    if isinstance(token, Mapping):
        token = cast("dict[str, object]", token).get("content")
    added = tokenizer_config.get("added_tokens_decoder")
    if isinstance(token, str) and isinstance(added, Mapping):
        # The decoder maps every special token's id to its record, which is
        # the same resolution as building the tokenizer.
        decoder = cast("Mapping[str, Mapping[str, object]]", added)
        for key, value in decoder.items():
            if value.get("content") == token:
                return int(key)
    raise ValueError(
        f"{path} names no pad token with a resolvable id; set "
        "step.pad_token_id explicitly.",
    )


def _hf_auxiliary_files(checkpoint: Path) -> tuple[Path, ...]:
    """Select the known text-model files carried into an HF export."""
    return tuple(
        path
        for name in _HF_AUXILIARY_FILE_NAMES
        if (path := checkpoint / name).is_file()
    )


def tiny_qwen35_config(
    *,
    vocab_size: int = 32,
    hidden_size: int = 16,
    tie_word_embeddings: bool = False,
) -> Qwen35.Config:
    """Return a real, CPU-sized Qwen3.5 configuration for ``exp_smoke``.

    The smoke model is the native Qwen3.5 implementation, not a fake language
    model. Its dimensions deliberately match the shared fixture PriML's HF
    parity tests use, so a smoke update exercises the actual hybrid
    linear/full-attention stack while remaining suitable for a unit test.
    ``hidden_size`` only scales that fixture down (goldens store the whole
    state twice, under a 32 KiB ceiling); it never changes the recipe.
    ``tie_word_embeddings`` selects the tied head the released 4B checkpoint
    uses (``proj_out`` borrows the embedding's weight) or the plain head the
    untied parity fixture uses, which is the choice the scoring path tests in
    both forms.
    """
    return _hf_projection_order(
        Qwen35.Config.from_hf(
            {
                "model_type": "qwen3_5_text",
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "intermediate_size": hidden_size * 3 // 2,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": hidden_size // 2,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": hidden_size // 4,
                "linear_value_head_dim": hidden_size // 4,
                "linear_conv_kernel_dim": 4,
                "layer_types": ["linear_attention", "full_attention"],
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                    "partial_rotary_factor": 0.5,
                    "mrope_section": [1, 1, 0],
                },
                "hidden_act": "silu",
                "rms_norm_eps": 1e-6,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "tie_word_embeddings": tie_word_embeddings,
            },
        ),
    )


def qwen35_4b_config() -> Qwen35.Config:
    """Return the released Qwen3.5-4B text architecture as printable config."""
    return _cluster_attention(
        _hf_projection_order(
            Qwen35.Config.from_hf(
                {
                    "model_type": "qwen3_5_text",
                    "vocab_size": 248_320,
                    "hidden_size": 2_560,
                    "intermediate_size": 9_216,
                    "num_hidden_layers": 32,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 32,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                    "linear_conv_kernel_dim": 4,
                    "full_attention_interval": 4,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 10_000_000.0,
                        "partial_rotary_factor": 0.25,
                        "mrope_section": [11, 11, 10],
                        "mrope_interleaved": True,
                    },
                    "hidden_act": "silu",
                    "rms_norm_eps": 1e-6,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "attn_output_gate": True,
                    "initializer_range": 0.02,
                    "max_position_embeddings": 262_144,
                    "tie_word_embeddings": True,
                },
            ),
        ),
    )


class TMaxDPPOTrainStep(TrainStep):
    """Teacher-forced Qwen3.5 scoring plus a DPPO update faithful to upstream."""

    class Config(
        Makes["TMaxDPPOTrainStep"],
        TrainStep.Config[Qwen35.Config],
        kw_only=True,
    ):
        """Model, optimizer, and DPPO trust-region settings."""

        model: Qwen35.Config = field(default_factory=tiny_qwen35_config)
        """Native Qwen3.5 policy architecture, sized for checkpoint-free runs.

        With ``model_path`` set the staged checkpoint's ``config.json`` IS the
        architecture and this field is replaced by it, so there is exactly one
        source of truth; the checkpoint below also supplies ``pad_token_id``.
        """

        model_path: Path | str | None = None
        """Local Hugging Face checkpoint directory for the cluster recipe."""

        base_dir: Path | str | None = None
        """Run directory supplied by the experiment."""

        working_dir: Path | str = "/"
        """This step's directory inside the experiment run."""

        export_dir: Path | str | None = None
        """Directory the trained policy is exported to at each epoch boundary.

        ``None`` never writes. The export carries the staged checkpoint's
        text ``config.json`` (copied verbatim for a text-only source, or
        derived from the released multimodal source's ``text_config``
        exactly the way upstream's own text-model save writes it), its
        tokenizer files, and ``model.safetensors`` in Hugging Face names --
        the format the rollout collector and the Terminal-Bench evaluation
        serve. The published run is a single finite epoch: its live dataset
        runs out after the configured number of updates, so the export runs
        at a real epoch boundary.
        """

        @override
        def finalize(self) -> Self:
            """Resolve working and export paths before building the step."""
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            if self.export_dir is not None:
                self.export_dir = resolve_working_dir(
                    self.working_dir,
                    self.export_dir,
                )
            return super().finalize()

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: PartialConfig(
                torch.optim.AdamW,
                lr=1e-6,
                betas=(0.9, 0.999),
                weight_decay=0.0,
            ),
        )
        """AdamW at the released TMax learning rate."""

        divergence_type: DivergenceType = "tv"
        """Binary DPPO divergence; the released 4B recipe uses total variation."""

        divergence_threshold: float = 0.1
        """Trust-region radius from the released 4B recipe."""

        temperature: float = 1.0
        """Teacher-forcing temperature, matching rollout sampling."""

        head_chunk_size: int | None = None
        """Scored positions per language-model-head chunk; ``None`` disables.

        The released 4B launcher fuses the head into a chunked loss
        (``--use_liger_grpo_loss --liger_grpo_loss_chunk_size 8``) so a packed
        row's ``[tokens, vocabulary]`` logits never exist whole; this is how
        PriML enforces the same limit (see
        :func:`priml.baselines.tmax.scoring.score_hidden_labels`). CPU parity
        paths leave it unset.
        """

        fp32_head: bool = False
        """Project hidden states and the language-model head in fp32."""

        pad_token_id: int | None = None
        """Id replaced by token 0 before gathering labels.

        ``None`` resolves from the checkpoint's ``config.json`` when
        ``model_path`` is set, else from the staged tokenizer's pad token --
        where upstream's trainer resolves it, and the only place the released
        source names one -- else token 0 for checkpoint-free synthetic runs.
        The data layer inherits the resolved value so packing and scoring
        remove the same token.
        """

        gradient_clip_norm: float = 1.0
        """The upstream trainer's global gradient clipping bound.

        The released 4B launcher passes no ``--max_grad_norm``, so the parser's
        default of 1.0 applies (``grpo_fast.py``'s ``set_defaults``). Set here
        rather than inherited: ``TrainStep.Config``'s own default is unclipped,
        so a run that set nothing would silently train without the clipping
        the recipe used.
        """

        compile: (
            Makeable[Callable[[Callable[..., object]], Callable[..., object]]] | None
        ) = None
        """Keep smoke and parity runs eager; cluster launch may opt into compile."""

    config: Config

    def __init__(self, config: Config) -> None:
        """Validate DPPO settings, resolve the checkpoint, build the step."""
        if config.temperature <= 0.0:
            raise ValueError("TMax temperature must be positive.")
        if config.divergence_threshold <= 0.0:
            raise ValueError("TMax DPPO divergence threshold must be positive.")
        if config.pad_token_id is not None and config.pad_token_id < 0:
            raise ValueError("TMax pad_token_id must be non-negative.")
        # One optimizer update per train_step, sized by records_per_update;
        # loop-level microbatch accumulation would be silently ignored here.
        if config.accumulate_grad_batches != 1:
            raise ValueError(
                "TMax DPPO takes exactly one optimizer update per train_step; "
                "size the update with records_per_update, not "
                "accumulate_grad_batches.",
            )
        checkpoint = Path(config.model_path) if config.model_path else None
        metadata: dict[str, object] | None = None
        if checkpoint is not None:
            metadata = _checkpoint_metadata(checkpoint)
            staged_model = _cluster_attention(
                _hf_projection_order(Qwen35.Config.from_hf(metadata)),
            )
            # The published experiment spells the full architecture in its
            # config. Refuse a differently staged checkpoint instead of
            # silently changing the printed experiment at runtime. The tiny
            # default remains a convenient fallback for tests and one-off
            # tools that did not declare an architecture.
            declared = config.model.copy_tree().finalize().pformat(finalize=False)
            default = (
                tiny_qwen35_config().copy_tree().finalize().pformat(finalize=False)
            )
            staged = staged_model.copy_tree().finalize().pformat(finalize=False)
            if declared != default:
                if declared != staged:
                    raise ValueError(
                        "The staged checkpoint architecture does not match "
                        "step.model; update the experiment config or stage the "
                        "checkpoint it declares.",
                    )
            else:
                config.model = staged_model
            if config.pad_token_id is None:
                if "pad_token_id" in metadata:
                    config.pad_token_id = int(cast(int, metadata["pad_token_id"]))
                else:
                    # The released source's config.json carries no pad id;
                    # upstream's trainer resolves the tokenizer's, and so
                    # does the staged tokenizer here.
                    config.pad_token_id = _tokenizer_pad_token_id(checkpoint)
        elif config.pad_token_id is None:
            config.pad_token_id = 0
        super().__init__(config)
        self.config = config
        self._dp_world = data_parallel_shape()[1]
        # Export re-reads both at the epoch boundary: the staged checkpoint
        # owns the config and the known tokenizer files the export carries.
        self._checkpoint = checkpoint
        self._checkpoint_metadata = metadata
        if checkpoint is not None:
            source = Qwen35.load(checkpoint, device=self.device, non_text="discard")
            _load_checkpoint_state(self.model, source.state_dict())
            del source

    @property
    def pad_token_id(self) -> int:
        """The resolved pad id; the data layer reuses it for packing."""
        return cast(int, self.config.pad_token_id)

    @override
    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Backprop one packed row at a time, then take the single update.

        Each row's backward frees its graph before the next forward, so peak
        activation memory is ONE row -- upstream runs one packed row per device
        per micro-batch -- instead of the whole update's rows.
        """
        rows = _rows(preprocessed_batch) or (preprocessed_batch,)
        token_count = _update_token_count(preprocessed_batch, rows)
        results: list[TrainStepOutput] = []
        for row in rows:
            result = self._loss(row, token_count=token_count, evaluate=False)
            # FSDP reduces gradients as a MEAN over data-parallel ranks while
            # each row's loss is already divided by the GLOBAL token count, so
            # scaling by the DP world size makes the reduction the true token
            # mean over the whole update: (1/W) * sum_r W * S_r / N.
            (result["loss"] * self._dp_world).backward()
            results.append(result)
        self.step()
        return _world_scaled(
            _merge_results(results, token_count=token_count),
            self._dp_world,
        )

    @override
    def train_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Compute the update's DPPO scalar without changing model weights."""
        rows = _rows(preprocessed_batch) or (preprocessed_batch,)
        token_count = _update_token_count(preprocessed_batch, rows)
        results = [
            self._loss(row, token_count=token_count, evaluate=False) for row in rows
        ]
        return _merge_results(results, token_count=token_count)

    @override
    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Compute one detached DPPO diagnostic loss per update."""
        with torch.inference_mode():
            rows = _rows(preprocessed_batch) or (preprocessed_batch,)
            token_count = _update_token_count(preprocessed_batch, rows)
            results = [
                self._loss(row, token_count=token_count, evaluate=True) for row in rows
            ]
            return _merge_results(results, token_count=token_count)

    @override
    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Move rows to the device and record per-row token counts on the host.

        Counting before the move keeps the update's denominator free of the
        per-row device sync a hot-loop ``.item()`` would cost.
        """
        rows = _rows(batch)
        if rows is None:
            return super().preprocess_batch(batch)
        move = super().preprocess_batch
        preprocessed = move(
            {key: value for key, value in batch.items() if key != "rows"},
        )
        preprocessed["rows"] = tuple(
            move(
                {
                    **row,
                    "row_response_tokens": int(
                        _tensor(row, "response_mask")[1:].bool().sum(),
                    ),
                },
            )
            for row in rows
        )
        return preprocessed

    def hf_state_dict(self) -> dict[str, Tensor]:
        """Gather this policy's weights in Hugging Face names and layout.

        Collective: ``full_tensor()`` all-gathers each sharded parameter, so
        EVERY rank must call this, and each ends up holding the whole policy.
        That is upstream's ``gather_whole_model`` default rather than a new
        cost, but it is why this call is not free.

        The conversion is the one the checkpoint export uses, which the
        weight-parity tests pin bit for bit. Both the export and the rollout
        engines' weight sync read it, so a served policy and a written
        checkpoint cannot disagree about names or layout.

        Returns:
          state: Hugging Face parameter names mapped to full tensors.

        """
        return export_hf_state_dict(
            {
                name: value.full_tensor() if isinstance(value, DTensor) else value
                for name, value in self.model.state_dict().items()
            },
            self.config.model,
        )

    def export_checkpoint(self, directory: Path | str) -> Path | None:
        """Write the trained policy as a Hugging Face text checkpoint.

        Every rank participates -- gathering FSDP shards is collective --
        but only rank zero serializes, so the directory is written once.

        Args:
          directory: Destination, created if absent.

        Returns:
          directory: The written checkpoint root on rank zero; ``None``
            elsewhere. Reloads bit-for-bit via :meth:`Qwen35.load`.

        Raises:
          ValueError: No staged source carries ``config.json`` and the
            tokenizer, or the staged config declares no text architecture
            to export.

        """
        if self._checkpoint_metadata is None or self._checkpoint is None:
            raise ValueError(
                "Checkpoint export needs the staged source (step.model_path) "
                "for config.json and the tokenizer; a checkpoint-free model "
                "has neither.",
            )
        metadata: dict[str, object] = self._checkpoint_metadata
        if "text_config" in metadata:
            # The staged source is the released multimodal form; the
            # export carries the text-only config upstream's own text-model
            # save writes, derived rather than rejected: the trainer behind
            # the released recipe holds the text model, and its save is what
            # the rollout collector and the evaluation serve.
            metadata = _text_only_config(metadata)
        state = self.hf_state_dict()
        if not is_rank_zero():
            return None
        return hub.save_hf_checkpoint(
            directory,
            metadata,
            state,
            auxiliary_files=_hf_auxiliary_files(self._checkpoint),
        )

    @override
    def on_epoch_end(self) -> None:
        """Flush a partial accumulation, then export the trained policy.

        The published run is a single epoch, so its one boundary is the end
        of training; with ``export_dir`` unset, this hook only flushes.
        """
        super().on_epoch_end()
        if self.config.export_dir is None:
            return
        self.export_checkpoint(self.config.export_dir)

    def _loss(
        self,
        batch: Mapping[str, object],
        *,
        token_count: float,
        evaluate: bool,
    ) -> TrainStepOutput:
        """Build the global-normalized DPPO loss and diagnostics for one row."""
        tokens = _tensor(batch, "query_responses").long()
        positions = _tensor(batch, "position_ids").long()
        segments = _tensor(batch, "attention_mask").long()
        response_mask = _tensor(batch, "response_mask").bool()
        behavior = _tensor(batch, "vllm_logprobs").float()
        advantages = _tensor(batch, "advantages").float()
        if tokens.ndim != 1:
            raise ValueError(f"Expected one packed row, got tokens {tokens.shape}.")
        if not all(
            value.shape == tokens.shape
            for value in (
                positions,
                segments,
                response_mask,
                behavior,
                advantages,
            )
        ):
            raise ValueError("Packed row tensors must all have the same shape.")

        forward: LogitModel = cast(
            LogitModel,
            self.call_eval if evaluate else self.__call__,
        )
        current = response_logprobs(
            forward,
            tokens=tokens,
            segments=segments,
            positions=positions,
            pad_token_id=self.pad_token_id,
            temperature=self.config.temperature,
            head_chunk_size=self.config.head_chunk_size,
            fp32_head=self.config.fp32_head,
        ).squeeze(0)

        # TMax's trainer aligns all packed metadata with the shifted model
        # outputs. The first token of each segment is prompt context, so the
        # prediction of it -- made from the previous segment's last token --
        # stays outside the response mask.
        shifted_mask = response_mask[1:]
        shifted_behavior = mask_logprobs(behavior[1:].unsqueeze(0), shifted_mask[None])[
            0
        ]
        shifted_advantages = advantages[1:]
        shifted_current = torch.nan_to_num(
            current,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        ratio = importance_ratio(shifted_current, shifted_behavior)
        policy_mask, divergence = dppo_mask(
            new_logprobs=shifted_current,
            behavior_logprobs=shifted_behavior,
            advantages=shifted_advantages,
            ratio=ratio,
            response_mask=shifted_mask,
            divergence_threshold=self.config.divergence_threshold,
            divergence_type=self.config.divergence_type,
        )
        per_token = dppo_token_loss(
            advantages=shifted_advantages,
            ratio=ratio,
            policy_mask=policy_mask,
        )
        scalar = masked_mean(per_token, shifted_mask, denominator=token_count)
        # A 0-dim share of the update's token mean, NOT the per-token vector:
        # the training loop reads ``loss.mean()``, and averaging a padded
        # per-token tensor would count the masked-out positions.
        valid = shifted_mask
        metrics: dict[str, float | Tensor] = {
            "dppo_loss": scalar.detach(),
            "ratio_mean": _row_mean(ratio, valid),
            "divergence_mean": _row_mean(divergence, valid),
            # float32 so the loop can stack it with the float loss scalars.
            "response_tokens": valid.sum(dtype=torch.float32),
        }
        return {
            "loss": scalar,
            "model": current.detach(),
            "metrics": metrics,
        }


def _checkpoint_metadata(checkpoint: Path) -> dict[str, object]:
    """Read and narrow one Hugging Face checkpoint's ``config.json``."""
    metadata_path = checkpoint / "config.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"TMax Qwen checkpoint config not found: {metadata_path}",
        )
    metadata = cast(object, json.loads(metadata_path.read_text(encoding="utf-8")))
    if not isinstance(metadata, dict):
        raise TypeError("Qwen checkpoint config.json must contain an object.")
    return cast(dict[str, object], metadata)


def _load_checkpoint_state(model: torch.nn.Module, state: dict[str, Tensor]) -> None:
    """Load full tensors into either a local or FSDP model."""
    if any(isinstance(value, DTensor) for value in model.state_dict().values()):
        set_model_state_dict(
            model,
            cast("dict[str, ValueType]", state),
            options=StateDictOptions(full_state_dict=True, strict=True),
        )
        return
    model.load_state_dict(state, strict=True)


def _tensor(batch: Mapping[str, object], name: str) -> Tensor:
    """Read one tensor field from a packed-row batch."""
    value = batch[name]
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    return value


def _rows(batch: Mapping[str, object]) -> tuple[Mapping[str, object], ...] | None:
    """Return a variable-length row batch, if the data layer supplied one."""
    value = batch.get("rows")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("rows must be a non-empty list or tuple.")
    row_values = cast(list[object] | tuple[object, ...], value)
    if not all(isinstance(row, Mapping) for row in row_values):
        raise TypeError("rows must contain mapping objects.")
    rows = cast(
        list[Mapping[str, object]] | tuple[Mapping[str, object], ...],
        row_values,
    )
    return tuple(rows)


def _row_response_tokens(row: Mapping[str, object]) -> int:
    """Read the host-side count ``preprocess_batch`` recorded, or recount."""
    value = row.get("row_response_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(_tensor(row, "response_mask")[1:].bool().sum())


def _global_token_count(batch: Mapping[str, object]) -> float | None:
    """Read the dataset-reported update-wide denominator, when present."""
    value = batch.get("response_token_count")
    if value is None:
        return None
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("response_token_count must be scalar.")
        count = float(value.item())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        count = float(value)
    else:
        raise TypeError("response_token_count must be a scalar number.")
    if not math.isfinite(count) or count <= 0.0:
        raise ValueError(
            f"response_token_count must be positive and finite, got {count}.",
        )
    return count


def _update_token_count(
    batch: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> float:
    """Return the update's GLOBAL shifted-response-token denominator.

    The dataset packs the whole update before slicing rows across ranks, so
    its count already spans the data-parallel world; without that count the
    batch is single-process and its own rows ARE the whole update.
    """
    global_count = _global_token_count(batch)
    if global_count is not None:
        return global_count
    local = float(sum(_row_response_tokens(row) for row in rows))
    if local <= 0.0:
        raise ValueError("A DPPO update needs at least one response token.")
    return local


def _row_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``values`` over ``mask``; an empty mask reports zero.

    Kept on device -- no ``.item()`` -- so the row's hot loop never syncs.
    """
    numerator = (values * mask).sum().detach()
    return numerator / mask.sum(dtype=torch.float32).clamp(min=1.0)


def _row_metrics(result: TrainStepOutput) -> dict[str, float | Tensor]:
    """Read a row's metrics -- the TypedDict marks the key optional."""
    metrics = result.get("metrics")
    if metrics is None:
        raise ValueError("A DPPO row result must carry its metrics.")
    return metrics


def _merge_results(
    results: list[TrainStepOutput],
    *,
    token_count: float,
) -> TrainStepOutput:
    """Combine per-row diagnostics into the update's single numbers.

    SUMS, not means: each row's loss is already divided by the update's global
    token count, so the sum is this rank's share of the DPPO objective.
    Ratio and divergence combine token-weighted over that same denominator.
    """
    if not results:
        raise ValueError("Cannot merge an empty DPPO result list.")
    losses = [result["loss"] for result in results]
    tokens = [
        cast(Tensor, _row_metrics(result)["response_tokens"]) for result in results
    ]
    total_tokens = torch.stack(tokens).sum()
    metrics: dict[str, float | Tensor] = {
        "dppo_loss": torch.stack(
            [cast(Tensor, _row_metrics(result)["dppo_loss"]) for result in results],
        ).sum(),
        "response_tokens": total_tokens,
    }
    for name in ("ratio_mean", "divergence_mean"):
        means = torch.stack(
            [cast(Tensor, _row_metrics(result)[name]) for result in results],
        )
        metrics[name] = (means * torch.stack(tokens)).sum() / token_count
    return {
        "loss": torch.stack(losses).sum(),
        "model": torch.cat(
            [result["model"].reshape(-1) for result in results],
        ),
        "metrics": metrics,
    }


def _world_scaled(result: TrainStepOutput, world: int) -> TrainStepOutput:
    """Pre-scale the update's loss and metrics by the DP world size.

    The training loop all-reduces these as a MEAN over ranks, so the world mean
    of ``world * (this rank's share of the global token mean)`` is exactly the
    DPPO loss of the whole update -- the same correction upstream applies
    before its own backward.
    """
    metrics = _row_metrics(result)
    return {
        "loss": result["loss"] * world,
        "model": result["model"],
        "metrics": {name: value * world for name, value in metrics.items()},
    }
