"""Bits per byte: a language-model score that survives a tokenizer change.

Cross-entropy per TOKEN is not comparable across vocabularies -- merging two
tokens into one lowers it without the model predicting anything better. Bits
per byte divides the same total surprise by the UTF-8 LENGTH of what was
predicted instead, so a change to the tokenizer moves both sides of the ratio
and the number stays meaningful.

Special tokens carry zero bytes and drop out of both sums: they mark document
boundaries rather than text, so charging the model for them would score a
formatting convention.

Sums accumulate in float64 and are all-reduced once at ``compute``, so a
distributed run reports the same number as a single process rather than a mean
of per-rank ratios -- which would weight a short final shard equally with a
full one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import math

from configgle import Fig

import torch
import torch.distributed as dist


if TYPE_CHECKING:
    from torch import Tensor


class BitsPerByte:
    """Validation bits per byte, lower being better.

    Consumes the per-token loss a language-model train step emits as its
    evaluation output: ``[B, S]`` nats, aligned with the batch's targets.
    """

    class Config(Fig["BitsPerByte"]):
        """Nothing to configure; the byte table travels with the batch."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """Zero both sums."""
        self.nats = 0.0
        self.bytes = 0

    def update(self, logits: Tensor, **batch: Any) -> None:
        """Accumulate one batch.

        Args:
          logits: ``[B, S]`` per-token cross-entropy in nats. Named for the
            metric protocol; a language-model step reduces to the loss before
            this point, since a full ``[B, S, V]`` logit tensor is far larger
            than the number this needs from it.
          **batch: Must carry ``label`` and ``token_bytes``.

        Raises:
          ValueError: If the per-token loss and the targets disagree in shape,
            which would silently pair each token's loss with another's length.

        """
        per_token = logits.detach().double()
        labels: Tensor = batch["label"].detach().to(torch.int64)
        token_bytes: Tensor = batch["token_bytes"].detach().to(labels.device)
        if per_token.shape != labels.shape:
            raise ValueError(
                f"per-token loss has shape {tuple(per_token.shape)} but the "
                f"targets have {tuple(labels.shape)}.",
            )
        lengths = token_bytes[labels.reshape(-1)]
        counted = lengths > 0
        self.nats += float(per_token.reshape(-1)[counted].sum().item())
        self.bytes += int(lengths[counted].sum().item())

    def compute(self) -> dict[str, float]:
        """Return bits per byte, summed across ranks first."""
        totals = torch.tensor([self.nats, float(self.bytes)], dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            # NCCL reduces only CUDA tensors; gloo only CPU ones. Move for the
            # former and come back, so ``.tolist()`` works either way.
            if dist.get_backend() != "gloo":
                totals = totals.to(torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            totals = totals.cpu()
        nats, counted = totals.tolist()
        return {"bpb": nats / (math.log(2) * counted) if counted else 0.0}

    def state_dict(self) -> dict[str, Any]:
        """Return the accumulated sums."""
        return {"nats": self.nats, "bytes": self.bytes}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore sums produced by :meth:`state_dict`."""
        self.nats = state_dict.get("nats", 0.0)
        self.bytes = state_dict.get("bytes", 0)
