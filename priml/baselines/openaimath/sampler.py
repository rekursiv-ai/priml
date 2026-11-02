"""Sample Lean actions from the rollout policy's own vLLM server.

The server is OpenAI-compatible and is served from the checkpoint the learner is
updating, so the policy that produced a transcript is the policy the update
scores. This module is the only place that talks to it.

It exists because the learner needs more than text: DPPO's ratio is anchored on
the behavior logprobs of the exact tokens that were sampled, so the transcript
must carry token ids and their logprobs. The OpenAI-compatible API reports
decoded pieces by default, and a piece cannot be turned back into its id: a
byte-level token such as either half of Lean's ``⟨`` decodes to ``�``, which
encodes to a different, valid token. So every request asks vLLM to report the
sampled ids themselves (``return_tokens_as_token_ids``), and a response that
does not carry them is an error rather than something to reconstruct.
"""

from __future__ import annotations

# ruff: noqa: S310 -- Every URL here is this baseline's own configured endpoint.
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
# Two edges are untyped: the rollout server's HTTP body (urllib returns Any)
# and the transformers tokenizer's stubbed members. Every value crossing them
# is narrowed before use, as in the TMax baseline's live.py.
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, cast

import asyncio
import json
import re
import urllib.error
import urllib.request

from priml.baselines.openaimath.episode import ChatFormat, SampledTokens


if TYPE_CHECKING:
    from pathlib import Path


COMPLETIONS_SUFFIX: Final = "/v1/completions"
"""The completions route of an OpenAI-compatible server.

Completions rather than chat completions, for one reason that decides the
architecture: vLLM accepts a list of token ids as the prompt here, so the
transcript the learner scores is built from ids the learner itself supplied, not
from text re-tokenized by the server.
"""

TOKEN_ID_PIECE: Final = re.compile(r"token_id:(\d+)")
"""How vLLM reports a sampled token under ``return_tokens_as_token_ids``."""

REASONING_BLOCK: Final = ("<think>", "</think>")
"""The reasoning markers of the Qwen3 chat templates this baseline serves.

When the template's generation prompt opens the first, a reply's directives are
read only after the second.
"""

_SENTINELS: Final = ("PRIMLUSERSENTINEL", "PRIMLREPLYSENTINEL", "PRIMLNEXTSENTINEL")
"""Message contents that cannot occur in a template, used to split its output."""


class TransformersTokenizer:
    """The policy checkpoint's own tokenizer.

    ``transformers`` is imported inside ``__init__`` rather than at module scope
    so the learner's unit tests, which use a synthetic tokenizer, never import
    it.
    """

    def __init__(self, path: str | Path) -> None:
        """Load the tokenizer stored beside a released or exported checkpoint."""
        from transformers import AutoTokenizer  # noqa: PLC0415, I001 -- The adapter keeps optional model dependencies off module import.

        self.path = str(path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.path)

    def chat_format(self) -> ChatFormat:
        """Render the checkpoint's own chat template into episode framing.

        The template is rendered twice with sentinel contents, once for a first
        user message and once for a user, assistant, user exchange, and split on
        the sentinels. That reads the framing off the checkpoint rather than
        restating it, so a policy with a different template is framed by its own.

        Returns:
          chat_format: The framing pieces.

        Raises:
          ValueError: The template does not have the single-stream shape the
            episode appends to, or its end-of-turn is not one token.

        """
        user, reply, after = _SENTINELS
        first = self._render([{"role": "user", "content": user}])
        prompt_prefix, found, generation_prefix = first.partition(user)
        exchange = self._render(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": reply},
                {"role": "user", "content": after},
            ],
        )
        end_of_turn = self._tokenizer.eos_token or ""
        between = exchange.partition(reply)[2].partition(after)[0]
        if (
            not found
            or not end_of_turn
            or not exchange.startswith(prompt_prefix)
            or not exchange.endswith(after + generation_prefix)
            or not between.startswith(end_of_turn)
        ):
            raise ValueError(
                f"The chat template at {self.path} does not frame a conversation "
                "as one appendable stream; set the episode framing explicitly.",
            )
        if len(self.encode(end_of_turn)) != 1:
            raise ValueError(
                f"End-of-turn {end_of_turn!r} is not one token in {self.path}.",
            )
        opens, closes = REASONING_BLOCK
        return ChatFormat(
            prompt_prefix=prompt_prefix,
            generation_prefix=generation_prefix,
            end_of_turn=end_of_turn,
            observation_prefix=between[len(end_of_turn) :],
            reasoning_end=closes if opens in generation_prefix else "",
        )

    def _render(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Render messages as text, ending with the opening of a reply."""
        return str(
            self._tokenizer.apply_chat_template(
                [dict(message) for message in messages],
                tokenize=False,
                add_generation_prompt=True,
            ),
        )

    def encode(self, text: str) -> list[int]:
        """Encode text without adding special tokens.

        The prompt is one continuous stream: the instruction, the action
        protocol, then the episode's own turns. Inserting BOS or EOS markers
        around the instruction would put tokens in the middle of that stream
        that the policy never chose and that the parser would have to know to
        skip.

        Args:
          text: Text to encode.

        Returns:
          tokens: Token ids.

        """
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids without dropping special tokens.

        Special tokens are kept so that parsing sees exactly what the policy
        emitted; a token the model used as a delimiter must not silently
        disappear from the text the action parser reads.

        Args:
          token_ids: Token ids.

        Returns:
          text: Decoded text.

        """
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)


def completion_request(  # noqa: PLR0917 -- One field per sampling parameter the request pins.
    model: str,
    prompt_tokens: Sequence[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop_token_ids: Sequence[int],
    seed: int | None = None,
) -> dict[str, object]:
    """Build one completions request body.

    The stop ids are sent explicitly rather than left to the server, because
    the server's default comes from checkpoint metadata: Qwen3.5-4B's
    ``config.json`` names ``<|endoftext|>`` as its EOS while its chat template
    ends a turn with ``<|im_end|>``, and the checkpoint ships no
    ``generation_config.json`` to settle it.

    Args:
      model: The served model name.
      prompt_tokens: The whole conversation so far.
      max_tokens: Most tokens this turn may generate.
      temperature: Sampling temperature.
      top_p: Nucleus sampling cutoff.
      stop_token_ids: Ids that end a turn.
      seed: Optional sampling seed.

    Returns:
      payload: The JSON body.

    """
    payload: dict[str, object] = {
        "model": model,
        "prompt": list(prompt_tokens),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "logprobs": 1,
        "n": 1,
        "stop_token_ids": list(stop_token_ids),
        # vLLM extension: report each sampled token as ``token_id:<id>``.
        "return_tokens_as_token_ids": True,
    }
    if seed is not None:
        payload["seed"] = seed
    return payload


def parse_completion(body: Mapping[str, object]) -> SampledTokens:
    """Narrow one completions response into a sampled turn.

    Args:
      body: The decoded response.

    Returns:
      sampled: The turn's tokens, logprobs, and stop reason.

    Raises:
      RuntimeError: The response carries no usable choice, or does not report
        the sampled token ids.
      TypeError: The response carries a field of the wrong type.

    """
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Rollout server returned no choices: {body!r}.")
    choice = cast(Mapping[str, object], choices[0])
    logprob_payload = choice.get("logprobs")
    if not isinstance(logprob_payload, Mapping):
        raise TypeError(
            "Rollout server returned no logprobs; DPPO cannot be anchored "
            f"without them: {choice!r}.",
        )
    logprob_payload = cast(Mapping[str, object], logprob_payload)
    raw_logprobs = logprob_payload.get("token_logprobs")
    if not isinstance(raw_logprobs, list):
        raise TypeError(f"Rollout server returned no token logprobs: {choice!r}.")
    logprobs: list[float] = []
    for index, value in enumerate(cast(list[object], raw_logprobs)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                f"Rollout server returned no logprob at position {index}; "
                "every generated token has one, so a null here means the "
                "request omitted logprobs.",
            )
        logprobs.append(float(value))
    finish_reason = choice.get("finish_reason")
    return SampledTokens(
        token_ids=_token_ids(logprob_payload.get("tokens"), len(logprobs)),
        logprobs=tuple(logprobs),
        finish_reason=str(finish_reason) if finish_reason else "stop",
    )


def _token_ids(pieces: object, count: int) -> tuple[int, ...]:
    """Read the sampled ids from ``return_tokens_as_token_ids`` pieces.

    Args:
      pieces: The response's ``logprobs.tokens`` value.
      count: Number of sampled positions.

    Returns:
      tokens: Token ids, one per sampled position.

    Raises:
      RuntimeError: The pieces are missing, misaligned with the logprobs, or
        are decoded text rather than ids.

    """
    if not isinstance(pieces, list) or len(cast(list[object], pieces)) != count:
        raise RuntimeError(
            f"Rollout server returned no token list aligned with its {count} logprobs.",
        )
    tokens: list[int] = []
    for index, piece in enumerate(cast(list[object], pieces)):
        match = TOKEN_ID_PIECE.fullmatch(str(piece))
        if match is None:
            raise RuntimeError(
                f"Sampled piece {index} is {piece!r}, not a token id; the "
                "server ignored return_tokens_as_token_ids, and decoded text "
                "cannot be turned back into the id that was sampled.",
            )
        tokens.append(int(match.group(1)))
    return tuple(tokens)


class VLLMSampler:
    """One sampled turn per call, with token ids and behavior logprobs."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        stop_token_ids: Sequence[int],
        temperature: float,
        top_p: float,
        seed: int | None = None,
        timeout_s: float = 900.0,
    ) -> None:
        """Bind the server endpoint and the sampling parameters.

        Args:
          base_url: Server root, for example ``http://127.0.0.1:8000``.
          model: The served model name, as vLLM reports it.
          stop_token_ids: Ids that end a turn: the chat template's end-of-turn.
          temperature: Sampling temperature; must match the learner's.
          top_p: Nucleus sampling cutoff.
          seed: Optional sampling seed for reproducibility.
          timeout_s: Per-request timeout. One turn of a Lean proof can be long;
            the default is generous rather than tight.

        """
        self.url = base_url.rstrip("/") + COMPLETIONS_SUFFIX
        self.model = model
        self.stop_token_ids = tuple(stop_token_ids)
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.timeout_s = timeout_s

    async def sample(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int,
    ) -> SampledTokens:
        """Sample one turn.

        The request is issued from a worker thread so that a slow completion
        cannot stall the caller's event loop, which also drives the sandbox.

        Args:
          prompt_tokens: The whole conversation so far.
          max_tokens: Most tokens this turn may generate.

        Returns:
          sampled: The turn's tokens, logprobs, and stop reason.

        """
        payload = completion_request(
            self.model,
            prompt_tokens,
            max_tokens,
            self.temperature,
            self.top_p,
            self.stop_token_ids,
            self.seed,
        )
        body = await asyncio.to_thread(self._post, payload)
        return parse_completion(body)

    def _post(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        """Send one request and return the decoded body."""
        request = urllib.request.Request(
            self.url,
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_s,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as error:
            raise RuntimeError(f"Rollout server {self.url} failed: {error}") from error
        return cast(Mapping[str, object], json.loads(body))
