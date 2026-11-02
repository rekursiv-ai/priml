"""Tests for the rollout server's request, response, and chat framing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio

import pytest

from priml.baselines.openaimath.episode import ChatFormat, SampledTokens
from priml.baselines.openaimath.sampler import (
    TransformersTokenizer,
    VLLMSampler,
    completion_request,
    parse_completion,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


def _body(
    tokens: Sequence[object],
    logprobs: Sequence[object],
    *,
    finish_reason: str | None = "stop",
) -> dict[str, object]:
    """Return a completions response shaped like vLLM's."""
    return {
        "choices": [
            {
                "text": "ignored",
                "finish_reason": finish_reason,
                "logprobs": {"tokens": list(tokens), "token_logprobs": list(logprobs)},
            },
        ],
    }


def test_completion_request_pins_the_stop_ids_and_asks_for_token_ids() -> None:
    """The server must stop on the template's turn end and report sampled ids."""
    payload = completion_request("policy", [1, 2], 16, 1.0, 1.0, [248_046])
    assert payload["prompt"] == [1, 2]
    assert payload["max_tokens"] == 16
    assert payload["stop_token_ids"] == [248_046]
    assert payload["return_tokens_as_token_ids"] is True
    assert payload["logprobs"] == 1
    assert "seed" not in payload
    seeded = completion_request("policy", [1], 1, 1.0, 1.0, [9], seed=3)
    assert seeded["seed"] == 3


def test_parse_completion_reads_the_sampled_ids() -> None:
    """Ids come from the pieces, never from re-encoding text."""
    sampled = parse_completion(
        _body(["token_id:5513", "token_id:248046"], [-0.25, -1.0]),
    )
    assert sampled == SampledTokens(
        token_ids=(5513, 248_046),
        logprobs=(-0.25, -1.0),
        finish_reason="stop",
    )


def test_parse_completion_keeps_a_length_stop() -> None:
    """A turn cut off by its budget is reported as such."""
    sampled = parse_completion(
        _body(["token_id:1"], [-0.5], finish_reason="length"),
    )
    assert sampled.finish_reason == "length"


def test_parse_completion_refuses_decoded_pieces() -> None:
    """Both byte halves of a split character decode alike; text cannot be trusted."""
    with pytest.raises(RuntimeError, match="not a token id"):
        parse_completion(_body(["\ufffd", "\ufffd"], [-0.1, -0.2]))


@pytest.mark.parametrize(
    ("body", "error", "match"),
    [
        ({"choices": []}, RuntimeError, "no choices"),
        ({"choices": [{"logprobs": None}]}, TypeError, "no logprobs"),
        (_body(["token_id:1"], [None]), TypeError, "position 0"),
        (_body(["token_id:1"], [-0.1, -0.2]), RuntimeError, "aligned"),
    ],
)
def test_parse_completion_refuses_an_unusable_response(
    body: Mapping[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """A response the learner could not score is an error, not a guess."""
    with pytest.raises(error, match=match):
        parse_completion(body)


def test_vllm_sampler_posts_one_request_per_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sampler sends the pinned request and parses the reply."""
    sampler = VLLMSampler(
        "http://127.0.0.1:8000/",
        "policy",
        stop_token_ids=(7,),
        temperature=1.0,
        top_p=0.9,
    )
    sent: list[Mapping[str, object]] = []

    def post(payload: Mapping[str, object]) -> Mapping[str, object]:
        sent.append(payload)
        return _body(["token_id:4", "token_id:7"], [-0.5, -0.1])

    monkeypatch.setattr(sampler, "_post", post)
    sampled = asyncio.run(sampler.sample([1, 2, 3], max_tokens=8))
    assert sampler.url == "http://127.0.0.1:8000/v1/completions"
    assert sampled.token_ids == (4, 7)
    assert sent[0]["stop_token_ids"] == [7]
    assert sent[0]["top_p"] == 0.9


class _FakeHFTokenizer:
    """The members of a transformers tokenizer the chat-format reader uses."""

    def __init__(
        self,
        render: Callable[[Sequence[Mapping[str, str]]], str],
        *,
        end_of_turn: str = "<|im_end|>",
    ) -> None:
        """Wrap a template renderer whose EOS is ``end_of_turn``."""
        self._render = render
        self.eos_token = end_of_turn

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        """Render with a generation prompt, as the reader asks."""
        assert tokenize is False
        assert add_generation_prompt is True
        return self._render(messages)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode the special markers as one token each."""
        del add_special_tokens
        return [0] if text in {"<|im_end|>", "<|im_start|>"} else [ord(c) for c in text]


def _qwen(messages: Sequence[Mapping[str, str]]) -> str:
    """Render the way Qwen3.5's template does, generation prompt included."""
    turns = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )
    return turns + "<|im_start|>assistant\n<think>\n"


def _tokenizer(fake: _FakeHFTokenizer) -> TransformersTokenizer:
    """Wrap a fake without loading anything from disk."""
    tokenizer = TransformersTokenizer.__new__(TransformersTokenizer)
    tokenizer.path = "fake"
    # Bypasses the attribute's transformers type, which the fake only mimics.
    object.__setattr__(tokenizer, "_tokenizer", fake)
    return tokenizer


def test_chat_format_reads_the_framing_off_the_template() -> None:
    """Qwen's template becomes one appendable stream with a reasoning block."""
    assert _tokenizer(_FakeHFTokenizer(_qwen)).chat_format() == ChatFormat(
        prompt_prefix="<|im_start|>user\n",
        generation_prefix="<|im_end|>\n<|im_start|>assistant\n<think>\n",
        end_of_turn="<|im_end|>",
        observation_prefix="\n<|im_start|>user\n",
        reasoning_end="</think>",
    )


def test_chat_format_without_a_reasoning_block_reads_whole_replies() -> None:
    """A template that opens no reasoning block parses every reply in full."""

    def plain(messages: Sequence[Mapping[str, str]]) -> str:
        return _qwen(messages).removesuffix("<think>\n")

    assert _tokenizer(_FakeHFTokenizer(plain)).chat_format().reasoning_end == ""


def test_chat_format_refuses_a_template_that_rewrites_history() -> None:
    """A template that changes earlier turns cannot be appended to."""

    def with_system(messages: Sequence[Mapping[str, str]]) -> str:
        prefix = "<|im_start|>system\nhi<|im_end|>\n" if len(messages) > 1 else ""
        return prefix + _qwen(messages)

    with pytest.raises(ValueError, match="appendable stream"):
        _tokenizer(_FakeHFTokenizer(with_system)).chat_format()


def test_chat_format_refuses_a_multi_token_end_of_turn() -> None:
    """The server stops on one id, so the turn end must be one token."""
    with pytest.raises(ValueError, match="not one token"):
        _tokenizer(
            _FakeHFTokenizer(
                lambda messages: _qwen(messages).replace("<|im_end|>", "</s>"),
                end_of_turn="</s>",
            ),
        ).chat_format()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
