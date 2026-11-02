"""Tests for the openai/math episode loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import base64
import shlex
import shutil
import subprocess

import pytest

from priml.baselines.openaimath.episode import (
    PROTOCOL,
    PROTOCOL_ERROR,
    SUBMISSION_MAX_FILE_BYTES,
    TRANSCRIPT_NAME,
    TRUNCATION_MARKER,
    Action,
    ChatFormat,
    Episode,
    EpisodeConfig,
    ProtocolError,
    SampledTokens,
    build_prompt,
    parse_action,
    parse_submission_listing,
    read_submission_command,
    resolve_workspace_path,
    run_episode,
    submission_digest,
    truncate_observation,
    write_file_command,
)


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


END = "¶"
"""The fake template's end-of-turn token: one character, so one token."""

PLAIN = ChatFormat(
    prompt_prefix="<u>",
    generation_prefix="</u><a>",
    end_of_turn=END,
    observation_prefix="<u>",
)
"""A template with no reasoning block, shaped like Qwen's."""

THINKING = ChatFormat(
    prompt_prefix="<u>",
    generation_prefix="</u><a><t>",
    end_of_turn=END,
    observation_prefix="<u>",
    reasoning_end="</t>",
)
"""A template whose replies open with a reasoning block, as Qwen3.5's do."""


class _CharTokenizer:
    """A reversible one-token-per-character tokenizer."""

    def encode(self, text: str) -> list[int]:
        """Encode text as one token per character."""
        return [ord(character) + 1 for character in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode one token per character."""
        return "".join(chr(token - 1) for token in token_ids)


def _sampled(text: str, *, finish_reason: str = "stop") -> SampledTokens:
    """Return one turn the way the rollout server reports it."""
    return SampledTokens(
        token_ids=tuple(_CharTokenizer().encode(text)),
        logprobs=tuple(-0.5 for _ in text),
        finish_reason=finish_reason,
    )


class _ScriptedSampler:
    """A sampler that returns scripted turns and records its prompts."""

    def __init__(self, turns: Sequence[str | SampledTokens]) -> None:
        """Script one reply per turn; a string is a turn that stopped on its own."""
        self._turns = [
            turn if isinstance(turn, SampledTokens) else _sampled(turn + END)
            for turn in turns
        ]
        self.prompts: list[tuple[int, ...]] = []
        self.budgets: list[int] = []

    async def sample(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int,
    ) -> SampledTokens:
        """Return the next scripted turn."""
        self.prompts.append(tuple(prompt_tokens))
        self.budgets.append(max_tokens)
        if not self._turns:
            raise AssertionError("the episode sampled more turns than were scripted")
        return self._turns.pop(0)


class _FakeEnvironment:
    """An in-memory sandbox."""

    def __init__(
        self,
        output: str = "no errors",
        *,
        refuse_writes: bool = False,
    ) -> None:
        """Record commands and writes instead of running them."""
        self.output = output
        self.refuse_writes = refuse_writes
        self.commands: list[str] = []
        self.files: dict[str, str] = {}

    async def run(self, command: str) -> str:
        """Record the command and return canned output."""
        self.commands.append(command)
        return self.output

    async def write_file(self, path: str, content: str) -> None:
        """Record the write, or refuse it."""
        if self.refuse_writes:
            raise ProtocolError(f"refusing to write {path}")
        self.files[path] = content


def _episode(
    turns: Sequence[str | SampledTokens],
    *,
    environment: _FakeEnvironment | None = None,
    config: EpisodeConfig | None = None,
    chat_format: ChatFormat = PLAIN,
) -> tuple[Episode, _ScriptedSampler, _FakeEnvironment]:
    """Run one episode over scripted turns."""
    sampler = _ScriptedSampler(turns)
    sandbox = environment or _FakeEnvironment()
    episode = asyncio.run(
        run_episode(
            instruction="prove the statement",
            tokenizer=_CharTokenizer(),
            sampler=sampler,
            environment=sandbox,
            chat_format=chat_format,
            config=config or EpisodeConfig(),
        ),
    )
    return episode, sampler, sandbox


def _text(tokens: Sequence[int]) -> str:
    """Decode a fake-tokenized stream."""
    return _CharTokenizer().decode(tokens)


def test_parse_action_reads_each_directive() -> None:
    """The three directives parse, and nothing else does."""
    assert parse_action("RUN: lake build Submission") == Action(
        kind="run",
        argument="lake build Submission",
    )
    assert parse_action("SUBMIT") == Action(kind="submit")
    assert parse_action("SUBMIT\nhere is my reasoning") == Action(kind="submit")
    edit = parse_action(
        "EDIT: Submission.lean\n```lean\ntheorem x : True := trivial\n```",
    )
    assert edit.kind == "edit"
    assert edit.argument == "Submission.lean"
    assert edit.content == "theorem x : True := trivial"
    assert parse_action("I will now prove it.") == Action(kind="invalid")
    assert parse_action("RUN:") == Action(kind="invalid")
    assert parse_action("EDIT: Submission.lean") == Action(kind="invalid")


def test_parse_action_is_case_insensitive_and_takes_the_first_directive() -> None:
    """A drifting reply still parses, and its first directive wins."""
    assert parse_action("run: ls") == Action(kind="run", argument="ls")
    assert parse_action("Submit") == Action(kind="submit")
    assert parse_action("RUN: ls\nSUBMIT") == Action(kind="run", argument="ls")


def test_parse_action_does_not_submit_on_prose() -> None:
    """Submitting is irreversible, so only a bare SUBMIT line submits."""
    assert parse_action("Submitting now would fail.") == Action(kind="invalid")
    assert parse_action("SUBMIT when it builds") == Action(kind="invalid")
    assert parse_action("Submitting would fail.\nRUN: ls") == Action(
        kind="run",
        argument="ls",
    )


def test_parse_action_reads_only_after_the_reasoning_block() -> None:
    """A model reasoning about an action has not taken it."""
    reply = "Should I SUBMIT?\nSUBMIT\n</t>\nRUN: lake build Submission"
    assert parse_action(reply, reasoning_end="</t>") == Action(
        kind="run",
        argument="lake build Submission",
    )
    assert parse_action("SUBMIT\nstill thinking", reasoning_end="</t>") == Action(
        kind="invalid",
    )
    # The last close wins, so a reply that quotes the marker still parses.
    assert parse_action("a </t> b\n</t>\nSUBMIT", reasoning_end="</t>") == Action(
        kind="submit",
    )


def test_parse_action_reads_the_first_lean_block() -> None:
    """An edit takes the first fenced Lean block, even when prose follows."""
    reply = (
        "EDIT: Submission.lean\n"
        "```lean\nfirst\n```\n"
        "and here is another\n"
        "```lean\nsecond\n```\n"
    )
    assert parse_action(reply).content == "first"


def test_parse_action_accepts_an_unclosed_lean_block() -> None:
    """A reply that runs out of budget mid-block still yields its content."""
    reply = "EDIT: Submission.lean\n```lean\nimport Mathlib\n"
    assert parse_action(reply).content.strip() == "import Mathlib"


def test_chat_format_requires_an_end_of_turn() -> None:
    """A reply with no stop token could never end on its own."""
    with pytest.raises(ValueError, match="end_of_turn"):
        ChatFormat(
            prompt_prefix="<u>",
            generation_prefix="<a>",
            end_of_turn="",
            observation_prefix="<u>",
        )


def test_build_prompt_frames_the_instruction_and_truncates_only_it() -> None:
    """The framing and protocol always fit; a long instruction is cut from its end."""
    framing = len(PLAIN.prompt_prefix) + len(PLAIN.generation_prefix)
    budget = framing + len(PROTOCOL) + 50
    tokens = build_prompt(
        "x" * 400,
        _CharTokenizer(),
        chat_format=PLAIN,
        max_prompt_tokens=budget,
    )
    assert len(tokens) == budget
    assert _text(tokens) == "<u>" + "x" * 50 + PROTOCOL + "</u><a>"


def test_build_prompt_rejects_a_budget_below_the_framing() -> None:
    """A prompt budget that cannot hold the protocol is refused."""
    with pytest.raises(ValueError, match="action protocol need"):
        build_prompt(
            "hi",
            _CharTokenizer(),
            chat_format=PLAIN,
            max_prompt_tokens=len(PROTOCOL),
        )


def test_truncate_observation_marks_a_cut_tail() -> None:
    """A long observation keeps its head and says the tail is missing."""
    text = "".join(f"line {index}\n" for index in range(200))
    tokens = truncate_observation(text, _CharTokenizer(), max_tokens=40)
    assert len(tokens) == 40
    assert _text(tokens).startswith("line 0")
    assert _text(tokens).endswith(TRUNCATION_MARKER)


def test_truncate_observation_never_exceeds_a_budget_below_the_marker() -> None:
    """The cap holds even when the marker itself would not fit."""
    tokens = truncate_observation("x" * 100, _CharTokenizer(), max_tokens=5)
    assert _text(tokens) == "xxxxx"


def test_truncate_observation_keeps_a_short_observation_whole() -> None:
    """An observation inside its budget is unchanged."""
    tokenizer = _CharTokenizer()
    assert truncate_observation("exit code 0", tokenizer, max_tokens=40) == tuple(
        tokenizer.encode("exit code 0"),
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("Submission.lean", "/workspace/Submission.lean"),
        ("./Submission.lean", "/workspace/Submission.lean"),
        ("Submission/Foo.lean", "/workspace/Submission/Foo.lean"),
        ("/workspace/Submission.lean", "/workspace/Submission.lean"),
        ("/workspace/Submission/Foo.lean", "/workspace/Submission/Foo.lean"),
    ],
)
def test_resolve_workspace_path_resolves_inside_the_workspace(
    path: str,
    expected: str,
) -> None:
    """Relative paths, and absolute ones the model saw in the sandbox, resolve once."""
    assert resolve_workspace_path(path, "/workspace") == expected


@pytest.mark.parametrize(
    "hostile",
    [
        "",
        ".",
        "..",
        "../x.lean",
        "Submission/../../x.lean",
        "/etc/passwd",
        "/workspace",
        "/workspace-other/x.lean",
        "/Submission/Foo.lean",
    ],
)
def test_resolve_workspace_path_rejects_escapes(hostile: str) -> None:
    """An edit path cannot leave the workspace or name the workspace itself."""
    with pytest.raises(
        ValueError,
        match=r"usable file path|escapes the workspace|outside the workspace",
    ):
        resolve_workspace_path(hostile, "/workspace")


def test_write_file_command_survives_hostile_content() -> None:
    """A proof containing shell syntax is written byte for byte."""
    content = "line one\nEOF\n'single' \"double\" $VAR `cmd` \\ backslash\n"
    command = write_file_command("/workspace/Submission.lean", content)
    words = shlex.split(command)
    encoded = words[words.index("printf") + 2]
    assert base64.b64decode(encoded).decode("utf-8") == content
    assert "EOF" not in command
    assert "$VAR" not in command


def test_write_file_command_creates_the_parent_directory() -> None:
    """A proof split into modules writes into a directory the image lacks."""
    command = write_file_command("/workspace/Submission/Part One.lean", "x")
    assert command.startswith("mkdir -p /workspace/Submission && ")
    assert shlex.split(command)[-1] == "/workspace/Submission/Part One.lean"


_NEEDS_SHELL = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in ("sh", "find", "base64", "wc", "tr")),
    reason="needs a POSIX shell with base64",
)


def _run_shell(command: str, cwd: Path) -> str:
    """Run a command in a POSIX shell, as the sandbox does, and return stdout."""
    completed = subprocess.run(  # noqa: S603 -- The command is the module's own, run on a scratch directory.
        ["sh", "-c", command],  # noqa: S607 -- A POSIX shell, as the sandbox runs it.
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout


@pytest.mark.cli_bash
@_NEEDS_SHELL
def test_read_submission_command_lists_the_graded_files(tmp_path: Path) -> None:
    """The listing round-trips the submission, its modules, and hostile text."""
    workspace = tmp_path / "workspace"
    (workspace / "Submission" / "Part One").mkdir(parents=True)
    hostile = "theorem x : True := trivial\n-- 'q' \"d\" $VAR `c` \\ \t tab\n"
    (workspace / "Submission.lean").write_text(hostile, encoding="utf-8")
    (workspace / "Submission" / "Part One" / "A.lean").write_text(
        "lemma a : 1 = 1 := rfl\n",
        encoding="utf-8",
    )
    (workspace / "Submission" / "notes.txt").write_text("ignored", encoding="utf-8")
    (workspace / "Submission" / "Big.lean").write_bytes(
        b"-" * (SUBMISSION_MAX_FILE_BYTES + 1),
    )
    (workspace / "Other.lean").write_text("not graded", encoding="utf-8")
    command = read_submission_command("Submission.lean", str(workspace))
    files = parse_submission_listing(_run_shell(command, workspace))
    assert files == {
        "Submission.lean": hostile,
        "Submission/Big.lean": None,
        "Submission/Part One/A.lean": "lemma a : 1 = 1 := rfl\n",
    }


@pytest.mark.cli_bash
@_NEEDS_SHELL
def test_read_submission_command_without_a_module_directory(tmp_path: Path) -> None:
    """The common case, no ``Submission/``, still lists the submission."""
    (tmp_path / "Submission.lean").write_text("x\n", encoding="utf-8")
    command = read_submission_command("Submission.lean", str(tmp_path))
    assert parse_submission_listing(_run_shell(command, tmp_path)) == {
        "Submission.lean": "x\n",
    }


def test_parse_submission_listing_skips_malformed_lines() -> None:
    """A line that is not two base64 fields is dropped, not guessed at."""
    path = base64.b64encode(b"Submission.lean").decode("ascii")
    body = base64.b64encode(b"proof").decode("ascii")
    output = f"{path}\t{body}\nno tab here\n{path}x\t{body}\n!!\t{body}\n"
    assert parse_submission_listing(output) == {"Submission.lean": "proof"}


def test_submission_digest_depends_on_paths_and_contents_only() -> None:
    """Equal files give equal digests, whatever order they were listed in."""
    files = {"Submission.lean": "a", "Submission/B.lean": "b"}
    reordered = {"Submission/B.lean": "b", "Submission.lean": "a"}
    assert submission_digest(files) == submission_digest(reordered)
    assert submission_digest(files) != submission_digest(
        {**files, "Submission.lean": "c"},
    )
    assert len(submission_digest({})) == 64


def test_run_episode_records_policy_and_sandbox_tokens_separately() -> None:
    """Sandbox output and chat framing are in the transcript but not trained on."""
    episode, sampler, sandbox = _episode(
        ["RUN: lake build Submission", "SUBMIT"],
        environment=_FakeEnvironment("build failed: unsolved goals"),
    )
    assert episode.submitted is True
    assert episode.finish_reason == "stop"
    # The end-of-turn token is stripped before the reply is parsed.
    assert sandbox.commands == ["lake build Submission"]
    first = "RUN: lake build Submission" + END
    framed = "<u>$ lake build Submission\nbuild failed: unsolved goals</u><a>"
    last = "SUBMIT" + END
    assert _text(episode.response_tokens) == first + framed + last
    end_framed = len(first) + len(framed)
    assert episode.tool_mask == (1,) * len(first) + (0,) * len(framed) + (1,) * len(
        last,
    )
    assert episode.logprobs[: len(first)] == (-0.5,) * len(first)
    assert episode.logprobs[len(first) : end_framed] == (0.0,) * len(framed)
    assert episode.logprobs[end_framed:] == (-0.5,) * len(last)
    assert episode.policy_token_count == len(first) + len(last)
    assert _text(episode.prompt_tokens).startswith("<u>prove the statement")
    assert len(sampler.prompts) == 2


def test_run_episode_feeds_observations_back_into_the_next_prompt() -> None:
    """The second turn sees the first turn, its output, and a fresh reply opening."""
    _, sampler, _ = _episode(
        ["RUN: ls", "SUBMIT"],
        environment=_FakeEnvironment("Submission.lean"),
    )
    second_prompt = _text(sampler.prompts[1])
    assert second_prompt.endswith(f"RUN: ls{END}<u>$ ls\nSubmission.lean</u><a>")


def test_run_episode_closes_a_reply_cut_off_by_its_budget() -> None:
    """A reply that never emitted its end-of-turn is closed by the framing."""
    episode, _, sandbox = _episode(
        [_sampled("RUN: ls", finish_reason="length"), "SUBMIT"],
    )
    assert sandbox.commands == ["ls"]
    text = _text(episode.response_tokens)
    assert text.startswith(f"RUN: ls{END}<u>$ ls\n")
    # The closing token was written by the framing, not by the policy.
    assert episode.tool_mask[len("RUN: ls")] == 0


def test_run_episode_reads_actions_only_after_the_reasoning_block() -> None:
    """With a reasoning template, a directive inside the reasoning is not acted on."""
    episode, _, sandbox = _episode(
        ["SUBMIT?\n</t>\nRUN: ls", "I should SUBMIT\nSUBMIT", "</t>\nSUBMIT"],
        chat_format=THINKING,
    )
    assert sandbox.commands == ["ls"]
    assert episode.turns == 3
    text = _text(episode.response_tokens)
    assert f"<u>{PROTOCOL_ERROR}</u><a><t>" in text


def test_run_episode_writes_an_edited_file() -> None:
    """An edit reaches the sandbox, and its reply is not a command."""
    _, _, sandbox = _episode(
        ["EDIT: Submission.lean\n```lean\ntheorem x : True := trivial\n```", "SUBMIT"],
    )
    assert sandbox.files == {"Submission.lean": "theorem x : True := trivial"}
    assert sandbox.commands == []


def test_run_episode_reports_a_refused_write_as_an_observation() -> None:
    """A refused edit becomes text the model can recover from."""
    episode, _, _ = _episode(
        ["EDIT: /etc/passwd\n```lean\nx\n```", "SUBMIT"],
        environment=_FakeEnvironment(refuse_writes=True),
    )
    assert episode.submitted is True
    assert "<u>refusing to write /etc/passwd</u><a>" in _text(episode.response_tokens)


def test_run_episode_reports_an_unparsable_reply() -> None:
    """A reply with no directive gets the protocol restated."""
    episode, _, _ = _episode(["let me think about it", "SUBMIT"])
    assert f"<u>{PROTOCOL_ERROR}</u><a>" in _text(episode.response_tokens)
    assert episode.turns == 2


def test_run_episode_stops_at_the_turn_budget() -> None:
    """An episode that never submits ends unsubmitted."""
    episode, sampler, _ = _episode(["RUN: ls"] * 8, config=EpisodeConfig(max_turns=3))
    assert episode.submitted is False
    assert episode.finish_reason == "length"
    assert episode.turns == 3
    assert len(sampler.prompts) == 3


def test_run_episode_keeps_every_token_inside_the_response_budget() -> None:
    """Policy, sandbox, and framing tokens all count, so the server never overflows."""
    episode, sampler, _ = _episode(
        ["RUN: ls"] * 8,
        config=EpisodeConfig(response_length=40, per_turn_max_tokens=20, max_turns=8),
    )
    assert episode.finish_reason == "length"
    assert len(episode.response_tokens) <= 40
    assert sampler.budgets[0] == 20
    assert all(budget <= 40 for budget in sampler.budgets)


def test_run_episode_refuses_a_sampler_that_ignores_its_budget() -> None:
    """A turn longer than requested would break the episode's context bound."""
    with pytest.raises(RuntimeError, match="4-token turn"):
        _episode(["RUN: ls"], config=EpisodeConfig(per_turn_max_tokens=4))


def test_episode_config_rejects_impossible_budgets() -> None:
    """Budgets must be positive for an episode to mean anything."""
    with pytest.raises(ValueError, match="max_turns"):
        EpisodeConfig(max_turns=0)
    with pytest.raises(ValueError, match="max_prompt_tokens"):
        EpisodeConfig(max_prompt_tokens=0)
    with pytest.raises(ValueError, match="max_observation_tokens"):
        EpisodeConfig(max_observation_tokens=0)


def test_transcript_name_is_a_single_file() -> None:
    """The transcript's name is the reader's and the writer's shared constant."""
    assert TRANSCRIPT_NAME.endswith(".jsonl")
    assert "/" not in TRANSCRIPT_NAME


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
