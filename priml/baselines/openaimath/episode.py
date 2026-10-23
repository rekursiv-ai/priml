"""Run one openai/math episode: sample an action, run it, record the token stream.

An episode is one Lean proving attempt. The model is the policy under training;
the sandbox holds Lean, Mathlib and a copy of the challenge statement. The model
asks for shell commands and file edits until it submits, and the reward is the
task's own verifier verdict, which is 1 only when Comparator accepts the proof.

Everything in this module is plain Python over two seams -- a sampler that
returns token ids with their behavior logprobs, and an environment that runs
commands -- so the whole loop is testable without Harbor, vLLM, or a sandbox.

The transcript it returns is what DPPO consumes: one prompt (the task
instruction) and one response (every token the policy emitted, in order, with
the sandbox's own output interleaved but flagged). Sandbox tokens are flagged
rather than dropped because the model needs them as context on later turns, and
masked because fitting the policy to compiler output would teach it to imitate
a compiler instead of to prove theorems.

The conversation is framed by the policy's own chat template (see
:class:`ChatFormat`), because the policy is a chat model: without its turn
markers it has no end-of-turn to stop on and keeps writing past its action.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

import base64
import binascii
import hashlib
import json
import re
import shlex


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


MAX_PROMPT_TOKENS: Final = 16_384
"""Token budget for the framed instruction plus the protocol.

Lean work needs more prompt room than a terminal task: the instruction carries
the full theorem statement, its imports, and the surrounding definitions.
"""

RESPONSE_LENGTH: Final = 65_536
"""Token budget for every response token of an episode, policy and sandbox."""

PER_TURN_MAX_TOKENS: Final = 16_384
"""Token budget for one sampled turn, reasoning included."""

MAX_TURNS: Final = 64
"""Most sampled turns before an episode ends unsubmitted."""

MAX_OBSERVATION_TOKENS: Final = 4_096
"""Token budget for one sandbox observation spliced into the transcript."""


PROTOCOL: Final = """
--- How to work on it ---
You are an autonomous Lean 4 programmer working in a sandbox. The statement file
`Submission.lean` in the working directory is the challenge, with `sorry` where the
proof belongs. Replace every `sorry` with a real proof. Do not change anything else:
the statements, names, namespaces and definitions must stay exactly as written.

Reply with exactly ONE action per turn, as the first line of your reply:

RUN: <shell command>
    Run it in the working directory and see its output. Use this to compile with
    `lake build Submission`, to inspect errors, and to read files.
EDIT: <path>
    Replace that file with the Lean code in the ```lean fenced block of this
    reply. The block must contain the COMPLETE new contents of the file.
SUBMIT
    Hand your proof to the grader. This ends the episode and cannot be undone.

`lake build Submission` must succeed before you submit; the grader re-checks the
proof with the Lean kernel and rejects anything that does not compile, anything
that uses `sorry`, `admit`, `native_decide` or a new `axiom`, and any change to
the statements. Long proofs may be split into modules under `Submission/` and
imported from `Submission.lean`. Anything you write outside those files is ignored.
"""
"""The action protocol appended to the task's own instruction.

The instruction supplied by the task explains the mathematics and where the
challenge file lives. It does not define how a language model should act, so
this text does, and it is the only part of the prompt this baseline authors.
"""

PROTOCOL_ERROR: Final = (
    "Your reply did not contain a usable action. Reply with exactly one of "
    "`RUN: <shell command>`, `EDIT: <path>` followed by a ```lean block, or "
    "`SUBMIT`, as the first line of your reply."
)

TRUNCATION_MARKER: Final = "\n[... output truncated ...]\n"

TRANSCRIPT_NAME: Final = "openaimath_episode.jsonl"
"""The flattened transcript each episode writes for the learner to read back.

Harbor has its own channel for rollouts, ``AgentContext.rollout_details``, but
it is per turn and it is reported rather than trained on. The learner needs one
prompt and one response per episode, exactly as DPPO scores it, so the agent
writes that record itself and the rollout runner reads it back from the trial's
agent directory. The name lives here, with the record's definition, so the
writer and the reader cannot drift apart.
"""

SUBMISSION_MAX_FILE_BYTES: Final = 1_000_000
"""Largest graded file whose contents the transcript records.

An edit is bounded by the episode's token budget, but a shell command can write
a file of any size, and the transcript is read back whole by the runner.
"""

LEAN_FENCE: Final = re.compile(r"```lean[ \t]*\n(.*?)(?:\n[ \t]*```|\Z)", re.DOTALL)
"""The first ```lean block of a reply; an unclosed block runs to the end."""

ActionKind = Literal["run", "edit", "submit", "invalid"]


class ProtocolError(Exception):
    """An action the environment refused: a bad path, or a failed write.

    Raised by an environment rather than returned as text so that the episode
    can turn it into an observation the model can read and recover from. A
    refused action is a model mistake, not a run failure: aborting the episode
    would throw away the rest of its turns and, with them, the reward signal
    the attempt was about to produce.
    """


@dataclass(frozen=True, slots=True)
class SampledTokens:
    """One sampled turn: the policy's own tokens with their behavior logprobs."""

    token_ids: tuple[int, ...]
    """Generated token ids, in order."""

    logprobs: tuple[float, ...]
    r"""Behavior logprobs :math:`\log\mu_{\theta'}` from the rollout server.

    DPPO anchors its ratio on these, so they are load-bearing rather than
    diagnostic.
    """

    finish_reason: str
    """Why the turn ended: ``"stop"`` at the end of a message, ``"length"``
    when the per-turn budget ran out."""


@runtime_checkable
class Sampler(Protocol):
    """The rollout policy: one completion per call, with token ids."""

    async def sample(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_tokens: int,
    ) -> SampledTokens:
        """Sample one turn from the current policy.

        Args:
          prompt_tokens: The whole conversation so far.
          max_tokens: Most tokens this turn may generate.

        Returns:
          sampled: The turn's tokens, logprobs, and stop reason.

        """
        ...


@runtime_checkable
class Tokenizer(Protocol):
    """Text and token ids, one vocabulary: the policy's own tokenizer."""

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids back to text."""
        ...


@runtime_checkable
class EpisodeEnvironment(Protocol):
    """A sandbox the episode can act in: run a command, write a file."""

    async def run(self, command: str) -> str:
        """Run a shell command in the workspace and return its combined output."""
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write a file in the workspace, replacing it."""
        ...


@dataclass(frozen=True, slots=True)
class ChatFormat:
    """The policy's chat template, rendered once into the text around each turn.

    The episode is one token stream, not a list of messages. Re-rendering the
    conversation through the template every turn would re-tokenize the policy's
    own tokens, and the learner must score exactly the ids that were sampled, so
    the template is rendered once into these pieces and each turn appends to the
    stream. Every piece is framing the policy did not choose, so every piece is
    masked out of the objective.
    """

    prompt_prefix: str
    """Opens the user message that holds the instruction."""

    generation_prefix: str
    """Closes a user message and opens the policy's reply."""

    end_of_turn: str
    """The token that closes the policy's reply; the rollout server stops on it."""

    observation_prefix: str
    """Follows a closed reply and opens the user message an observation goes in."""

    reasoning_end: str = ""
    """Closes the reasoning block :attr:`generation_prefix` opens, if it opens one.

    Directives are read only after it: a model thinking about ``SUBMIT`` has not
    submitted. Empty when the template opens no reasoning block.
    """

    def __post_init__(self) -> None:
        """Reject a format whose replies could never end on their own."""
        if not self.end_of_turn:
            raise ValueError("ChatFormat.end_of_turn must name the stop token.")


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    """Budgets and paths for one episode."""

    workspace: str = "/workspace"
    """The task's working directory, a Lake project."""

    submission_path: str = "Submission.lean"
    """The file the agent edits, relative to :attr:`workspace`."""

    max_prompt_tokens: int = MAX_PROMPT_TOKENS
    """Token budget for the framed instruction plus the protocol."""

    response_length: int = RESPONSE_LENGTH
    """Token budget for the whole response: policy tokens and sandbox tokens.

    The rollout server's context bound is ``max_prompt_tokens +
    response_length``, so an episode that respected only the policy's share
    would overflow the server on its last turns.
    """

    per_turn_max_tokens: int = PER_TURN_MAX_TOKENS
    """Token budget for one sampled turn."""

    max_turns: int = MAX_TURNS
    """Most sampled turns before the episode ends unsubmitted."""

    max_observation_tokens: int = MAX_OBSERVATION_TOKENS
    """Token budget for one sandbox observation spliced into the transcript."""

    def __post_init__(self) -> None:
        """Reject budgets that cannot produce a usable episode."""
        if self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive.")
        if self.response_length <= 0 or self.per_turn_max_tokens <= 0:
            raise ValueError("Response budgets must be positive.")
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive.")
        if self.max_observation_tokens <= 0:
            raise ValueError("max_observation_tokens must be positive.")


@dataclass(frozen=True, slots=True)
class Episode:
    """One finished episode, as the learner needs to see it.

    ``response_tokens`` is the complete token stream the episode produced: the
    policy's tokens and the sandbox's output, in the order they were appended to
    the conversation. ``tool_mask`` says which of them the policy wrote.
    """

    prompt_tokens: tuple[int, ...]
    """The instruction and protocol, tokenized."""

    response_tokens: tuple[int, ...]
    """Policy tokens and sandbox tokens, in transcript order."""

    logprobs: tuple[float, ...]
    """Behavior logprob per response token; exactly ``0.0`` on sandbox tokens.

    vLLM never sampled those tokens, so there is no behavior probability to
    report. The learner masks them out of the objective, and a zero here keeps
    the record's shape valid for the packing code that rejects NaN.
    """

    tool_mask: tuple[int, ...]
    """``1`` where the policy wrote the token, ``0`` on sandbox output."""

    finish_reason: str
    """``"stop"`` when the episode submitted; ``"length"`` when a budget ran out."""

    turns: int
    """Sampled turns the episode used."""

    submitted: bool
    """Whether the episode handed a proof to the grader."""

    @property
    def policy_token_count(self) -> int:
        """Number of tokens the policy actually emitted.

        Returns:
          count: Trainable token count before packing.

        """
        return sum(self.tool_mask)


@dataclass(frozen=True, slots=True)
class Action:
    """One parsed agent action."""

    kind: ActionKind
    """Which directive the reply started with, or ``"invalid"``."""

    argument: str = ""
    """The directive's argument: a command, or a path."""

    content: str = ""
    """The fenced Lean block, for ``"edit"``."""


def parse_action(text: str, *, reasoning_end: str = "") -> Action:
    """Read the first directive of a reply.

    The protocol asks for one action per turn and names it as the first line of
    the reply. Models drift, so the scan is line-based and takes the FIRST
    directive it finds anywhere in the reply, which keeps a reply that both
    narrates and acts deterministic to parse. A directive word is matched
    case-insensitively; anything before the directive is ignored, so a model may
    briefly explain itself first. ``SUBMIT`` must be the whole line, because it
    ends the episode irreversibly and prose such as "Submitting now would fail"
    starts with the same word.

    Args:
      text: The decoded turn, without its end-of-turn token.
      reasoning_end: The marker that closes the reply's reasoning block; only
        text after its last occurrence is read. Empty reads the whole reply.

    Returns:
      action: The parsed action, or ``Action("invalid")`` when no directive is
        usable, including a reply that never left its reasoning block.

    """
    if reasoning_end:
        _, marker, text = text.rpartition(reasoning_end)
        if not marker:
            return Action("invalid")
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper == "SUBMIT":
            return Action(kind="submit")
        if upper.startswith("RUN:"):
            command = stripped[len("RUN:") :].strip()
            return (
                Action(kind="run", argument=command) if command else Action("invalid")
            )
        if upper.startswith("EDIT:"):
            path = stripped[len("EDIT:") :].strip()
            block = LEAN_FENCE.search(text)
            if not path or block is None:
                return Action("invalid")
            return Action(kind="edit", argument=path, content=block.group(1))
    return Action("invalid")


def build_prompt(
    instruction: str,
    tokenizer: Tokenizer,
    *,
    chat_format: ChatFormat,
    max_prompt_tokens: int,
) -> tuple[int, ...]:
    """Tokenize the framed task instruction plus the action protocol.

    The framing and the protocol are budgeted first and never truncated: without
    them the model cannot act at all. The instruction is truncated from the end
    when it does not fit, which matters for the longest tasks, whose
    instructions embed a very large statement. Truncation is safe because the
    same statement is in the sandbox at ``Submission.lean``, which the protocol
    names, and because the grader compares the submission against its own copy
    rather than against the prompt.

    Args:
      instruction: The task's own instruction text.
      tokenizer: The policy's tokenizer.
      chat_format: The policy's chat framing.
      max_prompt_tokens: Budget for the whole prompt.

    Returns:
      tokens: The user-message opening, instruction, protocol, and the
        opening of the policy's first reply.

    Raises:
      ValueError: The framing and protocol alone do not fit the budget.

    """
    opening = tokenizer.encode(chat_format.prompt_prefix)
    protocol = tokenizer.encode(PROTOCOL)
    reply = tokenizer.encode(chat_format.generation_prefix)
    fixed = len(opening) + len(protocol) + len(reply)
    body_budget = max_prompt_tokens - fixed
    if body_budget <= 0:
        raise ValueError(
            f"The chat framing and action protocol need {fixed} tokens, more "
            f"than the {max_prompt_tokens}-token prompt budget.",
        )
    body = tokenizer.encode(instruction)
    return (*opening, *body[:body_budget], *protocol, *reply)


def truncate_observation(
    text: str,
    tokenizer: Tokenizer,
    *,
    max_tokens: int,
) -> tuple[int, ...]:
    """Tokenize a sandbox observation and cap its length.

    Compiler output is unbounded: one failing `lake build` can print tens of
    thousands of lines. The head is kept because Lean reports the first error
    and its context first, and a marker records that the tail was dropped. A
    budget too small for the marker keeps the head alone, because the cap is
    what keeps the episode inside the rollout server's context bound.

    Args:
      text: The observation text.
      tokenizer: The policy's tokenizer.
      max_tokens: Token budget for this observation.

    Returns:
      tokens: The observation's tokens, at most ``max_tokens`` of them.

    """
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return tuple(tokens)
    marker = tokenizer.encode(TRUNCATION_MARKER)
    if len(marker) >= max_tokens:
        return tuple(tokens[: max(max_tokens, 0)])
    return (*tokens[: max_tokens - len(marker)], *marker)


async def run_episode(
    *,
    instruction: str,
    tokenizer: Tokenizer,
    sampler: Sampler,
    environment: EpisodeEnvironment,
    chat_format: ChatFormat,
    config: EpisodeConfig,
) -> Episode:
    """Run one proving attempt to submission or to a budget.

    Every token appended after the prompt counts against
    ``config.response_length``: the policy's turns, the sandbox's observations,
    and the chat framing around them. That is what keeps a whole episode inside
    the rollout server's ``max_prompt_tokens + response_length`` context, and
    what lets the learner pack it into one row of the same length.

    Args:
      instruction: The task's instruction text, used as the prompt.
      tokenizer: The policy's tokenizer.
      sampler: The rollout policy.
      environment: The sandbox the episode acts in.
      chat_format: The policy's chat framing.
      config: Budgets and paths.

    Returns:
      episode: The recorded transcript, submitted or not.

    Raises:
      RuntimeError: The sampler returned more tokens than it was allowed.

    """
    prompt_tokens = build_prompt(
        instruction,
        tokenizer,
        chat_format=chat_format,
        max_prompt_tokens=config.max_prompt_tokens,
    )
    end_of_turn = tuple(tokenizer.encode(chat_format.end_of_turn))
    observation_prefix = tuple(tokenizer.encode(chat_format.observation_prefix))
    generation_prefix = tuple(tokenizer.encode(chat_format.generation_prefix))
    context = list(prompt_tokens)
    response: list[int] = []
    logprobs: list[float] = []
    tool_mask: list[int] = []

    def append(tokens: Sequence[int], token_logprobs: Sequence[float]) -> None:
        response.extend(tokens)
        logprobs.extend(token_logprobs)
        context.extend(tokens)

    turns = 0
    submitted = False
    for _ in range(config.max_turns):
        remaining = config.response_length - len(response)
        if remaining <= 0:
            break
        max_tokens = min(config.per_turn_max_tokens, remaining)
        sampled = await sampler.sample(context, max_tokens=max_tokens)
        if len(sampled.token_ids) > max_tokens:
            raise RuntimeError(
                f"The sampler returned {len(sampled.token_ids)} tokens for a "
                f"{max_tokens}-token turn.",
            )
        turns += 1
        append(sampled.token_ids, sampled.logprobs)
        tool_mask.extend([1] * len(sampled.token_ids))
        closed = (
            len(sampled.token_ids) >= len(end_of_turn)
            and sampled.token_ids[-len(end_of_turn) :] == end_of_turn
        )
        reply = sampled.token_ids[: -len(end_of_turn)] if closed else sampled.token_ids
        action = parse_action(
            tokenizer.decode(reply),
            reasoning_end=chat_format.reasoning_end,
        )
        if action.kind == "submit":
            submitted = True
            break
        observation = await _observation(environment, action)
        # A reply cut off by its token budget never emitted its end-of-turn, so
        # the framing closes it before the observation's message opens.
        opening = observation_prefix if closed else (*end_of_turn, *observation_prefix)
        body_budget = min(
            config.max_observation_tokens,
            config.response_length
            - len(response)
            - len(opening)
            - len(generation_prefix),
        )
        if body_budget <= 0:
            break
        # The sandbox and the chat template wrote these tokens; the policy did
        # not. They stay in the transcript as context and are excluded from the
        # objective.
        framed = (
            *opening,
            *truncate_observation(observation, tokenizer, max_tokens=body_budget),
            *generation_prefix,
        )
        append(framed, [0.0] * len(framed))
        tool_mask.extend([0] * len(framed))
    return Episode(
        prompt_tokens=prompt_tokens,
        response_tokens=tuple(response),
        logprobs=tuple(logprobs),
        tool_mask=tuple(tool_mask),
        finish_reason="stop" if submitted else "length",
        turns=turns,
        submitted=submitted,
    )


async def _observation(
    environment: EpisodeEnvironment,
    action: Action,
) -> str:
    """Carry out one action and return the text to append to the transcript."""
    if action.kind == "run":
        return f"$ {action.argument}\n{await environment.run(action.argument)}"
    if action.kind == "edit":
        try:
            await environment.write_file(action.argument, action.content)
        except ProtocolError as error:
            return str(error)
        return f"wrote {action.argument} ({len(action.content.encode('utf-8'))} bytes)"
    return PROTOCOL_ERROR


def resolve_workspace_path(path: str, workspace: str) -> str:
    """Resolve an edited path inside the workspace and reject escapes.

    The model writes the path, so it is untrusted: an edit outside the
    workspace would either fail in the sandbox or write somewhere the grader
    never reads, and both would look like a model failure rather than a
    protocol one. A relative path is taken relative to the workspace, and an
    absolute one must already be inside it, which is how the model sees its
    files when it runs ``pwd`` or reads a compiler error.

    Args:
      path: The path from the ``EDIT:`` directive.
      workspace: The task's working directory, absolute.

    Returns:
      absolute: The absolute path inside the workspace.

    Raises:
      ValueError: The path is empty, names the workspace itself, is absolute
        outside the workspace, or escapes it through ``..``.

    """
    root = PurePosixPath(workspace)
    candidate = PurePosixPath(path.strip())
    if candidate.is_absolute():
        if root not in candidate.parents:
            raise ValueError(f"Path is outside the workspace {workspace}: {path!r}.")
        candidate = candidate.relative_to(root)
    parts = [part for part in candidate.parts if part != "."]
    if not parts:
        raise ValueError(f"Not a usable file path: {path!r}.")
    if ".." in parts:
        raise ValueError(f"Path escapes the workspace: {path!r}.")
    return str(root.joinpath(*parts))


def write_file_command(path: str, content: str) -> str:
    """Build a shell command that writes ``content`` to ``path`` exactly.

    The content is base64-encoded, so a proof containing quotes, backslashes,
    dollar signs, or a line that looks like a heredoc terminator cannot break
    the command or be partially interpreted by the shell. The parent directory
    is created first, because the protocol invites proofs split into modules
    under ``Submission/``, which the image does not ship.

    Args:
      path: Absolute destination path.
      content: Exact file contents.

    Returns:
      command: A command for a POSIX shell.

    """
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    parent = str(PurePosixPath(path).parent)
    return (
        f"mkdir -p {shlex.quote(parent)} && "
        f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
    )


def read_submission_command(submission_path: str, workspace: str) -> str:
    """Build a shell command that lists the graded files with their contents.

    The graded files are the submission and any modules under the directory of
    the same name, which the protocol lets a long proof split into. Each file is
    one output line, ``base64(path) TAB base64(contents)``, so no path or Lean
    text can break the listing; a file over :data:`SUBMISSION_MAX_FILE_BYTES`
    gets ``-`` in place of its contents. The command uses only POSIX ``sh``,
    ``find``, ``wc``, ``tr`` and ``base64``, and runs from the workspace, so the
    paths it prints are relative to it.

    Args:
      submission_path: The graded file, relative to the workspace.
      workspace: The task's working directory, absolute.

    Returns:
      command: A command for a POSIX shell; parse its output with
        :func:`parse_submission_listing`.

    """
    target = PurePosixPath(resolve_workspace_path(submission_path, workspace))
    relative = target.relative_to(PurePosixPath(workspace))
    per_file = (
        'for f; do printf "%s\\t" "$(printf %s "$f" | base64 | tr -d "\\n")"; '
        f'if [ "$(wc -c < "$f")" -le {SUBMISSION_MAX_FILE_BYTES} ]; '
        'then base64 < "$f" | tr -d "\\n"; else printf %s -; fi; '
        'printf "\\n"; done'
    )
    roots = " ".join(
        shlex.quote(str(root)) for root in (relative, relative.with_suffix(""))
    )
    # A missing module directory makes find exit nonzero after listing the
    # rest, so the exit code says nothing; the listing is the result.
    return (
        f"find {roots} -type f -name '*.lean' "
        f"-exec sh -c {shlex.quote(per_file)} sh {{}} + 2>/dev/null"
    )


def parse_submission_listing(output: str) -> dict[str, str | None]:
    """Decode the output of :func:`read_submission_command`.

    Args:
      output: The command's standard output.

    Returns:
      files: Each graded file's path and contents, sorted by path; ``None``
        for a file too large to record. A malformed line is skipped.

    """
    files: dict[str, str | None] = {}
    for line in output.splitlines():
        path_part, separator, body = line.partition("\t")
        if not separator:
            continue
        try:
            path = base64.b64decode(path_part, validate=True).decode("utf-8")
            text = (
                None
                if body == "-"
                else base64.b64decode(body, validate=True).decode(
                    "utf-8",
                    errors="replace",
                )
            )
        except (binascii.Error, UnicodeDecodeError):
            continue
        files[path] = text
    return dict(sorted(files.items()))


def submission_digest(files: Mapping[str, str | None]) -> str:
    """Return a SHA-256 over the graded files' paths and contents.

    Two attempts that left the same files get the same digest, which is what
    lets graded attempts be joined and deduplicated without carrying the text.
    """
    canonical = json.dumps(dict(sorted(files.items())), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
