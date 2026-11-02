r"""Run the openai/math episode loop inside a Harbor trial.

Harbor owns everything around an episode: it builds the sandbox from the
dataset's shared snapshot, runs the task's collect hook, starts the separate
verifier sandbox, and records the verdict. This module is the adapter that gives
Harbor a policy-driven agent instead of a CLI one, and it is deliberately thin.
The episode itself lives in :mod:`priml.baselines.openaimath.episode`, which
knows nothing about Harbor, so the loop is testable without a sandbox.

Harbor imports this module by path, which is how a baseline plugs a policy into
the dataset's own verifier:

    harbor run -p <tasks> -e daytona --ek snapshot_template_name=<snap> \\
      --agent-import-path \\
      priml.baselines.openaimath.scripts.agent:OpenMathLeanAgent \\
      --ak base_url=http://127.0.0.1:8000 --ak tokenizer_path=/models/Qwen3.5-4B \\
      -i <task> -k 32 -o <job dir>

Because that import path is resolved by Harbor, this module imports Harbor at
import time, which is why it lives in ``scripts/``: the ``scripts`` tree is the
one place a baseline module may need an optional runtime that is not a PriML
dependency. The rollout runner launches Harbor as a subprocess and reads its
output off disk, and the unit tests drive
:func:`priml.baselines.openaimath.episode.run_episode` through its own seams, so
nothing else in the baseline imports Harbor.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast, override

import json
import time

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities

from priml.baselines.openaimath.episode import (
    MAX_OBSERVATION_TOKENS,
    MAX_PROMPT_TOKENS,
    MAX_TURNS,
    PER_TURN_MAX_TOKENS,
    RESPONSE_LENGTH,
    TRANSCRIPT_NAME,
    Episode,
    EpisodeConfig,
    ProtocolError,
    parse_submission_listing,
    read_submission_command,
    resolve_workspace_path,
    run_episode,
    submission_digest,
    write_file_command,
)
from priml.baselines.openaimath.sampler import TransformersTokenizer, VLLMSampler


if TYPE_CHECKING:
    from collections.abc import Mapping

    from harbor.environments.base import BaseEnvironment, ExecResult
    from harbor.models.agent.context import AgentContext


CHALLENGE_CHARACTERS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
)
"""Characters allowed in a challenge name before it reaches a shell command."""

EXEC_TIMEOUT_SEC: Final = 3_600
"""Per-command timeout. The task allows the agent four hours in total."""

WRITE_TIMEOUT_SEC: Final = 120
"""Timeout for writing one file, which is a local copy inside the sandbox."""


class OpenMathLeanAgent(BaseAgent):
    """Prove an openai/math statement in Lean, driven by the learner's policy.

    The policy is reached over HTTP, not imported: the learner holds the GPUs in
    its own process and serves the checkpoint under training, and this agent runs
    in Harbor's trial process. Every token it samples is recorded with its
    behavior logprob so the update can be anchored on the policy that produced it.
    """

    capabilities = AgentCapabilities()

    @staticmethod
    @override
    def name() -> str:
        return "openaimath-lean"

    @override
    def version(self) -> str:
        return "1.0.0"

    def __init__(  # noqa: PLR0917 -- The signature mirrors Harbor's --ak option surface.
        self,
        logs_dir: Path,
        task_dir: Path | None = None,
        base_url: str = "http://127.0.0.1:8000",
        model: str | None = None,
        tokenizer_path: str | None = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
        max_turns: int = MAX_TURNS,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
        response_length: int = RESPONSE_LENGTH,
        per_turn_max_tokens: int = PER_TURN_MAX_TOKENS,
        max_observation_tokens: int = MAX_OBSERVATION_TOKENS,
        workspace: str = "/workspace",
        submission_path: str = "Submission.lean",
        model_name: str | None = None,
        **kwargs: object,
    ) -> None:
        """Bind the policy endpoint and the episode budgets.

        Args:
          logs_dir: Harbor's per-trial agent log directory on the host.
          task_dir: The task's directory on the host; its ``tests/`` holds the
            challenge name and the challenge file staged for the policy.
          base_url: Root of the learner's OpenAI-compatible rollout server.
          model: Served model name; falls back to Harbor's ``model_name``.
          tokenizer_path: The policy checkpoint or its tokenizer directory.
          temperature: Sampling temperature; the released recipe's 1.0.
          top_p: Nucleus sampling cutoff; the released recipe's 1.0.
          seed: Optional sampling seed.
          max_turns: Most sampled turns per episode.
          max_prompt_tokens: Token budget for the instruction and protocol.
          response_length: Token budget for the whole episode's policy tokens.
          per_turn_max_tokens: Token budget for one sampled turn.
          max_observation_tokens: Token budget for one sandbox observation.
          workspace: The task's working directory, a Lake project.
          submission_path: The file graded, relative to the workspace.
          model_name: Harbor's ``-m`` value.
          **kwargs: Remaining Harbor agent options, passed to the base class.

        Raises:
          ValueError: No tokenizer path was supplied.

        """
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if tokenizer_path is None:
            raise ValueError(
                "The openai/math agent needs --ak tokenizer_path=<checkpoint> so "
                "the learner scores the same token ids the policy sampled.",
            )
        self._task_dir = task_dir
        self._base_url = base_url
        self._model = model or model_name or "policy"
        self._tokenizer_path = tokenizer_path
        self._temperature = temperature
        self._top_p = top_p
        self._seed = seed
        self._config = EpisodeConfig(
            workspace=workspace,
            submission_path=submission_path,
            max_prompt_tokens=max_prompt_tokens,
            response_length=response_length,
            per_turn_max_tokens=per_turn_max_tokens,
            max_turns=max_turns,
            max_observation_tokens=max_observation_tokens,
        )

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        """Nothing to install: the snapshot already holds Lean and Mathlib."""
        return

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run one proving attempt and record its token stream.

        The challenge file is uploaded to the graded path before the first turn, so
        the policy starts from a buildable project and the statement cannot drift
        from the one the grader holds. Submission is the policy's own decision:
        nothing here submits on its behalf.

        Args:
          instruction: The task's instruction text.
          environment: The trial's sandbox.
          context: Populated with token counts and per-turn rollout details.

        """
        challenge = self._challenge()
        await self._stage_submission(environment)
        tokenizer = TransformersTokenizer(self._tokenizer_path)
        chat_format = tokenizer.chat_format()
        sandbox = HarborEpisodeEnvironment(environment, self._config)
        started_at = time.time()
        episode = await run_episode(
            instruction=instruction,
            tokenizer=tokenizer,
            sampler=VLLMSampler(
                self._base_url,
                self._model,
                stop_token_ids=tokenizer.encode(chat_format.end_of_turn),
                temperature=self._temperature,
                top_p=self._top_p,
                seed=self._seed,
            ),
            environment=sandbox,
            chat_format=chat_format,
            config=self._config,
        )
        finished_at = time.time()
        submission = await sandbox.read_submission()
        self._write_transcript(
            challenge,
            episode,
            sandbox.seconds,
            submission=submission,
            started_at=started_at,
            finished_at=finished_at,
        )
        self._populate_context(context, challenge, episode)

    def _tests_dir(self) -> Path:
        """Return the task's trusted ``tests/`` directory on the host.

        Raises:
          ValueError: Harbor did not pass the task directory.

        """
        if self._task_dir is None:
            raise ValueError(
                "The openai/math agent needs Harbor's task_dir: the challenge it "
                "stages is the task's own tests/Challenge.lean.",
            )
        return Path(self._task_dir) / "tests"

    def _challenge(self) -> str:
        """Read the challenge's name from the task's ``tests/task.json``.

        Returns:
          name: The challenge's name.

        Raises:
          TypeError: The file is not a JSON object.
          ValueError: It names no usable challenge.

        """
        path = self._tests_dir() / "task.json"
        payload = cast(object, json.loads(path.read_text(encoding="utf-8")))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} is not a JSON object.")
        name = str(cast(dict[str, object], payload).get("challenge") or "")
        if not name or not set(name) <= CHALLENGE_CHARACTERS:
            raise ValueError(f"Unusable challenge name in {path}: {name!r}.")
        return name

    async def _stage_submission(self, environment: BaseEnvironment) -> None:
        """Upload the task's own challenge file to the graded path.

        The source is the task's ``tests/Challenge.lean``, the exact file the
        grader compares against, rather than a copy baked into the image: a
        lemma task's challenge exists only in its task directory, and for a
        published task the two are the same file. Staging is setup, not an
        action the policy chose, so it does not consume a turn.

        Args:
          environment: The trial's sandbox.

        """
        target = resolve_workspace_path(
            self._config.submission_path,
            self._config.workspace,
        )
        await environment.upload_file(self._tests_dir() / "Challenge.lean", target)

    def _write_transcript(
        self,
        challenge: str,
        episode: Episode,
        sandbox_seconds: float,
        *,
        submission: Mapping[str, str | None] | None,
        started_at: float,
        finished_at: float,
    ) -> None:
        """Write the episode's flattened record into the trial's agent directory.

        Besides the tokens the learner trains on, the record keeps the graded
        files as the episode left them, so a verdict can later be re-checked or
        learned from without replaying the episode, and the episode's wall-clock
        bounds, in seconds since the epoch.
        """
        record: dict[str, object] = {
            "challenge": challenge,
            "started_at": started_at,
            "finished_at": finished_at,
            "sandbox_seconds": sandbox_seconds,
            "submission_files": None if submission is None else dict(submission),
            "submission_sha256": (
                None if submission is None else submission_digest(submission)
            ),
            "session_id": self.session_id,
            "context_id": str(self.context_id) if self.context_id else None,
            "prompt_tokens": list(episode.prompt_tokens),
            "response_tokens": list(episode.response_tokens),
            "logprobs": list(episode.logprobs),
            "tool_mask": list(episode.tool_mask),
            "finish_reason": episode.finish_reason,
            "turns": episode.turns,
            "submitted": episode.submitted,
        }
        path = Path(self.logs_dir) / TRANSCRIPT_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _populate_context(
        self,
        context: AgentContext,
        challenge: str,
        episode: Episode,
    ) -> None:
        """Report token counts and the per-turn rollout details to Harbor."""
        context.n_input_tokens = len(episode.prompt_tokens)
        context.n_output_tokens = episode.policy_token_count
        context.rollout_details = [
            {
                "prompt_token_ids": [list(episode.prompt_tokens)],
                "completion_token_ids": [list(episode.response_tokens)],
                "logprobs": [list(episode.logprobs)],
                "extra": {"tool_mask": [list(episode.tool_mask)]},
            },
        ]
        context.metadata = {
            "challenge": challenge,
            "turns": episode.turns,
            "submitted": episode.submitted,
            "finish_reason": episode.finish_reason,
        }


class HarborEpisodeEnvironment:
    """A Harbor sandbox, narrowed to what one episode needs."""

    def __init__(self, environment: BaseEnvironment, config: EpisodeConfig) -> None:
        """Bind the sandbox and the workspace the episode acts in."""
        self._environment = environment
        self._config = config
        self.seconds = 0.0
        """Wall-clock seconds spent in sandbox commands, mostly Lean builds."""

    async def run(self, command: str) -> str:
        """Run a shell command in the workspace and return its combined output.

        A failing command is an observation, not an error: a compiler error is
        the most useful thing the policy can read.

        Args:
          command: The command from a ``RUN:`` directive.

        Returns:
          output: Standard output and error, with the exit code when nonzero.

        """
        result = await self._run(command)
        output = f"{result.stdout or ''}{result.stderr or ''}"
        if result.return_code != 0:
            output = f"{output}\n[exit code {result.return_code}]"
        return output

    async def write_file(self, path: str, content: str) -> None:
        """Write a file inside the workspace.

        Args:
          path: The path from an ``EDIT:`` directive, resolved inside the
            workspace.
          content: Exact file contents, base64-encoded through the shell so no
            Lean syntax can be interpreted by it.

        Raises:
          ProtocolError: The path escapes the workspace or the write failed.

        """
        try:
            target = resolve_workspace_path(path, self._config.workspace)
        except ValueError as error:
            raise ProtocolError(str(error)) from error
        result = await self._run(
            write_file_command(target, content),
            timeout_sec=WRITE_TIMEOUT_SEC,
        )
        if result.return_code != 0:
            raise ProtocolError(
                f"could not write {path}: "
                f"{result.stderr or result.stdout or 'no output'}",
            )

    async def read_submission(self) -> dict[str, str | None] | None:
        """Return the graded files as the episode left them.

        Not timed into :attr:`seconds`, which counts the policy's own commands.

        Returns:
          files: Each graded file's path and contents (see
            :func:`parse_submission_listing`), or ``None`` when the sandbox could
            not be read.

        """
        try:
            result = await self._environment.exec(
                command=read_submission_command(
                    self._config.submission_path,
                    self._config.workspace,
                ),
                cwd=self._config.workspace,
                timeout_sec=WRITE_TIMEOUT_SEC,
            )
        except Exception:  # noqa: BLE001 -- The listing is a record, not part of the attempt; failing here would cost the trial its transcript.
            return None
        return parse_submission_listing(result.stdout or "")

    async def _run(
        self,
        command: str,
        *,
        timeout_sec: float = EXEC_TIMEOUT_SEC,
    ) -> ExecResult:
        """Run one command in the episode's workspace, timing it."""
        started = time.monotonic()
        try:
            return await self._environment.exec(
                command=command,
                cwd=self._config.workspace,
                timeout_sec=int(timeout_sec),
            )
        finally:
            self.seconds += time.monotonic() - started
