"""Refactor executor — runs a transformation on the workspace between checkpoints.

Two interchangeable kinds:
  - ScriptRefactorExecutor: runs an arbitrary host subprocess
  - AgentRefactorExecutor: drives an SCBench Agent on the same session

Both mutate the workspace in place and then call session.finish_checkpoint so
the next feature checkpoint measures its diff against the refactored baseline.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

from slop_code import common
from slop_code.logging import get_logger

if TYPE_CHECKING:
    from slop_code.agent_runner.agent import Agent
    from slop_code.agent_runner.agent import AgentConfigBase
    from slop_code.agent_runner.credentials import ProviderCredential
    from slop_code.common.llms import ModelDefinition
    from slop_code.common.llms import ThinkingPreset
    from slop_code.execution import Session
    from slop_code.execution import SnapshotDiff

logger = get_logger(__name__)

# Refactor-on policy: which feature checkpoints trigger a refactor step.
# "all_but_last" is the default — ensures at least one feature checkpoint
# follows the refactor so we get a "churn after refactor" signal.
RefactorOnPolicy = str  # "all" | "all_but_last" | "last" | comma-sep names

REFACTOR_SUFFIX = "__refactor"


def should_refactor(
    on: RefactorOnPolicy,
    checkpoint_name: str,
    all_feature_checkpoint_names: list[str],
) -> bool:
    """Return True if a refactor step should follow this checkpoint."""
    if on == "all":
        return True
    if on == "all_but_last":
        return (
            not all_feature_checkpoint_names
            or checkpoint_name != all_feature_checkpoint_names[-1]
        )
    if on == "last":
        return (
            bool(all_feature_checkpoint_names)
            and checkpoint_name == all_feature_checkpoint_names[-1]
        )
    # Treat as comma-separated list of checkpoint names
    names = {n.strip() for n in on.split(",")}
    return checkpoint_name in names


# ---------------------------------------------------------------------------
# Specs (plain dataclasses — used as run-level config, not serialized to disk)
# ---------------------------------------------------------------------------


@dataclass
class ScriptRefactorSpec:
    """Run an arbitrary host script as the refactor step.

    Script contract:
      - Invoked as: command <target_dir> <artifacts_dir>
      - Mutates <target_dir> in place.
      - Exit 0 = success; nonzero = refactor failed.
      - May write <artifacts_dir>/usage.json with cost/step/token info.
    """

    command: str
    on: RefactorOnPolicy = "all_but_last"
    env_passthrough: list[str] = field(default_factory=list)
    timeout: int = 1800  # seconds


@dataclass
class AgentRefactorSpec:
    """Run an SCBench Agent as the refactor step on the same session."""

    agent_config: AgentConfigBase
    model_def: ModelDefinition
    credential: ProviderCredential
    prompt: str  # rendered prompt string (the refactor instruction)
    image: str
    on: RefactorOnPolicy = "all_but_last"
    verbose: bool = False
    thinking_preset: ThinkingPreset | None = None
    thinking_max_tokens: int | None = None


RefactorSpec = ScriptRefactorSpec | AgentRefactorSpec


# ---------------------------------------------------------------------------
# Executors
# ---------------------------------------------------------------------------


class RefactorError(Exception):
    """Raised when a refactor step fails."""


class ScriptRefactorExecutor:
    def __init__(self, spec: ScriptRefactorSpec) -> None:
        self._spec = spec

    def execute(self, session: Session, save_dir: Path) -> SnapshotDiff:
        """Run the script on the workspace and snapshot the result."""
        working_dir = session.workspace.working_dir
        artifacts_dir = save_dir / common.AGENT_DIR_NAME
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        for var in self._spec.env_passthrough:
            if var in os.environ:
                env[var] = os.environ[var]

        cmd = [self._spec.command, str(working_dir), str(artifacts_dir)]
        logger.info("Running script refactor", command=self._spec.command, working_dir=str(working_dir))

        stdout_log = artifacts_dir / "stdout.log"
        stderr_log = artifacts_dir / "stderr.log"

        try:
            with stdout_log.open("w") as out, stderr_log.open("w") as err:
                proc = subprocess.run(
                    cmd,
                    env=env,
                    timeout=self._spec.timeout,
                    stdout=out,
                    stderr=err,
                    cwd=str(working_dir),
                )
        except subprocess.TimeoutExpired as e:
            logger.error("Script refactor timed out", command=self._spec.command)
            raise RefactorError(f"Refactor script timed out after {self._spec.timeout}s") from e
        except Exception as e:
            logger.error("Script refactor failed to launch", command=self._spec.command, error=str(e))
            raise RefactorError(f"Refactor script failed to launch: {e}") from e

        if proc.returncode != 0:
            logger.error(
                "Script refactor returned nonzero exit code",
                returncode=proc.returncode,
                stderr_log=str(stderr_log),
            )
            raise RefactorError(f"Refactor script exited with code {proc.returncode}")

        snapshot_dir = save_dir / common.SNAPSHOT_DIR_NAME
        diff = session.finish_checkpoint(snapshot_dir)
        logger.info("Script refactor complete", diff=repr(diff))
        return diff


class AgentRefactorExecutor:
    def __init__(self, spec: AgentRefactorSpec, problem_name: str) -> None:
        self._spec = spec
        self._problem_name = problem_name

    def execute(self, session: Session, save_dir: Path) -> SnapshotDiff:
        """Run an SCBench Agent on the workspace and snapshot the result."""
        from slop_code.agent_runner.agent import Agent

        spec = self._spec
        agent: Agent = Agent.from_config(
            spec.agent_config,
            model=spec.model_def,
            credential=spec.credential,
            problem_name=self._problem_name,
            verbose=spec.verbose,
            image=spec.image,
            thinking_preset=spec.thinking_preset,
            thinking_max_tokens=spec.thinking_max_tokens,
        )

        snapshot_dir = save_dir / common.SNAPSHOT_DIR_NAME
        try:
            agent.setup(session=session)
            logger.info("Running agent refactor", problem=self._problem_name)
            agent.run_checkpoint(spec.prompt)
        except Exception as e:
            logger.error("Agent refactor inference error", error=str(e), exc_info=True)
            raise RefactorError(f"Agent refactor failed: {e}") from e
        finally:
            diff = session.finish_checkpoint(snapshot_dir)
            try:
                agent.cleanup()
            except Exception:
                logger.warning("Agent refactor cleanup failed", exc_info=True)

        logger.info("Agent refactor complete", diff=repr(diff))
        return diff


def make_executor(spec: RefactorSpec, problem_name: str) -> ScriptRefactorExecutor | AgentRefactorExecutor:
    if isinstance(spec, ScriptRefactorSpec):
        return ScriptRefactorExecutor(spec)
    return AgentRefactorExecutor(spec, problem_name)
