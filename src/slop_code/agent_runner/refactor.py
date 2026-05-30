"""Refactor executor — runs a transformation on the workspace between checkpoints.

Two interchangeable kinds:

  Script kind (ScriptRefactorExecutor / --refactor-command):
    Runs an arbitrary host subprocess with no SCBench agent involvement.
    The script drives whatever tools it likes (claude CLI, codex, custom binary, …)
    and mutates the workspace in place.  SCBench treats it as a black box.
    Configured via CLI --refactor-command; no prompt template needed.

  Agent kind (AgentRefactorExecutor / config YAML refactor.kind: agent):
    Spins up one of SCBench's registered agents (claude_code, codex, …) on the
    same session, running the refactor.jinja prompt template.  The agent is a
    full SCBench agent with usage/cost tracking, Docker container lifecycle, etc.
    Not exposed as a CLI flag; configure via a run config YAML.

Both kinds mutate the workspace in place and then call session.finish_checkpoint
so the next feature checkpoint measures its diff against the refactored baseline.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING

from slop_code import common
from slop_code.agent_runner.reporting import save_agent_artifacts
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

REFACTOR_SUFFIX = "__refactor"
REFACTOR_IDENTITY_FILENAME = "refactor_identity.json"

# Minimal host vars a refactor script needs just to launch (find interpreters,
# locate its config/credential dir, write temp files). Everything else from the
# host is withheld unless explicitly named in env_passthrough — see
# ScriptRefactorExecutor.execute.
_BASE_ENV_VARS = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TERM",
    "LANG",
)


def refactor_runs_after(
    name: str, all_names: list[str], spec: RefactorSpec | None
) -> bool:
    """Whether a refactor step is scheduled after checkpoint ``name``.

    The structural policy: a refactor runs after every checkpoint except the
    last, and only when a refactor spec is configured. This is *only* the
    scheduling rule — execution-time guards (e.g. skipping a checkpoint that
    errored) are the caller's concern and stay at the call site.
    """
    return spec is not None and name != all_names[-1]


def _hash_dir(root: Path | None) -> str:
    """Stable hash of a directory tree's contents, so swapping a refactorer's
    seeded ~/.claude (skills/commands/subagents) changes the refactor identity
    (and invalidates the resume cache)."""
    if root is None:
        return ""
    root = Path(root)
    h = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def compute_refactor_identity(spec: RefactorSpec) -> str:
    """Short hash identifying the refactor spec; used to invalidate resume cache on change.

    Script kind: hashes command.
    Agent kind:  hashes agent config type + model name + prompt content + the
                 seeded ~/.claude template directory's contents.
    """
    if isinstance(spec, ScriptRefactorSpec):
        content = f"script:{spec.command}"
    else:
        home_hash = _hash_dir(getattr(spec.agent_config, "claude_home", None))
        content = (
            f"agent:{type(spec.agent_config).__name__}:{spec.model_def.name}:"
            f"{spec.prompt}:{home_hash}"
        )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Specs (plain dataclasses — used as run-level config, not serialized to disk)
# ---------------------------------------------------------------------------


@dataclass
class ScriptRefactorSpec:
    """Run an arbitrary host script as the refactor step (script kind).

    The script is a black box — it drives whatever tools it likes (claude CLI,
    codex, a compiled binary, …) without any SCBench agent infrastructure.
    No prompt template is involved; the script owns its own prompting logic.

    Script contract:
      - Invoked as: command <target_dir> <artifacts_dir>
      - Mutates <target_dir> in place.
      - Exit 0 = success; nonzero = refactor failed.
      - May write <artifacts_dir>/usage.json with cost/step/token info.
    """

    command: str
    env_passthrough: list[str] = field(default_factory=list)
    timeout: int = 1800  # seconds


@dataclass
class AgentRefactorSpec:
    """Run an SCBench-registered agent as the refactor step (agent kind).

    Spins up a registered SCBench agent (claude_code, codex, …) on the same
    session, running it with the provided prompt (typically rendered from
    configs/prompts/refactor.jinja).  The agent has full usage/cost tracking
    and a Docker container lifecycle managed by SCBench.

    Configure via a run config YAML (refactor.kind: agent); not exposed as CLI
    flags since it requires agent/model/credential resolution done at load time.
    """

    agent_config: AgentConfigBase
    model_def: ModelDefinition
    credential: ProviderCredential
    prompt: str  # rendered prompt string (the refactor instruction)
    image: str
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
    # TODO(scbench): run the script inside a Docker container (via session.spawn / exec)  # noqa: TD003
    # instead of as a host subprocess, so an arbitrary orchestrator binary and any
    # coding-agent CLIs it spawns get the same sandbox the feature agents get
    # (no host filesystem/network access). The insulation invariant is unchanged:
    # the binary keeps its own ~/.claude outside working_dir, so nothing leaks to
    # the feature agent. Today this runs on the HOST (subprocess.run below).
    def __init__(self, spec: ScriptRefactorSpec) -> None:
        self._spec = spec

    def execute(self, session: Session, save_dir: Path) -> SnapshotDiff:
        """Run the script on the workspace and snapshot the result."""
        working_dir = session.workspace.working_dir
        artifacts_dir = save_dir / common.AGENT_DIR_NAME
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Strict whitelist: the script sees a minimal base plus only the vars the
        # caller explicitly passed through (e.g. --refactor-env ANTHROPIC_API_KEY).
        # Nothing else from the host leaks in.
        env = {
            var: os.environ[var]
            for var in (*_BASE_ENV_VARS, *self._spec.env_passthrough)
            if var in os.environ
        }
        # Forward locale vars so tool output/encoding stays sane.
        env.update({k: v for k, v in os.environ.items() if k.startswith("LC_")})

        cmd = [self._spec.command, str(working_dir), str(artifacts_dir)]
        logger.info(
            "Running script refactor",
            command=self._spec.command,
            working_dir=str(working_dir),
        )

        stdout_log = artifacts_dir / "stdout.log"
        stderr_log = artifacts_dir / "stderr.log"

        try:
            with stdout_log.open("w") as out, stderr_log.open("w") as err:
                proc = subprocess.run(  # noqa: S603  # command comes from trusted run config
                    cmd,
                    env=env,
                    timeout=self._spec.timeout,
                    stdout=out,
                    stderr=err,
                    cwd=str(working_dir),
                )
        except subprocess.TimeoutExpired as e:
            logger.error(
                "Script refactor timed out", command=self._spec.command
            )
            raise RefactorError(
                f"Refactor script timed out after {self._spec.timeout}s"
            ) from e
        except Exception as e:
            logger.error(
                "Script refactor failed to launch",
                command=self._spec.command,
                error=str(e),
            )
            raise RefactorError(f"Refactor script failed to launch: {e}") from e

        if proc.returncode != 0:
            logger.error(
                "Script refactor returned nonzero exit code",
                returncode=proc.returncode,
                stderr_log=str(stderr_log),
            )
            raise RefactorError(
                f"Refactor script exited with code {proc.returncode}"
            )

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
            logger.error(
                "Agent refactor inference error", error=str(e), exc_info=True
            )
            raise RefactorError(f"Agent refactor failed: {e}") from e
        finally:
            # Persist the refactor agent's native artifacts (stdout.jsonl stream,
            # traces) to save_dir/agent/ — the same trajectory the feature agent
            # saves per checkpoint. Best-effort and before cleanup so a partial
            # run is still inspectable; needed both for pilot analysis and so a
            # caller can verify which skill/subagent the refactor actually fired.
            try:
                save_agent_artifacts(save_dir, agent)
            except Exception:  # noqa: BLE001  # never mask the real refactor error
                logger.warning(
                    "Failed to save agent refactor artifacts", exc_info=True
                )
            try:
                agent.cleanup()
            except Exception:  # noqa: BLE001  # best-effort cleanup, never mask the real error
                logger.warning("Agent refactor cleanup failed", exc_info=True)

        # Snapshot only on success, matching ScriptRefactorExecutor: a failed refactor
        # must leave the pre-refactor baseline intact rather than re-baselining onto a
        # half-refactored workspace (which resume would later prefer and consume).
        diff = session.finish_checkpoint(snapshot_dir)
        logger.info("Agent refactor complete", diff=repr(diff))
        return diff


def make_executor(
    spec: RefactorSpec, problem_name: str
) -> ScriptRefactorExecutor | AgentRefactorExecutor:
    if isinstance(spec, ScriptRefactorSpec):
        return ScriptRefactorExecutor(spec)
    return AgentRefactorExecutor(spec, problem_name)
