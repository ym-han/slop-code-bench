"""Hypothesis property-based tests for the interleaved refactoring pipeline.

Spec reference: cc-notes/test/intermediate/spec_v2.md

Properties tested:

§A (the rolling-baseline property):
  - Every step's diff equals diff(B_step, T_step) where B_step is the workspace
    tree at entry and T_step is the workspace tree at exit.
  - Consequence: each step's diff created/modified/deleted sets match exactly the
    file operations chosen by the plan (given the running tree at that point).
  - Consequence: the next step's baseline is exactly the prior step's exit tree,
    so files a refactor wrote do NOT reappear as created in the subsequent
    checkpoint's diff (no phantom re-creation, §A D2).

§H (observable boundary conditions):
  - Text-only: generated files are decodable text, so the file-set mapping is exact.
  - Content equality: a "modify" must change bytes; identical overwrites produce no
    diff entry.
  - Snapshot accumulation: files created in earlier steps persist in every later
    snapshot, even after they stop appearing in diffs.

§I.2 (consistency / invalidation via _apply_refactor_consistency):
  - The first inconsistent c_i whose completed successor c_{i+1} triggers exactly
    c_{i+1} onward being invalidated (REFACTOR_CHANGED, then DEPENDS_ON_INVALID).
  - CI6: the triggering c_i stays completed.
  - CI7: no-successor inconsistency invalidates nothing.
  - CI8: successor not completed — cascade blocked.

§B (scheduling, pure):
  - With a spec: exactly the non-last names return True from refactor_runs_after.
  - Without a spec: all names return False.

§C (identity, pure):
  - Two specs with the same command have the same hash.
  - Changing the command changes the hash.
  - The hash is always 16 lowercase hex chars.

Implementation notes:
  - Tests use tempfile.TemporaryDirectory() inside the test body rather than the
    tmp_path fixture.  This is required because @given runs the test body multiple
    times against the same pytest fixture instance, which would cause directory
    collisions and health-check failures.
  - All tests are marked @pytest.mark.hypothesis; E2E pipeline tests are also
    marked @pytest.mark.slow since they spin up a local session.
"""

from __future__ import annotations

import json
import queue
import stat
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from slop_code.agent_runner import runner
from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentRunSpec
from slop_code.agent_runner.refactor import REFACTOR_IDENTITY_FILENAME
from slop_code.agent_runner.refactor import REFACTOR_SUFFIX
from slop_code.agent_runner.refactor import ScriptRefactorSpec
from slop_code.agent_runner.refactor import compute_refactor_identity
from slop_code.agent_runner.refactor import refactor_runs_after
from slop_code.agent_runner.resume import CheckpointStatus
from slop_code.agent_runner.resume import InvalidationReason
from slop_code.agent_runner.resume import _apply_refactor_consistency
from slop_code.common import DIFF_FILENAME
from slop_code.common import SNAPSHOT_DIR_NAME
from slop_code.evaluation import PassPolicy
from slop_code.evaluation import ProblemConfig
from slop_code.evaluation.config import CheckpointConfig

# ---------------------------------------------------------------------------
# Helpers (mirror of those in test_refactor_interleaved.py; copied here because
# tests/ has no __init__.py files — it is rootdir-based, not a package).
# ---------------------------------------------------------------------------


class DummyAgent(Agent):
    """A no-op agent that writes pre-canned files for each checkpoint."""

    def __init__(
        self,
        checkpoint_solutions: list[dict[str, str]],
        *,
        error_on_checkpoint: int | None = None,
    ) -> None:
        super().__init__(
            "dummy",
            "dummy",
            AgentCostLimits(step_limit=100, cost_limit=100, net_cost_limit=100),
            None,
            False,
        )
        self.checkpoint_solutions = checkpoint_solutions
        self.working_dir: Path | None = None
        self._chkpt_num = 0
        self._error_on_checkpoint = error_on_checkpoint

    def setup(self, session) -> None:
        self.working_dir = session.working_dir

    def run(self, task: str) -> None:
        pass

    def run_checkpoint(self, task: str):  # type: ignore[override]
        from datetime import datetime

        from slop_code.agent_runner.agent import CheckpointInferenceResult

        assert self.working_dir is not None
        chkpt = self._chkpt_num
        if self._error_on_checkpoint == chkpt:
            self._chkpt_num += 1
            now = datetime.now()
            return CheckpointInferenceResult(
                started=now,
                completed=now,
                elapsed=0.0,
                usage=self.usage.model_copy(deep=True),
                had_error=True,
                error_message="intentional test error",
            )
        for fname, content in self.checkpoint_solutions[chkpt].items():
            (self.working_dir / fname).write_text(content)
        self._chkpt_num += 1
        now = datetime.now()
        return CheckpointInferenceResult(
            started=now,
            completed=now,
            elapsed=0.0,
            usage=self.usage.model_copy(deep=True),
            had_error=False,
        )

    def supports_replay(self) -> bool:
        return False

    def reset(self) -> None:
        pass

    def save_artifacts(self, path: Path) -> None:
        pass

    def cleanup(self) -> None:
        pass

    @classmethod
    def _from_config(cls, config, problem_name, verbose, image):
        raise NotImplementedError


def _make_run_spec(problem: ProblemConfig, env) -> AgentRunSpec:
    return AgentRunSpec(
        seed=0,
        template="{{task}}",
        problem=problem,
        environment=env,
        image="test-image",
        pass_policy=PassPolicy.ANY,
        skip_evaluation=True,
        verbose=False,
    )


def _checkpoint_names_list(problem: ProblemConfig) -> list[str]:
    return list(problem.checkpoints.keys())


def _created_paths(diff_json: dict) -> set[str]:
    file_diffs: dict[str, dict] = diff_json.get("file_diffs", {})
    return {
        Path(path).name
        for path, fd in file_diffs.items()
        if fd.get("change_type") == "created"
    }


def _modified_paths(diff_json: dict) -> set[str]:
    file_diffs: dict[str, dict] = diff_json.get("file_diffs", {})
    return {
        Path(path).name
        for path, fd in file_diffs.items()
        if fd.get("change_type") == "modified"
    }


def _deleted_paths(diff_json: dict) -> set[str]:
    file_diffs: dict[str, dict] = diff_json.get("file_diffs", {})
    return {
        Path(path).name
        for path, fd in file_diffs.items()
        if fd.get("change_type") == "deleted"
    }


def _read_diff(output_dir: Path, step_dir_name: str) -> dict:
    diff_path = output_dir / step_dir_name / DIFF_FILENAME
    assert diff_path.exists(), f"diff.json missing under {step_dir_name!r}"
    return json.loads(diff_path.read_text())


def _snapshot_dir(output_dir: Path, step_dir_name: str) -> Path:
    return output_dir / step_dir_name / SNAPSHOT_DIR_NAME


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

_DOCKER_ENV_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "environments"
    / "docker-python3.12-uv.yaml"
)


def _docker_available() -> bool:
    try:
        import docker as docker_sdk

        client = docker_sdk.from_env()
        client.ping()
        client.close()
        return True
    except Exception:
        return False


def _load_docker_env():
    from slop_code.execution.docker_runtime import DockerEnvironmentSpec

    with _DOCKER_ENV_PATH.open() as f:
        config = yaml.safe_load(f)
    return DockerEnvironmentSpec(**config)


def cleanup_as_root(path: Path) -> None:
    if not path.exists():
        return
    subprocess.run(  # noqa: S603
        [  # noqa: S607
            "docker",
            "run",
            "--rm",
            "-v",
            f"{path}:/cleanup",
            "alpine:latest",
            "rm",
            "-rf",
            "/cleanup",
        ],
        capture_output=True,
        check=False,
    )
    if path.exists():
        path.rmdir()


# ---------------------------------------------------------------------------
# Problem factory
# ---------------------------------------------------------------------------


def _make_n_chkpt_problem(base_path: Path, n: int) -> ProblemConfig:
    """Synthesize a minimal N-checkpoint ProblemConfig in-memory."""
    assert n >= 1
    checkpoints = {
        f"checkpoint_{i}": CheckpointConfig(
            name=f"checkpoint_{i}",
            version=1,
            order=i,
            spec_override=f"spec for c{i}",
        )
        for i in range(1, n + 1)
    }
    return ProblemConfig(
        name=f"{n}_checkpoint_test_problem",
        path=base_path,
        version=1,
        description=f"Synthetic {n}-checkpoint problem for hypothesis tests.",
        tags=["test"],
        entry_file="solution",
        checkpoints=checkpoints,
    )


# ---------------------------------------------------------------------------
# Script helper
# ---------------------------------------------------------------------------


def _write_script(path: Path, body: str) -> str:
    """Write an executable shell script; return its path as a string."""
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


# ---------------------------------------------------------------------------
# Shared constants / strategies
# ---------------------------------------------------------------------------

# Small universe of filenames that avoid all default ignore globs and commas.
_FILENAME_POOL = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"]

# Text content that is guaranteed to be decodable ASCII (letters, digits, punctuation).
_text_content = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=1,
    max_size=40,
)


def _step_plan(
    existing_files: list[str], *, allow_deletes: bool = False
) -> st.SearchStrategy:
    """Strategy for one step's file operations.

    Returns a dict:
        {"creates": {fname: content}, "modifies": {fname: new_content}, "deletes": set[str]}

    "creates" only touches filenames NOT already in `existing_files`.
    "modifies" only touches filenames already in `existing_files`.
    "deletes" is a subset of `existing_files` (disjoint from "modifies").
    When `allow_deletes=False` (default), "deletes" is always empty — this keeps
    DummyAgent-only steps (feature checkpoints) simple, since DummyAgent cannot delete.
    Pass `allow_deletes=True` for refactor steps driven by the shell script side.
    """
    available_to_create = [f for f in _FILENAME_POOL if f not in existing_files]
    available_to_modify = list(existing_files)

    creates_strategy = st.fixed_dictionaries(
        {},
        optional=dict.fromkeys(available_to_create, _text_content),
    )

    if available_to_modify:
        modifies_strategy = st.fixed_dictionaries(
            {},
            optional=dict.fromkeys(available_to_modify, _text_content),
        )
    else:
        modifies_strategy = st.just({})

    if allow_deletes and available_to_modify:
        # Draw a subset of existing files to delete (may overlap "modifies" candidates,
        # but we enforce disjointness in post-processing below).
        deletes_strategy: st.SearchStrategy = st.frozensets(
            st.sampled_from(available_to_modify),
            max_size=len(available_to_modify),
        )
    else:
        deletes_strategy = st.just(frozenset())

    def _build(triple: tuple) -> dict:
        creates, modifies, deletes = triple
        # Enforce disjointness: a file being deleted cannot also be in modifies.
        modifies_filtered = {
            k: v for k, v in modifies.items() if k not in deletes
        }
        return {
            "creates": creates,
            "modifies": modifies_filtered,
            "deletes": set(deletes),
        }

    return st.tuples(creates_strategy, modifies_strategy, deletes_strategy).map(
        _build
    )


# ---------------------------------------------------------------------------
# §A + §H — rolling-baseline property: c1→r1→c2 pipeline
# ---------------------------------------------------------------------------


@pytest.mark.hypothesis
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _docker_available(), reason="Docker not available")
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=st.data())
def test_rolling_baseline_diff_invariant(data: st.DataObject) -> None:
    """§A rolling-baseline property over a c1→r1→c2 run.

    For each step, we track the "expected tree" as a dict {fname: content} and
    compare it against the actual diff produced by the pipeline.

    Invariants:
    1. c1 diff created-set == files c1 wrote (starts from empty tree).
    2. r1 diff created-set == files r1 created (not previously in tree).
       r1 diff modified-set == files r1 overwrote with *different* content.
       r1 diff deleted-set  == files r1 deleted (§A change-type function).
    3. c2 diff created-set == files c2 created (not in r1's exit tree).
       c2 diff modified-set == files c2 modified (different content than r1's exit).
    4. No phantom re-creation (§A D2): no file from r1's exit tree appears as
       'created' in c2's diff.
    5. Snapshot accumulation (§H): all files in r1's exit tree persist in c2's
       snapshot directory.  Files r1 deleted are absent from c2's snapshot.
    """
    from hypothesis import note

    # --- Build plan ----------------------------------------------------------
    c1_plan = data.draw(_step_plan([]), label="c1_plan")
    assume(c1_plan["creates"])  # c1 must write at least one file

    running_tree: dict[str, str] = {}
    running_tree.update(c1_plan["creates"])
    c1_creates = set(c1_plan["creates"])

    # r1 may delete files from the tree — only the refactor script side can do this.
    r1_plan = data.draw(
        _step_plan(list(running_tree), allow_deletes=True), label="r1_plan"
    )

    r1_new_creates: set[str] = set(r1_plan["creates"])
    r1_actual_modifies: set[str] = set()
    for fname, new_content in r1_plan["modifies"].items():
        if running_tree.get(fname) != new_content:
            r1_actual_modifies.add(fname)
    r1_deletes: set[str] = set(r1_plan["deletes"])

    running_tree.update(r1_plan["creates"])
    for fname, new_content in r1_plan["modifies"].items():
        running_tree[fname] = new_content
    for fname in r1_deletes:
        running_tree.pop(fname, None)

    tree_after_r1 = dict(running_tree)

    c2_plan = data.draw(_step_plan(list(running_tree)), label="c2_plan")
    c2_creates: set[str] = set(c2_plan["creates"])
    c2_actual_modifies: set[str] = set()
    for fname, new_content in c2_plan["modifies"].items():
        if tree_after_r1.get(fname) != new_content:
            c2_actual_modifies.add(fname)

    note(
        f"Plan summary:\n"
        f"  c1 creates={set(c1_plan['creates'])}\n"
        f"  r1 creates={r1_new_creates}, modifies={r1_actual_modifies}, deletes={r1_deletes}\n"
        f"  c2 creates={c2_creates}, modifies={c2_actual_modifies}\n"
        f"  tree_after_r1={set(tree_after_r1)}"
    )

    # --- Build solutions for DummyAgent --------------------------------------
    c1_solution: dict[str, str] = dict(c1_plan["creates"])

    c2_solution: dict[str, str] = dict(c2_plan["creates"])
    for fname, new_content in c2_plan["modifies"].items():
        if tree_after_r1.get(fname) != new_content:
            c2_solution[fname] = new_content

    # --- Build r1 script -----------------------------------------------------
    script_lines: list[str] = ['working_dir="$1"']
    for fname, content in r1_plan["creates"].items():
        safe = content.replace("'", "'\\''")
        script_lines.append(f"printf '%s' '{safe}' > \"$working_dir/{fname}\"")
    for fname, new_content in r1_plan["modifies"].items():
        safe = new_content.replace("'", "'\\''")
        script_lines.append(f"printf '%s' '{safe}' > \"$working_dir/{fname}\"")
    for fname in r1_deletes:
        script_lines.append(f'rm -f "$working_dir/{fname}"')
    script_body = "\n".join(script_lines)

    # --- Run pipeline in a fresh temp directory per example ------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        problem_dir = tmp / "problem"
        problem_dir.mkdir()
        problem = _make_n_chkpt_problem(problem_dir, 2)

        script_str = _write_script(tmp / "r1.sh", script_body)
        spec = ScriptRefactorSpec(command=script_str)

        output_dir = tmp / "output"
        output_dir.mkdir()
        docker_env = _load_docker_env()
        run_spec = _make_run_spec(problem, docker_env)
        agent = DummyAgent([c1_solution, c2_solution])

        try:
            runner.run_agent(
                run_spec=run_spec,
                agent=agent,
                output_path=output_dir,
                progress_queue=queue.Queue(),
                refactor_spec=spec,
            )
        except Exception:
            cleanup_as_root(output_dir)
            raise

        names = _checkpoint_names_list(problem)
        c1_name, c2_name = names[0], names[1]
        r1_dir_name = f"{c1_name}{REFACTOR_SUFFIX}"

        # 1. c1 diff: created-set == c1_creates
        c1_diff = _read_diff(output_dir, c1_name)
        actual_c1_creates = _created_paths(c1_diff)
        assert actual_c1_creates == c1_creates, (
            f"c1 diff created-set mismatch: expected {c1_creates}, got {actual_c1_creates}"
        )
        assert _modified_paths(c1_diff) == set(), (
            f"c1 diff unexpectedly has modified files: {_modified_paths(c1_diff)}"
        )

        # 2. r1 diff: created == r1_new_creates, modified == r1_actual_modifies,
        #             deleted == r1_deletes
        r1_diff = _read_diff(output_dir, r1_dir_name)
        assert _created_paths(r1_diff) == r1_new_creates, (
            f"r1 diff created-set: expected {r1_new_creates}, got {_created_paths(r1_diff)}"
        )
        assert _modified_paths(r1_diff) == r1_actual_modifies, (
            f"r1 diff modified-set: expected {r1_actual_modifies}, got {_modified_paths(r1_diff)}"
        )
        assert _deleted_paths(r1_diff) == r1_deletes, (
            f"r1 diff deleted-set: expected {r1_deletes}, got {_deleted_paths(r1_diff)}"
        )

        # 3. c2 diff: created == c2_creates, modified == c2_actual_modifies
        c2_diff = _read_diff(output_dir, c2_name)
        assert _created_paths(c2_diff) == c2_creates, (
            f"c2 diff created-set: expected {c2_creates}, got {_created_paths(c2_diff)}"
        )
        assert _modified_paths(c2_diff) == c2_actual_modifies, (
            f"c2 diff modified-set: expected {c2_actual_modifies}, got {_modified_paths(c2_diff)}"
        )

        # 4. No phantom re-creation (§A D2)
        phantom = _created_paths(c2_diff) & set(tree_after_r1)
        assert phantom == set(), (
            f"Phantom re-creation: {phantom} appeared as 'created' in c2's diff but "
            f"were already in the tree after r1 — c2 was diffed against the pre-refactor baseline"
        )

        # 5. Snapshot accumulation (§H): surviving files present, deleted files absent
        #    (unless c2 re-created them, in which case they are legitimately back).
        c2_snap = _snapshot_dir(output_dir, c2_name)
        for fname in tree_after_r1:
            assert (c2_snap / fname).exists(), (
                f"Snapshot accumulation violated: '{fname}' was in tree after r1 "
                f"but is missing from c2's snapshot"
            )
        for fname in r1_deletes - c2_creates:
            assert not (c2_snap / fname).exists(), (
                f"Deleted file '{fname}' still appears in c2's snapshot after r1 deleted it "
                f"and c2 did not re-create it"
            )

        cleanup_as_root(output_dir)


# ---------------------------------------------------------------------------
# §A + §H — rolling-baseline property: c1→r1→c2→r2→c3 pipeline (two refactors)
# ---------------------------------------------------------------------------


@pytest.mark.hypothesis
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _docker_available(), reason="Docker not available")
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(data=st.data())
def test_two_refactor_steps_rolling_baseline(data: st.DataObject) -> None:
    """§A rolling-baseline property over a c1→r1→c2→r2→c3 run (two refactor steps).

    Extends test_rolling_baseline_diff_invariant to cover two consecutive refactor
    boundaries.  The single ScriptRefactorSpec script dispatches on invocation order
    using a sentinel file in a stable tmp path baked into the script body:
    first call applies r1 operations; second call applies r2 operations.

    Invariants checked:
    1. c1/r1/c2/r2/c3 diffs each match their plan exactly (created/modified/deleted).
    2. No phantom re-creation across either refactor boundary (§A D2).
    3. Snapshot accumulation: c3's snapshot contains every file surviving the full chain.
    4. Files deleted by r1 or r2 (and not re-created) are absent from c3's snapshot.
    """
    from hypothesis import note

    # ------------------------------------------------------------------ plan --
    c1_plan = data.draw(_step_plan([]), label="c1_plan")
    assume(c1_plan["creates"])  # starting tree must be non-empty

    running_tree: dict[str, str] = {}
    running_tree.update(c1_plan["creates"])
    c1_creates = set(c1_plan["creates"])

    r1_plan = data.draw(
        _step_plan(list(running_tree), allow_deletes=True), label="r1_plan"
    )
    r1_new_creates: set[str] = set(r1_plan["creates"])
    r1_actual_modifies: set[str] = {
        fname
        for fname, new_content in r1_plan["modifies"].items()
        if running_tree.get(fname) != new_content
    }
    r1_deletes: set[str] = set(r1_plan["deletes"])

    running_tree.update(r1_plan["creates"])
    for fname, new_content in r1_plan["modifies"].items():
        running_tree[fname] = new_content
    for fname in r1_deletes:
        running_tree.pop(fname, None)
    tree_after_r1 = dict(running_tree)

    c2_plan = data.draw(_step_plan(list(running_tree)), label="c2_plan")
    c2_creates: set[str] = set(c2_plan["creates"])
    c2_actual_modifies: set[str] = {
        fname
        for fname, new_content in c2_plan["modifies"].items()
        if tree_after_r1.get(fname) != new_content
    }

    running_tree.update(c2_plan["creates"])
    for fname, new_content in c2_plan["modifies"].items():
        running_tree[fname] = new_content
    tree_after_c2 = dict(running_tree)

    r2_plan = data.draw(
        _step_plan(list(running_tree), allow_deletes=True), label="r2_plan"
    )
    r2_new_creates: set[str] = set(r2_plan["creates"])
    r2_actual_modifies: set[str] = {
        fname
        for fname, new_content in r2_plan["modifies"].items()
        if tree_after_c2.get(fname) != new_content
    }
    r2_deletes: set[str] = set(r2_plan["deletes"])

    running_tree.update(r2_plan["creates"])
    for fname, new_content in r2_plan["modifies"].items():
        running_tree[fname] = new_content
    for fname in r2_deletes:
        running_tree.pop(fname, None)
    tree_after_r2 = dict(running_tree)

    c3_plan = data.draw(_step_plan(list(running_tree)), label="c3_plan")
    c3_creates: set[str] = set(c3_plan["creates"])
    c3_actual_modifies: set[str] = {
        fname
        for fname, new_content in c3_plan["modifies"].items()
        if tree_after_r2.get(fname) != new_content
    }

    note(
        f"Plan summary:\n"
        f"  c1 creates={c1_creates}\n"
        f"  r1 creates={r1_new_creates}, modifies={r1_actual_modifies}, deletes={r1_deletes}\n"
        f"  c2 creates={c2_creates}, modifies={c2_actual_modifies}\n"
        f"  r2 creates={r2_new_creates}, modifies={r2_actual_modifies}, deletes={r2_deletes}\n"
        f"  c3 creates={c3_creates}, modifies={c3_actual_modifies}\n"
        f"  tree_after_r1={set(tree_after_r1)}\n"
        f"  tree_after_r2={set(tree_after_r2)}"
    )

    # --------------------------------------------------- DummyAgent solutions --
    c1_solution: dict[str, str] = dict(c1_plan["creates"])

    c2_solution: dict[str, str] = dict(c2_plan["creates"])
    for fname, new_content in c2_plan["modifies"].items():
        if tree_after_r1.get(fname) != new_content:
            c2_solution[fname] = new_content

    c3_solution: dict[str, str] = dict(c3_plan["creates"])
    for fname, new_content in c3_plan["modifies"].items():
        if tree_after_r2.get(fname) != new_content:
            c3_solution[fname] = new_content

    # ---------------------------------------------- dispatch script (r1 + r2) --
    # The script is invoked twice with the same command (once after c1 for r1,
    # once after c2 for r2).  It distinguishes invocations via a sentinel file
    # in a stable tmp path baked into the script body — entirely outside $1 so
    # the sentinel never appears in workspace diffs.
    def _ops_lines(plan: dict, varname: str = "working_dir") -> list[str]:
        lines: list[str] = []
        for fname, content in plan["creates"].items():
            safe = content.replace("'", "'\\''")
            lines.append(f"printf '%s' '{safe}' > \"${varname}/{fname}\"")
        for fname, new_content in plan["modifies"].items():
            safe = new_content.replace("'", "'\\''")
            lines.append(f"printf '%s' '{safe}' > \"${varname}/{fname}\"")
        for fname in plan["deletes"]:
            lines.append(f'rm -f "${varname}/{fname}"')
        return lines

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Sentinel lives inside the workspace under an ignored path (.ruff_cache/*)
        # so it never appears in diffs, yet survives across Docker invocations.
        sentinel_rel = ".ruff_cache/.refactor_sentinel"

        r1_lines = _ops_lines(r1_plan)
        r2_lines = _ops_lines(r2_plan)

        script_parts = ['working_dir="$1"']
        script_parts.append(f'sentinel="$working_dir/{sentinel_rel}"')
        script_parts.append('mkdir -p "$(dirname "$sentinel")"')
        script_parts.append('if [ ! -f "$sentinel" ]; then')
        script_parts.append('  touch "$sentinel"')
        # r1 operations (": " is a no-op so the branch is never empty)
        script_parts.extend(f"  {line}" for line in r1_lines)
        script_parts.append("  :")
        script_parts.append("else")
        # r2 operations
        script_parts.extend(f"  {line}" for line in r2_lines)
        script_parts.append("  :")
        script_parts.append("fi")
        script_body = "\n".join(script_parts)

        problem_dir = tmp / "problem"
        problem_dir.mkdir()
        problem = _make_n_chkpt_problem(problem_dir, 3)

        script_str = _write_script(tmp / "refactor.sh", script_body)
        spec = ScriptRefactorSpec(command=script_str)

        output_dir = tmp / "output"
        output_dir.mkdir()
        docker_env = _load_docker_env()
        run_spec = _make_run_spec(problem, docker_env)
        agent = DummyAgent([c1_solution, c2_solution, c3_solution])

        try:
            runner.run_agent(
                run_spec=run_spec,
                agent=agent,
                output_path=output_dir,
                progress_queue=queue.Queue(),
                refactor_spec=spec,
            )
        except Exception:
            cleanup_as_root(output_dir)
            raise

        names = _checkpoint_names_list(problem)
        c1_name, c2_name, c3_name = names[0], names[1], names[2]
        r1_dir_name = f"{c1_name}{REFACTOR_SUFFIX}"
        r2_dir_name = f"{c2_name}{REFACTOR_SUFFIX}"

        # 1. c1 diff
        c1_diff = _read_diff(output_dir, c1_name)
        assert _created_paths(c1_diff) == c1_creates, (
            f"c1 diff created-set mismatch: expected {c1_creates}, got {_created_paths(c1_diff)}"
        )
        assert _modified_paths(c1_diff) == set(), (
            f"c1 diff unexpectedly has modified files: {_modified_paths(c1_diff)}"
        )

        # 2. r1 diff
        r1_diff = _read_diff(output_dir, r1_dir_name)
        assert _created_paths(r1_diff) == r1_new_creates, (
            f"r1 diff created-set: expected {r1_new_creates}, got {_created_paths(r1_diff)}"
        )
        assert _modified_paths(r1_diff) == r1_actual_modifies, (
            f"r1 diff modified-set: expected {r1_actual_modifies}, got {_modified_paths(r1_diff)}"
        )
        assert _deleted_paths(r1_diff) == r1_deletes, (
            f"r1 diff deleted-set: expected {r1_deletes}, got {_deleted_paths(r1_diff)}"
        )

        # 3. c2 diff — no phantom re-creation across r1 boundary
        c2_diff = _read_diff(output_dir, c2_name)
        assert _created_paths(c2_diff) == c2_creates, (
            f"c2 diff created-set: expected {c2_creates}, got {_created_paths(c2_diff)}"
        )
        assert _modified_paths(c2_diff) == c2_actual_modifies, (
            f"c2 diff modified-set: expected {c2_actual_modifies}, got {_modified_paths(c2_diff)}"
        )
        phantom_r1 = _created_paths(c2_diff) & set(tree_after_r1)
        assert phantom_r1 == set(), (
            f"Phantom re-creation across r1→c2 boundary: {phantom_r1}"
        )

        # 4. r2 diff
        r2_diff = _read_diff(output_dir, r2_dir_name)
        assert _created_paths(r2_diff) == r2_new_creates, (
            f"r2 diff created-set: expected {r2_new_creates}, got {_created_paths(r2_diff)}"
        )
        assert _modified_paths(r2_diff) == r2_actual_modifies, (
            f"r2 diff modified-set: expected {r2_actual_modifies}, got {_modified_paths(r2_diff)}"
        )
        assert _deleted_paths(r2_diff) == r2_deletes, (
            f"r2 diff deleted-set: expected {r2_deletes}, got {_deleted_paths(r2_diff)}"
        )

        # 5. c3 diff — no phantom re-creation across r2 boundary
        c3_diff = _read_diff(output_dir, c3_name)
        assert _created_paths(c3_diff) == c3_creates, (
            f"c3 diff created-set: expected {c3_creates}, got {_created_paths(c3_diff)}"
        )
        assert _modified_paths(c3_diff) == c3_actual_modifies, (
            f"c3 diff modified-set: expected {c3_actual_modifies}, got {_modified_paths(c3_diff)}"
        )
        phantom_r2 = _created_paths(c3_diff) & set(tree_after_r2)
        assert phantom_r2 == set(), (
            f"Phantom re-creation across r2→c3 boundary: {phantom_r2}"
        )

        # 6. Snapshot accumulation (§H): c3's snapshot contains all surviving files
        c3_snap = _snapshot_dir(output_dir, c3_name)
        final_tree = dict(tree_after_r2)
        final_tree.update(c3_plan["creates"])
        for fname, new_content in c3_plan["modifies"].items():
            final_tree[fname] = new_content

        for fname in tree_after_r2:
            assert (c3_snap / fname).exists(), (
                f"Snapshot accumulation violated: '{fname}' was in tree after r2 "
                f"but is missing from c3's snapshot"
            )
        all_deletes = (r1_deletes | r2_deletes) - c3_creates
        for fname in all_deletes:
            if fname not in final_tree:
                assert not (c3_snap / fname).exists(), (
                    f"Deleted file '{fname}' still appears in c3's snapshot"
                )

        cleanup_as_root(output_dir)


# ---------------------------------------------------------------------------
# §H content equality: identical overwrite → no diff entry
# ---------------------------------------------------------------------------


@pytest.mark.hypothesis
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not _docker_available(), reason="Docker not available")
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(
    fname=st.sampled_from(_FILENAME_POOL),
    content=_text_content,
    sentinel_content=_text_content,
)
def test_identical_overwrite_no_diff_entry(
    fname: str, content: str, sentinel_content: str
) -> None:
    """§H: overwriting a file with identical bytes produces no diff entry.

    c1 writes fname with `content`. r1 writes the same fname with the same
    `content`. r1's diff must NOT list fname as modified or created.

    Positive control: r1 also creates a fresh sentinel file with *different*
    content.  That sentinel MUST appear as 'created' in r1's diff — confirming
    the diff machinery actually ran and didn't silently skip everything.
    """
    # Pick two distinct files from the pool: fname (for the identical overwrite)
    # and sentinel (to verify the diff machinery is running at all).
    sentinel_file = next(f for f in _FILENAME_POOL if f != fname)
    safe_content = content.replace("'", "'\\''")
    safe_sentinel = sentinel_content.replace("'", "'\\''")
    # The script overwrites fname identically AND creates a new sentinel file.
    script_body = (
        f"printf '%s' '{safe_content}' > \"$1/{fname}\"\n"
        f"printf '%s' '{safe_sentinel}' > \"$1/{sentinel_file}\""
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        problem_dir = tmp / "problem"
        problem_dir.mkdir()
        problem = _make_n_chkpt_problem(problem_dir, 2)

        script_str = _write_script(tmp / "noop.sh", script_body)
        spec = ScriptRefactorSpec(command=script_str)

        output_dir = tmp / "output"
        output_dir.mkdir()
        docker_env = _load_docker_env()
        run_spec = _make_run_spec(problem, docker_env)
        # c1 only writes fname; sentinel_file does not exist before r1 runs.
        agent = DummyAgent([{fname: content}, {}])

        try:
            runner.run_agent(
                run_spec=run_spec,
                agent=agent,
                output_path=output_dir,
                progress_queue=queue.Queue(),
                refactor_spec=spec,
            )
        except Exception:
            cleanup_as_root(output_dir)
            raise

        names = _checkpoint_names_list(problem)
        r1_dir_name = f"{names[0]}{REFACTOR_SUFFIX}"
        r1_diff = _read_diff(output_dir, r1_dir_name)

        # Negative assertion: identical overwrite must not appear in the diff.
        assert fname not in _modified_paths(r1_diff), (
            f"'{fname}' should NOT be modified — content unchanged. "
            f"Modified: {_modified_paths(r1_diff)}"
        )
        assert fname not in _created_paths(r1_diff), (
            f"'{fname}' should NOT be created — it already existed. "
            f"Created: {_created_paths(r1_diff)}"
        )

        # Positive control: the sentinel file is genuinely new and MUST appear
        # as 'created'. This rules out a silent-skip false pass where nothing
        # shows up in the diff because the refactor step never ran.
        assert sentinel_file in _created_paths(r1_diff), (
            f"Positive control failed: sentinel '{sentinel_file}' should be 'created' "
            f"in r1's diff but was not. Created: {_created_paths(r1_diff)}. "
            f"This may mean the refactor step was silently skipped."
        )

        cleanup_as_root(output_dir)


# ---------------------------------------------------------------------------
# §B scheduling — pure property over generated name lists
# ---------------------------------------------------------------------------


@pytest.mark.hypothesis
@settings(max_examples=50)
@given(
    names=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L",)),
            min_size=1,
            max_size=8,
        ),
        min_size=1,
        max_size=6,
    ).filter(lambda ns: len(ns) == len(set(ns))),
    with_spec=st.booleans(),
)
def test_scheduling_exactly_non_last_names(
    names: list[str], with_spec: bool
) -> None:
    """§B: with a spec, exactly the non-last names schedule a refactor; without, none do."""
    spec = ScriptRefactorSpec(command="echo") if with_spec else None
    results = {name: refactor_runs_after(name, names, spec) for name in names}

    if not with_spec:
        assert all(not v for v in results.values()), (
            f"Without a spec, all names should return False; got {results}"
        )
    else:
        for name in names[:-1]:
            assert results[name] is True, (
                f"Non-last name {name!r} should return True with a spec"
            )
        assert results[names[-1]] is False, (
            f"Last name {names[-1]!r} should return False even with a spec"
        )


# ---------------------------------------------------------------------------
# §C identity — pure properties
# ---------------------------------------------------------------------------


@pytest.mark.hypothesis
@settings(max_examples=50)
@given(
    cmd=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=1,
        max_size=60,
    )
)
def test_identity_is_16_hex_chars(cmd: str) -> None:
    """§C: identity is always 16 lowercase hex chars regardless of input."""
    h = compute_refactor_identity(ScriptRefactorSpec(command=cmd))
    assert len(h) == 16
    assert h == h.lower()
    assert all(c in "0123456789abcdef" for c in h)


@pytest.mark.hypothesis
@settings(max_examples=30)
@given(
    cmd_a=st.text(min_size=1, max_size=60),
    cmd_b=st.text(min_size=1, max_size=60),
)
def test_identity_sensitivity(cmd_a: str, cmd_b: str) -> None:
    """§C sensitivity: different commands produce different hashes (w.h.p.)."""
    assume(cmd_a != cmd_b)
    h_a = compute_refactor_identity(ScriptRefactorSpec(command=cmd_a))
    h_b = compute_refactor_identity(ScriptRefactorSpec(command=cmd_b))
    assert h_a != h_b


@pytest.mark.hypothesis
@settings(max_examples=30)
@given(
    cmd=st.text(min_size=1, max_size=60),
    timeout=st.integers(min_value=1, max_value=3600),
    env_vars=st.lists(st.text(min_size=1, max_size=20), max_size=5),
)
def test_identity_excluded_fields_do_not_affect_hash(
    cmd: str, timeout: int, env_vars: list[str]
) -> None:
    """§C: timeout and env_passthrough are excluded from identity (§C table, script row)."""
    base = ScriptRefactorSpec(command=cmd)
    h = compute_refactor_identity(base)
    assert (
        compute_refactor_identity(
            ScriptRefactorSpec(command=cmd, timeout=timeout)
        )
        == h
    )
    assert (
        compute_refactor_identity(
            ScriptRefactorSpec(command=cmd, env_passthrough=env_vars)
        )
        == h
    )


# ---------------------------------------------------------------------------
# §I.2 consistency / invalidation — pure property over fabricated layouts
# ---------------------------------------------------------------------------

_chkpt_name_list = st.lists(
    st.sampled_from([f"c{i}" for i in range(1, 8)]),
    min_size=2,
    max_size=6,
    unique=True,
)

_stale_hash = st.text(alphabet="0123456789abcdef", min_size=16, max_size=16)


def _completed_statuses(names: list[str]) -> list[CheckpointStatus]:
    return [CheckpointStatus(name=n, is_valid=True) for n in names]


def _write_identity(
    output_path: Path, checkpoint_name: str, hash_val: str
) -> None:
    refactor_dir = output_path / f"{checkpoint_name}{REFACTOR_SUFFIX}"
    refactor_dir.mkdir(parents=True, exist_ok=True)
    (refactor_dir / REFACTOR_IDENTITY_FILENAME).write_text(
        json.dumps({"identity_hash": hash_val})
    )


@pytest.mark.hypothesis
@settings(max_examples=50)
@given(
    all_names=_chkpt_name_list,
    current_command=st.text(min_size=1, max_size=40),
    old_hash=_stale_hash,
    data=st.data(),
)
def test_invalidation_first_inconsistent_completed_successor(
    all_names: list[str],
    current_command: str,
    old_hash: str,
    data: st.DataObject,
) -> None:
    """§I.2: first inconsistent c_i with a completed c_{i+1} → REFACTOR_CHANGED on c_{i+1}.

    CI6: the triggering c_i stays completed.
    Everything after c_{i+1} gets DEPENDS_ON_INVALID.

    The stale-index is drawn randomly in [0, len-2] so that the "find the *first*
    inconsistent" scan is exercised at positions other than 0.  All checkpoints
    strictly before the trigger get a *matching* identity file so the scan has to
    skip over them before finding the trigger.
    """
    from hypothesis import note

    spec = ScriptRefactorSpec(command=current_command)
    real_hash = compute_refactor_identity(spec)
    assume(old_hash != real_hash)

    # Draw the trigger index: any non-last position.
    stale_idx = data.draw(st.integers(0, len(all_names) - 2), label="stale_idx")
    triggering = all_names[stale_idx]
    first_successor = all_names[stale_idx + 1]

    note(
        f"all_names={all_names}, stale_idx={stale_idx}, "
        f"triggering={triggering!r}, first_successor={first_successor!r}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Checkpoints before the trigger: write a *matching* identity (no inconsistency).
        for name in all_names[:stale_idx]:
            _write_identity(tmp, name, real_hash)
        # The trigger itself: write a *stale* identity.
        _write_identity(tmp, triggering, old_hash)
        # Checkpoints after the trigger: no identity file needed (they're invalidated).

        completed = list(all_names)
        statuses = _completed_statuses(completed)

        new_completed, new_statuses = _apply_refactor_consistency(
            tmp, completed, statuses, all_names, spec
        )

        status_map = {s.name: s for s in new_statuses}

        # CI6: triggering checkpoint stays
        assert triggering in new_completed
        assert status_map[triggering].is_valid

        # All checkpoints before the trigger are unaffected
        for name in all_names[:stale_idx]:
            assert name in new_completed, (
                f"{name!r} before trigger should stay completed"
            )
            assert status_map[name].is_valid, (
                f"{name!r} before trigger should stay valid"
            )

        # first_successor gets REFACTOR_CHANGED
        assert first_successor not in new_completed
        assert (
            status_map[first_successor].reason
            == InvalidationReason.REFACTOR_CHANGED
        )

        # everything after first_successor gets DEPENDS_ON_INVALID
        for name in all_names[stale_idx + 2 :]:
            assert name not in new_completed
            assert (
                status_map[name].reason == InvalidationReason.DEPENDS_ON_INVALID
            )


@pytest.mark.hypothesis
@settings(max_examples=50)
@given(
    all_names=_chkpt_name_list,
)
def test_invalidation_no_spec_no_artifacts_clean(
    all_names: list[str],
) -> None:
    """§I.2 trivial case: no spec + no identity artifacts → zero invalidation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        completed = list(all_names)
        statuses = _completed_statuses(completed)

        new_completed, new_statuses = _apply_refactor_consistency(
            tmp, completed, statuses, all_names, None
        )

        assert new_completed == completed
        assert all(s.is_valid for s in new_statuses)


@pytest.mark.hypothesis
@settings(max_examples=50)
@given(
    all_names=_chkpt_name_list,
    current_command=st.text(min_size=1, max_size=40),
)
def test_invalidation_matching_hash_clean(
    all_names: list[str],
    current_command: str,
) -> None:
    """§I.2: current spec + matching identity hash on every non-last checkpoint → no invalidation."""
    spec = ScriptRefactorSpec(command=current_command)
    real_hash = compute_refactor_identity(spec)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for name in all_names[:-1]:
            _write_identity(tmp, name, real_hash)

        completed = list(all_names)
        statuses = _completed_statuses(completed)

        new_completed, new_statuses = _apply_refactor_consistency(
            tmp, completed, statuses, all_names, spec
        )

        assert new_completed == completed
        assert all(s.is_valid for s in new_statuses)


@pytest.mark.hypothesis
@settings(max_examples=40)
@given(
    all_names=st.lists(
        st.sampled_from([f"c{i}" for i in range(1, 8)]),
        min_size=2,
        max_size=6,
        unique=True,
    ),
    old_hash=_stale_hash,
    current_command=st.text(min_size=1, max_size=40),
)
def test_CI7_last_checkpoint_inconsistency_no_invalidation(
    all_names: list[str],
    old_hash: str,
    current_command: str,
) -> None:
    """§I.2 CI7: inconsistency at the last checkpoint has no successor → no cascade.

    The last name can never schedule a refactor (would_refactor=False), but we
    write an identity file there (did_refactor=True) to manufacture an
    inconsistency. Since it has no successor, nothing is invalidated.
    """
    spec = ScriptRefactorSpec(command=current_command)
    real_hash = compute_refactor_identity(spec)
    assume(old_hash != real_hash)

    last_name = all_names[-1]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Identity file on the last checkpoint — would_refactor=False, did_refactor=True
        _write_identity(tmp, last_name, old_hash)

        # Only last checkpoint completed; earlier ones are not
        completed = [last_name]
        statuses = _completed_statuses(completed)

        new_completed, _ = _apply_refactor_consistency(
            tmp, completed, statuses, all_names, spec
        )

        # CI7: no cascade
        assert new_completed == completed


@pytest.mark.hypothesis
@settings(max_examples=40)
@given(
    all_names=_chkpt_name_list,
    old_hash=_stale_hash,
    current_command=st.text(min_size=1, max_size=40),
)
def test_CI8_successor_not_completed_no_cascade(
    all_names: list[str],
    old_hash: str,
    current_command: str,
) -> None:
    """§I.2 CI8: inconsistent c_i with an uncompleted c_{i+1} → no cascade."""
    spec = ScriptRefactorSpec(command=current_command)
    real_hash = compute_refactor_identity(spec)
    assume(old_hash != real_hash)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Only the first checkpoint is completed; its successor is NOT.
        _write_identity(tmp, all_names[0], old_hash)

        completed = [all_names[0]]
        statuses = _completed_statuses(completed)

        new_completed, new_statuses = _apply_refactor_consistency(
            tmp, completed, statuses, all_names, spec
        )

        assert new_completed == completed
        assert all(s.is_valid for s in new_statuses)


@pytest.mark.hypothesis
@settings(max_examples=40)
@given(
    all_names=_chkpt_name_list,
    old_hash=_stale_hash,
    current_command=st.text(min_size=1, max_size=40),
)
def test_completed_checkpoints_never_contains_invalid(
    all_names: list[str],
    old_hash: str,
    current_command: str,
) -> None:
    """§I.3 RI: new_completed never contains a name marked invalid in new_statuses."""
    spec = ScriptRefactorSpec(command=current_command)
    real_hash = compute_refactor_identity(spec)
    assume(old_hash != real_hash)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        _write_identity(tmp, all_names[0], old_hash)

        completed = list(all_names)
        statuses = _completed_statuses(completed)

        new_completed, new_statuses = _apply_refactor_consistency(
            tmp, completed, statuses, all_names, spec
        )

        status_map = {s.name: s for s in new_statuses}

        # Invariant: no invalid name appears in new_completed
        for name in new_completed:
            assert status_map[name].is_valid, (
                f"completed list contains {name!r} but it is marked invalid"
            )

        # Invariant: every invalid name is absent from new_completed
        for status in new_statuses:
            if not status.is_valid:
                assert status.name not in new_completed, (
                    f"Invalid checkpoint {status.name!r} appears in completed list"
                )
