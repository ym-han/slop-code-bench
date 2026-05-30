"""Tests for the interleaved refactoring step machinery.

Coverage targets (from spec_v2.md):

Pure / unit (no session):
  - Scheduling: R1–R4
  - Identity: I1–I3
  - Executor selection: M1
  - Snapshot resolution: RS1, RS2
  - Consistency / invalidation: CI-INCONSISTENT (cases 1–3), CI4 cascade,
    CI6 off-by-one, CI7 last-boundary, CI8 successor-not-completed, CI9 reason wiring

E2E with DummyAgent + ScriptRefactorSpec:
  - D1/D2/D3 (headline): refactor snapshot + diff exist; refactor-created file
    does NOT reappear as "created" in the next checkpoint's diff
  - C1: errored checkpoint ⇒ no refactor directory after it
  - R2 / spec §3: no refactor directory after the final checkpoint
  - D4: no spec ⇒ no refactor directories
  - C6 + E1 + S3: nonzero exit ⇒ run still completes; no snapshot/diff, but
    identity file is present (C4)
  - S2: env whitelist — passed-through var present; non-passed-through absent
  - Resume CI4: change command ⇒ REFACTOR_CHANGED on successor, DEPENDS_ON_INVALID on rest
  - Resume RS1: restore uses refactored snapshot when it exists
"""

from __future__ import annotations

import json
import os
import queue
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

from slop_code.agent_runner import runner
from slop_code.agent_runner.agent import Agent
from slop_code.agent_runner.agent import AgentConfigBase
from slop_code.agent_runner.models import AgentCostLimits
from slop_code.agent_runner.models import AgentRunSpec
from slop_code.agent_runner.models import UsageTracker
from slop_code.agent_runner.refactor import REFACTOR_IDENTITY_FILENAME
from slop_code.agent_runner.refactor import REFACTOR_SUFFIX
from slop_code.agent_runner.refactor import AgentRefactorSpec
from slop_code.agent_runner.refactor import ScriptRefactorSpec
from slop_code.agent_runner.refactor import compute_refactor_identity
from slop_code.agent_runner.refactor import make_executor
from slop_code.agent_runner.refactor import refactor_runs_after
from slop_code.agent_runner.resume import CheckpointStatus
from slop_code.agent_runner.resume import InvalidationReason
from slop_code.agent_runner.resume import _apply_refactor_consistency
from slop_code.agent_runner.resume import _resolve_last_snapshot_dir
from slop_code.agent_runner.resume import detect_resume_point
from slop_code.common import DIFF_FILENAME
from slop_code.common import SNAPSHOT_DIR_NAME
from slop_code.evaluation import PassPolicy
from slop_code.evaluation import ProblemConfig
from slop_code.evaluation.config import CheckpointConfig
from slop_code.execution import LocalEnvironmentSpec
from slop_code.execution.models import CommandConfig
from slop_code.execution.models import EnvironmentConfig
from slop_code.execution.models import SetupConfig
from slop_code.execution.session import Session

# ---------------------------------------------------------------------------
# Helpers shared with runner_e2e_test.py (kept local to avoid coupling)
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

    def setup(self, session: Session) -> None:
        self.working_dir = session.working_dir

    def run(self, task: str) -> None:
        pass  # DummyAgent uses run_checkpoint API

    def run_checkpoint(self, task: str):  # type: ignore[override]
        assert self.working_dir is not None
        chkpt = self._chkpt_num
        if self._error_on_checkpoint == chkpt:
            # simulate an error – write nothing; the runner catches and
            # records had_error=True from the exception it raises
            self._chkpt_num += 1
            from datetime import datetime

            from slop_code.agent_runner.agent import CheckpointInferenceResult

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
        from datetime import datetime

        from slop_code.agent_runner.agent import CheckpointInferenceResult

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
    def _from_config(
        cls,
        config: AgentConfigBase,
        problem_name: str,
        verbose: bool,
        image: str | None,
    ) -> Agent:
        raise NotImplementedError


def _make_run_spec(
    problem: ProblemConfig, local_env: LocalEnvironmentSpec
) -> AgentRunSpec:
    return AgentRunSpec(
        seed=0,
        template="{{task}}",
        problem=problem,
        environment=local_env,
        image="test-image",
        pass_policy=PassPolicy.ANY,
        skip_evaluation=True,
        verbose=False,
    )


@pytest.fixture()
def resources_path() -> Path:
    return Path(__file__).parent / "resources"


@pytest.fixture()
def problem(resources_path: Path) -> ProblemConfig:
    return ProblemConfig.from_yaml(
        resources_path / "inventory_cli_debug_problem"
    )


@pytest.fixture()
def local_env() -> LocalEnvironmentSpec:
    return LocalEnvironmentSpec(
        name="local",
        type="local",
        setup=SetupConfig(eval_commands=[]),
        environment=EnvironmentConfig(include_os_env=True),
        commands=CommandConfig(command="uv run", entry_file="{entry_file}.py"),
    )


@pytest.fixture()
def run_spec(
    problem: ProblemConfig, local_env: LocalEnvironmentSpec
) -> AgentRunSpec:
    return _make_run_spec(problem, local_env)


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out


def _make_script(tmp_path: Path, body: str) -> str:
    """Write a shell script and make it executable; return its path as a string."""
    script = tmp_path / "refactor.sh"
    script.write_text("#!/bin/sh\n" + textwrap.dedent(body))
    script.chmod(
        script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    return str(script)


def _make_named_script(path: Path, body: str) -> str:
    """Write a named shell script to a specific path and make it executable."""
    path.write_text("#!/bin/sh\n" + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def _checkpoint_names(problem: ProblemConfig) -> list[str]:
    return list(problem.checkpoints.keys())


def _created_paths(diff_json: dict) -> set[str]:
    """Return the basenames of files marked 'created' in a serialized SnapshotDiff.

    A SnapshotDiff serializes to ``{"file_diffs": {<path>: {"change_type": ...}}}``;
    there is no top-level ``added_files`` key.  We key on basename because the diff
    path is relative to the workspace root.
    """
    file_diffs: dict[str, dict] = diff_json.get("file_diffs", {})
    return {
        Path(path).name
        for path, fd in file_diffs.items()
        if fd.get("change_type") == "created"
    }


def _modified_paths(diff_json: dict) -> set[str]:
    """Return the basenames of files marked 'modified' in a serialized SnapshotDiff."""
    file_diffs: dict[str, dict] = diff_json.get("file_diffs", {})
    return {
        Path(path).name
        for path, fd in file_diffs.items()
        if fd.get("change_type") == "modified"
    }


# ===========================================================================
# §3 Scheduling — pure unit tests (R1–R4)
# ===========================================================================


class TestScheduling:
    def test_R1_no_spec_always_false(self) -> None:
        """R1: no spec ⇒ refactor_runs_after is False for every name."""
        names = ["c1", "c2", "c3"]
        for name in names:
            assert refactor_runs_after(name, names, None) is False

    def test_R2_last_checkpoint_always_false(self) -> None:
        """R2: with a spec, the last checkpoint never triggers a refactor."""
        spec = ScriptRefactorSpec(command="echo")
        names = ["c1", "c2", "c3"]
        assert refactor_runs_after("c3", names, spec) is False

    def test_R3_non_last_with_spec_true(self) -> None:
        """R3: with a spec, every non-last checkpoint triggers a refactor."""
        spec = ScriptRefactorSpec(command="echo")
        names = ["c1", "c2", "c3"]
        assert refactor_runs_after("c1", names, spec) is True
        assert refactor_runs_after("c2", names, spec) is True

    def test_R4_purity_single_element(self) -> None:
        """R4 corner: single-checkpoint list has no non-last element."""
        spec = ScriptRefactorSpec(command="echo")
        assert refactor_runs_after("only", ["only"], spec) is False

    def test_R4_empty_names_raises_indexerror(self) -> None:
        """Degenerate boundary (A1): an empty names list has no last element, so the
        ``all_names[-1]`` lookup raises IndexError.  This characterizes (does not
        endorse) the current behavior — see ambiguity A1 in the spec.
        """
        spec = ScriptRefactorSpec(command="echo")
        with pytest.raises(IndexError):
            refactor_runs_after("anything", [], spec)


# ===========================================================================
# §6 Identity — pure unit tests (I1–I3)
# ===========================================================================


class TestIdentity:
    def _script(
        self, command: str = "echo hi", **kwargs: object
    ) -> ScriptRefactorSpec:
        return ScriptRefactorSpec(command=command, **kwargs)  # type: ignore[arg-type]

    def test_I1_script_same_command_same_hash(self) -> None:
        """I1: equal commands ⇒ equal hash."""
        assert compute_refactor_identity(
            self._script("foo")
        ) == compute_refactor_identity(self._script("foo"))

    def test_I1_script_different_command_different_hash(self) -> None:
        """I1: changing command changes identity."""
        assert compute_refactor_identity(
            self._script("foo")
        ) != compute_refactor_identity(self._script("bar"))

    def test_I1_script_timeout_and_env_do_not_affect_identity(self) -> None:
        """I1: timeout and env_passthrough are excluded from identity."""
        base = self._script("cmd")
        modified_timeout = ScriptRefactorSpec(command="cmd", timeout=9999)
        modified_env = ScriptRefactorSpec(
            command="cmd", env_passthrough=["FOO"]
        )
        assert compute_refactor_identity(base) == compute_refactor_identity(
            modified_timeout
        )
        assert compute_refactor_identity(base) == compute_refactor_identity(
            modified_env
        )

    def test_I3_form_16_hex_chars(self) -> None:
        """I3: identity is 16 lowercase hex chars."""
        h = compute_refactor_identity(self._script("anything"))
        assert len(h) == 16
        assert h == h.lower()
        assert all(c in "0123456789abcdef" for c in h)

    def test_I3_stable_across_calls(self) -> None:
        """I3: pure function — same spec always yields the same hash."""
        spec = self._script("stable_cmd")
        results = {compute_refactor_identity(spec) for _ in range(5)}
        assert len(results) == 1

    # --- I2: agent identity (config class + model name + prompt) -----------

    def _agent(
        self,
        *,
        config_cls: type = object,
        model_name: str = "m1",
        prompt: str = "refactor please",
        image: str = "img",
        credential: object | None = None,
    ) -> AgentRefactorSpec:
        return AgentRefactorSpec(
            agent_config=config_cls(),  # type: ignore[call-arg]
            model_def=type("M", (), {"name": model_name})(),
            credential=credential if credential is not None else object(),
            prompt=prompt,
            image=image,
        )

    def test_I2_agent_same_fields_same_hash(self) -> None:
        """I2: equal (config class, model name, prompt) ⇒ equal hash."""
        assert compute_refactor_identity(
            self._agent()
        ) == compute_refactor_identity(self._agent())

    def test_I2_agent_model_and_prompt_change_identity(self) -> None:
        """I2: changing model name or prompt changes identity."""
        base = self._agent()
        assert compute_refactor_identity(base) != compute_refactor_identity(
            self._agent(model_name="m2")
        )
        assert compute_refactor_identity(base) != compute_refactor_identity(
            self._agent(prompt="something else")
        )

    def test_I2_agent_image_and_credential_do_not_affect_identity(self) -> None:
        """I2: image and credential are excluded from identity."""
        base = self._agent()
        assert compute_refactor_identity(base) == compute_refactor_identity(
            self._agent(image="other-image")
        )
        assert compute_refactor_identity(base) == compute_refactor_identity(
            self._agent(credential=object())
        )

    def test_script_and_agent_identities_disjoint(self) -> None:
        """A script and an agent spec never collide (kind prefix differs)."""
        assert compute_refactor_identity(
            self._script("x")
        ) != compute_refactor_identity(self._agent())


# ===========================================================================
# §4.4 Executor selection — M1
# ===========================================================================


class TestExecutorSelection:
    def test_M1_script_spec_produces_script_executor(self) -> None:
        from slop_code.agent_runner.refactor import ScriptRefactorExecutor

        spec = ScriptRefactorSpec(command="echo")
        assert isinstance(
            make_executor(spec, "some_problem"), ScriptRefactorExecutor
        )

    def test_M1_agent_spec_produces_agent_executor(self) -> None:
        """M1: an agent spec selects the agent executor (problem name threaded in)."""
        from slop_code.agent_runner.refactor import AgentRefactorExecutor

        # AgentRefactorSpec is a plain dataclass with no validation, so stub objects
        # suffice to exercise the type-dispatch branch without agent infrastructure.
        spec = AgentRefactorSpec(
            agent_config=object(),  # type: ignore[arg-type]
            model_def=type("M", (), {"name": "m"})(),  # type: ignore[arg-type]
            credential=object(),  # type: ignore[arg-type]
            prompt="do it",
            image="img",
        )
        executor = make_executor(spec, "some_problem")
        assert isinstance(executor, AgentRefactorExecutor)
        assert executor._problem_name == "some_problem"


# ===========================================================================
# Resume §7.1 — snapshot resolution (RS1, RS2)
# ===========================================================================


class TestSnapshotResolution:
    def test_RS2_empty_completed_returns_none(self, tmp_path: Path) -> None:
        """RS2: no completed checkpoints ⇒ None."""
        assert _resolve_last_snapshot_dir(tmp_path, []) is None

    def test_RS1_prefers_refactor_snapshot(self, tmp_path: Path) -> None:
        """RS1: when refactor snapshot exists it is preferred over feature snapshot."""
        (tmp_path / "checkpoint_1" / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        refactor_snap = tmp_path / "checkpoint_1__refactor" / SNAPSHOT_DIR_NAME
        refactor_snap.mkdir(parents=True)
        result = _resolve_last_snapshot_dir(tmp_path, ["checkpoint_1"])
        assert result == refactor_snap

    def test_RS1_falls_back_to_feature_snapshot_when_no_refactor(
        self, tmp_path: Path
    ) -> None:
        """RS1 fallback: no refactor snapshot ⇒ feature snapshot returned."""
        snap = tmp_path / "checkpoint_1" / SNAPSHOT_DIR_NAME
        snap.mkdir(parents=True)
        result = _resolve_last_snapshot_dir(tmp_path, ["checkpoint_1"])
        assert result == snap

    def test_RS1_uses_last_completed(self, tmp_path: Path) -> None:
        """RS1: picks snapshot from the last completed checkpoint, not an earlier one."""
        for i in (1, 2):
            (tmp_path / f"checkpoint_{i}" / SNAPSHOT_DIR_NAME).mkdir(
                parents=True
            )
        # Refactor only after checkpoint_1
        refactor_snap_1 = (
            tmp_path / "checkpoint_1__refactor" / SNAPSHOT_DIR_NAME
        )
        refactor_snap_1.mkdir(parents=True)
        # Last completed is checkpoint_2, which has no refactor snapshot
        result = _resolve_last_snapshot_dir(
            tmp_path, ["checkpoint_1", "checkpoint_2"]
        )
        assert result == tmp_path / "checkpoint_2" / SNAPSHOT_DIR_NAME


# ===========================================================================
# Resume §7.2 — consistency / invalidation (unit, fabricated dirs)
# ===========================================================================


def _write_identity(
    output_path: Path, checkpoint_name: str, hash_val: str
) -> None:
    refactor_dir = output_path / f"{checkpoint_name}{REFACTOR_SUFFIX}"
    refactor_dir.mkdir(parents=True, exist_ok=True)
    (refactor_dir / REFACTOR_IDENTITY_FILENAME).write_text(
        json.dumps({"identity_hash": hash_val})
    )


def _completed_statuses(names: list[str]) -> list[CheckpointStatus]:
    return [CheckpointStatus(name=n, is_valid=True) for n in names]


class TestRefactorConsistency:
    """Unit tests for _apply_refactor_consistency — no session needed."""

    def test_no_spec_no_artifacts_clean(self, tmp_path: Path) -> None:
        """No spec + no identity files ⇒ no invalidation."""
        completed = ["c1", "c2"]
        statuses = _completed_statuses(completed)
        new_completed, new_statuses = _apply_refactor_consistency(
            tmp_path, completed, statuses, ["c1", "c2", "c3"], None
        )
        assert new_completed == completed
        assert all(s.is_valid for s in new_statuses)

    def test_CI_inconsistent_case1_added_refactor(self, tmp_path: Path) -> None:
        """CI-INCONSISTENT case 1a: refactor added (would_refactor=True, did_refactor=False)."""
        # No identity file exists for c1, but we now have a spec
        spec = ScriptRefactorSpec(command="something_new")
        completed = ["c1", "c2"]
        statuses = _completed_statuses(completed)
        new_completed, new_statuses = _apply_refactor_consistency(
            tmp_path, completed, statuses, ["c1", "c2", "c3"], spec
        )
        # c2 is the successor of c1 where the refactor was inconsistent
        assert "c2" not in new_completed
        assert new_completed == ["c1"]

    def test_CI_inconsistent_case1_removed_refactor(
        self, tmp_path: Path
    ) -> None:
        """CI-INCONSISTENT case 1b: refactor removed (would_refactor=False, did_refactor=True)."""
        _write_identity(tmp_path, "c1", "deadbeef12345678")
        completed = ["c1", "c2"]
        statuses = _completed_statuses(completed)
        new_completed, new_statuses = _apply_refactor_consistency(
            tmp_path, completed, statuses, ["c1", "c2", "c3"], None
        )
        assert "c2" not in new_completed

    def test_CI_inconsistent_case2_hash_changed(self, tmp_path: Path) -> None:
        """CI-INCONSISTENT case 2: both have refactor but hash differs."""
        spec = ScriptRefactorSpec(command="new_command")
        _write_identity(
            tmp_path, "c1", "0000000000000000"
        )  # different from real hash
        completed = ["c1", "c2"]
        statuses = _completed_statuses(completed)
        new_completed, new_statuses = _apply_refactor_consistency(
            tmp_path, completed, statuses, ["c1", "c2", "c3"], spec
        )
        assert "c2" not in new_completed

    def test_CI_inconsistent_case3_corrupt_identity_file(
        self, tmp_path: Path
    ) -> None:
        """CI-INCONSISTENT case 3: identity file unreadable/unparseable."""
        spec = ScriptRefactorSpec(command="cmd")
        refactor_dir = tmp_path / "c1__refactor"
        refactor_dir.mkdir(parents=True)
        (refactor_dir / REFACTOR_IDENTITY_FILENAME).write_text(
            "not valid json {{{{"
        )
        completed = ["c1", "c2"]
        statuses = _completed_statuses(completed)
        new_completed, new_statuses = _apply_refactor_consistency(
            tmp_path, completed, statuses, ["c1", "c2", "c3"], spec
        )
        assert "c2" not in new_completed

    def test_CI4_cascade_REFACTOR_CHANGED_then_DEPENDS_ON_INVALID(
        self, tmp_path: Path
    ) -> None:
        """CI4: first stale successor gets REFACTOR_CHANGED; rest get DEPENDS_ON_INVALID."""
        spec = ScriptRefactorSpec(command="new_cmd")
        # c1 ran a refactor with the old command; c2 and c3 completed on that stale baseline
        _write_identity(tmp_path, "c1", "0000000000000000")
        completed = ["c1", "c2", "c3"]
        statuses = _completed_statuses(completed)
        all_names = ["c1", "c2", "c3", "c4"]
        new_completed, new_statuses = _apply_refactor_consistency(
            tmp_path, completed, statuses, all_names, spec
        )
        assert new_completed == ["c1"]
        status_map = {s.name: s for s in new_statuses}
        assert status_map["c2"].reason == InvalidationReason.REFACTOR_CHANGED
        assert status_map["c3"].reason == InvalidationReason.DEPENDS_ON_INVALID

    def test_CI6_off_by_one_triggering_checkpoint_stays_completed(
        self, tmp_path: Path
    ) -> None:
        """CI6: the checkpoint *after* which the refactor ran stays completed; only its successor is stale."""
        spec = ScriptRefactorSpec(command="new_cmd")
        _write_identity(tmp_path, "c1", "0000000000000000")
        completed = ["c1", "c2"]
        statuses = _completed_statuses(completed)
        new_completed, _ = _apply_refactor_consistency(
            tmp_path, completed, statuses, ["c1", "c2", "c3"], spec
        )
        assert "c1" in new_completed
        assert "c2" not in new_completed

    def test_CI7_last_checkpoint_inconsistency_no_invalidation(
        self, tmp_path: Path
    ) -> None:
        """CI7: inconsistency at the last checkpoint has no successor to invalidate.

        Setup: ["c1", "c2", "c3"].  c1 and c3 are completed.  c3 is last.
        c3 has a stale identity file (would_refactor("c3", ..., spec) == False but
        did_refactor == True).  c1 is also inconsistent (would_refactor=True, no id
        file), BUT its successor c2 is NOT in completed — so CI8 blocks that cascade.
        The only remaining inconsistency is c3 (last), which has no successor, so CI7
        says nothing is invalidated.
        """
        spec = ScriptRefactorSpec(command="cmd")
        # c3 is last: would_refactor("c3", all_names, spec) == False,
        # yet an identity file sits there (did_refactor=True) — inconsistent.
        all_names = ["c1", "c2", "c3"]
        _write_identity(tmp_path, "c3", "deadbeef12345678")
        # c1 is completed, c2 is NOT completed (only c1 and c3 are completed)
        completed = ["c1", "c3"]
        statuses = _completed_statuses(completed)
        # c1 check: would_refactor("c1", all_names, spec)==True, did_refactor=False → inconsistent.
        # Successor of c1 is c2, which is NOT in completed → CI8 blocks cascade.
        # c3 check: would_refactor("c3", all_names, spec)==False, did_refactor=True → inconsistent.
        # idx+1 = 3 >= len(all_names) = 3 → no successor → CI7: nothing invalidated.
        new_completed, _ = _apply_refactor_consistency(
            tmp_path, completed, statuses, all_names, spec
        )
        assert new_completed == completed  # nothing invalidated

    def test_CI8_successor_not_completed_no_invalidation(
        self, tmp_path: Path
    ) -> None:
        """CI8: inconsistency whose successor is not completed ⇒ no cascade."""
        spec = ScriptRefactorSpec(command="new_cmd")
        _write_identity(tmp_path, "c1", "0000000000000000")
        # c2 is NOT in completed
        completed = ["c1"]
        statuses = _completed_statuses(completed)
        new_completed, _ = _apply_refactor_consistency(
            tmp_path, completed, statuses, ["c1", "c2", "c3"], spec
        )
        assert new_completed == ["c1"]  # nothing invalidated

    def test_CI9_reason_enum_string(self) -> None:
        """CI9: REFACTOR_CHANGED has enum string 'refactor_changed'."""
        assert InvalidationReason.REFACTOR_CHANGED.value == "refactor_changed"

    def test_CI9_reason_renders_in_summary(self) -> None:
        """CI9: reason renders as 'refactor spec changed' in format_resume_summary."""
        from slop_code.agent_runner.resume import ResumeInfo
        from slop_code.agent_runner.resume import format_resume_summary

        info = ResumeInfo(
            resume_from_checkpoint="c2",
            completed_checkpoints=["c1"],
            last_snapshot_dir=None,
            prior_usage=UsageTracker(),
            checkpoint_statuses=[
                CheckpointStatus(name="c1", is_valid=True),
                CheckpointStatus(
                    name="c2",
                    is_valid=False,
                    reason=InvalidationReason.REFACTOR_CHANGED,
                ),
            ],
            invalidated_checkpoints=["c2"],
        )
        summary = format_resume_summary(info)
        assert "refactor spec changed" in summary

    def test_RI1_artifact_path_also_checks_consistency(
        self, tmp_path: Path
    ) -> None:
        """RI1: consistency check runs in the artifact-fallback path too.

        We fabricate a directory layout that the artifact-fallback will accept
        as 'completed' and verify that an inconsistency in the identity file
        is caught, matching what the run_info.yaml path would do.
        """
        from slop_code.common import INFERENCE_RESULT_FILENAME
        from slop_code.common import SNAPSHOT_DIR_NAME

        spec = ScriptRefactorSpec(command="new_cmd")
        # Build two completed checkpoints (artifact layout)
        for name in ("c1", "c2"):
            d = tmp_path / name
            (d / SNAPSHOT_DIR_NAME).mkdir(parents=True)
            (d / INFERENCE_RESULT_FILENAME).write_text(
                json.dumps({"had_error": False, "usage": {"cost": 0.0}})
            )
        # c1 refactor ran with old hash
        _write_identity(tmp_path, "c1", "0000000000000000")

        result = detect_resume_point(
            tmp_path,
            ["c1", "c2", "c3"],
            refactor_spec=spec,
        )
        assert result is not None
        # c2 should have been invalidated because the refactor identity changed
        assert "c2" not in result.completed_checkpoints


# ===========================================================================
# E2E tests — DummyAgent + ScriptRefactorSpec (local session)
# ===========================================================================


def _run(
    problem: ProblemConfig,
    run_spec: AgentRunSpec,
    output_dir: Path,
    agent: DummyAgent,
    refactor_spec: ScriptRefactorSpec | None = None,
) -> dict:
    """Thin wrapper around runner.run_agent for E2E tests."""
    return runner.run_agent(
        run_spec=run_spec,
        agent=agent,
        output_path=output_dir,
        progress_queue=queue.Queue(),
        refactor_spec=refactor_spec,
    )


def _dummy_solutions(problem: ProblemConfig) -> list[dict[str, str]]:
    """Minimal per-checkpoint solutions that write a distinguishable file."""
    entry = problem.entry_file
    return [
        {f"{entry}.py": f"# checkpoint {i}"}
        for i, _ in enumerate(problem.checkpoints)
    ]


class TestE2ERefactor:
    """End-to-end tests: DummyAgent + ScriptRefactorSpec, local session."""

    # --- D4: no spec ⇒ no refactor dirs -----------------------------------

    def test_D4_no_spec_no_refactor_dirs(
        self,
        problem: ProblemConfig,
        run_spec: AgentRunSpec,
        output_dir: Path,
    ) -> None:
        """D4: without a refactor spec, no __refactor directories appear."""
        agent = DummyAgent(_dummy_solutions(problem))
        _run(problem, run_spec, output_dir, agent, refactor_spec=None)

        refactor_dirs = list(output_dir.glob("*__refactor"))
        assert refactor_dirs == []

    # --- R2: no refactor after last checkpoint -----------------------------

    def test_R2_no_refactor_after_last_checkpoint(
        self,
        problem: ProblemConfig,
        run_spec: AgentRunSpec,
        output_dir: Path,
        tmp_path: Path,
    ) -> None:
        """R2: even with a spec, the last checkpoint produces no refactor dir."""
        script = _make_script(tmp_path, "exit 0")
        spec = ScriptRefactorSpec(command=script)
        agent = DummyAgent(_dummy_solutions(problem))
        _run(problem, run_spec, output_dir, agent, refactor_spec=spec)

        names = _checkpoint_names(problem)
        last = names[-1]
        last_refactor = output_dir / f"{last}{REFACTOR_SUFFIX}"
        assert not last_refactor.exists()

    # --- C4 + E1 + S3: script exits nonzero --------------------------------

    def test_C4_identity_written_before_execution(
        self,
        problem: ProblemConfig,
        run_spec: AgentRunSpec,
        output_dir: Path,
        tmp_path: Path,
    ) -> None:
        """C4: identity file written before execution; E1/S3: no snapshot on failure."""
        # Script that always fails
        script = _make_script(tmp_path, "exit 1")
        spec = ScriptRefactorSpec(command=script)
        agent = DummyAgent(_dummy_solutions(problem))
        result = _run(problem, run_spec, output_dir, agent, refactor_spec=spec)

        # Run should still complete (C6)
        assert result["summary"]["state"] == "completed"

        names = _checkpoint_names(problem)
        # Refactor fires after every non-last checkpoint
        for name in names[:-1]:
            refactor_dir = output_dir / f"{name}{REFACTOR_SUFFIX}"
            # C4: identity file must exist (written before executing)
            assert (refactor_dir / REFACTOR_IDENTITY_FILENAME).exists(), (
                f"Missing identity file for {name}"
            )
            # E1 / S3: no snapshot or diff on failure
            assert not (refactor_dir / SNAPSHOT_DIR_NAME).exists(), (
                f"Unexpected snapshot for failed refactor after {name}"
            )
            assert not (refactor_dir / DIFF_FILENAME).exists(), (
                f"Unexpected diff.json for failed refactor after {name}"
            )

    # --- D1/D2/D3: headline diff semantics ---------------------------------

    def test_D1_D2_D3_diff_semantics(
        self,
        problem: ProblemConfig,
        run_spec: AgentRunSpec,
        output_dir: Path,
        tmp_path: Path,
    ) -> None:
        """D1/D2/D3: refactor snapshot+diff exist; refactor-created file NOT re-created in next diff.

        Concretely: the refactor script writes a sentinel file.  The next
        feature checkpoint writes its own file.  We verify that the next
        checkpoint's diff does NOT list the sentinel as 'added' (it was
        already in the refactored baseline).
        """
        names = _checkpoint_names(problem)
        assert len(names) >= 2, "Need at least 2 checkpoints for this test"

        sentinel = "refactor_sentinel.txt"
        # Script writes a sentinel file into the working directory (first arg)
        script = _make_script(
            tmp_path,
            f"""
            working_dir="$1"
            echo "refactored" > "$working_dir/{sentinel}"
            """,
        )
        spec = ScriptRefactorSpec(command=script)
        # Each checkpoint writes its OWN distinct file so we can positively assert
        # that c_2's diff *is* live (it lists c_2's new file as created) — otherwise
        # the D2 negative assertion could pass simply because every diff is empty.
        entry = problem.entry_file
        c2_file = "feature_c2.txt"
        solutions = [
            {f"{entry}.py": "# checkpoint 0"},
            {c2_file: "feature added in checkpoint 2"},
        ]
        agent = DummyAgent(solutions)
        _run(problem, run_spec, output_dir, agent, refactor_spec=spec)

        # D1: refactor snapshot + diff exist, and the refactor's *own* diff records
        # the sentinel as created (relative to the c_1 feature snapshot).  This is
        # the positive half: without it the D2 negative assertion below could pass
        # simply because the refactor never ran.
        first_name = names[0]
        refactor_dir = output_dir / f"{first_name}{REFACTOR_SUFFIX}"
        assert (refactor_dir / SNAPSHOT_DIR_NAME).exists(), (
            "Refactor snapshot missing (D1)"
        )
        refactor_diff_path = refactor_dir / DIFF_FILENAME
        assert refactor_diff_path.exists(), "Refactor diff.json missing (D1)"
        refactor_diff = json.loads(refactor_diff_path.read_text())
        assert sentinel in _created_paths(refactor_diff), (
            f"Sentinel '{sentinel}' should appear as 'created' in the refactor's own diff (D1) — "
            f"created files were {_created_paths(refactor_diff)}"
        )

        # D2 / D3: the sentinel must NOT reappear as 'created' in the second
        # checkpoint's diff, because c_2 was re-baselined onto the refactored tree
        # (which already contains the sentinel).  If it reappears, c_2 was diffed
        # against the pre-refactor snapshot — the exact bug this feature must avoid.
        second_name = names[1]
        second_diff_path = output_dir / second_name / DIFF_FILENAME
        assert second_diff_path.exists(), "Second checkpoint diff.json missing"
        second_diff = json.loads(second_diff_path.read_text())
        created_in_second = _created_paths(second_diff)
        # Positive control: c_2's own new file must show up as created, proving the
        # diff machinery is live (so the negative assertion below has teeth).
        assert c2_file in created_in_second, (
            f"c_2's own new file '{c2_file}' should be 'created' in {second_name}'s diff; "
            f"created files were {created_in_second}"
        )
        assert sentinel not in created_in_second, (
            f"Sentinel '{sentinel}' incorrectly appears as 'created' in {second_name}'s diff "
            f"(created files were {created_in_second}) — diff was not computed against the "
            "refactored baseline (D2 violated)"
        )

    # --- C1: errored checkpoint ⇒ no refactor dir after it ----------------

    def test_C1_errored_checkpoint_suppresses_refactor(
        self,
        problem: ProblemConfig,
        run_spec: AgentRunSpec,
        output_dir: Path,
        tmp_path: Path,
    ) -> None:
        """C1: a checkpoint that errors does not trigger its refactor."""
        script = _make_script(tmp_path, "exit 0")
        spec = ScriptRefactorSpec(command=script)

        # Make the first checkpoint error
        solutions = _dummy_solutions(problem)
        agent = DummyAgent(solutions, error_on_checkpoint=0)
        _run(problem, run_spec, output_dir, agent, refactor_spec=spec)

        names = _checkpoint_names(problem)
        first_refactor = output_dir / f"{names[0]}{REFACTOR_SUFFIX}"
        assert not first_refactor.exists(), (
            "Refactor directory must not exist after an errored checkpoint"
        )

    # --- S2: env whitelist -------------------------------------------------

    def test_S2_env_whitelist(
        self,
        problem: ProblemConfig,
        run_spec: AgentRunSpec,
        output_dir: Path,
        tmp_path: Path,
    ) -> None:
        """S2: non-passed-through host var absent; passed-through var present.

        The script dumps its env to artifacts_dir/env.txt.  We set a sentinel
        host var and verify: with no passthrough it's absent; with passthrough
        it's present.
        """
        from slop_code import common

        sentinel_var = "SLOP_TEST_SENTINEL_XYZ"
        sentinel_val = "test_value_12345"
        os.environ[sentinel_var] = sentinel_val

        def read_child_env(out: Path, name: str) -> dict[str, str]:
            """Parse the child's `env` dump into {var: value}, line by line.

            Line-based parsing avoids the brittleness of substring-matching the raw
            text (e.g. a value that happens to contain the var name).
            """
            env_file = (
                out
                / f"{name}{REFACTOR_SUFFIX}"
                / common.AGENT_DIR_NAME
                / "env.txt"
            )
            assert env_file.exists(), "Script did not write env.txt"
            result: dict[str, str] = {}
            for line in env_file.read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep:
                    result[key] = value
            return result

        try:
            # Script dumps env to the artifacts dir (second arg)
            script = _make_script(
                tmp_path,
                """
                artifacts="$2"
                env > "$artifacts/env.txt"
                """,
            )
            names = _checkpoint_names(problem)

            # First: without passthrough — sentinel absent from the child's env
            spec_no_pass = ScriptRefactorSpec(command=script)
            agent = DummyAgent(_dummy_solutions(problem))
            out1 = tmp_path / "out1"
            out1.mkdir()
            _run(problem, run_spec, out1, agent, refactor_spec=spec_no_pass)

            child_env = read_child_env(out1, names[0])
            assert sentinel_var not in child_env, (
                f"{sentinel_var} should not be in child env when not passed through; "
                f"child saw keys {sorted(child_env)}"
            )

            # Second: with passthrough — sentinel present with its exact value
            spec_with_pass = ScriptRefactorSpec(
                command=script, env_passthrough=[sentinel_var]
            )
            agent2 = DummyAgent(_dummy_solutions(problem))
            out2 = tmp_path / "out2"
            out2.mkdir()
            _run(problem, run_spec, out2, agent2, refactor_spec=spec_with_pass)

            child_env2 = read_child_env(out2, names[0])
            assert child_env2.get(sentinel_var) == sentinel_val, (
                f"{sentinel_var} should be in child env with value {sentinel_val!r} "
                f"when passed through; got {child_env2.get(sentinel_var)!r}"
            )
        finally:
            os.environ.pop(sentinel_var, None)


# ===========================================================================
# Resume E2E — detect_resume_point with refactor-aware invalidation
# ===========================================================================


class TestResumeRefactorAware:
    """Resume detection with fabricated output directories."""

    def _write_checkpoint_artifacts(self, output_path: Path, name: str) -> None:
        """Write minimal artifacts for a 'completed' checkpoint."""
        from slop_code.common import INFERENCE_RESULT_FILENAME
        from slop_code.common import SNAPSHOT_DIR_NAME

        d = output_path / name
        (d / SNAPSHOT_DIR_NAME).mkdir(parents=True)
        (d / INFERENCE_RESULT_FILENAME).write_text(
            json.dumps({"had_error": False, "usage": {"cost": 0.0}})
        )

    def test_resume_no_refactor_change_no_invalidation(
        self, tmp_path: Path
    ) -> None:
        """No spec change ⇒ completed checkpoints remain valid."""
        spec = ScriptRefactorSpec(command="cmd")
        real_hash = compute_refactor_identity(spec)
        self._write_checkpoint_artifacts(tmp_path, "c1")
        self._write_checkpoint_artifacts(tmp_path, "c2")
        _write_identity(tmp_path, "c1", real_hash)

        result = detect_resume_point(
            tmp_path, ["c1", "c2", "c3"], refactor_spec=spec
        )
        assert result is not None
        assert result.completed_checkpoints == ["c1", "c2"]

    def test_resume_CI4_refactor_changed_invalidates_successor(
        self, tmp_path: Path
    ) -> None:
        """CI4: changed command ⇒ REFACTOR_CHANGED on c2, DEPENDS_ON_INVALID on c3."""
        spec = ScriptRefactorSpec(command="new_command")
        self._write_checkpoint_artifacts(tmp_path, "c1")
        self._write_checkpoint_artifacts(tmp_path, "c2")
        self._write_checkpoint_artifacts(tmp_path, "c3")
        # c1 ran with old command
        _write_identity(tmp_path, "c1", "0000000000000000")
        # c2 also has a refactor (with old hash too) — both should be invalidated
        _write_identity(tmp_path, "c2", "0000000000000000")

        result = detect_resume_point(
            tmp_path, ["c1", "c2", "c3", "c4"], refactor_spec=spec
        )
        assert result is not None
        # c1 stays valid; c2 is the first stale successor
        assert result.completed_checkpoints == ["c1"]
        assert result.resume_from_checkpoint == "c2"
        status_map = {s.name: s for s in result.checkpoint_statuses}
        assert status_map["c2"].reason == InvalidationReason.REFACTOR_CHANGED
        assert status_map["c3"].reason == InvalidationReason.DEPENDS_ON_INVALID

    def test_resume_RS1_uses_refactor_snapshot(self, tmp_path: Path) -> None:
        """RS1: when the last completed checkpoint has a refactor snapshot, it is used as restore source.

        Scenario: c1 is completed and has a refactor snapshot; c2 has not started.
        The resume source must be c1's refactor snapshot, not c1's feature snapshot.
        """
        spec = ScriptRefactorSpec(command="cmd")
        real_hash = compute_refactor_identity(spec)

        # c1 is completed (feature artifacts present)
        self._write_checkpoint_artifacts(tmp_path, "c1")
        # c1 also has a successful refactor (refactor snapshot + identity)
        refactor_snap = tmp_path / "c1__refactor" / SNAPSHOT_DIR_NAME
        refactor_snap.mkdir(parents=True)
        _write_identity(tmp_path, "c1", real_hash)

        # c2 is not started at all (no directory)
        result = detect_resume_point(
            tmp_path, ["c1", "c2", "c3"], refactor_spec=spec
        )
        assert result is not None
        assert result.completed_checkpoints == ["c1"]
        assert result.resume_from_checkpoint == "c2"
        # RS1: restore from the refactored snapshot of c1 (not the feature snapshot)
        assert result.last_snapshot_dir == refactor_snap


# ===========================================================================
# Helpers for E2E diff-chain tests
# ===========================================================================


_DOCKER_ENV_PATH = (
    Path(__file__).parent.parent.parent
    / "configs"
    / "environments"
    / "docker-python3.12-uv.yaml"
)


def _docker_available() -> bool:
    """Check whether Docker is reachable (used for skipif markers)."""
    try:
        import docker as docker_sdk

        client = docker_sdk.from_env()
        client.ping()
        client.close()
        return True
    except Exception:
        return False


def cleanup_as_root(path: Path) -> None:
    """Remove a directory that may contain root-owned files created by Docker."""
    if not path.exists():
        return
    subprocess.run(
        [
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


def _load_docker_env():
    """Load the docker-python3.12-uv EnvironmentSpec from YAML."""
    from slop_code.execution.docker_runtime import DockerEnvironmentSpec

    with _DOCKER_ENV_PATH.open() as f:
        config = yaml.safe_load(f)
    return DockerEnvironmentSpec(**config)


def _make_3chkpt_problem(tmp_path: Path) -> ProblemConfig:
    """Synthesize a minimal 3-checkpoint ProblemConfig entirely in-memory.

    Uses spec_override so no spec files are needed on disk.
    """
    return ProblemConfig(
        name="three_checkpoint_test_problem",
        path=tmp_path,
        version=1,
        description="Synthetic 3-checkpoint problem for diff-chain tests.",
        tags=["test"],
        entry_file="solution",
        checkpoints={
            "checkpoint_1": CheckpointConfig(
                name="checkpoint_1",
                version=1,
                order=1,
                spec_override="spec for c1",
            ),
            "checkpoint_2": CheckpointConfig(
                name="checkpoint_2",
                version=1,
                order=2,
                spec_override="spec for c2",
            ),
            "checkpoint_3": CheckpointConfig(
                name="checkpoint_3",
                version=1,
                order=3,
                spec_override="spec for c3",
            ),
        },
    )


def _read_diff(output_dir: Path, step_dir_name: str) -> dict:
    """Load diff.json from a checkpoint or refactor directory."""
    diff_path = output_dir / step_dir_name / DIFF_FILENAME
    assert diff_path.exists(), f"diff.json missing under {step_dir_name!r}"
    return json.loads(diff_path.read_text())


# ===========================================================================
# §4.3 D3 — Diff chain E2E (TestE2EDiffChain)
# ===========================================================================


class TestE2EDiffChain:
    """E2E diff-chain tests covering multi-step chains and modify semantics.

    Scenario A — multi-step chain (c1→r1→c2→r2→c3):
        The refactor after c_i writes a new sentinel file.  D2 guarantees that
        sentinel does NOT reappear as 'created' in c_{i+1}'s diff (it is already
        part of the refactored baseline).  Positive controls confirm that each
        checkpoint's OWN new file IS listed as 'created' in its diff.

    Scenario B — refactor modifies an existing file:
        c1 writes file F with content A.  r1 overwrites F with content B →
        r1's diff records F as 'modified'.  c2's diff must NOT record F as
        'modified' (B is the new baseline).  c2 then writes F with content C →
        c2's diff records F as 'modified' (relative to B).

    Both scenarios are exercised twice: with LocalEnvironmentSpec and with the
    real Docker environment (marked @pytest.mark.integration).
    """

    # ------------------------------------------------------------------
    # Scenario A helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chain_solutions() -> list[dict[str, str]]:
        """Three checkpoint solutions: each writes its own distinguishable file."""
        return [
            {"c1_feature.txt": "feature written by c1"},
            {"c2_feature.txt": "feature written by c2"},
            {"c3_feature.txt": "feature written by c3"},
        ]

    @staticmethod
    def _assert_chain_scenario(output_dir: Path) -> None:
        """Core assertions for Scenario A — shared by local and Docker variants.

        Because the two refactors use different scripts, they cannot be combined
        into a single ScriptRefactorSpec.  Instead, this test verifies the
        fundamental D2/D3 invariant: the sentinel a refactor creates does NOT
        reappear as 'created' in the NEXT checkpoint's diff.  The test also
        checks the positive control (each checkpoint's own file IS created).
        """
        # r1's own diff: r1_sentinel must be listed as created (D1)
        r1_diff = _read_diff(output_dir, "checkpoint_1__refactor")
        assert "r1_sentinel.txt" in _created_paths(r1_diff), (
            "r1_sentinel.txt should appear as 'created' in r1's diff (D1 positive control)"
        )

        # c2's diff: c2_feature created (positive control); r1_sentinel NOT created (D2)
        c2_diff = _read_diff(output_dir, "checkpoint_2")
        assert "c2_feature.txt" in _created_paths(c2_diff), (
            "c2_feature.txt should appear as 'created' in c2's diff (positive control)"
        )
        assert "r1_sentinel.txt" not in _created_paths(c2_diff), (
            "r1_sentinel.txt must NOT appear as 'created' in c2's diff — "
            "c2 was re-baselined onto r1's snapshot, so the sentinel is already there (D2)"
        )

        # r2's own diff: r2_sentinel must be listed as created (D1)
        r2_diff = _read_diff(output_dir, "checkpoint_2__refactor")
        assert "r2_sentinel.txt" in _created_paths(r2_diff), (
            "r2_sentinel.txt should appear as 'created' in r2's diff (D1 positive control)"
        )

        # c3's diff: c3_feature created (positive control); r2_sentinel NOT created (D2)
        c3_diff = _read_diff(output_dir, "checkpoint_3")
        assert "c3_feature.txt" in _created_paths(c3_diff), (
            "c3_feature.txt should appear as 'created' in c3's diff (positive control)"
        )
        assert "r2_sentinel.txt" not in _created_paths(c3_diff), (
            "r2_sentinel.txt must NOT appear as 'created' in c3's diff — "
            "c3 was re-baselined onto r2's snapshot (D2)"
        )

    # ------------------------------------------------------------------
    # Scenario A — local
    # ------------------------------------------------------------------

    def test_ScenA_multi_step_chain_local(
        self,
        tmp_path: Path,
        local_env: LocalEnvironmentSpec,
    ) -> None:
        """Scenario A (local): c1→r1→c2→r2→c3 diff chain uses the correct baselines.

        Each refactor creates a sentinel file; D2 requires those sentinels do NOT
        reappear as 'created' in the subsequent checkpoint's diff.
        """
        problem = _make_3chkpt_problem(tmp_path / "problem")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Single script that writes r1_sentinel on first invocation and
        # r2_sentinel on the second, driven purely by which file already exists.
        # This exercises the two-refactor chain (r1 after c1, r2 after c2) with
        # a single ScriptRefactorSpec — the only supported API.
        combined_script = _make_named_script(
            tmp_path / "combined_refactor.sh",
            """
            working_dir="$1"
            if [ ! -f "$working_dir/r1_sentinel.txt" ]; then
                echo "r1" > "$working_dir/r1_sentinel.txt"
            else
                echo "r2" > "$working_dir/r2_sentinel.txt"
            fi
            """,
        )
        spec = ScriptRefactorSpec(command=combined_script)
        run_spec = _make_run_spec(problem, local_env)
        agent = DummyAgent(self._chain_solutions())
        _run(problem, run_spec, output_dir, agent, refactor_spec=spec)

        self._assert_chain_scenario(output_dir)

    # ------------------------------------------------------------------
    # Scenario A — Docker
    # ------------------------------------------------------------------

    @pytest.mark.integration
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_ScenA_multi_step_chain_docker(self, tmp_path: Path) -> None:
        """Scenario A (Docker): same diff-chain invariants verified against a real Docker environment."""
        docker_env = _load_docker_env()
        problem = _make_3chkpt_problem(tmp_path / "problem")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        combined_script = _make_named_script(
            tmp_path / "combined_refactor.sh",
            """
            working_dir="$1"
            if [ ! -f "$working_dir/r1_sentinel.txt" ]; then
                echo "r1" > "$working_dir/r1_sentinel.txt"
            else
                echo "r2" > "$working_dir/r2_sentinel.txt"
            fi
            """,
        )
        spec = ScriptRefactorSpec(command=combined_script)
        run_spec = _make_run_spec(problem, docker_env)
        agent = DummyAgent(self._chain_solutions())
        try:
            _run(problem, run_spec, output_dir, agent, refactor_spec=spec)
            self._assert_chain_scenario(output_dir)
        finally:
            cleanup_as_root(output_dir)

    # ------------------------------------------------------------------
    # Scenario B helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_modify_script(tmp_path: Path, shared_file: str) -> str:
        """Return a script path for Scenario B.

        The script overwrites shared_file with content B (simulating a refactor
        that rewrites an existing file rather than creating a new one).
        """
        return _make_named_script(
            tmp_path / "r1_modify.sh",
            f"""
            working_dir="$1"
            echo "content_B" > "$working_dir/{shared_file}"
            """,
        )

    @staticmethod
    def _assert_modify_scenario(output_dir: Path, shared_file: str) -> None:
        """Core assertions for Scenario B — shared by local and Docker variants."""
        # r1's diff: shared_file must be 'modified' (c1 wrote A; r1 rewrote to B)
        r1_diff = _read_diff(output_dir, "checkpoint_1__refactor")
        assert shared_file in _modified_paths(r1_diff), (
            f"'{shared_file}' should appear as 'modified' in r1's diff "
            f"(refactor rewrote content A → B)"
        )

        # c2's diff: shared_file must NOT be 'modified' — B is the new baseline;
        # c2 hasn't touched it yet (only c2_extra.txt was written by c2's solution).
        # It also must not appear as 'created' (it was already there).
        c2_diff = _read_diff(output_dir, "checkpoint_2")
        assert shared_file not in _modified_paths(c2_diff), (
            f"'{shared_file}' must NOT appear as 'modified' in c2's diff — "
            "the refactored content B is the new baseline for c2 (D2/D3)"
        )
        assert shared_file not in _created_paths(c2_diff), (
            f"'{shared_file}' must NOT appear as 'created' in c2's diff either"
        )

        # Positive control: c2 wrote an extra file that must show as created
        assert "c2_extra.txt" in _created_paths(c2_diff), (
            "c2_extra.txt should appear as 'created' in c2's diff (positive control)"
        )

    # ------------------------------------------------------------------
    # Scenario B — local
    # ------------------------------------------------------------------

    def test_ScenB_modify_chain_local(
        self,
        tmp_path: Path,
        local_env: LocalEnvironmentSpec,
    ) -> None:
        """Scenario B (local): refactor modifies file A→B; next checkpoint's diff uses B as baseline.

        c1 writes shared.txt with content A.
        r1 overwrites it with content B → r1's diff records it as 'modified'.
        c2 adds only c2_extra.txt; shared.txt must NOT appear as 'modified' in c2's diff.
        """
        shared_file = "shared.txt"
        problem = _make_3chkpt_problem(tmp_path / "problem")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        r1_script = self._make_modify_script(tmp_path, shared_file)
        spec = ScriptRefactorSpec(command=r1_script)

        # c1 writes shared.txt (content A); c2 writes ONLY c2_extra.txt (leaving shared.txt as B)
        solutions = [
            {shared_file: "content_A"},
            {"c2_extra.txt": "extra file from c2"},
            # c3 not needed for the assertions, but the runner expects 3 solutions
            {"c3_feature.txt": "c3 placeholder"},
        ]
        run_spec = _make_run_spec(problem, local_env)
        agent = DummyAgent(solutions)
        _run(problem, run_spec, output_dir, agent, refactor_spec=spec)

        self._assert_modify_scenario(output_dir, shared_file)

    # ------------------------------------------------------------------
    # Scenario B — Docker
    # ------------------------------------------------------------------

    @pytest.mark.integration
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_ScenB_modify_chain_docker(self, tmp_path: Path) -> None:
        """Scenario B (Docker): same modify-then-rebaseline invariants in a real Docker environment."""
        shared_file = "shared.txt"
        docker_env = _load_docker_env()
        problem = _make_3chkpt_problem(tmp_path / "problem")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        r1_script = self._make_modify_script(tmp_path, shared_file)
        spec = ScriptRefactorSpec(command=r1_script)

        solutions = [
            {shared_file: "content_A"},
            {"c2_extra.txt": "extra file from c2"},
            {"c3_feature.txt": "c3 placeholder"},
        ]
        run_spec = _make_run_spec(problem, docker_env)
        agent = DummyAgent(solutions)
        try:
            _run(problem, run_spec, output_dir, agent, refactor_spec=spec)
            self._assert_modify_scenario(output_dir, shared_file)
        finally:
            cleanup_as_root(output_dir)


# ===========================================================================
# Cross-check: diff.json claims vs snapshot directory contents
# ===========================================================================


def _snapshot_dir(output_dir: Path, step_dir_name: str) -> Path:
    return output_dir / step_dir_name / SNAPSHOT_DIR_NAME


def _snapshot_file_content(
    output_dir: Path, step_dir_name: str, filename: str
) -> str:
    p = _snapshot_dir(output_dir, step_dir_name) / filename
    assert p.exists(), (
        f"Expected {filename!r} in snapshot for {step_dir_name!r}"
    )
    return p.read_text()


def _diff_text_for(diff_json: dict, filename: str) -> str | None:
    """Return the diff_text for a file in a serialized SnapshotDiff, or None."""
    for path_str, fd in diff_json.get("file_diffs", {}).items():
        if Path(path_str).name == filename:
            return fd.get("diff_text")
    return None


class TestDiffVsSnapshot:
    """Cross-check diff.json entries against what is actually on disk in the snapshots.

    These tests run the Scenario B chain (c1 writes shared.txt with content A;
    r1 overwrites with content B; c2 adds c2_extra.txt) and verify:

    1. Files the diff claims were 'created' actually exist in the snapshot and
       contain the content that was written.
    2. Files the diff claims were 'modified' exist in the snapshot with the NEW
       content, and their diff_text shows the A→B transition explicitly.
    3. Files untouched by a step are absent from that step's diff entirely,
       but still present in the snapshot (carried forward from the prior baseline).
    4. The accumulated snapshot after c2 contains files from all prior steps
       (c1's files + r1's changes + c2's new files).
    """

    CONTENT_A = "content_A\n"
    CONTENT_B = "content_B\n"
    CONTENT_C = "content_C\n"

    @staticmethod
    def _run_scenario(tmp_path: Path, env) -> tuple[Path, str]:
        """Run Scenario B and return (output_dir, shared_file)."""
        shared_file = "shared.txt"
        problem = _make_3chkpt_problem(tmp_path / "problem")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # r1 rewrites shared.txt: A → B
        script = _make_named_script(
            tmp_path / "r1.sh",
            f'echo -n "content_B\\n" > "$1/{shared_file}"',
        )
        spec = ScriptRefactorSpec(command=script)
        solutions = [
            # c1: creates shared.txt with content A; also creates c1_only.txt
            {shared_file: "content_A\n", "c1_only.txt": "only in c1\n"},
            # c2: does NOT touch shared.txt; adds c2_extra.txt
            {"c2_extra.txt": "extra\n"},
            {"c3_feature.txt": "c3\n"},
        ]
        run_spec = _make_run_spec(problem, env)
        agent = DummyAgent(solutions)
        _run(problem, run_spec, output_dir, agent, refactor_spec=spec)
        return output_dir, shared_file

    # ------------------------------------------------------------------
    # 1. Created file: diff claims 'created' → file exists in snapshot
    #    with the content that was written
    # ------------------------------------------------------------------

    def test_created_file_exists_in_snapshot_with_correct_content(
        self, tmp_path: Path, local_env: LocalEnvironmentSpec
    ) -> None:
        """c2_extra.txt is 'created' in c2's diff → it exists in c2's snapshot with the right content."""
        output_dir, _ = self._run_scenario(tmp_path, local_env)

        c2_diff = _read_diff(output_dir, "checkpoint_2")
        assert "c2_extra.txt" in _created_paths(c2_diff), (
            "pre-condition: c2_extra.txt is created"
        )

        content = _snapshot_file_content(
            output_dir, "checkpoint_2", "c2_extra.txt"
        )
        assert content == "extra\n", f"snapshot content mismatch: {content!r}"

    # ------------------------------------------------------------------
    # 2. Modified file: diff claims 'modified' → snapshot has new content;
    #    diff_text shows the A→B transition
    # ------------------------------------------------------------------

    def test_modified_file_has_new_content_in_snapshot(
        self, tmp_path: Path, local_env: LocalEnvironmentSpec
    ) -> None:
        """r1 overwrites shared.txt A→B; r1's snapshot must contain content B."""
        output_dir, shared_file = self._run_scenario(tmp_path, local_env)

        r1_diff = _read_diff(output_dir, "checkpoint_1__refactor")
        assert shared_file in _modified_paths(r1_diff), (
            "pre-condition: shared.txt is modified in r1"
        )

        content = _snapshot_file_content(
            output_dir, "checkpoint_1__refactor", shared_file
        )
        assert "content_B" in content, (
            f"r1's snapshot should have content B after refactor; got {content!r}"
        )
        assert "content_A" not in content, (
            f"r1's snapshot must not retain content A; got {content!r}"
        )

    def test_diff_text_shows_ab_transition(
        self, tmp_path: Path, local_env: LocalEnvironmentSpec
    ) -> None:
        """r1's diff_text for shared.txt explicitly removes content A and adds content B."""
        output_dir, shared_file = self._run_scenario(tmp_path, local_env)

        r1_diff = _read_diff(output_dir, "checkpoint_1__refactor")
        diff_text = _diff_text_for(r1_diff, shared_file)
        assert diff_text is not None, (
            f"diff_text missing for {shared_file!r} in r1's diff"
        )
        assert "-content_A" in diff_text, (
            f"diff_text should remove content_A (got {diff_text!r})"
        )
        assert "+content_B" in diff_text, (
            f"diff_text should add content_B (got {diff_text!r})"
        )

    # ------------------------------------------------------------------
    # 3. Untouched file absent from step's diff, present in snapshot
    # ------------------------------------------------------------------

    def test_untouched_file_absent_from_diff_but_present_in_snapshot(
        self, tmp_path: Path, local_env: LocalEnvironmentSpec
    ) -> None:
        """shared.txt is untouched by c2 → absent from c2's diff, but still in c2's snapshot."""
        output_dir, shared_file = self._run_scenario(tmp_path, local_env)

        c2_diff = _read_diff(output_dir, "checkpoint_2")
        all_diff_files = {Path(p).name for p in c2_diff.get("file_diffs", {})}
        assert shared_file not in all_diff_files, (
            f"{shared_file!r} must not appear in c2's diff at all (untouched)"
        )

        # Still in the snapshot (carried forward)
        content = _snapshot_file_content(
            output_dir, "checkpoint_2", shared_file
        )
        assert "content_B" in content, (
            f"c2's snapshot should still have content B for {shared_file!r}; got {content!r}"
        )

    # ------------------------------------------------------------------
    # 4. Accumulated snapshot: c2's snapshot contains files from all prior steps
    # ------------------------------------------------------------------

    def test_accumulated_snapshot_contains_all_prior_files(
        self, tmp_path: Path, local_env: LocalEnvironmentSpec
    ) -> None:
        """c2's snapshot accumulates files from c1 (c1_only.txt, shared.txt) and r1 plus c2's own files."""
        output_dir, shared_file = self._run_scenario(tmp_path, local_env)

        # c1_only.txt was created by c1 and never deleted → must survive in c2's snapshot
        c1_only = _snapshot_file_content(
            output_dir, "checkpoint_2", "c1_only.txt"
        )
        assert c1_only == "only in c1\n", (
            f"c1_only.txt content mismatch: {c1_only!r}"
        )

        # shared.txt was modified by r1 to content B → c2's snapshot still has content B
        shared = _snapshot_file_content(output_dir, "checkpoint_2", shared_file)
        assert "content_B" in shared

        # c2's own file is present
        extra = _snapshot_file_content(
            output_dir, "checkpoint_2", "c2_extra.txt"
        )
        assert extra == "extra\n", f"c2_extra.txt content mismatch: {extra!r}"

    # ------------------------------------------------------------------
    # 5. c1's diff contains the files c1 created
    # ------------------------------------------------------------------

    def test_c1_diff_contains_created_files(
        self, tmp_path: Path, local_env: LocalEnvironmentSpec
    ) -> None:
        """c1 creates shared.txt and c1_only.txt → both appear as 'created' in checkpoint_1's diff."""
        output_dir, shared_file = self._run_scenario(tmp_path, local_env)

        c1_diff = _read_diff(output_dir, "checkpoint_1")
        created = _created_paths(c1_diff)
        assert shared_file in created, (
            f"{shared_file!r} must appear as 'created' in c1's diff; got {created!r}"
        )
        assert "c1_only.txt" in created, (
            f"c1_only.txt must appear as 'created' in c1's diff; got {created!r}"
        )

    # ------------------------------------------------------------------
    # Docker variants of the four tests above
    # ------------------------------------------------------------------

    @pytest.mark.integration
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_created_file_exists_in_snapshot_with_correct_content_docker(
        self, tmp_path: Path
    ) -> None:
        """Docker: c2_extra.txt 'created' in diff → correct content in snapshot."""
        docker_env = _load_docker_env()
        output_dir, _ = self._run_scenario(tmp_path, docker_env)
        try:
            c2_diff = _read_diff(output_dir, "checkpoint_2")
            assert "c2_extra.txt" in _created_paths(c2_diff)
            content = _snapshot_file_content(
                output_dir, "checkpoint_2", "c2_extra.txt"
            )
            assert content == "extra\n", (
                f"snapshot content mismatch: {content!r}"
            )
        finally:
            cleanup_as_root(output_dir)

    @pytest.mark.integration
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_modified_file_has_new_content_in_snapshot_docker(
        self, tmp_path: Path
    ) -> None:
        """Docker: r1's snapshot has content B for shared.txt."""
        docker_env = _load_docker_env()
        output_dir, shared_file = self._run_scenario(tmp_path, docker_env)
        try:
            content = _snapshot_file_content(
                output_dir, "checkpoint_1__refactor", shared_file
            )
            assert "content_B" in content
            assert "content_A" not in content
        finally:
            cleanup_as_root(output_dir)

    @pytest.mark.integration
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_diff_text_shows_ab_transition_docker(self, tmp_path: Path) -> None:
        """Docker: r1's diff_text for shared.txt shows A→B."""
        docker_env = _load_docker_env()
        output_dir, shared_file = self._run_scenario(tmp_path, docker_env)
        try:
            r1_diff = _read_diff(output_dir, "checkpoint_1__refactor")
            diff_text = _diff_text_for(r1_diff, shared_file)
            assert diff_text is not None
            assert "-content_A" in diff_text
            assert "+content_B" in diff_text
        finally:
            cleanup_as_root(output_dir)

    @pytest.mark.integration
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_untouched_file_absent_from_diff_but_present_in_snapshot_docker(
        self, tmp_path: Path
    ) -> None:
        """Docker: shared.txt untouched by c2 → absent from diff, present in snapshot."""
        docker_env = _load_docker_env()
        output_dir, shared_file = self._run_scenario(tmp_path, docker_env)
        try:
            c2_diff = _read_diff(output_dir, "checkpoint_2")
            all_diff_files = {
                Path(p).name for p in c2_diff.get("file_diffs", {})
            }
            assert shared_file not in all_diff_files
            content = _snapshot_file_content(
                output_dir, "checkpoint_2", shared_file
            )
            assert "content_B" in content
        finally:
            cleanup_as_root(output_dir)

    @pytest.mark.integration
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_accumulated_snapshot_contains_all_prior_files_docker(
        self, tmp_path: Path
    ) -> None:
        """Docker: c2's snapshot accumulates files from c1, r1, and c2."""
        docker_env = _load_docker_env()
        output_dir, shared_file = self._run_scenario(tmp_path, docker_env)
        try:
            c1_only = _snapshot_file_content(
                output_dir, "checkpoint_2", "c1_only.txt"
            )
            assert c1_only == "only in c1\n"
            shared = _snapshot_file_content(
                output_dir, "checkpoint_2", shared_file
            )
            assert "content_B" in shared
            extra = _snapshot_file_content(
                output_dir, "checkpoint_2", "c2_extra.txt"
            )
            assert extra == "extra\n"
        finally:
            cleanup_as_root(output_dir)

    @pytest.mark.integration
    @pytest.mark.skipif(not _docker_available(), reason="Docker not available")
    def test_c1_diff_contains_created_files_docker(
        self, tmp_path: Path
    ) -> None:
        """Docker: shared.txt and c1_only.txt appear as 'created' in checkpoint_1's diff."""
        docker_env = _load_docker_env()
        output_dir, shared_file = self._run_scenario(tmp_path, docker_env)
        try:
            c1_diff = _read_diff(output_dir, "checkpoint_1")
            created = _created_paths(c1_diff)
            assert shared_file in created, (
                f"{shared_file!r} must appear as 'created' in c1's diff; got {created!r}"
            )
            assert "c1_only.txt" in created, (
                f"c1_only.txt must appear as 'created' in c1's diff; got {created!r}"
            )
        finally:
            cleanup_as_root(output_dir)
