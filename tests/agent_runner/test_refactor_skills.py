"""Tests for agent-kind refactor: per-run ~/.claude seeding, isolation, and the
seed-aware resume identity.

The invariant under test: a refactorer can run arbitrary Claude Code assets
(skills, slash-commands, subagents), different runs can use different ones, and
a run sees only its own — because the assets are seeded into the agent's
per-instance ~/.claude, which is mounted only into that agent's container and
never lands in the workspace.
"""

from __future__ import annotations

import tempfile
import types
from pathlib import Path

import pytest

from slop_code.agent_runner.refactor import AgentRefactorSpec
from slop_code.agent_runner.refactor import ScriptRefactorSpec
from slop_code.agent_runner.refactor import _hash_dir
from slop_code.agent_runner.refactor import compute_refactor_identity


def _make_home(parent: Path, name: str, files: dict[str, str]) -> Path:
    """Build a ~/.claude-shaped template dir; files keyed by relative path."""
    home = parent / name
    for rel, body in files.items():
        dest = home / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body)
    return home


def _agent_spec(agent_config: object, prompt: str = "refactor") -> AgentRefactorSpec:
    # AgentRefactorSpec is a plain dataclass; stub the fields identity doesn't read.
    return AgentRefactorSpec(
        agent_config=agent_config,  # type: ignore[arg-type]
        model_def=types.SimpleNamespace(name="sonnet-4.6"),  # type: ignore[arg-type]
        credential=None,  # type: ignore[arg-type]
        prompt=prompt,
        image="img",
    )


# --- directory hashing -----------------------------------------------------


def test_hash_dir_content_sensitive(tmp_path: Path) -> None:
    a = _make_home(tmp_path / "v1", ".claude", {"skills/r/SKILL.md": "do X"})
    b = _make_home(tmp_path / "v2", ".claude", {"skills/r/SKILL.md": "do Y"})
    assert _hash_dir(a) != _hash_dir(b)


def test_hash_dir_stable(tmp_path: Path) -> None:
    a = _make_home(tmp_path, ".claude", {"skills/r/SKILL.md": "do X"})
    assert _hash_dir(a) == _hash_dir(a)


def test_hash_dir_none_is_empty() -> None:
    assert _hash_dir(None) == ""


def test_hash_dir_includes_relative_paths(tmp_path: Path) -> None:
    # Same file *content* at a different path must hash differently, so that
    # restructuring the seed dir invalidates the cache.
    a = _make_home(tmp_path / "a", ".claude", {"commands/one.md": "same"})
    b = _make_home(tmp_path / "b", ".claude", {"commands/two.md": "same"})
    assert _hash_dir(a) != _hash_dir(b)


# --- resume identity folds in the seed dir ---------------------------------


def test_identity_changes_when_seed_changes(tmp_path: Path) -> None:
    home_v1 = _make_home(tmp_path / "v1", ".claude", {"skills/r/SKILL.md": "A"})
    home_v2 = _make_home(tmp_path / "v2", ".claude", {"skills/r/SKILL.md": "B"})

    cfg_v1 = types.SimpleNamespace(claude_home=home_v1)
    cfg_v2 = types.SimpleNamespace(claude_home=home_v2)

    assert compute_refactor_identity(_agent_spec(cfg_v1)) != compute_refactor_identity(
        _agent_spec(cfg_v2)
    )


def test_identity_stable_for_same_seed(tmp_path: Path) -> None:
    home = _make_home(tmp_path, ".claude", {"skills/r/SKILL.md": "A"})
    cfg = types.SimpleNamespace(claude_home=home)
    assert compute_refactor_identity(_agent_spec(cfg)) == compute_refactor_identity(
        _agent_spec(cfg)
    )


def test_identity_changes_when_prompt_changes() -> None:
    cfg = types.SimpleNamespace(claude_home=None)
    a = compute_refactor_identity(_agent_spec(cfg, prompt="one"))
    b = compute_refactor_identity(_agent_spec(cfg, prompt="two"))
    assert a != b


def test_identity_is_16_hex_chars() -> None:
    cfg = types.SimpleNamespace(claude_home=None)
    h = compute_refactor_identity(_agent_spec(cfg))
    assert len(h) == 16
    int(h, 16)  # parses as hex


def test_script_and_agent_identities_differ() -> None:
    cfg = types.SimpleNamespace(claude_home=None)
    script_id = compute_refactor_identity(ScriptRefactorSpec(command="echo hi"))
    agent_id = compute_refactor_identity(_agent_spec(cfg))
    assert script_id != agent_id


# --- per-instance ~/.claude seeding ----------------------------------------


def _claude_agent(claude_home_template: Path | None):
    from slop_code.agent_runner.agents.claude_code.agent import ClaudeCodeAgent
    from slop_code.agent_runner.credentials import CredentialType
    from slop_code.agent_runner.credentials import ProviderCredential
    from slop_code.agent_runner.models import AgentCostLimits
    from slop_code.common.llms import APIPricing

    credential = ProviderCredential(
        provider="anthropic",
        value="x",
        source="ANTHROPIC_API_KEY",
        destination_key="ANTHROPIC_API_KEY",
        credential_type=CredentialType.ENV_VAR,
    )
    return ClaudeCodeAgent(
        problem_name="p",
        image="img",
        verbose=False,
        cost_limits=AgentCostLimits(
            step_limit=10, cost_limit=100.0, net_cost_limit=200.0
        ),
        pricing=APIPricing(input=0.5, output=2.0, cache_read=0.1),
        credential=credential,
        binary="claude",
        model="claude-sonnet-4-6",
        timeout=None,
        settings={"existingSetting": True},
        env={},
        extra_args=[],
        append_system_prompt=None,
        allowed_tools=[],
        disallowed_tools=[],
        permission_mode=None,
        base_url=None,
        thinking=None,
        max_thinking_tokens=None,
        max_output_tokens=None,
        claude_home_template=claude_home_template,
    )


def test_prepare_mounts_seeds_skills_commands_and_agents(tmp_path: Path) -> None:
    home = _make_home(
        tmp_path / "src",
        ".claude",
        {
            "skills/my-refactor/SKILL.md": "behavior-preserving only",
            "commands/tidy.md": "/tidy the code",
            "agents/reviewer.md": "you are a reviewer",
        },
    )
    agent = _claude_agent(home)
    agent._tmp_dir = tempfile.TemporaryDirectory()
    agent._workspace = tmp_path / "workspace"

    mounts = agent._prepare_mounts()

    claude_home = Path(agent._tmp_dir.name) / "claude_home"
    assert (claude_home / "skills" / "my-refactor" / "SKILL.md").read_text() == (
        "behavior-preserving only"
    )
    assert (claude_home / "commands" / "tidy.md").is_file()
    assert (claude_home / "agents" / "reviewer.md").is_file()
    # And that claude_home is what gets bind-mounted at ~/.claude.
    assert any(m["bind"].endswith("/.claude") for m in mounts.values())
    agent._tmp_dir.cleanup()


def test_harness_settings_overlay_template_settings(tmp_path: Path) -> None:
    # A settings.json carried by the template must NOT shadow the harness's own
    # (which holds auth/thinking). The harness writes last and wins.
    home = _make_home(
        tmp_path / "src", ".claude", {"settings.json": '{"existingSetting": false}'}
    )
    agent = _claude_agent(home)
    agent._tmp_dir = tempfile.TemporaryDirectory()
    agent._workspace = tmp_path / "workspace"

    agent._prepare_mounts()

    import json

    written = json.loads(
        (Path(agent._tmp_dir.name) / "claude_home" / "settings.json").read_text()
    )
    assert written["existingSetting"] is True  # harness value won
    agent._tmp_dir.cleanup()


def test_prepare_mounts_no_template_no_extra_dirs(tmp_path: Path) -> None:
    agent = _claude_agent(None)
    agent._tmp_dir = tempfile.TemporaryDirectory()
    agent._workspace = tmp_path / "workspace"

    agent._prepare_mounts()

    claude_home = Path(agent._tmp_dir.name) / "claude_home"
    assert not (claude_home / "skills").exists()
    assert not (claude_home / "commands").exists()
    assert (claude_home / "settings.json").is_file()
    agent._tmp_dir.cleanup()


def test_prepare_mounts_rejects_missing_template(tmp_path: Path) -> None:
    from slop_code.agent_runner.models import AgentError

    agent = _claude_agent(tmp_path / "does-not-exist")
    agent._tmp_dir = tempfile.TemporaryDirectory()
    agent._workspace = tmp_path / "workspace"

    with pytest.raises(AgentError):
        agent._prepare_mounts()
    agent._tmp_dir.cleanup()
