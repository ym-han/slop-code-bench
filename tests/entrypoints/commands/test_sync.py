from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from slop_code.entrypoints.commands import sync
from slop_code.problem_catalog import CatalogManifest


def test_sync_installs_latest_when_no_version(monkeypatch, tmp_path, capsys):
    calls: list[str | None] = []
    home = tmp_path / "scbench-home"
    manifest = CatalogManifest(version="v1.4.0", commit="deadbeef")

    monkeypatch.setattr(
        sync.problem_catalog,
        "load_manifest",
        lambda _home: None,
    )
    monkeypatch.setattr(
        sync.problem_catalog,
        "sync_catalog",
        lambda target_home, version: calls.append(version) or manifest,
    )
    monkeypatch.setattr(
        sync.problem_catalog,
        "get_catalog_root",
        lambda _home: home / "problems",
    )

    ctx = cast("Any", SimpleNamespace(obj=SimpleNamespace(scbench_home=home)))
    sync.sync_catalog(ctx, version=None)

    assert calls == [None]
    output = capsys.readouterr().out
    assert "Installed catalog version: v1.4.0" in output
    assert "Resolved commit: deadbeef" in output
    assert str(home / "problems") in output


def test_sync_installs_explicit_version(monkeypatch, tmp_path, capsys):
    calls: list[str | None] = []
    home = tmp_path / "scbench-home"
    manifest = CatalogManifest(version="v0.2", commit="cafebabe")

    monkeypatch.setattr(
        sync.problem_catalog,
        "load_manifest",
        lambda _home: None,
    )
    monkeypatch.setattr(
        sync.problem_catalog,
        "sync_catalog",
        lambda _home, version: calls.append(version) or manifest,
    )
    monkeypatch.setattr(
        sync.problem_catalog,
        "get_catalog_root",
        lambda _home: home / "problems",
    )

    ctx = cast("Any", SimpleNamespace(obj=SimpleNamespace(scbench_home=home)))
    sync.sync_catalog(ctx, version="v0.2")

    assert calls == ["v0.2"]
    output = capsys.readouterr().out
    assert "Installed catalog version: v0.2" in output
    assert "Resolved commit: cafebabe" in output


def test_sync_reports_noop_when_requested_version_already_current(
    monkeypatch,
    tmp_path,
    capsys,
):
    home = tmp_path / "scbench-home"
    existing = CatalogManifest(version="v0.2", commit="cafebabe")

    monkeypatch.setattr(
        sync.problem_catalog,
        "load_manifest",
        lambda _home: existing,
    )
    monkeypatch.setattr(
        sync.problem_catalog,
        "sync_catalog",
        lambda _home, _version: existing,
    )
    monkeypatch.setattr(
        sync.problem_catalog,
        "get_catalog_root",
        lambda _home: home / "problems",
    )

    ctx = cast("Any", SimpleNamespace(obj=SimpleNamespace(scbench_home=home)))
    sync.sync_catalog(ctx, version="v0.2")

    output = capsys.readouterr().out
    assert "already current" in output
