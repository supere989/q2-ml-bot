from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from tools.preflight_b2_dyn_invocation import (
    DynInvocationPreflightError,
    SCHEMA,
    preflight,
)


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", repo], check=True)
    (repo / "tracked").write_text("authority\n")
    subprocess.run(["git", "-C", repo, "add", "tracked"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repo,
            "-c",
            "user.name=B2 Test",
            "-c",
            "user.email=b2@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    return repo, commit


def _executable(tmp_path: Path, *, touch_output: bool = False) -> Path:
    executable = tmp_path / "q2-dyn-evidence"
    body = [
        "#!/usr/bin/env python3",
        "import pathlib, sys",
        "assert sys.argv[-2:] == ['--preflight-only', 'true']",
        "origin = sys.argv.index('--expected-origin')",
        "assert sys.argv[origin + 1] == '-512,-512,-512'",
    ]
    if touch_output:
        body.extend(
            [
                "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])",
                "output.mkdir(parents=True)",
            ]
        )
    body.append(f"print('{json.dumps({'passed': True, 'schema': SCHEMA}, separators=(',', ':'))}')")
    executable.write_text("\n".join(body) + "\n")
    executable.chmod(0o755)
    return executable


def _args(tmp_path: Path, repo: Path, commit: str, executable: Path) -> argparse.Namespace:
    root = tmp_path / "future-final-cohort"
    return argparse.Namespace(
        executable=executable,
        repo_root=repo,
        atlas=root / "analysis" / "map.atlas.bin",
        manifest=root / "analysis" / "map.atlas.manifest.json",
        bsp=root / "claims" / "map.bsp",
        expected_map_id="map",
        expected_origin="-512,-512,-512",
        expected_analyzer_authority="11" * 32,
        expected_crate_commit=commit,
        map_epoch=1,
        environment_steps=4000,
        samples=4000,
        output=root / "dyn-evidence",
        report=tmp_path / "dyn-argv-preflight.json",
    )


def _stub_binding(monkeypatch: pytest.MonkeyPatch, repo: Path, commit: str) -> None:
    tree = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD^{tree}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    binding = {
        "repository_commit": commit,
        "repository_tree": tree,
        "git_clean": True,
        "generator_sha256": "22" * 32,
        "routes_sha256": "33" * 32,
        "atlas_analyzer_authority_sha256": "44" * 32,
        "atlas_analyzer_authority_file_count": 1,
    }
    monkeypatch.setattr(
        "tools.preflight_b2_dyn_invocation.repository_binding", lambda _: binding
    )


def test_preflight_attests_separate_negative_origin_and_touches_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, commit = _repo(tmp_path)
    _stub_binding(monkeypatch, repo, commit)
    args = _args(tmp_path, repo, commit, _executable(tmp_path))

    report = preflight(args)

    assert report["passed"] is True
    assert report["producer_output_absent_before"] is True
    assert report["producer_output_absent_after"] is True
    assert not args.output.exists()
    origin = report["producer_argv"].index("--expected-origin")
    assert report["producer_argv"][origin + 1] == "-512,-512,-512"
    assert "--expected-origin=-512,-512,-512" not in report["producer_argv"]
    assert report["preflight_argv"][-2:] == ["--preflight-only", "true"]


def test_preflight_rejects_any_producer_output_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, commit = _repo(tmp_path)
    _stub_binding(monkeypatch, repo, commit)
    args = _args(tmp_path, repo, commit, _executable(tmp_path, touch_output=True))

    with pytest.raises(DynInvocationPreflightError, match="touched the producer output"):
        preflight(args)
