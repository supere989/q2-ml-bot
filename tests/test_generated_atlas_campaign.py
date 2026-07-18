from __future__ import annotations

import hashlib
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Callable

import pytest

from tools import run_generated_atlas_campaign as campaign
from tools.run_generator_cohort import (
    STAGE_SUFFIXES,
    canonical_bytes,
    file_sha256,
    load_declaration,
    verify_stage_membership,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_declaration_document() -> dict[str, object]:
    styles = (
        "open",
        "towers",
        "canyon",
        "pits",
        "arena_open",
        "arena_vertical",
        "arena_lanes",
    )
    maps = []
    for style_index, style in enumerate(styles):
        for member in range(4):
            ordinal = style_index * 4 + member
            maps.append({
                "ordinal": ordinal,
                "map": f"b2g26_test_fresh_{ordinal:02d}",
                "seed": 99_000_000 + ordinal,
                "style": style,
                "grid": 5,
                "observed_heat": None,
            })
    return {
        "schema": "q2-b2-generated-cohort-declaration-v1",
        "cohort_id": "b2g26_test_fresh_99000",
        "mode": "final",
        "selection": {
            "policy": "all-or-nothing",
            "required_map_count": 28,
            "required_maps_per_style": 4,
            "required_concrete_styles": list(styles),
            "timing": "declared-before-generation",
            "salvage_allowed": False,
            "replacement_allowed": False,
        },
        "generator": {
            "version": "v6",
            "grid": 5,
            "gym": False,
            "observed_heat": None,
        },
        "source_suffixes": [
            ".map",
            ".json",
            ".meta.json",
            ".lattice.json",
            ".routes.json",
        ],
        "implementation_binding": {
            "bind_repository_commit": True,
            "bind_repository_tree": True,
            "require_clean_git": True,
            "bind_atlas_analyzer_closure": True,
        },
        "maps": maps,
    }


def _declaration_path(claims_directory: Path) -> Path:
    return claims_directory.parent / "synthetic-unretired-declaration.json"


@lru_cache(maxsize=1)
def binding() -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
    ).strip()
    # Production deliberately rejects a dirty working closure while executing
    # the committed snapshot.  These unit tests inject their builder, so their
    # synthetic clean binding must likewise describe the exact HEAD archive,
    # not unrelated concurrent edits in the shared development worktree.
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", commit], cwd=ROOT
    )
    with tempfile.TemporaryDirectory(prefix="q2-campaign-test-head-") as raw:
        snapshot = Path(raw)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
            stream.extractall(snapshot)
        return {
            "repository_commit": commit,
            "repository_tree": tree,
            "git_clean": True,
            "atlas_analyzer_authority_sha256": (
                campaign.atlas_analyzer_authority_sha256(snapshot)
            ),
            "atlas_analyzer_authority_file_count": len(
                campaign.atlas_analyzer_authority_inputs(snapshot)
            ),
            "generator_sha256": file_sha256(snapshot / "maps/generator.py"),
            "routes_sha256": file_sha256(snapshot / "maps/routes.py"),
        }


def write_claims_stage(directory: Path) -> dict:
    declaration_path = _declaration_path(directory)
    declaration_path.write_bytes(
        canonical_bytes(_synthetic_declaration_document())
    )
    declaration, _digest = load_declaration(declaration_path)
    directory.mkdir()
    for row in declaration["maps"]:
        for suffix in STAGE_SUFFIXES["claims"]:
            (directory / f"{row['map']}{suffix}").write_bytes(
                f"claims:{row['ordinal']}:{row['map']}:{suffix}\n".encode()
            )
    return declaration


def write_map_build(
    declared: dict,
    claims_dir: Path,
    output_dir: Path,
    *,
    extra_file: bool = False,
    manifest_mutator: Callable[[dict], None] | None = None,
    tamper_atlas_after_manifest: bool = False,
) -> campaign.BuildProcessResult:
    name = declared["map"]
    atlas_path = output_dir / f"{name}.atlas.bin"
    atlas_path.write_bytes(f"atlas:{name}\n".encode())
    atlas_sha256 = file_sha256(atlas_path)
    for suffix in STAGE_SUFFIXES["analysis"]:
        path = output_dir / f"{name}{suffix}"
        if suffix in (
            ".atlas.bin",
            ".analysis.manifest.json",
            ".atlas.manifest.json",
        ):
            continue
        path.write_bytes(f"analysis:{name}:{suffix}\n".encode())
    bsp_sha256 = file_sha256(claims_dir / f"{name}.bsp")
    claims_sha256 = file_sha256(
        claims_dir / f"{name}.generator-claims.json"
    )
    authority = binding()
    atlas_manifest = {
        "schema_version": 1,
        "bsp": {"canonical_map_id": name, "sha256": bsp_sha256},
        "analyzer": {
            "sha256": authority["atlas_analyzer_authority_sha256"]
        },
        "generator": {"sha256": authority["generator_sha256"]},
        "artifacts": {
            f"{name}.atlas.bin": {
                "sha256_uncompressed": atlas_sha256,
                "uncompressed_size": atlas_path.stat().st_size,
            }
        },
    }
    atlas_manifest_path = output_dir / f"{name}.atlas.manifest.json"
    atlas_manifest_path.write_bytes(
        json.dumps(
            atlas_manifest, ensure_ascii=True, separators=(",", ":")
        ).encode("ascii") + b"\n"
    )
    cold_suffixes = {
        ".atlas.bin",
        ".atlas.bin.zst",
        ".navigation.bin.zst",
        ".visibility.bin.zst",
        ".design-signature.json",
        ".objectives.json",
    }
    artifact_hashes = {
        suffix: file_sha256(output_dir / f"{name}{suffix}")
        for suffix in cold_suffixes
    }
    semantic_hashes = {
        ".analysis.manifest.json": hashlib.sha256(
            f"analysis-semantic:{name}".encode()
        ).hexdigest(),
        ".atlas.manifest.json": hashlib.sha256(
            f"atlas-manifest-semantic:{name}".encode()
        ).hexdigest(),
    }
    full_cold = {
        "schema": campaign.FULL_COLD_SCHEMA,
        "independent_process_launches": 1,
        "artifact_count": 8,
        "artifact_sha256": artifact_hashes,
        "artifact_semantic_sha256": semantic_hashes,
        "cold_artifact_sha256": artifact_hashes,
        "cold_artifact_semantic_sha256": semantic_hashes,
        "verification": {
            "passed": True,
            "canonical_map_id": name,
            "bsp_sha256": bsp_sha256,
            "atlas_sha256": atlas_sha256,
        },
        "elapsed_milliseconds": 1,
        "timeout_limit_milliseconds": 300_000,
    }
    manifest = {
        "schema": campaign.ANALYSIS_MANIFEST_SCHEMA,
        "status": "passed",
        "deterministic_rebuild": True,
        "confidence": "high",
        "canonical_map_id": name,
        "bsp": {"sha256": bsp_sha256},
        "generator_claims_sha256": claims_sha256,
        "identity": {
            "bsp_sha256": bsp_sha256,
            "generator_claims_sha256": claims_sha256,
            "atlas_sha256": atlas_sha256,
            "analyzer_sha256": authority[
                "atlas_analyzer_authority_sha256"
            ],
            "atlas_manifest_sha256": file_sha256(atlas_manifest_path),
        },
        "artifacts": {
            "atlas": {"uncompressed_sha256": atlas_sha256},
            "atlas_manifest": {"sha256": file_sha256(atlas_manifest_path)},
        },
        "performance": {"full_cold_rebuild": full_cold},
    }
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    (output_dir / f"{name}.analysis.manifest.json").write_bytes(
        canonical_bytes(manifest)
    )
    if tamper_atlas_after_manifest:
        atlas_path.write_bytes(atlas_path.read_bytes() + b"tampered\n")
    manifest_path = output_dir / f"{name}.analysis.manifest.json"
    summary = {
        "schema": campaign.BUILD_SUMMARY_SCHEMA,
        "maps": [{
            "canonical_map_id": name,
            "bsp_sha256": file_sha256(claims_dir / f"{name}.bsp"),
            "atlas_sha256": atlas_sha256,
            "manifest_sha256": file_sha256(manifest_path),
        }],
    }
    (output_dir / campaign.BUILD_SUMMARY_NAME).write_bytes(
        canonical_bytes(summary)
    )
    if extra_file:
        (output_dir / "passing-subset.txt").write_text("not admissible\n")
    return campaign.BuildProcessResult(
        returncode=0,
        stdout=(json.dumps(summary, sort_keys=True) + "\n").encode(),
        stderr=f"atlas: {name} completed\n".encode(),
    )


def paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    return (
        tmp_path / "analysis",
        tmp_path / "diagnostics",
        tmp_path / "reports" / "atlas-build.json",
        tmp_path / "claims",
    )


def test_campaign_schema_freezes_exact_28_map_224_file_contract() -> None:
    schema = json.loads((
        ROOT / "schemas/q2-generated-atlas-build-campaign-v1.schema.json"
    ).read_text())

    assert schema["$id"] == (
        "urn:q2-ml:q2-generated-atlas-build-campaign-v1"
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema"]["const"] == campaign.CAMPAIGN_SCHEMA
    assert schema["properties"]["expected_map_count"]["const"] == 28
    assert schema["properties"]["maps"]["minItems"] == 28
    assert schema["properties"]["maps"]["maxItems"] == 28
    assert schema["properties"]["input_claims"]["allOf"][1][
        "properties"
    ]["expected_file_count"]["const"] == 224
    assert schema["properties"]["output_analysis"]["allOf"][1][
        "properties"
    ]["expected_file_count"]["const"] == 224
    assert "claims_snapshot_sha256" in schema["required"]
    success = schema["oneOf"][0]["properties"]
    assert success["passed"]["const"] is True
    assert success["published"]["const"] is True
    assert success["pass_count"]["const"] == 28
    assert success["failures"]["maxItems"] == 0
    assert set(schema["$defs"]["success_artifact_set"]["required"]) == set(
        STAGE_SUFFIXES["analysis"]
    )
    assert schema["$defs"]["success_map_row"]["allOf"][1][
        "properties"
    ]["analysis_manifest"]["$ref"].endswith("analysis_manifest_binding")


def test_default_builder_is_one_explicit_build_map_atlas_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    claims = tmp_path / "claims"
    declaration = write_claims_stage(claims)
    declared = declaration["maps"][0]
    calls: list[tuple[list[str], dict]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Completed", (), {
            "returncode": 0,
            "stdout": b"summary\n",
            "stderr": b"log\n",
        })()

    monkeypatch.setattr(campaign.subprocess, "run", run)
    execution_root = tmp_path / "execution"
    builder = campaign._default_builder_factory(
        execution_root=execution_root,
        client_root=tmp_path / "client",
        lithium_root=tmp_path / "lithium",
        hook_attestation=tmp_path / "hook.json",
        fall_oracle=tmp_path / "fall",
        packer=tmp_path / "pack",
        verifier=tmp_path / "verify",
    )
    result = builder(
        declared, claims, tmp_path / "output", execution_root
    )

    assert result == campaign.BuildProcessResult(0, b"summary\n", b"log\n")
    assert len(calls) == 1
    command, options = calls[0]
    assert command[1] == str(execution_root / "tools/build_map_atlas.py")
    assert command[command.index("--bsp") + 1] == str(
        claims / f"{declared['map']}.bsp"
    )
    assert command[command.index("--generator-claims") + 1] == str(
        claims / f"{declared['map']}.generator-claims.json"
    )
    assert command[command.index("--map-id") + 1] == declared["map"]
    assert options == {
        "cwd": execution_root,
        "stdout": campaign.subprocess.PIPE,
        "stderr": campaign.subprocess.PIPE,
        "check": False,
    }


def test_builds_declared_order_and_atomically_publishes_exact_224_files(
    tmp_path: Path,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    declaration = write_claims_stage(claims)
    calls: list[tuple[int, str]] = []

    def builder(declared, claims_dir, output_dir, execution_root):
        assert not analysis.exists()
        assert claims_dir != claims
        assert execution_root != ROOT
        assert not os.access(claims_dir, os.W_OK)
        assert not os.access(execution_root, os.W_OK)
        calls.append((declared["ordinal"], declared["map"]))
        return write_map_build(declared, claims_dir, output_dir)

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )

    expected = [(row["ordinal"], row["map"]) for row in declaration["maps"]]
    assert calls == expected
    assert report["schema"] == campaign.CAMPAIGN_SCHEMA
    assert report["passed"] is True
    assert report["published"] is True
    assert report["pass_count"] == 28
    assert report["implementation"] == binding()
    assert report["input_claims"]["expected_file_count"] == 224
    assert report["output_analysis"]["expected_file_count"] == 224
    assert report["output_analysis"]["actual_file_count"] == 224
    assert report["output_analysis"]["passed"] is True
    assert report_path.read_bytes() == canonical_bytes(report)
    assert verify_stage_membership(
        declaration, analysis, "analysis"
    )["passed"] is True
    assert not (analysis / campaign.BUILD_SUMMARY_NAME).exists()
    assert len([path for path in analysis.iterdir() if path.is_file()]) == 224
    assert len([path for path in diagnostics.iterdir() if path.is_file()]) == 84
    for row in report["maps"]:
        assert row["passed"] is True
        assert list(row["artifacts"]) == list(STAGE_SUFFIXES["analysis"])
        assert row["build_summary"]["maps"][0]["canonical_map_id"] == row["map"]
        summary_path = diagnostics / f"{row['map']}.build-summary.json"
        assert row["build_summary_sha256"] == file_sha256(summary_path)
        assert row["artifacts"][".analysis.manifest.json"]["sha256"] == (
            file_sha256(analysis / f"{row['map']}.analysis.manifest.json")
        )


def test_one_builder_failure_runs_full_declaration_and_publishes_no_subset(
    tmp_path: Path,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    declaration = write_claims_stage(claims)
    failed = declaration["maps"][13]["map"]
    calls: list[str] = []

    def builder(declared, claims_dir, output_dir, _execution_root):
        calls.append(declared["map"])
        if declared["map"] == failed:
            return campaign.BuildProcessResult(7, b"", b"oracle rejected map\n")
        return write_map_build(declared, claims_dir, output_dir)

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )

    assert calls == [row["map"] for row in declaration["maps"]]
    assert report["passed"] is False
    assert report["published"] is False
    assert report["pass_count"] == 27
    assert not analysis.exists()
    assert report_path.read_bytes() == canonical_bytes(report)
    assert report["maps"][13]["error"] == "build_map_atlas exited 7"
    assert f"{failed}: build_map_atlas exited 7" in report["failures"]
    assert sum(
        failure.startswith("analysis-stage: missing file")
        for failure in report["failures"]
    ) == 8
    assert (diagnostics / f"{failed}.build.stderr.log").read_bytes() == (
        b"oracle rejected map\n"
    )


def test_unexpected_builder_file_rejects_whole_campaign(tmp_path: Path) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    declaration = write_claims_stage(claims)
    bad = declaration["maps"][4]["map"]

    def builder(declared, claims_dir, output_dir, _execution_root):
        return write_map_build(
            declared,
            claims_dir,
            output_dir,
            extra_file=declared["map"] == bad,
        )

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )

    assert report["passed"] is False
    assert report["pass_count"] == 27
    assert not analysis.exists()
    assert report["maps"][4]["error"] == (
        "unexpected builder output passing-subset.txt"
    )
    assert not (diagnostics / f"{bad}.build-summary.json").exists()


def test_second_run_refuses_overwrite_before_builder(tmp_path: Path) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    write_claims_stage(claims)
    calls: list[str] = []

    def builder(declared, claims_dir, output_dir, _execution_root):
        calls.append(declared["map"])
        return write_map_build(declared, claims_dir, output_dir)

    first = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )
    assert first["passed"] is True
    first_call_count = len(calls)

    with pytest.raises(campaign.GeneratedAtlasCampaignError, match="already exists"):
        campaign.build_atlas_campaign(
            _declaration_path(claims),
            claims,
            analysis,
            tmp_path / "second-diagnostics",
            tmp_path / "second-report.json",
            _builder=builder,
            _binding=binding(),
        )
    assert len(calls) == first_call_count


def test_refuses_nested_report_and_symlink_stage_roots(tmp_path: Path) -> None:
    claims = tmp_path / "claims"
    write_claims_stage(claims)
    analysis = tmp_path / "analysis"
    with pytest.raises(campaign.GeneratedAtlasCampaignError, match="outside"):
        campaign.build_atlas_campaign(
            _declaration_path(claims),
            claims,
            analysis,
            tmp_path / "diagnostics",
            analysis / "report.json",
            _builder=lambda *_args: pytest.fail("builder called"),
            _binding=binding(),
        )
    with pytest.raises(campaign.GeneratedAtlasCampaignError, match="outside"):
        campaign.build_atlas_campaign(
            _declaration_path(claims),
            claims,
            tmp_path / "report-root" / "analysis",
            tmp_path / "inverse-diagnostics",
            tmp_path / "report-root",
            _builder=lambda *_args: pytest.fail("builder called"),
            _binding=binding(),
        )

    symlink = tmp_path / "analysis-link"
    symlink.symlink_to(tmp_path / "absent-analysis")
    with pytest.raises(campaign.GeneratedAtlasCampaignError, match="symlink"):
        campaign.build_atlas_campaign(
            _declaration_path(claims),
            claims,
            symlink,
            tmp_path / "different-diagnostics",
            tmp_path / "different-report.json",
            _builder=lambda *_args: pytest.fail("builder called"),
            _binding=binding(),
        )


def test_exact_claims_membership_failure_calls_no_builder(tmp_path: Path) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    declaration = write_claims_stage(claims)
    missing = f"{declaration['maps'][0]['map']}.generator-claims.json"
    (claims / missing).unlink()
    (claims / "replacement.generator-claims.json").write_bytes(b"replacement")
    calls: list[str] = []

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=lambda declared, *_args: calls.append(declared["map"]),
        _binding=binding(),
    )

    assert report["passed"] is False
    assert calls == []
    assert not analysis.exists()
    assert not diagnostics.exists()
    assert f"claims-stage: missing file {missing}" in report["failures"]
    assert (
        "claims-stage: unexpected file replacement.generator-claims.json"
        in report["failures"]
    )


def test_linux_noreplace_refuses_existing_empty_directory_and_symlink(
    tmp_path: Path,
) -> None:
    for destination_kind in ("directory", "symlink"):
        source = tmp_path / f"source-{destination_kind}"
        destination = tmp_path / f"destination-{destination_kind}"
        source.mkdir()
        (source / "owned.txt").write_text("owned\n")
        if destination_kind == "directory":
            destination.mkdir()
        else:
            destination.symlink_to(tmp_path / "attacker-target")

        with pytest.raises(
            campaign.GeneratedAtlasCampaignError,
            match="refusing overwrite",
        ):
            campaign._rename_noreplace(source, destination)

        assert source.is_dir()
        if destination_kind == "directory":
            assert destination.is_dir()
        else:
            assert destination.is_symlink()


def test_claims_and_execution_are_immutable_committed_snapshots(
    tmp_path: Path,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    declaration = write_claims_stage(claims)
    first = declaration["maps"][0]["map"]
    original_member = claims / f"{first}.bsp"
    original_payload = original_member.read_bytes()
    observed: list[tuple[Path, Path]] = []

    def builder(declared, claims_snapshot, output_dir, execution_snapshot):
        if declared["ordinal"] == 0:
            observed.append((claims_snapshot, execution_snapshot))
            assert claims_snapshot != claims
            assert execution_snapshot != ROOT
            assert not (execution_snapshot / ".git").exists()
            assert (claims_snapshot / f"{first}.bsp").read_bytes() == original_payload
            committed_builder = subprocess.check_output(
                ["git", "show", "HEAD:tools/build_map_atlas.py"], cwd=ROOT
            )
            assert (
                execution_snapshot / "tools/build_map_atlas.py"
            ).read_bytes() == committed_builder
            assert claims_snapshot.stat().st_mode & 0o222 == 0
            assert execution_snapshot.stat().st_mode & 0o222 == 0
            original_member.write_bytes(b"external mutation after snapshot\n")
        assert (claims_snapshot / f"{first}.bsp").read_bytes() == original_payload
        return write_map_build(declared, claims_snapshot, output_dir)

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )

    assert observed
    assert report["passed"] is True
    assert report["claims_snapshot_sha256"] == report["input_claims"][
        "report_sha256"
    ]
    assert original_member.read_bytes() != original_payload


def test_rejects_noncanonical_repo_root_before_reserving_outputs(
    tmp_path: Path,
) -> None:
    claims = tmp_path / "claims"
    write_claims_stage(claims)
    report = tmp_path / "report.json"

    with pytest.raises(
        campaign.GeneratedAtlasCampaignError,
        match="repo_root must be",
    ):
        campaign.build_atlas_campaign(
            _declaration_path(claims),
            claims,
            tmp_path / "analysis",
            tmp_path / "diagnostics",
            report,
            repo_root=tmp_path,
            _builder=lambda *_args: pytest.fail("builder called"),
            _binding=binding(),
        )
    assert not report.exists()


@pytest.mark.parametrize(
    ("mutator", "tamper_atlas", "failure_fragment"),
    [
        (
            lambda value: value.__setitem__("canonical_map_id", "wrong-map"),
            False,
            "identity/status",
        ),
        (
            lambda value: value["bsp"].__setitem__("sha256", "0" * 64),
            False,
            "identity/status",
        ),
        (
            lambda value: value.__setitem__(
                "generator_claims_sha256", "1" * 64
            ),
            False,
            "identity/status",
        ),
        (
            lambda value: value["identity"].__setitem__(
                "analyzer_sha256", "2" * 64
            ),
            False,
            "identity/status",
        ),
        (
            lambda value: value.__setitem__("status", "candidate"),
            False,
            "identity/status",
        ),
        (
            lambda value: value["performance"]["full_cold_rebuild"].__setitem__(
                "elapsed_milliseconds", 0
            ),
            False,
            "full-cold proof contract",
        ),
        (
            lambda value: value["performance"]["full_cold_rebuild"][
                "artifact_sha256"
            ].__setitem__(".objectives.json", "3" * 64),
            False,
            "full-cold proof digest differs",
        ),
        (None, True, "identity/status"),
    ],
)
def test_manifest_authority_tampering_rejects_complete_campaign(
    tmp_path: Path,
    mutator: Callable[[dict], None] | None,
    tamper_atlas: bool,
    failure_fragment: str,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    declaration = write_claims_stage(claims)
    bad = declaration["maps"][0]["map"]

    def builder(declared, claims_snapshot, output_dir, _execution_snapshot):
        return write_map_build(
            declared,
            claims_snapshot,
            output_dir,
            manifest_mutator=mutator if declared["ordinal"] == 0 else None,
            tamper_atlas_after_manifest=(
                tamper_atlas and declared["ordinal"] == 0
            ),
        )

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )

    assert report["passed"] is False
    assert report["published"] is False
    assert not analysis.exists()
    assert failure_fragment in report["maps"][0]["error"]
    assert any(failure.startswith(f"{bad}:") for failure in report["failures"])


def test_noncanonical_analysis_manifest_is_rejected(tmp_path: Path) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    write_claims_stage(claims)

    def builder(declared, claims_snapshot, output_dir, _execution_snapshot):
        result = write_map_build(declared, claims_snapshot, output_dir)
        if declared["ordinal"] == 0:
            path = output_dir / f"{declared['map']}.analysis.manifest.json"
            value = json.loads(path.read_bytes())
            path.write_text(json.dumps(value, indent=2) + "\n")
            summary_path = output_dir / campaign.BUILD_SUMMARY_NAME
            summary = json.loads(summary_path.read_bytes())
            summary["maps"][0]["manifest_sha256"] = file_sha256(path)
            summary_path.write_bytes(canonical_bytes(summary))
            return campaign.BuildProcessResult(
                0,
                (json.dumps(summary, sort_keys=True) + "\n").encode(),
                result.stderr,
            )
        return result

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )

    assert report["passed"] is False
    assert "not canonical compact sorted JSON" in report["maps"][0]["error"]


def test_atlas_manifest_authority_is_independently_bound(tmp_path: Path) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    write_claims_stage(claims)

    def builder(declared, claims_snapshot, output_dir, _execution_snapshot):
        result = write_map_build(declared, claims_snapshot, output_dir)
        if declared["ordinal"] != 0:
            return result
        name = declared["map"]
        atlas_manifest_path = output_dir / f"{name}.atlas.manifest.json"
        atlas_manifest = json.loads(atlas_manifest_path.read_bytes())
        atlas_manifest["generator"]["sha256"] = "4" * 64
        atlas_manifest_path.write_bytes(
            json.dumps(
                atlas_manifest, ensure_ascii=True, separators=(",", ":")
            ).encode("ascii") + b"\n"
        )
        analysis_manifest_path = output_dir / f"{name}.analysis.manifest.json"
        analysis_manifest = json.loads(analysis_manifest_path.read_bytes())
        replacement_sha = file_sha256(atlas_manifest_path)
        analysis_manifest["identity"]["atlas_manifest_sha256"] = replacement_sha
        analysis_manifest["artifacts"]["atlas_manifest"][
            "sha256"
        ] = replacement_sha
        analysis_manifest_path.write_bytes(canonical_bytes(analysis_manifest))
        summary_path = output_dir / campaign.BUILD_SUMMARY_NAME
        summary = json.loads(summary_path.read_bytes())
        summary["maps"][0]["manifest_sha256"] = file_sha256(
            analysis_manifest_path
        )
        summary_path.write_bytes(canonical_bytes(summary))
        return campaign.BuildProcessResult(
            0,
            (json.dumps(summary, sort_keys=True) + "\n").encode(),
            result.stderr,
        )

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )

    assert report["passed"] is False
    assert "Atlas manifest differs" in report["maps"][0]["error"]


def test_row_artifacts_are_derived_from_and_compared_to_incoming_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    declaration = write_claims_stage(claims)
    first = declaration["maps"][0]["map"]
    original_copy = campaign._exclusive_copy

    def corrupting_copy(source: Path, destination: Path) -> None:
        original_copy(source, destination)
        if destination.name == f"{first}.objectives.json":
            destination.write_bytes(destination.read_bytes() + b"post-copy drift\n")

    monkeypatch.setattr(campaign, "_exclusive_copy", corrupting_copy)

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=lambda declared, snapshot, output, _execution: write_map_build(
            declared, snapshot, output
        ),
        _binding=binding(),
    )

    assert report["passed"] is False
    assert not analysis.exists()
    assert report["maps"][0]["error"] == (
        "per-map artifacts differ from incoming membership"
    )
    corrupted_route = (
        f"analysis:{first}:.objectives.json\n".encode() + b"post-copy drift\n"
    )
    assert report["maps"][0]["artifacts"][".objectives.json"] == {
        "bytes": len(corrupted_route),
        "sha256": hashlib.sha256(corrupted_route).hexdigest(),
    }


def test_nested_claims_entry_and_nested_builder_output_are_rejected(
    tmp_path: Path,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    write_claims_stage(claims)
    (claims / "nested").mkdir()
    calls: list[str] = []

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=lambda declared, *_args: calls.append(declared["map"]),
        _binding=binding(),
    )

    assert report["passed"] is False
    assert calls == []
    assert "unexpected entry nested" in report["failures"][0]

    second_root = tmp_path / "second"
    second_root.mkdir()
    second_claims = second_root / "claims"
    write_claims_stage(second_claims)

    def nested_builder(declared, snapshot, output, _execution):
        result = write_map_build(declared, snapshot, output)
        if declared["ordinal"] == 0:
            (output / "nested").mkdir()
        return result

    second = campaign.build_atlas_campaign(
        _declaration_path(second_claims),
        second_claims,
        second_root / "analysis",
        second_root / "diagnostics",
        second_root / "report.json",
        _builder=nested_builder,
        _binding=binding(),
    )
    assert second["passed"] is False
    assert "nested directory nested" in second["maps"][0]["error"]


@pytest.mark.parametrize("destination_kind", ["directory", "symlink"])
def test_publication_race_preserves_attacker_destination_and_fails_report(
    tmp_path: Path, destination_kind: str,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    declaration = write_claims_stage(claims)

    def builder(declared, snapshot, output, _execution):
        result = write_map_build(declared, snapshot, output)
        if declared["ordinal"] == 27:
            if destination_kind == "directory":
                analysis.mkdir()
            else:
                analysis.symlink_to(tmp_path / "attacker-target")
        return result

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )

    assert report["passed"] is False
    assert report["published"] is False
    assert any("atomic-publication" in failure for failure in report["failures"])
    if destination_kind == "directory":
        assert analysis.is_dir() and not any(analysis.iterdir())
    else:
        assert analysis.is_symlink()


def test_passing_report_is_durable_before_analysis_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    write_claims_stage(claims)
    real_rename = campaign._rename_noreplace
    observed: list[dict] = []

    def attested_rename(source: Path, destination: Path) -> None:
        if destination == analysis:
            attestation = json.loads(report_path.read_bytes())
            assert attestation["passed"] is True
            assert attestation["published"] is True
            assert attestation["pass_count"] == 28
            observed.append(attestation)
        real_rename(source, destination)

    monkeypatch.setattr(campaign, "_rename_noreplace", attested_rename)
    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=lambda declared, snapshot, output, _execution: write_map_build(
            declared, snapshot, output
        ),
        _binding=binding(),
    )

    assert report["passed"] is True
    assert len(observed) == 1


def test_failed_post_publish_verification_quarantines_owned_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    write_claims_stage(claims)
    real_open = campaign._open_exact_flat_root

    def fail_final(path: Path, expected: set[str], label: str):
        if path == analysis:
            raise campaign.GeneratedAtlasCampaignError("injected final audit failure")
        return real_open(path, expected, label)

    monkeypatch.setattr(campaign, "_open_exact_flat_root", fail_final)
    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=lambda declared, snapshot, output, _execution: write_map_build(
            declared, snapshot, output
        ),
        _binding=binding(),
    )

    assert report["passed"] is False
    assert report["published"] is False
    assert not analysis.exists()
    quarantines = list(tmp_path.glob(".analysis.quarantine-*"))
    assert len(quarantines) == 1
    assert len(list(quarantines[0].iterdir())) == 224
    assert json.loads(report_path.read_bytes())["passed"] is False


def test_report_path_symlink_swap_prevents_any_publication(
    tmp_path: Path,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    write_claims_stage(claims)
    moved_report = tmp_path / "reserved-report-moved"

    def builder(declared, snapshot, output, _execution):
        result = write_map_build(declared, snapshot, output)
        if declared["ordinal"] == 27:
            report_path.rename(moved_report)
            report_path.symlink_to(tmp_path / "attacker-report")
        return result

    with pytest.raises(
        campaign.GeneratedAtlasCampaignError,
        match="report path was replaced",
    ):
        campaign.build_atlas_campaign(
            _declaration_path(claims),
            claims,
            analysis,
            diagnostics,
            report_path,
            _builder=builder,
            _binding=binding(),
        )
    assert not analysis.exists()
    assert report_path.is_symlink()


def test_diagnostics_root_symlink_swap_fails_closed_with_report(
    tmp_path: Path,
) -> None:
    analysis, diagnostics, report_path, claims = paths(tmp_path)
    write_claims_stage(claims)
    moved_diagnostics = tmp_path / "diagnostics-moved"

    def builder(declared, snapshot, output, _execution):
        result = write_map_build(declared, snapshot, output)
        if declared["ordinal"] == 0:
            diagnostics.rename(moved_diagnostics)
            diagnostics.symlink_to(moved_diagnostics, target_is_directory=True)
        return result

    report = campaign.build_atlas_campaign(
        _declaration_path(claims),
        claims,
        analysis,
        diagnostics,
        report_path,
        _builder=builder,
        _binding=binding(),
    )

    assert report["passed"] is False
    assert report["published"] is False
    assert not analysis.exists()
    assert report["maps"][0]["passed"] is False
    assert any(
        "build-diagnostics" in failure for failure in report["failures"]
    )
    assert json.loads(report_path.read_bytes()) == report
