from __future__ import annotations

from pathlib import Path
import json

import pytest

from harness.atlas_analyzer import _rust_struct_json
from tests.b2_qualification_native_fixtures import (
    IMPLEMENTATION, declaration, sha, stage_report, write_json,
)
from tools.assemble_b2_qualification import STAGES, _validate_infrastructure
from tools.b2_qualification_stage_support import QualificationStageError
from tools.run_b2_qualification_infrastructure import (
    PINNED_PYTHON_SHA256, _timeout_probe as real_timeout_probe,
    _membership_evidence, _syntax_evidence, produce_infrastructure_report,
)
from tools.run_generator_cohort import STAGE_SUFFIXES, canonical_bytes


def _inputs(tmp_path: Path, *, sparse: bool = False) -> dict[str, object]:
    declared = declaration()
    declaration_path = write_json(tmp_path / "declaration.json", declared)
    declaration_sha = sha(declaration_path.read_bytes())
    report_paths = {}
    report_hashes = {}
    previous = None
    sparse_counts = {
        "source": 28, "compile": 26, "compiled-cm-preflight": 25,
        "materialization": 24, "claims": 23, "atlas-build": 20,
        "generated-promotion": 20,
    }
    for stage in STAGES:
        count = (
            sparse_counts[stage]
            if sparse else
            (20 if stage in ("atlas-build", "generated-promotion") else 28)
        )
        criteria = None
        if stage == "source":
            criteria = {"source-static": True, "deterministic-cold-rebuild": True,
                        "cold-bytes-identical": True}
        elif stage == "atlas-build":
            criteria = {"full-atlas": True, "deterministic-cold-rebuild": True}
        report = stage_report(
            stage, declared, declaration_sha, previous, pass_count=count, criteria=criteria
        )
        path = write_json(tmp_path / f"{stage}-report.json", report)
        previous = sha(path.read_bytes())
        report_paths[stage] = path
        report_hashes[stage] = previous

    roots = {}
    for stage, suffix_key in (
        ("source", "source"), ("compile", "compiled"),
        ("materialization", "materialized"), ("claims", "claims"),
    ):
        root = tmp_path / f"{stage}-root"
        root.mkdir()
        for ordinal, row in enumerate(declared["maps"]):
            suffixes = STAGE_SUFFIXES[suffix_key]
            if sparse and stage == "compile":
                suffixes = (
                    STAGE_SUFFIXES["compiled"]
                    if ordinal < sparse_counts["compile"] else
                    STAGE_SUFFIXES["source"]
                )
            elif sparse and stage in ("materialization", "claims"):
                if ordinal >= sparse_counts[stage]:
                    continue
            for suffix in suffixes:
                (root / f"{row['map']}{suffix}").write_bytes(
                    f"{stage}:{row['map']}:{suffix}\n".encode()
                )
        roots[stage] = root
    analysis = tmp_path / "analysis-root"
    analysis.mkdir()
    for row in declared["maps"][:20]:
        name = row["map"]
        proof = {
            "artifact_sha256": {".atlas.bin": sha(f"artifact:{name}".encode())},
            "cold_artifact_sha256": {".atlas.bin": sha(f"artifact:{name}".encode())},
            "artifact_semantic_sha256": {
                ".analysis.manifest.json": sha(f"semantic:{name}".encode())},
            "cold_artifact_semantic_sha256": {
                ".analysis.manifest.json": sha(f"semantic:{name}".encode())},
            "elapsed_milliseconds": 10, "timeout_limit_milliseconds": 300_000,
            "sampled_process_tree_peak_rss_bytes": 1024,
            "peak_rss_limit_bytes": 512 * 1024 * 1024,
        }
        for suffix in STAGE_SUFFIXES["analysis"]:
            path = analysis / f"{name}{suffix}"
            if suffix not in (
                ".analysis.manifest.json", ".atlas.manifest.json",
            ):
                path.write_bytes(f"analysis:{name}:{suffix}\n".encode())
        atlas_manifest_path = analysis / f"{name}.atlas.manifest.json"
        atlas_manifest = {
            "schema_version": 1,
            "byte_order": "little",
            "atlas_magic": "Q2ATL001",
            "grid": {
                "origin": [-512, -512, -512],
                "model0_mins": [-321, -321, -321],
            },
        }
        writer_bytes = _rust_struct_json(atlas_manifest) + b"\n"
        assert writer_bytes != canonical_bytes(atlas_manifest)
        atlas_manifest_path.write_bytes(writer_bytes)
        atlas_manifest_sha = sha(atlas_manifest_path.read_bytes())
        (analysis / f"{name}.analysis.manifest.json").write_bytes(
            canonical_bytes({
                "artifacts": {
                    "atlas_manifest": {"sha256": atlas_manifest_sha},
                },
                "canonical_map_id": name,
                "identity": {
                    "atlas_manifest_sha256": atlas_manifest_sha,
                },
                "performance": {"full_cold_rebuild": proof},
            })
        )
    roots["atlas-build"] = analysis
    promotion = tmp_path / "promotion-evidence"
    promotion.mkdir()
    promotion_report = json.loads(
        report_paths["generated-promotion"].read_bytes()
    )
    for row in declared["maps"]:
        ordinal = row["ordinal"]
        name = row["map"]
        stage_row = promotion_report["maps"][ordinal]
        eligible = stage_row["passed"] is True
        evidence = {
            "schema": "q2-b2-qualification-promotion-map-evidence-v1",
            "ordinal": ordinal,
            "map": name,
            "atlas_evidence_sha256": sha(f"atlas:{name}".encode()),
            "runtime": {"fixture": True},
            "eligible": eligible,
            "validation": {} if eligible else None,
            "atlas_manifest_sha256": (
                sha((analysis / f"{name}.atlas.manifest.json").read_bytes())
                if eligible else None
            ),
            "analysis_manifest_sha256": (
                sha((analysis / f"{name}.analysis.manifest.json").read_bytes())
                if eligible else None
            ),
            "criteria": stage_row["criteria"],
            "failures": stage_row["failures"],
            "passed": stage_row["passed"],
        }
        evidence_path = write_json(
            promotion / f"{ordinal:03d}-{name}.json", evidence
        )
        stage_row["evidence_sha256"] = sha(evidence_path.read_bytes())
    write_json(report_paths["generated-promotion"], promotion_report)
    report_hashes["generated-promotion"] = sha(
        report_paths["generated-promotion"].read_bytes()
    )
    roots["generated-promotion"] = promotion
    syntax = write_json(tmp_path / "syntax.json", {
        "enumeration": "git-tracked", "failures": [], "file_count": 17,
        "files_sha256": sha(b"files"),
        "interpreter": {"executable": "/usr/bin/python3.10",
                        "implementation": "cpython",
                        "sha256": "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86",
                        "version": [3, 10, 12]},
        "passed": True, "schema": "q2-python-syntax-floor-v1",
    })
    return {
        "declaration": declaration_path, "reports": report_paths,
        "report_hashes": report_hashes, "roots": roots, "syntax": syntax,
        "evidence": tmp_path / "infrastructure-evidence",
        "report": tmp_path / "infrastructure-report.json",
    }


def _timeout_probe():
    return {
        "timed_out": True, "process_group_killed": True, "exit_code": -9,
        "timeout_milliseconds": 50, "probe": "injected-real-process-fixture",
    }


def _run(paths):
    roots = paths["roots"]
    return produce_infrastructure_report(
        declaration_path=paths["declaration"], stage_report_paths=paths["reports"],
        source_root=roots["source"], compiled_root=roots["compile"],
        materialized_root=roots["materialization"], claims_root=roots["claims"],
        analysis_root=roots["atlas-build"],
        promotion_evidence_root=roots["generated-promotion"],
        syntax_report_path=paths["syntax"], evidence_root=paths["evidence"],
        report_path=paths["report"], repo_root=Path("/fixture"),
        implementation_provider=lambda _root: IMPLEMENTATION,
        timeout_probe=_timeout_probe,
        runtime_provider=lambda: {
            "sha256": PINNED_PYTHON_SHA256, "version": [3, 11, 4],
            "implementation": "cpython", "executable": "/fixture/python",
            "bytes": 1,
        },
    )


def test_infrastructure_retains_seven_actual_evidence_documents(tmp_path: Path):
    paths = _inputs(tmp_path)
    report = _run(paths)
    assert report["pass_count"] == 7
    assert sorted(path.stem for path in paths["evidence"].iterdir()) == sorted(
        check["id"] for check in report["checks"]
    )
    binding = json.loads(
        (
            paths["evidence"]
            / "dyn-phase-b-atlas-manifest-binding.json"
        ).read_bytes()
    )["evidence"]
    assert binding["pass_count"] == 20
    assert binding["contract"] == (
        "production-phase-b-compact-atlas-manifest-and-promotion-seal"
    )


def test_infrastructure_rejects_manifest_digest_rewritten_in_sealed_promotion(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    first = declaration()["maps"][0]
    evidence_path = (
        paths["roots"]["generated-promotion"]
        / f"{first['ordinal']:03d}-{first['map']}.json"
    )
    evidence = json.loads(evidence_path.read_bytes())
    evidence["atlas_manifest_sha256"] = "00" * 32
    evidence_path.write_bytes(canonical_bytes(evidence))
    report_path = paths["reports"]["generated-promotion"]
    report = json.loads(report_path.read_bytes())
    report["maps"][0]["evidence_sha256"] = sha(evidence_path.read_bytes())
    report_path.write_bytes(canonical_bytes(report))

    with pytest.raises(
        QualificationStageError,
        match="promotion manifest digests differ",
    ):
        _run(paths)


def test_infrastructure_accepts_exact_sparse_stage_membership(tmp_path: Path):
    paths = _inputs(tmp_path, sparse=True)
    report = _run(paths)
    assert report["pass_count"] == 7
    exact = next(
        check for check in report["checks"]
        if check["id"] == "exact-stage-membership"
    )
    assert exact["passed"] is True
    for check in report["checks"]:
        raw = (paths["evidence"] / f"{check['id']}.json").read_bytes()
        assert sha(raw) == check["evidence_sha256"]
    _validate_infrastructure(
        paths["report"], declaration(), sha(paths["declaration"].read_bytes()),
        IMPLEMENTATION, paths["report_hashes"],
        {"cohorts": set(), "declarations": set(), "maps": set(),
         "seeds": set(), "digests": set()},
    )


def test_infrastructure_rejects_ambient_or_forged_syntax_runtime(tmp_path: Path):
    paths = _inputs(tmp_path)
    syntax = __import__("json").loads(paths["syntax"].read_bytes())
    syntax["interpreter"]["version"] = [3, 14, 0]
    paths["syntax"].write_bytes(canonical_bytes(syntax))
    with pytest.raises(QualificationStageError, match="WSL CPython 3.10.12"):
        _run(paths)
    assert not paths["evidence"].exists()


def test_syntax_floor_and_execution_runtime_are_distinct_authorities(tmp_path: Path):
    paths = _inputs(tmp_path)
    evidence = _syntax_evidence(paths["syntax"], {
        "sha256": PINNED_PYTHON_SHA256, "version": [3, 11, 4],
        "implementation": "cpython", "executable": "/fixture/python3.11",
        "bytes": 1,
    })
    assert evidence["syntax_interpreter"]["version"] == [3, 10, 12]
    assert evidence["execution_runtime"]["version"] == [3, 11, 4]


def test_syntax_evidence_rejects_wrong_execution_runtime(tmp_path: Path):
    paths = _inputs(tmp_path)
    with pytest.raises(QualificationStageError, match="pinned CPython 3.11.4"):
        _syntax_evidence(paths["syntax"], {
            "sha256": "00" * 32, "version": [3, 11, 4],
            "implementation": "cpython", "executable": "/fixture/python",
            "bytes": 1,
        })


def test_infrastructure_rejects_forged_stage_hash_chain(tmp_path: Path):
    paths = _inputs(tmp_path)
    atlas_path = paths["reports"]["atlas-build"]
    atlas = __import__("json").loads(atlas_path.read_bytes())
    atlas["input_report_sha256"] = "ab" * 32
    atlas_path.write_bytes(canonical_bytes(atlas))
    with pytest.raises(QualificationStageError, match="input report hash differs"):
        _run(paths)
    assert not paths["evidence"].exists()


def test_timeout_probe_actually_kills_the_process_group():
    result = real_timeout_probe()
    assert result["timed_out"] is True
    assert result["process_group_killed"] is True
    assert result["exit_code"] == -9


def test_exact_membership_accepts_honest_sparse_twenty_map_lifecycle(tmp_path: Path):
    paths = _inputs(tmp_path)
    declared = declaration()
    reports = {
        stage: json.loads(path.read_text())
        for stage, path in paths["reports"].items()
    }
    rejected = {row["map"] for row in declared["maps"][20:]}
    for stage in ("compile", "materialization", "claims"):
        for row in reports[stage]["maps"]:
            if row["map"] not in rejected:
                continue
            first = next(iter(row["criteria"]))
            row["criteria"][first] = False
            row["failures"] = ["honest sparse rejection"]
            row["passed"] = False
        reports[stage]["pass_count"] = 20
    roots = paths["roots"]
    for map_id in rejected:
        (roots["compile"] / f"{map_id}.bsp").unlink()
        for suffix in STAGE_SUFFIXES["materialized"]:
            (roots["materialization"] / f"{map_id}{suffix}").unlink()
        for suffix in STAGE_SUFFIXES["claims"]:
            (roots["claims"] / f"{map_id}{suffix}").unlink()
    evidence = _membership_evidence(declared, reports, roots)
    exact = evidence["exact_roots"]
    assert len(exact["source"]) == 28 * len(STAGE_SUFFIXES["source"])
    assert len(exact["compile"]) == 28 * len(STAGE_SUFFIXES["source"]) + 20
    assert len(exact["materialization"]) == 20 * len(STAGE_SUFFIXES["materialized"])
    assert len(exact["claims"]) == 20 * len(STAGE_SUFFIXES["claims"])
