from __future__ import annotations

from pathlib import Path

import pytest

import tools.run_b2_qualification_promotion as promotion
from tests.b2_qualification_native_fixtures import (
    IMPLEMENTATION, declaration, sha, stage_report, write_json,
)
from tools.b2_qualification_stage_support import QualificationStageError, file_record
from tools.run_generator_cohort import STAGE_SUFFIXES, canonical_bytes


def _inputs(tmp_path: Path) -> dict[str, Path]:
    declared = declaration()
    declaration_path = write_json(tmp_path / "declaration.json", declared)
    declaration_sha = sha(declaration_path.read_bytes())
    claims = tmp_path / "claims"
    analysis = tmp_path / "analysis"
    atlas_evidence = tmp_path / "atlas-evidence"
    claims.mkdir(); analysis.mkdir(); atlas_evidence.mkdir()
    atlas = stage_report("atlas-build", declared, declaration_sha, sha(b"claims"))
    for row in atlas["maps"]:
        for suffix in STAGE_SUFFIXES["claims"]:
            (claims / f"{row['map']}{suffix}").write_bytes(
                f"{row['map']}:{suffix}\n".encode()
            )
        artifacts = {}
        for suffix in STAGE_SUFFIXES["analysis"]:
            path = analysis / f"{row['map']}{suffix}"
            path.write_bytes(f"{row['map']}:{suffix}\n".encode())
            artifacts[suffix] = file_record(path)
        stem = f"{row['ordinal']:03d}-{row['map']}"
        stdout = atlas_evidence / f"{stem}.stdout.log"
        stderr = atlas_evidence / f"{stem}.stderr.log"
        stdout.write_bytes(b"atlas stdout\n"); stderr.write_bytes(b"")
        criteria = {
            "prior-claims-passed": True, "real-atlas-build": True,
            "full-atlas": True, "deterministic-cold-rebuild": True,
            "exact-analysis-membership": True,
        }
        evidence = {
            "schema": "q2-b2-qualification-atlas-map-evidence-v1",
            "ordinal": row["ordinal"], "map": row["map"],
            "claims_evidence_sha256": sha(f"claims:{row['map']}".encode()),
            "invoked": True, "exit_code": 0,
            "stdout": file_record(stdout), "stderr": file_record(stderr),
            "artifacts": artifacts,
            "analysis_binding": {"deterministic_rebuild": True},
            "build_summary": {"map": row["map"]}, "criteria": criteria,
            "failures": [], "passed": True,
        }
        evidence_path = atlas_evidence / f"{stem}.json"
        evidence_path.write_bytes(canonical_bytes(evidence))
        row["criteria"] = criteria
        row["evidence_sha256"] = sha(evidence_path.read_bytes())
    atlas_report = write_json(tmp_path / "atlas-report.json", atlas)
    repo = tmp_path / "repo"
    gate = repo / "docs/multires/B1-GATE.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(b"fixture-current-b1\n")
    return {
        "declaration": declaration_path, "claims": claims, "analysis": analysis,
        "atlas_evidence": atlas_evidence, "atlas_report": atlas_report,
        "repo": repo, "b1": gate, "evidence": tmp_path / "promotion-evidence",
        "report": tmp_path / "promotion-report.json",
    }


def test_promotion_calls_independent_validator_for_every_eligible_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    paths = _inputs(tmp_path)
    calls: list[str] = []

    def validator(map_path, analysis_path, *, b1_gate_path):
        calls.append(map_path.stem)
        return {
            "passed": True, "failures": [],
            "identities": {
                "analysis_sha256": file_record(analysis_path)["sha256"],
                "bsp_sha256": file_record(map_path.with_suffix(".bsp"))["sha256"],
            },
        }

    monkeypatch.setattr(promotion, "validate_promotion_report", lambda report: None)
    report = promotion.run_qualification_promotion(
        declaration_path=paths["declaration"], atlas_report_path=paths["atlas_report"],
        claims_root=paths["claims"], analysis_root=paths["analysis"],
        atlas_evidence_root=paths["atlas_evidence"], b1_gate_path=paths["b1"],
        evidence_root=paths["evidence"], report_path=paths["report"],
        repo_root=paths["repo"], jobs=4,
        implementation_provider=lambda _root: IMPLEMENTATION, validator=validator,
        runtime_provider=lambda: {"fixture": True},
    )
    assert report["pass_count"] == 28
    assert len(calls) == 28 and len(set(calls)) == 28
    assert len(list(paths["evidence"].iterdir())) == 28
    for row in report["maps"]:
        raw = (paths["evidence"] / f"{row['ordinal']:03d}-{row['map']}.json").read_bytes()
        assert sha(raw) == row["evidence_sha256"]


def test_promotion_rejects_artifact_changed_after_atlas_evidence(tmp_path: Path):
    paths = _inputs(tmp_path)
    first = declaration()["maps"][0]["map"]
    (paths["analysis"] / f"{first}.atlas.bin").write_bytes(b"forged\n")
    with pytest.raises(QualificationStageError, match="artifact bytes differ"):
        promotion.run_qualification_promotion(
            declaration_path=paths["declaration"],
            atlas_report_path=paths["atlas_report"], claims_root=paths["claims"],
            analysis_root=paths["analysis"], atlas_evidence_root=paths["atlas_evidence"],
            b1_gate_path=paths["b1"], evidence_root=paths["evidence"],
            report_path=paths["report"], repo_root=paths["repo"],
            implementation_provider=lambda _root: IMPLEMENTATION,
            runtime_provider=lambda: {"fixture": True},
        )
    assert not paths["evidence"].exists()
