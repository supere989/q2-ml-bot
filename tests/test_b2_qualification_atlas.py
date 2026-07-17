from __future__ import annotations

from pathlib import Path
import sys

import pytest

from tests.b2_qualification_native_fixtures import (
    IMPLEMENTATION, declaration, sha, stage_report, write_json,
)
from tools.assemble_b2_qualification import _validate_stage_report
from tools.b2_qualification_stage_support import (
    QualificationStageError, file_record,
)
from tools.run_b2_qualification_atlas import _default_builder, build_qualification_atlas
from tools.run_generated_atlas_campaign import BuildProcessResult
from tools.run_generator_cohort import STAGE_SUFFIXES, canonical_bytes


def _inputs(tmp_path: Path) -> dict[str, Path]:
    declared = declaration()
    declaration_path = write_json(tmp_path / "declaration.json", declared)
    declaration_sha = sha(declaration_path.read_bytes())
    claims_root = tmp_path / "claims"
    claims_root.mkdir()
    material_sha = sha(b"materialization-report")
    report = stage_report("claims", declared, declaration_sha, material_sha)
    for item in report["maps"]:
        files = {}
        for suffix in STAGE_SUFFIXES["claims"]:
            path = claims_root / f"{item['map']}{suffix}"
            path.write_bytes(f"{item['map']}:{suffix}\n".encode())
            files[suffix] = file_record(path)
        item["criteria"] = {
            "prior-stage-passed": True, "immutable-claims": True,
            "claims-membership": True, "input-stability": True,
        }
        item["evidence_sha256"] = sha(canonical_bytes({
            "map": item["map"], "prior_stage_passed": True, "files": files,
        }))
    claims_report = write_json(tmp_path / "claims-report.json", report)
    return {
        "declaration": declaration_path, "claims_root": claims_root,
        "claims_report": claims_report, "analysis": tmp_path / "analysis",
        "evidence": tmp_path / "atlas-evidence", "report": tmp_path / "atlas-report.json",
    }


def _snapshot(_repo: Path, destination: Path, _binding: object) -> None:
    destination.mkdir()


def _builder(declared, _claims, output, _execution):
    for suffix in STAGE_SUFFIXES["analysis"]:
        (output / f"{declared['map']}{suffix}").write_bytes(
            f"{declared['map']}:{suffix}\n".encode()
        )
    return BuildProcessResult(0, b'{"fixture":true}\n', b"")


def _inspector(output, declared, _claims, result, _binding):
    artifacts = {
        suffix: file_record(output / f"{declared['map']}{suffix}")
        for suffix in STAGE_SUFFIXES["analysis"]
    }
    return ({"map": declared["map"]}, result.stdout, artifacts, {
        "status": "passed", "deterministic_rebuild": True,
        "full_cold_proof_sha256": sha(declared["map"].encode()),
    })


def _run(paths: dict[str, Path]):
    return build_qualification_atlas(
        declaration_path=paths["declaration"],
        claims_report_path=paths["claims_report"], claims_root=paths["claims_root"],
        analysis_root=paths["analysis"], evidence_root=paths["evidence"],
        report_path=paths["report"], repo_root=Path("/fixture"), jobs=4,
        implementation_provider=lambda _root: IMPLEMENTATION,
        snapshotter=_snapshot, builder=_builder, inspector=_inspector,
    )


def test_atlas_runs_all_eligible_maps_and_emits_replayable_standard_stage(tmp_path: Path):
    paths = _inputs(tmp_path)
    report = _run(paths)
    assert report["pass_count"] == 28
    assert len(list(paths["analysis"].iterdir())) == 28 * 8
    assert len(list(paths["evidence"].iterdir())) == 28 * 3
    for row in report["maps"]:
        evidence = paths["evidence"] / f"{row['ordinal']:03d}-{row['map']}.json"
        assert sha(evidence.read_bytes()) == row["evidence_sha256"]
    declared = declaration()
    declaration_sha = sha(paths["declaration"].read_bytes())
    claims_sha = sha(paths["claims_report"].read_bytes())
    _validate_stage_report(
        paths["report"], "atlas-build", declared, declaration_sha,
        IMPLEMENTATION, claims_sha,
        {"cohorts": set(), "declarations": set(), "maps": set(),
         "seeds": set(), "digests": set()},
    )


def test_atlas_rejects_forged_claims_row_before_invocation(tmp_path: Path):
    paths = _inputs(tmp_path)
    claims = __import__("json").loads(paths["claims_report"].read_bytes())
    claims["maps"][7]["evidence_sha256"] = "ab" * 32
    paths["claims_report"].write_bytes(canonical_bytes(claims))
    with pytest.raises(QualificationStageError, match="claims evidence does not match bytes"):
        _run(paths)
    assert not paths["analysis"].exists()
    assert not paths["evidence"].exists()
    assert not paths["report"].exists()


def test_real_builder_wrapper_kills_timed_out_process_group(tmp_path: Path):
    execution = tmp_path / "execution"
    tools = execution / "tools"
    tools.mkdir(parents=True)
    (tools / "build_map_atlas.py").write_text(
        "import time\ntime.sleep(60)\n", encoding="ascii"
    )
    claims = tmp_path / "claims"; claims.mkdir()
    output = tmp_path / "output"; output.mkdir()
    builder = _default_builder(
        client_root=tmp_path, lithium_root=tmp_path,
        hook_attestation=tmp_path / "hook.json", fall_oracle=None,
        packer=None, verifier=None, timeout_seconds=0.05,
        python_executable=Path(sys.executable),
    )
    result = builder({"map": "b2q26_timeout"}, claims, output, execution)
    assert result.returncode < 0
    assert b"qualification timeout" in result.stderr
