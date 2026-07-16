from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/run_b2_compiled_boundary_qualification.py"
SPEC = importlib.util.spec_from_file_location("compiled_boundary_qualification", TOOL_PATH)
assert SPEC and SPEC.loader
qualification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualification)


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _fake_toolchain(tmp_path: Path, *, malformed_cm: bool = False) -> tuple[Path, Path, Path]:
    q2tool = tmp_path / "q2tool"
    _write_executable(
        q2tool,
        """#!/usr/bin/env python3
import pathlib, struct, sys
source = pathlib.Path(sys.argv[-1])
source.with_suffix('.bsp').write_bytes(b'IBSP' + struct.pack('<i', 38) + source.stem.encode())
""",
    )
    cm = tmp_path / "q2-cm-oracle"
    if malformed_cm:
        cm_source = """#!/usr/bin/env python3
print('{not-json')
"""
    else:
        cm_source = """#!/usr/bin/env python3
import hashlib, json, pathlib, sys
bsp = pathlib.Path(sys.argv[sys.argv.index('--map') + 1])
ceiling = int(bsp.stem.rsplit('_', 1)[-1])
digest = hashlib.sha256(bsp.read_bytes()).hexdigest()
for line in sys.stdin:
    request = json.loads(line)
    common = {'ok': True, 'id': request['id'], 'op': request['op'],
              'schema': 'q2-cm-oracle-v1', 'tool_identity': '1' * 64,
              'physics_identity': '2' * 64, 'map_sha256': digest,
              'map_checksum': 7}
    if request['op'] == 'identity':
        common.update({'model0': {'mins': [-145, -145, -17],
                                  'maxs': [145, 145, ceiling + 17], 'headnode': 0}})
    else:
        top = request['end'][2] + request['maxs'][2]
        clear = top < ceiling
        common.update({'fraction': 1 if clear else 0.5,
                       'startsolid': False, 'allsolid': False,
                       'endpos': request['end']})
    print(json.dumps(common, sort_keys=True, separators=(',', ':')))
"""
    _write_executable(cm, cm_source)
    basedir = tmp_path / "assets"
    (basedir / "baseq2").mkdir(parents=True)
    (basedir / "baseq2/pak0.pak").write_bytes(b"test-only-pak")
    return q2tool, cm, basedir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_phase_fans_out_and_proves_real_boundary_contract(tmp_path: Path) -> None:
    q2tool, cm, basedir = _fake_toolchain(tmp_path)
    work_root = tmp_path / "work"
    report_path = tmp_path / "proof.json"

    result = qualification.main([
        "--phase", "all", "--q2tool", str(q2tool), "--basedir", str(basedir),
        "--q2tool-sha256", _sha(q2tool), "--cm-oracle", str(cm),
        "--cm-oracle-sha256", _sha(cm), "--work-root", str(work_root),
        "--report", str(report_path), "--compile-workers", "3",
    ])

    assert result == 0
    raw = report_path.read_bytes()
    report = json.loads(raw)
    assert raw == qualification.canonical_bytes(report)
    assert report["schema"] == qualification.PROOF_SCHEMA
    assert report["status"] == "passed-non-admissible-qualification"
    assert report["admission"] == qualification.qualification_only()
    assert [row["column_clearance_milliunits"] for row in report["proofs"]] == [
        92_000, 92_000, 96_000,
    ]
    assert [row["column_clear_96"] for row in report["proofs"]] == [
        False, False, True,
    ]
    compile_report = json.loads((work_root / "compile-report.json").read_bytes())
    assert compile_report["parallel_compile_workers"] == 3
    assert {row["returncode"] for row in compile_report["compiled"]} == {0}
    assert compile_report["q2tool"]["sha256"] == hashlib.sha256(
        q2tool.read_bytes()
    ).hexdigest()


def test_split_compile_and_prove_rebinds_copied_bsps(tmp_path: Path) -> None:
    q2tool, cm, basedir = _fake_toolchain(tmp_path)
    remote_root = tmp_path / "remote"
    compile_report = tmp_path / "compile.json"
    assert qualification.main([
        "--phase", "compile", "--q2tool", str(q2tool), "--basedir", str(basedir),
        "--q2tool-sha256", _sha(q2tool),
        "--work-root", str(remote_root), "--report", str(compile_report),
    ]) == 0
    copied = tmp_path / "returned-bsps"
    copied.mkdir()
    for bsp in (remote_root / "compiled").glob("*.bsp"):
        (copied / bsp.name).write_bytes(bsp.read_bytes())
    proof = tmp_path / "local-proof.json"

    assert qualification.main([
        "--phase", "prove", "--compile-report", str(compile_report),
        "--compiled-dir", str(copied), "--cm-oracle", str(cm),
        "--q2tool-sha256", _sha(q2tool),
        "--cm-oracle-sha256", _sha(cm),
        "--report", str(proof),
    ]) == 0
    report = json.loads(proof.read_bytes())
    assert report["passed"] is True
    assert report["compile_evidence"]["sha256"] == hashlib.sha256(
        compile_report.read_bytes()
    ).hexdigest()


def test_malformed_cm_output_fails_closed_with_canonical_evidence(tmp_path: Path) -> None:
    q2tool, cm, basedir = _fake_toolchain(tmp_path, malformed_cm=True)
    report_path = tmp_path / "failure.json"

    result = qualification.main([
        "--phase", "all", "--q2tool", str(q2tool), "--basedir", str(basedir),
        "--q2tool-sha256", _sha(q2tool), "--cm-oracle", str(cm),
        "--cm-oracle-sha256", _sha(cm), "--work-root", str(tmp_path / "work"),
        "--report", str(report_path),
    ])

    assert result == 1
    raw = report_path.read_bytes()
    report = json.loads(raw)
    assert raw == qualification.canonical_bytes(report)
    assert report["passed"] is False
    assert report["status"] == "failed-non-admissible-qualification"
    assert report["admission"]["qualification_only"] is True
    assert "invalid JSON" in report["failure"]["message"]


def test_compile_report_must_be_canonical(tmp_path: Path) -> None:
    _q2tool, cm, _basedir = _fake_toolchain(tmp_path)
    compile_report = tmp_path / "compile.json"
    compile_report.write_text(json.dumps({"schema": qualification.COMPILE_SCHEMA}),
                              encoding="utf-8")
    report_path = tmp_path / "failure.json"

    result = qualification.main([
        "--phase", "prove", "--compile-report", str(compile_report),
        "--compiled-dir", str(tmp_path), "--cm-oracle", str(cm),
        "--q2tool-sha256", _sha(_q2tool),
        "--cm-oracle-sha256", _sha(cm),
        "--report", str(report_path),
    ])

    assert result == 1
    report = json.loads(report_path.read_bytes())
    assert "not canonical JSON" in report["failure"]["message"]
