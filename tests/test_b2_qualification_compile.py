from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import time
from dataclasses import replace

import pytest

from harness.ibsp38 import HEADER_SIZE, LUMP_NAMES
from tools.b2_qualification_toolchain import (
    ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
    load_toolchain_authority,
)
from tools.assemble_b2_qualification import (
    DECLARATION_SCHEMA,
    STAGE_SCHEMA,
    _validate_stage_report,
)
from tools.run_b2_qualification_compile import (
    CONCRETE_STYLES,
    SOURCE_SUFFIXES,
    QualificationCompileError,
    _file_sha256,
    _source_evidence_sha256,
    compile_qualification,
)
from tools.run_generator_cohort import canonical_bytes


IMPLEMENTATION = {
    "repository_commit": "12" * 20,
    "repository_tree": "34" * 20,
    "git_clean": True,
    "atlas_analyzer_authority_sha256": "56" * 32,
    "atlas_analyzer_authority_file_count": 1,
    "generator_sha256": "78" * 32,
    "routes_sha256": "9a" * 32,
}


def _sha256(payload: bytes) -> str:
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def _bsp() -> bytes:
    lumps = {name: b"" for name in LUMP_NAMES}
    lumps["entities"] = (
        b'{\n"classname" "worldspawn"\n}\n'
        b'{\n"classname" "info_player_deathmatch"\n"origin" "0 0 24"\n}\n\0'
    )
    lumps["planes"] = struct.pack("<4fi", 0.0, 0.0, 1.0, 0.0, 2)
    lumps["vertices"] = b"".join((
        struct.pack("<3f", 0.0, 0.0, 0.0),
        struct.pack("<3f", 64.0, 0.0, 0.0),
        struct.pack("<3f", 0.0, 64.0, 0.0),
    ))
    lumps["visibility"] = struct.pack("<i2i2B", 1, 12, 13, 1, 1)
    lumps["nodes"] = struct.pack(
        "<3i6h2H", 0, -1, -1, -64, -64, -64, 64, 64, 64, 0, 1
    )
    lumps["texinfo"] = struct.pack(
        "<8fii32si", *(0.0,) * 8, 0, 0, b"fixture/stone".ljust(32, b"\0"), -1
    )
    lumps["faces"] = struct.pack(
        "<Hhihh4Bi", 0, 0, 0, 3, 0, 0, 255, 255, 255, 0
    )
    lumps["lighting"] = b"\x80\x80\x80"
    lumps["leafs"] = struct.pack(
        "<ihh6h4H", 0, 0, 0, -64, -64, -64, 64, 64, 64, 0, 1, 0, 0
    )
    lumps["leaffaces"] = struct.pack("<H", 0)
    lumps["edges"] = struct.pack("<6H", 0, 1, 1, 2, 2, 0)
    lumps["surfedges"] = struct.pack("<3i", 0, 1, 2)
    lumps["models"] = struct.pack(
        "<9f3i", -64.0, -64.0, -24.0, 64.0, 64.0, 64.0,
        0.0, 0.0, 0.0, 0, 0, 1,
    )
    lumps["areas"] = struct.pack("<2i", 0, 0)
    header = bytearray(struct.pack("<4si", b"IBSP", 38))
    body = bytearray()
    cursor = HEADER_SIZE
    for name in LUMP_NAMES:
        payload = lumps[name]
        header.extend(struct.pack("<2i", cursor if payload else 0, len(payload)))
        body.extend(payload)
        cursor += len(payload)
    return bytes(header + body)


def _write_pak(path: Path) -> None:
    payload = b"colormap"
    name = b"PICS/COLORMAP.PCX"
    directory = struct.pack(
        "<56sii", name + b"\0" * (56 - len(name)), 12, len(payload)
    )
    path.write_bytes(
        struct.pack("<4sii", b"PACK", 12 + len(payload), len(directory))
        + payload + directory
    )


def _write_q2tool(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import sys
import time

expected = ['-bsp','-vis','-fast','-rad','-bounce','0','-threads','1','-basedir']
if sys.argv[1:10] != expected:
    raise SystemExit(90)
map_path = Path(sys.argv[11])
event_path = Path(os.environ['Q2_FAKE_EVENTS'])
with event_path.open('a', encoding='ascii') as stream:
    stream.write(json.dumps({'event':'start','map':map_path.stem,'time':time.time()}) + '\\n')
if map_path.stem == os.environ.get('Q2_FAKE_SLOW_MAP'):
    time.sleep(float(os.environ.get('Q2_FAKE_SLOW_SECONDS', '2')))
else:
    time.sleep(float(os.environ.get('Q2_FAKE_DELAY', '0.01')))
fail_maps = set(filter(None, os.environ.get('Q2_FAKE_FAIL_MAPS', '').split(',')))
if map_path.stem == os.environ.get('Q2_FAKE_FAIL_MAP') or map_path.stem in fail_maps:
    raise SystemExit(23)
shutil.copyfile(os.environ['Q2_FAKE_BSP'], map_path.with_suffix('.bsp'))
if map_path.stem == os.environ.get('Q2_FAKE_EXTRA_MAP'):
    map_path.with_suffix('.lin').write_text('extra', encoding='ascii')
if map_path.stem == os.environ.get('Q2_FAKE_MUTATE_MAP'):
    with open(os.environ['Q2_FAKE_MUTATE_PATH'], 'ab') as stream:
        stream.write(b'changed')
with event_path.open('a', encoding='ascii') as stream:
    stream.write(json.dumps({'event':'end','map':map_path.stem,'time':time.time()}) + '\\n')
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _declaration() -> dict[str, object]:
    maps = []
    for ordinal in range(28):
        style = CONCRETE_STYLES[ordinal // 4]
        maps.append({
            "ordinal": ordinal,
            "map": f"b2q26_compile_{style}_{27 - ordinal:02d}",
            "seed": 99000000 + ordinal,
            "style": style,
            "grid": 5,
            "observed_heat": None,
        })
    return {
        "schema": DECLARATION_SCHEMA,
        "qualification_id": "b2q26_compile_fixture",
        "mode": "qualification",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "toolchain_authority_sha256": ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
        "generator": {"version": "v6", "grid": 5, "gym": False, "observed_heat": None},
        "selection": {
            "required_map_count": 28,
            "required_concrete_styles": list(CONCRETE_STYLES),
            "required_maps_per_style": 4,
        },
        "implementation": IMPLEMENTATION,
        "maps": maps,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    declaration = _declaration()
    declaration_path = tmp_path / "qualification-declaration.json"
    declaration_path.write_bytes(canonical_bytes(declaration))
    source = tmp_path / "qualification-source"
    source.mkdir()
    records: dict[str, dict[str, dict[str, object]]] = {}
    rows = []
    for declared in declaration["maps"]:
        map_id = declared["map"]
        records[map_id] = {}
        for suffix in SOURCE_SUFFIXES:
            path = source / f"{map_id}{suffix}"
            path.write_text(f"{map_id}:{suffix}\n", encoding="ascii")
            records[map_id][suffix] = _file_record(path)
        rows.append({
            "ordinal": declared["ordinal"],
            "map": map_id,
            "criteria": {
                "source-files-complete": True,
                "cold-bytes-identical": True,
                "metadata-contract": True,
                "source-static": True,
                "route-contract": True,
                "spawn-origin-binding": True,
                "layout-unique": True,
            },
            "evidence_sha256": _source_evidence_sha256(map_id, records[map_id]),
            "failures": [],
            "passed": True,
        })
    source_report = {
        "schema": STAGE_SCHEMA,
        "qualification_id": declaration["qualification_id"],
        "mode": "qualification",
        "stage": "source",
        "non_admissible": True,
        "retryable": True,
        "final_cohort_authorized": False,
        "declaration_sha256": _sha256(declaration_path.read_bytes()),
        "implementation": IMPLEMENTATION,
        "toolchain_authority_sha256": ACCEPTED_TOOLCHAIN_AUTHORITY_SHA256,
        "input_report_sha256": None,
        "infrastructure_checks": {
            "source-static": True,
            "deterministic-cold-rebuild": True,
            "exact-membership": True,
            "bounded-parallel-workers": True,
            "input-stability": True,
        },
        "map_count": 28,
        "pass_count": 28,
        "maps": rows,
        "failures": [],
    }
    source_report_path = tmp_path / "source-report.json"
    source_report_path.write_bytes(canonical_bytes(source_report))
    basedir = tmp_path / "baseq2"
    basedir.mkdir()
    _write_pak(basedir / "pak0.pak")
    q2tool = tmp_path / "q2tool"
    _write_q2tool(q2tool)
    authority = replace(
        load_toolchain_authority(),
        q2tool_sha256=_file_sha256(q2tool),
        pak0_sha256=_file_sha256(basedir / "pak0.pak"),
        colormap_sha256=_sha256(b"colormap"),
    )
    bsp = tmp_path / "fixture.bsp"
    bsp.write_bytes(_bsp())
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("Q2_FAKE_BSP", str(bsp))
    monkeypatch.setenv("Q2_FAKE_EVENTS", str(events))
    return {
        "declaration": declaration,
        "declaration_path": declaration_path,
        "source_report": source_report_path,
        "source": source,
        "staging": tmp_path / "compiled-staging",
        "compiled": tmp_path / "compiled",
        "logs": tmp_path / "compile-logs",
        "report": tmp_path / "compile-report.json",
        "q2tool": q2tool,
        "basedir": basedir,
        "events": events,
        "authority": authority,
    }


def _run(paths: dict[str, object], *, jobs: int = 4, timeout: float = 5.0) -> dict:
    return compile_qualification(
        declaration_path=paths["declaration_path"],
        source_report_path=paths["source_report"],
        source_root=paths["source"],
        staging_root=paths["staging"],
        compiled_root=paths["compiled"],
        log_root=paths["logs"],
        report_path=paths["report"],
        q2tool=paths["q2tool"],
        basedir=paths["basedir"],
        jobs=jobs,
        timeout_seconds=timeout,
        implementation_provider=lambda _root: dict(IMPLEMENTATION),
        authority_provider=lambda _root: paths["authority"],
    )


def test_parallel_compile_publishes_in_declaration_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("Q2_FAKE_DELAY", "0.04")
    started = time.monotonic()
    report = _run(paths, jobs=4)
    elapsed = time.monotonic() - started
    assert elapsed < 1.1
    assert report["stage"] == "compile"
    assert report["pass_count"] == 28
    assert [row["map"] for row in report["maps"]] == [
        row["map"] for row in paths["declaration"]["maps"]
    ]
    events = [json.loads(line) for line in paths["events"].read_text().splitlines()]
    assert sum(event["event"] == "start" for event in events[:4]) == 4
    assert paths["compiled"].is_dir()
    assert not paths["staging"].exists()


def test_one_q2tool_failure_publishes_honest_sparse_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    failed = paths["declaration"]["maps"][6]["map"]
    monkeypatch.setenv("Q2_FAKE_FAIL_MAP", failed)
    report = _run(paths)
    assert paths["compiled"].is_dir()
    assert not paths["staging"].exists()
    assert report["pass_count"] == 27
    assert report["failures"] == []
    failed_row = next(row for row in report["maps"] if row["map"] == failed)
    assert failed_row["passed"] is False
    assert failed_row["criteria"]["compiled-stage-published"] is False
    assert not (paths["compiled"] / f"{failed}.bsp").exists()
    assert len(list(paths["compiled"].iterdir())) == 28 * len(SOURCE_SUFFIXES) + 27


def test_source_hash_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    trigger = paths["declaration"]["maps"][0]["map"]
    target = paths["source"] / f"{paths['declaration']['maps'][1]['map']}.meta.json"
    monkeypatch.setenv("Q2_FAKE_MUTATE_MAP", trigger)
    monkeypatch.setenv("Q2_FAKE_MUTATE_PATH", str(target))
    with pytest.raises(QualificationCompileError, match="input changed"):
        _run(paths)
    assert not paths["compiled"].exists()


def test_twenty_of_twenty_eight_compile_results_remain_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    rejected = [row["map"] for row in paths["declaration"]["maps"][:8]]
    monkeypatch.setenv("Q2_FAKE_FAIL_MAPS", ",".join(rejected))
    report = _run(paths)
    assert report["pass_count"] == 20
    assert [row["map"] for row in report["maps"] if not row["passed"]] == rejected
    assert len(list(paths["compiled"].iterdir())) == 28 * len(SOURCE_SUFFIXES) + 20


def test_timeout_kills_process_group_and_prevents_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    slow = paths["declaration"]["maps"][2]["map"]
    monkeypatch.setenv("Q2_FAKE_SLOW_MAP", slow)
    monkeypatch.setenv("Q2_FAKE_SLOW_SECONDS", "2")
    with pytest.raises(QualificationCompileError, match="timeout"):
        _run(paths, timeout=0.05)
    assert not paths["compiled"].exists()
    report = json.loads(paths["report"].read_text())
    row = next(row for row in report["maps"] if row["map"] == slow)
    assert row["criteria"]["q2tool-not-timed-out"] is False


def test_extra_compiler_output_fails_exact_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    extra = paths["declaration"]["maps"][4]["map"]
    monkeypatch.setenv("Q2_FAKE_EXTRA_MAP", extra)
    with pytest.raises(QualificationCompileError, match="membership differs"):
        _run(paths)
    assert not paths["compiled"].exists()


def test_report_is_canonical_exclusive_and_chained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    report = _run(paths)
    raw = paths["report"].read_bytes()
    assert raw == canonical_bytes(report)
    assert report["input_report_sha256"] == _sha256(
        paths["source_report"].read_bytes()
    )
    assert report["non_admissible"] is True
    assert report["retryable"] is True
    assert report["final_cohort_authorized"] is False
    first = report["maps"][0]
    evidence_path = paths["logs"] / (
        f"000-{first['map']}.evidence.json"
    )
    evidence = json.loads(evidence_path.read_text())
    assert evidence["q2tool"] == _file_record(paths["q2tool"])
    assert evidence["basedir"]["pak0"] == _file_record(
        paths["basedir"] / "pak0.pak"
    )
    assert _sha256(evidence_path.read_bytes()) == first["evidence_sha256"]
    summary, digest, passed = _validate_stage_report(
        paths["report"],
        "compile",
        paths["declaration"],
        _sha256(paths["declaration_path"].read_bytes()),
        IMPLEMENTATION,
        _sha256(paths["source_report"].read_bytes()),
        {"digests": set()},
    )
    assert summary["pass_count"] == 28
    assert digest == _sha256(raw)
    assert passed == {row["map"] for row in paths["declaration"]["maps"]}
    with pytest.raises(QualificationCompileError, match="output must be fresh"):
        _run(paths)


def test_final_mode_declaration_is_rejected_without_output_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    declaration = dict(paths["declaration"])
    declaration["mode"] = "final"
    paths["declaration_path"].write_bytes(canonical_bytes(declaration))
    with pytest.raises(QualificationCompileError, match="final-mode"):
        _run(paths)
    assert not paths["staging"].exists()
    assert not paths["logs"].exists()
    assert not paths["report"].exists()
