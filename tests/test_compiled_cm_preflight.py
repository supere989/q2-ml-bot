from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import struct

import pytest

from harness.ibsp38 import HEADER_SIZE, LUMP_NAMES
from tools.run_generator_cohort import canonical_bytes
import tools.run_compiled_cm_preflight as preflight


SPAWNS = (
    (0, 0, 24),
    (512, 0, 24),
    (1024, 0, 24),
    (1536, 0, 24),
    (0, 512, 24),
    (512, 512, 24),
    (1024, 512, 24),
    (1536, 512, 24),
)
LAVA_BOUNDS = (256, 256, 12, 512, 512, 20)
HURT_BOUNDS = (-64, -64, -240, 2048, 2048, -64)


def _box_brush(bounds, texture="fixture/stone", contents=0) -> str:
    x0, y0, z0, x1, y1, z1 = bounds
    faces = (
        ((x1, y0, z0), (x1, y1, z0), (x0, y1, z0)),
        ((x1, y1, z1), (x1, y0, z1), (x0, y0, z1)),
        ((x1, y0, z1), (x1, y0, z0), (x0, y0, z0)),
        ((x0, y1, z0), (x1, y1, z0), (x1, y1, z1)),
        ((x0, y0, z0), (x0, y1, z0), (x0, y1, z1)),
        ((x1, y1, z0), (x1, y0, z0), (x1, y0, z1)),
    )
    lines = ["{\n"]
    for first, second, third in faces:
        lines.append(
            f"( {first[0]} {first[1]} {first[2]} ) "
            f"( {second[0]} {second[1]} {second[2]} ) "
            f"( {third[0]} {third[1]} {third[2]} ) "
            f"{texture} 0 0 0 1 1 {contents} 0 0\n"
        )
    lines.append("}\n")
    return "".join(lines)


def _entity_text(spawns=SPAWNS, *, hazards=True) -> str:
    entities = ['{\n"classname" "worldspawn"\n']
    if hazards:
        entities.append(
            _box_brush(LAVA_BOUNDS, "fixture/lava", preflight.CONTENTS_LAVA)
        )
    entities.append('}\n')
    for x, y, z in spawns:
        entities.append(
            '{\n"classname" "info_player_deathmatch"\n'
            f'"origin" "{x} {y} {z}"\n}}\n'
        )
    if hazards:
        entities.append(
            '{\n"classname" "trigger_hurt"\n'
            '"dmg" "100000"\n"spawnflags" "12"\n'
            + _box_brush(HURT_BOUNDS)
            + '}\n'
        )
    return "".join(entities)


def _bsp_entity_text(spawns=SPAWNS, *, hazards=True) -> str:
    entities = ['{\n"classname" "worldspawn"\n}\n']
    for x, y, z in spawns:
        entities.append(
            '{\n"classname" "info_player_deathmatch"\n'
            f'"origin" "{x} {y} {z}"\n}}\n'
        )
    if hazards:
        entities.append(
            '{\n"model" "*1"\n"spawnflags" "12"\n'
            '"dmg" "100000"\n"classname" "trigger_hurt"\n}\n'
        )
    return "".join(entities)


def _build_bsp(
    spawns=SPAWNS, *, hazards=True, hurt_bounds=HURT_BOUNDS,
    lightdata=b"\x80\x80\x80",
) -> bytes:
    lumps = {name: b"" for name in LUMP_NAMES}
    lumps["entities"] = (
        _bsp_entity_text(spawns, hazards=hazards).encode("ascii") + b"\x00"
    )
    lumps["planes"] = struct.pack("<4fi", 0.0, 0.0, 1.0, 0.0, 2)
    lumps["vertices"] = b"".join((
        struct.pack("<3f", 0.0, 0.0, 0.0),
        struct.pack("<3f", 2048.0, 0.0, 0.0),
        struct.pack("<3f", 0.0, 2048.0, 0.0),
    ))
    lumps["visibility"] = struct.pack("<i2i2B", 1, 12, 13, 1, 1)
    lumps["nodes"] = struct.pack(
        "<3i6h2H", 0, -1, -1, -64, -64, -64, 2048, 2048, 256, 0, 1
    )
    texture = b"fixture/stone".ljust(32, b"\x00")
    lumps["texinfo"] = struct.pack(
        "<8fii32si", *(0.0,) * 8, 0, 0, texture, -1
    )
    lumps["faces"] = struct.pack(
        "<Hhihh4Bi", 0, 0, 0, 3, 0, 0, 255, 255, 255,
        0 if lightdata else -1,
    )
    lumps["lighting"] = lightdata
    lumps["leafs"] = struct.pack(
        "<ihh6h4H", 0, 0, 0, -64, -64, -64, 2048, 2048, 256, 0, 1, 0, 0
    )
    lumps["leaffaces"] = struct.pack("<H", 0)
    lumps["edges"] = struct.pack("<6H", 0, 1, 1, 2, 2, 0)
    lumps["surfedges"] = struct.pack("<3i", 0, 1, 2)
    models = [struct.pack(
        "<9f3i",
        -64.0, -64.0, -240.0, 2048.0, 2048.0, 256.0,
        0.0, 0.0, 0.0, 0, 0, 1,
    )]
    if hazards:
        models.append(struct.pack(
            "<9f3i",
            *(float(value) for value in hurt_bounds),
            0.0, 0.0, 0.0, 0, 0, 1,
        ))
    lumps["models"] = b"".join(models)
    lumps["areas"] = struct.pack("<2i", 0, 0)
    header = bytearray(struct.pack("<4si", b"IBSP", 38))
    cursor = HEADER_SIZE
    body = bytearray()
    ranges = []
    for name in LUMP_NAMES:
        content = lumps[name]
        ranges.append((cursor if content else 0, len(content)))
        body.extend(content)
        cursor += len(content)
    for offset, length in ranges:
        header.extend(struct.pack("<2i", offset, length))
    return bytes(header + body)


class FakeOracle:
    fail_columns_for: set[str] = set()
    fail_start_for: set[str] = set()
    error_for: set[str] = set()
    fail_hazards_for: set[str] = set()
    tool_identity = "11" * 32
    source_closure = "22" * 32

    def __init__(self, binary, bsp, kind, limits):
        del binary, kind, limits
        self.bsp = Path(bsp)
        self.map_name = self.bsp.stem
        if self.map_name in self.error_for:
            raise RuntimeError("synthetic oracle startup failure")
        digest = hashlib.sha256(self.bsp.read_bytes()).hexdigest()
        self.identity = {
            "map_sha256": digest,
            "tool_identity": self.tool_identity,
            "physics_identity": "33" * 32,
            "provenance": {"source_closure_sha256": self.source_closure},
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def call(self, requests):
        return [self._response(request) for request in requests]

    def _response(self, request):
        identifier = request["id"]
        if request["op"] in {"point_contents", "transformed_point_contents"}:
            contents = 0
            if self.map_name not in self.fail_hazards_for:
                contents = (
                    preflight.CONTENTS_LAVA
                    if ":lava:" in identifier else preflight.CONTENTS_SOLID
                )
            return {
                "id": identifier,
                "ok": True,
                "contents": contents,
            }
        failed = False
        if self.map_name in self.fail_start_for and identifier.endswith(":standing"):
            failed = True
        if self.map_name in self.fail_columns_for and ":column:" in identifier:
            height = int(identifier.rsplit(":", 1)[1])
            failed = height >= 40
        start = request["start"]
        end = request["end"]
        is_support = ":support" in identifier
        if is_support:
            endpos = [start[0], start[1], start[2] - 9.0]
            fraction = 0.1
        else:
            endpos = list(end)
            fraction = 0.0 if failed else 1.0
        return {
            "id": identifier,
            "ok": True,
            "startsolid": failed,
            "allsolid": failed,
            "fraction": fraction,
            "endpos": endpos,
            "plane": {"normal": [0.0, 0.0, 1.0]},
        }


@pytest.fixture(autouse=True)
def _reset_fake_oracle():
    FakeOracle.fail_columns_for = set()
    FakeOracle.fail_start_for = set()
    FakeOracle.error_for = set()
    FakeOracle.fail_hazards_for = set()


@pytest.fixture
def compiled_cohort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    declaration = (
        Path(__file__).resolve().parents[1]
        / "docs/multires/B2-GENERATED-COHORT-71442-DECLARATION.json"
    )
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    value = json.loads(declaration.read_text(encoding="ascii"))
    for row in value["maps"]:
        name = row["map"]
        (compiled / f"{name}.map").write_text(_entity_text(), encoding="ascii")
        for suffix in (".json", ".meta.json", ".routes.json"):
            (compiled / f"{name}{suffix}").write_bytes(b"{}\n")
        (compiled / f"{name}.lattice.json").write_text(
            json.dumps({"danger": [list(LAVA_BOUNDS)]}) + "\n",
            encoding="ascii",
        )
        (compiled / f"{name}.bsp").write_bytes(_build_bsp())
    oracle = tmp_path / "q2-cm-oracle"
    oracle.write_bytes(b"fake executable bytes\n")
    oracle.chmod(0o755)
    digest = hashlib.sha256(oracle.read_bytes()).hexdigest()
    gate = SimpleNamespace(
        cm_executable_sha256=digest,
        oracle_tool_identity=FakeOracle.tool_identity,
        oracle_source_closure_sha256=FakeOracle.source_closure,
    )
    monkeypatch.setattr(preflight, "load_b1_authority_gate", lambda root: gate)
    return declaration, compiled, oracle, value


def _report(compiled_cohort, *, jobs=8):
    declaration, compiled, oracle, _ = compiled_cohort
    return preflight.build_report(
        declaration_path=declaration,
        compiled_dir=compiled,
        cm_oracle=oracle,
        jobs=jobs,
        oracle_batch_timeout_seconds=1.0,
        oracle_factory=FakeOracle,
    )


def test_parallel_preflight_passes_and_binds_canonical_evidence(compiled_cohort):
    report = _report(compiled_cohort)
    assert report["passed"] is True
    assert report["pass_count"] == 28
    assert report["failure_count"] == 0
    assert report["execution"]["parallel_jobs"] == 8
    assert report["admission_status"] == "non-admissible-preflight-only"
    assert report["promotion_authority"] is False
    assert report["compiled_membership"]["report"]["actual_file_count"] == 168
    assert report["checks"]["compiled_lightdata_presence"] is True
    assert report["checks"]["basic_hazard_containment"] is True
    assert all(row["all_to_all_reachability"] == "not-evaluated-by-preflight"
               for row in report["maps"])
    first = report["maps"][0]
    assert first["compiled_lightdata"] == {
        "bytes": 3,
        "sha256": hashlib.sha256(b"\x80\x80\x80").hexdigest(),
        "present": True,
    }
    containment = first["basic_hazard_containment"]
    assert containment["declared_hazard_count"] == 2
    assert containment["checked_hazard_count"] == 2
    assert containment["passed"] is True
    assert [hazard["claim_id"] for hazard in containment["hazards"]] == [
        "hazard:hurt:0000", "hazard:lava:0000",
    ]
    assert all(hazard["probe_count"] == 7 for hazard in containment["hazards"])
    canonical_digest = report.pop("canonical_record_sha256")
    assert canonical_digest == hashlib.sha256(canonical_bytes(report)).hexdigest()


def test_preflight_rejects_real_92_unit_column_pattern(compiled_cohort):
    failed_map = compiled_cohort[3]["maps"][3]["map"]
    FakeOracle.fail_columns_for = {failed_map}
    report = _report(compiled_cohort)
    assert report["passed"] is False
    assert report["pass_count"] == 27
    row = report["maps"][3]
    assert row["passed"] is False
    assert all(spawn["column_clearance_milliunits"] == 92_000
               for spawn in row["spawns"])
    assert any("less than 96" in failure for failure in row["failures"])


def test_preflight_rejects_nonexact_compiled_membership(compiled_cohort):
    _, compiled, _, declaration = compiled_cohort
    missing_map = declaration["maps"][0]["map"]
    (compiled / f"{missing_map}.bsp").unlink()
    report = _report(compiled_cohort)
    assert report["passed"] is False
    assert report["maps"] == []
    assert any("missing file" in failure for failure in report["failures"])


def test_preflight_rejects_input_staleness(compiled_cohort, monkeypatch):
    original = preflight.verify_stage_membership
    calls = 0

    def stale_on_final(declaration, directory, stage):
        nonlocal calls
        calls += 1
        value = original(declaration, directory, stage)
        if calls == 2:
            value = deepcopy(value)
            value["maps"][0]["files"][".bsp"]["sha256"] = "ff" * 32
        return value

    monkeypatch.setattr(preflight, "verify_stage_membership", stale_on_final)
    report = _report(compiled_cohort, jobs=4)
    assert report["passed"] is False
    assert report["input_stability"]["compiled_membership"] is False
    assert any("changed during preflight" in failure for failure in report["failures"])


def test_preflight_records_oracle_process_error_without_partial_pass(compiled_cohort):
    failed_map = compiled_cohort[3]["maps"][11]["map"]
    FakeOracle.error_for = {failed_map}
    report = _report(compiled_cohort)
    assert report["passed"] is False
    assert report["pass_count"] == 27
    row = report["maps"][11]
    assert row["passed"] is False
    assert "synthetic oracle startup failure" in row["failures"][0]


def test_preflight_rejects_compiled_spawn_origin_drift(compiled_cohort):
    _, compiled, _, declaration = compiled_cohort
    failed_map = declaration["maps"][7]["map"]
    changed = list(SPAWNS)
    changed[-1] = (2048, 512, 24)
    (compiled / f"{failed_map}.bsp").write_bytes(_build_bsp(changed))
    report = _report(compiled_cohort)
    row = report["maps"][7]
    assert report["passed"] is False
    assert row["spawn_origin_sets_match"] is False
    assert "compiled BSP spawn origins differ from source map" in row["failures"]


def test_preflight_rejects_missing_compiled_lightdata(compiled_cohort):
    _, compiled, _, declaration = compiled_cohort
    failed_map = declaration["maps"][5]["map"]
    (compiled / f"{failed_map}.bsp").write_bytes(_build_bsp(lightdata=b""))
    report = _report(compiled_cohort)
    row = report["maps"][5]
    assert report["passed"] is False
    assert row["compiled_lightdata"]["bytes"] == 0
    assert row["compiled_lightdata"]["present"] is False
    assert "compiled BSP has no lightdata" in row["failures"]


def test_preflight_rejects_missing_compiled_hazard_contents(compiled_cohort):
    failed_map = compiled_cohort[3]["maps"][9]["map"]
    FakeOracle.fail_hazards_for = {failed_map}
    report = _report(compiled_cohort)
    row = report["maps"][9]
    containment = row["basic_hazard_containment"]
    assert report["passed"] is False
    assert containment["passed"] is False
    assert all(hazard["probe_count"] == 7 for hazard in containment["hazards"])
    assert any("no compiled lava contents" in failure
               for failure in containment["failures"])
    assert any("no compiled inline brush geometry" in failure
               for failure in containment["failures"])


def test_preflight_rejects_trigger_hurt_model_bound_drift(compiled_cohort):
    _, compiled, _, declaration = compiled_cohort
    failed_map = declaration["maps"][13]["map"]
    changed = (*HURT_BOUNDS[:3], HURT_BOUNDS[3] + 32, *HURT_BOUNDS[4:])
    (compiled / f"{failed_map}.bsp").write_bytes(
        _build_bsp(hurt_bounds=changed)
    )
    report = _report(compiled_cohort)
    containment = report["maps"][13]["basic_hazard_containment"]
    assert report["passed"] is False
    assert containment["passed"] is False
    assert "compiled BSP has undeclared trigger_hurt model bounds" in containment[
        "failures"
    ]
    hurt = containment["hazards"][0]
    assert hurt["probe_count"] == 0
    assert "no exact compiled trigger_hurt inline-model bounds" in hurt["failures"]


def test_zero_hazard_map_has_nonvacuous_exact_empty_result(compiled_cohort):
    _, compiled, _, declaration = compiled_cohort
    safe_map = declaration["maps"][17]["map"]
    (compiled / f"{safe_map}.map").write_text(
        _entity_text(hazards=False), encoding="ascii"
    )
    (compiled / f"{safe_map}.lattice.json").write_text(
        '{"danger":[]}\n', encoding="ascii"
    )
    (compiled / f"{safe_map}.bsp").write_bytes(_build_bsp(hazards=False))
    report = _report(compiled_cohort)
    containment = report["maps"][17]["basic_hazard_containment"]
    assert containment == {
        "declared_hazard_count": 0,
        "checked_hazard_count": 0,
        "hazards": [],
        "failures": [],
        "passed": True,
    }


def test_run_refuses_to_overwrite_report_before_any_work(
    compiled_cohort, tmp_path: Path
):
    declaration, compiled, oracle, _ = compiled_cohort
    output = tmp_path / "report.json"
    output.write_bytes(b"owner evidence\n")
    with pytest.raises(preflight.CompiledCmPreflightError, match="already exists"):
        preflight.run(
            declaration_path=declaration,
            compiled_dir=compiled,
            cm_oracle=oracle,
            output=output,
            jobs=2,
            oracle_batch_timeout_seconds=1.0,
        )
    assert output.read_bytes() == b"owner evidence\n"
