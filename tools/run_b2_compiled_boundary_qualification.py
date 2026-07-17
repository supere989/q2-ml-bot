#!/usr/bin/env python3
"""Qualify authored spawn headroom through real q2tool and Yamagi CM.

This is a disposable, retryable qualification lane.  Its evidence is never a
generated-cohort, Atlas, bundle, deployment, or training admission artifact.
Compilation and proof can run together, or as separate cross-host phases.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_b1_authority import canonical_cm_physics_identity  # noqa: E402
from tools.b2_qualification_toolchain import (  # noqa: E402
    Q2TOOL_FLAGS,
    ToolchainAuthority,
    ToolchainAuthorityError,
    inspect_baseq2_assets,
    inspect_q2tool,
    load_toolchain_authority,
)

DEFAULT_FIXTURE_DIR = ROOT / "tests/fixtures/compiled_boundary"
COMPILE_SCHEMA = "q2-b2-compiled-boundary-compile-v1"
PROOF_SCHEMA = "q2-b2-compiled-boundary-qualification-v1"
MASK_PLAYERSOLID = 33_619_971
STANDING_MINS = [-16, -16, -24]
STANDING_MAXS = [16, 16, 32]
AUTHORED_SPAWN = [0, 0, 24]
ENGINE_LINK_LIFT = 9
LINKED_SPAWN = [0, 0, 33]
COLUMN_REQUIREMENT = 96
TRACE_STEP = 4
CASES = (
    {"case_id": "spawn_ceiling_104", "ceiling_units": 104,
     "expected_clearance_units": 92, "expected_pass": False},
    {"case_id": "spawn_ceiling_105", "ceiling_units": 105,
     "expected_clearance_units": 92, "expected_pass": False},
    {"case_id": "spawn_ceiling_106", "ceiling_units": 106,
     "expected_clearance_units": 96, "expected_pass": True},
)


class QualificationError(RuntimeError):
    """A qualification input, tool, response, or boundary failed closed."""


_PLANE_RE = re.compile(
    r"^\s*\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)\s*"
    r"\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)\s*"
    r"\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)"
)
_PROPERTY_RE = re.compile(r'^\s*"([^"]+)"\s+"([^"]*)"\s*$')


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QualificationError(f"required regular file is absent: {path}")
    return {"path": str(path.absolute()), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size}


def write_canonical(path: Path, value: object) -> None:
    if not path.parent.is_dir():
        raise QualificationError(f"report parent does not exist: {path.parent}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def qualification_only() -> dict[str, bool]:
    return {
        "atlas_admissible": False,
        "bundle_admissible": False,
        "cohort_admissible": False,
        "deployment_admissible": False,
        "qualification_only": True,
        "training_admissible": False,
    }


def _parse_fixture_geometry(text: str, case_id: str) -> dict[str, Any]:
    entities: list[dict[str, str]] = []
    world_brushes: list[list[dict[str, Any]]] = []
    entity: dict[str, str] | None = None
    brush: list[dict[str, Any]] | None = None
    depth = 0
    for raw_line in text.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        if line == "{":
            depth += 1
            if depth == 1:
                entity = {}
            elif depth == 2:
                brush = []
            else:
                raise QualificationError(
                    f"fixture has unsupported nested geometry: {case_id}"
                )
            continue
        if line == "}":
            if depth == 2:
                if entity is None or entity.get("classname") != "worldspawn":
                    raise QualificationError(
                        f"non-world fixture brush is forbidden: {case_id}"
                    )
                if not brush:
                    raise QualificationError(f"empty fixture brush: {case_id}")
                world_brushes.append(brush)
                brush = None
            elif depth == 1:
                if entity is None:
                    raise QualificationError(f"fixture entity is malformed: {case_id}")
                entities.append(entity)
                entity = None
            else:
                raise QualificationError(f"fixture braces are unbalanced: {case_id}")
            depth -= 1
            continue
        if depth == 1:
            match = _PROPERTY_RE.fullmatch(line)
            if match is None or entity is None or match.group(1) in entity:
                raise QualificationError(
                    f"fixture entity property is malformed: {case_id}"
                )
            entity[match.group(1)] = match.group(2)
            continue
        if depth == 2:
            match = _PLANE_RE.match(line)
            if match is None or brush is None:
                raise QualificationError(f"fixture brush plane is malformed: {case_id}")
            values = [int(value) for value in match.groups()[:9]]
            points = [values[index:index + 3] for index in range(0, 9, 3)]
            brush.append({"points": points, "texture": match.group(10)})
            continue
        raise QualificationError(f"fixture content is outside an entity: {case_id}")
    if depth != 0 or entity is not None or brush is not None:
        raise QualificationError(f"fixture braces are unbalanced: {case_id}")
    worldspawns = [item for item in entities if item.get("classname") == "worldspawn"]
    spawns = [
        item for item in entities
        if item.get("classname") == "info_player_deathmatch"
    ]
    if len(worldspawns) != 1 or len(spawns) != 1:
        raise QualificationError(
            f"fixture must contain one worldspawn and one deathmatch spawn: {case_id}"
        )
    try:
        spawn_origin = [int(value) for value in spawns[0]["origin"].split()]
    except (KeyError, ValueError) as error:
        raise QualificationError(f"fixture spawn origin is malformed: {case_id}") from error
    if len(spawn_origin) != 3:
        raise QualificationError(f"fixture spawn origin is malformed: {case_id}")

    def horizontal_faces(texture: str) -> list[dict[str, Any]]:
        return [
            plane for candidate in world_brushes for plane in candidate
            if plane["texture"] == texture
            and len({point[2] for point in plane["points"]}) == 1
        ]

    floor_faces = horizontal_faces("e1u1/floor3_3")
    ceiling_faces = horizontal_faces("e1u1/ceil1_4")
    if len(floor_faces) != 2 or len(ceiling_faces) != 2:
        raise QualificationError(
            f"fixture floor/ceiling planes are not independently identifiable: {case_id}"
        )
    floor_heights = sorted({face["points"][0][2] for face in floor_faces})
    ceiling_heights = sorted({face["points"][0][2] for face in ceiling_faces})
    if (
        len(floor_heights) != 2 or floor_heights[1] - floor_heights[0] != 16
        or len(ceiling_heights) != 2
        or ceiling_heights[1] - ceiling_heights[0] != 16
    ):
        raise QualificationError(f"fixture slab thickness differs: {case_id}")
    floor_top = floor_heights[1]
    ceiling_bottom = ceiling_heights[0]
    for label, faces in (("floor", floor_faces), ("ceiling", ceiling_faces)):
        xs = {point[0] for face in faces for point in face["points"]}
        ys = {point[1] for face in faces for point in face["points"]}
        if min(xs) != -144 or max(xs) != 144 or min(ys) != -144 or max(ys) != 144:
            raise QualificationError(
                f"fixture {label} footprint differs: {case_id}"
            )
    return {
        "ceiling_bottom_units": ceiling_bottom,
        "floor_top_units": floor_top,
        "spawn_origin_units": spawn_origin,
        "world_brush_count": len(world_brushes),
    }


def fixture_records(
    fixture_dir: Path, authority: ToolchainAuthority | None = None,
) -> list[dict[str, Any]]:
    try:
        accepted = authority or load_toolchain_authority(ROOT)
    except ToolchainAuthorityError as error:
        raise QualificationError(
            f"canonical toolchain authority rejected: {error}"
        ) from error
    expected_dir = (ROOT / "tests/fixtures/compiled_boundary").resolve()
    if fixture_dir.resolve() != expected_dir:
        raise QualificationError("boundary fixture directory is not canonical")
    cases = {case["case_id"]: case for case in CASES}
    records = []
    for fixture in accepted.fixtures:
        case_id = str(fixture["case_id"])
        if case_id not in cases:
            raise QualificationError("toolchain fixture membership differs")
        case = cases[case_id]
        source = ROOT / str(fixture["relative_path"])
        record = file_record(source)
        if record["sha256"] != fixture["sha256"]:
            raise QualificationError(f"fixture bytes differ for {case_id}")
        try:
            geometry = _parse_fixture_geometry(
                source.read_text(encoding="ascii"), case_id
            )
        except UnicodeError as error:
            raise QualificationError(f"fixture is not ASCII: {case_id}") from error
        expected_geometry = {
            "ceiling_bottom_units": fixture["ceiling_bottom_units"],
            "floor_top_units": fixture["floor_top_units"],
            "spawn_origin_units": fixture["spawn_origin_units"],
        }
        if any(geometry[name] != value for name, value in expected_geometry.items()):
            raise QualificationError(f"fixture authored geometry differs for {case_id}")
        if case["ceiling_units"] != geometry["ceiling_bottom_units"]:
            raise QualificationError(f"fixture boundary case differs for {case_id}")
        records.append({**case, "geometry": geometry, "source": record})
    if {row["case_id"] for row in records} != set(cases):
        raise QualificationError("toolchain fixture membership differs")
    return records


def check_bsp(path: Path) -> dict[str, Any]:
    record = file_record(path)
    if record["size_bytes"] < 8:
        raise QualificationError(f"BSP is truncated: {path}")
    with path.open("rb") as stream:
        magic = stream.read(4)
        version = struct.unpack("<i", stream.read(4))[0]
    if magic != b"IBSP" or version != 38:
        raise QualificationError(f"q2tool emitted non-IBSP-38 output: {path}")
    return {**record, "ibsp_version": version}


def run_logged(command: Sequence[str], cwd: Path, stdout_path: Path,
               stderr_path: Path, timeout_seconds: float) -> int:
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            list(command), cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise QualificationError(
                f"q2tool timed out after {timeout_seconds:g} seconds"
            )


def compile_fixtures(*, q2tool: Path, basedir: Path, fixture_dir: Path,
                     work_root: Path, workers: int,
                     timeout_seconds: float,
                     authority: ToolchainAuthority | None = None) -> dict[str, Any]:
    if work_root.exists() or work_root.is_symlink():
        raise QualificationError(f"work root must not exist: {work_root}")
    if not 1 <= workers <= len(CASES):
        raise QualificationError("compile workers must be between 1 and 3")
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 86400:
        raise QualificationError("compile timeout must be in (0, 86400]")
    try:
        accepted = authority or load_toolchain_authority(ROOT)
        inspect_q2tool(q2tool, accepted)
        assets = inspect_baseq2_assets(basedir, accepted)
    except ToolchainAuthorityError as error:
        raise QualificationError(f"canonical toolchain rejected: {error}") from error
    tool_before = file_record(q2tool)
    if not os.access(q2tool, os.X_OK):
        raise QualificationError(f"q2tool is not executable: {q2tool}")
    fixtures = fixture_records(fixture_dir, accepted)
    compiled_dir = work_root / "compiled"
    log_dir = work_root / "logs"
    compiled_dir.mkdir(parents=True)
    log_dir.mkdir()
    for case in fixtures:
        shutil.copyfile(case["source"]["path"],
                        compiled_dir / f"{case['case_id']}.map")

    def compile_one(case: dict[str, Any]) -> dict[str, Any]:
        case_id = case["case_id"]
        source = compiled_dir / f"{case_id}.map"
        command = [str(q2tool.absolute()), *Q2TOOL_FLAGS,
                   str(basedir.absolute()), str(source.absolute())]
        stdout_path = log_dir / f"{case_id}.stdout.log"
        stderr_path = log_dir / f"{case_id}.stderr.log"
        returncode = run_logged(
            command, compiled_dir, stdout_path, stderr_path, timeout_seconds,
        )
        if returncode != 0:
            raise QualificationError(f"q2tool exited {returncode} for {case_id}")
        bsp = check_bsp(compiled_dir / f"{case_id}.bsp")
        return {
            "case_id": case_id,
            "command": command,
            "returncode": returncode,
            "bsp": bsp,
            "stderr": file_record(stderr_path),
            "stdout": file_record(stdout_path),
        }

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="b2-boundary-q2tool") as executor:
        compiled = list(executor.map(compile_one, fixtures))
    tool_after = file_record(q2tool)
    if tool_after != tool_before:
        raise QualificationError("q2tool changed during qualification compilation")
    for fixture in fixtures:
        if file_record(Path(fixture["source"]["path"])) != fixture["source"]:
            raise QualificationError("fixture changed during qualification compilation")
    if authority is None:
        try:
            if load_toolchain_authority(ROOT) != accepted:
                raise QualificationError(
                    "toolchain authority changed during qualification compilation"
                )
        except ToolchainAuthorityError as error:
            raise QualificationError(
                f"toolchain authority stability failed: {error}"
            ) from error
    return {
        "schema": COMPILE_SCHEMA,
        "status": "compiled-non-admissible-qualification",
        "passed": True,
        "admission": qualification_only(),
        "parallel_compile_workers": workers,
        "toolchain_authority": accepted.manifest_record(),
        "q2tool": tool_before,
        "basedir": assets,
        "fixed_q2tool_flags": list(accepted.q2tool_flags),
        "fixtures": fixtures,
        "compiled": sorted(compiled, key=lambda row: row["case_id"]),
    }


def load_compile_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = file_record(path)
    try:
        report = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise QualificationError(f"compile report is not JSON: {path}") from error
    if canonical_bytes(report) != path.read_bytes():
        raise QualificationError("compile report is not canonical JSON")
    if (report.get("schema") != COMPILE_SCHEMA or report.get("passed") is not True
            or report.get("status") != "compiled-non-admissible-qualification"
            or report.get("admission") != qualification_only()):
        raise QualificationError("compile report is not passing qualification evidence")
    return report, record


def invoke_cm(cm_oracle: Path, bsp: Path, requests: list[dict[str, Any]],
              timeout_seconds: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    command = [str(cm_oracle.absolute()), "--map", str(bsp.absolute())]
    payload = b"".join(canonical_bytes(request) for request in requests)
    try:
        result = subprocess.run(
            command, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise QualificationError(f"CM oracle timed out for {bsp.name}") from error
    if result.returncode != 0:
        raise QualificationError(
            f"CM oracle exited {result.returncode} for {bsp.name}"
        )
    try:
        records = [json.loads(line) for line in result.stdout.splitlines()]
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise QualificationError(f"CM oracle emitted invalid JSON for {bsp.name}") from error
    if len(records) != len(requests):
        raise QualificationError(f"CM oracle response count mismatch for {bsp.name}")
    for request, record in zip(requests, records):
        if record.get("id") != request["id"] or record.get("ok") is not True:
            raise QualificationError(f"CM oracle rejected or mismatched {request['id']}")
        if record.get("schema") != "q2-cm-oracle-v1":
            raise QualificationError("CM oracle schema mismatch")
    transcript = {
        "command": command,
        "returncode": result.returncode,
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
        "stderr_size_bytes": len(result.stderr),
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stdout_size_bytes": len(result.stdout),
    }
    return records, transcript


def prove_compilation(*, compile_report_path: Path, compiled_dir: Path,
                      cm_oracle: Path, fixture_dir: Path,
                      timeout_seconds: float,
                      expected_cm_oracle_sha256: str,
                      authority: ToolchainAuthority | None = None) -> dict[str, Any]:
    try:
        accepted = authority or load_toolchain_authority(ROOT)
    except ToolchainAuthorityError as error:
        raise QualificationError(
            f"canonical toolchain authority rejected: {error}"
        ) from error
    compile_report, compile_record = load_compile_report(compile_report_path)
    if compile_report.get("toolchain_authority") != accepted.manifest_record():
        raise QualificationError("compile evidence toolchain authority differs")
    if compile_report.get("q2tool", {}).get("sha256") != accepted.q2tool_sha256:
        raise QualificationError("compile evidence q2tool bytes differ")
    if compile_report.get("fixed_q2tool_flags") != list(accepted.q2tool_flags):
        raise QualificationError("compile evidence q2tool flags differ")
    basedir_record = compile_report.get("basedir")
    if (
        not isinstance(basedir_record, dict)
        or basedir_record.get("pak0", {}).get("sha256")
        != accepted.pak0_sha256
        or basedir_record.get("required_member", {}).get("sha256")
        != accepted.colormap_sha256
    ):
        raise QualificationError("compile evidence baseq2 assets differ")
    oracle_before = file_record(cm_oracle)
    if oracle_before["sha256"] != expected_cm_oracle_sha256:
        raise QualificationError("CM oracle SHA-256 does not match the explicit pin")
    if not os.access(cm_oracle, os.X_OK):
        raise QualificationError(f"CM oracle is not executable: {cm_oracle}")
    fixtures = fixture_records(fixture_dir, accepted)
    reported_fixtures = {row["case_id"]: row for row in compile_report["fixtures"]}
    reported_compiled = {row["case_id"]: row for row in compile_report["compiled"]}
    if set(reported_fixtures) != {row["case_id"] for row in fixtures}:
        raise QualificationError("compile report fixture membership mismatch")
    if set(reported_compiled) != set(reported_fixtures):
        raise QualificationError("compile report BSP membership mismatch")

    proofs = []
    for case in fixtures:
        case_id = case["case_id"]
        if reported_fixtures[case_id]["source"]["sha256"] != case["source"]["sha256"]:
            raise QualificationError(f"fixture digest mismatch for {case_id}")
        bsp_path = compiled_dir / f"{case_id}.bsp"
        bsp = check_bsp(bsp_path)
        reported_bsp = reported_compiled[case_id]["bsp"]
        for field in ("sha256", "size_bytes", "ibsp_version"):
            if bsp[field] != reported_bsp[field]:
                raise QualificationError(f"compiled BSP {field} mismatch for {case_id}")
        requests: list[dict[str, Any]] = [{"id": "identity", "op": "identity"}]
        requests.append({
            "id": "standing", "op": "box_trace", "start": LINKED_SPAWN,
            "end": LINKED_SPAWN, "mins": STANDING_MINS, "maxs": STANDING_MAXS,
            "mask": MASK_PLAYERSOLID,
        })
        for height in range(TRACE_STEP, 129, TRACE_STEP):
            requests.append({
                "id": f"column:{height}", "op": "box_trace",
                "start": LINKED_SPAWN,
                "end": [LINKED_SPAWN[0], LINKED_SPAWN[1], LINKED_SPAWN[2] + height],
                "mins": STANDING_MINS, "maxs": STANDING_MAXS,
                "mask": MASK_PLAYERSOLID,
            })
        responses, transcript = invoke_cm(
            cm_oracle, bsp_path, requests, timeout_seconds,
        )
        identity = responses[0]
        for field in ("tool_identity", "physics_identity", "map_sha256"):
            value = identity.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise QualificationError(f"CM identity lacks {field} for {case_id}")
        if identity["map_sha256"] != bsp["sha256"]:
            raise QualificationError(f"CM loaded the wrong BSP for {case_id}")
        expected_physics = canonical_cm_physics_identity(
            identity["tool_identity"], bsp["sha256"]
        )
        if identity["physics_identity"] != expected_physics:
            raise QualificationError(
                f"CM emitted a noncanonical physics identity for {case_id}"
            )
        standing = responses[1]
        if (standing.get("startsolid") is not False
                or standing.get("allsolid") is not False
                or standing.get("fraction") != 1):
            raise QualificationError(f"linked standing spawn is blocked for {case_id}")
        measurements = []
        clearance = 56
        blocked_at = None
        for height, response in zip(range(TRACE_STEP, 129, TRACE_STEP), responses[2:]):
            fraction = response.get("fraction")
            if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                    or not math.isfinite(fraction) or not 0 <= fraction <= 1
                    or not isinstance(response.get("startsolid"), bool)
                    or not isinstance(response.get("allsolid"), bool)):
                raise QualificationError(f"invalid CM trace response for {case_id}:{height}")
            clear = (fraction == 1 and response["startsolid"] is False
                     and response["allsolid"] is False)
            measurements.append({
                "sweep_units": height,
                "clear": clear,
                "fraction_millionths": round(float(fraction) * 1_000_000),
                "startsolid": response["startsolid"],
                "allsolid": response["allsolid"],
            })
            if blocked_at is None:
                if clear:
                    clearance = 56 + height
                else:
                    blocked_at = height
        passed_requirement = clearance >= COLUMN_REQUIREMENT
        if clearance != case["expected_clearance_units"]:
            raise QualificationError(
                f"{case_id} measured {clearance}u; expected "
                f"{case['expected_clearance_units']}u"
            )
        if passed_requirement is not case["expected_pass"]:
            raise QualificationError(f"{case_id} boundary disposition mismatch")
        proofs.append({
            "case_id": case_id,
            "authored_floor_to_ceiling_units": case["ceiling_units"],
            "expected_pass": case["expected_pass"],
            "bsp": bsp,
            "cm_identity": {
                "map_checksum": identity.get("map_checksum"),
                "map_sha256": identity["map_sha256"],
                "physics_identity": identity["physics_identity"],
                "schema": identity["schema"],
                "tool_identity": identity["tool_identity"],
            },
            "cm_transcript": transcript,
            "linked_standing_clear": True,
            "first_blocked_sweep_units": blocked_at,
            "column_clearance_milliunits": clearance * 1000,
            "column_clear_96": passed_requirement,
            "measurements": measurements,
        })
    oracle_after = file_record(cm_oracle)
    if oracle_after != oracle_before:
        raise QualificationError("CM oracle changed during qualification proof")
    return {
        "schema": PROOF_SCHEMA,
        "status": "passed-non-admissible-qualification",
        "passed": True,
        "admission": qualification_only(),
        "compile_evidence": compile_record,
        "toolchain_authority": accepted.manifest_record(),
        "q2tool": compile_report["q2tool"],
        "cm_oracle": oracle_before,
        "cm_oracle_sha256_pin": expected_cm_oracle_sha256,
        "contract": {
            "authored_spawn_origin": AUTHORED_SPAWN,
            "column_requirement_units": COLUMN_REQUIREMENT,
            "engine_link_lift_units": ENGINE_LINK_LIFT,
            "linked_spawn_origin": LINKED_SPAWN,
            "mask_playersolid": MASK_PLAYERSOLID,
            "standing_maxs": STANDING_MAXS,
            "standing_mins": STANDING_MINS,
            "trace_step_units": TRACE_STEP,
        },
        "proofs": sorted(proofs, key=lambda row: row["case_id"]),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--phase", choices=("compile", "prove", "all"), default="all")
    result.add_argument("--q2tool", type=Path)
    result.add_argument("--basedir", type=Path)
    result.add_argument("--cm-oracle", type=Path)
    result.add_argument("--cm-oracle-sha256")
    result.add_argument("--work-root", type=Path)
    result.add_argument("--compiled-dir", type=Path)
    result.add_argument("--compile-report", type=Path)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--compile-workers", type=int, default=3)
    result.add_argument("--compile-timeout-seconds", type=float, default=900.0)
    result.add_argument("--oracle-timeout-seconds", type=float, default=60.0)
    return result


def require(value: Path | None, name: str) -> Path:
    if value is None:
        raise QualificationError(f"{name} is required for the selected phase")
    return value.expanduser().absolute()


def require_sha256(value: str | None, name: str) -> str:
    if value is None or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise QualificationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    fixture_dir = DEFAULT_FIXTURE_DIR
    report_path = args.report.expanduser().absolute()
    try:
        authority = load_toolchain_authority(ROOT)
        if args.phase in ("compile", "all"):
            work_root = require(args.work_root, "--work-root")
            compile_report = compile_fixtures(
                q2tool=require(args.q2tool, "--q2tool"),
                basedir=require(args.basedir, "--basedir"), fixture_dir=fixture_dir,
                work_root=work_root, workers=args.compile_workers,
                timeout_seconds=args.compile_timeout_seconds,
                authority=authority,
            )
            if args.phase == "compile":
                write_canonical(report_path, compile_report)
                return 0
            compile_report_path = work_root / "compile-report.json"
            write_canonical(compile_report_path, compile_report)
            compiled_dir = work_root / "compiled"
        else:
            compile_report_path = require(args.compile_report, "--compile-report")
            compiled_dir = require(args.compiled_dir, "--compiled-dir")
        proof = prove_compilation(
            compile_report_path=compile_report_path, compiled_dir=compiled_dir,
            cm_oracle=require(args.cm_oracle, "--cm-oracle"),
            fixture_dir=fixture_dir, timeout_seconds=args.oracle_timeout_seconds,
            expected_cm_oracle_sha256=require_sha256(
                args.cm_oracle_sha256, "--cm-oracle-sha256"
            ),
            authority=authority,
        )
        write_canonical(report_path, proof)
        return 0
    except (OSError, QualificationError, ToolchainAuthorityError) as error:
        failure = {
            "schema": PROOF_SCHEMA if args.phase != "compile" else COMPILE_SCHEMA,
            "status": "failed-non-admissible-qualification",
            "passed": False,
            "admission": qualification_only(),
            "failure": {"phase": args.phase, "message": str(error)},
        }
        try:
            write_canonical(report_path, failure)
        except (OSError, QualificationError):
            pass
        print(f"qualification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
