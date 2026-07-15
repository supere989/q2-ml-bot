#!/usr/bin/env python3
"""Compile one declared generated-map cohort as an atomic, fail-closed stage.

The compiler copies the exact declared source population into a fresh
non-admissible staging directory, invokes q2tool in declaration order, and
publishes the complete compiled population with Linux RENAME_NOREPLACE only
after every member and the exact postcompile membership check pass.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    SOURCE_SUFFIXES,
    canonical_bytes,
    file_sha256,
    load_declaration,
    verify_stage_membership,
)
from tools.retired_cohort_registry import require_unretired_declaration  # noqa: E402


REPORT_SCHEMA = "q2-b2-generated-compile-cohort-v1"
Q2TOOL_FLAGS = (
    "-bsp",
    "-vis",
    "-fast",
    "-rad",
    "-bounce",
    "0",
    "-threads",
    "1",
    "-basedir",
)
REQUIRED_PAK_MEMBER = "pics/colormap.pcx"
AT_FDCWD = -100
RENAME_NOREPLACE = 1
DEFAULT_MAP_TIMEOUT_SECONDS = 3600.0


class CompileCohortError(RuntimeError):
    """Raised when a compiled cohort cannot be published atomically."""


def _absolute(path: Path) -> Path:
    return path.expanduser().absolute()


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing a racing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CompileCohortError("libc has no renameat2; atomic no-replace is required")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise CompileCohortError(
            f"atomic publish {source} -> {destination} failed: "
            f"{os.strerror(error_number)}"
        )
    _fsync_directory(source.parent)
    if destination.parent != source.parent:
        _fsync_directory(destination.parent)


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    return first == second or first in second.parents or second in first.parents


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise CompileCohortError(f"{label} must be a regular, non-symlink file: {path}")


def _file_record(path: Path) -> dict[str, Any]:
    _require_regular_file(path, "artifact")
    return {
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _hash_file_region(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise CompileCohortError("PAK member ends outside pak0.pak")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def inspect_basedir(basedir: Path) -> dict[str, Any]:
    """Bind pak0 and the colormap asset from the supplied baseq2 itself."""
    if basedir.is_symlink() or not basedir.is_dir():
        raise CompileCohortError(
            f"basedir must be an existing, non-symlink baseq2 directory: {basedir}"
        )
    pak_path = basedir / "pak0.pak"
    _require_regular_file(pak_path, "basedir/pak0.pak")
    pak_size = pak_path.stat().st_size
    try:
        with pak_path.open("rb") as stream:
            header = stream.read(12)
            if len(header) != 12:
                raise CompileCohortError("basedir/pak0.pak has a truncated header")
            magic, directory_offset, directory_length = struct.unpack("<4sii", header)
            if magic != b"PACK":
                raise CompileCohortError("basedir/pak0.pak has invalid PACK magic")
            if (
                directory_offset < 12
                or directory_length < 0
                or directory_length % 64 != 0
                or directory_offset + directory_length > pak_size
            ):
                raise CompileCohortError("basedir/pak0.pak has an invalid directory")
            stream.seek(directory_offset)
            directory = stream.read(directory_length)
    except OSError as exc:
        raise CompileCohortError(f"cannot inspect basedir/pak0.pak: {exc}") from exc

    matches: list[dict[str, Any]] = []
    for start in range(0, len(directory), 64):
        raw_name, member_offset, member_length = struct.unpack(
            "<56sii", directory[start : start + 64]
        )
        name_bytes = raw_name.split(b"\0", 1)[0]
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise CompileCohortError("basedir/pak0.pak has a non-ASCII member name") from exc
        if (
            member_offset < 12
            or member_length < 0
            or member_offset + member_length > pak_size
        ):
            raise CompileCohortError(
                f"basedir/pak0.pak member {name!r} points outside the archive"
            )
        if name.replace("\\", "/").casefold() == REQUIRED_PAK_MEMBER:
            matches.append({
                "name": name,
                "offset": member_offset,
                "bytes": member_length,
                "sha256": _hash_file_region(
                    pak_path, member_offset, member_length
                ),
            })
    if len(matches) != 1:
        raise CompileCohortError(
            "basedir/pak0.pak must contain exactly one case-insensitive "
            f"{REQUIRED_PAK_MEMBER} member; found {len(matches)}"
        )
    return {
        "basedir": str(basedir),
        "pak0": {
            "path": str(pak_path),
            **_file_record(pak_path),
        },
        "required_member": matches[0],
    }


def _source_hashes(
    declaration: Mapping[str, Any], source_root: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        str(row["map"]): {
            suffix: _file_record(source_root / f"{row['map']}{suffix}")
            for suffix in SOURCE_SUFFIXES
        }
        for row in declaration["maps"]
    }


def _require_flat_regular_stage(directory: Path, label: str) -> None:
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise CompileCohortError(
                f"{label} contains a non-regular or nested entry: {path.name}"
            )


def _check_preserved_sources(
    declaration: Mapping[str, Any],
    directory: Path,
    expected: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    for row in declaration["maps"]:
        name = str(row["map"])
        for suffix in SOURCE_SUFFIXES:
            actual = _file_record(directory / f"{name}{suffix}")
            if actual != expected[name][suffix]:
                raise CompileCohortError(
                    f"compiler changed source artifact {name}{suffix}"
                )


def _log_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        **_file_record(path),
    }


def _base_report(paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "cohort_id": None,
        "status": "initializing-non-admissible-staging",
        "passed": False,
        "bundle_admissible": False,
        "atlas_admissible": False,
        "declaration": None,
        "inputs": {key: str(value) for key, value in paths.items()},
        "contract": {
            "declaration_order_required": True,
            "fail_fast": True,
            "source_suffixes": list(SOURCE_SUFFIXES),
            "compiled_suffixes": [*SOURCE_SUFFIXES, ".bsp"],
            "q2tool_flags": list(Q2TOOL_FLAGS),
            "q2tool_threads": 1,
            "atomic_publish": "renameat2(RENAME_NOREPLACE)",
            "inputs_rehashed_before_publish": True,
            "timeout_termination": "SIGKILL isolated q2tool process group",
            "successful_prt_handling": "hash-recorded then durably removed",
            "failure_intermediates_retained": True,
        },
        "q2tool": None,
        "assets": None,
        "source_membership": None,
        "source_hashes": None,
        "postcompile_membership": None,
        "maps": [],
        "publication": {
            "compiled_stage_published": False,
            "staging_non_admissible": True,
            "atomic": False,
        },
        "failure": None,
    }


def _write_failure_report(
    report_path: Path,
    report: dict[str, Any],
    *,
    phase: str,
    message: str,
    failed_map: Mapping[str, Any] | None,
) -> None:
    report["status"] = "failed-non-admissible-staging"
    report["passed"] = False
    report["bundle_admissible"] = False
    report["atlas_admissible"] = False
    report["publication"]["compiled_stage_published"] = False
    report["publication"]["atomic"] = False
    report["failure"] = {
        "phase": phase,
        "message": message,
        "failed_map": dict(failed_map) if failed_map is not None else None,
    }
    _exclusive_write(report_path, canonical_bytes(report))


def compile_generated_cohort(
    declaration_path: Path,
    source_root: Path,
    staging_root: Path,
    publish_root: Path,
    log_root: Path,
    report_path: Path,
    q2tool: Path,
    basedir: Path,
    *,
    timeout_seconds: float = DEFAULT_MAP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Compile and atomically publish exactly one canonical declaration."""
    paths = {
        "declaration": _absolute(declaration_path),
        "source_root": _absolute(source_root),
        "staging_root": _absolute(staging_root),
        "publish_root": _absolute(publish_root),
        "log_root": _absolute(log_root),
        "report": _absolute(report_path),
        "q2tool": _absolute(q2tool),
        "basedir": _absolute(basedir),
    }
    try:
        declaration, declaration_sha256 = load_declaration(paths["declaration"])
        require_unretired_declaration(
            paths["declaration"], declaration, declaration_sha256
        )
    except GeneratorCohortError as exc:
        raise CompileCohortError(str(exc)) from exc
    report = _base_report(paths)
    phase = "preflight"
    failed_map: Mapping[str, Any] | None = None
    published = False
    report_destination = paths["report"]

    if report_destination.exists():
        raise CompileCohortError(f"report destination already exists: {report_destination}")
    for key in ("source_root", "staging_root", "publish_root", "log_root"):
        if _paths_overlap(report_destination, paths[key]):
            raise CompileCohortError(f"report must be outside {key}")
    if report_destination.parent.exists() and not report_destination.parent.is_dir():
        raise CompileCohortError(
            f"report parent is not a directory: {report_destination.parent}"
        )
    if not report_destination.parent.exists():
        report_destination.parent.mkdir(parents=True)
    try:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
            or timeout_seconds > 86400
        ):
            raise CompileCohortError(
                "timeout_seconds must be finite and in the range (0, 86400]"
            )
        timeout_seconds = float(timeout_seconds)
        report["contract"]["per_map_timeout_seconds"] = timeout_seconds
        report["cohort_id"] = declaration["cohort_id"]
        report["declaration"] = {
            "path": str(paths["declaration"]),
            "sha256": declaration_sha256,
        }

        directories = {
            key: paths[key]
            for key in ("source_root", "staging_root", "publish_root", "log_root")
        }
        for first_name, first in directories.items():
            for second_name, second in directories.items():
                if first_name < second_name and _paths_overlap(first, second):
                    raise CompileCohortError(
                        f"{first_name} and {second_name} must be disjoint"
                    )
        if paths["source_root"].is_symlink() or not paths["source_root"].is_dir():
            raise CompileCohortError("source_root must be an existing non-symlink directory")
        for key in ("staging_root", "publish_root", "log_root"):
            if paths[key].exists():
                raise CompileCohortError(f"{key} must not already exist: {paths[key]}")
            if not paths[key].parent.is_dir():
                raise CompileCohortError(
                    f"{key} parent directory does not exist: {paths[key].parent}"
                )
        if (
            paths["staging_root"].parent.stat().st_dev
            != paths["publish_root"].parent.stat().st_dev
        ):
            raise CompileCohortError(
                "staging_root and publish_root must be on one filesystem"
            )

        _require_regular_file(paths["q2tool"], "q2tool")
        if not os.access(paths["q2tool"], os.X_OK):
            raise CompileCohortError(f"q2tool is not executable: {paths['q2tool']}")
        report["q2tool"] = {
            "path": str(paths["q2tool"]),
            **_file_record(paths["q2tool"]),
        }
        report["assets"] = inspect_basedir(paths["basedir"])

        phase = "source-membership"
        source_membership = verify_stage_membership(
            declaration, paths["source_root"], "source"
        )
        report["source_membership"] = source_membership
        if source_membership["passed"] is not True:
            raise CompileCohortError(
                "source membership differs: "
                + "; ".join(source_membership["failures"])
            )
        _require_flat_regular_stage(paths["source_root"], "source_root")
        source_hashes = _source_hashes(declaration, paths["source_root"])
        report["source_hashes"] = source_hashes

        phase = "staging"
        paths["staging_root"].mkdir(mode=0o755)
        paths["log_root"].mkdir(mode=0o755)
        for row in declaration["maps"]:
            for suffix in SOURCE_SUFFIXES:
                name = f"{row['map']}{suffix}"
                shutil.copyfile(paths["source_root"] / name, paths["staging_root"] / name)
        staged_source = verify_stage_membership(
            declaration, paths["staging_root"], "source"
        )
        if staged_source["passed"] is not True:
            raise CompileCohortError(
                "staged source membership differs: "
                + "; ".join(staged_source["failures"])
            )
        _require_flat_regular_stage(paths["staging_root"], "staging_root")
        _check_preserved_sources(declaration, paths["staging_root"], source_hashes)

        phase = "compile"
        for row in declaration["maps"]:
            failed_map = {
                "ordinal": row["ordinal"],
                "map": row["map"],
                "seed": row["seed"],
                "style": row["style"],
            }
            ordinal = int(row["ordinal"])
            map_id = str(row["map"])
            map_path = paths["staging_root"] / f"{map_id}.map"
            bsp_path = paths["staging_root"] / f"{map_id}.bsp"
            prt_path = paths["staging_root"] / f"{map_id}.prt"
            if bsp_path.exists():
                raise CompileCohortError(f"staged BSP exists before compile: {bsp_path}")
            stdout_path = paths["log_root"] / f"{ordinal:03d}-{map_id}.stdout.log"
            stderr_path = paths["log_root"] / f"{ordinal:03d}-{map_id}.stderr.log"
            command = [
                str(paths["q2tool"]),
                *Q2TOOL_FLAGS,
                str(paths["basedir"]),
                str(map_path),
            ]
            invocation_error: str | None = None
            exit_code: int | None = None
            timed_out = False
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=paths["staging_root"],
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        start_new_session=True,
                    )
                    try:
                        exit_code = process.wait(timeout=timeout_seconds)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        invocation_error = (
                            "TimeoutExpired: exceeded per-map timeout of "
                            f"{timeout_seconds:g} seconds"
                        )
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        exit_code = process.wait()
                        stderr.write(
                            (
                                "\ncompile orchestrator: "
                                f"{invocation_error}; process group terminated\n"
                            ).encode("ascii")
                        )
                        stderr.flush()
                except OSError as exc:
                    invocation_error = f"{exc.__class__.__name__}: {exc}"
            bsp: dict[str, Any] | None = None
            bsp_error: str | None = None
            if bsp_path.exists() or bsp_path.is_symlink():
                try:
                    bsp = _file_record(bsp_path)
                except CompileCohortError as exc:
                    bsp_error = str(exc)
            prt: dict[str, Any] | None = None
            prt_error: str | None = None
            if prt_path.exists() or prt_path.is_symlink():
                try:
                    prt = _file_record(prt_path)
                except CompileCohortError as exc:
                    prt_error = str(exc)
            result: dict[str, Any] = {
                **failed_map,
                "command": command,
                "exit_code": exit_code,
                "invocation_error": invocation_error,
                "timed_out": timed_out,
                "stdout_log": _log_record(stdout_path, paths["log_root"]),
                "stderr_log": _log_record(stderr_path, paths["log_root"]),
                "bsp": bsp,
                "bsp_error": bsp_error,
                "prt": prt,
                "prt_error": prt_error,
                "prt_removed_after_success": False,
                "passed": False,
            }
            report["maps"].append(result)
            if timed_out:
                raise CompileCohortError(
                    f"q2tool timed out for {map_id} after "
                    f"{timeout_seconds:g} seconds"
                )
            if invocation_error is not None:
                raise CompileCohortError(
                    f"q2tool invocation failed for {map_id}: {invocation_error}"
                )
            if exit_code != 0:
                raise CompileCohortError(
                    f"q2tool failed for {map_id} with exit code {exit_code}"
                )
            if bsp_error is not None:
                raise CompileCohortError(
                    f"q2tool emitted an invalid BSP for {map_id}: {bsp_error}"
                )
            if result["bsp"] is None:
                raise CompileCohortError(
                    f"q2tool reported success but emitted no BSP for {map_id}"
                )
            if prt_error is not None:
                raise CompileCohortError(
                    f"q2tool emitted an invalid PRT for {map_id}: {prt_error}"
                )
            if prt is not None:
                prt_path.unlink()
                _fsync_directory(paths["staging_root"])
                if prt_path.exists() or prt_path.is_symlink():
                    raise CompileCohortError(
                        f"q2tool PRT cleanup did not remove {prt_path}"
                    )
                result["prt_removed_after_success"] = True
            _check_preserved_sources(
                declaration, paths["staging_root"], source_hashes
            )
            result["passed"] = True
            failed_map = None

        phase = "postcompile-membership"
        postcompile = verify_stage_membership(
            declaration, paths["staging_root"], "compiled"
        )
        report["postcompile_membership"] = postcompile
        if postcompile["passed"] is not True:
            raise CompileCohortError(
                "postcompile membership differs: "
                + "; ".join(postcompile["failures"])
            )
        _require_flat_regular_stage(paths["staging_root"], "staging_root")
        _check_preserved_sources(declaration, paths["staging_root"], source_hashes)
        if len(report["maps"]) != len(declaration["maps"]):
            raise CompileCohortError("compile result count differs from declaration")
        if not all(row["passed"] is True for row in report["maps"]):
            raise CompileCohortError("one or more compile results did not pass")
        phase = "input-stability"
        if _source_hashes(declaration, paths["source_root"]) != source_hashes:
            raise CompileCohortError("source_root changed during compilation")
        try:
            stable_declaration, stable_declaration_sha256 = load_declaration(
                paths["declaration"]
            )
        except GeneratorCohortError as exc:
            raise CompileCohortError(
                "declaration changed or became invalid during compilation"
            ) from exc
        if (
            stable_declaration_sha256 != declaration_sha256
            or stable_declaration != declaration
        ):
            raise CompileCohortError("declaration changed during compilation")
        if {
            "path": str(paths["q2tool"]),
            **_file_record(paths["q2tool"]),
        } != report["q2tool"]:
            raise CompileCohortError("q2tool changed during compilation")
        if inspect_basedir(paths["basedir"]) != report["assets"]:
            raise CompileCohortError("basedir assets changed during compilation")

        phase = "publish"
        report["status"] = "compiled-stage-published-non-admissible"
        report["passed"] = True
        report["failure"] = None
        report["publication"] = {
            "compiled_stage_published": True,
            "staging_non_admissible": True,
            "atomic": True,
        }
        _rename_noreplace(paths["staging_root"], paths["publish_root"])
        published = True
        phase = "report-publication"
        try:
            _exclusive_write(report_destination, canonical_bytes(report))
        except Exception:
            _rename_noreplace(paths["publish_root"], paths["staging_root"])
            published = False
            raise
        return report
    except Exception as exc:
        if published:
            try:
                _rename_noreplace(paths["publish_root"], paths["staging_root"])
                published = False
            except Exception as rollback_error:
                raise CompileCohortError(
                    f"{exc}; atomic publish rollback also failed: {rollback_error}"
                ) from exc
        message = str(exc)
        if not report_destination.exists():
            _write_failure_report(
                report_destination,
                report,
                phase=phase,
                message=message,
                failed_map=failed_map,
            )
        if isinstance(exc, CompileCohortError):
            raise
        if isinstance(exc, GeneratorCohortError):
            raise CompileCohortError(message) from exc
        raise CompileCohortError(message) from exc


def _summary(report: Mapping[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "cohort_id": report["cohort_id"],
        "status": report["status"],
        "map_count": len(report["maps"]),
        "report_sha256": file_sha256(report_path),
        "passed": report["passed"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--publish-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--q2tool", type=Path, required=True)
    parser.add_argument("--basedir", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_MAP_TIMEOUT_SECONDS,
        help="per-map q2tool timeout; must be in (0, 86400]",
    )
    args = parser.parse_args(argv)
    try:
        report = compile_generated_cohort(
            args.declaration,
            args.source_root,
            args.staging_root,
            args.publish_root,
            args.log_root,
            args.report,
            args.q2tool,
            args.basedir,
            timeout_seconds=args.timeout_seconds,
        )
    except CompileCohortError as exc:
        print(f"generated cohort compile failed: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_bytes(_summary(report, args.report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
