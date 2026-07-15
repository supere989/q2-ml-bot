#!/usr/bin/env python3
"""Build one exact generated Atlas cohort and publish it atomically.

The declared claims stage is the only input membership authority.  Every map
is built by a separate ``build_map_atlas.py`` process into private scratch
space.  The per-process summary and logs are retained in a separate diagnostics
root, while only the eight declared Atlas artifacts enter the unpublished
analysis root.  No analysis root is renamed into place unless all 28 builds and
the exact 224-file membership check pass.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    HEX_RE,
    GIT_HEX_RE,
    STAGE_SUFFIXES,
    canonical_bytes,
    file_sha256,
    load_declaration,
    repository_binding,
    verify_stage_membership,
)
from tools.retired_cohort_registry import require_unretired_declaration  # noqa: E402
from harness.atlas_source_closure import (  # noqa: E402
    atlas_analyzer_authority_inputs,
    atlas_analyzer_authority_sha256,
)


CAMPAIGN_SCHEMA = "q2-generated-atlas-build-campaign-v1"
BUILD_SUMMARY_SCHEMA = "q2-atlas-generated-build-v1"
BUILD_SUMMARY_NAME = "generated-build-summary.json"
ANALYSIS_MANIFEST_SCHEMA = "q2-atlas-analysis-v1"
FULL_COLD_SCHEMA = "q2-atlas-full-cold-proof-v1"
RENAME_NOREPLACE = 1
AT_FDCWD = -100
DEFAULT_CLIENT = Path("/home/raymondj/multires-worktrees/integration/q2-ml-client")
DEFAULT_LITHIUM = Path(
    "/home/raymondj/multires-worktrees/integration/q2-lithium-3zb2"
)
DEFAULT_ATTESTATION = Path(
    "/home/raymondj/multires-artifacts/atlas-v1/B1/"
    "hook-parity-pullspeed-1700.json"
)


class GeneratedAtlasCampaignError(ValueError):
    """Raised when an exact generated Atlas campaign cannot be published."""


@dataclass(frozen=True)
class BuildProcessResult:
    """Captured result of exactly one Atlas builder invocation."""

    returncode: int
    stdout: bytes
    stderr: bytes


Builder = Callable[[Mapping[str, Any], Path, Path, Path], BuildProcessResult]


@dataclass
class ReportReservation:
    """An exclusively created report inode retained across the transaction."""

    path: Path
    descriptor: int
    device: int
    inode: int

    def _require_owned_path(self) -> None:
        try:
            current = self.path.lstat()
        except OSError as exc:
            raise GeneratedAtlasCampaignError(
                "reserved campaign report path disappeared"
            ) from exc
        if (
            stat.S_ISLNK(current.st_mode)
            or current.st_dev != self.device
            or current.st_ino != self.inode
        ):
            raise GeneratedAtlasCampaignError(
                "reserved campaign report path was replaced"
            )

    def write(self, report: Mapping[str, Any]) -> None:
        payload = canonical_bytes(report)
        self._require_owned_path()
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        os.ftruncate(self.descriptor, 0)
        view = memoryview(payload)
        while view:
            written = os.write(self.descriptor, view)
            if written <= 0:
                raise OSError("campaign report write made no progress")
            view = view[written:]
        os.fsync(self.descriptor)
        self._require_owned_path()
        _fsync_directory(self.path.parent)

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def _reject_symlink_components(path: Path, label: str) -> None:
    """Reject symlinks in every existing component of an authority path."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if not current.exists() and not current.is_symlink():
            continue
        if current.is_symlink():
            raise GeneratedAtlasCampaignError(
                f"{label} path contains symlink component {current}"
            )


def _reserve_report(path: Path) -> ReportReservation:
    _reject_symlink_components(path.parent, "campaign report")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent, "campaign report")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
    )
    identity = os.fstat(descriptor)
    _fsync_directory(path.parent)
    return ReportReservation(
        path=path,
        descriptor=descriptor,
        device=identity.st_dev,
        inode=identity.st_ino,
    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename on Linux while refusing every existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise GeneratedAtlasCampaignError(
            "Linux renameat2(RENAME_NOREPLACE) is unavailable"
        )
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
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise GeneratedAtlasCampaignError(
            "analysis stage appeared before atomic publication; refusing overwrite"
        )
    if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise GeneratedAtlasCampaignError(
            "filesystem lacks renameat2(RENAME_NOREPLACE) authority"
        )
    raise OSError(error, os.strerror(error), str(destination))


def _canonical_error(exc: Exception) -> str:
    return " ".join(str(exc).replace("\n", " ").split())[:4096]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_canonical(value: object) -> str:
    return _sha256_bytes(canonical_bytes(value))


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _require_unpublished(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise GeneratedAtlasCampaignError(
            f"{label} already exists; refusing overwrite"
        )


def _require_plain_root(path: Path, label: str) -> None:
    if path.is_symlink():
        raise GeneratedAtlasCampaignError(f"{label} must not be a symlink")


def _require_separate_non_nested(
    roots: tuple[tuple[str, Path], ...], report_path: Path
) -> None:
    for index, (first_label, first) in enumerate(roots):
        _require_plain_root(first, first_label)
        for second_label, second in roots[index + 1 :]:
            if (
                first.resolve() == second.resolve()
                or _path_is_within(first, second)
                or _path_is_within(second, first)
            ):
                raise GeneratedAtlasCampaignError(
                    f"{first_label} and {second_label} must be separate "
                    "non-nested roots"
                )
    _require_plain_root(report_path, "campaign report")
    if any(
        _path_is_within(report_path, root)
        or _path_is_within(root, report_path)
        for _label, root in roots
    ):
        raise GeneratedAtlasCampaignError(
            "campaign report must be outside all exact and diagnostic roots"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _exclusive_copy(source: Path, destination: Path) -> None:
    descriptor = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
    )
    try:
        with source.open("rb") as input_stream, os.fdopen(
            descriptor, "wb"
        ) as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def _expected_stage_names(
    declaration: Mapping[str, Any], stage: str
) -> set[str]:
    return {
        f"{row['map']}{suffix}"
        for row in declaration["maps"]
        for suffix in STAGE_SUFFIXES[stage]
    }


def _open_exact_flat_root(
    path: Path, expected_names: set[str], label: str
) -> tuple[int, tuple[int, int]]:
    """Pin and validate a flat exact root without following a root swap."""

    _reject_symlink_components(path, label)
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
    except OSError as exc:
        raise GeneratedAtlasCampaignError(f"{label} is not a plain directory") from exc
    identity = os.fstat(descriptor)
    names = set(os.listdir(descriptor))
    missing = sorted(expected_names - names)
    unexpected = sorted(names - expected_names)
    failures = [f"missing file {name}" for name in missing]
    failures.extend(f"unexpected entry {name}" for name in unexpected)
    for name in sorted(names & expected_names):
        item = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(item.st_mode):
            failures.append(f"non-regular or nested entry {name}")
    if failures:
        os.close(descriptor)
        raise GeneratedAtlasCampaignError(
            f"{label} exact-root policy failed: " + "; ".join(failures)
        )
    return descriptor, (identity.st_dev, identity.st_ino)


def _require_root_identity(
    path: Path, identity: tuple[int, int], label: str
) -> None:
    try:
        current = path.lstat()
    except OSError as exc:
        raise GeneratedAtlasCampaignError(f"{label} root disappeared") from exc
    if (
        not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
    ):
        raise GeneratedAtlasCampaignError(f"{label} root was replaced")


def _copy_claims_snapshot(
    declaration: Mapping[str, Any], claims_dir: Path, destination: Path
) -> dict[str, Any]:
    """Copy an exact claims root through a pinned directory FD, then seal it."""

    expected = _expected_stage_names(declaration, "claims")
    descriptor, identity = _open_exact_flat_root(
        claims_dir, expected, "claims stage"
    )
    destination.mkdir(mode=0o700)
    try:
        for name in sorted(expected):
            source_descriptor = os.open(
                name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                source_identity = os.fstat(source_descriptor)
                if not stat.S_ISREG(source_identity.st_mode):
                    raise GeneratedAtlasCampaignError(
                        f"claims stage member changed type: {name}"
                    )
                destination_path = destination / name
                output_descriptor = os.open(
                    destination_path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o400,
                )
                with os.fdopen(output_descriptor, "wb") as output_stream:
                    while True:
                        block = os.read(source_descriptor, 1024 * 1024)
                        if not block:
                            break
                        output_stream.write(block)
                    output_stream.flush()
                    os.fsync(output_stream.fileno())
            finally:
                os.close(source_descriptor)
        if set(os.listdir(descriptor)) != expected:
            raise GeneratedAtlasCampaignError(
                "claims stage membership changed during immutable snapshot"
            )
        _require_root_identity(claims_dir, identity, "claims stage")
    finally:
        os.close(descriptor)
    _fsync_directory(destination)
    membership = verify_stage_membership(declaration, destination, "claims")
    if not membership["passed"]:
        raise GeneratedAtlasCampaignError(
            "immutable claims snapshot failed exact membership"
        )
    for path in destination.iterdir():
        path.chmod(0o400)
    destination.chmod(0o500)
    return membership


def _git_output(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _snapshot_repository(
    repo_root: Path, destination: Path, binding: Mapping[str, Any]
) -> None:
    """Extract the bound committed tree and make it the sole execution root."""

    if repo_root.resolve() != ROOT.resolve():
        raise GeneratedAtlasCampaignError(
            "repo_root must be the canonical q2-ml-bot ROOT"
        )
    if (
        _git_output(repo_root, "rev-parse", "HEAD")
        != binding["repository_commit"]
        or _git_output(repo_root, "rev-parse", "HEAD^{tree}")
        != binding["repository_tree"]
    ):
        raise GeneratedAtlasCampaignError(
            "implementation binding differs from the current committed tree"
        )
    completed = subprocess.run(
        ["git", "archive", "--format=tar", binding["repository_commit"]],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise GeneratedAtlasCampaignError(
            "cannot create committed implementation execution snapshot: "
            + completed.stderr.decode("utf-8", "replace").strip()
        )
    destination.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = Path(member.name)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or not (member.isfile() or member.isdir())
            ):
                raise GeneratedAtlasCampaignError(
                    "committed implementation archive contains unsafe entry"
                )
        archive.extractall(destination, members=members)
    if (
        atlas_analyzer_authority_sha256(destination)
        != binding["atlas_analyzer_authority_sha256"]
        or len(atlas_analyzer_authority_inputs(destination))
        != binding["atlas_analyzer_authority_file_count"]
        or file_sha256(destination / "maps/generator.py")
        != binding["generator_sha256"]
        or file_sha256(destination / "maps/routes.py") != binding["routes_sha256"]
    ):
        raise GeneratedAtlasCampaignError(
            "committed implementation snapshot differs from bound authority"
        )
    if not (destination / "tools/build_map_atlas.py").is_file():
        raise GeneratedAtlasCampaignError(
            "committed implementation snapshot lacks build_map_atlas.py"
        )
    for path in sorted(destination.rglob("*"), reverse=True):
        path.chmod(0o500 if path.is_dir() else 0o400)
    destination.chmod(0o500)


def _membership_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stage": report["stage"],
        "passed": report["passed"],
        "expected_map_count": report["expected_map_count"],
        "expected_file_count": report["expected_file_count"],
        "actual_file_count": report["actual_file_count"],
        "report_sha256": _sha256_canonical(report),
    }


def _validate_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(value)
    expected = {
        "repository_commit",
        "repository_tree",
        "git_clean",
        "atlas_analyzer_authority_sha256",
        "atlas_analyzer_authority_file_count",
        "generator_sha256",
        "routes_sha256",
    }
    if set(binding) != expected or binding.get("git_clean") is not True:
        raise GeneratedAtlasCampaignError(
            "implementation binding is incomplete or not clean"
        )
    for field in ("repository_commit", "repository_tree"):
        if not isinstance(binding[field], str) or not GIT_HEX_RE.fullmatch(
            binding[field]
        ):
            raise GeneratedAtlasCampaignError(
                f"implementation binding {field} is malformed"
            )
    for field in (
        "atlas_analyzer_authority_sha256",
        "generator_sha256",
        "routes_sha256",
    ):
        if not isinstance(binding[field], str) or not HEX_RE.fullmatch(
            binding[field]
        ):
            raise GeneratedAtlasCampaignError(
                f"implementation binding {field} is malformed"
            )
    count = binding["atlas_analyzer_authority_file_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise GeneratedAtlasCampaignError(
            "implementation binding analyzer authority file count is malformed"
        )
    return binding


def _default_builder_factory(
    *,
    execution_root: Path,
    client_root: Path,
    lithium_root: Path,
    hook_attestation: Path,
    fall_oracle: Path | None,
    packer: Path | None,
    verifier: Path | None,
) -> Builder:
    def build(
        declared: Mapping[str, Any],
        claims_dir: Path,
        output_dir: Path,
        supplied_execution_root: Path,
    ) -> BuildProcessResult:
        if supplied_execution_root != execution_root:
            raise GeneratedAtlasCampaignError(
                "builder execution root differs from committed snapshot"
            )
        name = declared["map"]
        command = [
            sys.executable,
            str(execution_root / "tools/build_map_atlas.py"),
            "--bsp",
            str(claims_dir / f"{name}.bsp"),
            "--map-id",
            name,
            "--generator-claims",
            str(claims_dir / f"{name}.generator-claims.json"),
            "--output",
            str(output_dir),
            "--client-root",
            str(client_root),
            "--lithium-root",
            str(lithium_root),
            "--hook-attestation",
            str(hook_attestation),
        ]
        for option, path in (
            ("--fall-oracle", fall_oracle),
            ("--packer", packer),
            ("--verifier", verifier),
        ):
            if path is not None:
                command.extend((option, str(path)))
        completed = subprocess.run(
            command,
            cwd=execution_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return BuildProcessResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    return build


def _regular_files_exact(directory: Path) -> tuple[set[str], list[str]]:
    files: set[str] = set()
    failures: list[str] = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            failures.append(f"builder output contains symlink {relative}")
        elif path.is_dir():
            failures.append(f"builder output contains nested directory {relative}")
        elif path.is_file():
            files.add(relative)
        else:
            failures.append(f"builder output contains non-regular entry {relative}")
    return files, failures


def _load_analysis_binding(
    output_dir: Path,
    declared: Mapping[str, Any],
    claims_dir: Path,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and bind the canonical passing full-cold analysis manifest."""

    name = declared["map"]
    manifest_path = output_dir / f"{name}.analysis.manifest.json"
    atlas_manifest_path = output_dir / f"{name}.atlas.manifest.json"
    atlas_path = output_dir / f"{name}.atlas.bin"
    try:
        manifest_payload = manifest_path.read_bytes()
        manifest = json.loads(manifest_payload)
        atlas_manifest_payload = atlas_manifest_path.read_bytes()
        atlas_manifest = json.loads(atlas_manifest_payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeneratedAtlasCampaignError(
            f"analysis manifest is unreadable: {_canonical_error(exc)}"
        ) from exc
    if manifest_payload != canonical_bytes(manifest):
        raise GeneratedAtlasCampaignError(
            "analysis manifest is not canonical compact sorted JSON"
        )
    compact_atlas_manifest = (
        json.dumps(
            atlas_manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=False,
        )
        + "\n"
    ).encode("ascii")
    if atlas_manifest_payload != compact_atlas_manifest:
        raise GeneratedAtlasCampaignError(
            "Atlas manifest is not canonical compact JSON"
        )

    bsp_sha256 = file_sha256(claims_dir / f"{name}.bsp")
    claims_sha256 = file_sha256(
        claims_dir / f"{name}.generator-claims.json"
    )
    atlas_sha256 = file_sha256(atlas_path)
    atlas_manifest_sha256 = file_sha256(atlas_manifest_path)
    try:
        identity = manifest["identity"]
        full_cold = manifest["performance"]["full_cold_rebuild"]
        atlas_artifact = manifest["artifacts"]["atlas"]
        atlas_manifest_artifact = manifest["artifacts"]["atlas_manifest"]
    except (KeyError, TypeError) as exc:
        raise GeneratedAtlasCampaignError(
            "analysis manifest lacks required identity or full-cold proof"
        ) from exc
    required_analysis = {
        "schema": ANALYSIS_MANIFEST_SCHEMA,
        "status": "passed",
        "deterministic_rebuild": True,
        "confidence": "high",
        "canonical_map_id": name,
        "bsp_sha256": bsp_sha256,
        "generator_claims_sha256": claims_sha256,
        "atlas_sha256": atlas_sha256,
        "analyzer_sha256": binding["atlas_analyzer_authority_sha256"],
        "atlas_manifest_sha256": atlas_manifest_sha256,
    }
    manifest_bsp = manifest.get("bsp")
    if not isinstance(manifest_bsp, Mapping) or not isinstance(identity, Mapping):
        raise GeneratedAtlasCampaignError(
            "analysis manifest BSP or identity record is malformed"
        )
    actual_analysis = {
        "schema": manifest.get("schema"),
        "status": manifest.get("status"),
        "deterministic_rebuild": manifest.get("deterministic_rebuild"),
        "confidence": manifest.get("confidence"),
        "canonical_map_id": manifest.get("canonical_map_id"),
        "bsp_sha256": manifest_bsp.get("sha256"),
        "generator_claims_sha256": manifest.get("generator_claims_sha256"),
        "atlas_sha256": identity.get("atlas_sha256"),
        "analyzer_sha256": identity.get("analyzer_sha256"),
        "atlas_manifest_sha256": identity.get("atlas_manifest_sha256"),
    }
    if actual_analysis != required_analysis or (
        identity.get("bsp_sha256") != bsp_sha256
        or identity.get("generator_claims_sha256") != claims_sha256
        or atlas_artifact.get("uncompressed_sha256") != atlas_sha256
        or atlas_manifest_artifact.get("sha256") != atlas_manifest_sha256
    ):
        raise GeneratedAtlasCampaignError(
            "analysis manifest identity/status differs from exact campaign authority"
        )

    expected_cold_artifacts = {
        ".atlas.bin",
        ".atlas.bin.zst",
        ".navigation.bin.zst",
        ".visibility.bin.zst",
        ".design-signature.json",
        ".routes.json",
    }
    expected_semantic = {
        ".analysis.manifest.json",
        ".atlas.manifest.json",
    }
    try:
        cold_artifacts = full_cold["artifact_sha256"]
        rebuilt_artifacts = full_cold["cold_artifact_sha256"]
        semantic_artifacts = full_cold["artifact_semantic_sha256"]
        cold_semantic_artifacts = full_cold["cold_artifact_semantic_sha256"]
        verification = full_cold["verification"]
    except (KeyError, TypeError) as exc:
        raise GeneratedAtlasCampaignError(
            "full-cold proof lacks required digest sets"
        ) from exc
    if not all(
        isinstance(value, Mapping)
        for value in (
            cold_artifacts,
            rebuilt_artifacts,
            semantic_artifacts,
            cold_semantic_artifacts,
        )
    ):
        raise GeneratedAtlasCampaignError(
            "full-cold proof digest sets are malformed"
        )
    if (
        full_cold.get("schema") != FULL_COLD_SCHEMA
        or full_cold.get("independent_process_launches") != 1
        or full_cold.get("artifact_count") != 8
        or full_cold.get("timeout_limit_milliseconds") != 300_000
        or isinstance(full_cold.get("elapsed_milliseconds"), bool)
        or not isinstance(full_cold.get("elapsed_milliseconds"), int)
        or not 1 <= full_cold["elapsed_milliseconds"] <= 300_000
        or set(cold_artifacts) != expected_cold_artifacts
        or set(rebuilt_artifacts) != expected_cold_artifacts
        or set(semantic_artifacts) != expected_semantic
        or set(cold_semantic_artifacts) != expected_semantic
    ):
        raise GeneratedAtlasCampaignError(
            "full-cold proof contract or exact digest membership differs"
        )
    for suffix in expected_cold_artifacts:
        actual_digest = file_sha256(output_dir / f"{name}{suffix}")
        if (
            cold_artifacts[suffix] != actual_digest
            or rebuilt_artifacts[suffix] != actual_digest
        ):
            raise GeneratedAtlasCampaignError(
                f"full-cold proof digest differs for {suffix}"
            )
    if any(
        not isinstance(digest, str) or not HEX_RE.fullmatch(digest)
        for digest in (
            *semantic_artifacts.values(),
            *cold_semantic_artifacts.values(),
        )
    ) or semantic_artifacts != cold_semantic_artifacts:
        raise GeneratedAtlasCampaignError(
            "full-cold semantic digest proof is malformed or asymmetric"
        )
    if (
        not isinstance(verification, Mapping)
        or verification.get("passed") is not True
        or verification.get("canonical_map_id") != name
        or verification.get("bsp_sha256") != bsp_sha256
        or verification.get("atlas_sha256") != atlas_sha256
    ):
        raise GeneratedAtlasCampaignError(
            "full-cold verifier binding differs from exact artifacts"
        )

    try:
        atlas_manifest_map = atlas_manifest["bsp"]
        atlas_manifest_analyzer = atlas_manifest["analyzer"]
        atlas_manifest_generator = atlas_manifest["generator"]
        atlas_manifest_atlas = atlas_manifest["artifacts"][f"{name}.atlas.bin"]
    except (KeyError, TypeError) as exc:
        raise GeneratedAtlasCampaignError(
            "Atlas manifest lacks required generated-map identities"
        ) from exc
    if (
        atlas_manifest_map.get("canonical_map_id") != name
        or atlas_manifest_map.get("sha256") != bsp_sha256
        or atlas_manifest_analyzer.get("sha256")
        != binding["atlas_analyzer_authority_sha256"]
        or not isinstance(atlas_manifest_generator, Mapping)
        or atlas_manifest_generator.get("sha256") != binding["generator_sha256"]
        or atlas_manifest_atlas.get("sha256_uncompressed") != atlas_sha256
        or atlas_manifest_atlas.get("uncompressed_size") != atlas_path.stat().st_size
    ):
        raise GeneratedAtlasCampaignError(
            "Atlas manifest differs from map/BSP/analyzer/Atlas authority"
        )
    return {
        "sha256": _sha256_bytes(manifest_payload),
        "atlas_manifest_sha256": atlas_manifest_sha256,
        "status": "passed",
        "deterministic_rebuild": True,
        "bsp_sha256": bsp_sha256,
        "generator_claims_sha256": claims_sha256,
        "analyzer_sha256": binding["atlas_analyzer_authority_sha256"],
        "atlas_sha256": atlas_sha256,
        "full_cold_proof_sha256": _sha256_canonical(full_cold),
    }


def _load_build_summary(
    output_dir: Path,
    declared: Mapping[str, Any],
    claims_dir: Path,
    stdout: bytes,
) -> tuple[dict[str, Any], bytes]:
    name = declared["map"]
    summary_path = output_dir / BUILD_SUMMARY_NAME
    try:
        payload = summary_path.read_bytes()
        summary = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeneratedAtlasCampaignError(
            f"builder summary is unreadable: {_canonical_error(exc)}"
        ) from exc
    if payload != canonical_bytes(summary):
        raise GeneratedAtlasCampaignError("builder summary is not canonical JSON")
    if set(summary) != {"schema", "maps"} or summary["schema"] != BUILD_SUMMARY_SCHEMA:
        raise GeneratedAtlasCampaignError("builder summary schema or keys differ")
    if not isinstance(summary["maps"], list) or len(summary["maps"]) != 1:
        raise GeneratedAtlasCampaignError("builder summary must contain exactly one map")
    row = summary["maps"][0]
    expected_keys = {
        "canonical_map_id",
        "bsp_sha256",
        "atlas_sha256",
        "manifest_sha256",
    }
    if not isinstance(row, Mapping) or set(row) != expected_keys:
        raise GeneratedAtlasCampaignError("builder summary map keys differ")
    try:
        stdout_summary = json.loads(stdout)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GeneratedAtlasCampaignError(
            f"builder stdout is not its JSON summary: {_canonical_error(exc)}"
        ) from exc
    if stdout_summary != summary:
        raise GeneratedAtlasCampaignError("builder stdout summary differs from file")
    manifest_path = output_dir / f"{name}.analysis.manifest.json"
    try:
        manifest = json.loads(manifest_path.read_bytes())
        atlas_sha256 = manifest["identity"]["atlas_sha256"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GeneratedAtlasCampaignError(
            f"analysis manifest identity is unreadable: {_canonical_error(exc)}"
        ) from exc
    expected = {
        "canonical_map_id": name,
        "bsp_sha256": file_sha256(claims_dir / f"{name}.bsp"),
        "atlas_sha256": atlas_sha256,
        "manifest_sha256": file_sha256(manifest_path),
    }
    if dict(row) != expected:
        raise GeneratedAtlasCampaignError(
            "builder summary identities differ from declared inputs or artifacts"
        )
    for field in ("bsp_sha256", "atlas_sha256", "manifest_sha256"):
        if not isinstance(row[field], str) or not HEX_RE.fullmatch(row[field]):
            raise GeneratedAtlasCampaignError(
                f"builder summary {field} is malformed"
            )
    return dict(summary), payload


def _inspect_map_build(
    output_dir: Path,
    declared: Mapping[str, Any],
    claims_dir: Path,
    result: BuildProcessResult,
    binding: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    if result.returncode != 0:
        raise GeneratedAtlasCampaignError(
            f"build_map_atlas exited {result.returncode}"
        )
    name = declared["map"]
    expected = {
        *(f"{name}{suffix}" for suffix in STAGE_SUFFIXES["analysis"]),
        BUILD_SUMMARY_NAME,
    }
    actual, failures = _regular_files_exact(output_dir)
    failures.extend(f"missing builder output {item}" for item in sorted(expected - actual))
    failures.extend(f"unexpected builder output {item}" for item in sorted(actual - expected))
    if failures:
        raise GeneratedAtlasCampaignError("; ".join(failures))
    summary, summary_payload = _load_build_summary(
        output_dir, declared, claims_dir, result.stdout
    )
    analysis_binding = _load_analysis_binding(
        output_dir, declared, claims_dir, binding
    )
    artifacts = {
        suffix: {
            "bytes": (output_dir / f"{name}{suffix}").stat().st_size,
            "sha256": file_sha256(output_dir / f"{name}{suffix}"),
        }
        for suffix in STAGE_SUFFIXES["analysis"]
    }
    return summary, summary_payload, artifacts, analysis_binding


def _empty_row(declared: Mapping[str, Any], error: str) -> dict[str, Any]:
    return {
        "ordinal": declared["ordinal"],
        "map": declared["map"],
        "passed": False,
        "bsp_sha256": None,
        "generator_claims_sha256": None,
        "artifacts": {},
        "analysis_manifest": None,
        "build_summary": None,
        "build_summary_sha256": None,
        "stdout": {"bytes": 0, "sha256": _sha256_bytes(b"")},
        "stderr": {"bytes": 0, "sha256": _sha256_bytes(b"")},
        "error": error,
    }


def _campaign_report(
    declaration: Mapping[str, Any],
    declaration_sha256: str,
    binding: Mapping[str, Any],
    claims_membership: Mapping[str, Any],
    analysis_membership: Mapping[str, Any],
    rows: list[dict[str, Any]],
    failures: list[str],
    *,
    published: bool,
) -> dict[str, Any]:
    passed = (
        published
        and not failures
        and claims_membership["passed"] is True
        and analysis_membership["passed"] is True
        and len(rows) == len(declaration["maps"])
        and all(row["passed"] is True for row in rows)
    )
    return {
        "schema": CAMPAIGN_SCHEMA,
        "cohort_id": declaration["cohort_id"],
        "declaration_sha256": declaration_sha256,
        "implementation": dict(binding),
        "claims_snapshot_sha256": _sha256_canonical(claims_membership),
        "input_claims": _membership_projection(claims_membership),
        "output_analysis": _membership_projection(analysis_membership),
        "expected_map_count": len(declaration["maps"]),
        "pass_count": sum(row["passed"] is True for row in rows),
        "published": published,
        "passed": passed,
        "maps": rows,
        "failures": sorted(set(failures)),
    }


def _quarantine_owned_root(path: Path) -> Path | None:
    """Move a just-published owned root out of selection without overwrite."""

    for ordinal in range(1000):
        candidate = path.parent / f".{path.name}.quarantine-{os.getpid()}-{ordinal}"
        try:
            _rename_noreplace(path, candidate)
            _fsync_directory(path.parent)
            return candidate
        except GeneratedAtlasCampaignError as exc:
            if "appeared before atomic publication" not in str(exc):
                return None
        except OSError:
            return None
    return None


def _remove_work_root(path: Path) -> None:
    """Remove our read-only snapshots without touching any external root."""

    if not path.exists():
        return
    for item in [path, *path.rglob("*")]:
        try:
            item.chmod(0o700 if item.is_dir() else 0o600)
        except OSError:
            pass
    shutil.rmtree(path, ignore_errors=True)


def build_atlas_campaign(
    declaration_path: Path,
    claims_dir: Path,
    analysis_dir: Path,
    diagnostics_dir: Path,
    report_path: Path,
    *,
    repo_root: Path = ROOT,
    client_root: Path = DEFAULT_CLIENT,
    lithium_root: Path = DEFAULT_LITHIUM,
    hook_attestation: Path = DEFAULT_ATTESTATION,
    fall_oracle: Path | None = None,
    packer: Path | None = None,
    verifier: Path | None = None,
    _builder: Builder | None = None,
    _binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and atomically publish the declaration's exact analysis stage."""

    if repo_root.resolve() != ROOT.resolve():
        raise GeneratedAtlasCampaignError(
            "repo_root must be the canonical q2-ml-bot ROOT"
        )
    declaration, declaration_sha256 = load_declaration(declaration_path)
    require_unretired_declaration(
        declaration_path, declaration, declaration_sha256
    )
    roots = (
        ("claims stage", claims_dir),
        ("analysis stage", analysis_dir),
        ("build diagnostics", diagnostics_dir),
    )
    _require_separate_non_nested(roots, report_path)
    _require_unpublished(analysis_dir, "analysis stage")
    _require_unpublished(diagnostics_dir, "build diagnostics")
    _require_unpublished(report_path, "campaign report")
    for label, path in roots:
        _reject_symlink_components(path, label)
    _reject_symlink_components(analysis_dir.parent, "analysis stage")
    binding = _validate_binding(
        _binding if _binding is not None else repository_binding(repo_root)
    )
    claims_membership = verify_stage_membership(declaration, claims_dir, "claims")
    analysis_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(analysis_dir.parent, "analysis stage")
    reservation = _reserve_report(report_path)
    try:
        work_root = Path(
            tempfile.mkdtemp(
                prefix=f".{analysis_dir.name}.incoming-", dir=analysis_dir.parent
            )
        )
    except Exception:
        reservation.close()
        raise
    incoming = work_root / "analysis"
    scratch = work_root / "scratch"
    claims_snapshot = work_root / "claims"
    execution_snapshot = work_root / "implementation"
    incoming.mkdir()
    scratch.mkdir()
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        if not claims_membership["passed"]:
            failures.extend(
                f"claims-stage: {failure}"
                for failure in claims_membership["failures"]
            )
            rows = [
                _empty_row(row, "exact claims-stage membership failed")
                for row in declaration["maps"]
            ]
            analysis_membership = verify_stage_membership(
                declaration, incoming, "analysis"
            )
            report = _campaign_report(
                declaration,
                declaration_sha256,
                binding,
                claims_membership,
                analysis_membership,
                rows,
                failures,
                published=False,
            )
            reservation.write(report)
            return report

        try:
            snapshot_membership = _copy_claims_snapshot(
                declaration, claims_dir, claims_snapshot
            )
            if canonical_bytes(snapshot_membership) != canonical_bytes(
                claims_membership
            ):
                raise GeneratedAtlasCampaignError(
                    "immutable claims snapshot differs from admitted input"
                )
            claims_membership = snapshot_membership
            _snapshot_repository(repo_root, execution_snapshot, binding)
        except (GeneratedAtlasCampaignError, OSError, subprocess.SubprocessError) as exc:
            error = _canonical_error(exc)
            failures.append(f"preflight: {error}")
            rows = [_empty_row(row, error) for row in declaration["maps"]]
            analysis_membership = verify_stage_membership(
                declaration, incoming, "analysis"
            )
            report = _campaign_report(
                declaration,
                declaration_sha256,
                binding,
                claims_membership,
                analysis_membership,
                rows,
                failures,
                published=False,
            )
            reservation.write(report)
            return report

        diagnostics_dir.mkdir(parents=True, exist_ok=False)
        _reject_symlink_components(diagnostics_dir, "build diagnostics")
        _fsync_directory(diagnostics_dir.parent)
        diagnostics_identity_stat = diagnostics_dir.lstat()
        diagnostics_identity = (
            diagnostics_identity_stat.st_dev,
            diagnostics_identity_stat.st_ino,
        )
        builder = _builder or _default_builder_factory(
            execution_root=execution_snapshot,
            client_root=client_root,
            lithium_root=lithium_root,
            hook_attestation=hook_attestation,
            fall_oracle=fall_oracle,
            packer=packer,
            verifier=verifier,
        )

        for declared_index, declared in enumerate(declaration["maps"]):
            name = declared["map"]
            map_output = scratch / name
            map_output.mkdir()
            try:
                result = builder(
                    declared,
                    claims_snapshot,
                    map_output,
                    execution_snapshot,
                )
                if not isinstance(result, BuildProcessResult):
                    raise GeneratedAtlasCampaignError(
                        "builder returned an unsupported result"
                    )
            except Exception as exc:
                result = BuildProcessResult(
                    returncode=1,
                    stdout=b"",
                    stderr=(_canonical_error(exc) + "\n").encode("utf-8"),
                )

            stdout_record = {
                "bytes": len(result.stdout),
                "sha256": _sha256_bytes(result.stdout),
            }
            stderr_record = {
                "bytes": len(result.stderr),
                "sha256": _sha256_bytes(result.stderr),
            }
            _exclusive_write(
                diagnostics_dir / f"{name}.build.stdout.log", result.stdout
            )
            _exclusive_write(
                diagnostics_dir / f"{name}.build.stderr.log", result.stderr
            )
            summary: dict[str, Any] | None = None
            summary_payload: bytes | None = None
            artifacts: dict[str, dict[str, Any]] = {}
            analysis_binding: dict[str, Any] | None = None
            error: str | None = None
            try:
                (
                    summary,
                    summary_payload,
                    artifacts,
                    analysis_binding,
                ) = _inspect_map_build(
                    map_output,
                    declared,
                    claims_snapshot,
                    result,
                    binding,
                )
                _exclusive_write(
                    diagnostics_dir / f"{name}.build-summary.json",
                    summary_payload,
                )
                for suffix in STAGE_SUFFIXES["analysis"]:
                    _exclusive_copy(
                        map_output / f"{name}{suffix}",
                        incoming / f"{name}{suffix}",
                    )
            except (GeneratedAtlasCampaignError, OSError, ValueError) as exc:
                error = _canonical_error(exc)
                failures.append(f"{name}: {error}")
            row = {
                "ordinal": declared["ordinal"],
                "map": name,
                "passed": error is None,
                "bsp_sha256": file_sha256(claims_snapshot / f"{name}.bsp"),
                "generator_claims_sha256": file_sha256(
                    claims_snapshot / f"{name}.generator-claims.json"
                ),
                "artifacts": artifacts,
                "analysis_manifest": analysis_binding,
                "build_summary": summary,
                "build_summary_sha256": (
                    None
                    if summary_payload is None
                    else _sha256_bytes(summary_payload)
                ),
                "stdout": stdout_record,
                "stderr": stderr_record,
                "error": error,
            }
            rows.append(row)
            shutil.rmtree(map_output, ignore_errors=True)
            try:
                _require_root_identity(
                    diagnostics_dir, diagnostics_identity, "build diagnostics"
                )
            except GeneratedAtlasCampaignError as exc:
                root_error = _canonical_error(exc)
                failures.append(f"build-diagnostics: {root_error}")
                row["passed"] = False
                row["error"] = root_error
                rows.extend(
                    _empty_row(remaining, root_error)
                    for remaining in declaration["maps"][declared_index + 1 :]
                )
                analysis_membership = verify_stage_membership(
                    declaration, incoming, "analysis"
                )
                report = _campaign_report(
                    declaration,
                    declaration_sha256,
                    binding,
                    claims_membership,
                    analysis_membership,
                    rows,
                    failures,
                    published=False,
                )
                reservation.write(report)
                return report

        analysis_membership = verify_stage_membership(
            declaration, incoming, "analysis"
        )
        membership_rows = {
            row["map"]: row["files"] for row in analysis_membership["maps"]
        }
        for row in rows:
            derived = membership_rows[row["map"]]
            if row["artifacts"] != derived:
                failures.append(
                    f"{row['map']}: per-map artifacts differ from incoming membership"
                )
                row["passed"] = False
                row["error"] = (
                    "per-map artifacts differ from incoming membership"
                    if row["error"] is None
                    else row["error"]
                )
            row["artifacts"] = derived
        failures.extend(
            f"analysis-stage: {failure}"
            for failure in analysis_membership["failures"]
        )
        current_claims_membership = verify_stage_membership(
            declaration, claims_snapshot, "claims"
        )
        if canonical_bytes(current_claims_membership) != canonical_bytes(
            claims_membership
        ):
            failures.append("immutable claims snapshot changed during build")
        if (
            atlas_analyzer_authority_sha256(execution_snapshot)
            != binding["atlas_analyzer_authority_sha256"]
        ):
            failures.append("committed implementation snapshot changed during build")
        try:
            _require_root_identity(
                diagnostics_dir, diagnostics_identity, "build diagnostics"
            )
        except GeneratedAtlasCampaignError as exc:
            failures.append(f"build-diagnostics: {_canonical_error(exc)}")
        if not failures and all(row["passed"] for row in rows):
            expected_diagnostics = {
                f"{row['map']}{suffix}"
                for row in declaration["maps"]
                for suffix in (
                    ".build.stdout.log",
                    ".build.stderr.log",
                    ".build-summary.json",
                )
            }
            try:
                diagnostics_descriptor, _identity = _open_exact_flat_root(
                    diagnostics_dir,
                    expected_diagnostics,
                    "build diagnostics",
                )
                os.close(diagnostics_descriptor)
            except GeneratedAtlasCampaignError as exc:
                failures.append(f"build-diagnostics: {_canonical_error(exc)}")

        if failures or not all(row["passed"] for row in rows):
            report = _campaign_report(
                declaration,
                declaration_sha256,
                binding,
                claims_membership,
                analysis_membership,
                rows,
                failures,
                published=False,
            )
            reservation.write(report)
            return report

        incoming_descriptor, incoming_identity = _open_exact_flat_root(
            incoming,
            _expected_stage_names(declaration, "analysis"),
            "incoming analysis stage",
        )
        os.close(incoming_descriptor)
        _fsync_directory(incoming)

        # The complete passing report is durable before the only publication
        # operation. A crash after rename therefore cannot leave an unattested
        # selected analysis root.
        passing_report = _campaign_report(
            declaration,
            declaration_sha256,
            binding,
            claims_membership,
            analysis_membership,
            rows,
            [],
            published=True,
        )
        reservation.write(passing_report)
        try:
            reservation._require_owned_path()
            _reject_symlink_components(analysis_dir.parent, "analysis stage")
            _rename_noreplace(incoming, analysis_dir)
            _fsync_directory(analysis_dir.parent)
        except (GeneratedAtlasCampaignError, OSError) as exc:
            failure_report = _campaign_report(
                declaration,
                declaration_sha256,
                binding,
                claims_membership,
                analysis_membership,
                rows,
                [f"atomic-publication: {_canonical_error(exc)}"],
                published=False,
            )
            reservation.write(failure_report)
            return failure_report

        post_failures: list[str] = []
        try:
            reservation._require_owned_path()
            _reject_symlink_components(analysis_dir.parent, "analysis stage")
            _require_root_identity(
                analysis_dir, incoming_identity, "published analysis stage"
            )
            final_descriptor, _identity = _open_exact_flat_root(
                analysis_dir,
                _expected_stage_names(declaration, "analysis"),
                "published analysis stage",
            )
            os.close(final_descriptor)
            final_membership = verify_stage_membership(
                declaration, analysis_dir, "analysis"
            )
            if canonical_bytes(final_membership) != canonical_bytes(
                analysis_membership
            ):
                post_failures.append(
                    "published analysis membership differs from incoming authority"
                )
        except (GeneratedAtlasCampaignError, OSError) as exc:
            post_failures.append(
                "post-publication exact-root verification failed: "
                + _canonical_error(exc)
            )
        if post_failures:
            quarantine = _quarantine_owned_root(analysis_dir)
            if quarantine is None:
                raise GeneratedAtlasCampaignError(
                    "owned invalid analysis root could not be quarantined; "
                    "durable pre-publication attestation retained"
                )
            failure_membership = verify_stage_membership(
                declaration, analysis_dir, "analysis"
            )
            failure_report = _campaign_report(
                declaration,
                declaration_sha256,
                binding,
                claims_membership,
                failure_membership,
                rows,
                post_failures,
                published=False,
            )
            reservation.write(failure_report)
            return failure_report
        return passing_report
    finally:
        _remove_work_root(work_root)
        reservation.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--claims-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--client-root", type=Path, default=DEFAULT_CLIENT)
    parser.add_argument("--lithium-root", type=Path, default=DEFAULT_LITHIUM)
    parser.add_argument(
        "--hook-attestation", type=Path, default=DEFAULT_ATTESTATION
    )
    parser.add_argument("--fall-oracle", type=Path)
    parser.add_argument("--packer", type=Path)
    parser.add_argument("--verifier", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_atlas_campaign(
            args.declaration,
            args.claims_dir,
            args.analysis_dir,
            args.diagnostics_dir,
            args.output,
            client_root=args.client_root,
            lithium_root=args.lithium_root,
            hook_attestation=args.hook_attestation,
            fall_oracle=args.fall_oracle,
            packer=args.packer,
            verifier=args.verifier,
        )
    except (GeneratedAtlasCampaignError, GeneratorCohortError, OSError) as exc:
        print(f"generated Atlas campaign failed: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
