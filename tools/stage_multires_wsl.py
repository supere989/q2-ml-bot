#!/usr/bin/env python3
"""Fail-closed, no-overlay staging of the multires source triple on WSL.

The public entry points are ``preflight`` (read-only) and ``stage``.  The
implementation deliberately transports committed Git objects rather than a
working-tree copy.  The same module is encoded into the SSH command and runs
the remote half, which gives the tests a local transport seam without a second
implementation drifting away from production behavior.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, BinaryIO, Mapping, Protocol, Sequence
import uuid


SCHEMA = "q2-multires-isolated-wsl-staging-v1"
PREFLIGHT_SCHEMA = "q2-multires-isolated-wsl-staging-preflight-v1"
TRANSFER_SCHEMA = "q2-multires-isolated-wsl-transfer-v1"
EXPECTED_HOST_IDENTITY = "DESKTOP-RTX2080"
DEFAULT_BRANCH = "feature/multires-map-atlas-v1"
DEFAULT_TOOLCHAIN_ROOT = (
    "/home/raymond/q2-multires-isolated/tooling/"
    "rust-1.96.1-x86_64-unknown-linux-gnu"
)
NORMATIVE_DOCUMENTS = (
    "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md",
    "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md",
)
REPOSITORY_NAMES = ("q2-ml-bot", "q2-ml-client", "q2-lithium-3zb2")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40,64}$")
SSH_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
BRANCH_RE = re.compile(r"^(?![-/])(?!.*(?:\.\.|//|@\{|\\))[A-Za-z0-9._/-]+(?<![/.])$")
MAX_TRANSFER_MEMBER_BYTES = 1 << 30
MAX_TRANSFER_BYTES = 3 << 30
MAX_FAILURE_DETAIL_CHARS = 512
ELIGIBLE_TRANSPORT_STATE: dict[str, bool | int] = {
    "shallow": False,
    "partial_clone": False,
    "replace_refs": 0,
    "alternates": False,
}


class StagingError(RuntimeError):
    """A fail-closed staging validation or transport failure."""


def _bounded_failure_detail(value: bytes | str) -> str:
    if isinstance(value, bytes):
        detail = value.decode("utf-8", "replace")
    else:
        detail = value
    detail = " ".join(detail.split())
    detail = re.sub(
        r"(?i)[\"']?\bauthorization\b[\"']?\s*(?::|=)?\s*[\"']?"
        r"\b(?:bearer|basic)\b\s+[^\s,;}\"']+",
        "Authorization=<redacted>",
        detail,
    )
    detail = re.sub(
        r"(?i)[\"']?\b(github_token|access_token|api_key|token|password|"
        r"secret|credential)\b[\"']?\s*(?:=|:)\s*[\"']?[^\s,;}\"']+",
        lambda match: f"{match.group(1)}=<redacted>",
        detail,
    )
    detail = re.sub(
        r"([A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@",
        r"\1<redacted>@",
        detail,
    )
    if not detail:
        detail = "no stderr"
    return detail[:MAX_FAILURE_DETAIL_CHARS]


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require_sha256(label: str, value: str) -> str:
    normalized = value.lower()
    if not SHA256_RE.fullmatch(normalized):
        raise StagingError(f"{label} must be a lowercase 64-character SHA-256")
    return normalized


def _require_git_oid(label: str, value: str) -> str:
    normalized = value.lower()
    if not GIT_OID_RE.fullmatch(normalized):
        raise StagingError(f"{label} is not a supported Git object ID")
    return normalized


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    input_bytes: bytes | None = None,
) -> bytes:
    completed = subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd is not None else None,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        stderr = _bounded_failure_detail(completed.stderr)
        raise StagingError(f"command failed ({argv[0]}): {stderr}")
    return completed.stdout


def _git(repo: Path, *args: str) -> bytes:
    return _run(("git", "-C", str(repo), *args))


def _assert_absolute_no_symlink(path: Path, *, must_exist: bool, label: str) -> None:
    if not path.is_absolute():
        raise StagingError(f"{label} must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if must_exist:
                raise StagingError(f"{label} does not exist: {path}")
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise StagingError(f"{label} contains a symlink component: {current}")


def _safe_branch(branch: str) -> str:
    if not BRANCH_RE.fullmatch(branch) or branch.endswith(".lock"):
        raise StagingError(f"unsafe or invalid branch name: {branch!r}")
    return branch


def _parse_tree_closure(raw: bytes) -> tuple[str, list[tuple[bytes, bytes, bytes]]]:
    """Hash the recursive tree listing and return validated blob entries."""

    digest = hashlib.sha256()
    entries: list[tuple[bytes, bytes, bytes]] = []
    seen: set[bytes] = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, path = record.split(b"\t", 1)
            mode, object_type, oid = metadata.split(b" ", 2)
        except ValueError as exc:
            raise StagingError("malformed git ls-tree record") from exc
        if mode == b"120000":
            raise StagingError(
                f"tracked symlink is forbidden: {path.decode('utf-8', 'replace')}"
            )
        if mode == b"160000" or object_type == b"commit":
            raise StagingError(
                f"gitlink/submodule is forbidden: {path.decode('utf-8', 'replace')}"
            )
        if mode not in (b"100644", b"100755") or object_type != b"blob":
            raise StagingError("unsupported object in source tree")
        pure = PurePosixPath(os.fsdecode(path))
        if pure.is_absolute() or not pure.parts or any(p in ("", ".", "..") for p in pure.parts):
            raise StagingError("unsafe path in source tree")
        if path in seen:
            raise StagingError("duplicate path in source tree")
        seen.add(path)
        digest.update(mode + b"\0" + oid + b"\0" + path + b"\0")
        entries.append((mode, oid, path))
    if not entries:
        raise StagingError("source repository has an empty tracked-file closure")
    return digest.hexdigest(), entries


def _repository_closures(repo: Path) -> tuple[str, str, int, int]:
    """Return Git-listing and independent SHA-256 content closures."""

    tree_sha256, entries = _parse_tree_closure(
        _git(repo, "ls-tree", "-r", "-z", "--full-tree", "HEAD")
    )
    request = b"".join(oid + b"\n" for _, oid, _ in entries)
    response = _run(
        ("git", "-C", str(repo), "cat-file", "--batch"), input_bytes=request
    )
    offset = 0
    content_digest = hashlib.sha256()
    total_bytes = 0
    for mode, expected_oid, path in entries:
        header_end = response.find(b"\n", offset)
        if header_end < 0:
            raise StagingError("truncated git cat-file batch header")
        header = response[offset:header_end].split(b" ")
        if len(header) != 3 or header[0] != expected_oid or header[1] != b"blob":
            raise StagingError("git cat-file returned an unexpected object")
        try:
            size = int(header[2])
        except ValueError as exc:
            raise StagingError("git cat-file returned an invalid blob size") from exc
        offset = header_end + 1
        end = offset + size
        if size < 0 or end >= len(response) or response[end : end + 1] != b"\n":
            raise StagingError("truncated git cat-file blob payload")
        blob_sha256 = hashlib.sha256(response[offset:end]).hexdigest().encode("ascii")
        content_digest.update(
            mode
            + b"\0"
            + path
            + b"\0"
            + str(size).encode("ascii")
            + b"\0"
            + blob_sha256
            + b"\0"
        )
        total_bytes += size
        offset = end + 1
    if offset != len(response):
        raise StagingError("git cat-file returned trailing data")
    return tree_sha256, content_digest.hexdigest(), len(entries), total_bytes


def inspect_repository(
    *,
    name: str,
    path: Path,
    expected_branch: str,
    expected_commit: str,
    expected_tree: str,
) -> dict[str, Any]:
    if name not in REPOSITORY_NAMES:
        raise StagingError(f"unknown repository name: {name}")
    expected_branch = _safe_branch(expected_branch)
    expected_commit = _require_git_oid(f"{name} expected commit", expected_commit)
    expected_tree = _require_git_oid(f"{name} expected tree", expected_tree)
    _assert_absolute_no_symlink(path, must_exist=True, label=f"{name} repository")
    if not path.is_dir():
        raise StagingError(f"{name} repository is not a directory: {path}")
    git_marker = path / ".git"
    if not os.path.lexists(git_marker) or git_marker.is_symlink():
        raise StagingError(f"{name} .git marker is missing or linked")
    try:
        actual_root = Path(
            _git(path, "rev-parse", "--path-format=absolute", "--show-toplevel")
            .decode("utf-8")
            .strip()
        )
    except StagingError as exc:
        raise StagingError(f"{name} is not a usable Git repository") from exc
    if actual_root != path:
        raise StagingError(f"{name} path is not its exact Git root: {path}")
    transport_state = _repository_transport_state(path)
    if transport_state != ELIGIBLE_TRANSPORT_STATE:
        raise StagingError(
            f"{name} source is not transport-eligible "
            f"({_transport_state_summary(transport_state)}); hydrate or repair "
            "the source repository explicitly before staging"
        )
    branch = _git(path, "symbolic-ref", "--quiet", "--short", "HEAD").decode().strip()
    commit = _git(path, "rev-parse", "--verify", "HEAD^{commit}").decode().strip().lower()
    tree = _git(path, "rev-parse", "--verify", "HEAD^{tree}").decode().strip().lower()
    if branch != expected_branch:
        raise StagingError(f"{name} branch mismatch: expected {expected_branch}, got {branch}")
    if commit != expected_commit:
        raise StagingError(f"{name} commit mismatch: expected {expected_commit}, got {commit}")
    if tree != expected_tree:
        raise StagingError(f"{name} tree mismatch: expected {expected_tree}, got {tree}")
    dirty = _git(path, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    if dirty:
        raise StagingError(f"{name} repository is dirty")
    closure_sha256, content_sha256, file_count, content_bytes = _repository_closures(path)
    return {
        "name": name,
        "source_path": str(path),
        "staged_path": f"repositories/{name}",
        "branch": branch,
        "commit": commit,
        "tree": tree,
        "tracked_file_count": file_count,
        "tracked_file_closure_sha256": closure_sha256,
        "tracked_content_closure_sha256": content_sha256,
        "tracked_content_bytes": content_bytes,
        "transport_eligibility": dict(transport_state),
    }


def inspect_sources(repository_specs: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    if [item["name"] for item in repository_specs] != list(REPOSITORY_NAMES):
        raise StagingError("repository source triple is incomplete or out of order")
    snapshots = [
        inspect_repository(
            name=item["name"],
            path=Path(item["path"]),
            expected_branch=item["branch"],
            expected_commit=item["commit"],
            expected_tree=item["tree"],
        )
        for item in repository_specs
    ]
    bot_root = Path(repository_specs[0]["path"])
    documents: list[dict[str, Any]] = []
    for relative in NORMATIVE_DOCUMENTS:
        path = bot_root / relative
        if not path.is_file() or path.is_symlink():
            raise StagingError(f"normative document missing or linked: {relative}")
        digest, size = sha256_file(path)
        documents.append({"path": relative, "sha256": digest, "size_bytes": size})
    snapshots[0]["normative_documents"] = documents
    return snapshots


def _remote_paths(request: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = Path(request["isolated_root"])
    destination = Path(request["destination"])
    toolchain = Path(request["toolchain_root"])
    _assert_absolute_no_symlink(root, must_exist=True, label="isolated root")
    if not root.is_dir():
        raise StagingError(f"isolated root is not a directory: {root}")
    _assert_absolute_no_symlink(destination.parent, must_exist=True, label="destination parent")
    if not destination.parent.is_dir():
        raise StagingError("destination parent is not a directory")
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise StagingError("destination is not beneath the declared isolated root") from exc
    if destination == root:
        raise StagingError("destination cannot equal the isolated root")
    if os.path.lexists(destination):
        raise StagingError(f"destination already exists: {destination}")
    _assert_absolute_no_symlink(toolchain, must_exist=True, label="toolchain root")
    if not toolchain.is_dir():
        raise StagingError("toolchain root is not a directory")
    return root, destination, toolchain


def _remote_identity_and_toolchain(request: Mapping[str, Any]) -> dict[str, Any]:
    actual_hostname = socket.gethostname().split(".", 1)[0]
    expected_hostname = request["expected_hostname"]
    if actual_hostname != expected_hostname:
        raise StagingError(
            f"host identity mismatch: expected {expected_hostname}, got {actual_hostname}"
        )
    root, destination, toolchain = _remote_paths(request)
    binaries: list[dict[str, Any]] = []
    for name in ("rustc", "cargo"):
        path = toolchain / "bin" / name
        _assert_absolute_no_symlink(path, must_exist=True, label=f"toolchain {name}")
        if not path.is_file():
            raise StagingError(f"toolchain binary is not a regular file: {path}")
        actual, size = sha256_file(path)
        expected = _require_sha256(f"expected {name} SHA-256", request[f"{name}_sha256"])
        if actual != expected:
            raise StagingError(
                f"toolchain {name} hash mismatch: expected {expected}, got {actual}"
            )
        binaries.append(
            {
                "name": name,
                "path": str(path),
                "sha256": actual,
                "size_bytes": size,
            }
        )
    return {
        "hostname": actual_hostname,
        "isolated_root": str(root),
        "destination": str(destination),
        "toolchain": {"root": str(toolchain), "binaries": binaries},
    }


def _validate_request(request: Mapping[str, Any]) -> None:
    if request.get("schema") != SCHEMA:
        raise StagingError("remote request schema mismatch")
    if request.get("expected_hostname") != EXPECTED_HOST_IDENTITY:
        raise StagingError("the only admitted WSL host identity is DESKTOP-RTX2080")
    repositories = request.get("repositories")
    if not isinstance(repositories, list) or [r.get("name") for r in repositories] != list(
        REPOSITORY_NAMES
    ):
        raise StagingError("remote request does not bind the exact source triple")
    for repository in repositories:
        _safe_branch(repository.get("branch", ""))
        _require_git_oid("repository commit", repository.get("commit", ""))
        _require_git_oid("repository tree", repository.get("tree", ""))
        _require_sha256(
            "repository file closure", repository.get("tracked_file_closure_sha256", "")
        )
        _require_sha256(
            "repository content closure", repository.get("tracked_content_closure_sha256", "")
        )
        if not isinstance(repository.get("tracked_file_count"), int) or repository[
            "tracked_file_count"
        ] <= 0:
            raise StagingError("invalid repository tracked-file count")
        if not isinstance(repository.get("tracked_content_bytes"), int) or repository[
            "tracked_content_bytes"
        ] < 0:
            raise StagingError("invalid repository tracked-content byte count")
        _require_transport_eligibility(repository.get("transport_eligibility"))
    documents = request.get("normative_documents")
    if not isinstance(documents, list) or [item.get("path") for item in documents] != list(
        NORMATIVE_DOCUMENTS
    ):
        raise StagingError("remote request lacks the exact normative document set")
    for item in documents:
        _require_sha256("normative document", item.get("sha256", ""))
    _require_sha256("rustc", request.get("rustc_sha256", ""))
    _require_sha256("cargo", request.get("cargo_sha256", ""))


def _require_transport_eligibility(value: Any) -> None:
    expected_keys = set(ELIGIBLE_TRANSPORT_STATE)
    if type(value) is not dict or set(value) != expected_keys:
        raise StagingError("repository transport eligibility key set is not sealed")
    if (
        type(value["shallow"]) is not bool
        or type(value["partial_clone"]) is not bool
        or type(value["replace_refs"]) is not int
        or type(value["alternates"]) is not bool
    ):
        raise StagingError("repository transport eligibility types are not sealed")
    if value != ELIGIBLE_TRANSPORT_STATE:
        raise StagingError("repository transport eligibility values are not sealed")


def _remote_preflight(request: Mapping[str, Any]) -> dict[str, Any]:
    _validate_request(request)
    remote = _remote_identity_and_toolchain(request)
    semantic = {
        "schema": SCHEMA,
        "host_identity": remote["hostname"],
        "isolated_root": remote["isolated_root"],
        "destination": remote["destination"],
        "repositories": request["repositories"],
        "normative_documents": request["normative_documents"],
        "toolchain": remote["toolchain"],
        "effects": {
            "public_runtime_changed": False,
            "services_changed": False,
            "trainer_changed_or_started": False,
        },
    }
    return {
        "schema": PREFLIGHT_SCHEMA,
        "passed": True,
        "read_only": True,
        "semantic": semantic,
        "semantic_sha256": sha256_bytes(canonical_bytes(semantic)),
    }


def _safe_transfer_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    expected = {"transfer.json"} | {f"bundles/{name}.bundle" for name in REPOSITORY_NAMES}
    found: dict[str, tarfile.TarInfo] = {}
    total = 0
    for member in archive:
        pure = PurePosixPath(member.name)
        if (
            pure.is_absolute()
            or not pure.parts
            or any(part in ("", ".", "..") for part in pure.parts)
            or member.name not in expected
        ):
            raise StagingError(f"unexpected or unsafe transfer member: {member.name!r}")
        if member.name in found:
            raise StagingError(f"duplicate transfer member: {member.name}")
        if not member.isfile() or member.issym() or member.islnk():
            raise StagingError(f"non-regular transfer member is forbidden: {member.name}")
        if member.size < 0 or member.size > MAX_TRANSFER_MEMBER_BYTES:
            raise StagingError(f"transfer member size is invalid: {member.name}")
        total += member.size
        if total > MAX_TRANSFER_BYTES:
            raise StagingError("transfer archive exceeds the hard byte limit")
        found[member.name] = member
    if set(found) != expected:
        missing = sorted(expected - set(found))
        raise StagingError(f"transfer archive membership mismatch; missing={missing}")
    return found


def _extract_transfer_spool(spool: Path, incoming: Path) -> dict[str, Any]:
    with tarfile.open(spool, mode="r:*") as archive:
        members = _safe_transfer_members(archive)
        for name in sorted(members):
            target = incoming / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if os.path.lexists(target):
                raise StagingError(f"duplicate extracted path: {name}")
            source = archive.extractfile(members[name])
            if source is None:
                raise StagingError(f"cannot read transfer member: {name}")
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
    raw_manifest = (incoming / "transfer.json").read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StagingError("transfer manifest is not valid JSON") from exc
    if raw_manifest != canonical_bytes(manifest):
        raise StagingError("transfer manifest is not canonical JSON")
    if manifest.get("schema") != TRANSFER_SCHEMA:
        raise StagingError("transfer manifest schema mismatch")
    return manifest


def _verify_transfer_manifest(
    manifest: Mapping[str, Any], request: Mapping[str, Any], incoming: Path
) -> None:
    repositories = manifest.get("repositories")
    if not isinstance(repositories, list) or len(repositories) != 3:
        raise StagingError("transfer manifest repository set is invalid")
    by_name = {item.get("name"): item for item in repositories}
    if set(by_name) != set(REPOSITORY_NAMES):
        raise StagingError("transfer manifest repository names are invalid")
    request_by_name = {item["name"]: item for item in request["repositories"]}
    for name in REPOSITORY_NAMES:
        transfer = by_name[name]
        if transfer.get("source") != request_by_name[name]:
            raise StagingError(f"transfer source binding mismatch for {name}")
        expected_hash = _require_sha256("bundle", transfer.get("bundle_sha256", ""))
        bundle_path = incoming / f"bundles/{name}.bundle"
        actual_hash, actual_size = sha256_file(bundle_path)
        if actual_hash != expected_hash or actual_size != transfer.get("bundle_size_bytes"):
            raise StagingError(f"bundle hash/size mismatch for {name}")


def _checkout_and_verify_repository(
    incoming: Path,
    repositories_root: Path,
    source: Mapping[str, Any],
    *,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    name = source["name"]
    destination = repositories_root / name
    destination.mkdir(mode=0o700)
    bundle = bundle_path or incoming / f"bundles/{name}.bundle"
    _run(("git", "init", "--quiet", str(destination)))
    _run(("git", "-C", str(destination), "bundle", "verify", str(bundle)))
    refspec = f"refs/heads/{source['branch']}:refs/heads/{source['branch']}"
    _run(("git", "-C", str(destination), "fetch", "--quiet", str(bundle), refspec))
    _run(("git", "-C", str(destination), "checkout", "--quiet", source["branch"]))
    branch = _git(destination, "symbolic-ref", "--quiet", "--short", "HEAD").decode().strip()
    commit = _git(destination, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    tree = _git(destination, "rev-parse", "--verify", "HEAD^{tree}").decode().strip()
    if branch != source["branch"] or commit != source["commit"] or tree != source["tree"]:
        raise StagingError(f"staged Git identity mismatch for {name}")
    if _git(destination, "status", "--porcelain=v1", "-z", "--untracked-files=all"):
        raise StagingError(f"staged repository is dirty: {name}")
    closure, content, count, content_bytes = _repository_closures(destination)
    if (
        closure != source["tracked_file_closure_sha256"]
        or content != source["tracked_content_closure_sha256"]
        or count != source["tracked_file_count"]
        or content_bytes != source["tracked_content_bytes"]
    ):
        raise StagingError(f"staged file closure mismatch for {name}")
    _run(("git", "-C", str(destination), "fsck", "--strict", "--no-dangling"))
    return dict(source)


def _atomic_write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(manifest))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    if stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode) != 0o600:
        raise StagingError("staging manifest mode is not 0600")


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise StagingError("renameat2 is required for atomic no-overlay publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        error = ctypes.get_errno()
        raise StagingError(f"atomic no-overlay publication failed: {os.strerror(error)}")


def _remote_stage(request: Mapping[str, Any], stream: BinaryIO) -> dict[str, Any]:
    preflight = _remote_preflight(request)
    _, destination, _ = _remote_paths(request)
    nonce = request.get("invocation_nonce", "")
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise StagingError("invalid staging invocation nonce")
    temporary = destination.parent / f".{destination.name}.partial-{nonce}"
    if os.path.lexists(temporary):
        raise StagingError("invocation temporary destination already exists")
    created = False
    try:
        temporary.mkdir(mode=0o700)
        created = True
        incoming = temporary / ".incoming"
        repositories_root = temporary / "repositories"
        incoming.mkdir(mode=0o700)
        repositories_root.mkdir(mode=0o700)
        spool = incoming / ".transfer.tar"
        descriptor = os.open(spool, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        total = 0
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_TRANSFER_BYTES:
                    raise StagingError("transfer stream exceeds the hard byte limit")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        transfer_manifest = _extract_transfer_spool(spool, incoming)
        _verify_transfer_manifest(transfer_manifest, request, incoming)
        verified = [
            _checkout_and_verify_repository(incoming, repositories_root, source)
            for source in request["repositories"]
        ]
        bot_root = repositories_root / "q2-ml-bot"
        for expected in request["normative_documents"]:
            path = bot_root / expected["path"]
            if path.is_symlink() or not path.is_file():
                raise StagingError(f"staged normative document missing: {expected['path']}")
            digest, size = sha256_file(path)
            if digest != expected["sha256"] or size != expected["size_bytes"]:
                raise StagingError(f"staged normative document mismatch: {expected['path']}")
        semantic = dict(preflight["semantic"])
        semantic["repositories"] = verified
        semantic_sha256 = sha256_bytes(canonical_bytes(semantic))
        manifest = {
            "schema": SCHEMA,
            "semantic": semantic,
            "semantic_sha256": semantic_sha256,
            "informational": {"staged_at_utc": request["staged_at_utc"]},
        }
        manifest_sha256 = sha256_bytes(canonical_bytes(manifest))
        _atomic_write_manifest(temporary / "staging-manifest.json", manifest)
        shutil.rmtree(incoming)
        if os.path.lexists(destination):
            raise StagingError("destination appeared during staging")
        _rename_noreplace(temporary, destination)
        created = False
        return {
            "schema": SCHEMA,
            "passed": True,
            "published": True,
            "destination": str(destination),
            "manifest_path": str(destination / "staging-manifest.json"),
            "manifest_sha256": manifest_sha256,
            "semantic_sha256": semantic_sha256,
            "effects": semantic["effects"],
        }
    finally:
        if created and os.path.lexists(temporary):
            shutil.rmtree(temporary)


class Transport(Protocol):
    def execute(
        self, mode: str, request: Mapping[str, Any], archive_path: Path | None
    ) -> dict[str, Any]: ...


class OpenSshTransport:
    def __init__(self, host: str, ssh_bin: str = "ssh", module_path: Path | None = None):
        if not SSH_HOST_RE.fullmatch(host):
            raise StagingError(f"unsafe SSH host alias: {host!r}")
        if not ssh_bin or "\0" in ssh_bin:
            raise StagingError("invalid SSH executable")
        self.host = host
        self.ssh_bin = ssh_bin
        self.module_path = module_path or Path(__file__)

    def execute(
        self, mode: str, request: Mapping[str, Any], archive_path: Path | None
    ) -> dict[str, Any]:
        encoded_module = base64.b64encode(self.module_path.read_bytes()).decode("ascii")
        encoded_request = base64.b64encode(canonical_bytes(request)).decode("ascii")
        bootstrap = (
            "import base64;"
            f"exec(compile(base64.b64decode('{encoded_module}'),"
            "'<stage_multires_wsl.py>','exec'))"
        )
        remote_argv = (
            "python3",
            "-c",
            bootstrap,
            "--remote-internal",
            mode,
            "--request-b64",
            encoded_request,
        )
        # OpenSSH passes one remote command string through the remote login
        # shell.  shlex.join performs the only quoting step; user values live
        # inside base64 JSON and are validated again remotely.
        import shlex

        command = shlex.join(remote_argv)
        stdin: BinaryIO | int
        archive: BinaryIO | None = None
        if archive_path is None:
            stdin = subprocess.DEVNULL
        else:
            archive = archive_path.open("rb")
            stdin = archive
        try:
            completed = subprocess.run(
                (self.ssh_bin, "--", self.host, command),
                stdin=stdin,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        finally:
            if archive is not None:
                archive.close()
        stdout = completed.stdout.decode("utf-8", "replace")
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as exc:
            diagnostic = _bounded_failure_detail(
                completed.stderr if completed.stderr.strip() else completed.stdout
            )
            raise StagingError(
                f"remote staging response was invalid: {diagnostic}"
            ) from exc
        if completed.returncode != 0 or not response.get("ok"):
            diagnostic = _bounded_failure_detail(
                str(response.get("error", "unknown error"))
            )
            raise StagingError(f"remote staging failed: {diagnostic}")
        return response["result"]


class LocalRemoteTransport:
    """Exact remote implementation seam for tests; never used by the CLI."""

    def __init__(self, hostname: str):
        self.hostname = hostname

    def execute(
        self, mode: str, request: Mapping[str, Any], archive_path: Path | None
    ) -> dict[str, Any]:
        original = socket.gethostname
        socket.gethostname = lambda: self.hostname  # type: ignore[method-assign]
        try:
            if mode == "preflight":
                if archive_path is not None:
                    raise StagingError("preflight cannot receive an archive")
                return _remote_preflight(request)
            if mode == "stage":
                if archive_path is None:
                    raise StagingError("stage requires an archive")
                with archive_path.open("rb") as stream:
                    return _remote_stage(request, stream)
            raise StagingError(f"unknown transport mode: {mode}")
        finally:
            socket.gethostname = original  # type: ignore[method-assign]


def _repository_transport_state(repo: Path) -> dict[str, bool | int]:
    shallow = _git(repo, "rev-parse", "--is-shallow-repository").strip() == b"true"
    config_entries: dict[str, str] = {}
    for record in _git(repo, "config", "--null", "--list").split(b"\0"):
        if not record:
            continue
        key, separator, value = record.partition(b"\n")
        if not separator:
            continue
        config_entries[key.decode("utf-8", "replace").lower()] = value.decode(
            "utf-8", "replace"
        )
    common_git_value = _git(repo, "rev-parse", "--git-common-dir").decode(
        "utf-8", "replace"
    ).strip()
    common_git = Path(common_git_value)
    if not common_git.is_absolute():
        common_git = repo / common_git
    promisor_packs = any((common_git / "objects" / "pack").glob("*.promisor"))
    partial_clone = (
        promisor_packs
        or "extensions.partialclone" in config_entries
        or any(
            (
                key.startswith("remote.")
                and (
                    key.endswith(".partialclonefilter")
                    or (
                        key.endswith(".promisor")
                        and value.lower() in ("1", "true", "yes", "on")
                    )
                )
            )
            for key, value in config_entries.items()
        )
    )
    replace_refs = len(
        [
            line for line in _git(
                repo, "for-each-ref", "--format=%(refname)", "refs/replace"
            ).splitlines()
            if line
        ]
    )
    alternates_value = _git(
        repo, "rev-parse", "--git-path", "objects/info/alternates"
    ).decode("utf-8", "replace").strip()
    alternates_path = Path(alternates_value)
    if not alternates_path.is_absolute():
        alternates_path = repo / alternates_path
    alternates = os.path.lexists(alternates_path) or bool(
        os.environ.get("GIT_ALTERNATE_OBJECT_DIRECTORIES")
    )
    return {
        "shallow": shallow,
        "partial_clone": partial_clone,
        "replace_refs": replace_refs,
        "alternates": alternates,
    }


def _transport_state_summary(state: Mapping[str, bool | int]) -> str:
    return ", ".join(
        (
            f"shallow={'true' if state['shallow'] else 'false'}",
            f"partial_clone={'true' if state['partial_clone'] else 'false'}",
            f"replace_refs={state['replace_refs']}",
            f"alternates={'true' if state['alternates'] else 'false'}",
        )
    )


def _prove_bundle_self_contained(
    repo: Mapping[str, Any], bundle: Path
) -> None:
    with tempfile.TemporaryDirectory(prefix="q2-multires-bundle-proof-") as temp_name:
        root = Path(temp_name)
        repositories = root / "repositories"
        repositories.mkdir(mode=0o700)
        _checkout_and_verify_repository(
            root / "unused-incoming",
            repositories,
            repo,
            bundle_path=bundle,
        )


def _bundle_repository(repo: Mapping[str, Any], bundle: Path) -> dict[str, Any]:
    source_path = Path(repo["source_path"])
    transport_state = _repository_transport_state(source_path)
    try:
        _run(
            (
                "git",
                "-C",
                str(source_path),
                "bundle",
                "create",
                str(bundle),
                f"refs/heads/{repo['branch']}",
            )
        )
        # Source-context verification is syntax-only assurance: a shallow or
        # otherwise incomplete source can satisfy prerequisites from its own
        # object database.  The empty-repository fetch below is authoritative.
        _run(("git", "-C", str(source_path), "bundle", "verify", str(bundle)))
        _prove_bundle_self_contained(repo, bundle)
    except (OSError, StagingError) as exc:
        try:
            bundle.unlink()
        except FileNotFoundError:
            pass
        detail = _bounded_failure_detail(str(exc))
        raise StagingError(
            f"bundle is not self-contained for {repo['name']} "
            f"({_transport_state_summary(transport_state)}): {detail}"
        ) from exc
    digest, size = sha256_file(bundle)
    public_source = {
        key: value
        for key, value in repo.items()
        if key not in ("source_path", "normative_documents")
    }
    return {
        "name": repo["name"],
        "source": public_source,
        "bundle_sha256": digest,
        "bundle_size_bytes": size,
    }


def _tar_add_regular(archive: tarfile.TarFile, path: Path, name: str) -> None:
    data = path.read_bytes()
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o600
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def build_transfer_archive(
    repositories: Sequence[Mapping[str, Any]], output: Path
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="q2-multires-bundles-") as temp_name:
        temp = Path(temp_name)
        bundle_records = [
            _bundle_repository(repo, temp / f"{repo['name']}.bundle") for repo in repositories
        ]
        transfer = {"schema": TRANSFER_SCHEMA, "repositories": bundle_records}
        transfer_path = temp / "transfer.json"
        transfer_path.write_bytes(canonical_bytes(transfer))
        with tarfile.open(output, mode="w") as archive:
            _tar_add_regular(archive, transfer_path, "transfer.json")
            for repository in bundle_records:
                name = repository["name"]
                _tar_add_regular(
                    archive, temp / f"{name}.bundle", f"bundles/{name}.bundle"
                )
    return transfer


def make_request(
    *,
    repositories: Sequence[Mapping[str, Any]],
    isolated_root: Path,
    destination: Path,
    toolchain_root: Path,
    rustc_sha256: str,
    cargo_sha256: str,
    staged_at_utc: str,
    invocation_nonce: str,
) -> dict[str, Any]:
    documents = repositories[0].get("normative_documents")
    if not isinstance(documents, list):
        raise StagingError("source inspection did not produce normative document hashes")
    public_repositories = []
    for repository in repositories:
        public_repositories.append(
            {key: value for key, value in repository.items() if key != "source_path" and key != "normative_documents"}
        )
    request = {
        "schema": SCHEMA,
        "expected_hostname": EXPECTED_HOST_IDENTITY,
        "isolated_root": str(isolated_root),
        "destination": str(destination),
        "toolchain_root": str(toolchain_root),
        "rustc_sha256": _require_sha256("rustc", rustc_sha256),
        "cargo_sha256": _require_sha256("cargo", cargo_sha256),
        "repositories": public_repositories,
        "normative_documents": documents,
        "staged_at_utc": staged_at_utc,
        "invocation_nonce": invocation_nonce,
    }
    _validate_request(request)
    return request


def execute(
    *,
    mode: str,
    repository_specs: Sequence[Mapping[str, str]],
    isolated_root: Path,
    destination: Path,
    toolchain_root: Path,
    rustc_sha256: str,
    cargo_sha256: str,
    transport: Transport,
    staged_at_utc: str | None = None,
    invocation_nonce: str | None = None,
) -> dict[str, Any]:
    if mode not in ("preflight", "stage"):
        raise StagingError(f"invalid execution mode: {mode}")
    repositories = inspect_sources(repository_specs)
    request = make_request(
        repositories=repositories,
        isolated_root=isolated_root,
        destination=destination,
        toolchain_root=toolchain_root,
        rustc_sha256=rustc_sha256,
        cargo_sha256=cargo_sha256,
        staged_at_utc=staged_at_utc
        or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        invocation_nonce=invocation_nonce or uuid.uuid4().hex,
    )
    with tempfile.TemporaryDirectory(prefix="q2-multires-transfer-") as temp_name:
        archive = Path(temp_name) / "source-triple.tar"
        build_transfer_archive(repositories, archive)
        # Close the source-of-check/time-of-bundle race.  A commit, branch,
        # tree, index, or untracked-file change aborts before remote mutation.
        if inspect_sources(repository_specs) != repositories:
            raise StagingError("source triple changed while the transfer was being assembled")
        if mode == "preflight":
            return transport.execute("preflight", request, None)
        return transport.execute("stage", request, archive)


def _repository_specs_from_args(args: argparse.Namespace) -> list[dict[str, str]]:
    return [
        {
            "name": "q2-ml-bot",
            "path": str(Path(args.bot_repo).absolute()),
            "branch": args.bot_branch,
            "commit": args.bot_commit,
            "tree": args.bot_tree,
        },
        {
            "name": "q2-ml-client",
            "path": str(Path(args.client_repo).absolute()),
            "branch": args.client_branch,
            "commit": args.client_commit,
            "tree": args.client_tree,
        },
        {
            "name": "q2-lithium-3zb2",
            "path": str(Path(args.game_repo).absolute()),
            "branch": args.game_branch,
            "commit": args.game_commit,
            "tree": args.game_tree,
        },
    ]


def _add_repository_args(parser: argparse.ArgumentParser, prefix: str, default_path: str) -> None:
    parser.add_argument(f"--{prefix}-repo", default=default_path)
    parser.add_argument(f"--{prefix}-branch", default=DEFAULT_BRANCH)
    parser.add_argument(f"--{prefix}-commit", required=True)
    parser.add_argument(f"--{prefix}-tree", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-internal", choices=("preflight", "stage"), help=argparse.SUPPRESS)
    parser.add_argument("--request-b64", help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="mode")
    for mode in ("preflight", "stage"):
        command = subparsers.add_parser(mode)
        _add_repository_args(command, "bot", str(Path.cwd()))
        _add_repository_args(
            command,
            "client",
            "/home/raymondj/multires-worktrees/integration/q2-ml-client",
        )
        _add_repository_args(
            command,
            "game",
            "/home/raymondj/multires-worktrees/integration/q2-lithium-3zb2",
        )
        command.add_argument("--host", default="wsl-box")
        command.add_argument("--expected-hostname", default=EXPECTED_HOST_IDENTITY)
        command.add_argument("--isolated-root", type=Path, required=True)
        command.add_argument("--destination", type=Path, required=True)
        command.add_argument("--toolchain-root", type=Path, default=Path(DEFAULT_TOOLCHAIN_ROOT))
        command.add_argument("--rustc-sha256", required=True)
        command.add_argument("--cargo-sha256", required=True)
        command.add_argument("--ssh-bin", default="ssh")
    return parser


def _remote_main(mode: str, encoded_request: str) -> int:
    try:
        raw = base64.b64decode(encoded_request, validate=True)
        request = json.loads(raw)
        if raw != canonical_bytes(request):
            raise StagingError("remote request is not canonical JSON")
        if mode == "preflight":
            result = _remote_preflight(request)
        else:
            result = _remote_stage(request, sys.stdin.buffer)
        sys.stdout.buffer.write(canonical_bytes({"ok": True, "result": result}))
        return 0
    except Exception as exc:  # fail closed without a remote traceback
        sys.stdout.buffer.write(canonical_bytes({"ok": False, "error": str(exc)}))
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.remote_internal:
        if not args.request_b64:
            parser.error("--request-b64 is required for remote execution")
        return _remote_main(args.remote_internal, args.request_b64)
    if args.mode not in ("preflight", "stage"):
        parser.error("preflight or stage is required")
    try:
        if args.expected_hostname != EXPECTED_HOST_IDENTITY:
            raise StagingError("expected hostname must be exactly DESKTOP-RTX2080")
        result = execute(
            mode=args.mode,
            repository_specs=_repository_specs_from_args(args),
            isolated_root=args.isolated_root,
            destination=args.destination,
            toolchain_root=args.toolchain_root,
            rustc_sha256=args.rustc_sha256,
            cargo_sha256=args.cargo_sha256,
            transport=OpenSshTransport(args.host, args.ssh_bin),
        )
    except StagingError as exc:
        sys.stderr.write(f"stage_multires_wsl: {exc}\n")
        return 2
    sys.stdout.buffer.write(canonical_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
