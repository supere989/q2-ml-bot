#!/usr/bin/env python3
"""Host-bound authorization primitives for the final B2 lifecycle.

The final B2 source boundary is intentionally one-shot.  A per-user
``Path.home()`` journal is not sufficient: an exact reviewed plan could be
copied to another LAN host, or invoked through a different account, and create
an unrelated local tombstone.  This module provides the small, deterministic
binding which the final lifecycle driver must place in its reviewed plan.

Public API
==========

``build_execution_binding``
    Builds a canonical binding for the only admitted WSL execution context.
    The caller supplies an already-provisioned journal root; the host identity
    is always read from ``/etc/machine-id`` and the hostname is always
    ``DESKTOP-RTX2080``.

``validate_execution_binding``
    Re-checks that a supplied binding still describes the live host, effective
    UID, private journal directory, and requested repository/workspace scope.

``open_secure_marker_journal``
    Opens the declaration-scoped journal through a validated directory file
    descriptor.  Every marker operation is relative to that descriptor, so a
    pathname replacement cannot redirect a final-lane authorization.

``build_source_authorization_marker`` / ``verify_source_authorization_marker``
    Construct and verify the canonical v4 write-ahead tombstone payload.  The
    payload binds the full execution binding, plan digest, immutable
    declaration path, and exact primary/cold/report source outputs.  Its leaf
    remains keyed solely by declaration digest so a consumed declaration
    cannot be reopened with another workspace or plan.

``validate_final_source_authorization``
    Read an already-created marker, revalidate its live host binding, and
    prove that a final source invocation uses the exact declaration and output
    paths authorized by the marker.  The generic source producer uses this at
    its final-declaration boundary, so the lifecycle policy cannot be bypassed
    by invoking the producer directly.

The journal deliberately fails closed.  A pre-existing, malformed, unreadable,
or partially-written marker conservatively consumes its declaration; a caller
must never unlink it or retry that final cohort.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
from typing import Any, Mapping


EXECUTION_BINDING_SCHEMA = "q2-b2-final-execution-binding-v2"
SOURCE_AUTHORIZATION_MARKER_SCHEMA = "q2-b2-source-authorization-consumed-v4"
SOURCE_AUTHORIZATION_STATUS = "source-authorization-consumed"
EXPECTED_EXECUTION_HOSTNAME = "DESKTOP-RTX2080"
EXPECTED_WSL_KERNEL_FRAGMENT = "microsoft-standard-WSL2"
MACHINE_ID_PATH = Path("/etc/machine-id")

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_HEX = re.compile(r"^[0-9a-f]{40}$")
_COHORT_ID = re.compile(r"^b2g[0-9]+_final_[0-9]+$")
_STRICT_ROOT_MODE = 0o700
_STRICT_MARKER_MODE = 0o600
_MAX_MARKER_BYTES = 1024 * 1024


class FinalExecutionBindingError(ValueError):
    """Raised when final-lane host authorization cannot be established."""


class SourceAuthorizationJournalError(FinalExecutionBindingError):
    """A journal failure annotated with whether the declaration is consumed."""

    def __init__(self, message: str, *, consumed: bool) -> None:
        super().__init__(message)
        self.consumed = consumed


def _canonical_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FinalExecutionBindingError(
            "authorization value is not canonical JSON"
        ) from error


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _absolute(path: Path | str, label: str) -> Path:
    try:
        candidate = Path(path).expanduser()
    except TypeError as error:
        raise FinalExecutionBindingError(f"{label} must be a path") from error
    if not candidate.is_absolute():
        raise FinalExecutionBindingError(f"{label} must be an absolute path")
    absolute = Path(os.path.abspath(candidate))
    if str(candidate) != str(absolute):
        raise FinalExecutionBindingError(
            f"{label} must be a normalized absolute path: {candidate}"
        )
    return absolute


def _assert_no_symlink_components(path: Path, label: str) -> None:
    """Reject a symlink in any existing component of an absolute path."""

    if not path.is_absolute():
        raise FinalExecutionBindingError(f"{label} must be absolute")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            entry = os.lstat(current)
        except FileNotFoundError:
            # A later component cannot exist if this one does not.  The caller
            # decides whether the whole path itself must exist.
            return
        except OSError as error:
            raise FinalExecutionBindingError(
                f"cannot inspect {label}: {current}"
            ) from error
        if stat.S_ISLNK(entry.st_mode):
            raise FinalExecutionBindingError(
                f"{label} traverses a symlink: {current}"
            )


def _stable_regular_bytes(path: Path, label: str) -> bytes:
    """Read one regular, non-symlink file with an inode-stability check."""

    absolute = _absolute(path, label)
    _assert_no_symlink_components(absolute, label)
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise FinalExecutionBindingError(f"{label} is unavailable: {absolute}") from error
    if not stat.S_ISREG(before.st_mode):
        raise FinalExecutionBindingError(f"{label} is not a regular file: {absolute}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise FinalExecutionBindingError(f"cannot read {label}: {absolute}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise FinalExecutionBindingError(f"{label} changed before read: {absolute}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino, after.st_size) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
        ):
            raise FinalExecutionBindingError(f"{label} changed during read: {absolute}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _machine_identity_record() -> dict[str, Any]:
    """Return the canonical, non-secret machine identity digest record."""

    path = _absolute(MACHINE_ID_PATH, "machine identity path")
    if path != MACHINE_ID_PATH:
        raise FinalExecutionBindingError(
            "machine identity path must remain the fixed /etc/machine-id"
        )
    raw = _stable_regular_bytes(path, "machine identity")
    identity = raw[:-1] if raw.endswith(b"\n") else raw
    if (
        len(raw) not in (32, 33)
        or (len(raw) == 33 and raw[-1:] != b"\n")
        or re.fullmatch(rb"[0-9a-f]{32}", identity) is None
    ):
        raise FinalExecutionBindingError(
            "machine identity must be 32 lowercase hexadecimal bytes with at most one trailing LF"
        )
    return {
        "path": str(path),
        "sha256": _sha256(identity),
    }


def _live_hostname() -> str:
    hostname = socket.gethostname().split(".", 1)[0]
    if hostname != EXPECTED_EXECUTION_HOSTNAME:
        raise FinalExecutionBindingError(
            "the only admitted final execution host is DESKTOP-RTX2080"
        )
    return hostname


def _live_kernel_release() -> str:
    """Require the planned WSL2 kernel rather than any POSIX host name match."""

    release = platform.release()
    if (
        not isinstance(release, str)
        or not release
        or EXPECTED_WSL_KERNEL_FRAGMENT not in release
    ):
        raise FinalExecutionBindingError(
            "the final execution context must be DESKTOP-RTX2080 WSL2"
        )
    return release


def _live_euid() -> int:
    try:
        value = os.geteuid()
    except AttributeError as error:
        raise FinalExecutionBindingError(
            "final execution requires a POSIX effective UID"
        ) from error
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalExecutionBindingError("effective UID is invalid")
    return value


def _state_root_record(path: Path | str, *, expected_uid: int) -> dict[str, Any]:
    """Validate the pre-provisioned private journal root and record its inode."""

    root = _absolute(path, "authorization state root")
    _assert_no_symlink_components(root, "authorization state root")
    try:
        entry = os.lstat(root)
    except OSError as error:
        raise FinalExecutionBindingError(
            f"authorization state root is unavailable: {root}"
        ) from error
    if not stat.S_ISDIR(entry.st_mode):
        raise FinalExecutionBindingError(
            f"authorization state root is not a directory: {root}"
        )
    mode = stat.S_IMODE(entry.st_mode)
    if mode != _STRICT_ROOT_MODE:
        raise FinalExecutionBindingError(
            f"authorization state root must have mode 0700: {root}"
        )
    if entry.st_uid != expected_uid:
        raise FinalExecutionBindingError(
            "authorization state root owner differs from the execution UID"
        )
    return {
        "path": str(root),
        "owner_uid": entry.st_uid,
        "mode": "0700",
        "device": entry.st_dev,
        "inode": entry.st_ino,
    }


def _scope_path(path: Path | str, label: str, *, require_directory: bool) -> Path:
    absolute = _absolute(path, label)
    _assert_no_symlink_components(absolute, label)
    if require_directory:
        try:
            entry = os.lstat(absolute)
        except OSError as error:
            raise FinalExecutionBindingError(f"{label} is unavailable: {absolute}") from error
        if not stat.S_ISDIR(entry.st_mode):
            raise FinalExecutionBindingError(f"{label} is not a directory: {absolute}")
    return absolute


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_scope(
    root: Path,
    *,
    repo_root: Path | str,
    workspace: Path | str,
) -> None:
    repository = _scope_path(repo_root, "repository root", require_directory=True)
    execution_workspace = _scope_path(
        workspace, "final execution workspace", require_directory=False
    )
    if _paths_overlap(root, repository):
        raise FinalExecutionBindingError(
            "authorization state root must be outside the repository"
        )
    if _paths_overlap(root, execution_workspace):
        raise FinalExecutionBindingError(
            "authorization state root must be outside the final execution workspace"
        )


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalExecutionBindingError(f"{label} must be an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise FinalExecutionBindingError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalExecutionBindingError(f"{label} must be a nonnegative integer")
    return value


def _normalize_implementation_binding(value: object) -> dict[str, Any]:
    """Normalize the source-producer implementation identity in a marker."""

    implementation = _require_mapping(value, "source implementation binding")
    expected = {
        "repository_commit",
        "repository_tree",
        "git_clean",
        "atlas_analyzer_authority_sha256",
        "atlas_analyzer_authority_file_count",
        "generator_sha256",
        "routes_sha256",
    }
    _require_exact_keys(implementation, expected, "source implementation binding")
    normalized: dict[str, Any] = {}
    for key in ("repository_commit", "repository_tree"):
        raw = implementation.get(key)
        if not isinstance(raw, str) or _GIT_HEX.fullmatch(raw) is None:
            raise FinalExecutionBindingError(
                f"source implementation binding {key} is malformed"
            )
        normalized[key] = raw
    if implementation.get("git_clean") is not True:
        raise FinalExecutionBindingError(
            "source implementation binding must be clean"
        )
    normalized["git_clean"] = True
    for key in (
        "atlas_analyzer_authority_sha256",
        "generator_sha256",
        "routes_sha256",
    ):
        raw = implementation.get(key)
        if not isinstance(raw, str) or _HEX64.fullmatch(raw) is None:
            raise FinalExecutionBindingError(
                f"source implementation binding {key} is malformed"
            )
        normalized[key] = raw
    count = _require_nonnegative_int(
        implementation.get("atlas_analyzer_authority_file_count"),
        "source implementation binding analyzer file count",
    )
    if count < 1:
        raise FinalExecutionBindingError(
            "source implementation binding analyzer file count must be positive"
        )
    normalized["atlas_analyzer_authority_file_count"] = count
    return normalized


def _normalize_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the canonical binding shape without reading live host state."""

    outer = _require_mapping(binding, "execution binding")
    _require_exact_keys(
        outer, {"schema", "host", "state_root"}, "execution binding"
    )
    if outer.get("schema") != EXECUTION_BINDING_SCHEMA:
        raise FinalExecutionBindingError("execution binding schema differs")

    host = _require_mapping(outer.get("host"), "execution binding host")
    _require_exact_keys(
        host,
        {"hostname", "kernel_release", "machine_identity", "euid"},
        "execution binding host",
    )
    hostname = host.get("hostname")
    if hostname != EXPECTED_EXECUTION_HOSTNAME:
        raise FinalExecutionBindingError("execution binding hostname differs")
    kernel_release = host.get("kernel_release")
    if (
        not isinstance(kernel_release, str)
        or EXPECTED_WSL_KERNEL_FRAGMENT not in kernel_release
    ):
        raise FinalExecutionBindingError("execution binding kernel release differs")
    machine = _require_mapping(host.get("machine_identity"), "machine identity")
    _require_exact_keys(machine, {"path", "sha256"}, "machine identity")
    machine_path = machine.get("path")
    if machine_path != str(MACHINE_ID_PATH):
        raise FinalExecutionBindingError(
            "execution binding machine identity path differs from /etc/machine-id"
        )
    machine_digest = machine.get("sha256")
    if not isinstance(machine_digest, str) or _HEX64.fullmatch(machine_digest) is None:
        raise FinalExecutionBindingError("execution binding machine identity digest is invalid")
    euid = _require_nonnegative_int(host.get("euid"), "execution binding euid")

    state_root = _require_mapping(outer.get("state_root"), "authorization state root")
    _require_exact_keys(
        state_root,
        {"path", "owner_uid", "mode", "device", "inode"},
        "authorization state root",
    )
    raw_root = state_root.get("path")
    if not isinstance(raw_root, str):
        raise FinalExecutionBindingError("authorization state root path is invalid")
    root = _absolute(raw_root, "authorization state root")
    if raw_root != str(root):
        raise FinalExecutionBindingError(
            "authorization state root path is not normalized"
        )
    owner_uid = _require_nonnegative_int(
        state_root.get("owner_uid"), "authorization state root owner UID"
    )
    if state_root.get("mode") != "0700":
        raise FinalExecutionBindingError("authorization state root mode differs")
    device = _require_nonnegative_int(
        state_root.get("device"), "authorization state root device"
    )
    inode = _require_nonnegative_int(
        state_root.get("inode"), "authorization state root inode"
    )
    if inode == 0:
        raise FinalExecutionBindingError("authorization state root inode is invalid")

    return {
        "schema": EXECUTION_BINDING_SCHEMA,
        "host": {
            "hostname": hostname,
            "kernel_release": kernel_release,
            "machine_identity": {
                "path": machine_path,
                "sha256": machine_digest,
            },
            "euid": euid,
        },
        "state_root": {
            "path": str(root),
            "owner_uid": owner_uid,
            "mode": "0700",
            "device": device,
            "inode": inode,
        },
    }


def build_execution_binding(
    *,
    state_root: Path | str,
    repo_root: Path | str,
    workspace: Path | str,
) -> dict[str, Any]:
    """Build a binding for the current admitted WSL host and explicit root.

    ``state_root`` must already exist, be an ordinary non-symlink directory
    owned by the effective UID, and have exactly mode ``0700``.  It must not
    overlap either the repository or the final execution workspace.
    """

    hostname = _live_hostname()
    kernel_release = _live_kernel_release()
    euid = _live_euid()
    state = _state_root_record(state_root, expected_uid=euid)
    _validate_scope(Path(state["path"]), repo_root=repo_root, workspace=workspace)
    binding = {
        "schema": EXECUTION_BINDING_SCHEMA,
        "host": {
            "hostname": hostname,
            "kernel_release": kernel_release,
            "machine_identity": _machine_identity_record(),
            "euid": euid,
        },
        "state_root": state,
    }
    # Keep the construction and validation paths identical so no unvalidated
    # field can enter a reviewed plan.
    return validate_execution_binding(
        binding, repo_root=repo_root, workspace=workspace
    )


def validate_execution_binding(
    binding: Mapping[str, Any],
    *,
    repo_root: Path | str,
    workspace: Path | str,
) -> dict[str, Any]:
    """Fail closed unless *binding* exactly matches the live WSL context."""

    normalized = _normalize_binding(binding)
    hostname = _live_hostname()
    kernel_release = _live_kernel_release()
    euid = _live_euid()
    host = normalized["host"]
    if host["hostname"] != hostname:
        raise FinalExecutionBindingError("execution hostname differs from plan binding")
    if host["kernel_release"] != kernel_release:
        raise FinalExecutionBindingError(
            "execution kernel release differs from plan binding"
        )
    if host["euid"] != euid:
        raise FinalExecutionBindingError("execution effective UID differs from plan binding")
    if host["machine_identity"] != _machine_identity_record():
        raise FinalExecutionBindingError(
            "execution machine identity differs from plan binding"
        )

    state = normalized["state_root"]
    live_state = _state_root_record(Path(state["path"]), expected_uid=euid)
    if live_state != state:
        raise FinalExecutionBindingError(
            "authorization state root identity differs from plan binding"
        )
    _validate_scope(
        Path(live_state["path"]), repo_root=repo_root, workspace=workspace
    )
    return normalized


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise FinalExecutionBindingError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _marker_leaf_name(declaration_sha256: object) -> str:
    return f"{_require_digest(declaration_sha256, 'declaration digest')}.json"


def _required_open_flags(*names: str) -> int:
    """Return Linux descriptor flags or fail closed when one is unavailable."""

    flags = 0
    for name in names:
        value = getattr(os, name, None)
        if not isinstance(value, int):
            raise FinalExecutionBindingError(
                f"final authorization requires OS flag {name}"
            )
        flags |= value
    return flags


def _root_fd_matches(binding: Mapping[str, Any], descriptor: int) -> None:
    """Require an already-open root descriptor to match the bound directory."""

    state_root = binding["state_root"]
    try:
        entry = os.fstat(descriptor)
    except OSError as error:
        raise FinalExecutionBindingError(
            "cannot inspect authorization state-root descriptor"
        ) from error
    expected = (
        state_root["device"],
        state_root["inode"],
        state_root["owner_uid"],
        _STRICT_ROOT_MODE,
    )
    actual = (
        entry.st_dev,
        entry.st_ino,
        entry.st_uid,
        stat.S_IMODE(entry.st_mode),
    )
    if not stat.S_ISDIR(entry.st_mode) or actual != expected:
        raise FinalExecutionBindingError(
            "authorization state-root descriptor differs from its bound identity"
        )


def _marker_stat_identity(entry: os.stat_result) -> tuple[int, int, int, int, int, int]:
    """Return the identity fields that must not change while a marker is read."""

    try:
        return (
            entry.st_dev,
            entry.st_ino,
            entry.st_size,
            entry.st_mtime_ns,
            entry.st_ctime_ns,
            stat.S_IMODE(entry.st_mode),
        )
    except AttributeError as error:
        raise FinalExecutionBindingError(
            "final authorization requires nanosecond marker stat fields"
        ) from error


class SecureMarkerJournal:
    """A directory-FD-anchored authorization journal.

    The pathname is used only to acquire a bound root descriptor and to report
    audit evidence.  All leaf I/O uses ``dir_fd`` with a one-component,
    declaration-derived filename.  This is deliberately a small capability:
    it cannot address arbitrary files below the state root.
    """

    def __init__(
        self,
        binding: Mapping[str, Any],
        *,
        repo_root: Path | str,
        workspace: Path | str,
    ) -> None:
        self._binding = validate_execution_binding(
            binding, repo_root=repo_root, workspace=workspace
        )
        self._repo_root = Path(repo_root)
        self._workspace = Path(workspace)
        self._root = Path(self._binding["state_root"]["path"])
        self._descriptor: int | None = None

    def __enter__(self) -> "SecureMarkerJournal":
        flags = _required_open_flags(
            "O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"
        )
        try:
            descriptor = os.open(str(self._root), flags)
        except OSError as error:
            raise FinalExecutionBindingError(
                "cannot open authorization state-root descriptor"
            ) from error
        self._descriptor = descriptor
        try:
            _root_fd_matches(self._binding, descriptor)
            # A root replacement between path validation and descriptor open is
            # caught by fstat above.  A replacement after open is caught here.
            self.revalidate_path_binding()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        """Close the anchored root descriptor exactly once."""

        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is not None:
            os.close(descriptor)

    def _fd(self) -> int:
        if self._descriptor is None:
            raise FinalExecutionBindingError("authorization journal is not open")
        return self._descriptor

    def marker_path(self, declaration_sha256: object) -> Path:
        """Return an audit-only spelling of a declaration-derived leaf path."""

        return self._root / _marker_leaf_name(declaration_sha256)

    def exists(self, declaration_sha256: object) -> bool:
        """Test one declaration leaf through the anchored root descriptor."""

        leaf = _marker_leaf_name(declaration_sha256)
        try:
            os.stat(leaf, dir_fd=self._fd(), follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise FinalExecutionBindingError(
                f"cannot inspect source authorization marker: {leaf}"
            ) from error
        return True

    def _marker_mode_and_owner(self, entry: os.stat_result, leaf: str) -> None:
        if not stat.S_ISREG(entry.st_mode):
            raise FinalExecutionBindingError(
                f"source authorization marker is not a regular file: {leaf}"
            )
        if stat.S_IMODE(entry.st_mode) != _STRICT_MARKER_MODE:
            raise FinalExecutionBindingError(
                f"source authorization marker must have mode 0600: {leaf}"
            )
        if entry.st_uid != self._binding["host"]["euid"]:
            raise FinalExecutionBindingError(
                f"source authorization marker owner differs from execution UID: {leaf}"
            )
        if entry.st_nlink != 1:
            raise FinalExecutionBindingError(
                f"source authorization marker link count differs: {leaf}"
            )
        if entry.st_size < 0 or entry.st_size > _MAX_MARKER_BYTES:
            raise FinalExecutionBindingError(
                f"source authorization marker size is invalid: {leaf}"
            )

    def revalidate_path_binding(self) -> None:
        """Ensure both pathname and held descriptor still name the bound root."""

        live = validate_execution_binding(
            self._binding, repo_root=self._repo_root, workspace=self._workspace
        )
        if _canonical_bytes(live) != _canonical_bytes(self._binding):
            raise FinalExecutionBindingError(
                "authorization state-root binding changed while journal was open"
            )
        _root_fd_matches(self._binding, self._fd())

    def read(self, declaration_sha256: object) -> bytes | None:
        """Read one stable marker leaf through the anchored root descriptor."""

        leaf = _marker_leaf_name(declaration_sha256)
        descriptor = self._fd()
        try:
            before = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise FinalExecutionBindingError(
                f"cannot inspect source authorization marker: {leaf}"
            ) from error
        self._marker_mode_and_owner(before, leaf)
        before_identity = _marker_stat_identity(before)
        flags = _required_open_flags(
            "O_RDONLY", "O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC"
        )
        try:
            marker_fd = os.open(leaf, flags, dir_fd=descriptor)
        except OSError as error:
            raise FinalExecutionBindingError(
                f"source authorization marker changed before read: {leaf}"
            ) from error
        try:
            opened = os.fstat(marker_fd)
            self._marker_mode_and_owner(opened, leaf)
            if _marker_stat_identity(opened) != before_identity:
                raise FinalExecutionBindingError(
                    f"source authorization marker changed before read: {leaf}"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                block = os.read(marker_fd, min(1024 * 1024, _MAX_MARKER_BYTES + 1))
                if not block:
                    break
                total += len(block)
                if total > _MAX_MARKER_BYTES:
                    raise FinalExecutionBindingError(
                        f"source authorization marker size is invalid: {leaf}"
                    )
                chunks.append(block)
            after = os.fstat(marker_fd)
            self._marker_mode_and_owner(after, leaf)
            if _marker_stat_identity(after) != before_identity:
                raise FinalExecutionBindingError(
                    f"source authorization marker changed during read: {leaf}"
                )
            return b"".join(chunks)
        finally:
            os.close(marker_fd)

    def create(self, declaration_sha256: object, payload: bytes) -> tuple[bytes, bool]:
        """Create a durable marker, or return an existing consumed marker.

        Once ``O_EXCL`` succeeds, any later failure is explicitly marked
        ``consumed=True``.  The new leaf is deliberately never unlinked.
        """

        if not isinstance(payload, bytes):
            raise FinalExecutionBindingError("source authorization payload is not bytes")
        if len(payload) > _MAX_MARKER_BYTES:
            raise FinalExecutionBindingError("source authorization payload is too large")
        leaf = _marker_leaf_name(declaration_sha256)
        self.revalidate_path_binding()
        flags = _required_open_flags(
            "O_WRONLY", "O_CREAT", "O_EXCL", "O_NOFOLLOW", "O_CLOEXEC"
        )
        try:
            marker_fd = os.open(leaf, flags, _STRICT_MARKER_MODE, dir_fd=self._fd())
        except FileExistsError:
            # A collision may represent another process which already crossed
            # the source boundary.  Even malformed or unreadable contents are
            # conservatively terminal for this declaration.
            try:
                existing = self.read(declaration_sha256)
                if existing is None:
                    raise FinalExecutionBindingError(
                        "source authorization marker collision disappeared"
                    )
                self.revalidate_path_binding()
            except FinalExecutionBindingError as error:
                raise SourceAuthorizationJournalError(
                    f"source authorization marker collision is terminal: {error}",
                    consumed=True,
                ) from error
            return existing, False
        except OSError as error:
            raise FinalExecutionBindingError(
                f"cannot create source authorization marker: {leaf}"
            ) from error

        try:
            offset = 0
            while offset < len(payload):
                written = os.write(marker_fd, payload[offset:])
                if written <= 0:
                    raise OSError("short source authorization marker write")
                offset += written
            os.fchmod(marker_fd, _STRICT_MARKER_MODE)
            os.fsync(marker_fd)
            os.close(marker_fd)
            marker_fd = -1
            os.fsync(self._fd())
            self.revalidate_path_binding()
        except BaseException as error:
            raise SourceAuthorizationJournalError(
                "source authorization journal creation was incomplete; declaration remains consumed",
                consumed=True,
            ) from error
        finally:
            if marker_fd >= 0:
                os.close(marker_fd)
        return payload, True


def open_secure_marker_journal(
    binding: Mapping[str, Any],
    *,
    repo_root: Path | str,
    workspace: Path | str,
) -> SecureMarkerJournal:
    """Return a context manager for the anchored final-lane journal."""

    return SecureMarkerJournal(
        binding, repo_root=repo_root, workspace=workspace
    )


def resolve_secure_marker_path(
    binding: Mapping[str, Any],
    declaration_sha256: object,
    *,
    repo_root: Path | str,
    workspace: Path | str,
) -> Path:
    """Return an audit-only marker spelling after an anchored inspection.

    New code must use :func:`open_secure_marker_journal` for marker I/O.
    This compatibility helper never grants a path-based write capability.
    """

    with open_secure_marker_journal(
        binding, repo_root=repo_root, workspace=workspace
    ) as journal:
        # Reject special/symlink leaves now, but do not let callers use this
        # returned spelling for security-sensitive I/O.
        journal.read(declaration_sha256)
        journal.revalidate_path_binding()
        return journal.marker_path(declaration_sha256)


def _require_cohort_id(value: object) -> str:
    if not isinstance(value, str) or _COHORT_ID.fullmatch(value) is None:
        raise FinalExecutionBindingError("final cohort ID is malformed")
    return value


def _normalized_workspace(value: Path | str) -> str:
    path = _absolute(value, "final execution workspace")
    _assert_no_symlink_components(path, "final execution workspace")
    return str(path)


def _normalized_regular_path(value: Path | str, label: str) -> str:
    """Return a direct, non-symlink regular-file path for marker identity."""

    path = _absolute(value, label)
    _assert_no_symlink_components(path, label)
    try:
        entry = os.lstat(path)
    except OSError as error:
        raise FinalExecutionBindingError(f"{label} is unavailable: {path}") from error
    if not stat.S_ISREG(entry.st_mode):
        raise FinalExecutionBindingError(f"{label} is not a regular file: {path}")
    return str(path)


def _normalized_source_outputs(
    *,
    workspace: Path | str,
    source_output: Path | str,
    source_cold: Path | str,
    source_report: Path | str,
) -> dict[str, str]:
    """Normalize the three source outputs and keep them inside one workspace."""

    workspace_path = Path(_normalized_workspace(workspace))
    values = {
        "primary": source_output,
        "cold": source_cold,
        "report": source_report,
    }
    normalized: dict[str, str] = {}
    paths: list[Path] = []
    for name, value in values.items():
        label = f"source {name} output"
        path = _absolute(value, label)
        _assert_no_symlink_components(path, label)
        try:
            relative = path.relative_to(workspace_path)
        except ValueError as error:
            raise FinalExecutionBindingError(
                f"{label} must remain inside the final execution workspace"
            ) from error
        if not relative.parts:
            raise FinalExecutionBindingError(
                f"{label} cannot equal the final execution workspace"
            )
        normalized[name] = str(path)
        paths.append(path)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise FinalExecutionBindingError(
                    "source primary/cold/report outputs must be distinct and disjoint"
                )
    return normalized


def build_source_authorization_marker(
    *,
    binding: Mapping[str, Any],
    cohort_id: object,
    declaration_sha256: object,
    declaration_path: Path | str,
    plan_sha256: object,
    workspace: Path | str,
    repo_root: Path | str,
    implementation: Mapping[str, Any],
    source_output: Path | str,
    source_cold: Path | str,
    source_report: Path | str,
) -> bytes:
    """Construct the canonical v4 write-ahead source-authorization payload.

    This validates the live binding before returning bytes; it intentionally
    does not create or pathname-resolve the leaf.  The anchored journal owns
    all leaf I/O.  ``plan_sha256`` is recorded for audit, while the declaration
    digest remains the no-retry key.
    """

    normalized = validate_execution_binding(
        binding, repo_root=repo_root, workspace=workspace
    )
    normalized_workspace = _normalized_workspace(workspace)
    outputs = _normalized_source_outputs(
        workspace=normalized_workspace,
        source_output=source_output,
        source_cold=source_cold,
        source_report=source_report,
    )
    body = {
        "schema": SOURCE_AUTHORIZATION_MARKER_SCHEMA,
        "status": SOURCE_AUTHORIZATION_STATUS,
        "cohort_id": _require_cohort_id(cohort_id),
        "declaration_sha256": _require_digest(
            declaration_sha256, "declaration digest"
        ),
        "declaration_path": _normalized_regular_path(
            declaration_path, "marker declaration"
        ),
        "plan_sha256": _require_digest(plan_sha256, "plan digest"),
        "workspace": normalized_workspace,
        "implementation": _normalize_implementation_binding(implementation),
        "source_outputs": outputs,
        "stage_started": "source",
        "immutable_no_retry": True,
        "execution_binding": normalized,
    }
    return _canonical_bytes(body)


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalExecutionBindingError(
                f"duplicate source authorization marker key: {key!r}"
            )
        result[key] = value
    return result


def _parse_source_authorization_marker(payload: bytes) -> dict[str, Any]:
    """Decode and normalize a v4 marker before comparing it to a caller."""

    if not isinstance(payload, bytes):
        raise FinalExecutionBindingError("source authorization marker is not bytes")
    try:
        loaded = json.loads(
            payload,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                FinalExecutionBindingError(
                    f"non-finite source authorization marker token: {token}"
                )
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FinalExecutionBindingError(
            "source authorization marker is unreadable"
        ) from error
    marker = _require_mapping(loaded, "source authorization marker")
    _require_exact_keys(
        marker,
        {
            "schema",
            "status",
            "cohort_id",
            "declaration_sha256",
            "declaration_path",
            "plan_sha256",
            "workspace",
            "implementation",
            "source_outputs",
            "stage_started",
            "immutable_no_retry",
            "execution_binding",
        },
        "source authorization marker",
    )
    if payload != _canonical_bytes(marker):
        raise FinalExecutionBindingError(
            "source authorization marker is not canonical JSON"
        )
    if marker.get("schema") != SOURCE_AUTHORIZATION_MARKER_SCHEMA:
        raise FinalExecutionBindingError("source authorization marker schema differs")
    if marker.get("status") != SOURCE_AUTHORIZATION_STATUS:
        raise FinalExecutionBindingError("source authorization marker status differs")
    if marker.get("stage_started") != "source":
        raise FinalExecutionBindingError("source authorization marker stage differs")
    if marker.get("immutable_no_retry") is not True:
        raise FinalExecutionBindingError(
            "source authorization marker immutable no-retry flag differs"
        )
    cohort_id = _require_cohort_id(marker.get("cohort_id"))
    declaration_sha256 = _require_digest(
        marker.get("declaration_sha256"), "marker declaration digest"
    )
    declaration_path = _normalized_regular_path(
        marker.get("declaration_path"), "marker declaration"
    )
    plan_sha256 = _require_digest(marker.get("plan_sha256"), "marker plan digest")
    workspace = _normalized_workspace(marker.get("workspace"))
    implementation = _normalize_implementation_binding(marker.get("implementation"))
    outputs_value = _require_mapping(
        marker.get("source_outputs"), "marker source outputs"
    )
    _require_exact_keys(
        outputs_value, {"primary", "cold", "report"}, "marker source outputs"
    )
    outputs = _normalized_source_outputs(
        workspace=workspace,
        source_output=outputs_value.get("primary"),
        source_cold=outputs_value.get("cold"),
        source_report=outputs_value.get("report"),
    )
    binding = _normalize_binding(
        _require_mapping(marker.get("execution_binding"), "marker execution binding")
    )
    return {
        "schema": SOURCE_AUTHORIZATION_MARKER_SCHEMA,
        "status": SOURCE_AUTHORIZATION_STATUS,
        "cohort_id": cohort_id,
        "declaration_sha256": declaration_sha256,
        "declaration_path": declaration_path,
        "plan_sha256": plan_sha256,
        "workspace": workspace,
        "implementation": implementation,
        "source_outputs": outputs,
        "stage_started": "source",
        "immutable_no_retry": True,
        "execution_binding": binding,
    }


def verify_source_authorization_marker(
    payload: bytes,
    *,
    binding: Mapping[str, Any],
    cohort_id: object,
    declaration_sha256: object,
    repo_root: Path | str,
    workspace: Path | str,
    implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a v4 tombstone against the live plan binding.

    The recorded plan digest and workspace intentionally need not match the
    current plan: any valid tombstone for the same declaration is terminal and
    blocks a retry under a different workspace/plan.
    """

    marker = _parse_source_authorization_marker(payload)
    expected_cohort = _require_cohort_id(cohort_id)
    expected_declaration = _require_digest(declaration_sha256, "declaration digest")
    if marker["cohort_id"] != expected_cohort:
        raise FinalExecutionBindingError(
            "source authorization marker cohort identity differs"
        )
    if marker["declaration_sha256"] != expected_declaration:
        raise FinalExecutionBindingError(
            "source authorization marker declaration identity differs"
        )
    expected_binding = validate_execution_binding(
        binding, repo_root=repo_root, workspace=workspace
    )
    marker_binding = marker["execution_binding"]
    if _canonical_bytes(marker_binding) != _canonical_bytes(expected_binding):
        raise FinalExecutionBindingError(
            "source authorization marker execution binding differs"
        )
    if implementation is not None and _canonical_bytes(
        marker["implementation"]
    ) != _canonical_bytes(_normalize_implementation_binding(implementation)):
        raise FinalExecutionBindingError(
            "source authorization marker implementation binding differs"
        )
    return marker


def validate_final_source_authorization(
    marker_path: Path | str,
    *,
    declaration_path: Path | str,
    declaration_sha256: object,
    cohort_id: object,
    source_output: Path | str,
    source_cold: Path | str,
    source_report: Path | str,
    repo_root: Path | str,
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the marker capability required by an active final producer.

    The marker is not merely an audit record: this validates its canonical
    payload, live host/UID/machine/journal binding, declaration identity, and
    all three exact source output paths before an active final source producer
    is allowed to create a byte.  Non-active declarations do not call this
    function and retain their reusable qualification/disposable workflow.
    """

    # The supplied path is untrusted capability input.  Read it only to obtain
    # a candidate binding, then re-open the marker through that binding's
    # anchored directory descriptor and authorize solely from the second read.
    path = _absolute(marker_path, "final source authorization marker")
    preliminary = _parse_source_authorization_marker(
        _stable_regular_bytes(path, "final source authorization marker")
    )
    workspace = Path(preliminary["workspace"])
    with open_secure_marker_journal(
        preliminary["execution_binding"],
        repo_root=repo_root,
        workspace=workspace,
    ) as journal:
        expected_marker = journal.marker_path(declaration_sha256)
        if path != expected_marker:
            raise FinalExecutionBindingError(
                "final source authorization marker path differs from the declaration journal leaf"
            )
        payload = journal.read(declaration_sha256)
        if payload is None:
            raise FinalExecutionBindingError(
                "final source authorization marker is absent from the declaration journal"
            )
        marker = _parse_source_authorization_marker(payload)
        binding = validate_execution_binding(
            marker["execution_binding"], repo_root=repo_root, workspace=workspace
        )
        verified = verify_source_authorization_marker(
            payload,
            binding=binding,
            cohort_id=cohort_id,
            declaration_sha256=declaration_sha256,
            repo_root=repo_root,
            workspace=workspace,
            implementation=implementation,
        )
        expected_declaration_path = _normalized_regular_path(
            declaration_path, "active final declaration"
        )
        if verified["declaration_path"] != expected_declaration_path:
            raise FinalExecutionBindingError(
                "final source authorization declaration path differs"
            )
        expected_outputs = _normalized_source_outputs(
            workspace=workspace,
            source_output=source_output,
            source_cold=source_cold,
            source_report=source_report,
        )
        if verified["source_outputs"] != expected_outputs:
            raise FinalExecutionBindingError(
                "final source authorization output paths differ"
            )
        journal.revalidate_path_binding()
        return verified


__all__ = [
    "EXECUTION_BINDING_SCHEMA",
    "EXPECTED_EXECUTION_HOSTNAME",
    "FinalExecutionBindingError",
    "MACHINE_ID_PATH",
    "SecureMarkerJournal",
    "SOURCE_AUTHORIZATION_MARKER_SCHEMA",
    "SourceAuthorizationJournalError",
    "build_execution_binding",
    "build_source_authorization_marker",
    "open_secure_marker_journal",
    "resolve_secure_marker_path",
    "validate_final_source_authorization",
    "validate_execution_binding",
    "verify_source_authorization_marker",
]
