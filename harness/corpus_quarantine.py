"""Fail-closed archive quarantine and corpus provenance for authored BSPs.

The quarantine scanner never extracts or executes archive content.  It hashes
bounded members in place, rejects active content and ambiguous paths, and hands
classic Quake II BSP bytes to :mod:`harness.ibsp38` for metadata-only parsing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import fnmatch
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import struct
from typing import BinaryIO, Iterable, Mapping, Sequence
import unicodedata
import zipfile

from .ibsp38 import BspLimits, BspMetadata, parse_ibsp38


class QuarantineError(ValueError):
    """Raised when an archive or its provenance fails admission."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_archive_bytes: int = 512 * 1024 * 1024
    max_total_uncompressed_bytes: int = 1024 * 1024 * 1024
    max_member_bytes: int = 128 * 1024 * 1024
    max_members: int = 8192
    max_compression_ratio: float = 100.0
    max_path_bytes: int = 512
    max_bsp_members: int = 256
    chunk_bytes: int = 1024 * 1024


@dataclass(frozen=True)
class ArchiveMember:
    path: str
    byte_count: int
    compressed_bytes: int
    sha256: str
    asset_class: str


@dataclass(frozen=True)
class AssetInventory:
    by_class: Mapping[str, int]
    members: tuple[str, ...]
    unresolved_texture_references: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "by_class": dict(sorted(self.by_class.items())),
            "members": list(self.members),
            "unresolved_texture_references": list(
                self.unresolved_texture_references
            ),
        }


@dataclass(frozen=True)
class ArchiveReport:
    schema: str
    archive_format: str
    archive_sha256: str
    archive_bytes: int
    total_uncompressed_bytes: int
    members: tuple[ArchiveMember, ...]
    bsp_metadata: Mapping[str, BspMetadata]
    assets: AssetInventory

    def to_dict(self, *, include_bsp_entities: bool = False) -> dict:
        return {
            "schema": self.schema,
            "archive_format": self.archive_format,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "total_uncompressed_bytes": self.total_uncompressed_bytes,
            "members": [asdict(member) for member in self.members],
            "bsp_metadata": {
                path: metadata.to_dict(include_entities=include_bsp_entities)
                for path, metadata in sorted(self.bsp_metadata.items())
            },
            "assets": self.assets.to_dict(),
        }


@dataclass(frozen=True)
class ProvenanceRecord:
    canonical_id: str
    bsp_member: str
    aliases: tuple[str, ...]
    source_url: str | None
    manual_origin: str | None
    author: str
    license_name: str
    license_evidence: str
    redistribution: str
    archive_sha256: str
    bsp_sha256: str

    def validate(self) -> None:
        allowed_id_characters = "abcdefghijklmnopqrstuvwxyz0123456789._-"
        if not self.canonical_id or any(
            character not in allowed_id_characters
            for character in self.canonical_id
        ):
            raise QuarantineError(
                f"invalid canonical corpus ID {self.canonical_id!r}"
            )
        if bool(self.source_url) == bool(self.manual_origin):
            raise QuarantineError(
                f"{self.canonical_id}: exactly one source URL or manual origin is required"
            )
        if not self.author.strip():
            raise QuarantineError(f"{self.canonical_id}: author is required")
        if not self.license_name.strip() or not self.license_evidence.strip():
            raise QuarantineError(
                f"{self.canonical_id}: license name and evidence are required"
            )
        if self.redistribution not in {"analysis-only", "redistributable"}:
            raise QuarantineError(
                f"{self.canonical_id}: invalid redistribution status"
            )
        for name, digest in (
            ("archive", self.archive_sha256),
            ("BSP", self.bsp_sha256),
        ):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise QuarantineError(
                    f"{self.canonical_id}: invalid {name} SHA-256"
                )
        alias_set = set()
        for alias in self.aliases:
            if (
                not alias
                or any(character not in allowed_id_characters for character in alias)
                or alias == self.canonical_id
                or alias in alias_set
            ):
                raise QuarantineError(
                    f"{self.canonical_id}: invalid or duplicate alias {alias!r}"
                )
            alias_set.add(alias)

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProvenanceRecord":
        record = cls(
            canonical_id=str(value.get("canonical_id", "")),
            bsp_member=str(value.get("bsp_member", "")),
            aliases=tuple(str(alias) for alias in value.get("aliases", ())),
            source_url=(
                str(value["source_url"]) if value.get("source_url") else None
            ),
            manual_origin=(
                str(value["manual_origin"]) if value.get("manual_origin") else None
            ),
            author=str(value.get("author", "")),
            license_name=str(value.get("license_name", "")),
            license_evidence=str(value.get("license_evidence", "")),
            redistribution=str(value.get("redistribution", "")),
            archive_sha256=str(value.get("archive_sha256", "")),
            bsp_sha256=str(value.get("bsp_sha256", "")),
        )
        record.validate()
        return record


@dataclass(frozen=True)
class CorpusEntry:
    canonical_id: str
    aliases: tuple[str, ...]
    bsp_member: str
    metadata: BspMetadata
    provenance: ProvenanceRecord


@dataclass(frozen=True)
class DuplicateClassification:
    left: str
    right: str
    kind: str
    score: float
    canonical_id: str | None


_NESTED_ARCHIVE_SUFFIXES = {
    ".zip", ".pk3", ".pak", ".tar", ".gz", ".tgz", ".bz2", ".xz",
    ".7z", ".rar", ".cab", ".iso",
}

_ACTIVE_SUFFIXES = {
    ".exe", ".com", ".dll", ".so", ".dylib", ".msi", ".scr",
    ".bat", ".cmd", ".ps1", ".sh", ".py", ".pyc", ".pl", ".rb",
    ".js", ".jar", ".vbs", ".app", ".cfg",
}

_ASSET_SUFFIXES = {
    ".bsp": "bsp",
    ".wal": "texture",
    ".pcx": "texture",
    ".tga": "texture",
    ".png": "texture",
    ".jpg": "texture",
    ".jpeg": "texture",
    ".wav": "audio",
    ".ogg": "audio",
    ".md2": "model",
    ".sp2": "model",
    ".dm2": "demo",
    ".txt": "documentation",
    ".md": "documentation",
    ".nfo": "documentation",
    ".loc": "metadata",
    ".arena": "metadata",
}

_ACTIVE_MAGICS = (
    b"MZ",
    b"\x7fELF",
    b"#!",
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
)


def _sha256_path(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_member_path(name: str, *, directory: bool, limit: int) -> str:
    if "\x00" in name:
        raise QuarantineError("archive member path contains NUL")
    candidate = name.replace("\\", "/")
    if directory:
        candidate = candidate.rstrip("/")
    if not candidate or candidate.startswith("/"):
        raise QuarantineError(f"unsafe archive member path {name!r}")
    if len(candidate.encode("utf-8")) > limit:
        raise QuarantineError(f"archive member path exceeds {limit} bytes")
    components = candidate.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise QuarantineError(f"unsafe archive member path {name!r}")
    if len(components[0]) >= 2 and components[0][1] == ":":
        raise QuarantineError(f"absolute drive path in archive member {name!r}")
    return PurePosixPath(*components).as_posix()


def _case_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _asset_class(path: str) -> str:
    return _ASSET_SUFFIXES.get(PurePosixPath(path).suffix.lower(), "other")


def _validate_member_name(path: str) -> None:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in _NESTED_ARCHIVE_SUFFIXES:
        raise QuarantineError(f"nested archive is forbidden: {path}")
    if suffix in _ACTIVE_SUFFIXES:
        raise QuarantineError(f"executable or script payload is forbidden: {path}")


def _hash_stream(
    handle: BinaryIO,
    *,
    expected_bytes: int,
    chunk_bytes: int,
    capture: bool,
) -> tuple[str, bytes | None]:
    digest = hashlib.sha256()
    received = 0
    chunks: list[bytes] | None = [] if capture else None
    prefix = b""
    while received < expected_bytes:
        chunk = handle.read(min(chunk_bytes, expected_bytes - received))
        if not chunk:
            break
        received += len(chunk)
        if len(prefix) < 8:
            prefix += chunk[:8 - len(prefix)]
        digest.update(chunk)
        if chunks is not None:
            chunks.append(chunk)
    if received != expected_bytes:
        raise QuarantineError(
            f"archive member size mismatch: declared {expected_bytes}, read {received}"
        )
    if any(prefix.startswith(magic) for magic in _ACTIVE_MAGICS):
        raise QuarantineError("member contains executable or script magic")
    return digest.hexdigest(), (b"".join(chunks) if chunks is not None else None)


def _validate_sizes(
    *,
    size: int,
    compressed: int,
    path: str,
    limits: ArchiveLimits,
) -> None:
    if size < 0 or compressed < 0:
        raise QuarantineError(f"negative member size for {path}")
    if size > limits.max_member_bytes:
        raise QuarantineError(
            f"member {path} is {size} bytes; limit is {limits.max_member_bytes}"
        )
    ratio = size / max(1, compressed)
    if ratio > limits.max_compression_ratio:
        raise QuarantineError(
            f"member {path} compression ratio {ratio:.2f} exceeds "
            f"{limits.max_compression_ratio:.2f}"
        )


def _scan_zip(
    path: Path,
    limits: ArchiveLimits,
    bsp_limits: BspLimits,
    include_bsp: Sequence[str],
) -> tuple[list[ArchiveMember], dict[str, BspMetadata]]:
    members: list[ArchiveMember] = []
    bsp_metadata: dict[str, BspMetadata] = {}
    seen: set[str] = set()
    total = 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > limits.max_members:
            raise QuarantineError(
                f"archive has {len(infos)} members; limit is {limits.max_members}"
            )
        normalized: list[tuple[zipfile.ZipInfo, str]] = []
        for info in infos:
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(unix_mode):
                raise QuarantineError(f"symlink member is forbidden: {info.filename}")
            if info.flag_bits & 0x1:
                raise QuarantineError(f"encrypted member is forbidden: {info.filename}")
            directory = info.is_dir()
            member_path = _normalize_member_path(
                info.filename, directory=directory, limit=limits.max_path_bytes
            )
            key = _case_key(member_path)
            if key in seen:
                raise QuarantineError(
                    f"duplicate or case-colliding member path: {member_path}"
                )
            seen.add(key)
            if directory:
                continue
            _validate_member_name(member_path)
            _validate_sizes(
                size=info.file_size,
                compressed=info.compress_size,
                path=member_path,
                limits=limits,
            )
            total += info.file_size
            if total > limits.max_total_uncompressed_bytes:
                raise QuarantineError("archive total uncompressed byte limit exceeded")
            normalized.append((info, member_path))

        for info, member_path in normalized:
            is_bsp = PurePosixPath(member_path).suffix.lower() == ".bsp"
            selected = is_bsp and _matches(member_path, include_bsp)
            with archive.open(info, "r") as handle:
                digest, content = _hash_stream(
                    handle,
                    expected_bytes=info.file_size,
                    chunk_bytes=limits.chunk_bytes,
                    capture=selected,
                )
                # Force ZipExtFile to verify end-of-stream and its CRC even when
                # the declared uncompressed size ended exactly on our last read.
                if handle.read(1):
                    raise QuarantineError(
                        f"archive member exceeds declared size: {member_path}"
                    )
            members.append(ArchiveMember(
                path=member_path,
                byte_count=info.file_size,
                compressed_bytes=info.compress_size,
                sha256=digest,
                asset_class=_asset_class(member_path),
            ))
            if selected:
                if len(bsp_metadata) >= limits.max_bsp_members:
                    raise QuarantineError("selected BSP member limit exceeded")
                assert content is not None
                bsp_metadata[member_path] = parse_ibsp38(content, limits=bsp_limits)
    return members, bsp_metadata


def _scan_pak(
    path: Path,
    limits: ArchiveLimits,
    bsp_limits: BspLimits,
    include_bsp: Sequence[str],
) -> tuple[list[ArchiveMember], dict[str, BspMetadata]]:
    members: list[ArchiveMember] = []
    bsp_metadata: dict[str, BspMetadata] = {}
    seen: set[str] = set()
    archive_size = path.stat().st_size
    with path.open("rb") as archive:
        header = archive.read(12)
        if len(header) != 12 or header[:4] != b"PACK":
            raise QuarantineError("invalid Quake PAK header")
        directory_offset, directory_length = struct.unpack_from("<ii", header, 4)
        if directory_offset < 12 or directory_length < 0 or directory_length % 64:
            raise QuarantineError("invalid Quake PAK directory range")
        directory_end = directory_offset + directory_length
        if directory_end > archive_size:
            raise QuarantineError("Quake PAK directory exceeds archive size")
        member_count = directory_length // 64
        if member_count > limits.max_members:
            raise QuarantineError(
                f"archive has {member_count} members; limit is {limits.max_members}"
            )
        archive.seek(directory_offset)
        directory = archive.read(directory_length)
        if len(directory) != directory_length:
            raise QuarantineError("truncated Quake PAK directory")

        entries: list[tuple[str, int, int]] = []
        ranges: list[tuple[int, int, str]] = []
        total = 0
        for index in range(member_count):
            raw_name, offset, size = struct.unpack_from(
                "<56sii", directory, index * 64
            )
            raw_name = raw_name.split(b"\x00", 1)[0]
            try:
                name = raw_name.decode("ascii")
            except UnicodeDecodeError as error:
                raise QuarantineError("PAK member name is not ASCII") from error
            member_path = _normalize_member_path(
                name, directory=False, limit=limits.max_path_bytes
            )
            key = _case_key(member_path)
            if key in seen:
                raise QuarantineError(
                    f"duplicate or case-colliding member path: {member_path}"
                )
            seen.add(key)
            _validate_member_name(member_path)
            _validate_sizes(
                size=size, compressed=size, path=member_path, limits=limits
            )
            end = offset + size
            if offset < 12 or end > archive_size:
                raise QuarantineError(f"member {member_path} exceeds PAK bounds")
            if offset < directory_end and end > directory_offset:
                raise QuarantineError(f"member {member_path} overlaps PAK directory")
            total += size
            if total > limits.max_total_uncompressed_bytes:
                raise QuarantineError("archive total uncompressed byte limit exceeded")
            entries.append((member_path, offset, size))
            if size:
                ranges.append((offset, end, member_path))

        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise QuarantineError(
                    f"overlapping PAK members: {previous[2]} and {current[2]}"
                )

        for member_path, offset, size in entries:
            is_bsp = PurePosixPath(member_path).suffix.lower() == ".bsp"
            selected = is_bsp and _matches(member_path, include_bsp)
            archive.seek(offset)
            digest, content = _hash_stream(
                archive,
                expected_bytes=size,
                chunk_bytes=limits.chunk_bytes,
                capture=selected,
            )
            members.append(ArchiveMember(
                path=member_path,
                byte_count=size,
                compressed_bytes=size,
                sha256=digest,
                asset_class=_asset_class(member_path),
            ))
            if selected:
                if len(bsp_metadata) >= limits.max_bsp_members:
                    raise QuarantineError("selected BSP member limit exceeded")
                assert content is not None
                bsp_metadata[member_path] = parse_ibsp38(content, limits=bsp_limits)
    return members, bsp_metadata


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return not patterns or any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _asset_inventory(
    members: Sequence[ArchiveMember],
    bsp_metadata: Mapping[str, BspMetadata],
) -> AssetInventory:
    by_class = Counter(member.asset_class for member in members)
    provided = {_case_key(member.path) for member in members}
    unresolved = set()
    for metadata in bsp_metadata.values():
        for texture in metadata.faces.texture_names:
            expected = _case_key(f"textures/{texture}.wal")
            if expected not in provided:
                unresolved.add(texture)
    return AssetInventory(
        by_class=dict(sorted(by_class.items())),
        members=tuple(member.path for member in sorted(members, key=lambda item: item.path)),
        unresolved_texture_references=tuple(sorted(unresolved)),
    )


def quarantine_archive(
    source: Path | str,
    *,
    limits: ArchiveLimits = ArchiveLimits(),
    bsp_limits: BspLimits = BspLimits(),
    include_bsp: Sequence[str] = (),
) -> ArchiveReport:
    """Scan an archive without extraction and parse selected IBSP-38 members."""

    path = Path(source)
    archive_size = path.stat().st_size
    if archive_size > limits.max_archive_bytes:
        raise QuarantineError(
            f"archive is {archive_size} bytes; limit is {limits.max_archive_bytes}"
        )
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic == b"PACK":
        archive_format = "quake-pak"
        members, bsp_metadata = _scan_pak(
            path, limits, bsp_limits, include_bsp
        )
    elif magic[:2] == b"PK":
        archive_format = "zip"
        try:
            members, bsp_metadata = _scan_zip(
                path, limits, bsp_limits, include_bsp
            )
        except zipfile.BadZipFile as error:
            raise QuarantineError("invalid ZIP archive") from error
    else:
        raise QuarantineError(f"unsupported archive magic {magic!r}")
    total = sum(member.byte_count for member in members)
    return ArchiveReport(
        schema="q2-corpus-quarantine-v1",
        archive_format=archive_format,
        archive_sha256=_sha256_path(path, limits.chunk_bytes),
        archive_bytes=archive_size,
        total_uncompressed_bytes=total,
        members=tuple(sorted(members, key=lambda member: member.path)),
        bsp_metadata=dict(sorted(bsp_metadata.items())),
        assets=_asset_inventory(members, bsp_metadata),
    )


def inventory_stock_pak(
    source: Path | str,
    *,
    limits: ArchiveLimits = ArchiveLimits(),
    bsp_limits: BspLimits = BspLimits(),
) -> ArchiveReport:
    """Inventory only the locally installed q2dm1..q2dm8 PAK members.

    A retail ``pak0.pak`` contains trusted engine configuration scripts and is
    therefore intentionally *not* an admissible incoming map-pack archive
    under :func:`quarantine_archive`.  This narrow lane validates the complete
    PAK directory and ranges, but reads and admits only the eight named stock
    BSP members.  It cannot be used for community archives or arbitrary paths.
    """

    expected = tuple(f"maps/q2dm{index}.bsp" for index in range(1, 9))
    path = Path(source)
    archive_size = path.stat().st_size
    if archive_size > limits.max_archive_bytes:
        raise QuarantineError(
            f"archive is {archive_size} bytes; limit is {limits.max_archive_bytes}"
        )
    with path.open("rb") as archive:
        header = archive.read(12)
        if len(header) != 12 or header[:4] != b"PACK":
            raise QuarantineError("invalid Quake PAK header")
        directory_offset, directory_length = struct.unpack_from("<ii", header, 4)
        if directory_offset < 12 or directory_length < 0 or directory_length % 64:
            raise QuarantineError("invalid Quake PAK directory range")
        directory_end = directory_offset + directory_length
        if directory_end > archive_size:
            raise QuarantineError("Quake PAK directory exceeds archive size")
        member_count = directory_length // 64
        if member_count > limits.max_members:
            raise QuarantineError("Quake PAK member limit exceeded")
        archive.seek(directory_offset)
        directory = archive.read(directory_length)
        entries: dict[str, tuple[int, int]] = {}
        raw_seen: set[str] = set()
        selected_seen: set[str] = set()
        ranges: list[tuple[int, int, str]] = []
        total = 0
        for index in range(member_count):
            raw_name, offset, size = struct.unpack_from(
                "<56sii", directory, index * 64
            )
            try:
                name = raw_name.split(b"\x00", 1)[0].decode("ascii")
            except UnicodeDecodeError as error:
                raise QuarantineError("PAK member name is not ASCII") from error
            if name in raw_seen:
                raise QuarantineError(f"duplicate PAK directory name: {name!r}")
            raw_seen.add(name)
            member_path = name
            if name in expected:
                member_path = _normalize_member_path(
                    name, directory=False, limit=limits.max_path_bytes
                )
                key = _case_key(member_path)
                if key in selected_seen:
                    raise QuarantineError(
                        f"duplicate selected stock member: {member_path}"
                    )
                selected_seen.add(key)
            _validate_sizes(
                size=size, compressed=size, path=member_path, limits=limits
            )
            end = offset + size
            if offset < 12 or end > archive_size:
                raise QuarantineError(f"member {member_path} exceeds PAK bounds")
            if offset < directory_end and end > directory_offset:
                raise QuarantineError(f"member {member_path} overlaps PAK directory")
            entries[member_path] = (offset, size)
            total += size
            if total > limits.max_total_uncompressed_bytes:
                raise QuarantineError("archive total uncompressed byte limit exceeded")
            if size:
                ranges.append((offset, end, member_path))
        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise QuarantineError(
                    f"overlapping PAK members: {previous[2]} and {current[2]}"
                )

        missing = [member for member in expected if member not in entries]
        if missing:
            raise QuarantineError(
                "stock PAK is missing required maps: " + ", ".join(missing)
            )
        members = []
        metadata = {}
        for member_path in expected:
            offset, size = entries[member_path]
            archive.seek(offset)
            digest, content = _hash_stream(
                archive,
                expected_bytes=size,
                chunk_bytes=limits.chunk_bytes,
                capture=True,
            )
            assert content is not None
            parsed = parse_ibsp38(content, limits=bsp_limits)
            if parsed.sha256 != digest:
                raise QuarantineError(f"stock member hash mismatch: {member_path}")
            members.append(ArchiveMember(
                path=member_path,
                byte_count=size,
                compressed_bytes=size,
                sha256=digest,
                asset_class="bsp",
            ))
            metadata[member_path] = parsed

    return ArchiveReport(
        schema="q2-stock-container-inventory-v1",
        archive_format="quake-pak-stock-container",
        archive_sha256=_sha256_path(path, limits.chunk_bytes),
        archive_bytes=archive_size,
        total_uncompressed_bytes=sum(member.byte_count for member in members),
        members=tuple(members),
        bsp_metadata=metadata,
        assets=_asset_inventory(members, metadata),
    )


def admit_corpus(
    report: ArchiveReport,
    provenance: Iterable[ProvenanceRecord],
) -> tuple[CorpusEntry, ...]:
    """Bind selected BSPs to complete provenance and license records."""

    records = list(provenance)
    if not records:
        raise QuarantineError("at least one provenance record is required")
    by_member: dict[str, ProvenanceRecord] = {}
    names: set[str] = set()
    for record in records:
        record.validate()
        member = _normalize_member_path(
            record.bsp_member, directory=False, limit=512
        )
        if member in by_member:
            raise QuarantineError(f"duplicate provenance for {member}")
        for name in (record.canonical_id, *record.aliases):
            folded = _case_key(name)
            if folded in names:
                raise QuarantineError(f"duplicate canonical ID or alias {name!r}")
            names.add(folded)
        metadata = report.bsp_metadata.get(member)
        if metadata is None:
            raise QuarantineError(
                f"provenance references unselected or missing BSP {member}"
            )
        if record.archive_sha256 != report.archive_sha256:
            raise QuarantineError(
                f"{record.canonical_id}: archive SHA-256 does not match quarantine"
            )
        if record.bsp_sha256 != metadata.sha256:
            raise QuarantineError(
                f"{record.canonical_id}: BSP SHA-256 does not match quarantine"
            )
        by_member[member] = record
    missing = sorted(set(report.bsp_metadata) - set(by_member))
    if missing:
        raise QuarantineError(
            "selected BSPs lack provenance: " + ", ".join(missing)
        )
    return tuple(sorted((
        CorpusEntry(
            canonical_id=record.canonical_id,
            aliases=record.aliases,
            bsp_member=member,
            metadata=report.bsp_metadata[member],
            provenance=record,
        )
        for member, record in by_member.items()
    ), key=lambda entry: entry.canonical_id))


def _weighted_jaccard(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    numerator = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    denominator = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    return numerator / denominator if denominator else 1.0


def _ratio(left: int, right: int) -> float:
    if left == right == 0:
        return 1.0
    return min(left, right) / max(left, right)


def near_duplicate_score(left: BspMetadata, right: BspMetadata) -> float:
    """Coordinate-free similarity used only to flag manual near-copy review."""

    class_score = _weighted_jaccard(
        left.entity_catalog.class_counts, right.entity_catalog.class_counts
    )
    item_score = _weighted_jaccard(
        left.entity_catalog.item_classes, right.entity_catalog.item_classes
    )
    structural = sum((
        _ratio(len(left.models), len(right.models)),
        _ratio(left.faces.count, right.faces.count),
        _ratio(left.lightmaps.byte_count, right.lightmaps.byte_count),
        _ratio(left.visibility.cluster_count, right.visibility.cluster_count),
        _ratio(len(left.entities), len(right.entities)),
    )) / 5.0
    return 0.35 * class_score + 0.25 * item_score + 0.40 * structural


def classify_duplicates(
    entries: Sequence[CorpusEntry],
    *,
    near_threshold: float = 0.94,
) -> tuple[DuplicateClassification, ...]:
    if not 0.0 <= near_threshold <= 1.0:
        raise ValueError("near_threshold must be in [0, 1]")
    output = []
    ordered = sorted(entries, key=lambda entry: entry.canonical_id)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if left.metadata.sha256 == right.metadata.sha256:
                kind = "exact"
                score = 1.0
                canonical = min(left.canonical_id, right.canonical_id)
            else:
                score = near_duplicate_score(left.metadata, right.metadata)
                kind = "near" if score >= near_threshold else "distinct"
                canonical = None
            output.append(DuplicateClassification(
                left=left.canonical_id,
                right=right.canonical_id,
                kind=kind,
                score=round(score, 8),
                canonical_id=canonical,
            ))
    return tuple(output)


def duplicate_signature(metadata: BspMetadata) -> dict:
    """Return the non-coordinate descriptor used for duplicate review."""

    value = {
        "schema": "q2-corpus-duplicate-signature-v1",
        "entity_classes": dict(sorted(metadata.entity_catalog.class_counts.items())),
        "item_classes": dict(sorted(metadata.entity_catalog.item_classes.items())),
        "counts": {
            "entities": len(metadata.entities),
            "models": len(metadata.models),
            "faces": metadata.faces.count,
            "lightdata_bytes": metadata.lightmaps.byte_count,
            "visibility_clusters": metadata.visibility.cluster_count,
        },
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    value["signature_sha256"] = hashlib.sha256(encoded).hexdigest()
    return value


def load_provenance(path: Path | str) -> tuple[ProvenanceRecord, ...]:
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "q2-corpus-provenance-v1":
        raise QuarantineError("unsupported provenance schema")
    records = value.get("records")
    if not isinstance(records, list):
        raise QuarantineError("provenance records must be a list")
    return tuple(ProvenanceRecord.from_dict(record) for record in records)


def write_json(path: Path | str, value: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return target
