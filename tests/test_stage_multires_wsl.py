from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile

import pytest

from tools import stage_multires_wsl as staging


BRANCH = "feature/multires-map-atlas-v1"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _make_repo(path: Path, name: str, *, symlink: bool = False) -> dict[str, str]:
    path.mkdir()
    subprocess.run(("git", "init", "--quiet", "-b", BRANCH, str(path)), check=True)
    _git(path, "config", "user.name", "staging test")
    _git(path, "config", "user.email", "staging@example.invalid")
    (path / "identity.txt").write_text(name + "\n", encoding="utf-8")
    if name == "q2-ml-bot":
        docs = path / "docs"
        docs.mkdir()
        (docs / "MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md").write_text(
            "authoritative design\n", encoding="utf-8"
        )
        (docs / "MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md").write_text(
            "authoritative plan\n", encoding="utf-8"
        )
    if symlink:
        os.symlink("identity.txt", path / "forbidden-link")
    _git(path, "add", ".")
    _git(path, "commit", "--quiet", "-m", "fixture")
    return {
        "name": name,
        "path": str(path),
        "branch": BRANCH,
        "commit": _git(path, "rev-parse", "HEAD"),
        "tree": _git(path, "rev-parse", "HEAD^{tree}"),
    }


@pytest.fixture
def staging_fixture(tmp_path: Path) -> dict[str, object]:
    sources = tmp_path / "sources"
    sources.mkdir()
    specs = [
        _make_repo(sources / "bot", "q2-ml-bot"),
        _make_repo(sources / "client", "q2-ml-client"),
        _make_repo(sources / "game", "q2-lithium-3zb2"),
    ]
    isolated = tmp_path / "remote" / "q2-multires-isolated"
    parent = isolated / "B6"
    parent.mkdir(parents=True)
    toolchain = isolated / "tooling" / "rust-1.96.1-x86_64-unknown-linux-gnu"
    binary_dir = toolchain / "bin"
    binary_dir.mkdir(parents=True)
    rustc = binary_dir / "rustc"
    cargo = binary_dir / "cargo"
    rustc.write_bytes(b"pinned-rustc\n")
    cargo.write_bytes(b"pinned-cargo\n")
    rustc.chmod(0o755)
    cargo.chmod(0o755)
    return {
        "specs": specs,
        "isolated": isolated,
        "destination": parent / "prototype-a",
        "toolchain": toolchain,
        "rustc_sha256": hashlib.sha256(rustc.read_bytes()).hexdigest(),
        "cargo_sha256": hashlib.sha256(cargo.read_bytes()).hexdigest(),
        "transport": staging.LocalRemoteTransport(staging.EXPECTED_HOST_IDENTITY),
    }


def _execute(fixture: dict[str, object], mode: str, **overrides: object) -> dict[str, object]:
    arguments = {
        "mode": mode,
        "repository_specs": fixture["specs"],
        "isolated_root": fixture["isolated"],
        "destination": fixture["destination"],
        "toolchain_root": fixture["toolchain"],
        "rustc_sha256": fixture["rustc_sha256"],
        "cargo_sha256": fixture["cargo_sha256"],
        "transport": fixture["transport"],
        "staged_at_utc": "2026-07-17T12:00:00Z",
        "invocation_nonce": "1" * 32,
    }
    arguments.update(overrides)
    return staging.execute(**arguments)  # type: ignore[arg-type]


def test_preflight_is_read_only_and_semantically_deterministic(
    staging_fixture: dict[str, object],
) -> None:
    destination = staging_fixture["destination"]
    first = _execute(staging_fixture, "preflight")
    second = _execute(
        staging_fixture,
        "preflight",
        staged_at_utc="2030-01-01T00:00:00Z",
        invocation_nonce="2" * 32,
    )
    assert first["passed"] is True
    assert first["read_only"] is True
    assert first["semantic"] == second["semantic"]
    assert first["semantic_sha256"] == second["semantic_sha256"]
    assert not Path(destination).exists()


def test_stage_reconstructs_exact_clean_triple_and_writes_canonical_0600_manifest(
    staging_fixture: dict[str, object],
) -> None:
    preflight = _execute(staging_fixture, "preflight")
    result = _execute(staging_fixture, "stage")
    destination = Path(staging_fixture["destination"])
    manifest_path = destination / "staging-manifest.json"
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    assert result["published"] is True
    assert raw == staging.canonical_bytes(manifest)
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert manifest["semantic"] == preflight["semantic"]
    assert manifest["semantic_sha256"] == preflight["semantic_sha256"]
    assert manifest["informational"] == {"staged_at_utc": "2026-07-17T12:00:00Z"}
    assert not (destination / ".incoming").exists()
    for source in staging_fixture["specs"]:  # type: ignore[assignment]
        repo = destination / "repositories" / source["name"]
        assert _git(repo, "symbolic-ref", "--short", "HEAD") == source["branch"]
        assert _git(repo, "rev-parse", "HEAD") == source["commit"]
        assert _git(repo, "rev-parse", "HEAD^{tree}") == source["tree"]
        assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    assert result["effects"] == {
        "public_runtime_changed": False,
        "services_changed": False,
        "trainer_changed_or_started": False,
    }


def test_local_transport_bundle_checkout_is_independent_of_cli_cwd(
    staging_fixture: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside-any-repository"
    outside.mkdir()
    probe = subprocess.run(
        ("git", "-C", str(outside), "rev-parse", "--git-dir"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert probe.returncode != 0
    monkeypatch.chdir(outside)
    result = _execute(staging_fixture, "stage")
    destination = Path(staging_fixture["destination"])
    assert result["published"] is True
    assert destination.is_dir()
    for source in staging_fixture["specs"]:  # type: ignore[assignment]
        repository = destination / "repositories" / source["name"]
        assert _git(repository, "rev-parse", "HEAD") == source["commit"]


class _BundleVerifyFailingTransport:
    def __init__(self, hostname: str, temp: Path):
        self.delegate = staging.LocalRemoteTransport(hostname)
        self.temp = temp

    def execute(
        self, mode: str, request: object, archive_path: Path | None
    ) -> dict[str, object]:
        if mode != "stage" or archive_path is None:
            return self.delegate.execute(mode, request, archive_path)  # type: ignore[arg-type]
        altered_path = self.temp / "bundle-verify-failure.tar"
        with tarfile.open(archive_path, "r") as source:
            members = {}
            for member in source.getmembers():
                extracted = source.extractfile(member)
                assert extracted is not None
                members[member.name] = extracted.read()
        bundle_name = "bundles/q2-ml-client.bundle"
        members[bundle_name] = b"not a valid git bundle\n"
        transfer = json.loads(members["transfer.json"])
        record = next(
            item for item in transfer["repositories"]
            if item["name"] == "q2-ml-client"
        )
        record["bundle_sha256"] = hashlib.sha256(members[bundle_name]).hexdigest()
        record["bundle_size_bytes"] = len(members[bundle_name])
        members["transfer.json"] = staging.canonical_bytes(transfer)
        with tarfile.open(altered_path, "w") as output:
            for name in sorted(members):
                info = tarfile.TarInfo(name)
                info.size = len(members[name])
                info.mode = 0o600
                output.addfile(
                    info, fileobj=__import__("io").BytesIO(members[name])
                )
        return self.delegate.execute(mode, request, altered_path)  # type: ignore[arg-type]


def test_remote_bundle_verify_failure_cleans_temporary_checkout_from_outside_cwd(
    staging_fixture: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside-any-repository-failure"
    outside.mkdir()
    monkeypatch.chdir(outside)
    destination = Path(staging_fixture["destination"])
    transport = _BundleVerifyFailingTransport(
        staging.EXPECTED_HOST_IDENTITY, tmp_path
    )
    with pytest.raises(staging.StagingError, match=r"command failed \(git\)"):
        _execute(staging_fixture, "stage", transport=transport)
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.partial-*"))


class _NeverCalledTransport:
    def execute(self, mode: str, request: object, archive_path: Path | None) -> dict[str, object]:
        raise AssertionError("transport must not run after a source-side failure")


def test_dirty_source_fails_before_transport(staging_fixture: dict[str, object]) -> None:
    bot = Path(staging_fixture["specs"][0]["path"])  # type: ignore[index]
    (bot / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    with pytest.raises(staging.StagingError, match="repository is dirty"):
        _execute(staging_fixture, "stage", transport=_NeverCalledTransport())
    assert not Path(staging_fixture["destination"]).exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("branch", "feature/wrong", "branch mismatch"),
        ("commit", "0" * 40, "commit mismatch"),
        ("tree", "0" * 40, "tree mismatch"),
    ],
)
def test_source_identity_mismatch_fails_before_transport(
    staging_fixture: dict[str, object], field: str, value: str, message: str
) -> None:
    specs = [dict(item) for item in staging_fixture["specs"]]  # type: ignore[arg-type]
    specs[1][field] = value
    with pytest.raises(staging.StagingError, match=message):
        _execute(
            staging_fixture,
            "preflight",
            repository_specs=specs,
            transport=_NeverCalledTransport(),
        )


def test_existing_destination_is_never_overlaid_or_deleted(
    staging_fixture: dict[str, object],
) -> None:
    destination = Path(staging_fixture["destination"])
    destination.mkdir()
    marker = destination / "old-stage-marker"
    marker.write_text("preserve me\n", encoding="utf-8")
    with pytest.raises(staging.StagingError, match="destination already exists"):
        _execute(staging_fixture, "stage")
    assert marker.read_text(encoding="utf-8") == "preserve me\n"
    assert not list(destination.parent.glob(f".{destination.name}.partial-*"))


def test_host_mismatch_fails_without_creating_or_deleting_any_stage(
    staging_fixture: dict[str, object],
) -> None:
    destination = Path(staging_fixture["destination"])
    old = destination.parent / "older-stage"
    old.mkdir()
    marker = old / "marker"
    marker.write_text("old\n", encoding="utf-8")
    transport = staging.LocalRemoteTransport("WRONG-HOST")
    with pytest.raises(staging.StagingError, match="host identity mismatch"):
        _execute(staging_fixture, "stage", transport=transport)
    assert not destination.exists()
    assert marker.read_text(encoding="utf-8") == "old\n"


def test_symlinked_remote_root_is_rejected(staging_fixture: dict[str, object]) -> None:
    real_root = Path(staging_fixture["isolated"])
    linked_root = real_root.parent / "linked-isolated"
    linked_root.symlink_to(real_root, target_is_directory=True)
    destination = linked_root / "B6" / "new-stage"
    with pytest.raises(staging.StagingError, match="symlink component"):
        _execute(
            staging_fixture,
            "preflight",
            isolated_root=linked_root,
            destination=destination,
        )
    assert not destination.exists()


class _ArchiveMutatingTransport:
    def __init__(self, hostname: str, temp: Path, mutation: str):
        self.delegate = staging.LocalRemoteTransport(hostname)
        self.temp = temp
        self.mutation = mutation

    def execute(
        self, mode: str, request: object, archive_path: Path | None
    ) -> dict[str, object]:
        if mode != "stage" or archive_path is None:
            return self.delegate.execute(mode, request, archive_path)  # type: ignore[arg-type]
        corrupted = self.temp / f"{self.mutation}.tar"
        if self.mutation == "partial":
            data = archive_path.read_bytes()
            corrupted.write_bytes(data[: max(1, len(data) // 3)])
        else:
            with tarfile.open(archive_path, "r") as source:
                members = {}
                for member in source.getmembers():
                    extracted = source.extractfile(member)
                    assert extracted is not None
                    members[member.name] = extracted.read()
            target_name = "bundles/q2-ml-client.bundle"
            altered = bytearray(members[target_name])
            altered[len(altered) // 2] ^= 0x01
            members[target_name] = bytes(altered)
            with tarfile.open(corrupted, "w") as output:
                for name in sorted(members):
                    info = tarfile.TarInfo(name)
                    info.size = len(members[name])
                    info.mode = 0o600
                    output.addfile(info, fileobj=__import__("io").BytesIO(members[name]))
        return self.delegate.execute(mode, request, corrupted)  # type: ignore[arg-type]


@pytest.mark.parametrize("mutation", ["partial", "hash"])
def test_partial_or_hash_corrupt_transfer_cleans_only_its_new_temporary_stage(
    staging_fixture: dict[str, object], tmp_path: Path, mutation: str
) -> None:
    destination = Path(staging_fixture["destination"])
    old = destination.parent / "older-stage"
    old.mkdir()
    marker = old / "marker"
    marker.write_text("preserve\n", encoding="utf-8")
    transport = _ArchiveMutatingTransport(staging.EXPECTED_HOST_IDENTITY, tmp_path, mutation)
    with pytest.raises((staging.StagingError, tarfile.TarError)):
        _execute(staging_fixture, "stage", transport=transport)
    assert not destination.exists()
    assert marker.read_text(encoding="utf-8") == "preserve\n"
    assert not list(destination.parent.glob(f".{destination.name}.partial-*"))


def test_toolchain_hash_mismatch_fails_closed(staging_fixture: dict[str, object]) -> None:
    with pytest.raises(staging.StagingError, match="toolchain rustc hash mismatch"):
        _execute(staging_fixture, "stage", rustc_sha256="0" * 64)
    assert not Path(staging_fixture["destination"]).exists()


def test_tracked_symlink_source_is_rejected(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "linked", "q2-ml-client", symlink=True)
    with pytest.raises(staging.StagingError, match="tracked symlink is forbidden"):
        staging.inspect_repository(
            name=repo["name"],
            path=Path(repo["path"]),
            expected_branch=repo["branch"],
            expected_commit=repo["commit"],
            expected_tree=repo["tree"],
        )
