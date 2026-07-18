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


def _make_shallow_missing_parent_repo(path: Path, name: str) -> dict[str, str]:
    full = path.with_name(path.name + "-full")
    _make_repo(full, name)
    (full / "child.txt").write_text("child commit\n", encoding="utf-8")
    _git(full, "add", "child.txt")
    _git(full, "commit", "--quiet", "-m", "child")
    subprocess.run(
        (
            "git", "clone", "--quiet", "--depth=1", "--branch", BRANCH,
            f"file://{full.resolve()}", str(path),
        ),
        check=True,
    )
    assert _git(path, "rev-parse", "--is-shallow-repository") == "true"
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


@pytest.mark.parametrize(
    ("field", "value", "shape"),
    (
        ("shallow", True, "value"),
        ("shallow", 0, "value"),
        ("replace_refs", False, "value"),
        ("replace_refs", 0.0, "value"),
        ("unexpected", False, "extra"),
        ("alternates", None, "missing"),
    ),
)
def test_request_rejects_forged_transport_eligibility(
    staging_fixture: dict[str, object], field: str, value: object, shape: str
) -> None:
    repositories = staging.inspect_sources(staging_fixture["specs"])  # type: ignore[arg-type]
    request = staging.make_request(
        repositories=repositories,
        isolated_root=Path(staging_fixture["isolated"]),
        destination=Path(staging_fixture["destination"]),
        toolchain_root=Path(staging_fixture["toolchain"]),
        rustc_sha256=str(staging_fixture["rustc_sha256"]),
        cargo_sha256=str(staging_fixture["cargo_sha256"]),
        staged_at_utc="2026-07-17T12:00:00Z",
        invocation_nonce="1" * 32,
    )
    eligibility = dict(staging.ELIGIBLE_TRANSPORT_STATE)
    if shape == "missing":
        eligibility.pop(field)
    else:
        eligibility[field] = value
    request["repositories"][0]["transport_eligibility"] = eligibility
    with pytest.raises(
        staging.StagingError, match=r"transport eligibility .*not sealed"
    ):
        staging._validate_request(request)


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
    assert all(
        repository["transport_eligibility"]
        == staging.ELIGIBLE_TRANSPORT_STATE
        for repository in manifest["semantic"]["repositories"]
    )
    assert result["effects"] == {
        "public_runtime_changed": False,
        "services_changed": False,
        "trainer_changed_or_started": False,
    }


def test_shallow_bundle_source_verify_is_not_standalone_proof(tmp_path: Path) -> None:
    spec = _make_shallow_missing_parent_repo(
        tmp_path / "shallow-client", "q2-ml-client"
    )
    source = Path(spec["path"])
    raw_bundle = tmp_path / "source-verified.bundle"
    subprocess.run(
        (
            "git", "-C", str(source), "bundle", "create", str(raw_bundle),
            f"refs/heads/{BRANCH}",
        ),
        check=True,
    )
    source_verify = subprocess.run(
        ("git", "-C", str(source), "bundle", "verify", str(raw_bundle)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert source_verify.returncode == 0
    empty = tmp_path / "fresh-empty-repository"
    subprocess.run(("git", "init", "--quiet", str(empty)), check=True)
    empty_verify = subprocess.run(
        ("git", "-C", str(empty), "bundle", "verify", str(raw_bundle)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert empty_verify.returncode == 0
    fetch = subprocess.run(
        (
            "git", "-C", str(empty), "fetch", str(raw_bundle),
            f"refs/heads/{BRANCH}:refs/heads/{BRANCH}",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert fetch.returncode != 0
    assert b"did not send all necessary objects" in fetch.stderr
    with pytest.raises(staging.StagingError) as raised:
        staging.inspect_repository(
            name=spec["name"],
            path=source,
            expected_branch=spec["branch"],
            expected_commit=spec["commit"],
            expected_tree=spec["tree"],
        )
    diagnostic = str(raised.value)
    assert "q2-ml-client source is not transport-eligible" in diagnostic
    assert "shallow=true" in diagnostic
    assert "partial_clone=false" in diagnostic
    assert "replace_refs=0" in diagnostic
    assert "alternates=false" in diagnostic
    assert len(diagnostic) < 1000


def test_command_failure_detail_is_bounded_and_redacted() -> None:
    secrets = (
        "bearer-abcdefghijklmnopqrstuvwxyz0123456789",
        "basic-abcdefghijklmnopqrstuvwxyz0123456789",
        "github-abcdefghijklmnopqrstuvwxyz0123456789",
        "access-abcdefghijklmnopqrstuvwxyz0123456789",
        "api-abcdefghijklmnopqrstuvwxyz0123456789",
    )
    detail = staging._bounded_failure_detail(
        f"Authorization: Bearer {secrets[0]} "
        f"Authorization=Basic {secrets[1]} "
        f"GITHUB_TOKEN={secrets[2]} access_token:{secrets[3]} "
        f'\"api_key\": \"{secrets[4]}\" '
        "https://user:password@example.invalid/ "
        + "x" * 2000
    )
    assert all(secret not in detail for secret in secrets)
    assert "user:password" not in detail
    assert detail.count("Authorization=<redacted>") == 2
    assert "GITHUB_TOKEN=<redacted>" in detail
    assert "access_token=<redacted>" in detail
    assert "api_key=<redacted>" in detail
    assert "https://<redacted>@example.invalid/" in detail
    assert len(detail) == staging.MAX_FAILURE_DETAIL_CHARS


@pytest.mark.parametrize("diagnostic_channel", ("stderr", "stdout"))
def test_openssh_invalid_json_diagnostics_are_bounded_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostic_channel: str
) -> None:
    module = tmp_path / "remote.py"
    module.write_text("pass\n", encoding="utf-8")
    secrets = (
        "bearer-ssh-secret-0123456789",
        "basic-ssh-secret-0123456789",
        "github-ssh-secret-0123456789",
        "access-ssh-secret-0123456789",
        "api-ssh-secret-0123456789",
    )
    adversarial = (
        f"Authorization: Bearer {secrets[0]} "
        f"Authorization: Basic {secrets[1]} "
        f"GITHUB_TOKEN={secrets[2]} access_token={secrets[3]} "
        f"api_key={secrets[4]} https://user:url-secret@example.invalid/ "
        + "z" * 2000
    ).encode()
    stdout = b"not-json " + (adversarial if diagnostic_channel == "stdout" else b"")
    stderr = adversarial if diagnostic_channel == "stderr" else b""
    monkeypatch.setattr(
        staging.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=1, stdout=stdout, stderr=stderr
        ),
    )
    transport = staging.OpenSshTransport("wsl-box", module_path=module)
    with pytest.raises(staging.StagingError) as raised:
        transport.execute("preflight", {}, None)
    diagnostic = str(raised.value)
    assert all(secret not in diagnostic for secret in secrets)
    assert "url-secret" not in diagnostic
    assert "<redacted>" in diagnostic
    assert len(diagnostic) < staging.MAX_FAILURE_DETAIL_CHARS + 100


def test_openssh_remote_json_error_is_bounded_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "remote.py"
    module.write_text("pass\n", encoding="utf-8")
    secrets = (
        "remote-bearer-secret-0123456789",
        "remote-github-secret-0123456789",
        "remote-api-secret-0123456789",
    )
    error = (
        f"Authorization: Bearer {secrets[0]} "
        f"GITHUB_TOKEN={secrets[1]} api_key={secrets[2]} "
        + "q" * 2000
    )
    stdout = json.dumps({"ok": False, "error": error}).encode()
    monkeypatch.setattr(
        staging.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=1, stdout=stdout, stderr=b""
        ),
    )
    transport = staging.OpenSshTransport("wsl-box", module_path=module)
    with pytest.raises(staging.StagingError) as raised:
        transport.execute("preflight", {}, None)
    diagnostic = str(raised.value)
    assert all(secret not in diagnostic for secret in secrets)
    assert "<redacted>" in diagnostic
    assert len(diagnostic) < staging.MAX_FAILURE_DETAIL_CHARS + 100


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


def test_preflight_bundle_proof_failure_never_reaches_transport(
    staging_fixture: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_bundle(repo: object, bundle: Path) -> None:
        del repo, bundle
        raise staging.StagingError("synthetic standalone bundle rejection")

    monkeypatch.setattr(staging, "_prove_bundle_self_contained", reject_bundle)
    destination = Path(staging_fixture["destination"])
    with pytest.raises(staging.StagingError, match="standalone bundle rejection"):
        _execute(
            staging_fixture,
            "preflight",
            transport=_NeverCalledTransport(),
        )
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.partial-*"))


def test_incomplete_source_bundle_fails_before_transport_and_leaves_no_stage(
    staging_fixture: dict[str, object], tmp_path: Path
) -> None:
    specs = [dict(item) for item in staging_fixture["specs"]]  # type: ignore[arg-type]
    specs[1] = _make_shallow_missing_parent_repo(
        tmp_path / "transport-shallow-client", "q2-ml-client"
    )
    destination = Path(staging_fixture["destination"])
    with pytest.raises(staging.StagingError, match="source is not transport-eligible"):
        _execute(
            staging_fixture,
            "stage",
            repository_specs=specs,
            transport=_NeverCalledTransport(),
        )
    assert not destination.exists()
    assert not list(destination.parent.glob(f".{destination.name}.partial-*"))


@pytest.mark.parametrize(
    "mutation", ("partial_clone", "promisor_pack", "alternates", "replace")
)
def test_nonstandalone_source_transport_state_is_rejected(
    tmp_path: Path, mutation: str
) -> None:
    spec = _make_repo(tmp_path / "source", "q2-ml-client")
    source = Path(spec["path"])
    if mutation == "partial_clone":
        _git(source, "config", "remote.origin.promisor", "true")
    elif mutation == "promisor_pack":
        common_git_value = _git(source, "rev-parse", "--git-common-dir")
        common_git = Path(common_git_value)
        if not common_git.is_absolute():
            common_git = source / common_git
        promisor = common_git / "objects" / "pack" / "fixture.promisor"
        promisor.parent.mkdir(parents=True, exist_ok=True)
        promisor.write_bytes(b"")
    elif mutation == "alternates":
        alternate_objects = tmp_path / "alternate-objects"
        alternate_objects.mkdir()
        alternates_value = _git(
            source, "rev-parse", "--git-path", "objects/info/alternates"
        )
        alternates = Path(alternates_value)
        if not alternates.is_absolute():
            alternates = source / alternates
        alternates.parent.mkdir(parents=True, exist_ok=True)
        alternates.write_text(str(alternate_objects) + "\n", encoding="utf-8")
    else:
        _git(
            source, "update-ref", f"refs/replace/{spec['commit']}",
            spec["commit"],
        )
    with pytest.raises(staging.StagingError, match="source is not transport-eligible"):
        staging.inspect_repository(
            name=spec["name"],
            path=source,
            expected_branch=spec["branch"],
            expected_commit=spec["commit"],
            expected_tree=spec["tree"],
        )


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
