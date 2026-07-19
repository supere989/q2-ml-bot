"""Host-bound one-shot authorization primitive tests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tools import final_execution_binding as binding_tool


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


@pytest.fixture
def execution_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """A synthetic DESKTOP-RTX2080 context without touching host identity."""

    machine_id = tmp_path / "machine-id"
    identity = b"0123456789abcdef0123456789abcdef\n"
    machine_id.write_bytes(identity)
    repo = tmp_path / "repo"
    repo.mkdir()
    declaration = repo / "B2-GENERATED-COHORT-71454-DECLARATION.json"
    declaration.write_bytes(b"{}\n")
    workspace = tmp_path / "final-workspace"
    state_root = tmp_path / "final-cohort-authorizations"
    state_root.mkdir()
    state_root.chmod(0o700)
    monkeypatch.setattr(binding_tool, "MACHINE_ID_PATH", machine_id)
    monkeypatch.setattr(
        binding_tool.socket, "gethostname", lambda: "DESKTOP-RTX2080.example"
    )
    monkeypatch.setattr(
        binding_tool.platform,
        "release",
        lambda: "6.6.87.2-microsoft-standard-WSL2",
    )
    return {
        "machine_id": machine_id,
        "machine_identity": identity[:-1],
        "repo": repo,
        "declaration": declaration,
        "workspace": workspace,
        "state_root": state_root,
    }


def _binding(context: dict[str, object]) -> dict[str, object]:
    return binding_tool.build_execution_binding(
        state_root=context["state_root"],  # type: ignore[arg-type]
        repo_root=context["repo"],  # type: ignore[arg-type]
        workspace=context["workspace"],  # type: ignore[arg-type]
    )


def _validate(context: dict[str, object], value: dict[str, object]) -> dict[str, object]:
    return binding_tool.validate_execution_binding(
        value,
        repo_root=context["repo"],  # type: ignore[arg-type]
        workspace=context["workspace"],  # type: ignore[arg-type]
    )


def _source_outputs(context: dict[str, object]) -> dict[str, Path]:
    workspace = context["workspace"]
    assert isinstance(workspace, Path)
    return {
        "primary": workspace / "source",
        "cold": workspace / "source-cold",
        "report": workspace / "reports/source-freeze.json",
    }


def _implementation() -> dict[str, object]:
    return {
        "repository_commit": "a" * 40,
        "repository_tree": "b" * 40,
        "git_clean": True,
        "atlas_analyzer_authority_sha256": "c" * 64,
        "atlas_analyzer_authority_file_count": 3,
        "generator_sha256": "d" * 64,
        "routes_sha256": "e" * 64,
    }


def _marker_payload(
    context: dict[str, object],
    value: dict[str, object],
    *,
    cohort: str,
    declaration_sha256: str,
    plan_sha256: str,
) -> bytes:
    outputs = _source_outputs(context)
    return binding_tool.build_source_authorization_marker(
        binding=value,
        cohort_id=cohort,
        declaration_sha256=declaration_sha256,
        declaration_path=context["declaration"],  # type: ignore[arg-type]
        plan_sha256=plan_sha256,
        workspace=context["workspace"],  # type: ignore[arg-type]
        repo_root=context["repo"],  # type: ignore[arg-type]
        implementation=_implementation(),
        source_output=outputs["primary"],
        source_cold=outputs["cold"],
        source_report=outputs["report"],
    )


def test_build_and_validate_binds_fixed_wsl_host_and_private_root(
    execution_context: dict[str, object],
) -> None:
    value = _binding(execution_context)

    assert value["schema"] == binding_tool.EXECUTION_BINDING_SCHEMA
    host = value["host"]  # type: ignore[index]
    state = value["state_root"]  # type: ignore[index]
    assert host["hostname"] == "DESKTOP-RTX2080"
    assert host["kernel_release"] == "6.6.87.2-microsoft-standard-WSL2"
    assert host["machine_identity"] == {
        "path": str(execution_context["machine_id"]),
        "sha256": hashlib.sha256(execution_context["machine_identity"]).hexdigest(),
    }
    assert host["euid"] == os.geteuid()
    assert state["path"] == str(execution_context["state_root"])
    assert state["owner_uid"] == os.geteuid()
    assert state["mode"] == "0700"
    assert isinstance(state["device"], int)
    assert isinstance(state["inode"], int)
    assert _validate(execution_context, value) == value


def test_other_host_or_machine_identity_cannot_replay_binding(
    execution_context: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _binding(execution_context)

    monkeypatch.setattr(binding_tool.socket, "gethostname", lambda: "OTHER-LAN-HOST")
    with pytest.raises(
        binding_tool.FinalExecutionBindingError,
        match="only admitted final execution host",
    ):
        _validate(execution_context, value)

    monkeypatch.setattr(
        binding_tool.socket, "gethostname", lambda: "DESKTOP-RTX2080"
    )
    machine_id = execution_context["machine_id"]
    assert isinstance(machine_id, Path)
    machine_id.write_bytes(b"abcdef0123456789abcdef0123456789\n")
    with pytest.raises(
        binding_tool.FinalExecutionBindingError,
        match="machine identity differs",
    ):
        _validate(execution_context, value)


def test_non_wsl_kernel_or_changed_release_cannot_replay_binding(
    execution_context: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _binding(execution_context)
    monkeypatch.setattr(binding_tool.platform, "release", lambda: "6.8.0-generic")
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="must be DESKTOP-RTX2080 WSL2"
    ):
        _validate(execution_context, value)

    monkeypatch.setattr(
        binding_tool.platform,
        "release",
        lambda: "6.6.88.1-microsoft-standard-WSL2",
    )
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="kernel release differs"
    ):
        _validate(execution_context, value)


def test_other_effective_uid_cannot_replay_binding(
    execution_context: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _binding(execution_context)
    monkeypatch.setattr(binding_tool.os, "geteuid", lambda: os.getuid() + 1)

    with pytest.raises(
        binding_tool.FinalExecutionBindingError,
        match="effective UID differs",
    ):
        _validate(execution_context, value)


def test_state_root_requires_exact_mode_owner_and_stable_inode(
    execution_context: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _binding(execution_context)
    real_euid = os.geteuid
    root = execution_context["state_root"]
    assert isinstance(root, Path)

    root.chmod(0o750)
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="must have mode 0700"
    ):
        _validate(execution_context, value)
    root.chmod(0o700)

    # An account different from the directory owner may not build a binding.
    monkeypatch.setattr(binding_tool.os, "geteuid", lambda: os.getuid() + 1)
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="owner differs"
    ):
        binding_tool.build_execution_binding(
            state_root=root,
            repo_root=execution_context["repo"],
            workspace=execution_context["workspace"],
        )
    monkeypatch.setattr(binding_tool.os, "geteuid", real_euid)

    moved = root.with_name("former-final-cohort-authorizations")
    root.rename(moved)
    root.mkdir()
    root.chmod(0o700)
    with pytest.raises(
        binding_tool.FinalExecutionBindingError,
        match="state root identity differs",
    ):
        _validate(execution_context, value)


def test_state_root_rejects_symlink_and_repository_or_workspace_overlap(
    execution_context: dict[str, object],
) -> None:
    root = execution_context["state_root"]
    assert isinstance(root, Path)
    link = root.with_name("authorization-link")
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="traverses a symlink"
    ):
        binding_tool.build_execution_binding(
            state_root=link,
            repo_root=execution_context["repo"],
            workspace=execution_context["workspace"],
        )

    repo = execution_context["repo"]
    assert isinstance(repo, Path)
    repo_state = repo / "journal"
    repo_state.mkdir()
    repo_state.chmod(0o700)
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="outside the repository"
    ):
        binding_tool.build_execution_binding(
            state_root=repo_state,
            repo_root=repo,
            workspace=execution_context["workspace"],
        )

    workspace = execution_context["workspace"]
    assert isinstance(workspace, Path)
    workspace.mkdir()
    workspace_state = workspace / "journal"
    workspace_state.mkdir()
    workspace_state.chmod(0o700)
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="outside the final execution workspace"
    ):
        binding_tool.build_execution_binding(
            state_root=workspace_state,
            repo_root=repo,
            workspace=workspace,
        )


def test_marker_resolution_is_declaration_scoped_and_rejects_symlink_leaf(
    execution_context: dict[str, object],
) -> None:
    value = _binding(execution_context)
    declaration = "a" * 64
    marker = binding_tool.resolve_secure_marker_path(
        value,
        declaration,
        repo_root=execution_context["repo"],
        workspace=execution_context["workspace"],
    )
    assert marker == execution_context["state_root"] / f"{declaration}.json"

    marker.symlink_to(execution_context["machine_id"])
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="not a regular file"
    ):
        binding_tool.resolve_secure_marker_path(
            value,
            declaration,
            repo_root=execution_context["repo"],
            workspace=execution_context["workspace"],
        )


def test_v4_marker_payload_is_canonical_and_blocks_different_workspace(
    execution_context: dict[str, object],
) -> None:
    value = _binding(execution_context)
    cohort = "b2g26_final_71454"
    declaration = "c" * 64
    payload = _marker_payload(
        execution_context, value, cohort=cohort,
        declaration_sha256=declaration, plan_sha256="d" * 64,
    )
    loaded = json.loads(payload)
    assert payload == _canonical(loaded)
    assert loaded["schema"] == binding_tool.SOURCE_AUTHORIZATION_MARKER_SCHEMA
    assert loaded["execution_binding"] == value
    assert loaded["declaration_path"] == str(execution_context["declaration"])
    assert loaded["source_outputs"] == {
        name: str(path) for name, path in _source_outputs(execution_context).items()
    }
    assert loaded["implementation"] == _implementation()

    # A tombstone is declaration-scoped, not workspace-scoped.  A second plan
    # cannot reopen the declaration merely by choosing another workspace.
    another_workspace = Path(execution_context["workspace"]).with_name("other")
    verified = binding_tool.verify_source_authorization_marker(
        payload,
        binding=value,
        cohort_id=cohort,
        declaration_sha256=declaration,
        repo_root=execution_context["repo"],
        workspace=another_workspace,
    )
    assert verified["plan_sha256"] == "d" * 64
    assert verified["workspace"] == str(execution_context["workspace"])


def test_v4_marker_rejects_noncanonical_or_wrong_execution_binding(
    execution_context: dict[str, object],
) -> None:
    value = _binding(execution_context)
    cohort = "b2g26_final_71454"
    declaration = "e" * 64
    payload = _marker_payload(
        execution_context, value, cohort=cohort,
        declaration_sha256=declaration, plan_sha256="f" * 64,
    )

    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="not canonical JSON"
    ):
        binding_tool.verify_source_authorization_marker(
            payload.rstrip(b"\n"),
            binding=value,
            cohort_id=cohort,
            declaration_sha256=declaration,
            repo_root=execution_context["repo"],
            workspace=execution_context["workspace"],
        )


def test_final_source_capability_binds_marker_declaration_and_outputs(
    execution_context: dict[str, object],
) -> None:
    value = _binding(execution_context)
    cohort = "b2g26_final_71454"
    declaration_sha256 = "f" * 64
    payload = _marker_payload(
        execution_context,
        value,
        cohort=cohort,
        declaration_sha256=declaration_sha256,
        plan_sha256="e" * 64,
    )
    state_root = execution_context["state_root"]
    assert isinstance(state_root, Path)
    marker = state_root / f"{declaration_sha256}.json"
    marker.write_bytes(payload)
    marker.chmod(0o600)
    outputs = _source_outputs(execution_context)

    verified = binding_tool.validate_final_source_authorization(
        marker,
        declaration_path=execution_context["declaration"],
        declaration_sha256=declaration_sha256,
        cohort_id=cohort,
        source_output=outputs["primary"],
        source_cold=outputs["cold"],
        source_report=outputs["report"],
        repo_root=execution_context["repo"],
        implementation=_implementation(),
    )
    assert verified["source_outputs"]["primary"] == str(outputs["primary"])

    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="output paths differ"
    ):
        binding_tool.validate_final_source_authorization(
            marker,
            declaration_path=execution_context["declaration"],
            declaration_sha256=declaration_sha256,
            cohort_id=cohort,
            source_output=outputs["cold"],
            source_cold=outputs["primary"],
            source_report=outputs["report"],
            repo_root=execution_context["repo"],
            implementation=_implementation(),
        )

    different_implementation = _implementation()
    different_implementation["generator_sha256"] = "0" * 64
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="implementation binding differs"
    ):
        binding_tool.validate_final_source_authorization(
            marker,
            declaration_path=execution_context["declaration"],
            declaration_sha256=declaration_sha256,
            cohort_id=cohort,
            source_output=outputs["primary"],
            source_cold=outputs["cold"],
            source_report=outputs["report"],
            repo_root=execution_context["repo"],
            implementation=different_implementation,
        )

    copied = Path(execution_context["repo"]) / "copied-declaration.json"
    copied.write_bytes(Path(execution_context["declaration"]).read_bytes())
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="declaration path differs"
    ):
        binding_tool.validate_final_source_authorization(
            marker,
            declaration_path=copied,
            declaration_sha256=declaration_sha256,
            cohort_id=cohort,
            source_output=outputs["primary"],
            source_cold=outputs["cold"],
            source_report=outputs["report"],
            repo_root=execution_context["repo"],
            implementation=_implementation(),
        )

    mutated = json.loads(payload)
    mutated["execution_binding"]["host"]["euid"] += 1
    with pytest.raises(
        binding_tool.FinalExecutionBindingError,
        match="marker execution binding differs",
    ):
        binding_tool.verify_source_authorization_marker(
                _canonical(mutated),
                binding=value,
                cohort_id=cohort,
                declaration_sha256=declaration_sha256,
            repo_root=execution_context["repo"],
            workspace=execution_context["workspace"],
        )


def test_secure_journal_creates_and_reads_only_through_bound_root(
    execution_context: dict[str, object],
) -> None:
    value = _binding(execution_context)
    declaration = "1" * 64
    payload = _marker_payload(
        execution_context,
        value,
        cohort="b2g26_final_71454",
        declaration_sha256=declaration,
        plan_sha256="2" * 64,
    )

    with binding_tool.open_secure_marker_journal(
        value,
        repo_root=execution_context["repo"],
        workspace=execution_context["workspace"],
    ) as journal:
        assert journal.exists(declaration) is False
        stored, created = journal.create(declaration, payload)
        assert created is True
        assert stored == payload
        assert journal.read(declaration) == payload
        marker = journal.marker_path(declaration)

    assert marker.read_bytes() == payload
    assert marker.stat().st_mode & 0o777 == 0o600


def test_secure_journal_rejects_root_replacement_before_or_after_open(
    execution_context: dict[str, object],
) -> None:
    value = _binding(execution_context)
    root = execution_context["state_root"]
    assert isinstance(root, Path)
    former = root.with_name("former-journal")
    root.rename(former)
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    with pytest.raises(
        binding_tool.FinalExecutionBindingError, match="state root identity differs"
    ):
        with binding_tool.open_secure_marker_journal(
            value,
            repo_root=execution_context["repo"],
            workspace=execution_context["workspace"],
        ):
            pass

    # Restore the original bound root then replace it only after its descriptor
    # is open; revalidation must still block source authorization.
    root.rmdir()
    former.rename(root)
    with binding_tool.open_secure_marker_journal(
        value,
        repo_root=execution_context["repo"],
        workspace=execution_context["workspace"],
    ) as journal:
        root.rename(former)
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        with pytest.raises(
            binding_tool.FinalExecutionBindingError,
            match="state root identity differs",
        ):
            journal.revalidate_path_binding()


def test_secure_journal_rejects_special_leaf_and_marks_partial_create_consumed(
    execution_context: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _binding(execution_context)
    fifo_declaration = "3" * 64
    root = execution_context["state_root"]
    assert isinstance(root, Path)
    fifo = root / f"{fifo_declaration}.json"
    os.mkfifo(fifo)
    with binding_tool.open_secure_marker_journal(
        value,
        repo_root=execution_context["repo"],
        workspace=execution_context["workspace"],
    ) as journal:
        assert journal.exists(fifo_declaration) is True
        with pytest.raises(
            binding_tool.FinalExecutionBindingError, match="not a regular file"
        ):
            journal.read(fifo_declaration)

    declaration = "4" * 64
    payload = _marker_payload(
        execution_context,
        value,
        cohort="b2g26_final_71454",
        declaration_sha256=declaration,
        plan_sha256="5" * 64,
    )

    def short_write(_descriptor: int, _payload: bytes) -> int:
        return 0

    monkeypatch.setattr(binding_tool.os, "write", short_write)
    with binding_tool.open_secure_marker_journal(
        value,
        repo_root=execution_context["repo"],
        workspace=execution_context["workspace"],
    ) as journal:
        with pytest.raises(binding_tool.SourceAuthorizationJournalError) as raised:
            journal.create(declaration, payload)
    assert raised.value.consumed is True
    marker = root / f"{declaration}.json"
    assert marker.is_file()
    assert marker.stat().st_mode & 0o777 == 0o600
