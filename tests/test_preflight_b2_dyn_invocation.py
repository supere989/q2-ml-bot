from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import pytest

from tools.bind_b2_dyn_origin import (
    ARTIFACT_PREFLIGHT_SCHEMA,
    DynOriginBindingError,
    bind_origin,
)
from tools.preflight_b2_dyn_invocation import (
    DynInvocationPreflightError,
    EXPECTED_STDOUT,
    SCHEMA,
    load_shape_preflight,
    preflight,
)
from tools.run_preflighted_b2_dyn import (
    execute,
    load_origin_binding,
)
from tools.run_generator_cohort import canonical_bytes


ANALYZER = "11" * 32
MAP_ID = "map"
ORIGIN = [-512, 0, -512]


def _repo(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", repo], check=True)
    (repo / "tracked").write_text("authority\n")
    declaration_path = (
        repo / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
    )
    declaration_path.parent.mkdir(parents=True)
    declaration_path.write_bytes(
        canonical_bytes(
            {
                "cohort_id": "b2g26_final_test",
                "maps": [
                    {"map": MAP_ID if ordinal == 0 else f"map_{ordinal:02d}"}
                    for ordinal in range(28)
                ],
            }
        )
    )
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repo,
            "-c",
            "user.name=B2 Test",
            "-c",
            "user.email=b2@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD^{tree}"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    binding = {
        "repository_commit": commit,
        "repository_tree": tree,
        "git_clean": True,
        "generator_sha256": "22" * 32,
        "routes_sha256": "33" * 32,
        "atlas_analyzer_authority_sha256": "44" * 32,
        "atlas_analyzer_authority_file_count": 1,
    }
    return repo, commit, binding


def _executable(tmp_path: Path, *, touch_output: bool = False) -> Path:
    executable = tmp_path / "q2-dyn-evidence"
    body = [
        "#!/usr/bin/env python3",
        "import json, pathlib, sys",
        "preflight = sys.argv[-2:] == ['--preflight-only', 'true']",
        "artifact = sys.argv[-2:] == ['--verify-artifacts-only', 'true']",
        "origins = [v for i, v in enumerate(sys.argv) if i and sys.argv[i-1] == '--expected-origin']",
        "if preflight and not origins:",
        f"    print('{json.dumps({'passed': True, 'schema': SCHEMA}, separators=(',', ':'))}')",
        "elif preflight:",
        "    assert origins == ['-512,0,-512']",
        f"    print('{json.dumps({'passed': True, 'schema': SCHEMA}, separators=(',', ':'))}')",
        "elif artifact:",
        "    assert origins == ['-512,0,-512']",
        f"    print('{json.dumps({'passed': True, 'schema': ARTIFACT_PREFLIGHT_SCHEMA}, separators=(',', ':'))}')",
        "else:",
        "    assert origins == ['-512,0,-512']",
        "    output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])",
        "    output.mkdir(parents=True)",
        "    report = output / 'b2-dyn-evidence.json'",
        "    report.write_text('{}\\n')",
        "    print(report)",
    ]
    if touch_output:
        body.insert(
            7,
            "    pathlib.Path(sys.argv[sys.argv.index('--output') + 1]).mkdir(parents=True)",
        )
    executable.write_text("\n".join(body) + "\n")
    executable.chmod(0o755)
    return executable


def _args(
    tmp_path: Path,
    repo: Path,
    commit: str,
    executable: Path,
) -> argparse.Namespace:
    root = tmp_path / "future-final-cohort"
    return argparse.Namespace(
        executable=executable,
        repo_root=repo,
        atlas=root / "analysis" / f"{MAP_ID}.atlas.bin",
        manifest=root / "analysis" / f"{MAP_ID}.atlas.manifest.json",
        bsp=root / "claims" / f"{MAP_ID}.bsp",
        expected_map_id=MAP_ID,
        expected_analyzer_authority=ANALYZER,
        expected_crate_commit=commit,
        map_epoch=1,
        environment_steps=4000,
        samples=4000,
        output=root / "dyn-evidence",
        report=tmp_path / "dyn-argv-shape-preflight.json",
    )


def _write_artifacts(args: argparse.Namespace) -> Path:
    args.atlas.parent.mkdir(parents=True)
    args.bsp.parent.mkdir(parents=True)
    atlas_header = bytearray(136)
    atlas_header[:8] = b"Q2ATL001"
    struct.pack_into("<HHIqqq", atlas_header, 8, 1, 0x454C, 136, *ORIGIN)
    args.atlas.write_bytes(bytes(atlas_header))
    args.bsp.write_bytes(b"bsp-bytes")
    atlas_sha = hashlib.sha256(args.atlas.read_bytes()).hexdigest()
    bsp_sha = hashlib.sha256(args.bsp.read_bytes()).hexdigest()
    manifest = {
        "analyzer": {"sha256": ANALYZER},
        "artifacts": {
            args.atlas.name: {
                "sha256_uncompressed": atlas_sha,
                "uncompressed_size": args.atlas.stat().st_size,
            }
        },
        "bsp": {
            "canonical_map_id": MAP_ID,
            "sha256": bsp_sha,
            "size_bytes": args.bsp.stat().st_size,
        },
        "grid": {
            "model0_mins": [-321, 191, -321],
            "origin": ORIGIN,
        },
    }
    args.manifest.write_bytes(canonical_bytes(manifest))
    manifest_sha = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    analysis_path = args.manifest.parent / f"{MAP_ID}.analysis.manifest.json"
    analysis = {
        "artifacts": {"atlas_manifest": {"sha256": manifest_sha}},
        "canonical_map_id": MAP_ID,
        "grid": {"origin": ORIGIN},
        "identity": {
            "analyzer_sha256": ANALYZER,
            "atlas_manifest_sha256": manifest_sha,
            "atlas_sha256": atlas_sha,
            "bsp_sha256": bsp_sha,
        },
    }
    analysis_path.write_bytes(canonical_bytes(analysis))
    promotion_path = args.output.parent / "reports" / "generated-promotion.json"
    promotion_path.parent.mkdir(parents=True)
    declaration_path = (
        args.repo_root / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
    )
    promotion_maps = [
        {
            "atlas_sha256": atlas_sha if ordinal == 0 else f"{ordinal + 1:064x}",
            "bsp_sha256": bsp_sha if ordinal == 0 else f"{ordinal + 101:064x}",
            "failures": [],
            "map": MAP_ID if ordinal == 0 else f"map_{ordinal:02d}",
            "passed": True,
        }
        for ordinal in range(28)
    ]
    promotion = {
        "cohort_id": "b2g26_final_test",
        "declaration_sha256": hashlib.sha256(
            declaration_path.read_bytes()
        ).hexdigest(),
        "expected_count": 28,
        "failures": [],
        "map_count": 28,
        "maps": promotion_maps,
        "pass_count": 28,
        "passed": True,
        "phase": "compiled_validation",
        "schema": "q2-generator-claim-campaign-v2",
    }
    promotion_path.write_bytes(canonical_bytes(promotion))
    return promotion_path


def _stub_binding(
    monkeypatch: pytest.MonkeyPatch,
    binding: dict[str, object],
) -> None:
    for module in (
        "tools.preflight_b2_dyn_invocation",
        "tools.bind_b2_dyn_origin",
        "tools.run_preflighted_b2_dyn",
    ):
        monkeypatch.setattr(f"{module}.repository_binding", lambda _: binding)


def _phase_a_and_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[argparse.Namespace, Path, Path, dict[str, object], dict[str, object]]:
    repo, commit, repository = _repo(tmp_path)
    _stub_binding(monkeypatch, repository)
    args = _args(tmp_path, repo, commit, _executable(tmp_path))
    shape = preflight(args)
    args.report.write_bytes(canonical_bytes(shape))
    promotion = _write_artifacts(args)
    binding_path = tmp_path / "dyn-origin-binding.json"
    binding = bind_origin(args.report, promotion, binding_path)
    binding_path.write_bytes(canonical_bytes(binding))
    return args, args.report, binding_path, shape, binding


def test_phase_a_defers_origin_and_touches_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, commit, repository = _repo(tmp_path)
    _stub_binding(monkeypatch, repository)
    args = _args(tmp_path, repo, commit, _executable(tmp_path))

    report = preflight(args)

    assert report["passed"] is True
    assert report["origin_binding_status"] == "deferred-until-promoted-artifact"
    assert not any(
        value.startswith("--expected-origin")
        for value in report["producer_argv_without_origin"]
    )
    assert report["preflight_argv"][-2:] == ["--preflight-only", "true"]
    assert not args.output.exists()


def test_phase_a_rejects_any_output_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, commit, repository = _repo(tmp_path)
    _stub_binding(monkeypatch, repository)
    args = _args(tmp_path, repo, commit, _executable(tmp_path, touch_output=True))

    with pytest.raises(DynInvocationPreflightError, match="touched the producer output"):
        preflight(args)


def test_phase_b_derives_asymmetric_origin_and_verifies_artifacts_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, _shape_path, _binding_path, _shape, binding = _phase_a_and_b(
        tmp_path, monkeypatch
    )

    assert binding["identity"]["origin"] == ORIGIN
    assert binding["identity"]["origin_token"] == "-512,0,-512"
    assert binding["artifact_preflight_argv"][-2:] == [
        "--verify-artifacts-only",
        "true",
    ]
    assert not args.output.exists()


def test_dual_authority_runner_executes_exact_bound_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, shape_path, binding_path, _shape, _binding = _phase_a_and_b(
        tmp_path, monkeypatch
    )

    published = execute(
        load_shape_preflight(shape_path),
        load_origin_binding(binding_path),
    )

    assert published == args.output / "b2-dyn-evidence.json"
    assert published.read_bytes() == b"{}\n"


def test_phase_b_rejects_manifest_analysis_origin_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, commit, repository = _repo(tmp_path)
    _stub_binding(monkeypatch, repository)
    args = _args(tmp_path, repo, commit, _executable(tmp_path))
    shape = preflight(args)
    args.report.write_bytes(canonical_bytes(shape))
    promotion = _write_artifacts(args)
    analysis_path = args.manifest.parent / f"{MAP_ID}.analysis.manifest.json"
    analysis = json.loads(analysis_path.read_bytes())
    analysis["grid"]["origin"] = [-512, -512, -512]
    analysis_path.write_bytes(canonical_bytes(analysis))

    with pytest.raises(DynOriginBindingError, match="origins differ"):
        bind_origin(args.report, promotion, tmp_path / "binding.json")


def test_phase_b_rejects_artifacts_absent_from_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, commit, repository = _repo(tmp_path)
    _stub_binding(monkeypatch, repository)
    args = _args(tmp_path, repo, commit, _executable(tmp_path))
    shape = preflight(args)
    args.report.write_bytes(canonical_bytes(shape))
    promotion = _write_artifacts(args)
    report = json.loads(promotion.read_bytes())
    report["maps"][0]["atlas_sha256"] = "ff" * 32
    promotion.write_bytes(canonical_bytes(report))

    with pytest.raises(DynOriginBindingError, match="not admitted by promotion"):
        bind_origin(args.report, promotion, tmp_path / "binding.json")


def test_runner_refuses_artifact_mutation_after_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, shape_path, binding_path, _shape, _binding = _phase_a_and_b(
        tmp_path, monkeypatch
    )
    args.atlas.write_bytes(b"mutated")

    with pytest.raises(ValueError, match="artifact bytes differ"):
        execute(load_shape_preflight(shape_path), load_origin_binding(binding_path))


def test_runner_repeats_rust_origin_fence_before_irreversible_execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, shape_path, binding_path, _shape, binding = _phase_a_and_b(
        tmp_path, monkeypatch
    )
    origin_ordinal = binding["producer_argv"].index("--expected-origin") + 1
    binding["producer_argv"][origin_ordinal] = "-512,-512,-512"
    binding["parser_preflight_argv"][origin_ordinal] = "-512,-512,-512"
    binding["artifact_preflight_argv"][origin_ordinal] = "-512,-512,-512"
    binding["identity"]["origin"] = [-512, -512, -512]
    binding["identity"]["origin_token"] = "-512,-512,-512"
    binding_path.write_bytes(canonical_bytes(binding))

    with pytest.raises(ValueError, match="final no-write Dyn artifact verification"):
        execute(load_shape_preflight(shape_path), load_origin_binding(binding_path))
    assert not args.output.exists()


def test_runner_refuses_declaration_mutation_after_phase_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, shape_path, binding_path, _shape, _binding = _phase_a_and_b(
        tmp_path, monkeypatch
    )
    declaration = (
        args.repo_root / "docs/multires/B2-GENERATED-COHORT-DECLARATION.json"
    )
    declaration.write_bytes(declaration.read_bytes() + b" ")

    with pytest.raises(ValueError, match="active-declaration bytes differ"):
        execute(load_shape_preflight(shape_path), load_origin_binding(binding_path))


def test_origin_binding_rejects_equals_glued_origin(tmp_path: Path) -> None:
    producer_argv = ["/dyn", "--expected-origin=-512,0,-512"]
    report = {
        "artifact_preflight_argv": [
            *producer_argv,
            "--verify-artifacts-only",
            "true",
        ],
        "artifact_preflight_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "artifact_preflight_stdout_sha256": "00" * 32,
        "artifacts": {},
        "declaration": {},
        "executable": {},
        "identity": {},
        "parser_preflight_argv": [*producer_argv, "--preflight-only", "true"],
        "parser_preflight_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "parser_preflight_stdout_sha256": hashlib.sha256(EXPECTED_STDOUT).hexdigest(),
        "passed": True,
        "producer_argv": producer_argv,
        "producer_output_absent_after": True,
        "producer_output_absent_before": True,
        "promotion": {},
        "repository": {},
        "schema": "q2-b2-dyn-origin-binding-v1",
        "shape_preflight": {},
    }
    path = tmp_path / "binding.json"
    path.write_bytes(canonical_bytes(report))

    with pytest.raises(ValueError, match="equals-glued"):
        load_origin_binding(path)


def test_shape_loader_rejects_concrete_origin(tmp_path: Path) -> None:
    producer_argv = ["/dyn", "--expected-origin", "-512,0,-512"]
    report = {
        "executable": {},
        "origin_binding_status": "deferred-until-promoted-artifact",
        "passed": True,
        "preflight_argv": [*producer_argv, "--preflight-only", "true"],
        "preflight_stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "preflight_stdout_sha256": hashlib.sha256(EXPECTED_STDOUT).hexdigest(),
        "producer_argv_without_origin": producer_argv,
        "producer_output_absent_after": True,
        "producer_output_absent_before": True,
        "repository": {},
        "schema": SCHEMA,
    }
    path = tmp_path / "shape.json"
    path.write_bytes(canonical_bytes(report))

    with pytest.raises(ValueError, match="must not contain a concrete origin"):
        load_shape_preflight(path)
