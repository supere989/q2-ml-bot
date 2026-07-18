from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import pytest

from tools import compile_generated_cohort as compiler
from tools import retired_cohort_registry as registry
from tools import run_generator_cohort as cohort


STYLES = cohort.CONCRETE_STYLES


def _declaration() -> dict[str, object]:
    maps = []
    for ordinal in range(28):
        style = STYLES[ordinal // 4]
        # Deliberately reverse lexical order so an implementation cannot hide
        # a glob/sort substitution for declaration order.
        name = f"future_{27 - ordinal:02d}_{style}"
        maps.append({
            "grid": 5,
            "map": name,
            "observed_heat": None,
            "ordinal": ordinal,
            "seed": 90000000 + ordinal,
            "style": style,
        })
    return {
        "cohort_id": "future_compile_cohort",
        "generator": {
            "grid": 5,
            "gym": False,
            "observed_heat": None,
            "version": "v6",
        },
        "implementation_binding": {
            "bind_atlas_analyzer_closure": True,
            "bind_repository_commit": True,
            "bind_repository_tree": True,
            "require_clean_git": True,
        },
        "maps": maps,
        "mode": "final",
        "schema": cohort.DECLARATION_SCHEMA,
        "selection": {
            "policy": "all-or-nothing",
            "replacement_allowed": False,
            "required_concrete_styles": list(STYLES),
            "required_map_count": 28,
            "required_maps_per_style": 4,
            "salvage_allowed": False,
            "timing": "declared-before-generation",
        },
        "source_suffixes": list(cohort.SOURCE_SUFFIXES),
    }


def _write_pak(path: Path, members: list[tuple[str, bytes]]) -> None:
    data = bytearray()
    directory = bytearray()
    member_offset = 12
    for name, payload in members:
        encoded = name.encode("ascii")
        assert len(encoded) < 56
        data.extend(payload)
        directory.extend(struct.pack(
            "<56sii", encoded + b"\0" * (56 - len(encoded)), member_offset, len(payload)
        ))
        member_offset += len(payload)
    directory_offset = 12 + len(data)
    path.write_bytes(
        struct.pack("<4sii", b"PACK", directory_offset, len(directory))
        + bytes(data)
        + bytes(directory)
    )


def _write_mock_q2tool(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

expected = [
    '-bsp', '-vis', '-fast', '-rad', '-bounce', '0',
    '-threads', '1', '-basedir',
]
if sys.argv[1:10] != expected:
    print('flags differ: ' + repr(sys.argv[1:10]), file=sys.stderr)
    raise SystemExit(91)
map_path = Path(sys.argv[11])
record = {
    'argv': sys.argv[1:],
    'cwd': os.getcwd(),
    'map': map_path.stem,
}
with open(os.environ['MOCK_Q2TOOL_ORDER'], 'a', encoding='utf-8') as stream:
    stream.write(json.dumps(record, sort_keys=True) + '\\n')
print('compile ' + map_path.stem, flush=True)
print('diagnostic ' + map_path.stem, file=sys.stderr, flush=True)
if os.environ.get('MOCK_Q2TOOL_EMIT_PRT') == '1':
    map_path.with_suffix('.prt').write_bytes(('PRT:' + map_path.stem).encode('ascii'))
if map_path.stem == os.environ.get('MOCK_Q2TOOL_EXTRA_MAP'):
    suffix = os.environ.get('MOCK_Q2TOOL_EXTRA_SUFFIX', '.lin')
    map_path.with_suffix(suffix).write_bytes(('EXTRA:' + map_path.stem).encode('ascii'))
if map_path.stem == os.environ.get('MOCK_Q2TOOL_SLOW_MAP'):
    time.sleep(float(os.environ.get('MOCK_Q2TOOL_SLEEP_SECONDS', '2')))
if map_path.stem == os.environ.get('MOCK_Q2TOOL_FAIL_MAP'):
    raise SystemExit(23)
map_path.with_suffix('.bsp').write_bytes(('BSP:' + map_path.stem).encode('ascii'))
if map_path.stem == os.environ.get('MOCK_Q2TOOL_MUTATE_MAP'):
    target = os.environ['MOCK_Q2TOOL_MUTATE_TARGET']
    with open(target, 'ab') as stream:
        stream.write(b'\\n# deterministic late mutation\\n')
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _fixture(tmp_path: Path) -> dict[str, object]:
    declaration = _declaration()
    declaration_path = tmp_path / "declaration.json"
    declaration_path.write_bytes(cohort.canonical_bytes(declaration))
    source = tmp_path / "source"
    source.mkdir()
    for row in declaration["maps"]:
        for suffix in cohort.SOURCE_SUFFIXES:
            (source / f"{row['map']}{suffix}").write_bytes(
                f"source:{row['ordinal']}:{row['map']}:{suffix}\n".encode("ascii")
            )
    basedir = tmp_path / "baseq2"
    basedir.mkdir()
    _write_pak(
        basedir / "pak0.pak",
        [
            ("maps/example.bsp", b"example"),
            ("PICS/COLORMAP.PCX", b"canonical-colormap-bytes"),
        ],
    )
    q2tool = tmp_path / "q2tool"
    _write_mock_q2tool(q2tool)
    return {
        "declaration": declaration,
        "declaration_path": declaration_path,
        "source": source,
        "staging": tmp_path / "compiled-staging",
        "publish": tmp_path / "compiled-published",
        "logs": tmp_path / "compile-logs",
        "report": tmp_path / "compile-report.json",
        "q2tool": q2tool,
        "basedir": basedir,
        "order": tmp_path / "q2tool-order.jsonl",
    }


def _run(
    paths: dict[str, object], *, timeout_seconds: float = 3600.0
) -> dict[str, object]:
    return compiler.compile_generated_cohort(
        paths["declaration_path"],
        paths["source"],
        paths["staging"],
        paths["publish"],
        paths["logs"],
        paths["report"],
        paths["q2tool"],
        paths["basedir"],
        timeout_seconds=timeout_seconds,
    )


def _load_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def test_retired_declaration_is_rejected_before_any_output_mutation(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "compiler-outputs"
    staging = output_root / "compiled-staging"
    publish = output_root / "compiled-published"
    logs = output_root / "compile-logs"
    report = output_root / "reports" / "compile-report.json"
    declaration = (
        compiler.ROOT
        / "docs/multires/B2-GENERATED-COHORT-71438-DECLARATION.json"
    )

    with pytest.raises(
        compiler.CompileCohortError, match="permanently retired"
    ) as caught:
        compiler.compile_generated_cohort(
            declaration,
            tmp_path / "missing-source",
            staging,
            publish,
            logs,
            report,
            tmp_path / "missing-q2tool",
            tmp_path / "missing-baseq2",
        )

    assert isinstance(caught.value.__cause__, registry.RetiredCohortRegistryError)
    assert not output_root.exists()
    assert not staging.exists()
    assert not publish.exists()
    assert not logs.exists()
    assert not report.parent.exists()


def test_retired_declaration_cli_refusal_is_one_line_and_creates_no_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_root = tmp_path / "compiler-outputs"
    declaration = (
        compiler.ROOT
        / "docs/multires/B2-GENERATED-COHORT-71438-DECLARATION.json"
    )

    exit_code = compiler.main(
        [
            "--declaration",
            str(declaration),
            "--source-root",
            str(output_root / "source"),
            "--staging-root",
            str(output_root / "staging"),
            "--publish-root",
            str(output_root / "published"),
            "--log-root",
            str(output_root / "logs"),
            "--report",
            str(output_root / "reports/compile.json"),
            "--q2tool",
            str(output_root / "q2tool"),
            "--basedir",
            str(output_root / "basedir"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert captured.err.startswith("generated cohort compile failed: ")
    assert "permanently retired" in captured.err
    assert not output_root.exists()


def test_rejects_parent_of_real_baseq2_instead_of_searching_for_assets(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    assets = tmp_path / "assets"
    real_baseq2 = assets / "baseq2"
    real_baseq2.mkdir(parents=True)
    _write_pak(
        real_baseq2 / "pak0.pak",
        [(compiler.REQUIRED_PAK_MEMBER, b"colormap")],
    )
    paths["basedir"] = assets

    with pytest.raises(
        compiler.CompileCohortError, match=r"basedir/pak0\.pak"
    ):
        _run(paths)

    report = _load_report(paths["report"])
    assert report["failure"]["phase"] == "preflight"
    assert report["publication"]["compiled_stage_published"] is False
    assert not paths["staging"].exists()
    assert not paths["publish"].exists()


def test_rejects_pak0_without_case_insensitive_colormap_member(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _write_pak(paths["basedir"] / "pak0.pak", [("pics/other.pcx", b"wrong")])

    with pytest.raises(
        compiler.CompileCohortError, match=r"pics/colormap\.pcx.*found 0"
    ):
        _run(paths)

    report = _load_report(paths["report"])
    assert report["status"] == "failed-non-admissible-staging"
    assert report["assets"] is None
    assert not paths["publish"].exists()


def test_source_membership_mismatch_fails_before_staging(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    (paths["source"] / "unexpected.txt").write_text("not declared\n")

    with pytest.raises(
        compiler.CompileCohortError, match=r"unexpected file unexpected\.txt"
    ):
        _run(paths)

    report = _load_report(paths["report"])
    assert report["failure"]["phase"] == "source-membership"
    assert report["source_membership"]["passed"] is False
    assert report["source_membership"]["actual_file_count"] == 141
    assert not paths["staging"].exists()
    assert not paths["logs"].exists()
    assert not paths["publish"].exists()


def test_missing_report_parent_is_created_for_canonical_failure_evidence(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    paths["report"] = tmp_path / "new-evidence" / "reports" / "compile.json"
    (paths["source"] / "unexpected.txt").write_text("not declared\n")

    with pytest.raises(compiler.CompileCohortError):
        _run(paths)

    report = _load_report(paths["report"])
    assert report["failure"]["phase"] == "source-membership"
    assert report["publication"]["compiled_stage_published"] is False


def test_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    sentinel = b"retained prior evidence\n"
    paths["report"].write_bytes(sentinel)

    with pytest.raises(compiler.CompileCohortError, match="already exists"):
        _run(paths)

    assert paths["report"].read_bytes() == sentinel
    assert not paths["staging"].exists()
    assert not paths["publish"].exists()


def test_q2tool_failure_is_fail_fast_and_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    names = [row["map"] for row in paths["declaration"]["maps"]]
    monkeypatch.setenv("MOCK_Q2TOOL_ORDER", str(paths["order"]))
    monkeypatch.setenv("MOCK_Q2TOOL_FAIL_MAP", names[2])

    with pytest.raises(
        compiler.CompileCohortError,
        match=rf"q2tool failed for {names[2]} with exit code 23",
    ):
        _run(paths)

    invocations = [
        json.loads(line) for line in paths["order"].read_text().splitlines()
    ]
    report = _load_report(paths["report"])
    assert [row["map"] for row in invocations] == names[:3]
    assert len(report["maps"]) == 3
    assert report["maps"][2]["exit_code"] == 23
    assert report["maps"][2]["passed"] is False
    assert report["failure"]["phase"] == "compile"
    assert report["failure"]["failed_map"]["ordinal"] == 2
    assert report["publication"] == {
        "atomic": False,
        "compiled_stage_published": False,
        "staging_non_admissible": True,
    }
    assert paths["staging"].is_dir()
    assert not paths["publish"].exists()
    assert len(list(paths["logs"].iterdir())) == 6
    assert set(path.stem for path in paths["staging"].glob("*.bsp")) == set(names[:2])


def test_nonzero_failure_retains_prt_as_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    first = paths["declaration"]["maps"][0]["map"]
    monkeypatch.setenv("MOCK_Q2TOOL_ORDER", str(paths["order"]))
    monkeypatch.setenv("MOCK_Q2TOOL_EMIT_PRT", "1")
    monkeypatch.setenv("MOCK_Q2TOOL_FAIL_MAP", first)

    with pytest.raises(compiler.CompileCohortError, match="exit code 23"):
        _run(paths)

    prt_path = paths["staging"] / f"{first}.prt"
    report = _load_report(paths["report"])
    result = report["maps"][0]
    assert prt_path.read_bytes() == f"PRT:{first}".encode("ascii")
    assert result["prt"] == {
        "bytes": prt_path.stat().st_size,
        "sha256": cohort.file_sha256(prt_path),
    }
    assert result["prt_error"] is None
    assert result["prt_removed_after_success"] is False
    assert result["passed"] is False
    assert not paths["publish"].exists()


def test_q2tool_timeout_closes_logs_fails_fast_and_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    first = paths["declaration"]["maps"][0]["map"]
    monkeypatch.setenv("MOCK_Q2TOOL_ORDER", str(paths["order"]))
    monkeypatch.setenv("MOCK_Q2TOOL_EMIT_PRT", "1")
    monkeypatch.setenv("MOCK_Q2TOOL_SLOW_MAP", first)
    monkeypatch.setenv("MOCK_Q2TOOL_SLEEP_SECONDS", "2")

    with pytest.raises(
        compiler.CompileCohortError,
        match=rf"q2tool timed out for {first} after 0\.05 seconds",
    ):
        _run(paths, timeout_seconds=0.05)

    report = _load_report(paths["report"])
    assert len(report["maps"]) == 1
    result = report["maps"][0]
    assert result["timed_out"] is True
    assert result["exit_code"] == -9
    assert result["invocation_error"] == (
        "TimeoutExpired: exceeded per-map timeout of 0.05 seconds"
    )
    assert report["contract"]["per_map_timeout_seconds"] == 0.05
    assert report["failure"]["phase"] == "compile"
    stdout = paths["logs"] / result["stdout_log"]["path"]
    stderr = paths["logs"] / result["stderr_log"]["path"]
    assert f"compile {first}" in stdout.read_text()
    assert f"diagnostic {first}" in stderr.read_text()
    assert "compile orchestrator: TimeoutExpired" in stderr.read_text()
    assert "process group terminated" in stderr.read_text()
    assert result["stdout_log"]["sha256"] == cohort.file_sha256(stdout)
    assert result["stderr_log"]["sha256"] == cohort.file_sha256(stderr)
    prt_path = paths["staging"] / f"{first}.prt"
    assert prt_path.is_file()
    assert result["prt"] == {
        "bytes": prt_path.stat().st_size,
        "sha256": cohort.file_sha256(prt_path),
    }
    assert result["prt_removed_after_success"] is False
    assert paths["staging"].is_dir()
    assert not paths["publish"].exists()


@pytest.mark.parametrize("target_name", ["declaration_path", "q2tool", "pak0"])
def test_late_input_mutation_is_rejected_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    paths = _fixture(tmp_path)
    last = paths["declaration"]["maps"][-1]["map"]
    mutation_target = (
        paths["basedir"] / "pak0.pak"
        if target_name == "pak0"
        else paths[target_name]
    )
    monkeypatch.setenv("MOCK_Q2TOOL_ORDER", str(paths["order"]))
    monkeypatch.setenv("MOCK_Q2TOOL_MUTATE_MAP", last)
    monkeypatch.setenv("MOCK_Q2TOOL_MUTATE_TARGET", str(mutation_target))

    with pytest.raises(compiler.CompileCohortError):
        _run(paths)

    report = _load_report(paths["report"])
    assert len(report["maps"]) == 28
    assert all(result["passed"] is True for result in report["maps"])
    assert report["postcompile_membership"]["passed"] is True
    assert report["failure"]["phase"] == "input-stability"
    expected_fragments = {
        "declaration_path": "declaration changed or became invalid during compilation",
        "q2tool": "q2tool changed during compilation",
        "pak0": "basedir assets changed during compilation",
    }
    assert expected_fragments[target_name] in report["failure"]["message"]
    assert report["publication"]["compiled_stage_published"] is False
    assert paths["staging"].is_dir()
    assert not paths["publish"].exists()


def test_success_uses_declaration_order_and_atomically_publishes_exact_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    names = [row["map"] for row in paths["declaration"]["maps"]]
    assert names != sorted(names)
    monkeypatch.setenv("MOCK_Q2TOOL_ORDER", str(paths["order"]))

    report = _run(paths)

    invocations = [
        json.loads(line) for line in paths["order"].read_text().splitlines()
    ]
    published_files = {
        path.relative_to(paths["publish"]).as_posix()
        for path in paths["publish"].iterdir()
        if path.is_file()
    }
    expected_files = {
        f"{name}{suffix}"
        for name in names
        for suffix in (*cohort.SOURCE_SUFFIXES, ".bsp")
    }
    colormap = b"canonical-colormap-bytes"

    assert [row["map"] for row in invocations] == names
    assert [row["map"] for row in report["maps"]] == names
    assert all(
        row["command"][1:10] == list(compiler.Q2TOOL_FLAGS)
        and row["command"][10] == str(paths["basedir"].absolute())
        and row["command"][11].endswith(f"/{row['map']}.map")
        for row in report["maps"]
    )
    assert all(row["exit_code"] == 0 and row["passed"] is True for row in report["maps"])
    assert all(
        row["prt"] is None
        and row["prt_error"] is None
        and row["prt_removed_after_success"] is False
        for row in report["maps"]
    )
    assert not paths["staging"].exists()
    assert paths["publish"].is_dir()
    assert published_files == expected_files
    assert len(published_files) == 28 * 6
    assert len(list(paths["logs"].iterdir())) == 28 * 2
    assert report["status"] == "compiled-stage-published-non-admissible"
    assert report["passed"] is True
    assert report["bundle_admissible"] is False
    assert report["atlas_admissible"] is False
    assert report["publication"] == {
        "atomic": True,
        "compiled_stage_published": True,
        "staging_non_admissible": True,
    }
    assert report["postcompile_membership"]["passed"] is True
    assert report["postcompile_membership"]["expected_file_count"] == 168
    assert report["postcompile_membership"]["actual_file_count"] == 168
    assert report["assets"]["basedir"] == str(paths["basedir"].absolute())
    assert report["assets"]["required_member"] == {
        "bytes": len(colormap),
        "name": "PICS/COLORMAP.PCX",
        "offset": 19,
        "sha256": hashlib.sha256(colormap).hexdigest(),
    }
    assert report["assets"]["pak0"]["sha256"] == cohort.file_sha256(
        paths["basedir"] / "pak0.pak"
    )
    assert report["q2tool"]["sha256"] == cohort.file_sha256(paths["q2tool"])
    assert paths["report"].read_bytes() == cohort.canonical_bytes(report)
    assert "recorded_at" not in report
    assert compiler.verify_stage_membership(
        paths["declaration"], paths["publish"], "compiled"
    )["passed"] is True


def test_success_records_and_durably_removes_prt_before_exact_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setenv("MOCK_Q2TOOL_ORDER", str(paths["order"]))
    monkeypatch.setenv("MOCK_Q2TOOL_EMIT_PRT", "1")

    report = _run(paths)

    for result in report["maps"]:
        payload = f"PRT:{result['map']}".encode("ascii")
        assert result["prt"] == {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        assert result["prt_error"] is None
        assert result["prt_removed_after_success"] is True
        assert result["passed"] is True
    assert not list(paths["publish"].glob("*.prt"))
    assert len(list(paths["publish"].iterdir())) == 168
    assert report["postcompile_membership"]["passed"] is True
    assert report["postcompile_membership"]["actual_file_count"] == 168


def test_unknown_q2tool_output_is_not_deleted_and_rejects_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    first = paths["declaration"]["maps"][0]["map"]
    monkeypatch.setenv("MOCK_Q2TOOL_ORDER", str(paths["order"]))
    monkeypatch.setenv("MOCK_Q2TOOL_EXTRA_MAP", first)
    monkeypatch.setenv("MOCK_Q2TOOL_EXTRA_SUFFIX", ".lin")

    with pytest.raises(
        compiler.CompileCohortError, match="postcompile membership differs"
    ):
        _run(paths)

    extra = paths["staging"] / f"{first}.lin"
    report = _load_report(paths["report"])
    assert extra.read_bytes() == f"EXTRA:{first}".encode("ascii")
    assert f"unexpected file {first}.lin" in report["postcompile_membership"][
        "failures"
    ]
    assert report["failure"]["phase"] == "postcompile-membership"
    assert report["publication"]["compiled_stage_published"] is False
    assert not paths["publish"].exists()
