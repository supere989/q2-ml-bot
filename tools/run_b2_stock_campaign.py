#!/usr/bin/env python3
"""Build and atomically publish exact stock q2dm1..q2dm8 B2 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_map_atlas import extract_stock  # noqa: E402
from tools.generator_claim_validator import validate_stock_analysis  # noqa: E402
from tools.run_generated_atlas_campaign import (  # noqa: E402
    GeneratedAtlasCampaignError,
    _fsync_directory,
    _rename_noreplace,
    _reserve_report,
)
from tools.run_generator_cohort import (  # noqa: E402
    STAGE_SUFFIXES,
    canonical_bytes,
    file_sha256,
    repository_binding,
)


SCHEMA = "q2-b2-stock-campaign-v1"
STOCK_IDS = tuple(f"q2dm{number}" for number in range(1, 9))


class B2StockCampaignError(ValueError):
    """Raised when the exact stock evidence bundle cannot be published."""


def _file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise B2StockCampaignError(f"stock input must be a regular file: {path}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _exact_flat(root: Path, expected: set[str], label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise B2StockCampaignError(f"{label} is not a plain directory")
    actual = {path.name for path in root.iterdir()}
    if actual != expected or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise B2StockCampaignError(
            f"{label} exact membership differs: "
            f"missing={sorted(expected - actual)!r} "
            f"unexpected={sorted(actual - expected)!r}"
        )


def _copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise B2StockCampaignError(f"stock build output is not regular: {source}")
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o444,
    )
    try:
        with source.open("rb") as input_stream, os.fdopen(
            descriptor, "wb"
        ) as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _root_records(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.iterdir())
    }


def run_stock_campaign(
    *,
    repo_root: Path,
    python: Path,
    stock_pak: Path,
    provenance: Path,
    stock_inventory: Path,
    b1_gate: Path,
    client_root: Path,
    lithium_root: Path,
    hook_attestation: Path,
    fall_oracle: Path,
    packer: Path,
    verifier: Path,
    output_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    if repo_root.resolve() != ROOT.resolve():
        raise B2StockCampaignError("stock repo root is not the executing repository")
    if not output_root.is_absolute() or not report_path.is_absolute():
        raise B2StockCampaignError("stock output and report paths must be absolute")
    if output_root.exists() or output_root.is_symlink():
        raise B2StockCampaignError("stock output root already exists")
    if report_path.exists() or report_path.is_symlink():
        raise B2StockCampaignError("stock report already exists")
    if not output_root.parent.is_dir() or output_root.parent.is_symlink():
        raise B2StockCampaignError("stock output parent is absent or a symlink")
    if not report_path.parent.is_dir() or report_path.parent.is_symlink():
        raise B2StockCampaignError("stock report parent is absent or a symlink")
    try:
        report_path.relative_to(output_root)
    except ValueError:
        pass
    else:
        raise B2StockCampaignError("stock report must be outside output root")

    inputs = {
        name: _file_record(path)
        for name, path in {
            "python": python,
            "stock_pak": stock_pak,
            "provenance": provenance,
            "stock_inventory": stock_inventory,
            "b1_gate": b1_gate,
            "hook_attestation": hook_attestation,
            "fall_oracle": fall_oracle,
            "packer": packer,
            "verifier": verifier,
            "cm_oracle": client_root / "release/q2-cm-oracle",
            "pmove_oracle": client_root / "release/q2-pmove-oracle",
            "hook_oracle": lithium_root / "tools/q2-hook-oracle",
        }.items()
    }
    before = repository_binding(repo_root)
    if before.get("git_clean") is not True:
        raise B2StockCampaignError("stock campaign repository is not clean")

    reservation = _reserve_report(report_path)
    work = Path(
        tempfile.mkdtemp(prefix=".b2-stock-campaign-", dir=output_root.parent)
    )
    published = False
    try:
        build_root = work / "analysis-build"
        bundle = work / "bundle"
        bsp_root = bundle / "bsp"
        analysis_root = bundle / "analysis"
        validation_root = bundle / "validation"
        build_root.mkdir()
        bundle.mkdir()
        bsp_root.mkdir()
        analysis_root.mkdir()
        validation_root.mkdir()

        command = [
            str(python),
            str(repo_root / "tools/build_map_atlas.py"),
            "--stock-pak", str(stock_pak),
            "--provenance", str(provenance),
            "--output", str(build_root),
            "--client-root", str(client_root),
            "--lithium-root", str(lithium_root),
            "--hook-attestation", str(hook_attestation),
            "--fall-oracle", str(fall_oracle),
            "--packer", str(packer),
            "--verifier", str(verifier),
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise B2StockCampaignError(
                f"stock Atlas builder failed with exit {completed.returncode}"
            )
        try:
            summary = json.loads(completed.stdout)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise B2StockCampaignError("stock Atlas builder stdout differs") from exc
        if (
            summary.get("schema") != "q2-atlas-stock-build-v1"
            or [row.get("canonical_map_id") for row in summary.get("maps", [])]
            != list(STOCK_IDS)
        ):
            raise B2StockCampaignError("stock Atlas build summary differs")

        extracted = extract_stock(stock_pak, bsp_root)
        if set(extracted) != set(STOCK_IDS):
            raise B2StockCampaignError("stock BSP extraction differs")
        analysis_names = {
            f"{map_id}{suffix}"
            for map_id in STOCK_IDS
            for suffix in STAGE_SUFFIXES["analysis"]
        }
        for name in sorted(analysis_names):
            _copy_regular(build_root / name, analysis_root / name)
        validation_names = {
            f"{map_id}.stock-validation.json" for map_id in STOCK_IDS
        }
        rows = []
        for map_id in STOCK_IDS:
            validation = validate_stock_analysis(
                bsp_root / f"{map_id}.bsp",
                analysis_root / f"{map_id}.analysis.manifest.json",
                b1_gate_path=b1_gate,
                stock_provenance_path=provenance,
                stock_inventory_path=stock_inventory,
            )
            if validation.get("passed") is not True:
                raise B2StockCampaignError(f"stock validation failed for {map_id}")
            validation_path = validation_root / f"{map_id}.stock-validation.json"
            _write_new(validation_path, canonical_bytes(validation))
            rows.append({
                "map": map_id,
                "bsp": _file_record(bsp_root / f"{map_id}.bsp"),
                "analysis_manifest": _file_record(
                    analysis_root / f"{map_id}.analysis.manifest.json"
                ),
                "validation": _file_record(validation_path),
                "passed": True,
            })

        _exact_flat(
            bsp_root, {f"{map_id}.bsp" for map_id in STOCK_IDS}, "stock BSP root"
        )
        _exact_flat(analysis_root, analysis_names, "stock analysis root")
        _exact_flat(validation_root, validation_names, "stock validation root")
        if repository_binding(repo_root) != before:
            raise B2StockCampaignError("repository changed during stock campaign")

        report = {
            "schema": SCHEMA,
            "passed": True,
            "repository": before,
            "inputs": inputs,
            "builder": {
                "command": command,
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
            },
            "map_count": len(rows),
            "maps": rows,
            "roots": {
                "bsp": _root_records(bsp_root),
                "analysis": _root_records(analysis_root),
                "validation": _root_records(validation_root),
            },
            "failures": [],
        }
        _fsync_directory(bsp_root)
        _fsync_directory(analysis_root)
        _fsync_directory(validation_root)
        _fsync_directory(bundle)
        _rename_noreplace(bundle, output_root)
        published = True
        reservation.write(report)
        return report
    finally:
        reservation.close()
        if work.exists() and not work.is_symlink():
            shutil.rmtree(work, ignore_errors=True)
        if not published and output_root.exists() and not output_root.is_symlink():
            # Publication can happen only as the final single-root rename. If a
            # later report fsync fails, retain the exact output as terminal
            # forensic evidence; never silently remove or reopen it.
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--stock-pak", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--stock-inventory", type=Path, required=True)
    parser.add_argument("--b1-gate", type=Path, required=True)
    parser.add_argument("--client-root", type=Path, required=True)
    parser.add_argument("--lithium-root", type=Path, required=True)
    parser.add_argument("--hook-attestation", type=Path, required=True)
    parser.add_argument("--fall-oracle", type=Path, required=True)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_stock_campaign(
            repo_root=args.repo_root,
            python=args.python,
            stock_pak=args.stock_pak,
            provenance=args.provenance,
            stock_inventory=args.stock_inventory,
            b1_gate=args.b1_gate,
            client_root=args.client_root,
            lithium_root=args.lithium_root,
            hook_attestation=args.hook_attestation,
            fall_oracle=args.fall_oracle,
            packer=args.packer,
            verifier=args.verifier,
            output_root=args.output_root,
            report_path=args.report,
        )
    except (
        B2StockCampaignError,
        GeneratedAtlasCampaignError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(f"B2 stock campaign refused: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
