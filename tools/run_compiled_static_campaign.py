#!/usr/bin/env python3
"""Publish exact compiled/static validation for one declared B2 cohort."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_generator_cohort import (  # noqa: E402
    GeneratorCohortError,
    _default_static_validator,
    canonical_bytes,
    load_declaration,
    verify_stage_membership,
)


SCHEMA = "q2-generator-v6-compiled-static-campaign-v1"


class CompiledStaticCampaignError(ValueError):
    """Raised when exact compiled input cannot produce a complete report."""


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def validate_compiled_static(
    declaration_path: Path,
    compiled_dir: Path,
    *,
    validator: Callable[[Path], Mapping[str, Any]] = _default_static_validator,
) -> dict[str, Any]:
    declaration, _declaration_sha256 = load_declaration(declaration_path)
    membership = verify_stage_membership(declaration, compiled_dir, "compiled")
    if membership["passed"] is not True:
        raise CompiledStaticCampaignError(
            "compiled exact membership failed: "
            + "; ".join(membership["failures"])
        )
    rows = []
    failures = []
    for declared in declaration["maps"]:
        try:
            result = dict(validator(compiled_dir / f"{declared['map']}.map"))
        except Exception as exc:
            raise CompiledStaticCampaignError(
                f"{declared['map']} static validator raised: {exc}"
            ) from exc
        if result.get("map") != declared["map"]:
            failures.append(f"{declared['map']}: validator map identity differs")
        if result.get("static_ok") is not True:
            failures.append(f"{declared['map']}: static validation failed")
        rows.append(result)
    return {
        "schema": SCHEMA,
        "map_count": len(rows),
        "pass_count": sum(row.get("static_ok") is True for row in rows),
        "passed": not failures,
        "maps": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--compiled-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise CompiledStaticCampaignError("compiled static report already exists")
        if _path_is_within(args.output, args.compiled_dir):
            raise CompiledStaticCampaignError(
                "compiled static report must be outside the exact compiled root"
            )
        report = validate_compiled_static(args.declaration, args.compiled_dir)
        _exclusive_write(args.output, canonical_bytes(report))
    except (CompiledStaticCampaignError, GeneratorCohortError, OSError) as exc:
        print(f"compiled static campaign failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_bytes(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
