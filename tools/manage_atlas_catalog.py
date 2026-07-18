#!/usr/bin/env python3
"""Author or validate one immutable multi-map Atlas catalog."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.atlas_catalog import (  # noqa: E402
    AtlasCatalogError, author_catalog, canonical_bytes, load_atlas_catalog,
)


def _load_spec(path: Path) -> dict:
    source = Path(path).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise AtlasCatalogError("map spec must be an absolute regular non-symlink file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AtlasCatalogError("map spec is invalid JSON") from error
    if not isinstance(value, dict):
        raise AtlasCatalogError("map spec must be a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    author = commands.add_parser("author")
    author.add_argument("--map-spec", type=Path, action="append", required=True)
    author.add_argument("--output", type=Path, required=True)
    author.add_argument("--rust-extension", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--catalog", type=Path, required=True)
    validate.add_argument("--expected-catalog-sha256")
    validate.add_argument("--rust-extension", type=Path, required=True)
    return parser


def _load_extension(path: Path):
    source = Path(path).expanduser()
    if not source.is_absolute() or source.is_symlink() or not source.is_file():
        raise AtlasCatalogError(
            "Rust extension must be an absolute regular non-symlink file"
        )
    spec = importlib.util.spec_from_file_location("q2_lattice_rs", source.resolve())
    if spec is None or spec.loader is None:
        raise AtlasCatalogError("Rust extension cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        extension = _load_extension(args.rust_extension)
        if args.command == "author":
            output = Path(args.output).expanduser()
            if not output.is_absolute() or output.is_symlink() or output.exists():
                raise AtlasCatalogError("catalog output must be a new absolute non-symlink path")
            output.parent.mkdir(parents=True, exist_ok=True)
            document = author_catalog(
                [_load_spec(path) for path in args.map_spec],
                catalog_path=output,
                rust_extension_path=Path(args.rust_extension),
                extension_module=extension,
            )
            descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(canonical_bytes(document) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            catalog = load_atlas_catalog(
                output, expected_sha256=document["atlas_catalog_sha256"],
                extension_module=extension,
            )
        else:
            catalog = load_atlas_catalog(
                args.catalog, expected_sha256=args.expected_catalog_sha256,
                extension_module=extension,
            )
        result = {
            "schema": "q2-multires-atlas-catalog-validation-v1",
            "status": "pass", "catalog": str(catalog.path),
            "catalog_file_sha256": catalog.file_sha256,
            "atlas_catalog_sha256": catalog.atlas_catalog_sha256,
            "map_names": [record.map_name for record in catalog.maps],
            "map_count": len(catalog.maps), "client_count": 4,
        }
    except (AtlasCatalogError, OSError) as error:
        print(f"Atlas catalog failed: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
