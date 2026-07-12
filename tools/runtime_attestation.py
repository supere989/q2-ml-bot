#!/usr/bin/env python3
"""Build or verify a path-independent Q2 rollout runtime manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.runtime_attestation import (
    AttestationError,
    build_runtime_manifest,
    load_runtime_manifest,
    verify_runtime_manifest,
    write_runtime_manifest,
)


def _runtime_pair(value: str) -> tuple[str, object]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("runtime values must be KEY=JSON_VALUE")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("runtime key cannot be empty")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = raw
    return key, parsed


def _environment_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("environment values must be KEY=VALUE")
    key, raw = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError("environment key cannot be empty")
    return key, raw


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--q2-root",
        type=Path,
        default=Path(os.environ.get("Q2_ROOT", "q2_lithium_merge")),
    )
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--map", action="append", dest="maps", default=[])
    parser.add_argument(
        "--map-glob",
        action="append",
        default=[],
        help="glob relative to baseq2/maps; matched BSP stems are attested",
    )
    parser.add_argument("--rust-extension", type=Path)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument(
        "--runtime",
        action="append",
        default=[],
        type=_runtime_pair,
        metavar="KEY=JSON_VALUE",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        type=_environment_pair,
        metavar="KEY=VALUE",
        help="override an attested environment value in this process",
    )
    parser.add_argument(
        "--hmac-key-env",
        default="",
        help="name of an environment variable containing the signing key",
    )


def _maps(args) -> list[str]:
    names = list(args.maps)
    for pattern in args.map_glob:
        for directory in (
            args.q2_root / "lithium" / "maps",
            args.q2_root / "baseq2" / "maps",
        ):
            names.extend(path.stem for path in directory.glob(pattern) if path.suffix == ".bsp")
    return sorted(set(names))


def _hmac_key(args) -> bytes | None:
    if not args.hmac_key_env:
        return None
    value = os.environ.get(args.hmac_key_env)
    if value is None:
        raise AttestationError(
            f"signing-key environment variable is unset: {args.hmac_key_env}"
        )
    return value.encode("utf-8")


def _build(args) -> dict:
    environment = dict(os.environ)
    for key, value in args.env:
        environment[key] = value
        # Policy/protocol modules use import-time environment switches.
        os.environ[key] = value
    runtime = dict(args.runtime)
    return build_runtime_manifest(
        q2_root=args.q2_root,
        source_root=args.source_root,
        map_names=_maps(args),
        rust_extension=args.rust_extension,
        source_revision=args.source_revision or None,
        runtime_config=runtime,
        environment=environment,
        policy_checkpoint=args.policy_checkpoint,
        hmac_key=_hmac_key(args),
    )


def _emit(value: dict, output: str) -> None:
    if output == "-":
        print(json.dumps(value, indent=2, sort_keys=True))
    else:
        write_runtime_manifest(Path(output), value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="measure and seal the current runtime")
    _add_build_arguments(build)
    build.add_argument("--output", default="-")

    verify = commands.add_parser(
        "verify", help="measure the current runtime and compare it to an expected manifest"
    )
    _add_build_arguments(verify)
    verify.add_argument("--expected", type=Path, required=True)
    verify.add_argument("--current-output", default="")
    verify.add_argument("--require-signature", action="store_true")

    validate = commands.add_parser(
        "validate", help="validate the digest/signature of a stored manifest"
    )
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--hmac-key-env", default="")
    validate.add_argument("--require-signature", action="store_true")

    args = parser.parse_args()
    try:
        if args.command == "build":
            manifest = _build(args)
            _emit(manifest, args.output)
            print(
                json.dumps({
                    "event": "runtime_manifest_built",
                    "manifest_sha256": manifest["manifest_sha256"],
                    "output": args.output,
                }, sort_keys=True),
                file=sys.stderr,
            )
            return 0
        if args.command == "verify":
            expected = load_runtime_manifest(args.expected)
            current = _build(args)
            if args.current_output:
                _emit(current, args.current_output)
            result = verify_runtime_manifest(
                current,
                expected=expected,
                hmac_key=_hmac_key(args),
                require_signature=args.require_signature,
            )
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
            return 0 if result.valid else 2

        manifest = load_runtime_manifest(args.manifest)
        result = verify_runtime_manifest(
            manifest,
            hmac_key=_hmac_key(args),
            require_signature=args.require_signature,
        )
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return 0 if result.valid else 2
    except AttestationError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
