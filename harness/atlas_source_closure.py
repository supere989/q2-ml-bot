"""Single source of truth for the Atlas analyzer authority closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ATLAS_ANALYZER_FIXED_INPUTS = (
    "harness/atlas_analyzer.py",
    "harness/atlas_source_closure.py",
    "harness/atlas_b1_authority.py",
    "harness/atlas_drop_replay.py",
    "harness/atlas_exact_drops.py",
    "harness/atlas_entity_semantics.py",
    "harness/atlas_surface_bands.py",
    "harness/atlas_teleporter_edges.py",
    "harness/generated_claim_probes.py",
    "harness/hook_claims_v3.py",
    "harness/ibsp38.py",
    "tools/atlas_cold_worker.py",
    "docs/MULTIRES-LATTICE-MAP-ATLAS-DESIGN-2026-07-14.md",
    "docs/MULTIRES-LATTICE-MAP-ATLAS-PLAN-2026-07-14.md",
    "docs/multires/B1-GATE.json",
    "docs/multires/B2-EXACT-DROP-REPLAY.md",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atlas_analyzer_authority_inputs(repo_root: Path) -> tuple[Path, ...]:
    """Return the complete, deterministic analyzer source authority set."""

    root = repo_root.resolve()
    fixed = [root / relative for relative in ATLAS_ANALYZER_FIXED_INPUTS]
    rust = sorted((root / "crates/q2-lattice/src").rglob("*.rs"))
    inputs = tuple(sorted((*fixed, *rust)))
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Atlas analyzer authority input missing: "
            + ", ".join(str(path) for path in missing)
        )
    return inputs


def atlas_analyzer_authority_sha256(repo_root: Path) -> str:
    """Hash canonical relative paths and bytes for the shared closure."""

    root = repo_root.resolve()
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
        }
        for path in atlas_analyzer_authority_inputs(root)
    ]
    payload = json.dumps(
        records, ensure_ascii=True, separators=(",", ":"), sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ATLAS_ANALYZER_FIXED_INPUTS",
    "atlas_analyzer_authority_inputs",
    "atlas_analyzer_authority_sha256",
]
