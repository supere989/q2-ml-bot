"""Extension-backed qualification for the production multires provider.

Unlike the small adapter tests, this lane builds and loads the actual PyO3
extension.  A temporary Rust fixture builder reuses the crate's canonical
Atlas test constructors, then emits two complete Atlas/objective maps and four
Q2LAT002 snapshots per map.  No Python lattice or permissive runtime double is
used anywhere in these tests.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from harness.causal_protocol import CausalFlags, CausalTelemetry
from harness.client_env import Q2NetworkClientEnv
from harness.client_protocol import ClientTelemetry
from harness.multires_admission import (
    MultiresAdmissionError,
    SpatialProviderBinding,
    validate_spatial_provider_frame,
)
from harness.multires_query_source import OwnObservationQuerySource
from harness.protocol import (
    ActionDebugIndex,
    ML_ENTITY_EPOCH_SHIFT,
    Observation,
)
from harness.rust_multires_provider import (
    RustAtlasProviderFactory,
    RustAtlasSpatialProvider,
    RustMapArtifacts,
    RustSpatialProviderError,
)


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MANIFEST_SHA256 = hashlib.sha256(
    b"extension-backed-provider-qualification-runtime-v1"
).hexdigest()
OTHER_RUNTIME_MANIFEST_SHA256 = hashlib.sha256(
    b"unattested-provider-runtime"
).hexdigest()
CLIENT_COUNT = 4
CLIENT_LIFE_EPOCH = 3


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
    )
    if completed.returncode:
        pytest.fail(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-12000:]}"
        )


@pytest.fixture(scope="session")
def rust_extension():
    _run(
        ["cargo", "build", "-p", "q2-lattice", "--features", "python"],
        cwd=ROOT,
    )
    candidates = sorted((ROOT / "target" / "debug").glob("*q2_lattice_rs*.so"))
    if len(candidates) != 1:
        pytest.fail(f"expected one built q2_lattice_rs extension, found {candidates!r}")
    path = candidates[0].resolve()
    sys.modules.pop("q2_lattice_rs", None)
    spec = importlib.util.spec_from_file_location("q2_lattice_rs", path)
    if spec is None or spec.loader is None:
        pytest.fail(f"cannot load Rust extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FIXTURE_BUILDER = r'''
include!(r#"__ATLAS_SCHEMA_RS__"#);

fn write(path: &std::path::Path, payload: &[u8]) {
    std::fs::write(path, payload).unwrap();
}

fn canonical_json(value: &serde_json::Value) -> Vec<u8> {
    let mut payload = serde_json::to_vec(value).unwrap();
    payload.push(b'\n');
    payload
}

fn emit_map(root: &std::path::Path, name: &str, epoch: u64, bsp_bytes: &[u8]) {
    use q2_lattice_rs::atlas::{
        ArtifactManifest, AtlasLevel, AtlasLimits, AtlasRuntime, GridIndex,
        OBJECTIVE_MEDIA_TYPE, install_static_costs,
        install_static_hazard_clearances, sha256_hex,
    };
    use q2_lattice_rs::dynstate::{
        DynFence, DynLimits, DynState, encode_snapshot,
    };

    let directory = root.join(name);
    std::fs::create_dir_all(&directory).unwrap();
    let limits = AtlasLimits::default();
    let mut atlas = artifact(false);
    install_static_costs(&mut atlas.l1).unwrap();
    install_static_hazard_clearances(&mut atlas.l1).unwrap();
    let raw = atlas.encode_uncompressed(&limits).unwrap();
    let atlas_sha256 = sha256_hex(&raw);
    let bsp_sha256 = sha256_hex(bsp_bytes);

    let mut atlas_manifest = manifest(&raw, raw.len() as u64);
    atlas_manifest.bsp.canonical_map_id = name.to_owned();
    atlas_manifest.bsp.sha256 = bsp_sha256.clone();
    atlas_manifest.bsp.size_bytes = bsp_bytes.len() as u64;
    atlas_manifest.oracles = oracle_admissions(&atlas_manifest.bsp, true, true);
    atlas_manifest.artifacts.clear();
    atlas_manifest.artifacts.insert(
        format!("{name}.atlas.bin"),
        ArtifactManifest::from_uncompressed(
            "application/vnd.q2.atlas-v1",
            &raw,
            raw.len() as u64,
            atlas_manifest.counts.named_counts(),
        ),
    );

    let objectives = canonical_json(&serde_json::json!({
        "atlas_sha256": atlas_sha256,
        "bsp_sha256": bsp_sha256,
        "canonical_map_id": name,
        "objectives": [],
        "origin": atlas.origin.0,
        "schema": "q2-atlas-objectives-v1",
    }));
    atlas_manifest.artifacts.insert(
        format!("{name}.objectives.json"),
        ArtifactManifest::from_uncompressed(
            OBJECTIVE_MEDIA_TYPE,
            &objectives,
            objectives.len() as u64,
            std::collections::BTreeMap::from([("objectives".to_owned(), 0)]),
        ),
    );
    let atlas_manifest_bytes = atlas_manifest.canonical_json(&limits).unwrap();

    // This also proves that the generated fixture crosses the same strict
    // admission path used by the PyO3 AtlasRuntime constructor.
    AtlasRuntime::from_bytes(
        &atlas_manifest_bytes,
        &format!("{name}.atlas.bin"),
        &raw,
        &format!("{name}.objectives.json"),
        &objectives,
        bsp_bytes,
        name,
        epoch,
        &limits,
    ).unwrap();

    write(&directory.join(format!("{name}.bsp")), bsp_bytes);
    write(&directory.join(format!("{name}.atlas.bin")), &raw);
    write(
        &directory.join(format!("{name}.atlas.bin.zst")),
        &q2_lattice_rs::atlas::encode_zstd_envelope(&raw, &limits).unwrap(),
    );
    write(
        &directory.join(format!("{name}.atlas.manifest.json")),
        &atlas_manifest_bytes,
    );
    write(
        &directory.join(format!("{name}.objectives.json")),
        &objectives,
    );

    let fence = DynFence {
        atlas_sha256: <[u8; 32]>::try_from(
            (0..32)
                .map(|index| u8::from_str_radix(&atlas_sha256[index * 2..index * 2 + 2], 16).unwrap())
                .collect::<Vec<_>>()
        ).unwrap(),
        map_sha256: <[u8; 32]>::try_from(
            (0..32)
                .map(|index| u8::from_str_radix(&bsp_sha256[index * 2..index * 2 + 2], 16).unwrap())
                .collect::<Vec<_>>()
        ).unwrap(),
        origin: atlas.origin,
        map_epoch: epoch,
    };
    let dyn_limits = DynLimits::default();
    for client in 0..4_u32 {
        let state = DynState::new(fence, client, 4, 0, &dyn_limits).unwrap();
        let snapshot = encode_snapshot(&state, &dyn_limits).unwrap();
        write(
            &directory.join(format!("client{client}.q2lat002")),
            &snapshot,
        );
    }
    let point = atlas.origin.center(GridIndex::new(0, 0, -2), AtlasLevel::L1);
    let metadata = canonical_json(&serde_json::json!({
        "atlas_origin": atlas.origin.0,
        "atlas_sha256": atlas_sha256,
        "bsp_sha256": bsp_sha256,
        "map_epoch": epoch,
        "map_name": name,
        "world_position": point,
    }));
    write(&directory.join("fixture-metadata.json"), &metadata);
}

fn main() {
    let root = std::path::PathBuf::from(std::env::args_os().nth(1).unwrap());
    std::fs::create_dir_all(&root).unwrap();
    emit_map(&root, "qualmap0", 0, b"extension-provider-fixture-bsp-0");
    emit_map(&root, "qualmap1", 1, b"extension-provider-fixture-bsp-1");
}
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle_manifest(directory: Path, map_name: str) -> Path:
    # The production v3 bundle contract is an exact set, not a minimum set.
    # Only the BSP/Atlas/objectives are interpreted by this qualification;
    # the remaining members are deterministic, digest-bound valid stand-ins
    # for their independently qualified producers.
    standins = {
        f"{map_name}.json": b"{}\n",
        f"{map_name}.lattice.json": b"{}\n",
        f"{map_name}.routes.json": b"{}\n",
        f"{map_name}.analysis.manifest.json": b"{}\n",
        f"{map_name}.navigation.bin.zst": b"qualification-navigation-v1",
        f"{map_name}.visibility.bin.zst": b"qualification-visibility-v1",
        f"{map_name}.design-signature.json": b"{}\n",
    }
    for name, payload in standins.items():
        (directory / name).write_bytes(payload)
    file_names = tuple(
        f"{map_name}{suffix}"
        for suffix in (".bsp", ".json", ".lattice.json", ".routes.json")
    )
    analysis_names = tuple(
        f"{map_name}{suffix}"
        for suffix in (
            ".analysis.manifest.json",
            ".atlas.manifest.json",
            ".atlas.bin.zst",
            ".navigation.bin.zst",
            ".visibility.bin.zst",
            ".design-signature.json",
            ".objectives.json",
        )
    )
    files = {name: _sha256(directory / name) for name in file_names}
    analysis = {name: _sha256(directory / name) for name in analysis_names}
    document = {
        "analysis_file_sizes": {
            name: (directory / name).stat().st_size for name in analysis
        },
        "analysis_files": analysis,
        "bundle_version": 3,
        "file_sizes": {name: (directory / name).stat().st_size for name in files},
        "files": files,
        "name": map_name,
    }
    path = directory / f"{map_name}.bundle.json"
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def real_provider_artifacts(tmp_path_factory, rust_extension):
    root = tmp_path_factory.mktemp("rust-provider-extension")
    crate = root / "fixture-builder"
    source = crate / "src"
    source.mkdir(parents=True)
    cargo = f'''[package]
name = "q2-provider-qualification-fixture"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
q2-lattice = {{ path = {json.dumps(str(ROOT / "crates" / "q2-lattice"))} }}
serde_json = "1.0"
'''
    (crate / "Cargo.toml").write_text(cargo, encoding="utf-8")
    builder = _FIXTURE_BUILDER.replace(
        "__ATLAS_SCHEMA_RS__",
        str(ROOT / "crates" / "q2-lattice" / "tests" / "atlas_schema.rs"),
    )
    (source / "main.rs").write_text(builder, encoding="utf-8")
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(ROOT / "target" / "provider-qualification-fixture")
    output = root / "artifacts"
    _run(
        ["cargo", "run", "--quiet", "--manifest-path", str(crate / "Cargo.toml"), "--", str(output)],
        cwd=ROOT,
        env=env,
    )
    result = {}
    for name in ("qualmap0", "qualmap1"):
        directory = output / name
        metadata = json.loads((directory / "fixture-metadata.json").read_text())
        metadata["directory"] = directory
        metadata["bundle_manifest"] = _write_bundle_manifest(directory, name)
        result[name] = metadata
    return result


def _causal(frame: int, *, private_variant: bool = False) -> CausalTelemetry:
    flags = (
        CausalFlags.ECHO_VALID
        | CausalFlags.FACTS_COMPLETE
        | CausalFlags.TRANSITION_TRAINABLE
    )
    values = dict(
        tick=frame,
        client_life_epoch=CLIENT_LIFE_EPOCH,
        target_id=0,
        target_epoch=0,
        environmental_source_id=0,
        environmental_source_epoch=0,
        environmental_mod=0,
        environmental_damage=0,
        crouch_edge_id=0,
        crouch_edge_epoch=0,
        echo_tick=max(1, frame - 1),
        action_generation=2,
        hook_zone_id=0,
        hook_attempt_tick=0,
        hook_action_generation=0,
        flags=flags,
    )
    if private_variant:
        values.update(
            target_id=7,
            target_epoch=2,
            environmental_source_id=91,
            environmental_source_epoch=4,
            environmental_mod=22,
            environmental_damage=3,
            flags=(
                flags
                | CausalFlags.TARGET_VALID
                | CausalFlags.TARGET_HIT
                | CausalFlags.ENV_SOURCE_EVIDENCE
                | CausalFlags.ENV_DAMAGE
            ),
        )
    return CausalTelemetry(**values)


def _observation(
    position: list[float] | tuple[float, float, float],
    *,
    public_event: bool = False,
    visible_target: bool = False,
) -> Observation:
    entities = np.zeros((8, 9), dtype=np.float32)
    entity_debug = np.zeros((8, 4), dtype=np.uint32)
    count = 0
    if visible_target:
        entities[0, :3] = (64.0, 0.0, 0.0)
        entities[0, 6] = 100.0
        entities[0, 7] = 1.0
        entities[0, 8] = 0.75
        entity_debug[0, 0] = 7
        entity_debug[0, 3] = 2 << ML_ENTITY_EPOCH_SHIFT
        count = 1
    action_debug = np.zeros(15, dtype=np.float64)
    action_debug[ActionDebugIndex.TICK] = 1
    return Observation(
        tick=1,
        bot_slot=1,
        yaw=0.0,
        pitch=0.0,
        self_state=np.asarray(
            [*position, 0.0, 0.0, 0.0, 100.0, 0.0, 1.0, 10.0],
            dtype=np.float32,
        ),
        entities=entities,
        entity_count=count,
        rays=np.zeros((16, 4), dtype=np.float32),
        hook_zones=np.zeros((4, 8), dtype=np.float32),
        hook_zone_count=0,
        audio=np.zeros(5, dtype=np.float32),
        reward_damage_dealt=5.0 if public_event else 0.0,
        reward_damage_taken=4.0 if public_event else 0.0,
        reward_kill=0.0,
        reward_death=0.0,
        reward_item_pickup=0.0,
        reward_hook_traversal=0.0,
        reward_damage_taken_prox=0.0,
        reward_offense=0.0,
        reward_survival=0.0,
        rune_flags=np.zeros(5, dtype=np.float32),
        inbound_dmg_dir=np.zeros(3, dtype=np.float32),
        inbound_dmg_dist=-1.0,
        inbound_dmg_recency=0.0,
        actual_ducked=False,
        standing_blocked=False,
        water_vertical_mode=False,
        is_terminal=False,
        terminal_reason=0,
        self_debug=np.zeros(4, dtype=np.uint32),
        entity_debug=entity_debug,
        action_debug=action_debug,
    )


def _telemetry(
    artifact: dict,
    client_index: int,
    frame: int,
    *,
    public_event: bool = False,
    visible_target: bool = False,
    private_variant: bool = False,
) -> ClientTelemetry:
    observation = _observation(
        artifact["world_position"],
        public_event=public_event,
        visible_target=visible_target,
    )
    observation.tick = frame
    observation.bot_slot = client_index + 1
    observation.action_debug[ActionDebugIndex.TICK] = frame
    return ClientTelemetry(
        sequence=frame,
        client_slot=client_index + 1,
        server_frame=frame,
        client_id=f"client-{client_index}",
        map_name=str(artifact["map_name"]),
        observation=observation,
        causal=_causal(frame, private_variant=private_variant),
    )


def _provider(rust_extension, artifact: dict, client_index: int):
    directory = Path(artifact["directory"])
    objectives = directory / f"{artifact['map_name']}.objectives.json"
    source = OwnObservationQuerySource(
        objectives_path=objectives,
        expected_objectives_sha256=_sha256(objectives),
        expected_atlas_sha256=str(artifact["atlas_sha256"]),
        atlas_origin=tuple(artifact["atlas_origin"]),
        map_name=str(artifact["map_name"]),
        client_id=f"client-{client_index}",
    )
    return RustAtlasSpatialProvider.from_admitted_bundle(
        extension_module=rust_extension,
        bundle_manifest_path=Path(artifact["bundle_manifest"]),
        uncompressed_atlas_path=directory / f"{artifact['map_name']}.atlas.bin",
        dyn_snapshot_path=directory / f"client{client_index}.q2lat002",
        input_source=source,
        expected_atlas_sha256=str(artifact["atlas_sha256"]),
        runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
        map_epoch=int(artifact["map_epoch"]),
        rust_client_id=client_index,
        client_count=CLIENT_COUNT,
        environment_steps=0,
        atlas_origin=tuple(artifact["atlas_origin"]),
    )


def test_actual_extension_four_clients_ingest_query_and_isolate_dyn(
    rust_extension, real_provider_artifacts
):
    artifact = real_provider_artifacts["qualmap0"]
    providers = [_provider(rust_extension, artifact, index) for index in range(4)]
    for index, provider in enumerate(providers):
        # A fresh Q2LAT002 wrapper has no life/frame cursor.  The real Rust
        # runtime must reject a feature query until the provider commits the
        # first (possibly empty) old-step -> new-step ingest transaction.
        with pytest.raises(ValueError):
            provider.dyn_runtime.feature_block(
                list(artifact["world_position"]), 0.0, 0, index,
                CLIENT_LIFE_EPOCH, 0, 10, (0.0, 0.5, 0.25), None,
            )
        snapshot = bytes(provider.dyn_runtime.snapshot_bytes())
        assert hashlib.sha256(snapshot).hexdigest() == (
            provider.dyn_runtime.snapshot_sha256
        )
    baseline = [
        provider.sample(_telemetry(artifact, index, 10), episode_projection=False)
        for index, provider in enumerate(providers)
    ]
    for index, frame in enumerate(baseline):
        spatial = validate_spatial_provider_frame(
            frame,
            _telemetry(artifact, index, 10),
            expected_atlas_sha256=str(artifact["atlas_sha256"]),
            expected_runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
            expected_map_epoch=0,
            require_private_reward_evidence=True,
        )
        assert spatial.dyn.shape == (24,)
        assert spatial.recovery.shape == (16,)
        assert spatial.objectives.shape == (4, 15)
        evidence = frame.private_reward_evidence
        assert evidence is not None
        assert evidence.cost_to_safety_q8 == 0
        assert evidence.hazard_component_id == 0
        assert evidence.hazard_component_epoch == 0

    changed = providers[0].sample(
        _telemetry(artifact, 0, 11, public_event=True, visible_target=True),
        episode_projection=False,
    )
    unchanged = [
        providers[index].sample(
            _telemetry(artifact, index, 11), episode_projection=False
        )
        for index in range(1, 4)
    ]
    assert changed.spatial.dyn[1] > baseline[0].spatial.dyn[1]
    assert providers[0].dyn_runtime.accepted_event_count == 3
    assert providers[0].dyn_runtime.last_event_id == (11 << 3) | 3
    changed_snapshot = bytes(providers[0].dyn_runtime.snapshot_bytes())
    assert hashlib.sha256(changed_snapshot).hexdigest() == (
        providers[0].dyn_runtime.snapshot_sha256
    )
    assert all(
        np.array_equal(frame.spatial.dyn, baseline[index + 1].spatial.dyn)
        for index, frame in enumerate(unchanged)
    )


def test_concrete_factory_binds_artifacts_and_rotates_real_extension(
    rust_extension, real_provider_artifacts
):
    def configured(artifact):
        directory = Path(artifact["directory"])
        return RustMapArtifacts(
            bundle_manifest_path=Path(artifact["bundle_manifest"]),
            uncompressed_atlas_path=(
                directory / f"{artifact['map_name']}.atlas.bin"
            ),
            dyn_snapshot_path=directory / "client0.q2lat002",
            expected_atlas_sha256=str(artifact["atlas_sha256"]),
        )

    factory = RustAtlasProviderFactory(
        extension_module=rust_extension,
        artifacts_by_map={
            name: configured(artifact)
            for name, artifact in real_provider_artifacts.items()
        },
        runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
        bound_client_id="client-0",
        rust_client_id=0,
        client_count=CLIENT_COUNT,
    )
    first_artifact = real_provider_artifacts["qualmap0"]
    first = factory.create(_telemetry(first_artifact, 0, 10), map_epoch=0)
    assert isinstance(first, SpatialProviderBinding)
    assert first.atlas_sha256 == first_artifact["atlas_sha256"]
    first.provider.sample(
        _telemetry(first_artifact, 0, 10), episode_projection=False
    )
    with pytest.raises(RustSpatialProviderError, match="repeated or regressed"):
        factory.create(_telemetry(first_artifact, 0, 11), map_epoch=0)

    unknown = replace(
        _telemetry(first_artifact, 0, 11), map_name="not-admitted"
    )
    with pytest.raises(RustSpatialProviderError, match="no admitted"):
        factory.create(unknown, map_epoch=1)

    second_artifact = real_provider_artifacts["qualmap1"]
    second = factory.create(_telemetry(second_artifact, 0, 10), map_epoch=1)
    assert second.atlas_sha256 == second_artifact["atlas_sha256"]
    second.provider.sample(
        _telemetry(second_artifact, 0, 10), episode_projection=False
    )


def test_actual_extension_checkpoint_restore_is_bit_exact(
    rust_extension, real_provider_artifacts
):
    artifact = real_provider_artifacts["qualmap0"]
    provider = _provider(rust_extension, artifact, 0)
    provider.sample(_telemetry(artifact, 0, 10), episode_projection=False)
    provider.sample(
        _telemetry(artifact, 0, 11, public_event=True, visible_target=True),
        episode_projection=False,
    )
    checkpoint, checkpoint_sha256 = provider.dyn_checkpoint()
    original = provider.dyn_runtime
    restored = rust_extension.DynRuntime.from_checkpoint(
        checkpoint,
        checkpoint_sha256,
        str(artifact["atlas_sha256"]),
        str(artifact["bsp_sha256"]),
        list(artifact["atlas_origin"]),
        0,
        0,
        CLIENT_COUNT,
        2,
        CLIENT_LIFE_EPOCH,
        11,
    )
    for runtime in (original, restored):
        snapshot = bytes(runtime.snapshot_bytes())
        assert hashlib.sha256(snapshot).hexdigest() == runtime.snapshot_sha256
    for runtime in (original, restored):
        runtime.ingest_events(
            str(artifact["atlas_sha256"]),
            str(artifact["bsp_sha256"]),
            list(artifact["atlas_origin"]),
            0,
            0,
            CLIENT_LIFE_EPOCH,
            12,
            2,
            3,
            [],
        )
    args = (
        list(artifact["world_position"]), 0.0, 0, 0,
        CLIENT_LIFE_EPOCH, 3, 12, (0.0, 0.5, 0.25), None,
    )
    assert np.array_equal(original.feature_block(*args), restored.feature_block(*args))
    assert bytes(original.checkpoint_bytes()) == bytes(restored.checkpoint_bytes())
    assert original.checkpoint_sha256 == restored.checkpoint_sha256


def test_private_causal_flags_cannot_change_public_rust_features(
    rust_extension, real_provider_artifacts
):
    artifact = real_provider_artifacts["qualmap0"]
    ordinary = _provider(rust_extension, artifact, 0)
    adversarial = _provider(rust_extension, artifact, 0)
    public = _telemetry(artifact, 0, 10, public_event=True, visible_target=True)
    private = replace(public, causal=_causal(10, private_variant=True))
    left = ordinary.sample(public, episode_projection=False)
    right = adversarial.sample(private, episode_projection=False)
    assert np.array_equal(left.spatial.to_vector(), right.spatial.to_vector())
    assert ordinary.dyn_checkpoint() == adversarial.dyn_checkpoint()


def test_actual_extension_projection_and_all_identity_fences(
    rust_extension, real_provider_artifacts
):
    artifact = real_provider_artifacts["qualmap0"]
    private_provider = _provider(rust_extension, artifact, 0)
    public_provider = _provider(rust_extension, artifact, 0)
    telemetry = _telemetry(artifact, 0, 10)
    private = private_provider.sample(telemetry, episode_projection=False)
    public = public_provider.sample(telemetry, episode_projection=True)
    assert public.private_reward_evidence is None
    assert private.private_reward_evidence is not None
    assert np.array_equal(public.spatial.to_vector(), private.spatial.to_vector())
    validate_spatial_provider_frame(
        public,
        telemetry,
        expected_atlas_sha256=str(artifact["atlas_sha256"]),
        expected_runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
        expected_map_epoch=0,
        require_private_reward_evidence=False,
    )
    with pytest.raises(MultiresAdmissionError, match="runtime_manifest_sha256"):
        validate_spatial_provider_frame(
            public,
            telemetry,
            expected_atlas_sha256=str(artifact["atlas_sha256"]),
            expected_runtime_manifest_sha256=OTHER_RUNTIME_MANIFEST_SHA256,
            expected_map_epoch=0,
            require_private_reward_evidence=False,
        )

    directory = Path(artifact["directory"])
    common = dict(
        extension_module=rust_extension,
        bundle_manifest_path=Path(artifact["bundle_manifest"]),
        uncompressed_atlas_path=directory / "qualmap0.atlas.bin",
        dyn_snapshot_path=directory / "client0.q2lat002",
        input_source=OwnObservationQuerySource(
            objectives_path=directory / "qualmap0.objectives.json",
            expected_objectives_sha256=_sha256(directory / "qualmap0.objectives.json"),
            expected_atlas_sha256=str(artifact["atlas_sha256"]),
            atlas_origin=tuple(artifact["atlas_origin"]),
            map_name="qualmap0",
            client_id="client-0",
        ),
        expected_atlas_sha256=str(artifact["atlas_sha256"]),
        runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
        map_epoch=0,
        client_count=CLIENT_COUNT,
        environment_steps=0,
    )
    with pytest.raises((ValueError, RustSpatialProviderError)):
        RustAtlasSpatialProvider.from_admitted_bundle(
            **common,
            rust_client_id=1,
            atlas_origin=tuple(artifact["atlas_origin"]),
        )
    wrong_origin = tuple(artifact["atlas_origin"][:2]) + (
        int(artifact["atlas_origin"][2]) + 256,
    )
    with pytest.raises((ValueError, RustSpatialProviderError)):
        RustAtlasSpatialProvider.from_admitted_bundle(
            **common,
            rust_client_id=0,
            atlas_origin=wrong_origin,
        )


def test_real_provider_factory_rotates_to_fresh_map_epoch(
    rust_extension, real_provider_artifacts
):
    class Factory:
        def __init__(self):
            self.created = []

        def create(self, telemetry, *, map_epoch):
            artifact = real_provider_artifacts[telemetry.map_name]
            assert map_epoch == artifact["map_epoch"]
            provider = _provider(rust_extension, artifact, 0)
            self.created.append(provider)
            return SpatialProviderBinding(
                provider=provider,
                atlas_sha256=str(artifact["atlas_sha256"]),
            )

    factory = Factory()
    env = Q2NetworkClientEnv(
        server="127.0.0.1:27910",
        telemetry_server="127.0.0.1:27911",
        telemetry_token="qualification",
        client_binary="/nonexistent/yquake2",
        client_root="/nonexistent/q2",
        client_id="client-0",
        multires_spatial_provider_factory=factory,
        expected_runtime_manifest_sha256=RUNTIME_MANIFEST_SHA256,
    )
    first_artifact = real_provider_artifacts["qualmap0"]
    second_artifact = real_provider_artifacts["qualmap1"]
    first = _telemetry(first_artifact, 0, 10)
    second = _telemetry(second_artifact, 0, 10)
    initial, _ = env.initial_result(first, vector=True)
    rotated, reward, *_ = env.transition_result(second, vector=True)
    assert initial.shape == rotated.shape == (298,)
    assert reward == 0.0
    assert len(factory.created) == 2
    with pytest.raises(RustSpatialProviderError, match="closed"):
        factory.created[0].sample(first, episode_projection=True)
    env.close()
    with pytest.raises(RustSpatialProviderError, match="closed"):
        factory.created[1].sample(second, episode_projection=True)


def _dyn_runtime_state(runtime) -> tuple:
    """Exact persisted state/cursors used by atomic-retry assertions."""
    return (
        bytes(runtime.snapshot_bytes()),
        runtime.snapshot_sha256,
        bytes(runtime.checkpoint_bytes()),
        runtime.checkpoint_sha256,
        runtime.environment_steps,
        runtime.client_life_epoch,
        runtime.server_frame,
        runtime.last_event_id,
        runtime.accepted_event_count,
    )


def test_actual_extension_commit_frame_rolls_back_after_ingest_failure_and_retries(
    rust_extension, real_provider_artifacts
):
    """A query failure after candidate ingest cannot poison the exact retry."""
    artifact = real_provider_artifacts["qualmap0"]
    failed = _provider(rust_extension, artifact, 0).dyn_runtime
    clean = _provider(rust_extension, artifact, 0).dyn_runtime
    common = (
        str(artifact["atlas_sha256"]),
        str(artifact["bsp_sha256"]),
        list(artifact["atlas_origin"]),
        0,
        0,
        CLIENT_LIFE_EPOCH,
    )
    position = list(artifact["world_position"])

    # Establish an identical checkpointable runtime cursor before the
    # adversarial transaction.  Frame 10 then ingests a valid event on the
    # candidate before its non-finite feature query must reject.
    for runtime in (failed, clean):
        runtime.commit_frame(
            *common, 9, 0, 1, [], position, 0.0, (0.0, 0.5, 0.25), None
        )
    before = _dyn_runtime_state(failed)
    event = [((10 << 3) | 1, "engagement", position)]
    with pytest.raises(ValueError, match="non-finite"):
        failed.commit_frame(
            *common,
            10,
            1,
            2,
            event,
            [float("nan"), position[1], position[2]],
            0.0,
            (0.0, 0.5, 0.25),
            None,
        )
    assert _dyn_runtime_state(failed) == before

    retried = failed.commit_frame(
        *common, 10, 1, 2, event, position, 0.0, (0.0, 0.5, 0.25), None
    )
    reference = clean.commit_frame(
        *common, 10, 1, 2, event, position, 0.0, (0.0, 0.5, 0.25), None
    )
    assert retried[0] == reference[0]
    assert np.array_equal(retried[1], reference[1])
    assert retried[2] == reference[2]
    assert _dyn_runtime_state(failed) == _dyn_runtime_state(clean)


def test_provider_rolls_back_staged_source_and_retries_same_frame_identically(
    rust_extension, real_provider_artifacts
):
    """Atlas rejection after source staging leaves both components untouched."""
    artifact = real_provider_artifacts["qualmap0"]
    failed = _provider(rust_extension, artifact, 0)
    clean = _provider(rust_extension, artifact, 0)
    baseline = _telemetry(artifact, 0, 9)
    failed.sample(baseline, episode_projection=False)
    clean.sample(baseline, episode_projection=False)

    class FailGuideOnce:
        def __init__(self, inner):
            self.inner = inner
            self.fail = True

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def guide_features(self, *args, **kwargs):
            if self.fail:
                self.fail = False
                raise RuntimeError("qualification failure after source stage")
            return self.inner.guide_features(*args, **kwargs)

    failed.atlas_runtime = FailGuideOnce(failed.atlas_runtime)
    before_dyn = _dyn_runtime_state(failed.dyn_runtime)
    before_steps = failed.input_source.environment_steps
    telemetry = _telemetry(
        artifact, 0, 10, public_event=True, visible_target=True
    )
    with pytest.raises(RuntimeError, match="failure after source stage"):
        failed.sample(telemetry, episode_projection=False)
    assert failed.input_source.environment_steps == before_steps
    assert _dyn_runtime_state(failed.dyn_runtime) == before_dyn

    retried = failed.sample(telemetry, episode_projection=False)
    reference = clean.sample(telemetry, episode_projection=False)
    assert np.array_equal(
        retried.spatial.to_vector(), reference.spatial.to_vector()
    )
    assert retried.private_reward_evidence == reference.private_reward_evidence
    assert failed.input_source.environment_steps == before_steps + 1
    assert _dyn_runtime_state(failed.dyn_runtime) == _dyn_runtime_state(
        clean.dyn_runtime
    )


def test_episode_projection_is_an_empty_event_resync_barrier_without_replay(
    rust_extension, real_provider_artifacts
):
    """Reset projection consumes public edges but never persists their events."""
    artifact = real_provider_artifacts["qualmap0"]
    provider = _provider(rust_extension, artifact, 0)

    projection = provider.sample(
        _telemetry(
            artifact, 0, 10, public_event=True, visible_target=True
        ),
        episode_projection=True,
    )
    assert projection.private_reward_evidence is None
    assert provider.dyn_runtime.environment_steps == 1
    assert provider.dyn_runtime.server_frame == 10
    assert provider.dyn_runtime.last_event_id == 0
    assert provider.dyn_runtime.accepted_event_count == 0

    # The same visible identity remains in the same L2 cell.  Its suppressed
    # opportunity edge is part of the frame-10 resync barrier, not a deferred
    # event that may appear in the first trainable transition.
    ordinary = provider.sample(
        _telemetry(artifact, 0, 11, visible_target=True),
        episode_projection=False,
    )
    assert ordinary.private_reward_evidence is not None
    assert provider.dyn_runtime.environment_steps == 2
    assert provider.dyn_runtime.server_frame == 11
    assert provider.dyn_runtime.last_event_id == 0
    assert provider.dyn_runtime.accepted_event_count == 0

    # Fresh factual damage on the next ordinary frame is accepted normally;
    # only engagement/threat are new because opportunity was consumed by the
    # barrier rather than replayed.
    provider.sample(
        _telemetry(
            artifact, 0, 12, public_event=True, visible_target=True
        ),
        episode_projection=False,
    )
    assert provider.dyn_runtime.environment_steps == 3
    assert provider.dyn_runtime.accepted_event_count == 2
    assert provider.dyn_runtime.last_event_id == (12 << 3) | 2

    # Held accepted-fire and held-death facts have the same barrier contract.
    # Use independent real runtimes so death's required thermal retirement
    # cannot obscure the opportunity-edge assertion above.
    fire_provider = _provider(rust_extension, artifact, 1)
    projected_fire = _telemetry(artifact, 1, 10)
    projected_fire.observation.action_debug[ActionDebugIndex.ACCEPTED] = 1
    projected_fire.observation.action_debug[ActionDebugIndex.FIRE] = 1
    fire_provider.sample(projected_fire, episode_projection=True)
    held_fire = _telemetry(artifact, 1, 11)
    held_fire.observation.action_debug[ActionDebugIndex.ACCEPTED] = 1
    held_fire.observation.action_debug[ActionDebugIndex.FIRE] = 1
    fire_provider.sample(held_fire, episode_projection=False)
    assert fire_provider.dyn_runtime.environment_steps == 2
    assert fire_provider.dyn_runtime.last_event_id == 0
    assert fire_provider.dyn_runtime.accepted_event_count == 0

    death_provider = _provider(rust_extension, artifact, 2)
    projected_death = _telemetry(artifact, 2, 10)
    projected_death.observation.self_state[6] = 0.0
    projected_death.observation.reward_death = 1.0
    death_provider.sample(projected_death, episode_projection=True)
    held_death = _telemetry(artifact, 2, 11)
    held_death.observation.self_state[6] = 0.0
    held_death.observation.reward_death = 1.0
    death_provider.sample(held_death, episode_projection=False)
    assert death_provider.dyn_runtime.environment_steps == 2
    assert death_provider.dyn_runtime.last_event_id == 0
    assert death_provider.dyn_runtime.accepted_event_count == 0
