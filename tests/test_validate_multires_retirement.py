import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

try:
    import torch
except ImportError:  # production qualification requires it; unit host may not
    torch = None

from harness.multires_contract import (
    ACTION_DIM,
    FEATURE_SCHEMA_SHA256,
    OBS_DIM,
    POLICY_GENERATION,
    POSTURE_CLASSES,
)
from harness.multires_lineage import save_attested_checkpoint
from harness.atlas_catalog import (
    ATLAS_MAP_SPEC_SCHEMA,
    author_catalog,
    canonical_bytes as catalog_canonical_bytes,
)
from harness.multires_runtime import (
    B4_ACTION_MAGIC,
    B4_CAUSAL_MAGIC,
    B4_CAUSAL_PACKET_BYTES,
    B4_CAUSAL_VERSION,
    B4_CLIENT_WIRE_VERSION,
    B4_OBSERVATION_MAGIC,
    B4_PROTOCOL_GENERATION,
    B4_ROLLOUT_SCHEMA,
    B4_TEACHER_VERSION,
)
from harness.multires_training_config import MultiresTrainingConfiguration
from tools.validate_multires_retirement import (
    COLD_START_SCHEMA,
    PRE_B6_COLD_START_SCHEMA,
    PRE_B6_REPORT_SCHEMA,
    RetirementValidationError,
    validate_retirement,
)
import tools.validate_multires_retirement as retirement_module
import train.multires_one_run as one_run_module

if torch is not None:
    from models.multires_policy import MultiresQ2BotPolicy


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/multires/M4-RUNTIME-RETIREMENT.json"
TOOL = ROOT / "tools/validate_multires_retirement.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _catalog_fixture(runtime: Path, bundle: Path):
    map_name = "retirement_fixture"
    bsp = runtime / f"{map_name}.bsp"
    atlas = runtime / f"{map_name}.atlas.bin"
    objectives = runtime / f"{map_name}.objectives.json"
    atlas_manifest = runtime / f"{map_name}.atlas.manifest.json"
    extension_path = runtime / "q2_lattice_rs.so"
    bsp.write_bytes(b"IBSP-retirement-fixture")
    atlas.write_bytes(b"ATLAS-retirement-fixture")
    objectives.write_text(json.dumps({
        "map_name": map_name,
        "atlas_sha256": _sha(atlas),
    }, sort_keys=True), encoding="utf-8")
    atlas_manifest.write_text(json.dumps({
        "grid": {"origin": [0, 0, 0]},
        "bsp": {
            "canonical_map_id": map_name,
            "sha256": _sha(bsp),
            "size_bytes": bsp.stat().st_size,
        },
        "artifacts": {f"{map_name}.atlas.bin": {
            "sha256_uncompressed": _sha(atlas),
            "uncompressed_size": atlas.stat().st_size,
        }},
    }, sort_keys=True), encoding="utf-8")
    _json(bundle, {
        "bundle_version": 3,
        "artifact_state": "admitted",
        "name": map_name,
        "files": {f"{map_name}.bsp": _sha(bsp)},
        "analysis_files": {
            f"{map_name}.atlas.manifest.json": _sha(atlas_manifest),
            f"{map_name}.objectives.json": _sha(objectives),
        },
    })
    extension_path.write_bytes(b"test-only-catalog-extension-authority")

    class Runtime:
        @staticmethod
        def empty(atlas_sha, map_sha, origin, epoch, client, count, steps):
            return SimpleNamespace(
                atlas_sha256=atlas_sha,
                map_sha256=map_sha,
                origin=tuple(origin),
                map_epoch=epoch,
                client_id=client,
                client_count=count,
                environment_steps=steps,
                client_life_epoch=0,
                server_frame=0,
                last_event_id=0,
                accepted_event_count=0,
                cell_count=0,
            )

    extension = SimpleNamespace(
        __file__=str(extension_path.resolve()), DynRuntime=Runtime
    )
    catalog = runtime / "atlas-catalog.json"
    document = author_catalog(
        [{
            "schema": ATLAS_MAP_SPEC_SCHEMA,
            "map_name": map_name,
            "bsp": str(bsp.resolve()),
            "atlas": str(atlas.resolve()),
            "atlas_manifest": str(atlas_manifest.resolve()),
            "bundle_manifest": str(bundle.resolve()),
            "objectives": str(objectives.resolve()),
        }],
        catalog_path=catalog,
        rust_extension_path=extension_path,
        extension_module=extension,
    )
    catalog.write_bytes(catalog_canonical_bytes(document) + b"\n")
    return catalog, document["atlas_catalog_sha256"], extension


def _fixture(tmp_path: Path) -> dict:
    operational = tmp_path / "operational"
    runtime = operational / "runtime"
    run = operational / "runs/public_network_multires_atlas_fresh_v1"
    roots = {
        "current_run_root": run,
        "checkpoint_root": run / "checkpoints",
        "tensorboard_root": run / "tensorboard",
        "rollout_root": run / "evidence" / "rollouts",
        "update_root": run / "evidence" / "updates",
        "season_report_root": run / "season",
    }
    for value in (runtime, *roots.values()):
        value.mkdir(parents=True, exist_ok=True)

    runtime_digest = hashlib.sha256(b"sealed-runtime").hexdigest()
    optimizer_configuration = {
        "class": "torch.optim.Adam", "learning_rate": 1e-5, "kwargs": {}
    }
    training_configuration = MultiresTrainingConfiguration.create(
        reward={"damage_reward": 0.003},
        guide_dropout={"global_probability": 0.1},
        ppo={"clip_coef": 0.2},
    )
    training = runtime / "training.json"
    training.write_text(training_configuration.to_json(), encoding="utf-8")
    bundle = runtime / "map.bundle.json"
    catalog, atlas_catalog_digest, extension_module = _catalog_fixture(
        runtime, bundle
    )
    catalog_document = json.loads(catalog.read_text(encoding="utf-8"))
    atlas_digest = catalog_document["maps"][0]["atlas_sha256"]
    checkpoint = roots["checkpoint_root"] / "fresh-step-zero.pt"
    if torch is not None:
        torch.manual_seed(23)
        policy = MultiresQ2BotPolicy()
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=optimizer_configuration["learning_rate"]
        )
        manifest = save_attested_checkpoint(
            checkpoint,
            policy,
            atlas_catalog_sha256=atlas_catalog_digest,
            runtime_manifest_sha256=runtime_digest,
            training_config=training_configuration,
            initialization="random",
            training_step=0,
            optimizer=optimizer,
        )
    else:
        with zipfile.ZipFile(checkpoint, "w") as archive:
            archive.writestr("unavailable/data.pkl", b"torch unavailable fixture")
        manifest = type("Manifest", (), {
            "checkpoint_format": "q2-multires-attested-checkpoint-v2",
            "policy_generation": "multires-atlas-policy-v1",
            "architecture": "models.multires_policy.MultiresQ2BotPolicy",
            "initialization": "random", "training_step": 0,
            "observation_dim": 298, "action_dim": 8, "posture_classes": 3,
        })()
    checkpoint_attestation = runtime / "checkpoint-attestation.json"
    checkpoint_manifest = {
        "schema": "q2-multires-cold-checkpoint-v1",
        "checkpoint_sha256": _sha(checkpoint),
        "checkpoint_format": manifest.checkpoint_format,
        "policy_generation": manifest.policy_generation,
        "architecture": manifest.architecture,
        "initialization": manifest.initialization,
        "training_step": manifest.training_step,
        "observation_dim": manifest.observation_dim,
        "action_dim": manifest.action_dim,
        "posture_classes": manifest.posture_classes,
        "optimizer_state": "fresh-empty",
        "normalization_state": "fresh-empty",
        "atlas_catalog_sha256": atlas_catalog_digest,
    }
    _json(checkpoint_attestation, checkpoint_manifest)
    evidence = runtime / "runtime-evidence.json"
    _json(evidence, {
        "policy_generation": POLICY_GENERATION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "posture_classes": POSTURE_CLASSES,
        "protocol_generation": B4_PROTOCOL_GENERATION,
        "observation_magic": B4_OBSERVATION_MAGIC,
        "action_magic": B4_ACTION_MAGIC,
        "client_wire_version": B4_CLIENT_WIRE_VERSION,
        "teacher_version": B4_TEACHER_VERSION,
        "rollout_schema": B4_ROLLOUT_SCHEMA,
        "atlas_sha256": atlas_digest,
        "public_teacher_packing_separate": True,
        "public_teacher_field_violations": 0,
        "recovery_width": 16,
        "guide_width": 60,
        "causal_magic": B4_CAUSAL_MAGIC,
        "causal_version": B4_CAUSAL_VERSION,
        "causal_packet_bytes": B4_CAUSAL_PACKET_BYTES,
        "runtime_manifest_sha256": runtime_digest,
    })
    dyn = runtime / "client-00.q2lat"
    dyn.write_bytes(b"Q2LAT002" + b"fresh-dyn-fixture")
    selector = runtime / "service-selector.sh"
    selector.write_text(
        "#!/bin/sh\nexec python3 -m train.multires_service\n", encoding="utf-8"
    )

    training_runtime = runtime / "multires-training-runtime.json"
    _json(training_runtime, {
        "schema": "q2-multires-primary-training-runtime-v1",
        "proof_module": "train.multires_one_run",
        "trainer_module": "train.multires_primary",
        "atlas_catalog": {"path": str(catalog), "sha256": _sha(catalog)},
        "atlas_catalog_sha256": atlas_catalog_digest,
        "checkpoint": {
            "mode": "fresh-step-zero", "path": str(checkpoint),
            "sha256": _sha(checkpoint),
            "lineage_root_sha256": getattr(
                manifest, "lineage_root_sha256", "cd" * 32
            ),
        },
    })
    integration_envelope = runtime / "integration-envelope.json"
    _json(integration_envelope, {
        "schema": "multires-integration-evidence-v1", "evidence": {}
    })
    integration_report = runtime / "integration-report.json"
    _json(integration_report, {
        "schema": "multires-integration-report-v1", "overall": "pass"
    })

    manifest_digest = _sha(MANIFEST)
    cold = runtime / "cold-start.json"
    document = {
        "schema": COLD_START_SCHEMA,
        "retirement_manifest_sha256": manifest_digest,
        "runtime_manifest_sha256": runtime_digest,
        "atlas_catalog_sha256": atlas_catalog_digest,
        "optimizer": optimizer_configuration,
        "selectors": {
            "service_module": "train.multires_service",
            "proof_module": "train.multires_one_run",
            "trainer_module": "train.multires_primary",
            "client_builder": "harness.client_batch.build_network_client_batch",
            "collector": "harness.multires_collector.MultiresSynchronousCollector",
            "policy_class": "models.multires_policy.MultiresQ2BotPolicy",
            "provider_class": (
                "harness.rust_multires_provider.RustAtlasSpatialProvider"
            ),
            "rust_dyn_call": "q2_lattice_rs.DynRuntime.commit_frame",
        },
        "lineage": {
            "run_tag": "public_network_multires_atlas_fresh_v1",
            "initialization": "random",
            "training_step": 0,
            "optimizer_state": "fresh-empty",
            "normalization_state": "fresh-empty",
            **{name: str(path) for name, path in roots.items()},
        },
        "inputs": {
            "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
            "checkpoint_attestation": {
                "path": str(checkpoint_attestation),
                "sha256": _sha(checkpoint_attestation),
            },
            "runtime_evidence": {"path": str(evidence), "sha256": _sha(evidence)},
            "training_manifest": {
                "path": str(training), "sha256": _sha(training)
            },
            "bundle_manifest": {"path": str(bundle), "sha256": _sha(bundle)},
            "atlas_catalog": {"path": str(catalog), "sha256": _sha(catalog)},
            "dyn_snapshots": [{"path": str(dyn), "sha256": _sha(dyn)}],
            "training_runtime": {
                "path": str(training_runtime), "sha256": _sha(training_runtime),
            },
            "trainer_checkpoint": {
                "path": str(checkpoint), "sha256": _sha(checkpoint),
            },
            "trainer_current_season": None,
            "integration_envelope": {
                "path": str(integration_envelope),
                "sha256": _sha(integration_envelope),
            },
            "integration_report": {
                "path": str(integration_report),
                "sha256": _sha(integration_report),
            },
            "integration_bot_source": {
                "commit": "a" * 40, "tree": "b" * 40,
            },
        },
    }
    _json(cold, document)
    return {
        "operational": operational,
        "runtime": runtime,
        "selector": selector,
        "cold": cold,
        "cold_document": document,
        "checkpoint": checkpoint,
        "checkpoint_attestation": checkpoint_attestation,
        "checkpoint_manifest": checkpoint_manifest,
        "evidence": evidence,
        "training": training,
        "bundle": bundle,
        "catalog": catalog,
        "atlas_catalog_digest": atlas_catalog_digest,
        "extension_module": extension_module,
        "training_runtime": training_runtime,
        "integration_envelope": integration_envelope,
        "integration_report": integration_report,
        "dyn": dyn,
        "manifest": MANIFEST,
        "manifest_digest": manifest_digest,
    }


def _validate(fixture: dict, **overrides):
    bypass_checkpoint = overrides.pop("_bypass_checkpoint", torch is None)
    arguments = {
        "manifest_path": fixture["manifest"],
        "expected_manifest_sha256": fixture["manifest_digest"],
        "cold_start_path": fixture["cold"],
        "operational_roots": [fixture["operational"]],
        "service_selector_files": [fixture["selector"]],
    }
    arguments.update(overrides)
    original_checkpoint_loader = retirement_module._load_checkpoint
    original_extension_loader = one_run_module._load_extension
    if bypass_checkpoint:
        # Preserve coverage of all non-checkpoint retirement predicates on a
        # minimal unit host. Dedicated real-envelope tests above/below are skipped
        # rather than substituting this bypass as checkpoint evidence.
        retirement_module._load_checkpoint = lambda *args, **kwargs: None
    one_run_module._load_extension = lambda _path: fixture["extension_module"]
    try:
        return validate_retirement(**arguments)
    finally:
        retirement_module._load_checkpoint = original_checkpoint_loader
        one_run_module._load_extension = original_extension_loader


def _refresh(fixture: dict, key: str) -> None:
    record_key = {
        "checkpoint": "checkpoint",
        "checkpoint_attestation": "checkpoint_attestation",
        "evidence": "runtime_evidence",
        "bundle": "bundle_manifest",
        "dyn": "dyn_snapshots",
    }[key]
    path = fixture[key]
    if key == "dyn":
        fixture["cold_document"]["inputs"][record_key][0]["sha256"] = _sha(path)
    else:
        fixture["cold_document"]["inputs"][record_key]["sha256"] = _sha(path)
    _json(fixture["cold"], fixture["cold_document"])


def _refresh_checkpoint(fixture: dict) -> None:
    fixture["checkpoint_manifest"]["checkpoint_sha256"] = _sha(
        fixture["checkpoint"]
    )
    _json(fixture["checkpoint_attestation"], fixture["checkpoint_manifest"])
    _refresh(fixture, "checkpoint")
    _refresh(fixture, "checkpoint_attestation")
    fixture["cold_document"]["inputs"]["trainer_checkpoint"]["sha256"] = (
        _sha(fixture["checkpoint"])
    )
    training_runtime = json.loads(fixture["training_runtime"].read_text())
    training_runtime["checkpoint"]["sha256"] = _sha(fixture["checkpoint"])
    _json(fixture["training_runtime"], training_runtime)
    fixture["cold_document"]["inputs"]["training_runtime"]["sha256"] = _sha(
        fixture["training_runtime"]
    )
    _json(fixture["cold"], fixture["cold_document"])


@pytest.mark.skipif(torch is None, reason="real weights-only loader requires PyTorch")
def test_fresh_gate_is_canonical_read_only_and_cli_executable(tmp_path):
    fixture = _fixture(tmp_path)
    before = {
        path.relative_to(fixture["operational"]): _sha(path)
        for path in fixture["operational"].rglob("*")
        if path.is_file()
    }
    report = _validate(fixture)
    assert report["status"] == "pass"
    assert report["manifest_sha256"] == fixture["manifest_digest"]
    assert report["read_only"] is True
    after = {
        path.relative_to(fixture["operational"]): _sha(path)
        for path in fixture["operational"].rglob("*")
        if path.is_file()
    }
    assert after == before

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--manifest", str(MANIFEST),
            "--expected-manifest-sha256", fixture["manifest_digest"],
            "--cold-start", str(fixture["cold"]),
            "--operational-root", str(fixture["operational"]),
            "--service-selector", str(fixture["selector"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "pass"


def test_manifest_digest_canonicality_and_placeholder_are_fail_closed(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(RetirementValidationError, match="SHA-256 differs"):
        _validate(fixture, expected_manifest_sha256=hashlib.sha256(b"wrong").hexdigest())

    pretty = tmp_path / "pretty-retirement.json"
    pretty.write_text(
        json.dumps(json.loads(MANIFEST.read_text()), indent=2), encoding="utf-8"
    )
    with pytest.raises(RetirementValidationError, match="not canonical"):
        _validate(
            fixture,
            manifest_path=pretty,
            expected_manifest_sha256=_sha(pretty),
        )

    fixture["cold_document"]["lineage"]["run_tag"] = "${RUN_TAG}"
    _json(fixture["cold"], fixture["cold_document"])
    with pytest.raises(RetirementValidationError, match="placeholder"):
        _validate(fixture)


def test_explicit_pre_b6_mode_has_distinct_schema_and_cannot_enter_service_mode(tmp_path):
    fixture = _fixture(tmp_path)
    document = fixture["cold_document"]
    document["schema"] = PRE_B6_COLD_START_SCHEMA
    document["inputs"] = {
        name: document["inputs"][name]
        for name in (
            "checkpoint", "checkpoint_attestation", "runtime_evidence",
            "training_manifest", "bundle_manifest", "atlas_catalog",
            "dyn_snapshots",
        )
    }
    _json(fixture["cold"], document)
    report = _validate(fixture, mode="pre-b6")
    assert report["schema"] == PRE_B6_REPORT_SCHEMA
    assert report["mode"] == "pre-b6"
    assert report["read_only"] is True

    with pytest.raises(RetirementValidationError, match="schema"):
        _validate(fixture)

    document["schema"] = COLD_START_SCHEMA
    _json(fixture["cold"], document)
    with pytest.raises(RetirementValidationError, match="schema"):
        _validate(fixture, mode="pre-b6")


@pytest.mark.parametrize(
    "field,value,label",
    [
        ("client_wire_version", 4, "B4/QM3C"),
        ("teacher_version", 2, "B4/QM3C"),
        ("rollout_schema", "ppo-telemetry-v8", "B4/QM3C"),
        ("causal_magic", 0x514D4C50, "B4/QM3C"),
    ],
)
def test_legacy_protocol_inputs_fail(field, value, label, tmp_path):
    fixture = _fixture(tmp_path)
    evidence = json.loads(fixture["evidence"].read_text())
    evidence[field] = value
    _json(fixture["evidence"], evidence)
    _refresh(fixture, "evidence")
    with pytest.raises(RetirementValidationError, match=label):
        _validate(fixture)


def test_legacy_dyn_bundle_and_optimizer_sidecar_fail(tmp_path):
    fixture = _fixture(tmp_path)
    fixture["dyn"].write_bytes(b"Q2LAT001" + b"retired")
    _refresh(fixture, "dyn")
    with pytest.raises(RetirementValidationError, match="Q2LAT001|not Q2LAT002"):
        _validate(fixture)

    fixture = _fixture(tmp_path / "bundle")
    _json(fixture["bundle"], {"bundle_version": 2, "artifact_state": "admitted"})
    _refresh(fixture, "bundle")
    with pytest.raises(
        RetirementValidationError,
        match="bundle-v3|bundle_version|bundle_manifest record digest",
    ):
        _validate(fixture)

    fixture = _fixture(tmp_path / "optimizer")
    fixture["checkpoint_manifest"]["optimizer_state"] = "legacy-restored"
    _json(fixture["checkpoint_attestation"], fixture["checkpoint_manifest"])
    _refresh(fixture, "checkpoint_attestation")
    with pytest.raises(RetirementValidationError, match="not fresh multires"):
        _validate(fixture, _bypass_checkpoint=False)


@pytest.mark.skipif(torch is None, reason="real weights-only loader requires PyTorch")
def test_marker_only_checkpoint_and_fresh_sidecar_are_non_admissible(tmp_path):
    fixture = _fixture(tmp_path)
    with zipfile.ZipFile(fixture["checkpoint"], "w") as archive:
        archive.writestr(
            "marker/data.pkl",
            b"checkpoint_format manifest policy_state optimizer_state "
            b"q2-multires-attested-checkpoint-v1 multires-atlas-policy-v1 "
            b"models.multires_policy.MultiresQ2BotPolicy",
        )
    _refresh_checkpoint(fixture)
    with pytest.raises(RetirementValidationError, match="real attested weights-only"):
        _validate(fixture)


@pytest.mark.parametrize(
    "selector",
    ["train.ppo", "models.policy.Q2BotPolicy", "public_network_thermal_bc_live_v2"],
)
def test_service_and_operational_legacy_selectors_fail(selector, tmp_path):
    fixture = _fixture(tmp_path)
    fixture["selector"].write_text(
        f"exec python3 -m train.multires_service # {selector}\n", encoding="utf-8"
    )
    with pytest.raises(RetirementValidationError, match="retired value"):
        _validate(fixture)

    fixture = _fixture(tmp_path / "operational-file")
    (fixture["runtime"] / "fallback.cfg").write_text(selector, encoding="utf-8")
    with pytest.raises(RetirementValidationError, match="retired value"):
        _validate(fixture)


def test_unselected_checkpoint_symlink_path_escape_and_historical_overlap_fail(tmp_path):
    fixture = _fixture(tmp_path)
    (fixture["runtime"] / "optimizer.pt").write_bytes(b"retired")
    with pytest.raises(RetirementValidationError, match="checkpoint|retired value"):
        _validate(fixture)

    fixture = _fixture(tmp_path / "symlink")
    (fixture["runtime"] / "linked").symlink_to(fixture["bundle"])
    with pytest.raises(RetirementValidationError, match="symlink"):
        _validate(fixture)

    fixture = _fixture(tmp_path / "escape")
    escaped = tmp_path / "outside.pt"
    escaped.write_bytes(fixture["checkpoint"].read_bytes())
    fixture["cold_document"]["inputs"]["checkpoint"] = {
        "path": str(escaped), "sha256": _sha(escaped)
    }
    _json(fixture["cold"], fixture["cold_document"])
    with pytest.raises(RetirementValidationError, match="escapes"):
        _validate(fixture)

    fixture = _fixture(tmp_path / "overlap")
    custom = copy.deepcopy(json.loads(MANIFEST.read_text()))
    custom["historical_evidence"][0]["path"] = str(fixture["operational"])
    custom_manifest = tmp_path / "overlap-retirement.json"
    custom_manifest.write_bytes(
        (json.dumps(custom, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    custom_digest = _sha(custom_manifest)
    fixture["cold_document"]["retirement_manifest_sha256"] = custom_digest
    _json(fixture["cold"], fixture["cold_document"])
    with pytest.raises(RetirementValidationError, match="not disjoint"):
        _validate(
            fixture,
            manifest_path=custom_manifest,
            expected_manifest_sha256=custom_digest,
        )
