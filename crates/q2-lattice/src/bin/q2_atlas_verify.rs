use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use q2_lattice_rs::atlas::{AtlasCounts, AtlasLimits, AtlasManifest, sha256_hex};
use serde::Serialize;

const ATLAS_MEDIA_TYPE: &str = "application/vnd.q2.atlas-v1";
const SUMMARY_SCHEMA: &str = "q2-atlas-verification-v1";
const DESIGN_SHA256: &str = "c55fc7ffc32bd0e88410b8493b46c179f3333f3806632ff8e6530f1c717508e6";

#[derive(Debug, Serialize)]
struct VerificationSummary {
    schema: &'static str,
    passed: bool,
    canonical_map_id: String,
    bsp_sha256: String,
    manifest_sha256: String,
    artifact_name: String,
    atlas_sha256: String,
    origin: [i64; 3],
    counts: AtlasCounts,
    collision_contract_sha256: String,
}

fn read_bounded(path: &Path, maximum: usize, label: &str) -> Result<Vec<u8>, String> {
    let read_limit = u64::try_from(maximum)
        .map_err(|_| format!("{label} byte limit does not fit u64"))?
        .checked_add(1)
        .ok_or_else(|| format!("{label} byte limit overflow"))?;
    let file = File::open(path)
        .map_err(|error| format!("cannot open {label} {}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    if bytes.len() > maximum {
        return Err(format!(
            "{label} bytes {} exceed default limit {maximum}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn atlas_artifact_name(manifest: &AtlasManifest) -> Result<String, String> {
    let mut candidates = manifest
        .artifacts
        .iter()
        .filter(|(_, identity)| identity.media_type == ATLAS_MEDIA_TYPE)
        .map(|(name, _)| name);
    let Some(name) = candidates.next() else {
        return Err(format!(
            "manifest has no {ATLAS_MEDIA_TYPE} artifact identity"
        ));
    };
    if candidates.next().is_some() {
        return Err(format!(
            "manifest has multiple {ATLAS_MEDIA_TYPE} artifact identities"
        ));
    }
    Ok(name.clone())
}

fn verify(manifest_bytes: &[u8], atlas_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let limits = AtlasLimits::default();
    let manifest = AtlasManifest::from_canonical_json(manifest_bytes, &limits)
        .map_err(|error| error.to_string())?;
    if manifest.specification_sha256 != DESIGN_SHA256 {
        return Err(format!(
            "manifest specification {} != authoritative design {DESIGN_SHA256}",
            manifest.specification_sha256
        ));
    }
    let artifact_name = atlas_artifact_name(&manifest)?;
    let artifact = manifest
        .decode_and_verify_atlas_artifact(&artifact_name, atlas_bytes, &limits)
        .map_err(|error| error.to_string())?;
    let summary = VerificationSummary {
        schema: SUMMARY_SCHEMA,
        passed: true,
        canonical_map_id: manifest.bsp.canonical_map_id.clone(),
        bsp_sha256: manifest.bsp.sha256.clone(),
        manifest_sha256: sha256_hex(manifest_bytes),
        artifact_name,
        atlas_sha256: sha256_hex(atlas_bytes),
        origin: artifact.origin.0,
        counts: AtlasCounts::from_artifact(&artifact),
        collision_contract_sha256: manifest.oracles.collision_oracle.contract_sha256.clone(),
    };
    let mut output = serde_json::to_vec(&summary).map_err(|error| error.to_string())?;
    output.push(b'\n');
    Ok(output)
}

fn run(arguments: &[OsString]) -> Result<Vec<u8>, String> {
    if arguments.len() != 3 {
        return Err(
            "usage: q2-atlas-verify CANONICAL_ATLAS_MANIFEST.json RAW_ATLAS.bin".to_owned(),
        );
    }
    let limits = AtlasLimits::default();
    let manifest = read_bounded(
        Path::new(&arguments[1]),
        limits.max_manifest_bytes,
        "Atlas manifest",
    )?;
    let atlas = read_bounded(
        Path::new(&arguments[2]),
        limits.max_atlas_decompressed_bytes,
        "raw Atlas",
    )?;
    verify(&manifest, &atlas)
}

fn main() {
    match run(&std::env::args_os().collect::<Vec<_>>()) {
        Ok(output) => print!("{}", String::from_utf8_lossy(&output)),
        Err(error) => {
            eprintln!("q2-atlas-verify: {error}");
            std::process::exit(65);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use q2_lattice_rs::atlas::{
        ATLAS_CELL_SIZES, ATLAS_MAGIC, ATLAS_SCHEMA_VERSION, ArtifactManifest, AtlasArtifact,
        AtlasOrigin, B1AuthorityExecutables, B1AuthorityIdentities, B1AuthorityIdentity,
        B1NormativeDocuments, B1RuntimeAuthoritySeal, BspIdentity, COLLISION_ORACLE_NAME,
        COLLISION_ORACLE_SCHEMA, CollisionOracleAdmission, CollisionParameters,
        CollisionSourceClosure, FALL_ORACLE_NAME, FALL_ORACLE_SCHEMA, FallOracleAdmission,
        FallParameters, FallSourceClosure, GridManifest, HullManifest, MASK_PLAYERSOLID_V1,
        MASK_SHOT_V1, ManifestBudgets, ORACLE_SEMANTIC_VERSION, OracleAdmissions, OracleBspBinding,
        OracleToolIdentity, PMOVE_ORACLE_NAME, PMOVE_ORACLE_SCHEMA, PmoveOracleAdmission,
        PmoveParameters, PmoveSourceClosure, ToolIdentity,
    };
    use serde_json::Value;

    use super::*;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let ordinal = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("q2-atlas-verify-{}-{ordinal}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn fixture() -> (AtlasManifest, Vec<u8>) {
        let limits = AtlasLimits::default();
        let bsp = BspIdentity {
            canonical_map_id: "verify-fixture".to_owned(),
            sha256: digest(0x10),
            provenance_sha256: digest(0x11),
            size_bytes: 1024,
            ibsp_version: 38,
        };
        let collision = CollisionOracleAdmission {
            tool: OracleToolIdentity {
                name: COLLISION_ORACLE_NAME.to_owned(),
                schema: COLLISION_ORACLE_SCHEMA.to_owned(),
                version: ORACLE_SEMANTIC_VERSION,
                executable_sha256:
                    "781edaee1b9317766dbf831ad5edc8b5fdebe696969ca1efe0e54e2f3e5c7d1e".to_owned(),
                tool_identity_sha256:
                    "50fd0df0a296c54d0060dc2406977b1c78f9392ea1643e010f6759622557cfdf".to_owned(),
                physics_identity_sha256: digest(0x13),
            },
            bsp: OracleBspBinding {
                sha256: bsp.sha256.clone(),
                provenance_sha256: bsp.provenance_sha256.clone(),
            },
            parameters: CollisionParameters {
                mask_playersolid: MASK_PLAYERSOLID_V1,
                mask_shot: MASK_SHOT_V1,
            },
            source: CollisionSourceClosure {
                collision_sha256: digest(0x14),
                shared_header_sha256: digest(0x15),
                shared_source_sha256: digest(0x16),
                build_contract: "fixture cc-v1".to_owned(),
            },
            contract_sha256: String::new(),
        }
        .seal();
        let pmove = PmoveOracleAdmission {
            tool: OracleToolIdentity {
                name: PMOVE_ORACLE_NAME.to_owned(),
                schema: PMOVE_ORACLE_SCHEMA.to_owned(),
                version: ORACLE_SEMANTIC_VERSION,
                executable_sha256:
                    "66b481e924ec3d0a5e4eaf5458dd34cfe3c0927d5b7650455bceb368666718e4".to_owned(),
                tool_identity_sha256:
                    "50fd0df0a296c54d0060dc2406977b1c78f9392ea1643e010f6759622557cfdf".to_owned(),
                physics_identity_sha256: digest(0x1a),
            },
            bsp: OracleBspBinding {
                sha256: bsp.sha256.clone(),
                provenance_sha256: bsp.provenance_sha256.clone(),
            },
            parameters: PmoveParameters {
                gravity: 800,
                airaccelerate_f32_bits: 0.0_f32.to_bits(),
                constants: "fixture-pmove-constants".to_owned(),
            },
            source: PmoveSourceClosure {
                collision_sha256: collision.source.collision_sha256.clone(),
                pmove_sha256: digest(0x1b),
                shared_header_sha256: collision.source.shared_header_sha256.clone(),
                shared_source_sha256: collision.source.shared_source_sha256.clone(),
                build_contract: collision.source.build_contract.clone(),
            },
            contract_sha256: String::new(),
        }
        .seal();
        let fall_constants = "player_model=255,noclip=1,grapple_fly=0,release_grace=0.2,delta_scale=0.0001,water1=0.5,water2=0.25,water3=suppress,footstep=1,short=15,damage=30,far=55,fall_value_scale=0.5,fall_value_max=40,fall_time=0.3,damage_divisor=2,df_no_falling=8";
        let fall = FallOracleAdmission {
            tool: OracleToolIdentity {
                name: FALL_ORACLE_NAME.to_owned(),
                schema: FALL_ORACLE_SCHEMA.to_owned(),
                version: ORACLE_SEMANTIC_VERSION,
                executable_sha256:
                    "dfdcf7ed74cc3ad7b8aa73df86986a8a4a31207da98ccffb4dd61673c324bef8".to_owned(),
                tool_identity_sha256:
                    "8f6706edf203bb75451fd148943fb9d0425a1b112f086b6886788434973117d5".to_owned(),
                physics_identity_sha256:
                    "8b1f06550cb546d329bbce209f2b13248810fc10e0e19799616a338c0f633582".to_owned(),
            },
            parameters: FallParameters {
                fall_damagemod_f32_bits: 1.0_f32.to_bits(),
                deathmatch: true,
                dmflags: 0,
                constants: fall_constants.to_owned(),
            },
            source: FallSourceClosure {
                shared_c_sha256: "6d30c143e359e18784615ad0f4b21a85b3b4b9b2d4b841792685b19a88a7b6d8"
                    .to_owned(),
                shared_h_sha256: "debd12ca5315cfa9e6cff714bad2d2c2fe708e378a37776e6665793aa3967357"
                    .to_owned(),
                integration_sha256:
                    "326f59b12ee60bd93252a9d5a39428c535097ed6bc7a6258f2326a3cdb12ed62".to_owned(),
                game_header_sha256:
                    "da27f13498fb7120b037b2a6b6ce0a36f4e90a90d1caf0c09c7aaeb1c8310877".to_owned(),
                constants_sha256:
                    "6274fdec332e9d51db6f1b8ca8a836835902e5269957f62c72e3d63a3a54c703".to_owned(),
                build_contract: "lithium-linux-c99-o1-f32-shared-fall-v1".to_owned(),
                tool_closure_sha256:
                    "8f6706edf203bb75451fd148943fb9d0425a1b112f086b6886788434973117d5".to_owned(),
            },
            contract_sha256: String::new(),
        }
        .seal();
        let artifact = AtlasArtifact::empty(AtlasOrigin([0, 0, 0]));
        let raw = artifact.encode_uncompressed(&limits).unwrap();
        let counts = AtlasCounts::from_artifact(&artifact);
        let artifacts = BTreeMap::from([(
            "verify-fixture.atlas.bin".to_owned(),
            ArtifactManifest::from_uncompressed(
                ATLAS_MEDIA_TYPE,
                &raw,
                raw.len() as u64,
                counts.named_counts(),
            ),
        )]);
        let manifest = AtlasManifest {
            schema_version: ATLAS_SCHEMA_VERSION,
            byte_order: "little".to_owned(),
            atlas_magic: String::from_utf8_lossy(ATLAS_MAGIC).into_owned(),
            specification_sha256: DESIGN_SHA256.to_owned(),
            bsp: bsp.clone(),
            analyzer: ToolIdentity {
                name: "fixture-analyzer".to_owned(),
                version: "1".to_owned(),
                sha256: digest(0x18),
            },
            oracles: OracleAdmissions {
                b1_runtime_authority_seal: B1RuntimeAuthoritySeal {
                    schema: "q2-b1-runtime-authority-seal-v1".to_owned(),
                    normative_documents: B1NormativeDocuments {
                        design_sha256:
                            "c55fc7ffc32bd0e88410b8493b46c179f3333f3806632ff8e6530f1c717508e6"
                                .to_owned(),
                        plan_sha256:
                            "371577feb8c40f542c90eec4b4aa91ef84c4a8e2019bf1614e59c46aedfec410"
                                .to_owned(),
                    },
                    hook_parity_attestation_sha256:
                        "2e473d8face6b89f5b32798ddc5264bb8cc406e8dc29fd837e85bbd11b53d5ab"
                            .to_owned(),
                    fixture_bsp_sha256:
                        "ed6c3ae52dffce93b932756486fdaea3992f6a8ce68dddf2fbfd4281e4515b3f"
                            .to_owned(),
                    analysis_bsp_sha256: bsp.sha256.clone(),
                    executables: B1AuthorityExecutables {
                        cm_sha256: collision.tool.executable_sha256.clone(),
                        pmove_sha256: pmove.tool.executable_sha256.clone(),
                        hook_sha256:
                            "cd8bc4107ae2e9f4ac006fbe469b360832db80b96a5597c2e5dfe12c32dc9284"
                                .to_owned(),
                        fall_sha256: fall.tool.executable_sha256.clone(),
                    },
                    identities: B1AuthorityIdentities {
                        collision: B1AuthorityIdentity {
                            tool_identity: collision.tool.tool_identity_sha256.clone(),
                            physics_identity: collision.tool.physics_identity_sha256.clone(),
                        },
                        pmove: B1AuthorityIdentity {
                            tool_identity: pmove.tool.tool_identity_sha256.clone(),
                            physics_identity: pmove.tool.physics_identity_sha256.clone(),
                        },
                        hook: B1AuthorityIdentity {
                            tool_identity:
                                "9c47e3339df3f194c7729ea95c1955708540411d8baffc8208aa92349e1d2e78"
                                    .to_owned(),
                            physics_identity:
                                "38f441106d653997466f8ace13baebe5e5515d6b77a7edf535a1d93576eef9d3"
                                    .to_owned(),
                        },
                        fall: B1AuthorityIdentity {
                            tool_identity: fall.tool.tool_identity_sha256.clone(),
                            physics_identity: fall.tool.physics_identity_sha256.clone(),
                        },
                    },
                },
                collision_oracle: collision,
                fall_oracle: fall,
                pmove_oracle: Some(pmove),
                hook_oracle: None,
            },
            generator: None,
            grid: GridManifest {
                origin: [0, 0, 0],
                model0_mins: [0, 0, 0],
                model0_maxs: [256, 256, 256],
                cell_sizes: ATLAS_CELL_SIZES.map(|value| value as u32),
                l0_chunk_dimensions: [16, 16, 16],
            },
            player_hulls: vec![
                HullManifest {
                    name: "standing".to_owned(),
                    mins: [-16, -16, -24],
                    maxs: [16, 16, 32],
                },
                HullManifest {
                    name: "crouched".to_owned(),
                    mins: [-16, -16, -24],
                    maxs: [16, 16, 4],
                },
            ],
            channels: vec![],
            artifacts,
            counts,
            budgets: ManifestBudgets::from(&limits),
            build_peak_rss_bytes: 1,
            limitations: vec![],
            confidence_summary: "verified empty fixture".to_owned(),
        };
        (manifest, raw)
    }

    fn write_fixture(
        directory: &TestDirectory,
        manifest_bytes: &[u8],
        atlas_bytes: &[u8],
    ) -> Vec<OsString> {
        let manifest_path = directory.0.join("atlas.manifest.json");
        let atlas_path = directory.0.join("atlas.bin");
        fs::write(&manifest_path, manifest_bytes).unwrap();
        fs::write(&atlas_path, atlas_bytes).unwrap();
        vec![
            OsString::from("q2-atlas-verify"),
            manifest_path.into_os_string(),
            atlas_path.into_os_string(),
        ]
    }

    #[test]
    fn cli_accepts_only_a_canonical_bound_manifest_and_raw_atlas() {
        let limits = AtlasLimits::default();
        let (manifest, raw) = fixture();
        let manifest_bytes = manifest.canonical_json(&limits).unwrap();
        let directory = TestDirectory::new();
        let arguments = write_fixture(&directory, &manifest_bytes, &raw);

        let first = run(&arguments).unwrap();
        let second = run(&arguments).unwrap();
        assert_eq!(first, second);
        assert!(first.ends_with(b"\n"));
        let summary: Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(summary["schema"], SUMMARY_SCHEMA);
        assert_eq!(summary["passed"], true);
        assert_eq!(summary["canonical_map_id"], "verify-fixture");
        assert_eq!(summary["manifest_sha256"], sha256_hex(&manifest_bytes));
        assert_eq!(summary["atlas_sha256"], sha256_hex(&raw));
        assert_eq!(summary["origin"], serde_json::json!([0, 0, 0]));
    }

    #[test]
    fn cli_rejects_wrong_arity_noncanonical_manifest_and_corrupt_atlas() {
        assert!(run(&[OsString::from("q2-atlas-verify")]).is_err());
        assert!(
            run(&[
                OsString::from("q2-atlas-verify"),
                OsString::from("one"),
                OsString::from("two"),
                OsString::from("three"),
            ])
            .is_err()
        );

        let limits = AtlasLimits::default();
        let (manifest, raw) = fixture();
        let mut noncanonical = manifest.canonical_json(&limits).unwrap();
        noncanonical.pop();
        let directory = TestDirectory::new();
        let arguments = write_fixture(&directory, &noncanonical, &raw);
        assert!(run(&arguments).unwrap_err().contains("not canonical"));

        let canonical = manifest.canonical_json(&limits).unwrap();
        let mut corrupt = raw;
        corrupt[0] ^= 1;
        let arguments = write_fixture(&directory, &canonical, &corrupt);
        assert!(run(&arguments).is_err());
    }

    #[test]
    fn verifier_rejects_ambiguous_atlas_authority_and_origin_substitution() {
        let limits = AtlasLimits::default();
        let (mut manifest, raw) = fixture();
        let duplicate = manifest.artifacts.values().next().unwrap().clone();
        manifest
            .artifacts
            .insert("substitute.atlas.bin".to_owned(), duplicate);
        let ambiguous = manifest.canonical_json(&limits).unwrap();
        assert!(verify(&ambiguous, &raw).unwrap_err().contains("multiple"));

        manifest.artifacts.remove("substitute.atlas.bin");
        manifest.grid.origin = [-256, 0, 0];
        manifest.grid.model0_mins = [-1, 0, 0];
        let wrong_origin = manifest.canonical_json(&limits).unwrap();
        assert!(verify(&wrong_origin, &raw).unwrap_err().contains("origin"));
    }

    #[test]
    fn verifier_fixture_rejects_superseded_normative_authority() {
        let limits = AtlasLimits::default();
        let (mut manifest, raw) = fixture();
        manifest.specification_sha256 =
            "eab02d2269f250a26f45bb5d3b1f66ffab2c34ba3ee958d2f8b5bd2a14fef8b5".to_owned();
        let manifest_bytes = manifest.canonical_json(&limits).unwrap();
        assert!(
            verify(&manifest_bytes, &raw)
                .unwrap_err()
                .contains("authoritative design")
        );

        let (mut manifest, _) = fixture();
        manifest
            .oracles
            .b1_runtime_authority_seal
            .normative_documents
            .design_sha256 =
            "eab02d2269f250a26f45bb5d3b1f66ffab2c34ba3ee958d2f8b5bd2a14fef8b5".to_owned();
        assert!(manifest.canonical_json(&limits).is_err());

        let (mut manifest, _) = fixture();
        manifest
            .oracles
            .b1_runtime_authority_seal
            .normative_documents
            .plan_sha256 =
            "970e97b9478b27ad1f1cd35d29a74b2ed2cd51ed1ae8b4af82605615d5b5ba6b".to_owned();
        assert!(manifest.canonical_json(&limits).is_err());
    }
}
