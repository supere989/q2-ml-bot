use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use q2_lattice_rs::atlas::{AtlasCounts, AtlasLimits, AtlasManifest, sha256_hex};
use serde::Serialize;

const ATLAS_MEDIA_TYPE: &str = "application/vnd.q2.atlas-v1";
const SUMMARY_SCHEMA: &str = "q2-atlas-verification-v1";

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
        AtlasOrigin, BspIdentity, COLLISION_ORACLE_NAME, COLLISION_ORACLE_SCHEMA,
        CollisionOracleAdmission, CollisionParameters, CollisionSourceClosure, GridManifest,
        HullManifest, MASK_PLAYERSOLID_V1, MASK_SHOT_V1, ManifestBudgets, ORACLE_SEMANTIC_VERSION,
        OracleAdmissions, OracleBspBinding, OracleToolIdentity, ToolIdentity,
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
                executable_sha256: digest(0x12),
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
            specification_sha256: digest(0x17),
            bsp,
            analyzer: ToolIdentity {
                name: "fixture-analyzer".to_owned(),
                version: "1".to_owned(),
                sha256: digest(0x18),
            },
            oracles: OracleAdmissions {
                collision_oracle: collision,
                pmove_oracle: None,
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
}
