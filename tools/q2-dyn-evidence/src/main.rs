use std::ffi::{CString, OsString};
use std::fs::{self, File, OpenOptions};
use std::hint::black_box;
use std::io::{Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use q2_lattice_rs::atlas::{
    AtlasAggregateCell, AtlasArtifact, AtlasCounts, AtlasLevel, AtlasLimits, AtlasManifest,
    AtlasOrigin, sha256_hex,
};
use q2_lattice_rs::dynstate::{
    BatchExpectation, DYN_FEATURE_WIDTH, DYN_MAGIC, DYN_SCHEMA_VERSION, DynBatch, DynCell,
    DynError, DynFeatureInput, DynFence, DynLimits, DynState, PersistentChannels,
    RETIRED_DYN_MAGIC, decode_snapshot, encode_snapshot,
};
use serde::Serialize;

const REPORT_SCHEMA: &str = "q2-b2-dyn-evidence-v1";
const ATLAS_MEDIA_TYPE: &str = "application/vnd.q2.atlas-v1";
const DESIGN_SHA256: &str = "eab02d2269f250a26f45bb5d3b1f66ffab2c34ba3ee958d2f8b5bd2a14fef8b5";
const CLIENT_COUNT: u32 = 4;
const MIN_SAMPLES: usize = 2_000;
const DEFAULT_SAMPLES: usize = 4_000;
const WARMUP_SAMPLES: usize = 256;
const TOTAL_P99_LIMIT_NS: u64 = 500_000;
const FOUR_CLIENT_BYTE_LIMIT: usize = 8 * 1024 * 1024;
const MAX_REPRESENTATIVE_L2_CELLS: usize = 10_000;
const REPORT_NAME: &str = "b2-dyn-evidence.json";
const SNAPSHOT_HEADER_BYTES: usize = 208;
const BUILD_COMMIT: Option<&str> = option_env!("Q2_LATTICE_CRATE_COMMIT");
const BUILD_HELPER_SOURCE_SHA256: &str = env!("Q2_DYN_HELPER_SOURCE_CLOSURE_SHA256");
const BUILD_LATTICE_SOURCE_SHA256: &str = env!("Q2_LATTICE_SOURCE_CLOSURE_SHA256");
const SOURCE_CLOSURE_ALGORITHM: &str = "sha256(canonical-json([{path,sha256},...]))-v1";
const COMMIT_BINDING_ALGORITHM: &str =
    "sha256(canonical-json({repo_commit,source_closure_sha256}))-v1";
static NEXT_STAGING_DIRECTORY: AtomicU64 = AtomicU64::new(0);

type AppResult<T> = Result<T, String>;

#[derive(Clone, Debug, Eq, PartialEq)]
struct Arguments {
    repo_root: PathBuf,
    atlas: PathBuf,
    manifest: PathBuf,
    bsp: PathBuf,
    output: PathBuf,
    expected_map_id: String,
    expected_origin: [i64; 3],
    expected_analyzer_authority: String,
    expected_crate_commit: String,
    map_epoch: u64,
    environment_steps: u64,
    samples: usize,
}

#[derive(Debug, Serialize)]
struct EvidenceReport {
    schema: &'static str,
    passed: bool,
    authority: AuthorityEvidence,
    provenance: ProvenanceEvidence,
    host: HostEvidence,
    atlas: AtlasEvidence,
    dyn_state: DynEvidence,
    negative_fences_and_limits: NegativeEvidence,
    performance: PerformanceEvidence,
}

#[derive(Debug, Serialize)]
struct AuthorityEvidence {
    specification_sha256: String,
    analyzer_name: String,
    analyzer_version: String,
    analyzer_authority_sha256: String,
    crate_commit: String,
    executable_sha256: String,
    canonical_map_id: String,
    map_epoch: u64,
    environment_steps: u64,
}

#[derive(Debug, Serialize)]
struct HostEvidence {
    hostname: String,
    kernel_release: String,
    architecture: &'static str,
}

#[derive(Debug, Serialize)]
struct ProvenanceEvidence {
    embedded_repo_commit: String,
    executable: FileEvidence,
    helper_source_closure: SourceClosureEvidence,
    q2_lattice_source_closure: SourceClosureEvidence,
}

#[derive(Debug, Serialize)]
struct SourceInputEvidence {
    path: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Serialize)]
struct SourceClosureEvidence {
    algorithm: &'static str,
    sha256: String,
    embedded_sha256: String,
    repo_commit: String,
    commit_binding_algorithm: &'static str,
    commit_bound_sha256: String,
    inputs: Vec<SourceInputEvidence>,
}

#[derive(Clone, Debug, Serialize)]
struct FileEvidence {
    path: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Serialize)]
struct AtlasEvidence {
    manifest: FileEvidence,
    artifact: FileEvidence,
    bsp: FileEvidence,
    origin: [i64; 3],
    counts: AtlasCounts,
    resident_bytes: usize,
    representative_l2_cells: usize,
    lookup: &'static str,
}

#[derive(Debug, Serialize)]
struct SnapshotEvidence {
    client_id: u32,
    file: FileEvidence,
    magic: String,
    schema_version: u16,
    l2_cells: usize,
    l3_cells: usize,
    resident_bytes: usize,
    byte_identical_roundtrip: bool,
}

#[derive(Debug, Serialize)]
struct DynEvidence {
    snapshot_magic: String,
    schema_version: u16,
    client_ids: Vec<u32>,
    client_count: u32,
    common_environment_steps: u64,
    population: &'static str,
    snapshots: Vec<SnapshotEvidence>,
    combined_compressed_bytes: usize,
    combined_resident_bytes: usize,
    combined_limit_bytes: usize,
    batch_ids_and_step_admitted: bool,
}

#[derive(Debug, Serialize)]
struct NegativeEvidence {
    stale_atlas_sha256_rejected: bool,
    stale_map_sha256_rejected: bool,
    stale_origin_rejected: bool,
    stale_map_epoch_rejected: bool,
    stale_environment_step_rejected: bool,
    wrong_client_count_rejected: bool,
    duplicate_client_rejected: bool,
    retired_schema_rejected: bool,
    mixed_schema_rejected: bool,
    payload_digest_corruption_rejected: bool,
    cell_size_mismatch_rejected: bool,
    soft_compressed_limit_reported: bool,
    hard_compressed_limit_rejected: bool,
    hard_resident_limit_rejected: bool,
    materialized_cell_limit_rejected: bool,
}

impl NegativeEvidence {
    fn passed(&self) -> bool {
        self.stale_atlas_sha256_rejected
            && self.stale_map_sha256_rejected
            && self.stale_origin_rejected
            && self.stale_map_epoch_rejected
            && self.stale_environment_step_rejected
            && self.wrong_client_count_rejected
            && self.duplicate_client_rejected
            && self.retired_schema_rejected
            && self.mixed_schema_rejected
            && self.payload_digest_corruption_rejected
            && self.cell_size_mismatch_rejected
            && self.soft_compressed_limit_reported
            && self.hard_compressed_limit_rejected
            && self.hard_resident_limit_rejected
            && self.materialized_cell_limit_rejected
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
struct LatencyDistribution {
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    max_ns: u64,
}

#[derive(Debug, Serialize)]
struct PerformanceEvidence {
    scope: &'static str,
    resident_samples: usize,
    warmup_samples: usize,
    clients_per_sample: u32,
    atlas_lookup: LatencyDistribution,
    dyn_feature_assembly: LatencyDistribution,
    total: LatencyDistribution,
    total_p99_limit_ns: u64,
    total_p99_passed: bool,
    feature_width: usize,
}

struct StagingDirectory {
    path: PathBuf,
    keep: bool,
}

impl StagingDirectory {
    fn create(output: &Path) -> AppResult<Self> {
        let parent = output_parent(output);
        if !parent.is_dir() {
            return Err(format!(
                "output parent is not a directory: {}",
                parent.display()
            ));
        }
        let name = output
            .file_name()
            .and_then(|value| value.to_str())
            .ok_or_else(|| "output directory requires a UTF-8 final component".to_owned())?;
        let ordinal = NEXT_STAGING_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".{name}.partial-{}-{ordinal}", std::process::id()));
        fs::create_dir(&path).map_err(|error| {
            format!(
                "cannot create exclusive staging directory {}: {error}",
                path.display()
            )
        })?;
        Ok(Self { path, keep: false })
    }

    fn publish(mut self, output: &Path) -> AppResult<()> {
        sync_directory(&self.path)?;
        rename_noreplace(&self.path, output).map_err(|error| {
            format!(
                "cannot exclusively publish {} as {}: {error}",
                self.path.display(),
                output.display()
            )
        })?;
        self.keep = true;
        sync_directory(output_parent(output))?;
        Ok(())
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if !self.keep {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn output_parent(output: &Path) -> &Path {
    output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn rename_noreplace(source: &Path, destination: &Path) -> std::io::Result<()> {
    let source = CString::new(source.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "source path contains a NUL byte",
        )
    })?;
    let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "destination path contains a NUL byte",
        )
    })?;
    // SAFETY: both pointers reference live, NUL-terminated C strings for the
    // duration of the call. AT_FDCWD makes both paths relative to this process,
    // and RENAME_NOREPLACE is the required atomic no-overwrite operation.
    let result = unsafe {
        libc::syscall(
            libc::SYS_renameat2,
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

fn usage() -> &'static str {
    "usage: q2-dyn-evidence \
  --repo-root Q2_ML_BOT_CHECKOUT \
  --atlas RAW_ATLAS.bin --manifest CANONICAL_ATLAS_MANIFEST.json --bsp MAP.bsp \
  --expected-map-id MAP --expected-origin X,Y,Z \
  --expected-analyzer-authority SHA256 --expected-crate-commit GIT_COMMIT \
  --map-epoch N --environment-steps N --output NEW_DIRECTORY [--samples N]\n\
The binary must be compiled with Q2_LATTICE_CRATE_COMMIT set to the same full Git commit."
}

fn parse_arguments(arguments: &[OsString]) -> AppResult<Arguments> {
    let mut repo_root = None;
    let mut atlas = None;
    let mut manifest = None;
    let mut bsp = None;
    let mut output = None;
    let mut expected_map_id = None;
    let mut expected_origin = None;
    let mut expected_analyzer_authority = None;
    let mut expected_crate_commit = None;
    let mut map_epoch = None;
    let mut environment_steps = None;
    let mut samples = DEFAULT_SAMPLES;
    let mut index = 1;
    while index < arguments.len() {
        let flag = arguments[index]
            .to_str()
            .ok_or_else(|| "command-line flags must be UTF-8".to_owned())?;
        if flag == "--help" || flag == "-h" {
            return Err(usage().to_owned());
        }
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?
            .clone();
        match flag {
            "--repo-root" => set_once(&mut repo_root, PathBuf::from(value), flag)?,
            "--atlas" => set_once(&mut atlas, PathBuf::from(value), flag)?,
            "--manifest" => set_once(&mut manifest, PathBuf::from(value), flag)?,
            "--bsp" => set_once(&mut bsp, PathBuf::from(value), flag)?,
            "--output" => set_once(&mut output, PathBuf::from(value), flag)?,
            "--expected-map-id" => {
                let parsed = utf8_value(&value, flag)?;
                if parsed.is_empty() {
                    return Err("--expected-map-id cannot be empty".to_owned());
                }
                set_once(&mut expected_map_id, parsed, flag)?;
            }
            "--expected-origin" => {
                set_once(
                    &mut expected_origin,
                    parse_origin(&utf8_value(&value, flag)?)?,
                    flag,
                )?;
            }
            "--expected-analyzer-authority" => {
                let parsed = utf8_value(&value, flag)?;
                validate_sha256(&parsed, flag)?;
                set_once(&mut expected_analyzer_authority, parsed, flag)?;
            }
            "--expected-crate-commit" => {
                let parsed = utf8_value(&value, flag)?;
                validate_commit(&parsed)?;
                set_once(&mut expected_crate_commit, parsed, flag)?;
            }
            "--map-epoch" => {
                let parsed = parse_u64(&value, flag)?;
                if parsed == 0 {
                    return Err("--map-epoch must be nonzero".to_owned());
                }
                set_once(&mut map_epoch, parsed, flag)?;
            }
            "--environment-steps" => {
                set_once(&mut environment_steps, parse_u64(&value, flag)?, flag)?;
            }
            "--samples" => {
                samples = usize::try_from(parse_u64(&value, flag)?)
                    .map_err(|_| "--samples does not fit usize".to_owned())?;
            }
            _ => return Err(format!("unknown flag {flag}\n{}", usage())),
        }
        index += 2;
    }
    if samples < MIN_SAMPLES {
        return Err(format!("--samples must be at least {MIN_SAMPLES}"));
    }
    Ok(Arguments {
        repo_root: required(repo_root, "--repo-root")?,
        atlas: required(atlas, "--atlas")?,
        manifest: required(manifest, "--manifest")?,
        bsp: required(bsp, "--bsp")?,
        output: required(output, "--output")?,
        expected_map_id: required(expected_map_id, "--expected-map-id")?,
        expected_origin: required(expected_origin, "--expected-origin")?,
        expected_analyzer_authority: required(
            expected_analyzer_authority,
            "--expected-analyzer-authority",
        )?,
        expected_crate_commit: required(expected_crate_commit, "--expected-crate-commit")?,
        map_epoch: required(map_epoch, "--map-epoch")?,
        environment_steps: required(environment_steps, "--environment-steps")?,
        samples,
    })
}

fn set_once<T>(destination: &mut Option<T>, value: T, flag: &str) -> AppResult<()> {
    if destination.replace(value).is_some() {
        return Err(format!("duplicate {flag}"));
    }
    Ok(())
}

fn required<T>(value: Option<T>, flag: &str) -> AppResult<T> {
    value.ok_or_else(|| format!("missing required {flag}"))
}

fn utf8_value(value: &OsString, flag: &str) -> AppResult<String> {
    value
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{flag} value must be UTF-8"))
}

fn parse_u64(value: &OsString, flag: &str) -> AppResult<u64> {
    utf8_value(value, flag)?
        .parse()
        .map_err(|error| format!("invalid {flag}: {error}"))
}

fn parse_origin(value: &str) -> AppResult<[i64; 3]> {
    let fields: Vec<_> = value.split(',').collect();
    if fields.len() != 3 {
        return Err("--expected-origin must be X,Y,Z".to_owned());
    }
    let parsed = fields
        .iter()
        .map(|field| field.parse::<i64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("invalid --expected-origin: {error}"))?;
    Ok([parsed[0], parsed[1], parsed[2]])
}

fn validate_sha256(value: &str, label: &str) -> AppResult<()> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{label} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(())
}

fn validate_commit(value: &str) -> AppResult<()> {
    if value.len() != 40
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return Err("--expected-crate-commit must be a full lowercase 40-hex commit".to_owned());
    }
    Ok(())
}

fn parse_digest(value: &str, label: &str) -> AppResult<[u8; 32]> {
    validate_sha256(value, label)?;
    let mut output = [0_u8; 32];
    for (index, destination) in output.iter_mut().enumerate() {
        *destination = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|error| format!("invalid {label}: {error}"))?;
    }
    Ok(output)
}

fn read_bounded(path: &Path, maximum: usize, label: &str) -> AppResult<Vec<u8>> {
    let read_limit = u64::try_from(maximum)
        .map_err(|_| format!("{label} limit does not fit u64"))?
        .checked_add(1)
        .ok_or_else(|| format!("{label} limit overflow"))?;
    let file = File::open(path)
        .map_err(|error| format!("cannot open {label} {}: {error}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    if bytes.len() > maximum {
        return Err(format!(
            "{label} bytes {} exceed limit {maximum}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn sole_atlas_artifact_name(manifest: &AtlasManifest) -> AppResult<String> {
    let mut candidates = manifest
        .artifacts
        .iter()
        .filter(|(_, identity)| identity.media_type == ATLAS_MEDIA_TYPE)
        .map(|(name, _)| name);
    let name = candidates
        .next()
        .ok_or_else(|| format!("manifest has no {ATLAS_MEDIA_TYPE} artifact"))?;
    if candidates.next().is_some() {
        return Err(format!(
            "manifest has multiple {ATLAS_MEDIA_TYPE} artifacts"
        ));
    }
    Ok(name.clone())
}

fn canonical_path(path: &Path, label: &str) -> AppResult<String> {
    fs::canonicalize(path)
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))?
        .to_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("canonical {label} path is not UTF-8"))
}

fn file_evidence(path: &Path, bytes: &[u8], label: &str) -> AppResult<FileEvidence> {
    Ok(FileEvidence {
        path: canonical_path(path, label)?,
        sha256: sha256_hex(bytes),
        size_bytes: bytes.len() as u64,
    })
}

#[derive(Serialize)]
struct SourceHashRecord<'a> {
    path: &'a str,
    sha256: &'a str,
}

#[derive(Serialize)]
struct CommitBinding<'a> {
    repo_commit: &'a str,
    source_closure_sha256: &'a str,
}

fn source_closure_evidence(
    repo_root: &Path,
    paths: Vec<PathBuf>,
    embedded_sha256: &str,
    embedded_repo_commit: &str,
    label: &str,
) -> AppResult<SourceClosureEvidence> {
    validate_sha256(embedded_sha256, label)?;
    validate_commit(embedded_repo_commit)?;
    let mut inputs = Vec::with_capacity(paths.len());
    for path in paths {
        let bytes = fs::read(&path)
            .map_err(|error| format!("cannot read {label} input {}: {error}", path.display()))?;
        let relative = path
            .strip_prefix(repo_root)
            .map_err(|_| {
                format!(
                    "{label} input is outside repository root: {}",
                    path.display()
                )
            })?
            .to_str()
            .ok_or_else(|| format!("{label} input path is not UTF-8: {}", path.display()))?
            .replace('\\', "/");
        inputs.push(SourceInputEvidence {
            path: relative,
            sha256: sha256_hex(&bytes),
            size_bytes: bytes.len() as u64,
        });
    }
    inputs.sort_by(|left, right| left.path.cmp(&right.path));
    if inputs.windows(2).any(|pair| pair[0].path == pair[1].path) {
        return Err(format!("{label} source closure contains duplicate paths"));
    }
    let records: Vec<_> = inputs
        .iter()
        .map(|input| SourceHashRecord {
            path: &input.path,
            sha256: &input.sha256,
        })
        .collect();
    let payload = serde_json::to_vec(&records).map_err(|error| error.to_string())?;
    let sha256 = sha256_hex(&payload);
    if sha256 != embedded_sha256 {
        return Err(format!(
            "computed {label} source closure {sha256} != embedded {embedded_sha256}"
        ));
    }
    let commit_binding = serde_json::to_vec(&CommitBinding {
        repo_commit: embedded_repo_commit,
        source_closure_sha256: &sha256,
    })
    .map_err(|error| error.to_string())?;
    Ok(SourceClosureEvidence {
        algorithm: SOURCE_CLOSURE_ALGORITHM,
        sha256,
        embedded_sha256: embedded_sha256.to_owned(),
        repo_commit: embedded_repo_commit.to_owned(),
        commit_binding_algorithm: COMMIT_BINDING_ALGORITHM,
        commit_bound_sha256: sha256_hex(&commit_binding),
        inputs,
    })
}

fn helper_source_paths(repo_root: &Path) -> AppResult<Vec<PathBuf>> {
    let base = repo_root.join("tools/q2-dyn-evidence");
    let mut paths = vec![
        base.join("Cargo.lock"),
        base.join("Cargo.toml"),
        base.join("README.md"),
        base.join("build.rs"),
    ];
    collect_rust_sources(&base.join("src"), &mut paths)?;
    canonical_source_paths(paths, "helper")
}

fn lattice_source_paths(repo_root: &Path) -> AppResult<Vec<PathBuf>> {
    let base = repo_root.join("crates/q2-lattice");
    let mut paths = vec![base.join("Cargo.toml")];
    collect_rust_sources(&base.join("src"), &mut paths)?;
    canonical_source_paths(paths, "q2-lattice")
}

fn collect_rust_sources(directory: &Path, output: &mut Vec<PathBuf>) -> AppResult<()> {
    let mut entries: Vec<_> = fs::read_dir(directory)
        .map_err(|error| {
            format!(
                "cannot read source directory {}: {error}",
                directory.display()
            )
        })?
        .map(|entry| {
            entry
                .map(|value| value.path())
                .map_err(|error| format!("cannot read entry in {}: {error}", directory.display()))
        })
        .collect::<AppResult<_>>()?;
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rust_sources(&path, output)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
    Ok(())
}

fn canonical_source_paths(mut paths: Vec<PathBuf>, label: &str) -> AppResult<Vec<PathBuf>> {
    paths.sort();
    paths.dedup();
    if let Some(missing) = paths.iter().find(|path| !path.is_file()) {
        return Err(format!(
            "{label} source-closure input missing: {}",
            missing.display()
        ));
    }
    Ok(paths)
}

fn representative_cell(cell: &AtlasAggregateCell, client_id: u32) -> AppResult<DynCell> {
    let confidence = (f32::from(cell.confidence) / f32::from(u16::MAX)).clamp(0.05, 1.0);
    let index_signal = ((cell.index.x.unsigned_abs()
        ^ cell.index.y.unsigned_abs().rotate_left(7)
        ^ cell.index.z.unsigned_abs().rotate_left(13)
        ^ client_id)
        % 17) as f32
        / 16.0;
    DynCell::new(
        PersistentChannels {
            engagement: 0.5 + index_signal,
            threat: 0.25 + f32::from(cell.hazard_severity) / 255.0,
            opportunity: 0.25 + f32::from(cell.standing_passable as u8),
            self_fire: 0.125 + f32::from(cell.crouched_passable as u8) * 0.25,
            deaths: 0.125 + f32::from(cell.hazard_severity) / 510.0,
        },
        1.0 + index_signal,
        confidence,
    )
    .map_err(|error| error.to_string())
}

fn build_states(
    artifact: &AtlasArtifact,
    fence: DynFence,
    environment_steps: u64,
    limits: &DynLimits,
) -> AppResult<Vec<DynState>> {
    if artifact.l2.is_empty() {
        return Err("admitted Atlas has no L2 cells for resident Dyn evidence".to_owned());
    }
    let cells = &artifact.l2[..artifact.l2.len().min(MAX_REPRESENTATIVE_L2_CELLS)];
    let mut states = Vec::with_capacity(CLIENT_COUNT as usize);
    for client_id in 0..CLIENT_COUNT {
        let mut state = DynState::new(fence, client_id, CLIENT_COUNT, environment_steps, limits)
            .map_err(|error| error.to_string())?;
        for cell in cells {
            state
                .upsert_l2(
                    fence,
                    cell.index,
                    representative_cell(cell, client_id)?,
                    limits,
                )
                .map_err(|error| error.to_string())?;
        }
        states.push(state);
    }
    Ok(states)
}

fn altered(mut digest: [u8; 32]) -> [u8; 32] {
    digest[0] ^= 0x80;
    digest
}

fn reject_fence(snapshot: &[u8], fence: DynFence, limits: &DynLimits, field: &'static str) -> bool {
    matches!(
        decode_snapshot(snapshot, fence, limits),
        Err(DynError::FenceMismatch(actual)) if actual == field
    )
}

fn snapshot_with_corrupted_payload(snapshot: &[u8]) -> AppResult<Vec<u8>> {
    let compressed = snapshot
        .get(SNAPSHOT_HEADER_BYTES..)
        .ok_or_else(|| "cannot corrupt a truncated Dyn snapshot".to_owned())?;
    let mut payload = zstd::stream::decode_all(compressed)
        .map_err(|error| format!("cannot decode Dyn payload for negative test: {error}"))?;
    let first = payload
        .first_mut()
        .ok_or_else(|| "cannot corrupt an empty Dyn payload".to_owned())?;
    *first ^= 0x01;
    let recompressed = zstd::stream::encode_all(payload.as_slice(), 3)
        .map_err(|error| format!("cannot recompress Dyn negative payload: {error}"))?;
    let mut corrupted = snapshot[..SNAPSHOT_HEADER_BYTES].to_vec();
    corrupted[32..40].copy_from_slice(&(recompressed.len() as u64).to_le_bytes());
    corrupted.extend_from_slice(&recompressed);
    Ok(corrupted)
}

fn run_negative_checks(
    snapshots: &[Vec<u8>],
    states: &[DynState],
    fence: DynFence,
    environment_steps: u64,
    limits: &DynLimits,
) -> AppResult<NegativeEvidence> {
    let mut stale_atlas = fence;
    stale_atlas.atlas_sha256 = altered(stale_atlas.atlas_sha256);
    let mut stale_map = fence;
    stale_map.map_sha256 = altered(stale_map.map_sha256);
    let mut stale_origin = fence;
    stale_origin.origin.0[0] = stale_origin.origin.0[0].saturating_add(256);
    let mut stale_epoch = fence;
    stale_epoch.map_epoch = stale_epoch.map_epoch.wrapping_add(1);
    let refs: Vec<_> = snapshots.iter().map(Vec::as_slice).collect();
    let expectation = BatchExpectation {
        fence,
        client_count: CLIENT_COUNT,
        environment_steps,
    };

    let duplicate_refs = [refs[0], refs[0], refs[2], refs[3]];
    let duplicate_client_rejected = matches!(
        DynBatch::decode(&duplicate_refs, expectation, limits),
        Err(DynError::InvalidFormat(message)) if message.contains("duplicate Dyn client id")
    );
    let stale_environment_step_rejected = matches!(
        DynBatch::decode(
            &refs,
            BatchExpectation {
                environment_steps: environment_steps.wrapping_add(1),
                ..expectation
            },
            limits,
        ),
        Err(DynError::StaleEnvironmentSteps { .. })
    );
    let wrong_client_count_rejected = matches!(
        DynBatch::decode(
            &refs[..3],
            BatchExpectation {
                client_count: 3,
                ..expectation
            },
            limits,
        ),
        Err(DynError::FenceMismatch("client_count"))
    );
    let mut retired = snapshots[0].clone();
    retired[..DYN_MAGIC.len()].copy_from_slice(RETIRED_DYN_MAGIC);
    let retired_schema_rejected = matches!(
        decode_snapshot(&retired, fence, limits),
        Err(DynError::RetiredSchema)
    );
    let mut mixed = snapshots[0].clone();
    mixed[8..10].copy_from_slice(&(DYN_SCHEMA_VERSION + 1).to_le_bytes());
    let mixed_schema_rejected = matches!(
        decode_snapshot(&mixed, fence, limits),
        Err(DynError::MixedSchema {
            expected: DYN_SCHEMA_VERSION,
            found
        }) if found == DYN_SCHEMA_VERSION + 1
    );
    let corrupted = snapshot_with_corrupted_payload(&snapshots[0])?;
    let payload_digest_corruption_rejected = matches!(
        decode_snapshot(&corrupted, fence, limits),
        Err(DynError::DigestMismatch)
    );
    let mut wrong_l2_size = snapshots[0].clone();
    wrong_l2_size[160..164].copy_from_slice(&32_u32.to_le_bytes());
    let mut wrong_l3_size = snapshots[0].clone();
    wrong_l3_size[164..168].copy_from_slice(&128_u32.to_le_bytes());
    let cell_size_mismatch_rejected = [wrong_l2_size, wrong_l3_size].iter().all(|snapshot| {
        matches!(
            decode_snapshot(snapshot, fence, limits),
            Err(DynError::InvalidFormat(message)) if message.contains("cell-size fence")
        )
    });
    let combined_compressed = snapshots.iter().map(Vec::len).sum::<usize>();
    let combined_resident = states
        .iter()
        .map(DynState::resident_bytes_estimate)
        .sum::<usize>();
    if combined_compressed == 0 || combined_resident == 0 {
        return Err("nonempty snapshots unexpectedly have a zero batch budget".to_owned());
    }
    let soft_limits = DynLimits {
        batch_soft_compressed_bytes: combined_compressed - 1,
        ..limits.clone()
    };
    let soft_compressed_limit_reported = DynBatch::decode(&refs, expectation, &soft_limits)
        .map(|batch| batch.report.soft_limit_exceeded)
        .unwrap_or(false);
    let hard_compressed_limits = DynLimits {
        batch_hard_compressed_bytes: combined_compressed - 1,
        ..limits.clone()
    };
    let hard_compressed_limit_rejected = matches!(
        DynBatch::decode(&refs, expectation, &hard_compressed_limits),
        Err(DynError::LimitExceeded(message)) if message.contains("batch compressed bytes")
    );
    let hard_resident_limits = DynLimits {
        batch_hard_resident_bytes: combined_resident - 1,
        ..limits.clone()
    };
    let hard_resident_limit_rejected = matches!(
        DynBatch::decode(&refs, expectation, &hard_resident_limits),
        Err(DynError::LimitExceeded(message)) if message.contains("batch resident bytes")
    );
    let cell_limits = DynLimits {
        max_materialized_cells: 0,
        ..limits.clone()
    };
    let mut empty = DynState::new(fence, 0, CLIENT_COUNT, environment_steps, &cell_limits)
        .map_err(|error| error.to_string())?;
    let first = states[0]
        .l2_cells()
        .next()
        .ok_or_else(|| "representative Dyn state has no L2 cells".to_owned())?;
    let materialized_cell_limit_rejected = matches!(
        empty.upsert_l2(fence, *first.0, *first.1, &cell_limits),
        Err(DynError::LimitExceeded(message)) if message.contains("materialized cells")
    );

    Ok(NegativeEvidence {
        stale_atlas_sha256_rejected: reject_fence(
            &snapshots[0],
            stale_atlas,
            limits,
            "atlas_sha256",
        ),
        stale_map_sha256_rejected: reject_fence(&snapshots[0], stale_map, limits, "map_sha256"),
        stale_origin_rejected: reject_fence(&snapshots[0], stale_origin, limits, "origin"),
        stale_map_epoch_rejected: reject_fence(&snapshots[0], stale_epoch, limits, "map_epoch"),
        stale_environment_step_rejected,
        wrong_client_count_rejected,
        duplicate_client_rejected,
        retired_schema_rejected,
        mixed_schema_rejected,
        payload_digest_corruption_rejected,
        cell_size_mismatch_rejected,
        soft_compressed_limit_reported,
        hard_compressed_limit_rejected,
        hard_resident_limit_rejected,
        materialized_cell_limit_rejected,
    })
}

fn percentile(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    debug_assert!(!sorted.is_empty());
    let rank = sorted
        .len()
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[rank]
}

fn distribution(mut samples: Vec<u64>) -> LatencyDistribution {
    samples.sort_unstable();
    LatencyDistribution {
        p50_ns: percentile(&samples, 50, 100),
        p95_ns: percentile(&samples, 95, 100),
        p99_ns: percentile(&samples, 99, 100),
        max_ns: *samples.last().expect("minimum sample count is nonzero"),
    }
}

fn duration_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn atlas_lookup(artifact: &AtlasArtifact, position: [f64; 3]) -> AppResult<&AtlasAggregateCell> {
    let index = artifact
        .origin
        .index(position, AtlasLevel::L2)
        .map_err(|error| error.to_string())?;
    artifact
        .l2
        .binary_search_by_key(&index, |cell| cell.index)
        .map(|ordinal| &artifact.l2[ordinal])
        .map_err(|_| format!("representative position has no admitted Atlas L2 cell {index:?}"))
}

fn feature_input(
    fence: DynFence,
    position: [f64; 3],
    sample: usize,
    client: usize,
) -> DynFeatureInput {
    DynFeatureInput {
        fence,
        world_position: position,
        yaw_degrees: ((sample * 17 + client * 73) % 360) as f32,
        thermal: None,
        survivability: [0.0, 0.75, 0.5],
        search_radius: 2048.0,
        score_scale: 8.0,
    }
}

fn exercise_sample(
    artifact: &AtlasArtifact,
    states: &[DynState],
    positions: &[[f64; 3]],
    sample: usize,
) -> AppResult<(u64, u64, u64)> {
    let total_start = Instant::now();
    let mut atlas_ns = 0_u64;
    let mut dyn_ns = 0_u64;
    for (client, state) in states.iter().enumerate() {
        let position = positions[(sample + client * 97) % positions.len()];
        let atlas_start = Instant::now();
        black_box(atlas_lookup(artifact, position)?);
        atlas_ns = atlas_ns.saturating_add(duration_ns(atlas_start));
        let dyn_start = Instant::now();
        black_box(
            state
                .feature_block(feature_input(state.fence(), position, sample, client))
                .map_err(|error| error.to_string())?,
        );
        dyn_ns = dyn_ns.saturating_add(duration_ns(dyn_start));
    }
    Ok((duration_ns(total_start), atlas_ns, dyn_ns))
}

fn measure_performance(
    artifact: &AtlasArtifact,
    states: &[DynState],
    sample_count: usize,
) -> AppResult<PerformanceEvidence> {
    let positions: Vec<_> = artifact
        .l2
        .iter()
        .map(|cell| artifact.origin.center(cell.index, AtlasLevel::L2))
        .collect();
    if positions.is_empty() {
        return Err("Atlas has no representative L2 positions".to_owned());
    }
    for sample in 0..WARMUP_SAMPLES {
        black_box(exercise_sample(artifact, states, &positions, sample)?);
    }
    let mut total = Vec::with_capacity(sample_count);
    let mut atlas = Vec::with_capacity(sample_count);
    let mut dyn_feature = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        let measured = exercise_sample(artifact, states, &positions, sample + WARMUP_SAMPLES)?;
        total.push(measured.0);
        atlas.push(measured.1);
        dyn_feature.push(measured.2);
    }
    let total = distribution(total);
    Ok(PerformanceEvidence {
        scope: "one accepted resident transition: exact admitted Atlas L2 lookup plus 24-float Dyn feature assembly for clients 0..3",
        resident_samples: sample_count,
        warmup_samples: WARMUP_SAMPLES,
        clients_per_sample: CLIENT_COUNT,
        atlas_lookup: distribution(atlas),
        dyn_feature_assembly: distribution(dyn_feature),
        total,
        total_p99_limit_ns: TOTAL_P99_LIMIT_NS,
        total_p99_passed: total.p99_ns < TOTAL_P99_LIMIT_NS,
        feature_width: DYN_FEATURE_WIDTH,
    })
}

fn host_evidence() -> AppResult<HostEvidence> {
    Ok(HostEvidence {
        hostname: read_trimmed(Path::new("/proc/sys/kernel/hostname"), "hostname")?,
        kernel_release: read_trimmed(Path::new("/proc/sys/kernel/osrelease"), "kernel release")?,
        architecture: std::env::consts::ARCH,
    })
}

fn read_trimmed(path: &Path, label: &str) -> AppResult<String> {
    let value = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(format!("{label} is empty"));
    }
    Ok(trimmed.to_owned())
}

fn executable_evidence() -> AppResult<(PathBuf, Vec<u8>)> {
    let path = std::env::current_exe()
        .map_err(|error| format!("cannot locate current executable: {error}"))?;
    let bytes = fs::read(&path)
        .map_err(|error| format!("cannot hash executable {}: {error}", path.display()))?;
    Ok((path, bytes))
}

fn write_new(path: &Path, bytes: &[u8]) -> AppResult<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("cannot sync {}: {error}", path.display()))
}

fn sync_directory(path: &Path) -> AppResult<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("cannot sync directory {}: {error}", path.display()))
}

fn execute(arguments: Arguments) -> AppResult<(PathBuf, bool)> {
    let embedded_commit = BUILD_COMMIT.ok_or_else(|| {
        "binary lacks embedded Q2_LATTICE_CRATE_COMMIT; rebuild with the documented environment variable"
            .to_owned()
    })?;
    validate_commit(embedded_commit)?;
    if embedded_commit != arguments.expected_crate_commit {
        return Err(format!(
            "embedded crate commit {embedded_commit} != expected {}",
            arguments.expected_crate_commit
        ));
    }
    let repo_root = fs::canonicalize(&arguments.repo_root).map_err(|error| {
        format!(
            "cannot canonicalize repository root {}: {error}",
            arguments.repo_root.display()
        )
    })?;
    if !repo_root.is_dir() {
        return Err(format!(
            "repository root is not a directory: {}",
            repo_root.display()
        ));
    }
    let helper_source_closure = source_closure_evidence(
        &repo_root,
        helper_source_paths(&repo_root)?,
        BUILD_HELPER_SOURCE_SHA256,
        embedded_commit,
        "helper",
    )?;
    let q2_lattice_source_closure = source_closure_evidence(
        &repo_root,
        lattice_source_paths(&repo_root)?,
        BUILD_LATTICE_SOURCE_SHA256,
        embedded_commit,
        "q2-lattice",
    )?;

    let atlas_limits = AtlasLimits::default();
    let manifest_bytes = read_bounded(
        &arguments.manifest,
        atlas_limits.max_manifest_bytes,
        "Atlas manifest",
    )?;
    let manifest = AtlasManifest::from_canonical_json(&manifest_bytes, &atlas_limits)
        .map_err(|error| error.to_string())?;
    if manifest.specification_sha256 != DESIGN_SHA256 {
        return Err(format!(
            "manifest specification {} != authoritative design {DESIGN_SHA256}",
            manifest.specification_sha256
        ));
    }
    if manifest.bsp.canonical_map_id != arguments.expected_map_id {
        return Err(format!(
            "manifest map {} != expected {}",
            manifest.bsp.canonical_map_id, arguments.expected_map_id
        ));
    }
    if manifest.analyzer.sha256 != arguments.expected_analyzer_authority {
        return Err(format!(
            "manifest analyzer authority {} != expected {}",
            manifest.analyzer.sha256, arguments.expected_analyzer_authority
        ));
    }
    let artifact_name = sole_atlas_artifact_name(&manifest)?;
    let supplied_name = arguments
        .atlas
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "Atlas path requires a UTF-8 filename".to_owned())?;
    if supplied_name != artifact_name {
        return Err(format!(
            "supplied Atlas filename {supplied_name} != manifest artifact {artifact_name}"
        ));
    }
    let atlas_bytes = read_bounded(
        &arguments.atlas,
        atlas_limits.max_atlas_decompressed_bytes,
        "raw Atlas",
    )?;
    let artifact = manifest
        .decode_and_verify_atlas_artifact(&artifact_name, &atlas_bytes, &atlas_limits)
        .map_err(|error| error.to_string())?;
    if artifact.origin.0 != arguments.expected_origin {
        return Err(format!(
            "Atlas origin {:?} != expected {:?}",
            artifact.origin.0, arguments.expected_origin
        ));
    }
    let bsp_maximum = usize::try_from(manifest.bsp.size_bytes)
        .map_err(|_| "manifest BSP size does not fit usize".to_owned())?;
    let bsp_bytes = read_bounded(&arguments.bsp, bsp_maximum, "BSP")?;
    if bsp_bytes.len() as u64 != manifest.bsp.size_bytes
        || sha256_hex(&bsp_bytes) != manifest.bsp.sha256
    {
        return Err("supplied BSP bytes do not match the admitted manifest identity".to_owned());
    }

    let fence = DynFence {
        atlas_sha256: parse_digest(&sha256_hex(&atlas_bytes), "Atlas SHA-256")?,
        map_sha256: parse_digest(&manifest.bsp.sha256, "BSP SHA-256")?,
        origin: AtlasOrigin(arguments.expected_origin),
        map_epoch: arguments.map_epoch,
    };
    fence.validate().map_err(|error| error.to_string())?;
    let dyn_limits = DynLimits::default();
    let states = build_states(&artifact, fence, arguments.environment_steps, &dyn_limits)?;
    let snapshots: Vec<_> = states
        .iter()
        .map(|state| encode_snapshot(state, &dyn_limits).map_err(|error| error.to_string()))
        .collect::<AppResult<_>>()?;
    let roundtrips: Vec<_> = snapshots
        .iter()
        .map(|snapshot| {
            let restored =
                decode_snapshot(snapshot, fence, &dyn_limits).map_err(|error| error.to_string())?;
            encode_snapshot(&restored, &dyn_limits)
                .map(|encoded| encoded == *snapshot)
                .map_err(|error| error.to_string())
        })
        .collect::<AppResult<_>>()?;
    if roundtrips.iter().any(|passed| !passed) {
        return Err("Q2LAT002 byte-identical round-trip failed".to_owned());
    }
    let refs: Vec<_> = snapshots.iter().map(Vec::as_slice).collect();
    let batch = DynBatch::decode(
        &refs,
        BatchExpectation {
            fence,
            client_count: CLIENT_COUNT,
            environment_steps: arguments.environment_steps,
        },
        &dyn_limits,
    )
    .map_err(|error| error.to_string())?;
    let combined_compressed = batch.report.compressed_bytes;
    let combined_resident = batch.report.resident_bytes;
    let combined_budgets_passed =
        combined_compressed < FOUR_CLIENT_BYTE_LIMIT && combined_resident < FOUR_CLIENT_BYTE_LIMIT;
    let negatives = run_negative_checks(
        &snapshots,
        &batch.states,
        fence,
        arguments.environment_steps,
        &dyn_limits,
    )?;
    let performance = measure_performance(&artifact, &batch.states, arguments.samples)?;

    let staging = StagingDirectory::create(&arguments.output)?;
    let mut snapshot_reports = Vec::with_capacity(CLIENT_COUNT as usize);
    for (client, snapshot) in snapshots.iter().enumerate() {
        if !snapshot.starts_with(DYN_MAGIC) {
            return Err(format!("client {client} snapshot lacks Q2LAT002 magic"));
        }
        let filename = format!("client{client}.q2lat002");
        let path = staging.path.join(&filename);
        write_new(&path, snapshot)?;
        snapshot_reports.push(SnapshotEvidence {
            client_id: client as u32,
            file: FileEvidence {
                path: filename,
                sha256: sha256_hex(snapshot),
                size_bytes: snapshot.len() as u64,
            },
            magic: String::from_utf8_lossy(DYN_MAGIC).into_owned(),
            schema_version: DYN_SCHEMA_VERSION,
            l2_cells: batch.states[client].l2_len(),
            l3_cells: batch.states[client].l3_len(),
            resident_bytes: batch.states[client].resident_bytes_estimate(),
            byte_identical_roundtrip: roundtrips[client],
        });
    }
    let (executable_path, executable_bytes) = executable_evidence()?;
    let executable_file = file_evidence(&executable_path, &executable_bytes, "helper executable")?;
    let passed = negatives.passed()
        && combined_budgets_passed
        && performance.total_p99_passed
        && batch
            .states
            .iter()
            .map(DynState::client_id)
            .eq(0..CLIENT_COUNT);
    let report = EvidenceReport {
        schema: REPORT_SCHEMA,
        passed,
        authority: AuthorityEvidence {
            specification_sha256: manifest.specification_sha256.clone(),
            analyzer_name: manifest.analyzer.name.clone(),
            analyzer_version: manifest.analyzer.version.clone(),
            analyzer_authority_sha256: manifest.analyzer.sha256.clone(),
            crate_commit: embedded_commit.to_owned(),
            executable_sha256: executable_file.sha256.clone(),
            canonical_map_id: manifest.bsp.canonical_map_id.clone(),
            map_epoch: arguments.map_epoch,
            environment_steps: arguments.environment_steps,
        },
        provenance: ProvenanceEvidence {
            embedded_repo_commit: embedded_commit.to_owned(),
            executable: executable_file,
            helper_source_closure,
            q2_lattice_source_closure,
        },
        host: host_evidence()?,
        atlas: AtlasEvidence {
            manifest: file_evidence(&arguments.manifest, &manifest_bytes, "Atlas manifest")?,
            artifact: file_evidence(&arguments.atlas, &atlas_bytes, "Atlas")?,
            bsp: file_evidence(&arguments.bsp, &bsp_bytes, "BSP")?,
            origin: artifact.origin.0,
            counts: AtlasCounts::from_artifact(&artifact),
            resident_bytes: artifact.resident_bytes_estimate(),
            representative_l2_cells: batch.states[0].l2_len(),
            lookup: "origin-indexed exact L2 aggregate binary search in the admitted resident Atlas",
        },
        dyn_state: DynEvidence {
            snapshot_magic: String::from_utf8_lossy(DYN_MAGIC).into_owned(),
            schema_version: DYN_SCHEMA_VERSION,
            client_ids: batch.states.iter().map(DynState::client_id).collect(),
            client_count: CLIENT_COUNT,
            common_environment_steps: arguments.environment_steps,
            population: "deterministic per-client representative channels over admitted Atlas L2 cells; authority identities are never synthetic",
            snapshots: snapshot_reports,
            combined_compressed_bytes: combined_compressed,
            combined_resident_bytes: combined_resident,
            combined_limit_bytes: FOUR_CLIENT_BYTE_LIMIT,
            batch_ids_and_step_admitted: true,
        },
        negative_fences_and_limits: negatives,
        performance,
    };
    // Serialize through Value: serde_json's default Map is a BTreeMap, making
    // every object key order canonical rather than retaining Rust struct order.
    let canonical_report = serde_json::to_value(&report).map_err(|error| error.to_string())?;
    let mut report_bytes =
        serde_json::to_vec(&canonical_report).map_err(|error| error.to_string())?;
    report_bytes.push(b'\n');
    write_new(&staging.path.join(REPORT_NAME), &report_bytes)?;
    staging.publish(&arguments.output)?;
    let report_path = arguments.output.join(REPORT_NAME);
    eprintln!(
        "q2-dyn-evidence: executable={} report={} passed={passed}",
        executable_path.display(),
        report_path.display()
    );
    Ok((report_path, passed))
}

fn main() {
    let arguments = match parse_arguments(&std::env::args_os().collect::<Vec<_>>()) {
        Ok(arguments) => arguments,
        Err(error) => {
            eprintln!("q2-dyn-evidence: {error}");
            std::process::exit(64);
        }
    };
    match execute(arguments) {
        Ok((path, true)) => println!("{}", path.display()),
        Ok((path, false)) => {
            eprintln!(
                "q2-dyn-evidence: gate failed; evidence written to {}",
                path.display()
            );
            std::process::exit(1);
        }
        Err(error) => {
            eprintln!("q2-dyn-evidence: {error}");
            std::process::exit(65);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use q2_lattice_rs::atlas::GridIndex;

    use super::*;

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn complete_arguments() -> Vec<OsString> {
        [
            "q2-dyn-evidence",
            "--repo-root",
            "/repo/q2-ml-bot",
            "--atlas",
            "map.atlas.bin",
            "--manifest",
            "map.atlas.manifest.json",
            "--bsp",
            "map.bsp",
            "--output",
            "evidence",
            "--expected-map-id",
            "map",
            "--expected-origin",
            "-256,0,512",
            "--expected-analyzer-authority",
            &"11".repeat(32),
            "--expected-crate-commit",
            &"22".repeat(20),
            "--map-epoch",
            "17",
            "--environment-steps",
            "4096",
            "--samples",
            "2000",
        ]
        .into_iter()
        .map(OsString::from)
        .collect()
    }

    #[test]
    fn cli_requires_explicit_authority_origin_epoch_and_minimum_samples() {
        let parsed = parse_arguments(&complete_arguments()).unwrap();
        assert_eq!(parsed.expected_origin, [-256, 0, 512]);
        assert_eq!(parsed.map_epoch, 17);
        assert_eq!(parsed.samples, MIN_SAMPLES);

        let mut too_few = complete_arguments();
        let ordinal = too_few
            .iter()
            .position(|value| value == "--samples")
            .unwrap();
        too_few[ordinal + 1] = "1999".into();
        assert!(parse_arguments(&too_few).unwrap_err().contains("at least"));
    }

    #[test]
    fn percentile_uses_nearest_rank_without_underflow() {
        let values: Vec<_> = (1..=2_000).collect();
        assert_eq!(percentile(&values, 50, 100), 1_000);
        assert_eq!(percentile(&values, 95, 100), 1_900);
        assert_eq!(percentile(&values, 99, 100), 1_980);
        assert_eq!(percentile(&[7], 99, 100), 7);
    }

    #[test]
    fn digests_are_canonical_lowercase_and_round_trip() {
        let text = "a5".repeat(32);
        assert_eq!(parse_digest(&text, "test").unwrap(), [0xa5; 32]);
        assert!(parse_digest(&"A5".repeat(32), "test").is_err());
        assert!(parse_digest("00", "test").is_err());
    }

    #[test]
    fn output_directory_is_exclusive_and_staging_is_cleaned() {
        let ordinal = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "q2-dyn-evidence-test-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let output = root.join("evidence");
        {
            let staging = StagingDirectory::create(&output).unwrap();
            assert!(staging.path.is_dir());
        }
        assert_eq!(fs::read_dir(&root).unwrap().count(), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn no_replace_publication_preserves_a_racing_destination() {
        let ordinal = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "q2-dyn-evidence-noreplace-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let output = root.join("evidence");
        let staging = StagingDirectory::create(&output).unwrap();
        write_new(&staging.path.join("candidate"), b"candidate").unwrap();
        fs::create_dir(&output).unwrap();
        write_new(&output.join("winner"), b"existing").unwrap();
        let error = staging.publish(&output).unwrap_err();
        assert!(error.contains("File exists"));
        assert_eq!(fs::read(output.join("winner")).unwrap(), b"existing");
        assert!(!output.join("candidate").exists());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn two_publishers_racing_for_one_output_have_exactly_one_winner() {
        use std::sync::{Arc, Barrier};

        let ordinal = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "q2-dyn-evidence-race-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let output = root.join("evidence");
        let first = StagingDirectory::create(&output).unwrap();
        let second = StagingDirectory::create(&output).unwrap();
        write_new(&first.path.join("publisher"), b"first").unwrap();
        write_new(&second.path.join("publisher"), b"second").unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = [first, second]
            .into_iter()
            .map(|staging| {
                let barrier = Arc::clone(&barrier);
                let output = output.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    staging.publish(&output)
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
        assert!(matches!(
            fs::read(output.join("publisher")).unwrap().as_slice(),
            b"first" | b"second"
        ));
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn relative_output_uses_the_current_directory_as_parent() {
        assert_eq!(output_parent(Path::new("evidence")), Path::new("."));
        assert_eq!(
            output_parent(Path::new("campaign/evidence")),
            Path::new("campaign")
        );
    }

    #[test]
    fn exact_l2_lookup_accepts_only_a_real_resident_key() {
        let limits = AtlasLimits::default();
        let mut artifact = AtlasArtifact::empty(AtlasOrigin([0, 0, 0]));
        artifact.l2.push(AtlasAggregateCell {
            index: GridIndex::new(1, 2, 3),
            contents_flags: 0,
            hazard_types: 0,
            hazard_severity: 0,
            standing_passable: true,
            crouched_passable: true,
            clearance: 64,
            cost_to_safety: 0,
            confidence: u16::MAX,
        });
        artifact.validate(&limits).unwrap();
        let position = artifact
            .origin
            .center(GridIndex::new(1, 2, 3), AtlasLevel::L2);
        assert_eq!(
            atlas_lookup(&artifact, position).unwrap().index,
            GridIndex::new(1, 2, 3)
        );
        assert!(atlas_lookup(&artifact, [0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn expanded_negative_matrix_rejects_every_stale_or_corrupt_variant() {
        let limits = DynLimits::default();
        let fence = DynFence {
            atlas_sha256: [0xa5; 32],
            map_sha256: [0x5a; 32],
            origin: AtlasOrigin([-256, -512, 0]),
            map_epoch: 17,
        };
        let mut states = Vec::new();
        for client_id in 0..CLIENT_COUNT {
            let mut state = DynState::new(fence, client_id, CLIENT_COUNT, 4096, &limits).unwrap();
            state
                .upsert_l2(
                    fence,
                    GridIndex::new(client_id as i32, 0, 0),
                    DynCell::new(
                        PersistentChannels {
                            engagement: 1.0,
                            threat: 2.0,
                            opportunity: 3.0,
                            self_fire: 4.0,
                            deaths: 5.0,
                        },
                        6.0,
                        0.75,
                    )
                    .unwrap(),
                    &limits,
                )
                .unwrap();
            states.push(state);
        }
        let snapshots: Vec<_> = states
            .iter()
            .map(|state| encode_snapshot(state, &limits).unwrap())
            .collect();
        let evidence = run_negative_checks(&snapshots, &states, fence, 4096, &limits).unwrap();
        assert!(evidence.passed());
        assert!(evidence.materialized_cell_limit_rejected);
        assert!(evidence.payload_digest_corruption_rejected);
        assert!(evidence.cell_size_mismatch_rejected);
    }

    #[test]
    fn runtime_source_closures_match_the_build_time_embeds() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let repo_root = manifest_dir
            .parent()
            .and_then(Path::parent)
            .unwrap()
            .to_path_buf();
        let helper = source_closure_evidence(
            &repo_root,
            helper_source_paths(&repo_root).unwrap(),
            BUILD_HELPER_SOURCE_SHA256,
            &"11".repeat(20),
            "helper",
        )
        .unwrap();
        let lattice = source_closure_evidence(
            &repo_root,
            lattice_source_paths(&repo_root).unwrap(),
            BUILD_LATTICE_SOURCE_SHA256,
            &"11".repeat(20),
            "q2-lattice",
        )
        .unwrap();
        assert_eq!(helper.sha256, helper.embedded_sha256);
        assert_eq!(lattice.sha256, lattice.embedded_sha256);
        assert!(
            helper
                .inputs
                .iter()
                .any(|input| input.path.ends_with("src/main.rs"))
        );
        assert!(
            lattice
                .inputs
                .iter()
                .any(|input| input.path.ends_with("src/dynstate.rs"))
        );
    }

    #[test]
    fn report_serialization_primitive_is_compact_and_recursively_key_sorted() {
        #[derive(Serialize)]
        struct OutOfOrder {
            zebra: u8,
            alpha: Nested,
        }

        #[derive(Serialize)]
        struct Nested {
            yellow: u8,
            beta: u8,
        }

        let value = serde_json::to_value(OutOfOrder {
            zebra: 1,
            alpha: Nested { yellow: 2, beta: 3 },
        })
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&value).unwrap(),
            br#"{"alpha":{"beta":3,"yellow":2},"zebra":1}"#
        );
    }
}
