use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::admission::validate_digest;
use super::{
    ATLAS_CELL_SIZES, ATLAS_MAGIC, ATLAS_SCHEMA_VERSION, AtlasArtifact, AtlasError, AtlasLevel,
    AtlasLimits, AtlasOrigin, AtlasResult, HOOK_RECOVERY_WALK_BUDGET_TICKS, OracleAdmissions,
    RECOVERY_GAME_TICK_HZ, RECOVERY_WALK_SPEED_Q8_PER_SECOND,
};

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolIdentity {
    pub name: String,
    pub version: String,
    pub sha256: String,
}

impl ToolIdentity {
    fn validate(&self, field: &str) -> AtlasResult<()> {
        if self.name.is_empty() || self.version.is_empty() {
            return Err(AtlasError::InvalidFormat(format!(
                "{field} identity has an empty name/version"
            )));
        }
        validate_digest(field, &self.sha256)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BspIdentity {
    pub canonical_map_id: String,
    pub sha256: String,
    pub provenance_sha256: String,
    pub size_bytes: u64,
    pub ibsp_version: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridManifest {
    pub origin: [i64; 3],
    pub model0_mins: [i64; 3],
    pub model0_maxs: [i64; 3],
    pub cell_sizes: [u32; 4],
    pub l0_chunk_dimensions: [u16; 3],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HullManifest {
    pub name: String,
    pub mins: [i16; 3],
    pub maxs: [i16; 3],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChannelManifest {
    pub store: String,
    pub level: u8,
    pub name: String,
    pub encoding: String,
    pub persistence: String,
}

impl ChannelManifest {
    fn key(&self) -> (&str, u8, &str) {
        (&self.store, self.level, &self.name)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub media_type: String,
    pub sha256_uncompressed: String,
    pub uncompressed_size: u64,
    pub compressed_size: u64,
    pub counts: BTreeMap<String, u64>,
}

impl ArtifactManifest {
    pub fn from_uncompressed(
        media_type: impl Into<String>,
        payload: &[u8],
        compressed_size: u64,
        counts: BTreeMap<String, u64>,
    ) -> Self {
        Self {
            media_type: media_type.into(),
            sha256_uncompressed: sha256_hex(payload),
            uncompressed_size: payload.len() as u64,
            compressed_size,
            counts,
        }
    }

    fn validate(&self, name: &str, limits: &AtlasLimits) -> AtlasResult<()> {
        if self.media_type.is_empty() {
            return Err(AtlasError::InvalidFormat(format!(
                "artifact {name} has empty media type"
            )));
        }
        validate_digest(name, &self.sha256_uncompressed)?;
        if self.uncompressed_size > limits.max_atlas_decompressed_bytes as u64 {
            return Err(AtlasError::LimitExceeded(format!(
                "artifact {name} uncompressed bytes {} > {}",
                self.uncompressed_size, limits.max_atlas_decompressed_bytes
            )));
        }
        if self.compressed_size > limits.max_compressed_payload_bytes as u64 {
            return Err(AtlasError::LimitExceeded(format!(
                "artifact {name} compressed bytes {} > {}",
                self.compressed_size, limits.max_compressed_payload_bytes
            )));
        }
        Ok(())
    }

    pub fn verify_uncompressed(&self, payload: &[u8]) -> AtlasResult<()> {
        if self.uncompressed_size != payload.len() as u64
            || self.sha256_uncompressed != sha256_hex(payload)
        {
            return Err(AtlasError::DigestMismatch);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtlasCounts {
    pub l0_chunks: u64,
    pub l1_nodes: u64,
    pub l1_edges: u64,
    pub l2_cells: u64,
    pub l3_cells: u64,
}

impl AtlasCounts {
    pub fn from_artifact(artifact: &AtlasArtifact) -> Self {
        Self {
            l0_chunks: artifact.l0.len() as u64,
            l1_nodes: artifact.l1.nodes().len() as u64,
            l1_edges: artifact.l1.edges().len() as u64,
            l2_cells: artifact.l2.len() as u64,
            l3_cells: artifact.l3.len() as u64,
        }
    }

    pub fn named_counts(self) -> BTreeMap<String, u64> {
        BTreeMap::from([
            ("l0_chunks".to_owned(), self.l0_chunks),
            ("l1_nodes".to_owned(), self.l1_nodes),
            ("l1_edges".to_owned(), self.l1_edges),
            ("l2_cells".to_owned(), self.l2_cells),
            ("l3_cells".to_owned(), self.l3_cells),
        ])
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestBudgets {
    pub max_l0_chunks: u64,
    pub max_l0_decompressed_bytes: u64,
    pub max_atlas_decompressed_bytes: u64,
    pub max_atlas_resident_bytes: u64,
    pub max_build_rss_bytes: u64,
}

/// Physics/cadence identity required by the teacher-only hook-necessity
/// evaluator. Mixed or missing values make the Atlas inadmissible.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryPhysicsManifest {
    pub hook_walk_budget_ticks: u32,
    pub game_tick_hz: u32,
    pub walk_speed_q8_per_second: u32,
}

impl Default for RecoveryPhysicsManifest {
    fn default() -> Self {
        Self {
            hook_walk_budget_ticks: HOOK_RECOVERY_WALK_BUDGET_TICKS,
            game_tick_hz: RECOVERY_GAME_TICK_HZ,
            walk_speed_q8_per_second: RECOVERY_WALK_SPEED_Q8_PER_SECOND,
        }
    }
}

impl From<&AtlasLimits> for ManifestBudgets {
    fn from(limits: &AtlasLimits) -> Self {
        Self {
            max_l0_chunks: limits.max_l0_chunks as u64,
            max_l0_decompressed_bytes: limits.max_l0_decompressed_bytes as u64,
            max_atlas_decompressed_bytes: limits.max_atlas_decompressed_bytes as u64,
            max_atlas_resident_bytes: limits.max_atlas_resident_bytes as u64,
            max_build_rss_bytes: limits.max_build_rss_bytes,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AtlasManifest {
    pub schema_version: u16,
    pub byte_order: String,
    pub atlas_magic: String,
    pub specification_sha256: String,
    pub bsp: BspIdentity,
    pub analyzer: ToolIdentity,
    pub oracles: OracleAdmissions,
    pub recovery_physics: RecoveryPhysicsManifest,
    pub generator: Option<ToolIdentity>,
    pub grid: GridManifest,
    pub player_hulls: Vec<HullManifest>,
    pub channels: Vec<ChannelManifest>,
    pub artifacts: BTreeMap<String, ArtifactManifest>,
    pub counts: AtlasCounts,
    pub budgets: ManifestBudgets,
    pub build_peak_rss_bytes: u64,
    pub limitations: Vec<String>,
    pub confidence_summary: String,
}

impl AtlasManifest {
    pub fn canonical_json(&self, limits: &AtlasLimits) -> AtlasResult<Vec<u8>> {
        let canonical = self.canonicalized();
        canonical.validate(limits)?;
        let mut bytes = serde_json::to_vec(&canonical)?;
        bytes.push(b'\n');
        if bytes.len() > limits.max_manifest_bytes {
            return Err(AtlasError::LimitExceeded(format!(
                "manifest bytes {} > {}",
                bytes.len(),
                limits.max_manifest_bytes
            )));
        }
        Ok(bytes)
    }

    pub fn from_canonical_json(bytes: &[u8], limits: &AtlasLimits) -> AtlasResult<Self> {
        if bytes.len() > limits.max_manifest_bytes {
            return Err(AtlasError::LimitExceeded(format!(
                "manifest bytes {} > {}",
                bytes.len(),
                limits.max_manifest_bytes
            )));
        }
        let manifest: Self = serde_json::from_slice(bytes)?;
        manifest.validate(limits)?;
        if manifest.canonical_json(limits)? != bytes {
            return Err(AtlasError::InvalidFormat(
                "manifest JSON is not canonical".to_owned(),
            ));
        }
        Ok(manifest)
    }

    pub fn validate(&self, limits: &AtlasLimits) -> AtlasResult<()> {
        if self.schema_version != ATLAS_SCHEMA_VERSION {
            return Err(AtlasError::MixedSchema {
                expected: ATLAS_SCHEMA_VERSION,
                found: self.schema_version,
            });
        }
        if self.byte_order != "little" || self.atlas_magic.as_bytes() != ATLAS_MAGIC {
            return Err(AtlasError::InvalidFormat(
                "manifest byte order or Atlas magic mismatch".to_owned(),
            ));
        }
        validate_digest("specification", &self.specification_sha256)?;
        if self.bsp.canonical_map_id.is_empty()
            || self.bsp.size_bytes == 0
            || self.bsp.ibsp_version != 38
        {
            return Err(AtlasError::InvalidFormat(
                "invalid IBSP-38 identity".to_owned(),
            ));
        }
        validate_digest("BSP", &self.bsp.sha256)?;
        validate_digest("BSP provenance", &self.bsp.provenance_sha256)?;
        self.analyzer.validate("analyzer")?;
        self.oracles.admit(&self.bsp)?;
        if self.recovery_physics != RecoveryPhysicsManifest::default() {
            return Err(AtlasError::InvalidFormat(
                "manifest recovery physics/cadence identity differs".to_owned(),
            ));
        }
        let pmove = self.oracles.pmove_oracle.as_ref().ok_or_else(|| {
            AtlasError::InvalidFormat(
                "manifest recovery physics requires Pmove authority".to_owned(),
            )
        })?;
        if !pmove
            .parameters
            .constants
            .split(',')
            .any(|item| item == "max=300")
        {
            return Err(AtlasError::InvalidFormat(
                "manifest recovery walk speed differs from Pmove maxspeed".to_owned(),
            ));
        }
        if let Some(generator) = &self.generator {
            generator.validate("generator")?;
        }
        let expected_origin = AtlasOrigin::snapped(self.grid.model0_mins)?;
        if self.grid.cell_sizes != ATLAS_CELL_SIZES.map(|value| value as u32)
            || self.grid.l0_chunk_dimensions != [16, 16, 16]
            || self.grid.origin != expected_origin.0
            || self
                .grid
                .model0_mins
                .iter()
                .zip(self.grid.model0_maxs)
                .any(|(min, max)| *min > max)
        {
            return Err(AtlasError::InvalidFormat(
                "manifest grid contract mismatch".to_owned(),
            ));
        }
        for level in AtlasLevel::ALL {
            expected_origin.index_integer(self.grid.model0_mins, level)?;
            expected_origin.index_integer(self.grid.model0_maxs, level)?;
        }
        let expected_hulls = [
            ("standing", [-16, -16, -24], [16, 16, 32]),
            ("crouched", [-16, -16, -24], [16, 16, 4]),
        ];
        if self.player_hulls.len() != expected_hulls.len()
            || self
                .player_hulls
                .iter()
                .zip(expected_hulls)
                .any(|(actual, expected)| {
                    actual.name != expected.0
                        || actual.mins != expected.1
                        || actual.maxs != expected.2
                })
        {
            return Err(AtlasError::InvalidFormat(
                "manifest player hulls do not match Quake II v1 hulls".to_owned(),
            ));
        }
        if self
            .channels
            .windows(2)
            .any(|pair| pair[0].key() >= pair[1].key())
        {
            return Err(AtlasError::InvalidFormat(
                "manifest channels are not canonically ordered/unique".to_owned(),
            ));
        }
        for channel in &self.channels {
            if channel.store != "Atlas"
                || channel.level > 3
                || channel.name.is_empty()
                || channel.encoding.is_empty()
                || channel.persistence != "map-static"
            {
                return Err(AtlasError::InvalidFormat(format!(
                    "invalid channel {}",
                    channel.name
                )));
            }
        }
        if self.artifacts.is_empty() {
            return Err(AtlasError::InvalidFormat(
                "manifest has no artifacts".to_owned(),
            ));
        }
        for (name, artifact) in &self.artifacts {
            artifact.validate(name, limits)?;
        }
        validate_count(self.counts.l0_chunks, limits.max_l0_chunks, "L0 chunks")?;
        validate_count(self.counts.l1_nodes, limits.max_l1_nodes, "L1 nodes")?;
        validate_count(self.counts.l1_edges, limits.max_l1_edges, "L1 edges")?;
        validate_count(self.counts.l2_cells, limits.max_l2_cells, "L2 cells")?;
        validate_count(self.counts.l3_cells, limits.max_l3_cells, "L3 cells")?;
        let expected_budgets = ManifestBudgets::from(limits);
        if self.budgets != expected_budgets {
            return Err(AtlasError::InvalidFormat(
                "manifest budgets do not match the admission limits".to_owned(),
            ));
        }
        if self.build_peak_rss_bytes > limits.max_build_rss_bytes {
            return Err(AtlasError::LimitExceeded(format!(
                "build peak RSS {} > {}",
                self.build_peak_rss_bytes, limits.max_build_rss_bytes
            )));
        }
        if self.limitations.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(AtlasError::InvalidFormat(
                "manifest limitations are not sorted/unique".to_owned(),
            ));
        }
        if self.confidence_summary.is_empty() {
            return Err(AtlasError::InvalidFormat(
                "manifest confidence summary is empty".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn verify_atlas_artifact(
        &self,
        artifact_name: &str,
        payload: &[u8],
        artifact: &AtlasArtifact,
        limits: &AtlasLimits,
    ) -> AtlasResult<()> {
        self.validate(limits)?;
        artifact.validate(limits)?;
        let identity = self.artifacts.get(artifact_name).ok_or_else(|| {
            AtlasError::InvalidFormat(format!(
                "manifest has no identity for artifact {artifact_name}"
            ))
        })?;
        identity.verify_uncompressed(payload)?;
        let encoded = artifact.encode_uncompressed(limits)?;
        if encoded != payload {
            return Err(AtlasError::InvalidFormat(
                "decoded Atlas artifact does not match the attested payload".to_owned(),
            ));
        }
        if artifact.origin.0 != self.grid.origin {
            return Err(AtlasError::InvalidFormat(
                "decoded Atlas origin does not match the manifest grid".to_owned(),
            ));
        }
        let admission = self.oracles.admit(&self.bsp)?;
        artifact.l1.validate(&admission, limits)?;
        let artifact_counts = AtlasCounts::from_artifact(artifact);
        if self.counts != artifact_counts {
            return Err(AtlasError::InvalidFormat(
                "manifest Atlas counts do not match decoded artifact".to_owned(),
            ));
        }
        if identity.counts != artifact_counts.named_counts() {
            return Err(AtlasError::InvalidFormat(
                "artifact-local counts do not match decoded Atlas artifact".to_owned(),
            ));
        }
        Ok(())
    }

    /// Decode and admit an Atlas payload as one operation. Raw Atlas decoding
    /// is structural only because the canonical manifest is a separate file.
    pub fn decode_and_verify_atlas_artifact(
        &self,
        artifact_name: &str,
        payload: &[u8],
        limits: &AtlasLimits,
    ) -> AtlasResult<AtlasArtifact> {
        let artifact = AtlasArtifact::decode_uncompressed(payload, limits)?;
        self.verify_atlas_artifact(artifact_name, payload, &artifact, limits)?;
        Ok(artifact)
    }

    fn canonicalized(&self) -> Self {
        let mut output = self.clone();
        output
            .channels
            .sort_by(|left, right| left.key().cmp(&right.key()));
        output.limitations.sort();
        output
    }
}

fn validate_count(value: u64, maximum: usize, name: &str) -> AtlasResult<()> {
    if value > maximum as u64 {
        return Err(AtlasError::LimitExceeded(format!(
            "{name} {value} > {maximum}"
        )));
    }
    Ok(())
}
