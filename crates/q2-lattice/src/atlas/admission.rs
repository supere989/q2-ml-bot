use serde::{Deserialize, Serialize};

use super::{AtlasError, AtlasResult, BspIdentity, EdgeType, sha256_hex};

pub const ORACLE_SEMANTIC_VERSION: u16 = 1;
pub const COLLISION_ORACLE_NAME: &str = "q2-cm-oracle";
pub const COLLISION_ORACLE_SCHEMA: &str = "q2-cm-oracle-v1";
pub const PMOVE_ORACLE_NAME: &str = "q2-pmove-oracle";
pub const PMOVE_ORACLE_SCHEMA: &str = "q2-pmove-oracle-v1";
pub const HOOK_ORACLE_NAME: &str = "q2-hook-oracle";
pub const HOOK_ORACLE_SCHEMA: &str = "q2-hook-oracle-v1";
pub const HOOK_PARITY_NAME: &str = "q2-hook-q2ded-parity";
pub const HOOK_PARITY_SCHEMA: &str = "q2-hook-parity-v1";
pub const HOOK_PARITY_CASES_V1: u32 = 8;
pub const B1_RUNTIME_AUTHORITY_SEAL_SCHEMA: &str = "q2-b1-runtime-authority-seal-v1";

pub const MASK_PLAYERSOLID_V1: u32 = 33_619_971;
pub const MASK_SHOT_V1: u32 = 100_663_299;

pub(crate) fn validate_digest(name: &str, digest: &str) -> AtlasResult<()> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || digest.bytes().all(|byte| byte == b'0')
    {
        return Err(AtlasError::InvalidFormat(format!(
            "{name} is not a nonzero lowercase SHA-256 digest"
        )));
    }
    Ok(())
}

fn validate_text(name: &str, value: &str) -> AtlasResult<()> {
    if value.is_empty()
        || value.len() > 4096
        || value.trim() != value
        || value.chars().any(char::is_control)
    {
        return Err(AtlasError::InvalidFormat(format!(
            "{name} is empty, noncanonical, or too large"
        )));
    }
    Ok(())
}

fn canonical_digest<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("oracle closure structs are always serializable");
    sha256_hex(&bytes)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleToolIdentity {
    pub name: String,
    pub schema: String,
    pub version: u16,
    pub executable_sha256: String,
    pub physics_identity_sha256: String,
}

impl OracleToolIdentity {
    fn validate(&self, expected_name: &str, expected_schema: &str, field: &str) -> AtlasResult<()> {
        if self.name != expected_name
            || self.schema != expected_schema
            || self.version != ORACLE_SEMANTIC_VERSION
        {
            return Err(AtlasError::InvalidFormat(format!(
                "{field} expected {expected_name}/{expected_schema}/v{ORACLE_SEMANTIC_VERSION}"
            )));
        }
        validate_digest(&format!("{field} executable"), &self.executable_sha256)?;
        validate_digest(
            &format!("{field} physics identity"),
            &self.physics_identity_sha256,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleBspBinding {
    pub sha256: String,
    pub provenance_sha256: String,
}

impl OracleBspBinding {
    fn validate(&self, bsp: &BspIdentity, field: &str) -> AtlasResult<()> {
        validate_digest(&format!("{field} BSP"), &self.sha256)?;
        validate_digest(&format!("{field} BSP provenance"), &self.provenance_sha256)?;
        if self.sha256 != bsp.sha256 || self.provenance_sha256 != bsp.provenance_sha256 {
            return Err(AtlasError::InvalidFormat(format!(
                "{field} is bound to a different BSP or provenance record"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollisionParameters {
    pub mask_playersolid: u32,
    pub mask_shot: u32,
}

impl CollisionParameters {
    fn validate(&self) -> AtlasResult<()> {
        if self.mask_playersolid != MASK_PLAYERSOLID_V1 || self.mask_shot != MASK_SHOT_V1 {
            return Err(AtlasError::InvalidFormat(
                "collision oracle mask contract mismatch".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollisionSourceClosure {
    pub collision_sha256: String,
    pub shared_header_sha256: String,
    pub shared_source_sha256: String,
    pub build_contract: String,
}

impl CollisionSourceClosure {
    fn validate(&self, field: &str) -> AtlasResult<()> {
        validate_digest(&format!("{field} collision source"), &self.collision_sha256)?;
        validate_digest(
            &format!("{field} shared header"),
            &self.shared_header_sha256,
        )?;
        validate_digest(
            &format!("{field} shared source"),
            &self.shared_source_sha256,
        )?;
        validate_text(&format!("{field} build contract"), &self.build_contract)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollisionOracleAdmission {
    pub tool: OracleToolIdentity,
    pub bsp: OracleBspBinding,
    pub parameters: CollisionParameters,
    pub source: CollisionSourceClosure,
    pub contract_sha256: String,
}

#[derive(Serialize)]
struct CollisionContractPayload<'a> {
    tool: &'a OracleToolIdentity,
    bsp: &'a OracleBspBinding,
    parameters: &'a CollisionParameters,
    source: &'a CollisionSourceClosure,
}

impl CollisionOracleAdmission {
    pub fn canonical_contract_sha256(&self) -> String {
        canonical_digest(&CollisionContractPayload {
            tool: &self.tool,
            bsp: &self.bsp,
            parameters: &self.parameters,
            source: &self.source,
        })
    }

    pub fn seal(mut self) -> Self {
        self.contract_sha256 = self.canonical_contract_sha256();
        self
    }

    fn validate(&self, bsp: &BspIdentity) -> AtlasResult<()> {
        self.tool.validate(
            COLLISION_ORACLE_NAME,
            COLLISION_ORACLE_SCHEMA,
            "collision oracle",
        )?;
        self.bsp.validate(bsp, "collision oracle")?;
        self.parameters.validate()?;
        self.source.validate("collision oracle")?;
        validate_digest("collision oracle contract", &self.contract_sha256)?;
        if self.contract_sha256 != self.canonical_contract_sha256() {
            return Err(AtlasError::InvalidFormat(
                "collision oracle semantic closure mismatch".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PmoveParameters {
    pub gravity: u16,
    /// Exact IEEE-754 binary32 representation returned by the pinned oracle.
    pub airaccelerate_f32_bits: u32,
    pub constants: String,
}

impl PmoveParameters {
    fn validate(&self) -> AtlasResult<()> {
        let airaccelerate = f32::from_bits(self.airaccelerate_f32_bits);
        if self.gravity > i16::MAX as u16 || !airaccelerate.is_finite() || airaccelerate < 0.0 {
            return Err(AtlasError::InvalidFormat(
                "pmove oracle parameters are outside the v1 engine range".to_owned(),
            ));
        }
        validate_text("pmove constants", &self.constants)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PmoveSourceClosure {
    pub collision_sha256: String,
    pub pmove_sha256: String,
    pub shared_header_sha256: String,
    pub shared_source_sha256: String,
    pub build_contract: String,
}

impl PmoveSourceClosure {
    fn validate(&self) -> AtlasResult<()> {
        validate_digest("pmove collision source", &self.collision_sha256)?;
        validate_digest("pmove source", &self.pmove_sha256)?;
        validate_digest("pmove shared header", &self.shared_header_sha256)?;
        validate_digest("pmove shared source", &self.shared_source_sha256)?;
        validate_text("pmove build contract", &self.build_contract)
    }

    fn validate_collision_closure(&self, collision: &CollisionSourceClosure) -> AtlasResult<()> {
        if self.collision_sha256 != collision.collision_sha256
            || self.shared_header_sha256 != collision.shared_header_sha256
            || self.shared_source_sha256 != collision.shared_source_sha256
            || self.build_contract != collision.build_contract
        {
            return Err(AtlasError::InvalidFormat(
                "pmove source closure does not match collision oracle".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PmoveOracleAdmission {
    pub tool: OracleToolIdentity,
    pub bsp: OracleBspBinding,
    pub parameters: PmoveParameters,
    pub source: PmoveSourceClosure,
    pub contract_sha256: String,
}

#[derive(Serialize)]
struct PmoveContractPayload<'a> {
    tool: &'a OracleToolIdentity,
    bsp: &'a OracleBspBinding,
    parameters: &'a PmoveParameters,
    source: &'a PmoveSourceClosure,
}

impl PmoveOracleAdmission {
    pub fn canonical_contract_sha256(&self) -> String {
        canonical_digest(&PmoveContractPayload {
            tool: &self.tool,
            bsp: &self.bsp,
            parameters: &self.parameters,
            source: &self.source,
        })
    }

    pub fn seal(mut self) -> Self {
        self.contract_sha256 = self.canonical_contract_sha256();
        self
    }

    fn validate(&self, bsp: &BspIdentity, collision: &CollisionOracleAdmission) -> AtlasResult<()> {
        self.tool
            .validate(PMOVE_ORACLE_NAME, PMOVE_ORACLE_SCHEMA, "pmove oracle")?;
        self.bsp.validate(bsp, "pmove oracle")?;
        self.parameters.validate()?;
        self.source.validate()?;
        self.source.validate_collision_closure(&collision.source)?;
        validate_digest("pmove oracle contract", &self.contract_sha256)?;
        if self.contract_sha256 != self.canonical_contract_sha256() {
            return Err(AtlasError::InvalidFormat(
                "pmove oracle semantic closure mismatch".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HookParameters {
    pub hook_speed_f32_bits: u32,
    pub hook_pullspeed_f32_bits: u32,
    pub hook_sky: bool,
    pub hook_maxtime_f32_bits: u32,
    pub full_velocity_overwrite: bool,
}

impl HookParameters {
    fn validate(&self) -> AtlasResult<()> {
        for (name, bits) in [
            ("hook_speed", self.hook_speed_f32_bits),
            ("hook_pullspeed", self.hook_pullspeed_f32_bits),
            ("hook_maxtime", self.hook_maxtime_f32_bits),
        ] {
            let value = f32::from_bits(bits);
            if !value.is_finite() || value <= 0.0 {
                return Err(AtlasError::InvalidFormat(format!(
                    "{name} is not a finite positive binary32 value"
                )));
            }
        }
        if !self.full_velocity_overwrite {
            return Err(AtlasError::InvalidFormat(
                "hook v1 requires full velocity overwrite".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HookSourceClosure {
    pub shared_c_sha256: String,
    pub shared_h_sha256: String,
    pub integration_sha256: String,
    pub math_sha256: String,
    pub build_contract: String,
}

impl HookSourceClosure {
    fn validate(&self) -> AtlasResult<()> {
        validate_digest("hook shared C source", &self.shared_c_sha256)?;
        validate_digest("hook shared header", &self.shared_h_sha256)?;
        validate_digest("hook integration source", &self.integration_sha256)?;
        validate_digest("hook math source", &self.math_sha256)?;
        validate_text("hook build contract", &self.build_contract)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HookParityAttestation {
    pub name: String,
    pub schema: String,
    pub version: u16,
    pub passed: bool,
    pub case_count: u32,
    pub fixture_bsp_sha256: String,
    pub fixture_provenance_sha256: String,
    pub fixture_collision_physics_identity_sha256: String,
    pub fixture_pmove_physics_identity_sha256: String,
    pub hook_physics_identity_sha256: String,
    pub collision_tool_sha256: String,
    pub pmove_tool_sha256: String,
    pub hook_tool_sha256: String,
    pub q2ded_sha256: String,
    pub game_module_sha256: String,
    pub transcript_sha256: String,
    pub attestation_sha256: String,
}

#[derive(Serialize)]
struct HookParityPayload<'a> {
    name: &'a str,
    schema: &'a str,
    version: u16,
    passed: bool,
    case_count: u32,
    fixture_bsp_sha256: &'a str,
    fixture_provenance_sha256: &'a str,
    fixture_collision_physics_identity_sha256: &'a str,
    fixture_pmove_physics_identity_sha256: &'a str,
    hook_physics_identity_sha256: &'a str,
    collision_tool_sha256: &'a str,
    pmove_tool_sha256: &'a str,
    hook_tool_sha256: &'a str,
    q2ded_sha256: &'a str,
    game_module_sha256: &'a str,
    transcript_sha256: &'a str,
}

impl HookParityAttestation {
    pub fn canonical_attestation_sha256(&self) -> String {
        canonical_digest(&HookParityPayload {
            name: &self.name,
            schema: &self.schema,
            version: self.version,
            passed: self.passed,
            case_count: self.case_count,
            fixture_bsp_sha256: &self.fixture_bsp_sha256,
            fixture_provenance_sha256: &self.fixture_provenance_sha256,
            fixture_collision_physics_identity_sha256: &self
                .fixture_collision_physics_identity_sha256,
            fixture_pmove_physics_identity_sha256: &self.fixture_pmove_physics_identity_sha256,
            hook_physics_identity_sha256: &self.hook_physics_identity_sha256,
            collision_tool_sha256: &self.collision_tool_sha256,
            pmove_tool_sha256: &self.pmove_tool_sha256,
            hook_tool_sha256: &self.hook_tool_sha256,
            q2ded_sha256: &self.q2ded_sha256,
            game_module_sha256: &self.game_module_sha256,
            transcript_sha256: &self.transcript_sha256,
        })
    }

    pub fn seal(mut self) -> Self {
        self.attestation_sha256 = self.canonical_attestation_sha256();
        self
    }

    fn validate(
        &self,
        collision: &CollisionOracleAdmission,
        pmove: &PmoveOracleAdmission,
        hook: &OracleToolIdentity,
    ) -> AtlasResult<()> {
        if self.name != HOOK_PARITY_NAME
            || self.schema != HOOK_PARITY_SCHEMA
            || self.version != ORACLE_SEMANTIC_VERSION
            || !self.passed
            || self.case_count != HOOK_PARITY_CASES_V1
        {
            return Err(AtlasError::InvalidFormat(
                "hook parity semantic contract mismatch".to_owned(),
            ));
        }
        for (name, digest) in [
            ("hook parity fixture BSP", self.fixture_bsp_sha256.as_str()),
            (
                "hook parity fixture provenance",
                self.fixture_provenance_sha256.as_str(),
            ),
            (
                "hook parity collision physics",
                self.fixture_collision_physics_identity_sha256.as_str(),
            ),
            (
                "hook parity pmove physics",
                self.fixture_pmove_physics_identity_sha256.as_str(),
            ),
            (
                "hook parity hook physics",
                self.hook_physics_identity_sha256.as_str(),
            ),
            (
                "hook parity collision tool",
                self.collision_tool_sha256.as_str(),
            ),
            ("hook parity pmove tool", self.pmove_tool_sha256.as_str()),
            ("hook parity hook tool", self.hook_tool_sha256.as_str()),
            ("hook parity q2ded", self.q2ded_sha256.as_str()),
            ("hook parity game module", self.game_module_sha256.as_str()),
            ("hook parity transcript", self.transcript_sha256.as_str()),
            ("hook parity attestation", self.attestation_sha256.as_str()),
        ] {
            validate_digest(name, digest)?;
        }
        if self.collision_tool_sha256 != collision.tool.executable_sha256
            || self.pmove_tool_sha256 != pmove.tool.executable_sha256
            || self.hook_tool_sha256 != hook.executable_sha256
            || self.hook_physics_identity_sha256 != hook.physics_identity_sha256
        {
            return Err(AtlasError::InvalidFormat(
                "hook parity tool/physics closure mismatch".to_owned(),
            ));
        }
        if self.attestation_sha256 != self.canonical_attestation_sha256() {
            return Err(AtlasError::InvalidFormat(
                "hook parity canonical attestation mismatch".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HookOracleAdmission {
    pub tool: OracleToolIdentity,
    pub bsp: OracleBspBinding,
    pub parameters: HookParameters,
    pub source: HookSourceClosure,
    pub parity: HookParityAttestation,
    pub contract_sha256: String,
}

#[derive(Serialize)]
struct HookContractPayload<'a> {
    tool: &'a OracleToolIdentity,
    bsp: &'a OracleBspBinding,
    parameters: &'a HookParameters,
    source: &'a HookSourceClosure,
    parity: &'a HookParityAttestation,
}

impl HookOracleAdmission {
    pub fn canonical_contract_sha256(&self) -> String {
        canonical_digest(&HookContractPayload {
            tool: &self.tool,
            bsp: &self.bsp,
            parameters: &self.parameters,
            source: &self.source,
            parity: &self.parity,
        })
    }

    pub fn seal(mut self) -> Self {
        self.contract_sha256 = self.canonical_contract_sha256();
        self
    }

    fn validate(
        &self,
        bsp: &BspIdentity,
        collision: &CollisionOracleAdmission,
        pmove: &PmoveOracleAdmission,
    ) -> AtlasResult<()> {
        self.tool
            .validate(HOOK_ORACLE_NAME, HOOK_ORACLE_SCHEMA, "hook oracle")?;
        self.bsp.validate(bsp, "hook oracle")?;
        self.parameters.validate()?;
        self.source.validate()?;
        self.parity.validate(collision, pmove, &self.tool)?;
        validate_digest("hook oracle contract", &self.contract_sha256)?;
        if self.contract_sha256 != self.canonical_contract_sha256() {
            return Err(AtlasError::InvalidFormat(
                "hook oracle semantic closure mismatch".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct B1NormativeDocuments {
    pub design_sha256: String,
    pub plan_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct B1AuthorityExecutables {
    pub cm_sha256: String,
    pub pmove_sha256: String,
    pub hook_sha256: String,
    pub fall_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct B1AuthorityIdentity {
    pub tool_identity: String,
    pub physics_identity: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct B1AuthorityIdentities {
    pub collision: B1AuthorityIdentity,
    pub pmove: B1AuthorityIdentity,
    pub hook: B1AuthorityIdentity,
    pub fall: B1AuthorityIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct B1RuntimeAuthoritySeal {
    pub schema: String,
    pub normative_documents: B1NormativeDocuments,
    pub hook_parity_attestation_sha256: String,
    pub fixture_bsp_sha256: String,
    pub analysis_bsp_sha256: String,
    pub executables: B1AuthorityExecutables,
    pub identities: B1AuthorityIdentities,
}

impl B1RuntimeAuthoritySeal {
    fn validate(&self, bsp: &BspIdentity, admissions: &OracleAdmissions) -> AtlasResult<()> {
        if self.schema != B1_RUNTIME_AUTHORITY_SEAL_SCHEMA {
            return Err(AtlasError::InvalidFormat(
                "B1 runtime authority seal schema mismatch".to_owned(),
            ));
        }
        for (name, digest) in [
            ("B1 design", &self.normative_documents.design_sha256),
            ("B1 plan", &self.normative_documents.plan_sha256),
            ("B1 hook attestation", &self.hook_parity_attestation_sha256),
            ("B1 fixture BSP", &self.fixture_bsp_sha256),
            ("B1 analysis BSP", &self.analysis_bsp_sha256),
            ("B1 CM executable", &self.executables.cm_sha256),
            ("B1 Pmove executable", &self.executables.pmove_sha256),
            ("B1 hook executable", &self.executables.hook_sha256),
            ("B1 fall executable", &self.executables.fall_sha256),
            (
                "B1 collision tool",
                &self.identities.collision.tool_identity,
            ),
            (
                "B1 collision physics",
                &self.identities.collision.physics_identity,
            ),
            ("B1 Pmove tool", &self.identities.pmove.tool_identity),
            ("B1 Pmove physics", &self.identities.pmove.physics_identity),
            ("B1 hook tool", &self.identities.hook.tool_identity),
            ("B1 hook physics", &self.identities.hook.physics_identity),
            ("B1 fall tool", &self.identities.fall.tool_identity),
            ("B1 fall physics", &self.identities.fall.physics_identity),
        ] {
            validate_digest(name, digest)?;
        }
        if self.analysis_bsp_sha256 != bsp.sha256 {
            return Err(AtlasError::InvalidFormat(
                "B1 runtime authority seal is bound to a different BSP".to_owned(),
            ));
        }
        let collision = &admissions.collision_oracle.tool;
        if self.executables.cm_sha256 != collision.executable_sha256
            || self.identities.collision.physics_identity != collision.physics_identity_sha256
        {
            return Err(AtlasError::InvalidFormat(
                "B1 collision authority differs from oracle admission".to_owned(),
            ));
        }
        let pmove = admissions.pmove_oracle.as_ref().ok_or_else(|| {
            AtlasError::InvalidFormat(
                "B1 runtime authority seal requires Pmove admission".to_owned(),
            )
        })?;
        if self.executables.pmove_sha256 != pmove.tool.executable_sha256
            || self.identities.pmove.physics_identity != pmove.tool.physics_identity_sha256
        {
            return Err(AtlasError::InvalidFormat(
                "B1 Pmove authority differs from oracle admission".to_owned(),
            ));
        }
        if let Some(hook) = &admissions.hook_oracle {
            if self.executables.hook_sha256 != hook.tool.executable_sha256
                || self.identities.hook.physics_identity != hook.tool.physics_identity_sha256
            {
                return Err(AtlasError::InvalidFormat(
                    "B1 hook authority differs from oracle admission".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OracleAdmissions {
    pub b1_runtime_authority_seal: B1RuntimeAuthoritySeal,
    pub collision_oracle: CollisionOracleAdmission,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pmove_oracle: Option<PmoveOracleAdmission>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hook_oracle: Option<HookOracleAdmission>,
}

impl OracleAdmissions {
    pub fn admit(&self, bsp: &BspIdentity) -> AtlasResult<EdgeAdmission> {
        self.b1_runtime_authority_seal.validate(bsp, self)?;
        self.collision_oracle.validate(bsp)?;
        if let Some(pmove) = &self.pmove_oracle {
            pmove.validate(bsp, &self.collision_oracle)?;
        }
        if let Some(hook) = &self.hook_oracle {
            let pmove = self.pmove_oracle.as_ref().ok_or_else(|| {
                AtlasError::InvalidFormat(
                    "hook oracle admission requires the companion pmove oracle".to_owned(),
                )
            })?;
            hook.validate(bsp, &self.collision_oracle, pmove)?;
        }
        Ok(EdgeAdmission {
            pmove: self.pmove_oracle.is_some(),
            hook: self.hook_oracle.is_some(),
        })
    }
}

/// Edge capabilities can only be constructed by successfully validating the
/// mandatory collision admission and any optional trajectory admissions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EdgeAdmission {
    pmove: bool,
    hook: bool,
}

impl EdgeAdmission {
    pub fn admits_pmove(self) -> bool {
        self.pmove
    }

    pub fn admits_hook(self) -> bool {
        self.hook
    }

    pub(crate) fn validate_edge_type(self, edge_type: EdgeType) -> AtlasResult<()> {
        match edge_type {
            EdgeType::Jump | EdgeType::ControlledDrop if !self.pmove => {
                Err(AtlasError::InvalidFormat(format!(
                    "{edge_type:?} edge requires an admitted pmove oracle"
                )))
            }
            EdgeType::Hook if !(self.hook && self.pmove) => Err(AtlasError::InvalidFormat(
                "Hook edge requires admitted hook, pmove, and q2ded parity contracts".to_owned(),
            )),
            _ => Ok(()),
        }
    }
}
