use std::cmp::Reverse;
use std::collections::BTreeSet;

use super::{AtlasError, AtlasLevel, AtlasOrigin, AtlasResult, COST_INFINITY, GridIndex};

pub const GUIDE_CANDIDATE_LIMIT: usize = 4;
pub const GUIDE_CANDIDATE_WIDTH: usize = 15;
pub const GUIDE_FEATURE_WIDTH: usize = GUIDE_CANDIDATE_LIMIT * GUIDE_CANDIDATE_WIDTH;
pub const GUIDE_DIRECTION_WORLD_SCALE: f32 = 4096.0;
pub const GUIDE_COST_WORLD_SCALE: f32 = 4096.0;

/// Frozen public objective classes. Safe recovery deliberately has no entry;
/// it belongs exclusively to the recovery block.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum ObjectiveClass {
    Weapon = 0,
    Ammunition = 1,
    Health = 2,
    Armor = 3,
    Powerup = 4,
    Rune = 5,
    Control = 6,
    SpawnEgress = 7,
}

impl TryFrom<u8> for ObjectiveClass {
    type Error = AtlasError;

    fn try_from(value: u8) -> AtlasResult<Self> {
        match value {
            0 => Ok(Self::Weapon),
            1 => Ok(Self::Ammunition),
            2 => Ok(Self::Health),
            3 => Ok(Self::Armor),
            4 => Ok(Self::Powerup),
            5 => Ok(Self::Rune),
            6 => Ok(Self::Control),
            7 => Ok(Self::SpawnEgress),
            _ => Err(AtlasError::InvalidFormat(format!(
                "unknown objective class {value}"
            ))),
        }
    }
}

impl ObjectiveClass {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Weapon => "weapon",
            Self::Ammunition => "ammunition",
            Self::Health => "health",
            Self::Armor => "armor",
            Self::Powerup => "powerup",
            Self::Rune => "rune",
            Self::Control => "control",
            Self::SpawnEgress => "spawn_egress",
        }
    }

    pub fn from_name(value: &str) -> AtlasResult<Self> {
        match value {
            "weapon" => Ok(Self::Weapon),
            "ammunition" => Ok(Self::Ammunition),
            "health" => Ok(Self::Health),
            "armor" => Ok(Self::Armor),
            "powerup" => Ok(Self::Powerup),
            "rune" => Ok(Self::Rune),
            "control" => Ok(Self::Control),
            "spawn_egress" => Ok(Self::SpawnEgress),
            _ => Err(AtlasError::InvalidFormat(format!(
                "unknown objective class name {value}"
            ))),
        }
    }

    pub fn from_classname(classname: &str) -> Option<Self> {
        if classname.starts_with("weapon_") {
            Some(Self::Weapon)
        } else if classname.starts_with("ammo_")
            || matches!(classname, "item_pack" | "item_bandolier")
        {
            Some(Self::Ammunition)
        } else if classname.starts_with("item_health")
            || matches!(classname, "item_adrenaline" | "item_ancient_head")
        {
            Some(Self::Health)
        } else if classname.starts_with("item_armor_")
            || matches!(classname, "item_power_shield" | "item_power_screen")
        {
            Some(Self::Armor)
        } else if matches!(
            classname,
            "item_quad"
                | "item_invulnerability"
                | "item_silencer"
                | "item_breather"
                | "item_enviro"
        ) {
            Some(Self::Powerup)
        } else if classname.starts_with("rune_") || classname.starts_with("item_rune_") {
            Some(Self::Rune)
        } else if classname.starts_with("item_flag_") || classname.starts_with("item_tech") {
            Some(Self::Control)
        } else if classname == "info_player_deathmatch" {
            Some(Self::SpawnEgress)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GuideCandidate {
    pub class: ObjectiveClass,
    pub world_point: [f64; 3],
    pub cost_q8: u32,
    pub risk: u16,
    pub confidence: u16,
    /// Per-client belief only. Exact global item timers are not accepted by
    /// this public packing API.
    pub availability_belief: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuideFeatureBlock {
    pub values: [f32; GUIDE_FEATURE_WIDTH],
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CandidateKey {
    // Frozen selection order: strongest per-client availability belief, then
    // lower traversal cost/risk, higher confidence, class, and L1 key.
    unavailable_q16: Reverse<u16>,
    cost_q8: u32,
    risk: u16,
    confidence: Reverse<u16>,
    class: ObjectiveClass,
    index: GridIndex,
}

#[derive(Clone, Copy, Debug)]
struct AdmittedCandidate {
    candidate: GuideCandidate,
    availability_q16: u16,
    key: CandidateKey,
}

pub(crate) fn pack_guide_features(
    origin: AtlasOrigin,
    world_position: [f64; 3],
    yaw_degrees: f32,
    candidates: &[GuideCandidate],
) -> AtlasResult<GuideFeatureBlock> {
    if world_position.iter().any(|value| !value.is_finite()) || !yaw_degrees.is_finite() {
        return Err(AtlasError::Coordinate(
            "guide query has a non-finite pose".to_owned(),
        ));
    }

    let mut admitted = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.world_point.iter().any(|value| !value.is_finite())
            || !candidate.availability_belief.is_finite()
            || !(0.0..=1.0).contains(&candidate.availability_belief)
            || candidate.cost_q8 == COST_INFINITY
        {
            return Err(AtlasError::InvalidFormat(
                "guide candidate has invalid public fields".to_owned(),
            ));
        }
        let index = origin.index(candidate.world_point, AtlasLevel::L1)?;
        let availability_q16 = (candidate.availability_belief * f32::from(u16::MAX)).round() as u16;
        admitted.push(AdmittedCandidate {
            candidate: *candidate,
            availability_q16,
            key: CandidateKey {
                unavailable_q16: Reverse(availability_q16),
                cost_q8: candidate.cost_q8,
                risk: candidate.risk,
                confidence: Reverse(candidate.confidence),
                class: candidate.class,
                index,
            },
        });
    }
    admitted.sort_by_key(|candidate| candidate.key);
    // Two objectives of the same class in one L1 cell are observationally the
    // same public guidepost. Deterministically retain the better-ranked one.
    let mut seen = BTreeSet::new();
    admitted.retain(|candidate| seen.insert((candidate.candidate.class, candidate.key.index)));
    admitted.truncate(GUIDE_CANDIDATE_LIMIT);

    let mut values = [0.0; GUIDE_FEATURE_WIDTH];
    for (slot, admitted) in admitted.iter().enumerate() {
        let offset = slot * GUIDE_CANDIDATE_WIDTH;
        let direction = yaw_local_scaled(
            world_position,
            admitted.candidate.world_point,
            yaw_degrees,
            GUIDE_DIRECTION_WORLD_SCALE,
        );
        values[offset..offset + 3].copy_from_slice(&direction);
        values[offset + 3] = normalize_cost(admitted.candidate.cost_q8, GUIDE_COST_WORLD_SCALE);
        values[offset + 4] = f32::from(admitted.candidate.risk) / f32::from(u16::MAX);
        values[offset + 5] = f32::from(admitted.candidate.confidence) / f32::from(u16::MAX);
        values[offset + 6] = f32::from(admitted.availability_q16) / f32::from(u16::MAX);
        values[offset + 7 + admitted.candidate.class as usize] = 1.0;
    }
    Ok(GuideFeatureBlock { values })
}

pub(crate) fn yaw_local_unit(delta: [f64; 3], yaw_degrees: f32) -> [f32; 3] {
    let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
    if length <= f64::EPSILON {
        return [0.0; 3];
    }
    let normalized = [
        (delta[0] / length) as f32,
        (delta[1] / length) as f32,
        (delta[2] / length) as f32,
    ];
    let yaw = yaw_degrees.to_radians();
    let (sin, cos) = yaw.sin_cos();
    [
        (normalized[0] * cos + normalized[1] * sin).clamp(-1.0, 1.0),
        (normalized[0] * sin - normalized[1] * cos).clamp(-1.0, 1.0),
        normalized[2].clamp(-1.0, 1.0),
    ]
}

fn yaw_local_scaled(source: [f64; 3], target: [f64; 3], yaw_degrees: f32, scale: f32) -> [f32; 3] {
    let delta = [
        (target[0] - source[0]) as f32,
        (target[1] - source[1]) as f32,
        (target[2] - source[2]) as f32,
    ];
    let yaw = yaw_degrees.to_radians();
    let (sin, cos) = yaw.sin_cos();
    let inverse_scale = scale.recip();
    [
        ((delta[0] * cos + delta[1] * sin) * inverse_scale).clamp(-1.0, 1.0),
        ((delta[0] * sin - delta[1] * cos) * inverse_scale).clamp(-1.0, 1.0),
        (delta[2] * inverse_scale).clamp(-1.0, 1.0),
    ]
}

pub(crate) fn normalize_cost(cost_q8: u32, world_scale: f32) -> f32 {
    if cost_q8 == COST_INFINITY {
        return 1.0;
    }
    ((cost_q8 as f64 / 256.0) as f32 / world_scale).clamp(0.0, 1.0)
}

pub const GUIDE_FEATURE_NAMES: [&str; GUIDE_FEATURE_WIDTH] = guide_feature_names();

const fn guide_feature_names() -> [&'static str; GUIDE_FEATURE_WIDTH] {
    [
        "guide_0_forward",
        "guide_0_quake_right",
        "guide_0_up",
        "guide_0_cost",
        "guide_0_risk",
        "guide_0_confidence",
        "guide_0_availability_belief",
        "guide_0_class_weapon",
        "guide_0_class_ammunition",
        "guide_0_class_health",
        "guide_0_class_armor",
        "guide_0_class_powerup",
        "guide_0_class_rune",
        "guide_0_class_control",
        "guide_0_class_spawn_egress",
        "guide_1_forward",
        "guide_1_quake_right",
        "guide_1_up",
        "guide_1_cost",
        "guide_1_risk",
        "guide_1_confidence",
        "guide_1_availability_belief",
        "guide_1_class_weapon",
        "guide_1_class_ammunition",
        "guide_1_class_health",
        "guide_1_class_armor",
        "guide_1_class_powerup",
        "guide_1_class_rune",
        "guide_1_class_control",
        "guide_1_class_spawn_egress",
        "guide_2_forward",
        "guide_2_quake_right",
        "guide_2_up",
        "guide_2_cost",
        "guide_2_risk",
        "guide_2_confidence",
        "guide_2_availability_belief",
        "guide_2_class_weapon",
        "guide_2_class_ammunition",
        "guide_2_class_health",
        "guide_2_class_armor",
        "guide_2_class_powerup",
        "guide_2_class_rune",
        "guide_2_class_control",
        "guide_2_class_spawn_egress",
        "guide_3_forward",
        "guide_3_quake_right",
        "guide_3_up",
        "guide_3_cost",
        "guide_3_risk",
        "guide_3_confidence",
        "guide_3_availability_belief",
        "guide_3_class_weapon",
        "guide_3_class_ammunition",
        "guide_3_class_health",
        "guide_3_class_armor",
        "guide_3_class_powerup",
        "guide_3_class_rune",
        "guide_3_class_control",
        "guide_3_class_spawn_egress",
    ]
}
