use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

use q2_lattice_rs::atlas::{
    AtlasAggregateCell, AtlasArtifact, AtlasLevel, AtlasLimits, AtlasOrigin, BspIdentity,
    COST_INFINITY, EdgeInput, EdgeType, GridIndex, L0BitPlane, L0Chunk, L0ScalarPlane, L1Graph,
    L1Node, OracleAdmissions, SparseL0, Stance, sha256_hex,
};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    origin: [i64; 3],
    bsp: BspIdentity,
    oracles: OracleAdmissions,
    chunks: Vec<ChunkPlan>,
    nodes: Vec<NodePlan>,
    edges: Vec<EdgePlan>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChunkPlan {
    key: [i32; 3],
    bits: BTreeMap<String, Vec<usize>>,
    scalars: BTreeMap<String, Vec<[usize; 2]>>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NodePlan {
    index: [i32; 3],
    flags: u16,
    floor_normal_class: u8,
    clearance_height: u16,
    hazard_types: u16,
    hazard_severity: u8,
    hazard_clearance: i32,
    cost_to_safety: u32,
    region_id: u32,
    confidence: u16,
    evidence: u16,
    contents_flags: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgePlan {
    source: [i32; 3],
    target: [i32; 3],
    edge_type: String,
    stance: String,
    flags: u16,
    blocker: u32,
    cost: u32,
    risk: u16,
    confidence: u16,
    evidence: u16,
    validation_version: u16,
    auxiliary: u32,
}

fn index(value: [i32; 3]) -> GridIndex {
    GridIndex::new(value[0], value[1], value[2])
}

fn bit_plane(name: &str) -> Result<L0BitPlane, String> {
    Ok(match name {
        "solid" => L0BitPlane::Solid,
        "window" => L0BitPlane::Window,
        "playerclip" => L0BitPlane::PlayerClip,
        "water" => L0BitPlane::Water,
        "slime" => L0BitPlane::Slime,
        "lava" => L0BitPlane::Lava,
        "ladder" => L0BitPlane::Ladder,
        "hurt" => L0BitPlane::Hurt,
        "teleport_trigger" => L0BitPlane::TeleportTrigger,
        "mover_reference_solid" => L0BitPlane::MoverReferenceSolid,
        "mover_swept_envelope" => L0BitPlane::MoverSweptEnvelope,
        "hookable_surface" => L0BitPlane::HookableSurface,
        "standing_forbidden_origin" => L0BitPlane::StandingForbiddenOrigin,
        "crouched_forbidden_origin" => L0BitPlane::CrouchedForbiddenOrigin,
        "unknown" => L0BitPlane::Unknown,
        "spawn_column" => L0BitPlane::SpawnColumn,
        "hook_corridor" => L0BitPlane::HookCorridor,
        _ => return Err(format!("unknown L0 bit plane {name}")),
    })
}

fn scalar_plane(name: &str) -> Result<L0ScalarPlane, String> {
    Ok(match name {
        "current_direction" => L0ScalarPlane::CurrentDirection,
        "hazard_severity" => L0ScalarPlane::HazardSeverity,
        "confidence" => L0ScalarPlane::Confidence,
        _ => return Err(format!("unknown L0 scalar plane {name}")),
    })
}

fn edge_type(name: &str) -> Result<EdgeType, String> {
    Ok(match name {
        "walk" => EdgeType::Walk,
        "strafe_walk" => EdgeType::StrafeWalk,
        "step" => EdgeType::Step,
        "jump" => EdgeType::Jump,
        "controlled_drop" => EdgeType::ControlledDrop,
        "crouch_enter" => EdgeType::CrouchEnter,
        "crouch_hold" => EdgeType::CrouchHold,
        "crouch_exit" => EdgeType::CrouchExit,
        "water_transition" => EdgeType::WaterTransition,
        "mover" => EdgeType::Mover,
        "teleporter" => EdgeType::Teleporter,
        "hook" => EdgeType::Hook,
        _ => return Err(format!("unknown edge type {name}")),
    })
}

fn stance(name: &str) -> Result<Stance, String> {
    Ok(match name {
        "standing" => Stance::Standing,
        "crouched" => Stance::Crouched,
        "either" => Stance::Either,
        "water" => Stance::Water,
        _ => return Err(format!("unknown stance {name}")),
    })
}

fn aggregate(nodes: &[NodePlan], level: AtlasLevel) -> Vec<AtlasAggregateCell> {
    let mut groups: BTreeMap<GridIndex, Vec<&NodePlan>> = BTreeMap::new();
    for node in nodes {
        let mut parent = index(node.index).parent();
        if level == AtlasLevel::L3 {
            parent = parent.parent();
        }
        groups.entry(parent).or_default().push(node);
    }
    groups
        .into_iter()
        .map(|(key, children)| {
            let mut contents = 0_u32;
            let mut hazards = 0_u16;
            let mut severity = 0_u8;
            let mut clearance = u16::MAX;
            let mut cost = COST_INFINITY;
            let mut confidence = u16::MAX;
            let mut standing = false;
            let mut crouched = false;
            for child in children {
                contents |= child.contents_flags;
                hazards |= child.hazard_types;
                severity = severity.max(child.hazard_severity);
                clearance = clearance.min(child.clearance_height);
                cost = cost.min(child.cost_to_safety);
                confidence = confidence.min(child.confidence);
                standing |= child.flags & (1 << 4) != 0;
                crouched |= child.flags & (1 << 5) != 0;
            }
            AtlasAggregateCell {
                index: key,
                contents_flags: contents,
                hazard_types: hazards,
                hazard_severity: severity,
                standing_passable: standing,
                crouched_passable: crouched,
                clearance,
                cost_to_safety: cost,
                confidence,
            }
        })
        .collect()
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn run() -> Result<(), String> {
    let arguments: Vec<_> = env::args_os().collect();
    if arguments.len() != 4 {
        return Err("usage: q2-atlas-pack PLAN.json ATLAS.bin ATLAS.bin.zst".to_owned());
    }
    let plan_bytes = fs::read(&arguments[1]).map_err(|error| error.to_string())?;
    let plan: Plan = serde_json::from_slice(&plan_bytes).map_err(|error| error.to_string())?;
    if plan.schema != "q2-atlas-build-plan-v1" {
        return Err("unsupported Atlas build-plan schema".to_owned());
    }
    let admission = plan
        .oracles
        .admit(&plan.bsp)
        .map_err(|error| error.to_string())?;
    let limits = AtlasLimits::default();
    let mut l0 = SparseL0::new();
    for item in plan.chunks {
        let mut chunk = L0Chunk::new(index(item.key));
        for (name, cells) in item.bits {
            let plane = bit_plane(&name)?;
            for cell in cells {
                chunk
                    .set_bit(plane, cell, true)
                    .map_err(|error| error.to_string())?;
            }
        }
        for (name, cells) in item.scalars {
            let plane = scalar_plane(&name)?;
            for [cell, value] in cells {
                let value = u8::try_from(value).map_err(|_| "L0 scalar exceeds u8")?;
                chunk
                    .set_scalar(plane, cell, value)
                    .map_err(|error| error.to_string())?;
            }
        }
        l0.insert(chunk, &limits)
            .map_err(|error| error.to_string())?;
    }
    let nodes: Vec<_> = plan
        .nodes
        .iter()
        .map(|item| L1Node {
            index: index(item.index),
            flags: item.flags,
            floor_normal_class: item.floor_normal_class,
            clearance_height: item.clearance_height,
            hazard_types: item.hazard_types,
            hazard_severity: item.hazard_severity,
            hazard_clearance: item.hazard_clearance,
            cost_to_safety: item.cost_to_safety,
            region_id: item.region_id,
            confidence: item.confidence,
            evidence: item.evidence,
            contents_flags: item.contents_flags,
        })
        .collect();
    let edges: Vec<_> = plan
        .edges
        .into_iter()
        .map(|item| {
            Ok(EdgeInput {
                source: index(item.source),
                target: index(item.target),
                edge_type: edge_type(&item.edge_type)?,
                stance: stance(&item.stance)?,
                flags: item.flags,
                blocker: item.blocker,
                cost: item.cost,
                risk: item.risk,
                confidence: item.confidence,
                evidence: item.evidence,
                validation_version: item.validation_version,
                auxiliary: item.auxiliary,
            })
        })
        .collect::<Result<_, String>>()?;
    let graph =
        L1Graph::build(nodes, edges, &admission, &limits).map_err(|error| error.to_string())?;
    let artifact = AtlasArtifact {
        origin: AtlasOrigin(plan.origin),
        l0,
        l1: graph,
        l2: aggregate(&plan.nodes, AtlasLevel::L2),
        l3: aggregate(&plan.nodes, AtlasLevel::L3),
    };
    let raw = artifact
        .encode_uncompressed(&limits)
        .map_err(|error| error.to_string())?;
    let compressed = artifact
        .encode_zstd(&limits)
        .map_err(|error| error.to_string())?;
    write(Path::new(&arguments[2]), &raw)?;
    write(Path::new(&arguments[3]), &compressed)?;
    let summary = json!({
        "schema": "q2-atlas-pack-result-v1",
        "uncompressed_sha256": sha256_hex(&raw),
        "uncompressed_bytes": raw.len(),
        "compressed_sha256": sha256_hex(&compressed),
        "compressed_bytes": compressed.len(),
        "l0_chunks": artifact.l0.len(),
        "l1_nodes": artifact.l1.nodes().len(),
        "l1_edges": artifact.l1.edges().len(),
        "l2_cells": artifact.l2.len(),
        "l3_cells": artifact.l3.len(),
        "resident_bytes_estimate": artifact.resident_bytes_estimate(),
    });
    println!(
        "{}",
        serde_json::to_string(&summary).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("q2-atlas-pack: {error}");
        std::process::exit(65);
    }
}
