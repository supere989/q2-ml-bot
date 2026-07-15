use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::fs;
use std::path::Path;

use q2_lattice_rs::atlas::{
    AtlasAggregateCell, AtlasArtifact, AtlasLimits, AtlasOrigin, BspIdentity, COST_INFINITY,
    ConservativeChild, CorridorWitness, EdgeInput, EdgeType, GridIndex, L0BitPlane, L0Chunk,
    L0ScalarPlane, L1Graph, L1Node, NodeFlags, OracleAdmissions, SparseL0, Stance,
    aggregate_conservative, sha256_hex,
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
        "monsterclip" => L0BitPlane::MonsterClipDiagnostic,
        "water" => L0BitPlane::Water,
        "slime" => L0BitPlane::Slime,
        "lava" => L0BitPlane::Lava,
        "mist" => L0BitPlane::Mist,
        "ladder" => L0BitPlane::Ladder,
        "hurt" => L0BitPlane::Hurt,
        "push_or_gravity" => L0BitPlane::PushOrGravity,
        "teleport_trigger" => L0BitPlane::TeleportTrigger,
        "mover_reference_solid" => L0BitPlane::MoverReferenceSolid,
        "mover_swept_envelope" => L0BitPlane::MoverSweptEnvelope,
        "areaportal" => L0BitPlane::AreaPortal,
        "sky" => L0BitPlane::Sky,
        "slick" => L0BitPlane::Slick,
        "warp" => L0BitPlane::Warp,
        "nodraw" => L0BitPlane::NoDrawDiagnostic,
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
        "ladder" => EdgeType::Ladder,
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

#[derive(Clone, Copy, Debug)]
struct AggregateSource {
    index: GridIndex,
    contents_flags: u32,
    hazard_types: u16,
    hazard_severity: u8,
    clearance: u16,
    cost_to_safety: u32,
    confidence: u16,
    standing_passable: bool,
    crouched_passable: bool,
}

impl AggregateSource {
    fn from_node(node: &NodePlan) -> Self {
        Self {
            index: index(node.index),
            contents_flags: node.contents_flags,
            hazard_types: node.hazard_types,
            hazard_severity: node.hazard_severity,
            clearance: node.clearance_height,
            cost_to_safety: node.cost_to_safety,
            confidence: node.confidence,
            standing_passable: node.flags & NodeFlags::STANDING_PASSABLE != 0,
            crouched_passable: node.flags & NodeFlags::CROUCHED_PASSABLE != 0,
        }
    }

    fn from_aggregate(cell: &AtlasAggregateCell) -> Self {
        Self {
            index: cell.index,
            contents_flags: cell.contents_flags,
            hazard_types: cell.hazard_types,
            hazard_severity: cell.hazard_severity,
            clearance: cell.clearance,
            cost_to_safety: cell.cost_to_safety,
            confidence: cell.confidence,
            standing_passable: cell.standing_passable,
            crouched_passable: cell.crouched_passable,
        }
    }

    fn passable(self, standing: bool) -> bool {
        if standing {
            self.standing_passable
        } else {
            self.crouched_passable
        }
    }
}

#[derive(Clone, Debug, Default)]
struct StanceAdjacency {
    standing: BTreeMap<GridIndex, BTreeSet<GridIndex>>,
    crouched: BTreeMap<GridIndex, BTreeSet<GridIndex>>,
}

impl StanceAdjacency {
    fn from_edges(edges: &[EdgePlan]) -> Result<Self, String> {
        let mut result = Self::default();
        for edge in edges {
            let source = index(edge.source);
            let target = index(edge.target);
            match stance(&edge.stance)? {
                Stance::Standing => {
                    result.standing.entry(source).or_default().insert(target);
                }
                Stance::Crouched => {
                    result.crouched.entry(source).or_default().insert(target);
                }
                Stance::Either => {
                    result.standing.entry(source).or_default().insert(target);
                    result.crouched.entry(source).or_default().insert(target);
                }
                Stance::Water => {}
            }
        }
        Ok(result)
    }

    fn for_stance(&self, standing: bool) -> &BTreeMap<GridIndex, BTreeSet<GridIndex>> {
        if standing {
            &self.standing
        } else {
            &self.crouched
        }
    }

    fn coarsened(&self) -> Self {
        Self {
            standing: coarsen_edges(&self.standing),
            crouched: coarsen_edges(&self.crouched),
        }
    }
}

fn coarsen_edges(
    edges: &BTreeMap<GridIndex, BTreeSet<GridIndex>>,
) -> BTreeMap<GridIndex, BTreeSet<GridIndex>> {
    let mut result: BTreeMap<GridIndex, BTreeSet<GridIndex>> = BTreeMap::new();
    for (source, targets) in edges {
        let coarse_source = source.parent();
        for target in targets {
            let coarse_target = target.parent();
            if coarse_source != coarse_target {
                result
                    .entry(coarse_source)
                    .or_default()
                    .insert(coarse_target);
            }
        }
    }
    result
}

fn blocked_child() -> ConservativeChild {
    ConservativeChild {
        contents_flags: 0,
        hazard_types: 0,
        hazard_severity: 0,
        clearance: 0,
        cost_to_safety: COST_INFINITY,
        confidence: 0,
        standing_passable: false,
        crouched_passable: false,
        standing_reachable: false,
        crouched_reachable: false,
    }
}

fn child_ordinal(child: GridIndex) -> u8 {
    let x = child.x.rem_euclid(4) as u8;
    let y = child.y.rem_euclid(4) as u8;
    let z = child.z.rem_euclid(4) as u8;
    x + 4 * y + 16 * z
}

fn is_parent_boundary(child: GridIndex) -> bool {
    [child.x, child.y, child.z]
        .into_iter()
        .any(|value| matches!(value.rem_euclid(4), 0 | 3))
}

/// Return every stance-clear child that has a directed, stance-legal path to
/// the containing parent's boundary. The reverse flood is only an efficient
/// way to prove the forward child-to-boundary reachability relation.
fn boundary_reachable(
    cells: &BTreeMap<GridIndex, AggregateSource>,
    adjacency: &BTreeMap<GridIndex, BTreeSet<GridIndex>>,
    standing: bool,
) -> BTreeSet<GridIndex> {
    let mut reverse: BTreeMap<GridIndex, BTreeSet<GridIndex>> = BTreeMap::new();
    for (source, targets) in adjacency {
        let Some(source_cell) = cells.get(source) else {
            continue;
        };
        if !source_cell.passable(standing) {
            continue;
        }
        for target in targets {
            if cells
                .get(target)
                .is_some_and(|cell| cell.passable(standing))
            {
                reverse.entry(*target).or_default().insert(*source);
            }
        }
    }

    let mut reachable = BTreeSet::new();
    let mut pending = VecDeque::new();
    for (child, cell) in cells {
        if cell.passable(standing) && is_parent_boundary(*child) {
            reachable.insert(*child);
            pending.push_back(*child);
        }
    }
    while let Some(target) = pending.pop_front() {
        if let Some(sources) = reverse.get(&target) {
            for source in sources {
                if reachable.insert(*source) {
                    pending.push_back(*source);
                }
            }
        }
    }
    reachable
}

fn witness(reachable: &BTreeSet<GridIndex>) -> Option<CorridorWitness> {
    if reachable.is_empty() {
        return None;
    }
    let mut child_indices: Vec<_> = reachable.iter().copied().map(child_ordinal).collect();
    child_indices.sort_unstable();
    child_indices.dedup();
    Some(CorridorWitness {
        child_indices,
        reaches_boundary: true,
    })
}

fn aggregate_level(
    cells: &[AggregateSource],
    adjacency: &StanceAdjacency,
) -> Result<Vec<AtlasAggregateCell>, String> {
    let mut groups: BTreeMap<GridIndex, BTreeMap<GridIndex, AggregateSource>> = BTreeMap::new();
    for cell in cells {
        groups
            .entry(cell.index.parent())
            .or_default()
            .insert(cell.index, *cell);
    }
    groups
        .into_iter()
        .map(|(parent, group)| {
            let standing_reachable = boundary_reachable(&group, adjacency.for_stance(true), true);
            let crouched_reachable = boundary_reachable(&group, adjacency.for_stance(false), false);
            let standing_witness = witness(&standing_reachable);
            let crouched_witness = witness(&crouched_reachable);
            let mut children = vec![blocked_child(); 64];
            for (child_index, cell) in &group {
                let ordinal = child_ordinal(*child_index) as usize;
                children[ordinal] = ConservativeChild {
                    contents_flags: cell.contents_flags,
                    hazard_types: cell.hazard_types,
                    hazard_severity: cell.hazard_severity,
                    clearance: cell.clearance,
                    cost_to_safety: cell.cost_to_safety,
                    confidence: cell.confidence,
                    standing_passable: cell.standing_passable,
                    crouched_passable: cell.crouched_passable,
                    standing_reachable: standing_reachable.contains(child_index),
                    crouched_reachable: crouched_reachable.contains(child_index),
                };
            }
            let aggregate = aggregate_conservative(
                &children,
                standing_witness.as_ref(),
                crouched_witness.as_ref(),
            )
            .map_err(|error| error.to_string())?;
            Ok(AtlasAggregateCell {
                index: parent,
                contents_flags: aggregate.contents_flags,
                hazard_types: aggregate.hazard_types,
                hazard_severity: aggregate.hazard_severity,
                standing_passable: aggregate.standing_passable,
                crouched_passable: aggregate.crouched_passable,
                clearance: aggregate.clearance,
                cost_to_safety: aggregate.cost_to_safety,
                confidence: aggregate.confidence,
            })
        })
        .collect()
}

fn parse_peak_rss_bytes(status: &str) -> Result<u64, String> {
    let line = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))
        .ok_or_else(|| "Linux /proc/self/status has no VmHWM peak-RSS field".to_owned())?;
    let fields: Vec<_> = line.split_whitespace().collect();
    if fields.len() != 3 || fields[0] != "VmHWM:" || fields[2] != "kB" {
        return Err("Linux VmHWM peak-RSS field has an unsupported format".to_owned());
    }
    let kibibytes = fields[1]
        .parse::<u64>()
        .map_err(|_| "Linux VmHWM peak-RSS value is not an integer".to_owned())?;
    if kibibytes == 0 {
        return Err("Linux VmHWM peak-RSS value is zero".to_owned());
    }
    kibibytes
        .checked_mul(1024)
        .ok_or_else(|| "Linux VmHWM peak-RSS byte count overflow".to_owned())
}

#[cfg(target_os = "linux")]
fn measured_peak_rss_bytes() -> Result<u64, String> {
    let status = fs::read_to_string("/proc/self/status")
        .map_err(|error| format!("cannot read Linux peak RSS: {error}"))?;
    parse_peak_rss_bytes(&status)
}

#[cfg(not(target_os = "linux"))]
fn measured_peak_rss_bytes() -> Result<u64, String> {
    Err("peak RSS measurement is unavailable outside Linux".to_owned())
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
    let l1_adjacency = StanceAdjacency::from_edges(&plan.edges)?;
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
    let l1_sources: Vec<_> = plan.nodes.iter().map(AggregateSource::from_node).collect();
    let l2 = aggregate_level(&l1_sources, &l1_adjacency)?;
    let l2_sources: Vec<_> = l2.iter().map(AggregateSource::from_aggregate).collect();
    let l3 = aggregate_level(&l2_sources, &l1_adjacency.coarsened())?;
    let artifact = AtlasArtifact {
        origin: AtlasOrigin(plan.origin),
        l0,
        l1: graph,
        l2,
        l3,
    };
    let raw = artifact
        .encode_uncompressed(&limits)
        .map_err(|error| error.to_string())?;
    let compressed = artifact
        .encode_zstd(&limits)
        .map_err(|error| error.to_string())?;
    let build_peak_rss_bytes = measured_peak_rss_bytes()?;
    if build_peak_rss_bytes > limits.max_build_rss_bytes {
        return Err(format!(
            "Atlas build peak RSS {build_peak_rss_bytes} > {}",
            limits.max_build_rss_bytes
        ));
    }
    // Peak RSS is operational evidence, not canonical Atlas content. Keep the
    // exact, naturally variable process measurement out of deterministic JSON
    // artifacts while still reporting it to the invoking gate log.
    eprintln!(
        "q2-atlas-pack: build_peak_rss_bytes={build_peak_rss_bytes} max_build_rss_bytes={} passed=true",
        limits.max_build_rss_bytes
    );
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
        "build_peak_rss_measurement": "linux_proc_self_status_vmhwm",
        "build_peak_rss_bytes": build_peak_rss_bytes,
        "max_build_rss_bytes": limits.max_build_rss_bytes,
        "build_peak_rss_gate_passed": true,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn source(index: [i32; 3], cost: u32, hazards: u16) -> AggregateSource {
        AggregateSource {
            index: super::index(index),
            contents_flags: 0,
            hazard_types: hazards,
            hazard_severity: u8::from(hazards != 0) * 200,
            clearance: 128,
            cost_to_safety: cost,
            confidence: u16::MAX,
            standing_passable: true,
            crouched_passable: false,
        }
    }

    #[test]
    fn aggregation_rejects_an_interior_origin_without_a_boundary_corridor() {
        let result =
            aggregate_level(&[source([1, 1, 1], 1, 0)], &StanceAdjacency::default()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(!result[0].standing_passable);
        assert!(!result[0].crouched_passable);
        assert_eq!(result[0].cost_to_safety, COST_INFINITY);
    }

    #[test]
    fn aggregation_uses_only_stance_reachable_costs_and_preserves_hazards() {
        let cells = [
            source([1, 1, 1], 100, 0),
            source([2, 1, 1], 200, 0),
            source([3, 1, 1], 300, 0),
            source([1, 2, 1], 1, 0x20),
        ];
        let mut adjacency = StanceAdjacency::default();
        adjacency
            .standing
            .entry(super::index([1, 1, 1]))
            .or_default()
            .insert(super::index([2, 1, 1]));
        adjacency
            .standing
            .entry(super::index([2, 1, 1]))
            .or_default()
            .insert(super::index([3, 1, 1]));

        let result = aggregate_level(&cells, &adjacency).unwrap();
        assert!(result[0].standing_passable);
        assert_eq!(result[0].cost_to_safety, 100);
        assert_eq!(result[0].hazard_types, 0x20);
        assert_eq!(result[0].hazard_severity, 200);
    }

    #[test]
    fn aggregation_is_stance_specific_and_coarsens_only_cross_parent_edges() {
        let edge = EdgePlan {
            source: [3, 0, 0],
            target: [4, 0, 0],
            edge_type: "walk".to_owned(),
            stance: "standing".to_owned(),
            flags: 0,
            blocker: 0,
            cost: 1,
            risk: 0,
            confidence: u16::MAX,
            evidence: 1,
            validation_version: 1,
            auxiliary: u32::MAX,
        };
        let adjacency = StanceAdjacency::from_edges(&[edge]).unwrap();
        assert!(
            adjacency
                .standing
                .get(&super::index([3, 0, 0]))
                .is_some_and(|targets| targets.contains(&super::index([4, 0, 0])))
        );
        assert!(adjacency.crouched.is_empty());
        assert!(
            adjacency
                .coarsened()
                .standing
                .get(&super::index([0, 0, 0]))
                .is_some_and(|targets| targets.contains(&super::index([1, 0, 0])))
        );
    }

    #[test]
    fn linux_peak_rss_parser_is_strict_and_byte_exact() {
        assert_eq!(
            parse_peak_rss_bytes("Name:\tq2-atlas-pack\nVmHWM:\t  12345 kB\n").unwrap(),
            12_641_280
        );
        assert!(parse_peak_rss_bytes("VmRSS:\t12345 kB\n").is_err());
        assert!(parse_peak_rss_bytes("VmHWM:\t12345 MB\n").is_err());
        assert!(parse_peak_rss_bytes("VmHWM:\t0 kB\n").is_err());
    }
}
