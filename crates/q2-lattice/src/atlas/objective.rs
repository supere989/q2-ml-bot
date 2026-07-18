use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use serde::{Deserialize, Serialize};

use super::guide::{GuideCandidate, GuideFeatureBlock, ObjectiveClass, pack_guide_features};
use super::{
    AtlasError, AtlasLevel, AtlasOrigin, AtlasResult, COST_INFINITY, EdgeType, GridIndex, L1Graph,
    NodeFlags, sha256_hex,
};

pub const OBJECTIVE_SCHEMA: &str = "q2-atlas-objectives-v1";
pub const OBJECTIVE_MEDIA_TYPE: &str = "application/vnd.q2.atlas-objectives-v1";
pub const OBJECTIVE_ARTIFACT_SUFFIX: &str = ".objectives.json";
pub const OBJECTIVE_CLASS_COUNT: usize = 8;
pub const OBJECTIVE_LIMIT: usize = 8192;
pub const OBJECTIVE_TARGET_MAX_DISTANCE: f64 = 160.0;
pub const DEFAULT_AVAILABILITY_BELIEF: f32 = 0.5;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtlasObjective {
    pub objective_id: u32,
    pub class: ObjectiveClass,
    pub classname: String,
    pub world_milliunits: [i64; 3],
    pub target_l1: GridIndex,
    pub risk: u16,
    pub confidence: u16,
}

impl AtlasObjective {
    pub fn world_point(&self) -> [f64; 3] {
        self.world_milliunits.map(|value| value as f64 / 1000.0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtlasObjectives {
    pub canonical_map_id: String,
    pub bsp_sha256: String,
    pub atlas_sha256: String,
    pub origin: AtlasOrigin,
    pub objectives: Vec<AtlasObjective>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectiveBelief {
    pub objective_id: u32,
    pub availability_belief: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct ObjectiveGuide {
    fields: [Option<ObjectiveField>; OBJECTIVE_CLASS_COUNT],
}

#[derive(Clone, Debug)]
struct ObjectiveField {
    cells: Vec<Option<RouteCell>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RouteCell {
    cost_q8: u32,
    risk: u16,
    confidence: u16,
    objective_ordinal: u32,
    next_ordinal: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RouteRank {
    cost_q8: u32,
    risk: u16,
    confidence: Reverse<u16>,
    objective_id: u32,
    next_ordinal: u32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ObjectiveDocument {
    atlas_sha256: String,
    bsp_sha256: String,
    canonical_map_id: String,
    objectives: Vec<ObjectiveRecord>,
    origin: [i64; 3],
    schema: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ObjectiveRecord {
    class: String,
    classname: String,
    confidence: u16,
    l1_index: [i32; 3],
    objective_id: u32,
    risk: u16,
    world_milliunits: [i64; 3],
}

impl AtlasObjectives {
    pub fn from_canonical_json(
        payload: &[u8],
        expected_map_id: &str,
        expected_bsp_sha256: &str,
        expected_atlas_sha256: &str,
        expected_origin: AtlasOrigin,
        graph: &L1Graph,
    ) -> AtlasResult<Self> {
        let value: serde_json::Value = serde_json::from_slice(payload)?;
        let mut canonical = serde_json::to_vec(&value)?;
        canonical.push(b'\n');
        if canonical != payload {
            return Err(AtlasError::InvalidFormat(
                "objective artifact is not canonical compact sorted JSON plus LF".to_owned(),
            ));
        }
        let document: ObjectiveDocument = serde_json::from_value(value)?;
        if document.schema != OBJECTIVE_SCHEMA
            || document.canonical_map_id != expected_map_id
            || document.bsp_sha256 != expected_bsp_sha256
            || document.atlas_sha256 != expected_atlas_sha256
            || AtlasOrigin(document.origin) != expected_origin
        {
            return Err(AtlasError::InvalidFormat(
                "objective artifact identity differs from admitted Atlas/map".to_owned(),
            ));
        }
        if document.objectives.len() > OBJECTIVE_LIMIT {
            return Err(AtlasError::LimitExceeded(format!(
                "objective count {} > {OBJECTIVE_LIMIT}",
                document.objectives.len()
            )));
        }
        let mut objectives = Vec::with_capacity(document.objectives.len());
        for record in document.objectives {
            let class = ObjectiveClass::from_name(&record.class)?;
            if ObjectiveClass::from_classname(&record.classname) != Some(class) {
                return Err(AtlasError::InvalidFormat(format!(
                    "objective {} classname/class mapping is invalid",
                    record.objective_id
                )));
            }
            if record.classname.is_empty()
                || record.classname.len() > 128
                || !record
                    .classname
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
            {
                return Err(AtlasError::InvalidFormat(format!(
                    "objective {} classname is not canonical",
                    record.objective_id
                )));
            }
            let target_l1 =
                GridIndex::new(record.l1_index[0], record.l1_index[1], record.l1_index[2]);
            let node_ordinal = graph.node_ordinal(target_l1).ok_or_else(|| {
                AtlasError::InvalidFormat(format!(
                    "objective {} target is not an admitted L1 node",
                    record.objective_id
                ))
            })?;
            let node = graph.nodes()[node_ordinal];
            if node.flags & NodeFlags::SUPPORTED_FLOOR == 0
                || node.flags & (NodeFlags::STANDING_PASSABLE | NodeFlags::CROUCHED_PASSABLE) == 0
            {
                return Err(AtlasError::InvalidFormat(format!(
                    "objective {} target is not supported/passable",
                    record.objective_id
                )));
            }
            let world_point = record.world_milliunits.map(|value| value as f64 / 1000.0);
            let center = expected_origin.center(target_l1, AtlasLevel::L1);
            let distance = world_point
                .iter()
                .zip(center)
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f64>()
                .sqrt();
            if !distance.is_finite() || distance > OBJECTIVE_TARGET_MAX_DISTANCE {
                return Err(AtlasError::InvalidFormat(format!(
                    "objective {} is too far from its admitted L1 target",
                    record.objective_id
                )));
            }
            objectives.push(AtlasObjective {
                objective_id: record.objective_id,
                class,
                classname: record.classname,
                world_milliunits: record.world_milliunits,
                target_l1,
                risk: record.risk,
                confidence: record.confidence,
            });
        }
        if objectives
            .windows(2)
            .any(|pair| pair[0].objective_id >= pair[1].objective_id)
        {
            return Err(AtlasError::InvalidFormat(
                "objectives are not strictly ordered by stable ID".to_owned(),
            ));
        }
        Ok(Self {
            canonical_map_id: document.canonical_map_id,
            bsp_sha256: document.bsp_sha256,
            atlas_sha256: document.atlas_sha256,
            origin: expected_origin,
            objectives,
        })
    }

    pub fn sha256(payload: &[u8]) -> String {
        sha256_hex(payload)
    }

    pub fn resident_bytes_estimate(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.objectives.capacity() * std::mem::size_of::<AtlasObjective>()
            + self
                .objectives
                .iter()
                .map(|objective| objective.classname.capacity())
                .sum::<usize>()
            + self.canonical_map_id.capacity()
            + self.bsp_sha256.capacity()
            + self.atlas_sha256.capacity()
    }
}

impl ObjectiveGuide {
    pub(crate) fn build(objectives: &AtlasObjectives, graph: &L1Graph) -> AtlasResult<Self> {
        let mut fields: [Option<ObjectiveField>; OBJECTIVE_CLASS_COUNT] =
            std::array::from_fn(|_| None);
        for (class_number, field) in fields.iter_mut().enumerate() {
            let class = ObjectiveClass::try_from(class_number as u8)?;
            let sources: Vec<_> = objectives
                .objectives
                .iter()
                .enumerate()
                .filter(|(_, objective)| objective.class == class)
                .collect();
            if !sources.is_empty() {
                *field = Some(ObjectiveField::build(graph, &sources)?);
            }
        }
        Ok(Self { fields })
    }

    pub(crate) fn resident_bytes_estimate(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .fields
                .iter()
                .flatten()
                .map(|field| field.cells.capacity() * std::mem::size_of::<Option<RouteCell>>())
                .sum::<usize>()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn features(
        &self,
        objectives: &AtlasObjectives,
        graph: &L1Graph,
        origin: AtlasOrigin,
        current: usize,
        world_position: [f64; 3],
        yaw_degrees: f32,
        beliefs: &[ObjectiveBelief],
    ) -> AtlasResult<GuideFeatureBlock> {
        let mut admitted_beliefs = BTreeMap::new();
        for belief in beliefs {
            if !belief.availability_belief.is_finite()
                || !(0.0..=1.0).contains(&belief.availability_belief)
                || admitted_beliefs
                    .insert(belief.objective_id, belief.availability_belief)
                    .is_some()
            {
                return Err(AtlasError::InvalidFormat(
                    "objective beliefs are invalid or duplicate".to_owned(),
                ));
            }
            if objectives
                .objectives
                .binary_search_by_key(&belief.objective_id, |objective| objective.objective_id)
                .is_err()
            {
                return Err(AtlasError::InvalidFormat(format!(
                    "objective belief {} is not in the admitted artifact",
                    belief.objective_id
                )));
            }
        }
        let mut candidates = Vec::with_capacity(OBJECTIVE_CLASS_COUNT);
        for field in self.fields.iter().flatten() {
            let Some(route) = field.cells[current] else {
                continue;
            };
            let objective = &objectives.objectives[route.objective_ordinal as usize];
            let target = if route.next_ordinal as usize == current {
                objective.world_point()
            } else {
                origin.center(
                    graph.nodes()[route.next_ordinal as usize].index,
                    AtlasLevel::L1,
                )
            };
            candidates.push(GuideCandidate {
                class: objective.class,
                world_point: target,
                cost_q8: route.cost_q8,
                risk: route.risk,
                confidence: route.confidence,
                availability_belief: admitted_beliefs
                    .get(&objective.objective_id)
                    .copied()
                    .unwrap_or(DEFAULT_AVAILABILITY_BELIEF),
            });
        }
        pack_guide_features(origin, world_position, yaw_degrees, &candidates)
    }
}

impl ObjectiveField {
    fn build(graph: &L1Graph, sources: &[(usize, &AtlasObjective)]) -> AtlasResult<Self> {
        let mut incoming = vec![Vec::new(); graph.nodes().len()];
        for source in 0..graph.nodes().len() {
            for edge in graph.outgoing(source).unwrap_or_default() {
                if matches!(edge.edge_type, EdgeType::Hook | EdgeType::Mover) {
                    continue;
                }
                incoming[edge.target as usize].push((source, *edge));
            }
        }
        let mut cells = vec![None; graph.nodes().len()];
        let mut heap = BinaryHeap::new();
        for (objective_ordinal, objective) in sources {
            let target = graph.node_ordinal(objective.target_l1).ok_or_else(|| {
                AtlasError::InvalidFormat("objective field target disappeared".to_owned())
            })?;
            let cell = RouteCell {
                cost_q8: 0,
                risk: objective.risk,
                confidence: objective.confidence.min(graph.nodes()[target].confidence),
                objective_ordinal: *objective_ordinal as u32,
                next_ordinal: target as u32,
            };
            if cells[target].is_none_or(|current| {
                route_rank(cell, objective.objective_id)
                    < route_rank(current, sources_id(current, sources))
            }) {
                cells[target] = Some(cell);
                heap.push(Reverse((route_rank(cell, objective.objective_id), target)));
            }
        }
        while let Some(Reverse((rank, node))) = heap.pop() {
            let Some(current) = cells[node] else {
                continue;
            };
            let objective_id = objectives_id(current, sources);
            if rank != route_rank(current, objective_id) {
                continue;
            }
            for (predecessor, edge) in &incoming[node] {
                let Some(cost_q8) = current.cost_q8.checked_add(edge.cost) else {
                    continue;
                };
                if cost_q8 == COST_INFINITY {
                    continue;
                }
                let candidate = RouteCell {
                    cost_q8,
                    risk: current.risk.max(edge.risk),
                    confidence: current
                        .confidence
                        .min(edge.confidence)
                        .min(graph.nodes()[*predecessor].confidence),
                    objective_ordinal: current.objective_ordinal,
                    next_ordinal: node as u32,
                };
                let candidate_rank = route_rank(candidate, objective_id);
                let replace = cells[*predecessor].is_none_or(|existing| {
                    candidate_rank < route_rank(existing, objectives_id(existing, sources))
                });
                if replace {
                    cells[*predecessor] = Some(candidate);
                    heap.push(Reverse((candidate_rank, *predecessor)));
                }
            }
        }
        Ok(Self { cells })
    }
}

fn objectives_id(cell: RouteCell, sources: &[(usize, &AtlasObjective)]) -> u32 {
    let ordinal = cell.objective_ordinal as usize;
    sources
        .iter()
        .find_map(|(candidate, objective)| {
            (*candidate == ordinal).then_some(objective.objective_id)
        })
        .expect("route cells retain an admitted class source")
}

fn sources_id(cell: RouteCell, sources: &[(usize, &AtlasObjective)]) -> u32 {
    objectives_id(cell, sources)
}

fn route_rank(cell: RouteCell, objective_id: u32) -> RouteRank {
    RouteRank {
        cost_q8: cell.cost_q8,
        risk: cell.risk,
        confidence: Reverse(cell.confidence),
        objective_id,
        next_ordinal: cell.next_ordinal,
    }
}
