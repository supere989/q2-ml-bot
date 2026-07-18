use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use super::guide::{normalize_cost, yaw_local_unit};
use super::{
    AtlasError, AtlasLevel, AtlasOrigin, AtlasResult, COST_INFINITY, EdgeRecord, EdgeType,
    GridIndex, L1Graph, NodeFlags,
};

pub const RECOVERY_FEATURE_WIDTH: usize = 16;
pub const RECOVERY_REPAIR_NODE_LIMIT: usize = 4096;
/// Frozen teacher/debug walking horizon at the authoritative 10 Hz cadence.
/// This value never enters the public recovery feature block or static solve.
pub const HOOK_RECOVERY_WALK_BUDGET_TICKS: u32 = 15;
pub const RECOVERY_GAME_TICK_HZ: u32 = 10;
pub const RECOVERY_WALK_SPEED_Q8_PER_SECOND: u32 = 300 * 256;
pub const RECOVERY_WALK_DISTANCE_Q8_PER_TICK: u32 =
    RECOVERY_WALK_SPEED_Q8_PER_SECOND / RECOVERY_GAME_TICK_HZ;
/// "Current two-L2-cell neighborhood" is frozen as a Chebyshev radius of two
/// L2 cells around the query cell. Sparse enumeration still hard-fails above
/// the independent 4096-node work ceiling.
pub const RECOVERY_REPAIR_L2_RADIUS: i32 = 2;
pub const RECOVERY_COST_WORLD_SCALE: f32 = 4096.0;
pub const RECOVERY_CLEARANCE_WORLD_SCALE: f32 = 128.0;
pub const RECOVERY_TTI_SCALE_SECONDS: f32 = 5.0;
/// One half L1 cell (8 world units), in Q8, from a cell center to a shared
/// hazard/safe face. Boundary cells therefore never encode ambiguous zero.
pub const HAZARD_CLEARANCE_BOUNDARY_Q8: u32 = 8 * 256;
/// No admitted static path from a safe cell to any hazard boundary.
pub const HAZARD_CLEARANCE_UNREACHABLE_SAFE: i32 = i32::MAX;
/// No admitted static path from a hazardous cell to any safe boundary.
pub const HAZARD_CLEARANCE_UNREACHABLE_HAZARD: i32 = i32::MIN;
pub const RECOVERY_EVIDENCE_SCHEMA: &str = "q2-recovery-evidence-v1";
pub const RECOVERY_EVIDENCE_FIELD_NAMES: [&str; 8] = [
    "l1_index",
    "cost_to_safety_q8",
    "signed_safe_clearance_q8",
    "hazard_types",
    "hazard_severity",
    "atlas_region_id",
    "hazard_component_id",
    "confidence",
];

/// Frozen public five-bit collapse. Richer Atlas-internal classifications may
/// exist, but the policy sees only these stable groups.
pub struct PolicyHazardBits;

impl PolicyHazardBits {
    pub const LAVA: u16 = 1 << 0;
    pub const SLIME: u16 = 1 << 1;
    pub const HURT: u16 = 1 << 2;
    pub const VOID_OR_LETHAL_DROP: u16 = 1 << 3;
    pub const CRUSH_OR_CURRENT: u16 = 1 << 4;
    pub const MASK: u16 = (1 << 5) - 1;
}

#[derive(Clone, Debug, Default)]
pub struct RecoveryOverlay {
    pub blocked_nodes: BTreeSet<GridIndex>,
    pub dynamic_penalty_q8: BTreeMap<GridIndex, u32>,
    /// Mover/blocker identities that are currently traversable. A zero blocker
    /// identity is never accepted as an enabled dynamic edge.
    pub enabled_mover_blockers: BTreeSet<u32>,
}

impl RecoveryOverlay {
    pub fn validate(&self) -> AtlasResult<()> {
        if self
            .dynamic_penalty_q8
            .values()
            .any(|penalty| *penalty == COST_INFINITY)
            || self.enabled_mover_blockers.contains(&0)
        {
            return Err(AtlasError::InvalidFormat(
                "recovery overlay contains an infinite penalty or zero mover identity".to_owned(),
            ));
        }
        Ok(())
    }

    fn is_empty(&self) -> bool {
        self.blocked_nodes.is_empty()
            && self.dynamic_penalty_q8.is_empty()
            && self.enabled_mover_blockers.is_empty()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RecoveryQuery<'a> {
    pub world_position: [f64; 3],
    pub yaw_degrees: f32,
    pub overlay: &'a RecoveryOverlay,
    /// Zero/None means unavailable. The runtime does not guess approach
    /// direction from scalar clearance alone.
    pub time_to_impact_seconds: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecoveryFeatureBlock {
    pub values: [f32; RECOVERY_FEATURE_WIDTH],
    pub repaired_nodes: usize,
    pub evidence: RecoveryEvidence,
}

/// Raw, private reward/teacher evidence from the same admitted L1 lookup as
/// the public normalized Recovery16 block. Values retain Atlas fixed-point
/// units and sentinels; consumers must not reconstruct them from policy slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecoveryEvidence {
    pub l1_index: GridIndex,
    /// Q8 path cost. [`COST_INFINITY`] is a valid unreachable value.
    pub cost_to_safety_q8: u32,
    /// Signed Q8 distance from the hazard/safety boundary; negative is inside.
    pub signed_safe_clearance_q8: i32,
    pub hazard_types: u16,
    pub hazard_severity: u8,
    /// Atlas L1 SCC/traversability identity. This is never a causal hazard ID.
    pub atlas_region_id: u32,
    /// Deterministic Atlas hazard basin. Zero means no reachable static hazard.
    pub hazard_component_id: u32,
    pub confidence: u16,
}

/// Private teacher/debug result for one admitted L1 recovery pose.  The
/// policy never receives this structure or a derived ``must_hook`` bit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HookNecessityEvidence {
    pub walking_budget_ticks: u32,
    pub evaluated_hook_edges: u32,
    pub walking_reaches_safety_within_budget: bool,
    pub hook_path_reaches_safety: bool,
    pub hook_lowers_recovery_cost: bool,
    pub hook_was_necessary: bool,
}

impl RecoveryEvidence {
    pub fn hazard_component_epoch(self, map_epoch: u64) -> u64 {
        if self.hazard_component_id == 0 {
            0
        } else {
            map_epoch
        }
    }
}

/// Map-static hazard components plus their deterministic recovery basins.
/// IDs are 1-based in minimum `(iz,iy,ix)` seed order; zero is reserved for a
/// safe graph component with no admitted hazard seed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HazardComponentField {
    ids: Vec<u32>,
    component_count: u32,
}

impl HazardComponentField {
    pub fn build(graph: &L1Graph) -> AtlasResult<Self> {
        let adjacency = static_undirected_adjacency(graph)?;
        let hazardous: Vec<_> = graph.nodes().iter().map(is_hazardous).collect();
        let mut ids = vec![0_u32; graph.nodes().len()];
        let mut component_count = 0_u32;
        for seed in 0..graph.nodes().len() {
            if !hazardous[seed] || ids[seed] != 0 {
                continue;
            }
            component_count = component_count.checked_add(1).ok_or_else(|| {
                AtlasError::LimitExceeded("hazard component count overflow".to_owned())
            })?;
            ids[seed] = component_count;
            let mut pending = vec![seed];
            while let Some(source) = pending.pop() {
                for &(target, _) in &adjacency[source] {
                    if hazardous[target] && ids[target] == 0 {
                        ids[target] = component_count;
                        pending.push(target);
                    }
                }
            }
        }

        // Extend each component through safe space as a deterministic nearest
        // basin. This keeps the entry component visible after first arrival at
        // positive clearance. Equal-distance ties choose the lower stable ID.
        let mut distances = vec![u64::MAX; graph.nodes().len()];
        let mut routes = BinaryHeap::new();
        for node in 0..graph.nodes().len() {
            if ids[node] != 0 {
                distances[node] = 0;
                routes.push(Reverse((0_u64, ids[node], node)));
            }
        }
        while let Some(Reverse((distance, component_id, source))) = routes.pop() {
            if (distances[source], ids[source]) != (distance, component_id) {
                continue;
            }
            for &(target, edge_cost) in &adjacency[source] {
                if hazardous[target] && ids[target] != component_id {
                    continue;
                }
                let candidate = distance.checked_add(u64::from(edge_cost)).ok_or_else(|| {
                    AtlasError::LimitExceeded("hazard-component distance overflow".to_owned())
                })?;
                if (candidate, component_id) < (distances[target], ids[target]) {
                    distances[target] = candidate;
                    ids[target] = component_id;
                    routes.push(Reverse((candidate, component_id, target)));
                }
            }
        }
        Ok(Self {
            ids,
            component_count,
        })
    }

    pub fn component_count(&self) -> u32 {
        self.component_count
    }

    pub fn id_at_ordinal(&self, ordinal: usize) -> AtlasResult<u32> {
        self.ids.get(ordinal).copied().ok_or_else(|| {
            AtlasError::InvalidFormat("hazard component L1 ordinal is invalid".to_owned())
        })
    }

    pub fn resident_bytes_estimate(&self) -> usize {
        std::mem::size_of::<Self>() + self.ids.capacity() * std::mem::size_of::<u32>()
    }
}

/// Deterministic fixed-point multi-source solve over public walking edges.
/// Hook, mover-dependent, and blocker-dependent edges are excluded offline.
pub fn solve_static_costs(graph: &L1Graph) -> AtlasResult<Vec<u32>> {
    let nodes = graph.nodes();
    let mut reverse = vec![Vec::<(usize, u32)>::new(); nodes.len()];
    for source in 0..nodes.len() {
        let outgoing = graph
            .outgoing(source)
            .ok_or_else(|| AtlasError::InvalidFormat("L1 CSR source is unavailable".to_owned()))?;
        for edge in outgoing {
            if static_edge(edge) {
                reverse[edge.target as usize].push((source, edge.cost));
            }
        }
    }
    for incoming in &mut reverse {
        incoming.sort_unstable();
    }

    let mut costs = vec![COST_INFINITY; nodes.len()];
    let mut pending = BinaryHeap::new();
    for (ordinal, node) in nodes.iter().enumerate() {
        if node.flags & NodeFlags::SAFE_TO_STAND != 0 {
            costs[ordinal] = 0;
            pending.push(Reverse((0_u32, ordinal)));
        }
    }
    while let Some(Reverse((cost, target))) = pending.pop() {
        if costs[target] != cost {
            continue;
        }
        for &(source, edge_cost) in &reverse[target] {
            let Some(candidate) = cost.checked_add(edge_cost) else {
                continue;
            };
            if candidate < costs[source] {
                costs[source] = candidate;
                pending.push(Reverse((candidate, source)));
            }
        }
    }
    Ok(costs)
}

pub fn install_static_costs(graph: &mut L1Graph) -> AtlasResult<()> {
    let costs = solve_static_costs(graph)?;
    graph.set_costs_to_safety(&costs)
}

/// Evaluate the frozen hook-necessity teacher label without changing the
/// public walking-only solve.  A label is true only when ordinary admitted
/// movement cannot reach a safe node within the Q8 distance available over 15
/// ticks at the manifest-bound 10 Hz/300 unit-per-second physics identity, an
/// admitted current hook action can reach safety, and that route has lower Q8
/// cost than the best ordinary route (or the ordinary route is unreachable).
pub fn evaluate_hook_necessity(
    graph: &L1Graph,
    current: usize,
    overlay: &RecoveryOverlay,
) -> AtlasResult<HookNecessityEvidence> {
    overlay.validate()?;
    let current_node = graph.nodes().get(current).ok_or_else(|| {
        AtlasError::InvalidFormat("hook-necessity L1 ordinal is invalid".to_owned())
    })?;
    if overlay.blocked_nodes.contains(&current_node.index) {
        return Err(AtlasError::InvalidFormat(
            "hook-necessity pose is dynamically blocked".to_owned(),
        ));
    }
    let evaluated_hook_edges = graph
        .outgoing(current)
        .into_iter()
        .flatten()
        .filter(|edge| edge.edge_type == EdgeType::Hook && edge.blocker == 0)
        .count();
    let evaluated_hook_edges = u32::try_from(evaluated_hook_edges).map_err(|_| {
        AtlasError::LimitExceeded("hook-necessity edge count exceeds u32".to_owned())
    })?;
    let walking_route = shortest_recovery_route(graph, current, overlay)?;
    let walking_distance = shortest_walking_distance_q8(graph, current, overlay)?;
    let walking_budget_q8 =
        u64::from(HOOK_RECOVERY_WALK_BUDGET_TICKS) * u64::from(RECOVERY_WALK_DISTANCE_Q8_PER_TICK);
    let walking_reaches_safety_within_budget =
        walking_distance.is_some_and(|distance| distance <= walking_budget_q8);
    let hook_route = shortest_current_hook_route(graph, current, overlay)?;
    let hook_path_reaches_safety = hook_route.is_some();
    let hook_lowers_recovery_cost =
        hook_route.is_some_and(|hook| walking_route.is_none_or(|walking| hook.0 < walking.0));
    let hook_was_necessary = !walking_reaches_safety_within_budget
        && hook_path_reaches_safety
        && hook_lowers_recovery_cost;
    Ok(HookNecessityEvidence {
        walking_budget_ticks: HOOK_RECOVERY_WALK_BUDGET_TICKS,
        evaluated_hook_edges,
        walking_reaches_safety_within_budget,
        hook_path_reaches_safety,
        hook_lowers_recovery_cost,
        hook_was_necessary,
    })
}

/// Return `(repaired_cost_q8, traversal_distance_q8)` for the deterministic
/// least-repaired-cost ordinary route. Traversal distance is accumulated from
/// the graph's frozen Q8 world-distance cost without dynamic hazard penalty;
/// the 10 Hz conversion is therefore not inferred from edge count.
fn shortest_recovery_route(
    graph: &L1Graph,
    current: usize,
    overlay: &RecoveryOverlay,
) -> AtlasResult<Option<(u64, u64)>> {
    let mut best = vec![(u64::MAX, u64::MAX); graph.nodes().len()];
    let mut pending = BinaryHeap::new();
    best[current] = (0, 0);
    pending.push(Reverse((
        0_u64,
        0_u64,
        source_sort_index(graph, current),
        current,
    )));
    while let Some(Reverse((cost, distance, _index, source))) = pending.pop() {
        if best[source] != (cost, distance) {
            continue;
        }
        if graph.nodes()[source].flags & NodeFlags::SAFE_TO_STAND != 0 {
            return Ok(Some((cost, distance)));
        }
        for edge in graph.outgoing(source).into_iter().flatten() {
            if !runtime_edge(edge, overlay) || edge.cost == COST_INFINITY {
                continue;
            }
            let target = edge.target as usize;
            if target >= graph.nodes().len() {
                return Err(AtlasError::InvalidFormat(
                    "hook-necessity edge target is invalid".to_owned(),
                ));
            }
            if overlay.blocked_nodes.contains(&graph.nodes()[target].index) {
                continue;
            }
            let penalty = overlay
                .dynamic_penalty_q8
                .get(&graph.nodes()[source].index)
                .copied()
                .unwrap_or(0);
            let Some(candidate_cost) = cost
                .checked_add(u64::from(edge.cost))
                .and_then(|value| value.checked_add(u64::from(penalty)))
            else {
                continue;
            };
            let Some(candidate_distance) = distance.checked_add(u64::from(edge.cost)) else {
                continue;
            };
            if (candidate_cost, candidate_distance) < best[target] {
                best[target] = (candidate_cost, candidate_distance);
                pending.push(Reverse((
                    candidate_cost,
                    candidate_distance,
                    source_sort_index(graph, target),
                    target,
                )));
            }
        }
    }
    Ok(None)
}

fn shortest_walking_distance_q8(
    graph: &L1Graph,
    current: usize,
    overlay: &RecoveryOverlay,
) -> AtlasResult<Option<u64>> {
    let mut best = vec![u64::MAX; graph.nodes().len()];
    let mut pending = BinaryHeap::new();
    best[current] = 0;
    pending.push(Reverse((0_u64, source_sort_index(graph, current), current)));
    while let Some(Reverse((distance, _index, source))) = pending.pop() {
        if best[source] != distance {
            continue;
        }
        if graph.nodes()[source].flags & NodeFlags::SAFE_TO_STAND != 0 {
            return Ok(Some(distance));
        }
        for edge in graph.outgoing(source).into_iter().flatten() {
            if !runtime_edge(edge, overlay) || edge.cost == COST_INFINITY {
                continue;
            }
            let target = edge.target as usize;
            if target >= graph.nodes().len() {
                return Err(AtlasError::InvalidFormat(
                    "hook-necessity edge target is invalid".to_owned(),
                ));
            }
            if overlay.blocked_nodes.contains(&graph.nodes()[target].index) {
                continue;
            }
            let Some(candidate) = distance.checked_add(u64::from(edge.cost)) else {
                continue;
            };
            if candidate < best[target] {
                best[target] = candidate;
                pending.push(Reverse((
                    candidate,
                    source_sort_index(graph, target),
                    target,
                )));
            }
        }
    }
    Ok(None)
}

/// Evaluate only an admissible hook action available at the current pose.
/// Lookahead hook edges cannot manufacture a positive label for the current
/// action. After the first hook, the remainder is ordinary recovery.
fn shortest_current_hook_route(
    graph: &L1Graph,
    current: usize,
    overlay: &RecoveryOverlay,
) -> AtlasResult<Option<(u64, u64)>> {
    let source_penalty = overlay
        .dynamic_penalty_q8
        .get(&graph.nodes()[current].index)
        .copied()
        .unwrap_or(0);
    let mut candidates = Vec::new();
    for edge in graph.outgoing(current).into_iter().flatten() {
        if edge.edge_type != EdgeType::Hook || edge.blocker != 0 || edge.cost == COST_INFINITY {
            continue;
        }
        let target = edge.target as usize;
        if target >= graph.nodes().len() {
            return Err(AtlasError::InvalidFormat(
                "hook-necessity edge target is invalid".to_owned(),
            ));
        }
        if overlay.blocked_nodes.contains(&graph.nodes()[target].index) {
            continue;
        }
        let continuation = shortest_recovery_route(graph, target, overlay)?;
        let Some((continuation_cost, continuation_distance)) = continuation else {
            continue;
        };
        let Some(cost) = continuation_cost
            .checked_add(u64::from(edge.cost))
            .and_then(|value| value.checked_add(u64::from(source_penalty)))
        else {
            continue;
        };
        let Some(distance) = continuation_distance.checked_add(u64::from(edge.cost)) else {
            continue;
        };
        candidates.push((cost, distance, graph.nodes()[target].index));
    }
    candidates.sort_unstable();
    Ok(candidates
        .first()
        .map(|candidate| (candidate.0, candidate.1)))
}

fn source_sort_index(graph: &L1Graph, ordinal: usize) -> GridIndex {
    graph.nodes()[ordinal].index
}

/// Deterministic signed distance to the nearest hazard/safe boundary over the
/// admitted, map-static L1 graph. Hook, mover, and blocker-dependent edges are
/// excluded, and directionality is ignored for geometric boundary clearance.
pub fn solve_static_hazard_clearances(graph: &L1Graph) -> AtlasResult<Vec<i32>> {
    let nodes = graph.nodes();
    let adjacency = static_undirected_adjacency(graph)?;
    let hazardous: Vec<_> = nodes.iter().map(is_hazardous).collect();
    let mut distances = vec![u64::MAX; nodes.len()];
    let mut pending = BinaryHeap::new();
    for source in 0..nodes.len() {
        if adjacency[source]
            .iter()
            .any(|(target, _)| hazardous[*target] != hazardous[source])
        {
            distances[source] = u64::from(HAZARD_CLEARANCE_BOUNDARY_Q8);
            pending.push(Reverse((distances[source], source)));
        }
    }
    while let Some(Reverse((distance, source))) = pending.pop() {
        if distances[source] != distance {
            continue;
        }
        for &(target, edge_cost) in &adjacency[source] {
            if hazardous[target] != hazardous[source] {
                continue;
            }
            let candidate = distance.checked_add(u64::from(edge_cost)).ok_or_else(|| {
                AtlasError::LimitExceeded("hazard-clearance distance overflow".to_owned())
            })?;
            if candidate < distances[target] {
                distances[target] = candidate;
                pending.push(Reverse((candidate, target)));
            }
        }
    }
    distances
        .into_iter()
        .zip(hazardous)
        .map(|(distance, hazard)| {
            if distance == u64::MAX {
                return Ok(if hazard {
                    HAZARD_CLEARANCE_UNREACHABLE_HAZARD
                } else {
                    HAZARD_CLEARANCE_UNREACHABLE_SAFE
                });
            }
            let finite = i32::try_from(distance).map_err(|_| {
                AtlasError::LimitExceeded(
                    "finite hazard clearance exceeds signed Q8 encoding".to_owned(),
                )
            })?;
            Ok(if hazard { -finite } else { finite })
        })
        .collect()
}

pub fn install_static_hazard_clearances(graph: &mut L1Graph) -> AtlasResult<()> {
    let clearances = solve_static_hazard_clearances(graph)?;
    graph.set_hazard_clearances(&clearances)
}

pub fn validate_static_hazard_clearances(graph: &L1Graph) -> AtlasResult<()> {
    let expected = solve_static_hazard_clearances(graph)?;
    for (node, expected_clearance) in graph.nodes().iter().zip(expected) {
        if node.hazard_clearance != expected_clearance {
            return Err(AtlasError::InvalidFormat(format!(
                "L1 node {:?} hazard clearance {} != deterministic solve {expected_clearance}",
                node.index, node.hazard_clearance
            )));
        }
    }
    Ok(())
}

fn is_hazardous(node: &super::L1Node) -> bool {
    node.hazard_types != 0 || node.hazard_severity != 0
}

fn static_undirected_adjacency(graph: &L1Graph) -> AtlasResult<Vec<Vec<(usize, u32)>>> {
    let mut adjacency = vec![Vec::<(usize, u32)>::new(); graph.nodes().len()];
    for source in 0..graph.nodes().len() {
        for edge in graph.outgoing(source).into_iter().flatten() {
            if !static_edge(edge) || edge.cost == COST_INFINITY {
                continue;
            }
            let target = edge.target as usize;
            if target >= graph.nodes().len() {
                return Err(AtlasError::InvalidFormat(
                    "hazard adjacency target is invalid".to_owned(),
                ));
            }
            adjacency[source].push((target, edge.cost));
            adjacency[target].push((source, edge.cost));
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    Ok(adjacency)
}

/// Fail closed unless stored L1 costs are exactly the authoritative Rust solve
/// and every finite unsafe cell descends or is explicitly mover-gated.
pub fn validate_static_costs(graph: &L1Graph) -> AtlasResult<()> {
    let expected = solve_static_costs(graph)?;
    for (ordinal, (node, expected_cost)) in graph.nodes().iter().zip(expected).enumerate() {
        if node.cost_to_safety != expected_cost {
            return Err(AtlasError::InvalidFormat(format!(
                "L1 node {:?} cost-to-safety {} != deterministic solve {expected_cost}",
                node.index, node.cost_to_safety
            )));
        }
        if node.cost_to_safety == COST_INFINITY
            || node.flags & NodeFlags::SAFE_TO_STAND != 0
            || node.flags & NodeFlags::MOVER_GATED_PLATEAU != 0
        {
            continue;
        }
        let descends = graph
            .outgoing(ordinal)
            .into_iter()
            .flatten()
            .filter(|edge| static_edge(edge))
            .any(|edge| graph.nodes()[edge.target as usize].cost_to_safety < node.cost_to_safety);
        if !descends {
            return Err(AtlasError::InvalidFormat(format!(
                "finite unsafe L1 node {:?} has no decreasing-cost edge or mover plateau",
                node.index
            )));
        }
    }
    Ok(())
}

pub fn recovery_features(
    origin: AtlasOrigin,
    graph: &L1Graph,
    query: RecoveryQuery<'_>,
) -> AtlasResult<RecoveryFeatureBlock> {
    let current = resolve_recovery_node(origin, graph, query.world_position)?;
    let hazard_components = HazardComponentField::build(graph)?;
    recovery_features_at(origin, graph, &hazard_components, current, query)
}

pub(crate) fn resolve_recovery_node(
    origin: AtlasOrigin,
    graph: &L1Graph,
    world_position: [f64; 3],
) -> AtlasResult<usize> {
    if world_position.iter().any(|value| !value.is_finite()) {
        return Err(AtlasError::Coordinate(
            "recovery query has a non-finite position".to_owned(),
        ));
    }
    let index = origin.index(world_position, AtlasLevel::L1)?;
    graph.node_ordinal(index).ok_or_else(|| {
        AtlasError::InvalidFormat(format!("recovery pose has no exact L1 node {index:?}"))
    })
}

pub(crate) fn recovery_features_at(
    origin: AtlasOrigin,
    graph: &L1Graph,
    hazard_components: &HazardComponentField,
    current: usize,
    query: RecoveryQuery<'_>,
) -> AtlasResult<RecoveryFeatureBlock> {
    if query.world_position.iter().any(|value| !value.is_finite()) || !query.yaw_degrees.is_finite()
    {
        return Err(AtlasError::Coordinate(
            "recovery query has a non-finite pose".to_owned(),
        ));
    }
    if query
        .time_to_impact_seconds
        .is_some_and(|value| !value.is_finite() || value < 0.0)
    {
        return Err(AtlasError::InvalidFormat(
            "recovery time-to-impact is invalid".to_owned(),
        ));
    }
    query.overlay.validate()?;
    let index = graph
        .nodes()
        .get(current)
        .ok_or_else(|| AtlasError::InvalidFormat("recovery L1 ordinal is invalid".to_owned()))?
        .index;
    if query.overlay.blocked_nodes.contains(&index) {
        return Err(AtlasError::InvalidFormat(
            "recovery pose is dynamically blocked".to_owned(),
        ));
    }

    let repaired = if query.overlay.is_empty() {
        BTreeMap::new()
    } else {
        repair_local_costs(graph, current, query.overlay)?
    };
    let repaired_nodes = repaired.len();
    let cost_at = |ordinal: usize| {
        repaired
            .get(&ordinal)
            .copied()
            .unwrap_or(graph.nodes()[ordinal].cost_to_safety)
    };
    let current_cost = cost_at(current);

    let mut candidates = Vec::new();
    for edge in graph.outgoing(current).into_iter().flatten() {
        if !runtime_edge(edge, query.overlay) {
            continue;
        }
        let target = edge.target as usize;
        let target_node = &graph.nodes()[target];
        if query.overlay.blocked_nodes.contains(&target_node.index) {
            continue;
        }
        let target_cost = cost_at(target);
        if target_cost < current_cost
            && let Some(route_cost) = target_cost.checked_add(edge.cost)
        {
            candidates.push((route_cost, edge.edge_type as u8, target_node.index, target));
        }
    }
    candidates.sort_unstable();
    // A public "real branch" is frozen as a distinct legal first-hop target;
    // parallel edge records to the same target do not invent an alternate.
    candidates.dedup_by_key(|candidate| candidate.3);

    if current_cost != COST_INFINITY
        && graph.nodes()[current].flags & NodeFlags::SAFE_TO_STAND == 0
        && graph.nodes()[current].flags & NodeFlags::MOVER_GATED_PLATEAU == 0
        && candidates.is_empty()
    {
        return Err(AtlasError::InvalidFormat(
            "finite repaired recovery cell has no decreasing-cost edge".to_owned(),
        ));
    }

    let direction = |candidate: Option<&(u32, u8, GridIndex, usize)>| {
        candidate.map_or([0.0; 3], |candidate| {
            let target = origin.center(candidate.2, AtlasLevel::L1);
            yaw_local_unit(
                [
                    target[0] - query.world_position[0],
                    target[1] - query.world_position[1],
                    target[2] - query.world_position[2],
                ],
                query.yaw_degrees,
            )
        })
    };
    let primary = direction(candidates.first());
    let alternate = direction(candidates.get(1));
    let node = &graph.nodes()[current];
    let public_hazards = node.hazard_types & PolicyHazardBits::MASK;
    let mut values = [0.0; RECOVERY_FEATURE_WIDTH];
    for (bit, value) in values.iter_mut().enumerate().take(5) {
        *value = f32::from(public_hazards & (1 << bit) != 0);
    }
    values[5] = f32::from(node.hazard_severity) / f32::from(u8::MAX);
    values[6] = (f32::from(node.clearance_height) / RECOVERY_CLEARANCE_WORLD_SCALE).clamp(0.0, 1.0);
    values[7] = normalize_cost(current_cost, RECOVERY_COST_WORLD_SCALE);
    values[8] = f32::from(node.confidence) / f32::from(u16::MAX);
    values[9..12].copy_from_slice(&primary);
    values[12..15].copy_from_slice(&alternate);
    values[15] = if public_hazards == 0 && node.hazard_severity == 0 {
        0.0
    } else {
        query.time_to_impact_seconds.map_or(0.0, |seconds| {
            (seconds / RECOVERY_TTI_SCALE_SECONDS).clamp(0.0, 1.0)
        })
    };
    Ok(RecoveryFeatureBlock {
        values,
        repaired_nodes,
        evidence: RecoveryEvidence {
            l1_index: node.index,
            cost_to_safety_q8: current_cost,
            signed_safe_clearance_q8: node.hazard_clearance,
            hazard_types: node.hazard_types,
            hazard_severity: node.hazard_severity,
            atlas_region_id: node.region_id,
            hazard_component_id: hazard_components.id_at_ordinal(current)?,
            confidence: node.confidence,
        },
    })
}

fn repair_local_costs(
    graph: &L1Graph,
    current: usize,
    overlay: &RecoveryOverlay,
) -> AtlasResult<BTreeMap<usize, u32>> {
    let center = graph.nodes()[current].index.parent();
    let local = graph.ordinals_in_l2_radius(
        center,
        RECOVERY_REPAIR_L2_RADIUS,
        RECOVERY_REPAIR_NODE_LIMIT,
    )?;
    let local_lookup: BTreeMap<usize, usize> = local
        .iter()
        .copied()
        .enumerate()
        .map(|(local, global)| (global, local))
        .collect();
    let mut reverse = vec![Vec::<(usize, u32)>::new(); local.len()];
    let mut costs = vec![COST_INFINITY; local.len()];
    let mut pending = BinaryHeap::new();

    for (local_source, &global_source) in local.iter().enumerate() {
        let source_node = &graph.nodes()[global_source];
        if overlay.blocked_nodes.contains(&source_node.index) {
            continue;
        }
        let penalty = overlay
            .dynamic_penalty_q8
            .get(&source_node.index)
            .copied()
            .unwrap_or(0);
        if source_node.flags & NodeFlags::SAFE_TO_STAND != 0 && penalty == 0 {
            costs[local_source] = 0;
            pending.push(Reverse((0_u32, local_source)));
        }
        for edge in graph.outgoing(global_source).into_iter().flatten() {
            if !runtime_edge(edge, overlay) {
                continue;
            }
            let global_target = edge.target as usize;
            let target_node = &graph.nodes()[global_target];
            if overlay.blocked_nodes.contains(&target_node.index) {
                continue;
            }
            if let Some(&local_target) = local_lookup.get(&global_target) {
                reverse[local_target].push((local_source, edge.cost));
            } else if target_node.cost_to_safety != COST_INFINITY
                && let Some(candidate) = target_node
                    .cost_to_safety
                    .checked_add(edge.cost)
                    .and_then(|value| value.checked_add(penalty))
                && candidate < costs[local_source]
            {
                costs[local_source] = candidate;
                pending.push(Reverse((candidate, local_source)));
            }
        }
    }
    for incoming in &mut reverse {
        incoming.sort_unstable();
    }
    while let Some(Reverse((cost, local_target))) = pending.pop() {
        if costs[local_target] != cost {
            continue;
        }
        for &(local_source, edge_cost) in &reverse[local_target] {
            let source_node = &graph.nodes()[local[local_source]];
            let penalty = overlay
                .dynamic_penalty_q8
                .get(&source_node.index)
                .copied()
                .unwrap_or(0);
            let Some(candidate) = cost
                .checked_add(edge_cost)
                .and_then(|value| value.checked_add(penalty))
            else {
                continue;
            };
            if candidate < costs[local_source] {
                costs[local_source] = candidate;
                pending.push(Reverse((candidate, local_source)));
            }
        }
    }
    Ok(local.into_iter().zip(costs).collect())
}

fn static_edge(edge: &EdgeRecord) -> bool {
    edge.edge_type != EdgeType::Hook && edge.edge_type != EdgeType::Mover && edge.blocker == 0
}

fn runtime_edge(edge: &EdgeRecord, overlay: &RecoveryOverlay) -> bool {
    if edge.edge_type == EdgeType::Hook {
        return false;
    }
    if edge.edge_type == EdgeType::Mover || edge.blocker != 0 {
        return edge.blocker != 0 && overlay.enabled_mover_blockers.contains(&edge.blocker);
    }
    true
}

pub const RECOVERY_FEATURE_NAMES: [&str; RECOVERY_FEATURE_WIDTH] = [
    "recovery_hazard_lava",
    "recovery_hazard_slime",
    "recovery_hazard_hurt",
    "recovery_hazard_void_or_lethal_drop",
    "recovery_hazard_crush_or_current",
    "recovery_hazard_strength",
    "recovery_hull_clearance",
    "recovery_cost_to_safety",
    "recovery_confidence",
    "recovery_primary_forward",
    "recovery_primary_quake_right",
    "recovery_primary_up",
    "recovery_alternate_forward",
    "recovery_alternate_quake_right",
    "recovery_alternate_up",
    "recovery_time_to_impact",
];
