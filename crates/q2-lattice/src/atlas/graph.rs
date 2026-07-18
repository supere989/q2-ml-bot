use std::collections::BTreeMap;

use super::{AtlasError, AtlasLimits, AtlasResult, EdgeAdmission, GridIndex};

pub const COST_INFINITY: u32 = u32::MAX;

pub struct NodeFlags;

impl NodeFlags {
    pub const STANDING_CLEAR: u16 = 1 << 0;
    pub const CROUCHED_CLEAR: u16 = 1 << 1;
    pub const SAFE_TO_STAND: u16 = 1 << 2;
    pub const SUPPORTED_FLOOR: u16 = 1 << 3;
    pub const STANDING_PASSABLE: u16 = 1 << 4;
    pub const CROUCHED_PASSABLE: u16 = 1 << 5;
    pub const MOVER_GATED_PLATEAU: u16 = 1 << 6;
    pub const KNOWN_MASK: u16 = (1 << 7) - 1;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EdgeType {
    Walk = 0,
    StrafeWalk = 1,
    Step = 2,
    Jump = 3,
    ControlledDrop = 4,
    CrouchEnter = 5,
    CrouchHold = 6,
    CrouchExit = 7,
    WaterTransition = 8,
    Mover = 9,
    Teleporter = 10,
    Hook = 11,
    Ladder = 12,
}

impl EdgeType {
    pub(crate) fn from_u8(value: u8) -> AtlasResult<Self> {
        match value {
            0 => Ok(Self::Walk),
            1 => Ok(Self::StrafeWalk),
            2 => Ok(Self::Step),
            3 => Ok(Self::Jump),
            4 => Ok(Self::ControlledDrop),
            5 => Ok(Self::CrouchEnter),
            6 => Ok(Self::CrouchHold),
            7 => Ok(Self::CrouchExit),
            8 => Ok(Self::WaterTransition),
            9 => Ok(Self::Mover),
            10 => Ok(Self::Teleporter),
            11 => Ok(Self::Hook),
            12 => Ok(Self::Ladder),
            _ => Err(AtlasError::InvalidFormat(format!(
                "unknown L1 edge type {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Stance {
    Standing = 0,
    Crouched = 1,
    Either = 2,
    Water = 3,
}

impl Stance {
    pub(crate) fn from_u8(value: u8) -> AtlasResult<Self> {
        match value {
            0 => Ok(Self::Standing),
            1 => Ok(Self::Crouched),
            2 => Ok(Self::Either),
            3 => Ok(Self::Water),
            _ => Err(AtlasError::InvalidFormat(format!(
                "unknown L1 edge stance {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct L1Node {
    pub index: GridIndex,
    pub flags: u16,
    pub floor_normal_class: u8,
    pub clearance_height: u16,
    pub hazard_types: u16,
    pub hazard_severity: u8,
    /// Fixed point, 1/256-unit quantum. Negative values are inside a hazard.
    pub hazard_clearance: i32,
    pub cost_to_safety: u32,
    pub region_id: u32,
    pub confidence: u16,
    pub evidence: u16,
    pub contents_flags: u32,
}

impl L1Node {
    pub(crate) fn validate(&self) -> AtlasResult<()> {
        if self.flags & !NodeFlags::KNOWN_MASK != 0 {
            return Err(AtlasError::InvalidFormat(format!(
                "L1 node {:?} has unknown flags {:#x}",
                self.index,
                self.flags & !NodeFlags::KNOWN_MASK
            )));
        }
        if self.flags & NodeFlags::SAFE_TO_STAND != 0
            && self.flags & (NodeFlags::STANDING_CLEAR | NodeFlags::SUPPORTED_FLOOR)
                != (NodeFlags::STANDING_CLEAR | NodeFlags::SUPPORTED_FLOOR)
        {
            return Err(AtlasError::InvalidFormat(format!(
                "safe L1 node {:?} lacks standing clearance/support",
                self.index
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EdgeInput {
    pub source: GridIndex,
    pub target: GridIndex,
    pub edge_type: EdgeType,
    pub stance: Stance,
    pub flags: u16,
    pub blocker: u32,
    pub cost: u32,
    pub risk: u16,
    pub confidence: u16,
    pub evidence: u16,
    pub validation_version: u16,
    /// Index into a type-specific side table. `u32::MAX` means none.
    pub auxiliary: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EdgeRecord {
    pub target: u32,
    pub edge_type: EdgeType,
    pub stance: Stance,
    pub flags: u16,
    pub blocker: u32,
    pub cost: u32,
    pub risk: u16,
    pub confidence: u16,
    pub evidence: u16,
    pub validation_version: u16,
    pub auxiliary: u32,
}

impl EdgeRecord {
    pub const ENCODED_BYTES: usize = 28;

    fn sort_key(
        self,
        nodes: &[L1Node],
    ) -> (
        u8,
        GridIndex,
        Stance,
        u16,
        u32,
        u32,
        u16,
        u16,
        u16,
        u16,
        u32,
    ) {
        (
            self.edge_type as u8,
            nodes[self.target as usize].index,
            self.stance,
            self.flags,
            self.blocker,
            self.cost,
            self.risk,
            self.confidence,
            self.evidence,
            self.validation_version,
            self.auxiliary,
        )
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct L1Graph {
    nodes: Vec<L1Node>,
    offsets: Vec<u32>,
    edges: Vec<EdgeRecord>,
}

impl L1Graph {
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            offsets: vec![0],
            edges: Vec::new(),
        }
    }

    pub fn build(
        mut nodes: Vec<L1Node>,
        edges: Vec<EdgeInput>,
        admission: &EdgeAdmission,
        limits: &AtlasLimits,
    ) -> AtlasResult<Self> {
        if nodes.len() > limits.max_l1_nodes {
            return Err(AtlasError::LimitExceeded(format!(
                "L1 node count {} > {}",
                nodes.len(),
                limits.max_l1_nodes
            )));
        }
        if edges.len() > limits.max_l1_edges {
            return Err(AtlasError::LimitExceeded(format!(
                "L1 edge count {} > {}",
                edges.len(),
                limits.max_l1_edges
            )));
        }
        nodes.sort_by_key(|node| node.index);
        for node in &nodes {
            node.validate()?;
        }
        if nodes.windows(2).any(|pair| pair[0].index == pair[1].index) {
            return Err(AtlasError::InvalidFormat(
                "duplicate L1 node key".to_owned(),
            ));
        }
        if nodes.len() > u32::MAX as usize || edges.len() > u32::MAX as usize {
            return Err(AtlasError::LimitExceeded(
                "CSR u32 cardinality exceeded".to_owned(),
            ));
        }
        let node_indices: BTreeMap<_, _> = nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.index, index as u32))
            .collect();
        let mut adjacency = vec![Vec::new(); nodes.len()];
        for edge in edges {
            let source = *node_indices.get(&edge.source).ok_or_else(|| {
                AtlasError::InvalidFormat(format!(
                    "edge source {:?} is not an L1 node",
                    edge.source
                ))
            })?;
            let target = *node_indices.get(&edge.target).ok_or_else(|| {
                AtlasError::InvalidFormat(format!(
                    "edge target {:?} is not an L1 node",
                    edge.target
                ))
            })?;
            if edge.cost == COST_INFINITY {
                return Err(AtlasError::InvalidFormat(
                    "materialized traversal edge has infinite cost".to_owned(),
                ));
            }
            if edge.evidence == 0 || edge.validation_version == 0 {
                return Err(AtlasError::InvalidFormat(
                    "materialized traversal edge lacks evidence or validation version".to_owned(),
                ));
            }
            admission.validate_edge(edge.edge_type, edge.evidence, edge.validation_version)?;
            adjacency[source as usize].push(EdgeRecord {
                target,
                edge_type: edge.edge_type,
                stance: edge.stance,
                flags: edge.flags,
                blocker: edge.blocker,
                cost: edge.cost,
                risk: edge.risk,
                confidence: edge.confidence,
                evidence: edge.evidence,
                validation_version: edge.validation_version,
                auxiliary: edge.auxiliary,
            });
        }

        let mut offsets = Vec::with_capacity(nodes.len() + 1);
        let mut packed = Vec::with_capacity(adjacency.iter().map(Vec::len).sum());
        offsets.push(0);
        for mut outgoing in adjacency {
            outgoing.sort_by_key(|edge| edge.sort_key(&nodes));
            if outgoing.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(AtlasError::InvalidFormat("duplicate L1 edge".to_owned()));
            }
            packed.extend(outgoing);
            offsets.push(packed.len() as u32);
        }
        let graph = Self {
            nodes,
            offsets,
            edges: packed,
        };
        graph.validate(admission, limits)?;
        Ok(graph)
    }

    pub fn nodes(&self) -> &[L1Node] {
        &self.nodes
    }

    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub fn edges(&self) -> &[EdgeRecord] {
        &self.edges
    }

    pub fn resident_bytes_estimate(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.nodes.capacity() * std::mem::size_of::<L1Node>()
            + self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.edges.capacity() * std::mem::size_of::<EdgeRecord>()
    }

    pub fn outgoing(&self, node: usize) -> Option<&[EdgeRecord]> {
        let start = *self.offsets.get(node)? as usize;
        let end = *self.offsets.get(node + 1)? as usize;
        Some(&self.edges[start..end])
    }

    /// Resolve one canonical grid key without allocating a runtime index.
    ///
    /// L1 nodes are already stored in strict `(iz,iy,ix)` order, so binary
    /// search keeps resident Atlas memory inside the frozen budget.
    pub fn node_ordinal(&self, index: GridIndex) -> Option<usize> {
        self.nodes
            .binary_search_by_key(&index, |node| node.index)
            .ok()
    }

    /// Enumerate sparse L1 ordinals whose L2 parents fall within a fixed
    /// Chebyshev radius. The implementation performs bounded canonical row
    /// searches rather than a full hot-path node scan.
    pub fn ordinals_in_l2_radius(
        &self,
        center: GridIndex,
        radius: i32,
        limit: usize,
    ) -> AtlasResult<Vec<usize>> {
        if radius < 0 || limit == 0 {
            return Err(AtlasError::InvalidFormat(
                "L2 neighborhood radius/limit is invalid".to_owned(),
            ));
        }
        let l1_bound = |coordinate: i32, upper: bool| -> AtlasResult<i32> {
            let parent = i64::from(coordinate)
                .checked_add(if upper {
                    i64::from(radius) + 1
                } else {
                    -i64::from(radius)
                })
                .ok_or_else(|| AtlasError::Coordinate("L2 neighborhood overflow".to_owned()))?;
            let scaled = parent
                .checked_mul(4)
                .and_then(|value| {
                    if upper {
                        value.checked_sub(1)
                    } else {
                        Some(value)
                    }
                })
                .ok_or_else(|| AtlasError::Coordinate("L1 neighborhood overflow".to_owned()))?;
            i32::try_from(scaled)
                .map_err(|_| AtlasError::Coordinate("L1 neighborhood exceeds i32".to_owned()))
        };
        let x_min = l1_bound(center.x, false)?;
        let x_max = l1_bound(center.x, true)?;
        let y_min = l1_bound(center.y, false)?;
        let y_max = l1_bound(center.y, true)?;
        let z_min = l1_bound(center.z, false)?;
        let z_max = l1_bound(center.z, true)?;
        let mut ordinals = Vec::new();
        for z in z_min..=z_max {
            for y in y_min..=y_max {
                let lower = GridIndex::new(x_min, y, z);
                let upper = GridIndex::new(x_max, y, z);
                let start = self.nodes.partition_point(|node| node.index < lower);
                let end = self.nodes.partition_point(|node| node.index <= upper);
                if ordinals.len().saturating_add(end.saturating_sub(start)) > limit {
                    return Err(AtlasError::LimitExceeded(format!(
                        "local recovery neighborhood exceeds {limit} L1 nodes"
                    )));
                }
                ordinals.extend(start..end);
            }
        }
        Ok(ordinals)
    }

    /// Install one complete, independently computed scalar recovery field.
    /// Partial updates are forbidden because L2/L3 aggregation and runtime
    /// descent validation must observe one coherent field.
    pub fn set_costs_to_safety(&mut self, costs: &[u32]) -> AtlasResult<()> {
        if costs.len() != self.nodes.len() {
            return Err(AtlasError::InvalidFormat(format!(
                "cost-to-safety field has {} entries for {} L1 nodes",
                costs.len(),
                self.nodes.len()
            )));
        }
        for (node, cost) in self.nodes.iter_mut().zip(costs) {
            node.cost_to_safety = *cost;
        }
        Ok(())
    }

    /// Install one complete deterministic signed hazard-clearance field.
    pub fn set_hazard_clearances(&mut self, clearances: &[i32]) -> AtlasResult<()> {
        if clearances.len() != self.nodes.len() {
            return Err(AtlasError::InvalidFormat(format!(
                "hazard-clearance field has {} entries for {} L1 nodes",
                clearances.len(),
                self.nodes.len()
            )));
        }
        for (node, clearance) in self.nodes.iter_mut().zip(clearances) {
            node.hazard_clearance = *clearance;
        }
        Ok(())
    }

    pub fn validate(&self, admission: &EdgeAdmission, limits: &AtlasLimits) -> AtlasResult<()> {
        self.validate_structure(limits)?;
        for edge in &self.edges {
            admission.validate_edge(edge.edge_type, edge.evidence, edge.validation_version)?;
        }
        Ok(())
    }

    pub(crate) fn validate_structure(&self, limits: &AtlasLimits) -> AtlasResult<()> {
        if self.nodes.len() > limits.max_l1_nodes || self.edges.len() > limits.max_l1_edges {
            return Err(AtlasError::LimitExceeded(
                "L1 graph exceeds configured cardinality".to_owned(),
            ));
        }
        if self.offsets.len() != self.nodes.len() + 1
            || self.offsets.first() != Some(&0)
            || self.offsets.last().copied().map(|value| value as usize) != Some(self.edges.len())
            || self.offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(AtlasError::InvalidFormat(
                "invalid L1 CSR offsets".to_owned(),
            ));
        }
        if self
            .nodes
            .windows(2)
            .any(|pair| pair[0].index >= pair[1].index)
        {
            return Err(AtlasError::InvalidFormat(
                "L1 nodes are not strictly ordered (iz,iy,ix)".to_owned(),
            ));
        }
        for node in &self.nodes {
            node.validate()?;
        }
        for (node_index, range) in self.offsets.windows(2).enumerate() {
            let start = range[0] as usize;
            let end = range[1] as usize;
            let outgoing = &self.edges[start..end];
            if outgoing
                .iter()
                .any(|edge| edge.target as usize >= self.nodes.len())
            {
                return Err(AtlasError::InvalidFormat(
                    "L1 edge target exceeds node count".to_owned(),
                ));
            }
            if outgoing.iter().any(|edge| edge.cost == COST_INFINITY) {
                return Err(AtlasError::InvalidFormat(
                    "L1 edge has infinite traversal cost".to_owned(),
                ));
            }
            if outgoing
                .iter()
                .any(|edge| edge.evidence == 0 || edge.validation_version == 0)
            {
                return Err(AtlasError::InvalidFormat(
                    "L1 edge lacks evidence or validation version".to_owned(),
                ));
            }
            if outgoing
                .windows(2)
                .any(|pair| pair[0].sort_key(&self.nodes) >= pair[1].sort_key(&self.nodes))
            {
                return Err(AtlasError::InvalidFormat(format!(
                    "L1 adjacency {node_index} is not canonical"
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn from_canonical_parts(
        nodes: Vec<L1Node>,
        offsets: Vec<u32>,
        edges: Vec<EdgeRecord>,
        limits: &AtlasLimits,
    ) -> AtlasResult<Self> {
        let graph = Self {
            nodes,
            offsets,
            edges,
        };
        graph.validate_structure(limits)?;
        Ok(graph)
    }
}
