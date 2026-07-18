use super::{AtlasError, AtlasResult, COST_INFINITY};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConservativeChild {
    pub contents_flags: u32,
    pub hazard_types: u16,
    pub hazard_severity: u8,
    pub clearance: u16,
    pub cost_to_safety: u32,
    pub confidence: u16,
    pub standing_passable: bool,
    pub crouched_passable: bool,
    pub standing_reachable: bool,
    pub crouched_reachable: bool,
}

/// A caller-supplied proof that selected child cells form a stance-legal path
/// from a finer clear origin to a relevant parent boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CorridorWitness {
    pub child_indices: Vec<u8>,
    pub reaches_boundary: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StaticAggregate {
    pub contents_flags: u32,
    pub hazard_types: u16,
    pub hazard_severity: u8,
    pub clearance: u16,
    pub cost_to_safety: u32,
    pub confidence: u16,
    pub standing_passable: bool,
    pub crouched_passable: bool,
}

/// Aggregate one complete 4x4x4 child block. Sparse/missing children must be
/// supplied by the analyzer as blocked/unknown children; omitting them would
/// allow hazards or blockage to disappear through sparsity.
pub fn aggregate_conservative(
    children: &[ConservativeChild],
    standing_corridor: Option<&CorridorWitness>,
    crouched_corridor: Option<&CorridorWitness>,
) -> AtlasResult<StaticAggregate> {
    if children.len() != 64 {
        return Err(AtlasError::InvalidFormat(format!(
            "conservative parent requires 64 children, got {}",
            children.len()
        )));
    }
    let contents_flags = children
        .iter()
        .fold(0_u32, |aggregate, child| aggregate | child.contents_flags);
    let hazard_types = children
        .iter()
        .fold(0_u16, |aggregate, child| aggregate | child.hazard_types);
    let hazard_severity = children
        .iter()
        .map(|child| child.hazard_severity)
        .max()
        .unwrap_or(0);
    let clearance = children
        .iter()
        .map(|child| child.clearance)
        .min()
        .unwrap_or(0);
    let confidence = children
        .iter()
        .map(|child| child.confidence)
        .min()
        .unwrap_or(0);
    let cost_to_safety = children
        .iter()
        .filter(|child| child.standing_reachable || child.crouched_reachable)
        .map(|child| child.cost_to_safety)
        .min()
        .unwrap_or(COST_INFINITY);

    Ok(StaticAggregate {
        contents_flags,
        hazard_types,
        hazard_severity,
        clearance,
        cost_to_safety,
        confidence,
        standing_passable: corridor_is_valid(children, standing_corridor, true)?,
        crouched_passable: corridor_is_valid(children, crouched_corridor, false)?,
    })
}

fn corridor_is_valid(
    children: &[ConservativeChild],
    witness: Option<&CorridorWitness>,
    standing: bool,
) -> AtlasResult<bool> {
    let Some(witness) = witness else {
        return Ok(false);
    };
    if witness.child_indices.is_empty() || !witness.reaches_boundary {
        return Ok(false);
    }
    let mut previous = None;
    for &index in &witness.child_indices {
        if index >= 64 {
            return Err(AtlasError::InvalidFormat(format!(
                "corridor child index {index} is outside 4x4x4 block"
            )));
        }
        if previous.is_some_and(|value| index <= value) {
            return Err(AtlasError::InvalidFormat(
                "corridor child indices must be strictly increasing".to_owned(),
            ));
        }
        previous = Some(index);
        let child = children[index as usize];
        let legal = if standing {
            child.standing_passable && child.standing_reachable
        } else {
            child.crouched_passable && child.crouched_reachable
        };
        if !legal {
            return Ok(false);
        }
    }
    Ok(true)
}
