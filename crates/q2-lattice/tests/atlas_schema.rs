use std::collections::BTreeMap;
use std::time::Instant;

use q2_lattice_rs::atlas::{
    ATLAS_CELL_SIZES, ATLAS_MAGIC, ATLAS_SCHEMA_VERSION, ArtifactManifest, AtlasAggregateCell,
    AtlasArtifact, AtlasCounts, AtlasLevel, AtlasLimits, AtlasManifest, AtlasOrigin,
    B1AuthorityExecutables, B1AuthorityIdentities, B1AuthorityIdentity, B1NormativeDocuments,
    B1RuntimeAuthoritySeal, BspIdentity, COLLISION_ORACLE_NAME, COLLISION_ORACLE_SCHEMA,
    COST_INFINITY, ChannelManifest, CollisionOracleAdmission, CollisionParameters,
    CollisionSourceClosure, ConservativeChild, CorridorWitness, EdgeInput, EdgeType, GridIndex,
    GridManifest, HOOK_ORACLE_NAME, HOOK_ORACLE_SCHEMA, HOOK_PARITY_CASES_V1, HOOK_PARITY_NAME,
    HOOK_PARITY_SCHEMA, HookOracleAdmission, HookParameters, HookParityAttestation,
    HookSourceClosure, HullManifest, L0Address, L0BitPlane, L0Chunk, L0ScalarPlane, L1Graph,
    L1Node, MASK_PLAYERSOLID_V1, MASK_SHOT_V1, ManifestBudgets, NodeFlags, ORACLE_SEMANTIC_VERSION,
    OracleAdmissions, OracleBspBinding, OracleToolIdentity, PMOVE_ORACLE_NAME, PMOVE_ORACLE_SCHEMA,
    PmoveOracleAdmission, PmoveParameters, PmoveSourceClosure, SparseL0, Stance, ToolIdentity,
    aggregate_conservative, decode_zstd_envelope, encode_zstd_envelope, sha256_hex,
};

fn node(index: GridIndex) -> L1Node {
    L1Node {
        index,
        flags: NodeFlags::STANDING_CLEAR
            | NodeFlags::CROUCHED_CLEAR
            | NodeFlags::SAFE_TO_STAND
            | NodeFlags::SUPPORTED_FLOOR
            | NodeFlags::STANDING_PASSABLE
            | NodeFlags::CROUCHED_PASSABLE,
        floor_normal_class: 1,
        clearance_height: 96,
        hazard_types: 0,
        hazard_severity: 0,
        hazard_clearance: 8 * 256,
        cost_to_safety: 256,
        region_id: 7,
        confidence: u16::MAX,
        evidence: 1,
        contents_flags: 0,
    }
}

fn edge(source: GridIndex, target: GridIndex, edge_type: EdgeType) -> EdgeInput {
    EdgeInput {
        source,
        target,
        edge_type,
        stance: Stance::Standing,
        flags: 0,
        blocker: 0,
        cost: 256,
        risk: 0,
        confidence: u16::MAX,
        evidence: 1,
        validation_version: 1,
        auxiliary: u32::MAX,
    }
}

fn digest(byte: u8) -> String {
    format!("{byte:02x}").repeat(32)
}

fn bsp_identity() -> BspIdentity {
    BspIdentity {
        canonical_map_id: "fixture".to_owned(),
        sha256: digest(0xcd),
        provenance_sha256: digest(0xce),
        size_bytes: 1024,
        ibsp_version: 38,
    }
}

fn oracle_tool(name: &str, schema: &str, executable: u8, physics: u8) -> OracleToolIdentity {
    OracleToolIdentity {
        name: name.to_owned(),
        schema: schema.to_owned(),
        version: ORACLE_SEMANTIC_VERSION,
        executable_sha256: digest(executable),
        physics_identity_sha256: digest(physics),
    }
}

fn oracle_binding(bsp: &BspIdentity) -> OracleBspBinding {
    OracleBspBinding {
        sha256: bsp.sha256.clone(),
        provenance_sha256: bsp.provenance_sha256.clone(),
    }
}

fn oracle_admissions(bsp: &BspIdentity, pmove: bool, hook: bool) -> OracleAdmissions {
    let collision = CollisionOracleAdmission {
        tool: oracle_tool(COLLISION_ORACLE_NAME, COLLISION_ORACLE_SCHEMA, 0x11, 0x12),
        bsp: oracle_binding(bsp),
        parameters: CollisionParameters {
            mask_playersolid: MASK_PLAYERSOLID_V1,
            mask_shot: MASK_SHOT_V1,
        },
        source: CollisionSourceClosure {
            collision_sha256: digest(0x13),
            shared_header_sha256: digest(0x14),
            shared_source_sha256: digest(0x15),
            build_contract: "cc-v1 -fno-strict-aliasing".to_owned(),
        },
        contract_sha256: String::new(),
    }
    .seal();
    let pmove_admission = pmove.then(|| {
        PmoveOracleAdmission {
            tool: oracle_tool(PMOVE_ORACLE_NAME, PMOVE_ORACLE_SCHEMA, 0x21, 0x22),
            bsp: oracle_binding(bsp),
            parameters: PmoveParameters {
                gravity: 800,
                airaccelerate_f32_bits: 0.0_f32.to_bits(),
                constants: "stopspeed=100;maxspeed=300;duckspeed=100;accelerate=10;wateraccelerate=10;friction=6;waterfriction=1;waterspeed=400".to_owned(),
            },
            source: PmoveSourceClosure {
                collision_sha256: collision.source.collision_sha256.clone(),
                pmove_sha256: digest(0x23),
                shared_header_sha256: collision.source.shared_header_sha256.clone(),
                shared_source_sha256: collision.source.shared_source_sha256.clone(),
                build_contract: collision.source.build_contract.clone(),
            },
            contract_sha256: String::new(),
        }
        .seal()
    });
    let hook_admission = hook.then(|| {
        let pmove = pmove_admission
            .as_ref()
            .expect("hook fixture always includes companion pmove");
        let tool = oracle_tool(HOOK_ORACLE_NAME, HOOK_ORACLE_SCHEMA, 0x31, 0x32);
        let parity = HookParityAttestation {
            name: HOOK_PARITY_NAME.to_owned(),
            schema: HOOK_PARITY_SCHEMA.to_owned(),
            version: ORACLE_SEMANTIC_VERSION,
            passed: true,
            case_count: HOOK_PARITY_CASES_V1,
            fixture_bsp_sha256: digest(0x33),
            fixture_provenance_sha256: digest(0x34),
            fixture_collision_physics_identity_sha256: digest(0x35),
            fixture_pmove_physics_identity_sha256: digest(0x36),
            hook_physics_identity_sha256: tool.physics_identity_sha256.clone(),
            collision_tool_sha256: collision.tool.executable_sha256.clone(),
            pmove_tool_sha256: pmove.tool.executable_sha256.clone(),
            hook_tool_sha256: tool.executable_sha256.clone(),
            q2ded_sha256: digest(0x37),
            game_module_sha256: digest(0x38),
            transcript_sha256: digest(0x39),
            attestation_sha256: String::new(),
        }
        .seal();
        HookOracleAdmission {
            tool,
            bsp: oracle_binding(bsp),
            parameters: HookParameters {
                hook_speed_f32_bits: 900.0_f32.to_bits(),
                hook_pullspeed_f32_bits: 700.0_f32.to_bits(),
                hook_sky: false,
                hook_maxtime_f32_bits: 5.0_f32.to_bits(),
                full_velocity_overwrite: true,
            },
            source: HookSourceClosure {
                shared_c_sha256: digest(0x3a),
                shared_h_sha256: digest(0x3b),
                integration_sha256: digest(0x3c),
                math_sha256: digest(0x3d),
                build_contract: "cc-v1 -fno-strict-aliasing".to_owned(),
            },
            parity,
            contract_sha256: String::new(),
        }
        .seal()
    });
    let pmove_tool = pmove_admission.as_ref().map(|item| &item.tool);
    let hook_tool = hook_admission.as_ref().map(|item| &item.tool);
    OracleAdmissions {
        b1_runtime_authority_seal: B1RuntimeAuthoritySeal {
            schema: "q2-b1-runtime-authority-seal-v1".to_owned(),
            normative_documents: B1NormativeDocuments {
                design_sha256: digest(0x41),
                plan_sha256: digest(0x42),
            },
            hook_parity_attestation_sha256: digest(0x43),
            fixture_bsp_sha256: digest(0x44),
            analysis_bsp_sha256: bsp.sha256.clone(),
            executables: B1AuthorityExecutables {
                cm_sha256: collision.tool.executable_sha256.clone(),
                pmove_sha256: pmove_tool
                    .map_or_else(|| digest(0x45), |item| item.executable_sha256.clone()),
                hook_sha256: hook_tool
                    .map_or_else(|| digest(0x46), |item| item.executable_sha256.clone()),
                fall_sha256: digest(0x47),
            },
            identities: B1AuthorityIdentities {
                collision: B1AuthorityIdentity {
                    tool_identity: digest(0x48),
                    physics_identity: collision.tool.physics_identity_sha256.clone(),
                },
                pmove: B1AuthorityIdentity {
                    tool_identity: digest(0x49),
                    physics_identity: pmove_tool
                        .map_or_else(|| digest(0x4a), |item| item.physics_identity_sha256.clone()),
                },
                hook: B1AuthorityIdentity {
                    tool_identity: digest(0x4b),
                    physics_identity: hook_tool
                        .map_or_else(|| digest(0x4c), |item| item.physics_identity_sha256.clone()),
                },
                fall: B1AuthorityIdentity {
                    tool_identity: digest(0x4d),
                    physics_identity: digest(0x4e),
                },
            },
        },
        collision_oracle: collision,
        pmove_oracle: pmove_admission,
        hook_oracle: hook_admission,
    }
}

fn artifact(reverse_insert: bool) -> AtlasArtifact {
    let limits = AtlasLimits::default();
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
    let mut chunks = SparseL0::new();
    let mut source = vec![GridIndex::new(2, -1, 0), GridIndex::new(-3, 4, -2)];
    if reverse_insert {
        source.reverse();
    }
    for key in source {
        let cell = 17 + key.x.unsigned_abs() as usize;
        let mut chunk = L0Chunk::new(key);
        chunk.set_bit(L0BitPlane::Solid, cell, true).unwrap();
        chunk
            .set_bit(L0BitPlane::StandingForbiddenOrigin, 1024, true)
            .unwrap();
        chunk
            .set_scalar(L0ScalarPlane::Confidence, 17, 240)
            .unwrap();
        chunks.insert(chunk, &limits).unwrap();
    }
    let a = GridIndex::new(-1, 0, -2);
    let b = GridIndex::new(0, 0, -2);
    let c = GridIndex::new(0, 0, 1);
    let nodes = if reverse_insert {
        vec![node(c), node(a), node(b)]
    } else {
        vec![node(b), node(c), node(a)]
    };
    let edges = if reverse_insert {
        vec![edge(a, c, EdgeType::Jump), edge(a, b, EdgeType::Walk)]
    } else {
        vec![edge(a, b, EdgeType::Walk), edge(a, c, EdgeType::Jump)]
    };
    AtlasArtifact {
        origin: AtlasOrigin::snapped([-1, 0, 511]).unwrap(),
        l0: chunks,
        l1: L1Graph::build(nodes, edges, &admission, &limits).unwrap(),
        l2: vec![AtlasAggregateCell {
            index: GridIndex::new(-1, 0, 0),
            contents_flags: 3,
            hazard_types: 2,
            hazard_severity: 9,
            standing_passable: true,
            crouched_passable: true,
            clearance: 72,
            cost_to_safety: 512,
            confidence: 60_000,
        }],
        l3: vec![],
    }
}

#[test]
fn snapped_origin_and_signed_floor_are_mathematical_at_every_level() {
    assert_eq!(
        AtlasOrigin::snapped([-1, -256, -257]).unwrap().0,
        [-256, -256, -512]
    );
    let origin = AtlasOrigin([0, 0, 0]);
    for level in AtlasLevel::ALL {
        let size = level.cell_size();
        for world in -1025_i64..=1025 {
            let actual = origin.index_integer([world, world, world], level).unwrap();
            let expected = world.div_euclid(size) as i32;
            assert_eq!(actual, GridIndex::new(expected, expected, expected));
            let center = origin.center(actual, level);
            assert!(center[0] >= (actual.x as i64 * size) as f64);
            assert!(center[0] < ((actual.x as i64 + 1) * size) as f64);
        }
    }
    assert_eq!(
        origin
            .index([-0.001, -4.001, -16.001], AtlasLevel::L0)
            .unwrap(),
        GridIndex::new(-1, -2, -5)
    );
    for world in -1025_i64..=1025 {
        let indices = AtlasLevel::ALL.map(|level| {
            origin
                .index_integer([world, -world, world / 3], level)
                .unwrap()
        });
        assert_eq!(indices[0].parent(), indices[1]);
        assert_eq!(indices[1].parent(), indices[2]);
        assert_eq!(indices[2].parent(), indices[3]);
    }
    assert!(origin.index([f64::NAN, 0.0, 0.0], AtlasLevel::L0).is_err());
    assert!(
        origin
            .index_integer([i64::MAX, 0, 0], AtlasLevel::L0)
            .is_err()
    );
}

#[test]
fn parent_child_properties_hold_for_negative_nonmultiples() {
    for z in -33..=33 {
        for y in -17..=17 {
            for x in -19..=19 {
                let child = GridIndex::new(x, y, z);
                let parent = child.parent();
                assert_eq!(parent.x, x.div_euclid(4));
                assert_eq!(parent.y, y.div_euclid(4));
                assert_eq!(parent.z, z.div_euclid(4));
                let minimum = parent.child_min().unwrap();
                assert!((minimum.x..minimum.x + 4).contains(&x));
                assert!((minimum.y..minimum.y + 4).contains(&y));
                assert!((minimum.z..minimum.z + 4).contains(&z));
            }
        }
    }
    for parent in [GridIndex::new(-2, -1, -3), GridIndex::new(0, 0, 0)] {
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    assert_eq!(parent.child(x, y, z).unwrap().parent(), parent);
                }
            }
        }
    }
}

#[test]
fn l0_chunk_address_uses_euclidean_remainder() {
    let address = L0Address::from_l0_index(GridIndex::new(-1, -16, -17));
    assert_eq!(address.chunk, GridIndex::new(-1, -1, -2));
    assert_eq!(address.local, [15, 0, 15]);
    assert_eq!(address.linear, 15 + 15 * 256);
    let origin = AtlasOrigin([-256, 0, 256]);
    for world in -513_i64..=513 {
        let l0 = origin
            .index_integer([world, -world, world / 2], AtlasLevel::L0)
            .unwrap();
        let l2 = origin
            .index_integer([world, -world, world / 2], AtlasLevel::L2)
            .unwrap();
        assert_eq!(L0Address::from_l0_index(l0).chunk, l2);
    }
}

#[test]
fn l0_is_sparse_and_enforces_chunk_and_payload_guards() {
    let mut l0 = SparseL0::new();
    assert!(l0.is_empty());
    assert!(
        l0.insert(L0Chunk::new(GridIndex::default()), &AtlasLimits::default())
            .is_err()
    );

    let mut chunk = L0Chunk::new(GridIndex::new(-1, 0, 0));
    chunk.set_bit(L0BitPlane::Lava, 4095, true).unwrap();
    chunk
        .set_scalar(L0ScalarPlane::HazardSeverity, 4095, 255)
        .unwrap();
    assert!(chunk.bit(L0BitPlane::Lava, 4095).unwrap());
    assert_eq!(
        chunk.scalar(L0ScalarPlane::HazardSeverity, 4095).unwrap(),
        255
    );

    let too_small = AtlasLimits {
        max_l0_decompressed_bytes: 100,
        ..AtlasLimits::default()
    };
    assert!(l0.insert(chunk.clone(), &too_small).is_err());
    l0.insert(chunk, &AtlasLimits::default()).unwrap();
    assert_eq!(l0.len(), 1);

    let one_chunk = AtlasLimits {
        max_l0_chunks: 1,
        ..AtlasLimits::default()
    };
    let mut another = L0Chunk::new(GridIndex::new(0, 0, 0));
    another.set_bit(L0BitPlane::Solid, 0, true).unwrap();
    assert!(l0.insert(another, &one_chunk).is_err());
}

#[test]
fn l1_graph_is_deterministic_zyx_csr() {
    let limits = AtlasLimits::default();
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, false).admit(&bsp).unwrap();
    let a = GridIndex::new(10, 0, -1);
    let b = GridIndex::new(-5, 2, -1);
    let c = GridIndex::new(0, 0, 0);
    let graph = L1Graph::build(
        vec![node(c), node(a), node(b)],
        vec![edge(a, c, EdgeType::Jump), edge(a, b, EdgeType::Walk)],
        &admission,
        &limits,
    )
    .unwrap();
    assert_eq!(
        graph
            .nodes()
            .iter()
            .map(|entry| entry.index)
            .collect::<Vec<_>>(),
        vec![a, b, c]
    );
    assert_eq!(graph.offsets(), &[0, 2, 2, 2]);
    let outgoing = graph.outgoing(0).unwrap();
    assert_eq!(outgoing[0].edge_type, EdgeType::Walk);
    assert_eq!(outgoing[1].edge_type, EdgeType::Jump);

    let duplicate = edge(a, b, EdgeType::Walk);
    assert!(
        L1Graph::build(
            vec![node(a), node(b)],
            vec![duplicate, duplicate],
            &admission,
            &limits,
        )
        .is_err()
    );
}

fn two_node_graph(
    edge_type: EdgeType,
    evidence: u16,
    validation_version: u16,
    admission: &q2_lattice_rs::atlas::EdgeAdmission,
) -> Result<L1Graph, q2_lattice_rs::atlas::AtlasError> {
    let source = GridIndex::new(0, 0, 0);
    let target = GridIndex::new(1, 0, 0);
    let mut candidate = edge(source, target, edge_type);
    candidate.evidence = evidence;
    candidate.validation_version = validation_version;
    L1Graph::build(
        vec![node(source), node(target)],
        vec![candidate],
        admission,
        &AtlasLimits::default(),
    )
}

#[test]
fn graph_trajectory_edges_require_admitted_optional_oracles() {
    let bsp = bsp_identity();
    assert!(oracle_admissions(&bsp, false, false).admit(&bsp).is_err());

    let pmove_only = oracle_admissions(&bsp, true, false).admit(&bsp).unwrap();
    assert!(two_node_graph(EdgeType::Jump, 1, 1, &pmove_only).is_ok());
    assert!(two_node_graph(EdgeType::ControlledDrop, 1, 1, &pmove_only).is_ok());
    assert!(two_node_graph(EdgeType::Hook, 1, 1, &pmove_only).is_err());

    let full = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
    assert!(two_node_graph(EdgeType::Hook, 1, 1, &full).is_ok());
}

#[test]
fn every_materialized_edge_requires_nonzero_evidence_and_validation_version() {
    let bsp = bsp_identity();
    let admission = oracle_admissions(&bsp, true, true).admit(&bsp).unwrap();
    for edge_type in [
        EdgeType::Walk,
        EdgeType::StrafeWalk,
        EdgeType::Step,
        EdgeType::Jump,
        EdgeType::ControlledDrop,
        EdgeType::CrouchEnter,
        EdgeType::CrouchHold,
        EdgeType::CrouchExit,
        EdgeType::WaterTransition,
        EdgeType::Mover,
        EdgeType::Teleporter,
        EdgeType::Hook,
    ] {
        assert!(two_node_graph(edge_type, 0, 1, &admission).is_err());
        assert!(two_node_graph(edge_type, 1, 0, &admission).is_err());
        assert!(two_node_graph(edge_type, 1, 1, &admission).is_ok());
    }
}

#[test]
fn oracle_contract_rejects_tool_schema_version_bsp_parameter_and_source_mismatches() {
    let bsp = bsp_identity();
    let valid = oracle_admissions(&bsp, true, true);
    valid.admit(&bsp).unwrap();

    for mutation in 0..3 {
        let mut candidate = valid.clone();
        let tool = &mut candidate.collision_oracle.tool;
        match mutation {
            0 => tool.name.push_str("-wrong"),
            1 => tool.schema.push_str("-wrong"),
            _ => tool.version += 1,
        }
        assert!(candidate.admit(&bsp).is_err());
    }
    for mutation in 0..3 {
        let mut candidate = valid.clone();
        let tool = &mut candidate.pmove_oracle.as_mut().unwrap().tool;
        match mutation {
            0 => tool.name.push_str("-wrong"),
            1 => tool.schema.push_str("-wrong"),
            _ => tool.version += 1,
        }
        assert!(candidate.admit(&bsp).is_err());
    }
    for mutation in 0..3 {
        let mut candidate = valid.clone();
        let tool = &mut candidate.hook_oracle.as_mut().unwrap().tool;
        match mutation {
            0 => tool.name.push_str("-wrong"),
            1 => tool.schema.push_str("-wrong"),
            _ => tool.version += 1,
        }
        assert!(candidate.admit(&bsp).is_err());
    }

    let mut wrong_bsp = valid.clone();
    wrong_bsp.collision_oracle.bsp.sha256 = digest(0xee);
    assert!(wrong_bsp.admit(&bsp).is_err());
    let mut wrong_provenance = valid.clone();
    wrong_provenance
        .pmove_oracle
        .as_mut()
        .unwrap()
        .bsp
        .provenance_sha256 = digest(0xef);
    assert!(wrong_provenance.admit(&bsp).is_err());
    let mut wrong_hook_bsp = valid.clone();
    wrong_hook_bsp.hook_oracle.as_mut().unwrap().bsp.sha256 = digest(0xf0);
    assert!(wrong_hook_bsp.admit(&bsp).is_err());

    let mut wrong_mask = valid.clone();
    wrong_mask.collision_oracle.parameters.mask_playersolid ^= 1;
    wrong_mask.collision_oracle = wrong_mask.collision_oracle.seal();
    assert!(wrong_mask.admit(&bsp).is_err());
    let mut wrong_parameter_closure = valid.clone();
    wrong_parameter_closure
        .pmove_oracle
        .as_mut()
        .unwrap()
        .parameters
        .gravity = 799;
    assert!(wrong_parameter_closure.admit(&bsp).is_err());
    let mut illegal_airaccelerate = valid.clone();
    let pmove = illegal_airaccelerate.pmove_oracle.take().unwrap();
    let mut pmove = pmove;
    pmove.parameters.airaccelerate_f32_bits = (-1.0_f32).to_bits();
    illegal_airaccelerate.pmove_oracle = Some(pmove.seal());
    assert!(illegal_airaccelerate.admit(&bsp).is_err());
    let mut wrong_source_closure = valid.clone();
    wrong_source_closure
        .hook_oracle
        .as_mut()
        .unwrap()
        .source
        .math_sha256 = digest(0xaa);
    assert!(wrong_source_closure.admit(&bsp).is_err());
    let mut zero_pullspeed = valid.clone();
    let hook = zero_pullspeed.hook_oracle.take().unwrap();
    let mut hook = hook;
    hook.parameters.hook_pullspeed_f32_bits = 0.0_f32.to_bits();
    zero_pullspeed.hook_oracle = Some(hook.seal());
    assert!(zero_pullspeed.admit(&bsp).is_err());
    let mut wrong_tool_closure = valid.clone();
    wrong_tool_closure.collision_oracle.tool.executable_sha256 = digest(0xab);
    assert!(wrong_tool_closure.admit(&bsp).is_err());

    for field in 0..3 {
        let mut zero_contract = valid.clone();
        match field {
            0 => zero_contract.collision_oracle.contract_sha256 = "00".repeat(32),
            1 => zero_contract.pmove_oracle.as_mut().unwrap().contract_sha256 = "00".repeat(32),
            _ => zero_contract.hook_oracle.as_mut().unwrap().contract_sha256 = "00".repeat(32),
        }
        assert!(zero_contract.admit(&bsp).is_err());
    }

    let mut wrong_companion_source = valid.clone();
    let pmove = wrong_companion_source.pmove_oracle.take().unwrap();
    let mut pmove = pmove;
    pmove.source.collision_sha256 = digest(0xac);
    wrong_companion_source.pmove_oracle = Some(pmove.seal());
    assert!(wrong_companion_source.admit(&bsp).is_err());

    let mut wrong_overwrite_law = valid;
    let hook = wrong_overwrite_law.hook_oracle.take().unwrap();
    let mut hook = hook;
    hook.parameters.full_velocity_overwrite = false;
    wrong_overwrite_law.hook_oracle = Some(hook.seal());
    assert!(wrong_overwrite_law.admit(&bsp).is_err());
}

#[test]
fn hook_requires_companion_pmove_and_canonical_q2ded_parity() {
    let bsp = bsp_identity();
    let valid = oracle_admissions(&bsp, true, true);

    let mut missing_parity = serde_json::to_value(&valid).unwrap();
    missing_parity["hook_oracle"]
        .as_object_mut()
        .unwrap()
        .remove("parity");
    assert!(serde_json::from_value::<OracleAdmissions>(missing_parity).is_err());

    let mut missing_pmove = valid.clone();
    missing_pmove.pmove_oracle = None;
    assert!(missing_pmove.admit(&bsp).is_err());

    for mutation in 0..5 {
        let mut candidate = valid.clone();
        let hook = candidate.hook_oracle.as_mut().unwrap();
        match mutation {
            0 => hook.parity.name.push_str("-wrong"),
            1 => hook.parity.schema.push_str("-wrong"),
            2 => hook.parity.version += 1,
            3 => hook.parity.passed = false,
            _ => hook.parity.case_count -= 1,
        }
        hook.parity = hook.parity.clone().seal();
        *hook = hook.clone().seal();
        assert!(candidate.admit(&bsp).is_err());
    }

    let mut stale_attestation = valid.clone();
    stale_attestation
        .hook_oracle
        .as_mut()
        .unwrap()
        .parity
        .transcript_sha256 = digest(0xba);
    assert!(stale_attestation.admit(&bsp).is_err());

    let mut wrong_tool = valid.clone();
    let hook = wrong_tool.hook_oracle.as_mut().unwrap();
    hook.parity.hook_tool_sha256 = digest(0xbb);
    hook.parity = hook.parity.clone().seal();
    *hook = hook.clone().seal();
    assert!(wrong_tool.admit(&bsp).is_err());

    let mut zero_digest = valid;
    let hook = zero_digest.hook_oracle.as_mut().unwrap();
    hook.parity.q2ded_sha256 = "00".repeat(32);
    hook.parity = hook.parity.clone().seal();
    *hook = hook.clone().seal();
    assert!(zero_digest.admit(&bsp).is_err());
}

#[test]
fn conservative_aggregation_never_hides_hazard_or_blockage() {
    let base = ConservativeChild {
        contents_flags: 1,
        hazard_types: 0,
        hazard_severity: 0,
        clearance: 128,
        cost_to_safety: 1024,
        confidence: u16::MAX,
        standing_passable: true,
        crouched_passable: true,
        standing_reachable: true,
        crouched_reachable: true,
    };
    for hazard_index in 0..64 {
        let mut children = vec![base; 64];
        children[hazard_index].contents_flags = 0x80;
        children[hazard_index].hazard_types = 0x20;
        children[hazard_index].hazard_severity = 250;
        children[hazard_index].clearance = 3;
        children[hazard_index].confidence = 17;
        let all = CorridorWitness {
            child_indices: (0..64).collect(),
            reaches_boundary: true,
        };
        let aggregate = aggregate_conservative(&children, Some(&all), Some(&all)).unwrap();
        assert_eq!(aggregate.contents_flags & 0x80, 0x80);
        assert_eq!(aggregate.hazard_types & 0x20, 0x20);
        assert_eq!(aggregate.hazard_severity, 250);
        assert_eq!(aggregate.clearance, 3);
        assert_eq!(aggregate.confidence, 17);
        assert!(aggregate.standing_passable);

        children[hazard_index].standing_passable = false;
        let aggregate = aggregate_conservative(&children, Some(&all), Some(&all)).unwrap();
        assert!(!aggregate.standing_passable);
        assert!(aggregate.crouched_passable);
    }
    let unreachable = vec![
        ConservativeChild {
            cost_to_safety: 2,
            standing_reachable: false,
            crouched_reachable: false,
            ..base
        };
        64
    ];
    assert_eq!(
        aggregate_conservative(&unreachable, None, None)
            .unwrap()
            .cost_to_safety,
        COST_INFINITY
    );
    assert!(aggregate_conservative(&unreachable[..63], None, None).is_err());
}

#[test]
fn canonical_serialization_is_order_independent_and_round_trips() {
    let limits = AtlasLimits::default();
    let first = artifact(false).encode_uncompressed(&limits).unwrap();
    let second = artifact(true).encode_uncompressed(&limits).unwrap();
    assert_eq!(first, second);
    assert_eq!(&first[..8], ATLAS_MAGIC);
    assert_eq!(
        sha256_hex(&first),
        "2577be4ec3f813ce001bd2bd82de39722d1631f787301a993de89b99bc3a9774"
    );
    let restored = AtlasArtifact::decode_uncompressed(&first, &limits).unwrap();
    assert_eq!(restored.encode_uncompressed(&limits).unwrap(), first);

    let envelope = encode_zstd_envelope(&first, &limits).unwrap();
    assert_eq!(decode_zstd_envelope(&envelope, &limits).unwrap(), first);
    assert_eq!(
        AtlasArtifact::decode_zstd(&envelope, &limits).unwrap(),
        restored
    );
    assert_eq!(restored.encode_zstd(&limits).unwrap(), envelope);
}

#[test]
fn empty_atlas_has_a_canonical_round_trip() {
    let limits = AtlasLimits::default();
    let artifact = AtlasArtifact::empty(AtlasOrigin([0, 0, 0]));
    let bytes = artifact.encode_uncompressed(&limits).unwrap();
    assert_eq!(bytes.len(), 148); // 136-byte header plus one empty CSR offset.
    let restored = AtlasArtifact::decode_uncompressed(&bytes, &limits).unwrap();
    assert_eq!(restored, artifact);
    assert_eq!(restored.encode_uncompressed(&limits).unwrap(), bytes);
}

#[test]
fn corrupt_oversized_and_mixed_schema_payloads_fail_closed() {
    let limits = AtlasLimits::default();
    let valid = artifact(false).encode_uncompressed(&limits).unwrap();

    let mut mixed = valid.clone();
    mixed[8..10].copy_from_slice(&(ATLAS_SCHEMA_VERSION + 1).to_le_bytes());
    assert!(AtlasArtifact::decode_uncompressed(&mixed, &limits).is_err());
    assert!(AtlasArtifact::decode_uncompressed(&valid[..valid.len() - 1], &limits).is_err());

    let mut oversized = valid.clone();
    // Header offset 56 is the first u64 count (L0 chunks).
    oversized[56..64].copy_from_slice(&((limits.max_l0_chunks + 1) as u64).to_le_bytes());
    assert!(AtlasArtifact::decode_uncompressed(&oversized, &limits).is_err());

    let l0_bytes = u64::from_le_bytes(valid[96..104].try_into().unwrap()) as usize;
    let node_bytes = u64::from_le_bytes(valid[104..112].try_into().unwrap()) as usize;
    let graph_start = 136 + l0_bytes + node_bytes;
    let offset_count =
        u64::from_le_bytes(valid[graph_start..graph_start + 8].try_into().unwrap()) as usize;
    let first_edge = graph_start + 8 + offset_count * 4;
    let mut zero_evidence = valid.clone();
    zero_evidence[first_edge + 20..first_edge + 22].copy_from_slice(&0_u16.to_le_bytes());
    assert!(AtlasArtifact::decode_uncompressed(&zero_evidence, &limits).is_err());
    let mut zero_validation_version = valid.clone();
    zero_validation_version[first_edge + 22..first_edge + 24].copy_from_slice(&0_u16.to_le_bytes());
    assert!(AtlasArtifact::decode_uncompressed(&zero_validation_version, &limits).is_err());

    let envelope = encode_zstd_envelope(&valid, &limits).unwrap();
    let mut corrupt_digest = envelope.clone();
    corrupt_digest[32] ^= 0xff;
    assert!(decode_zstd_envelope(&corrupt_digest, &limits).is_err());
    assert!(decode_zstd_envelope(&envelope[..envelope.len() - 1], &limits).is_err());

    let mut tiny = limits.clone();
    tiny.max_atlas_decompressed_bytes = 64;
    assert!(encode_zstd_envelope(&valid, &tiny).is_err());
    assert!(decode_zstd_envelope(&envelope, &tiny).is_err());
    let mut no_resident_capacity = limits.clone();
    no_resident_capacity.max_atlas_resident_bytes = 0;
    assert!(artifact(false).validate(&no_resident_capacity).is_err());
}

fn identity(name: &str, byte: u8) -> ToolIdentity {
    ToolIdentity {
        name: name.to_owned(),
        version: "1".to_owned(),
        sha256: format!("{byte:02x}").repeat(32),
    }
}

fn manifest(payload: &[u8], compressed_size: u64) -> AtlasManifest {
    let limits = AtlasLimits::default();
    let bsp = bsp_identity();
    let counts = AtlasCounts {
        l0_chunks: 2,
        l1_nodes: 3,
        l1_edges: 2,
        l2_cells: 1,
        l3_cells: 0,
    };
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "fixture.atlas.bin.zst".to_owned(),
        ArtifactManifest::from_uncompressed(
            "application/vnd.q2.atlas-v1",
            payload,
            compressed_size,
            counts.named_counts(),
        ),
    );
    AtlasManifest {
        schema_version: ATLAS_SCHEMA_VERSION,
        byte_order: "little".to_owned(),
        atlas_magic: String::from_utf8_lossy(ATLAS_MAGIC).into_owned(),
        specification_sha256: "ab".repeat(32),
        bsp: bsp.clone(),
        analyzer: identity("q2-atlas", 1),
        oracles: oracle_admissions(&bsp, true, true),
        generator: None,
        grid: GridManifest {
            origin: [-256, 0, 256],
            model0_mins: [-1, 0, 300],
            model0_maxs: [300, 600, 900],
            cell_sizes: ATLAS_CELL_SIZES.map(|value| value as u32),
            l0_chunk_dimensions: [16, 16, 16],
        },
        player_hulls: vec![
            HullManifest {
                name: "standing".to_owned(),
                mins: [-16, -16, -24],
                maxs: [16, 16, 32],
            },
            HullManifest {
                name: "crouched".to_owned(),
                mins: [-16, -16, -24],
                maxs: [16, 16, 4],
            },
        ],
        channels: vec![
            ChannelManifest {
                store: "Atlas".to_owned(),
                level: 1,
                name: "clearance".to_owned(),
                encoding: "u16".to_owned(),
                persistence: "map-static".to_owned(),
            },
            ChannelManifest {
                store: "Atlas".to_owned(),
                level: 0,
                name: "solid".to_owned(),
                encoding: "bitplane".to_owned(),
                persistence: "map-static".to_owned(),
            },
        ],
        artifacts,
        counts,
        budgets: ManifestBudgets::from(&limits),
        build_peak_rss_bytes: 64 * 1024 * 1024,
        limitations: vec!["zero hook edges".to_owned(), "fixture only".to_owned()],
        confidence_summary: "oracle-backed fixture".to_owned(),
    }
}

#[test]
fn manifest_is_canonical_bound_and_strict() {
    let limits = AtlasLimits::default();
    let payload = artifact(false).encode_uncompressed(&limits).unwrap();
    let envelope = encode_zstd_envelope(&payload, &limits).unwrap();
    let source = manifest(&payload, envelope.len() as u64);
    let bytes = source.canonical_json(&limits).unwrap();
    assert_eq!(
        sha256_hex(&bytes),
        "be5ca201540e86b3686eb58f45ea64271e4366303a09e2ae1f38ef723281b970"
    );
    let restored = AtlasManifest::from_canonical_json(&bytes, &limits).unwrap();
    assert_eq!(restored.channels[0].level, 0);
    assert_eq!(
        restored.limitations,
        vec!["fixture only", "zero hook edges"]
    );
    assert_eq!(restored.canonical_json(&limits).unwrap(), bytes);
    restored
        .verify_atlas_artifact("fixture.atlas.bin.zst", &payload, &artifact(false), &limits)
        .unwrap();
    assert_eq!(
        restored
            .decode_and_verify_atlas_artifact("fixture.atlas.bin.zst", &payload, &limits)
            .unwrap(),
        artifact(false)
    );

    let mut substitute = artifact(false);
    let a = GridIndex::new(-1, 0, -2);
    let b = GridIndex::new(0, 0, -2);
    let c = GridIndex::new(0, 0, 1);
    let admission = restored.oracles.admit(&restored.bsp).unwrap();
    substitute.l1 = L1Graph::build(
        vec![node(a), node(b), node(c)],
        vec![edge(a, b, EdgeType::StrafeWalk), edge(a, c, EdgeType::Jump)],
        &admission,
        &limits,
    )
    .unwrap();
    assert_eq!(AtlasCounts::from_artifact(&substitute), restored.counts);
    assert!(
        restored
            .verify_atlas_artifact("fixture.atlas.bin.zst", &payload, &substitute, &limits,)
            .is_err()
    );

    let mut lying_artifact_counts = restored.clone();
    lying_artifact_counts
        .artifacts
        .get_mut("fixture.atlas.bin.zst")
        .unwrap()
        .counts
        .insert("l1_edges".to_owned(), 999);
    assert!(
        lying_artifact_counts
            .verify_atlas_artifact("fixture.atlas.bin.zst", &payload, &artifact(false), &limits,)
            .is_err()
    );
    let mut tampered_payload = payload.clone();
    tampered_payload[136] ^= 1;
    assert!(
        restored
            .verify_atlas_artifact(
                "fixture.atlas.bin.zst",
                &tampered_payload,
                &artifact(false),
                &limits,
            )
            .is_err()
    );

    let pretty = serde_json::to_vec_pretty(&restored).unwrap();
    assert!(AtlasManifest::from_canonical_json(&pretty, &limits).is_err());
    let mut mixed = restored.clone();
    mixed.schema_version += 1;
    assert!(mixed.canonical_json(&limits).is_err());
    let mut bad_digest = restored.clone();
    bad_digest
        .oracles
        .pmove_oracle
        .as_mut()
        .unwrap()
        .tool
        .executable_sha256 = "ABC".to_owned();
    assert!(bad_digest.canonical_json(&limits).is_err());
    let mut duplicate_channel = restored.clone();
    duplicate_channel
        .channels
        .push(duplicate_channel.channels[0].clone());
    assert!(duplicate_channel.canonical_json(&limits).is_err());
    let mut out_of_range = restored.clone();
    out_of_range.grid.origin = [0, 0, 0];
    out_of_range.grid.model0_mins = [0, 0, 0];
    out_of_range.grid.model0_maxs = [i64::MAX, 1, 1];
    assert!(out_of_range.canonical_json(&limits).is_err());
    let mut over_rss = restored;
    over_rss.build_peak_rss_bytes = limits.max_build_rss_bytes + 1;
    assert!(over_rss.canonical_json(&limits).is_err());
}

#[test]
fn missing_pmove_authority_is_not_a_canonical_runtime() {
    let limits = AtlasLimits::default();
    let payload = artifact(false).encode_uncompressed(&limits).unwrap();
    let envelope = encode_zstd_envelope(&payload, &limits).unwrap();
    let mut collision_only = manifest(&payload, envelope.len() as u64);
    collision_only.oracles.pmove_oracle = None;
    collision_only.oracles.hook_oracle = None;
    assert!(collision_only.canonical_json(&limits).is_err());
    let bytes = manifest(&payload, envelope.len() as u64)
        .canonical_json(&limits)
        .unwrap();
    let mut explicit_null: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    explicit_null["oracles"]["pmove_oracle"] = serde_json::Value::Null;
    assert!(
        AtlasManifest::from_canonical_json(&serde_json::to_vec(&explicit_null).unwrap(), &limits,)
            .is_err()
    );
    let mut explicit_hook_null: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    explicit_hook_null["oracles"]["hook_oracle"] = serde_json::Value::Null;
    assert!(
        AtlasManifest::from_canonical_json(
            &serde_json::to_vec(&explicit_hook_null).unwrap(),
            &limits,
        )
        .is_err()
    );
}

#[test]
fn collision_oracle_is_mandatory_and_manifest_artifact_verification_rechecks_edges() {
    let limits = AtlasLimits::default();
    let decoded_artifact = artifact(false);
    let payload = decoded_artifact.encode_uncompressed(&limits).unwrap();
    let envelope = encode_zstd_envelope(&payload, &limits).unwrap();
    let source = manifest(&payload, envelope.len() as u64);
    let bytes = source.canonical_json(&limits).unwrap();
    let mut missing_collision: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    missing_collision["oracles"]
        .as_object_mut()
        .unwrap()
        .remove("collision_oracle");
    assert!(
        AtlasManifest::from_canonical_json(
            &serde_json::to_vec(&missing_collision).unwrap(),
            &limits,
        )
        .is_err()
    );

    let mut no_trajectory = source;
    no_trajectory.oracles.pmove_oracle = None;
    no_trajectory.oracles.hook_oracle = None;
    assert!(no_trajectory.canonical_json(&limits).is_err());
    assert!(
        no_trajectory
            .verify_atlas_artifact(
                "fixture.atlas.bin.zst",
                &payload,
                &decoded_artifact,
                &limits,
            )
            .is_err()
    );
}

#[test]
#[ignore = "manual release-mode storage benchmark"]
fn benchmark_sparse_storage_envelope_and_load() {
    let limits = AtlasLimits::default();
    let started = Instant::now();
    let mut l0 = SparseL0::new();
    for ordinal in 0..limits.max_l0_chunks {
        let mut chunk = L0Chunk::new(GridIndex::new(
            (ordinal % 40) as i32 - 20,
            ((ordinal / 40) % 30) as i32 - 15,
            (ordinal / 1200) as i32,
        ));
        for plane in [
            L0BitPlane::Solid,
            L0BitPlane::StandingForbiddenOrigin,
            L0BitPlane::CrouchedForbiddenOrigin,
        ] {
            chunk.set_bit(plane, ordinal % 4096, true).unwrap();
        }
        chunk
            .set_scalar(L0ScalarPlane::HazardSeverity, ordinal % 4096, 200)
            .unwrap();
        l0.insert(chunk, &limits).unwrap();
    }
    let build_elapsed = started.elapsed();
    let artifact = AtlasArtifact {
        origin: AtlasOrigin([0, 0, 0]),
        l0,
        l1: L1Graph::empty(),
        l2: Vec::new(),
        l3: Vec::new(),
    };
    let started = Instant::now();
    let bytes = artifact.encode_uncompressed(&limits).unwrap();
    let encode_elapsed = started.elapsed();
    let started = Instant::now();
    let envelope = encode_zstd_envelope(&bytes, &limits).unwrap();
    let compress_elapsed = started.elapsed();
    let started = Instant::now();
    let decoded = decode_zstd_envelope(&envelope, &limits).unwrap();
    let decompress_elapsed = started.elapsed();
    let started = Instant::now();
    let restored = AtlasArtifact::decode_uncompressed(&decoded, &limits).unwrap();
    let decode_elapsed = started.elapsed();
    assert_eq!(restored.l0.len(), limits.max_l0_chunks);
    eprintln!(
        "sparse_chunks={} raw_bytes={} zstd_bytes={} build_us={} encode_us={} compress_us={} decompress_us={} decode_us={}",
        restored.l0.len(),
        bytes.len(),
        envelope.len(),
        build_elapsed.as_micros(),
        encode_elapsed.as_micros(),
        compress_elapsed.as_micros(),
        decompress_elapsed.as_micros(),
        decode_elapsed.as_micros(),
    );
}
