import struct
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from harness.causal_protocol import (
    CAUSAL_FIELD_NAMES,
    CAUSAL_FLAGS_MASK,
    CAUSAL_TELEMETRY_FMT,
    CAUSAL_TELEMETRY_SIZE,
    ML_CAUSAL_MAGIC,
    ML_CAUSAL_VERSION,
    CausalFlags,
    parse_causal_telemetry,
)


ROOT = Path(__file__).resolve().parents[1]


def _causal(
    *, flags=CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
        | CausalFlags.TRANSITION_TRAINABLE,
    tick=77, life=4, target_id=0, target_epoch=0,
    source_id=0, source_epoch=0, environmental_mod=0,
    environmental_damage=0, crouch_id=0, crouch_epoch=0,
    echo_tick=76, action_generation=13, hook_zone_id=0,
    hook_attempt_tick=0, hook_action_generation=0,
    routed_role=True,
):
    if routed_role:
        flags |= CausalFlags.ROLE_PLAYING | CausalFlags.ROLE_PUBLIC_PM_NORMAL
    return struct.pack(
        CAUSAL_TELEMETRY_FMT,
        ML_CAUSAL_MAGIC, ML_CAUSAL_VERSION, CAUSAL_TELEMETRY_SIZE,
        int(flags), tick, life, target_id, target_epoch, source_id,
        source_epoch, environmental_mod, environmental_damage, crouch_id,
        crouch_epoch, echo_tick, action_generation, hook_zone_id,
        hook_attempt_tick, hook_action_generation, 0,
    )


def test_private_causal_contract_has_frozen_named_80_byte_layout():
    assert CAUSAL_TELEMETRY_SIZE == 80
    assert len(CAUSAL_FIELD_NAMES) == len(set(CAUSAL_FIELD_NAMES)) == 20
    assert CAUSAL_FIELD_NAMES[4:10] == (
        "tick", "client_life_epoch", "target_id", "target_epoch",
        "environmental_source_id", "environmental_source_epoch",
    )
    assert ML_CAUSAL_VERSION == 2
    assert CAUSAL_FLAGS_MASK == (1 << 22) - 1


def test_byte_parser_accepts_complete_nontrainable_settling_boundary():
    packet = bytearray(_causal())
    settling_flags = (
        CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
        | CausalFlags.ROLE_PLAYING | CausalFlags.ROLE_PUBLIC_PM_NORMAL
    )
    struct.pack_into("<I", packet, 12, int(settling_flags))

    parsed = parse_causal_telemetry(
        bytes(packet), expected_tick=77, require_action_generation=True,
    )

    assert parsed is not None
    assert parsed.echo_valid
    assert parsed.facts_complete
    assert not parsed.transition_trainable


def test_routed_parser_requires_playing_and_normal_for_trainable_packets():
    assert parse_causal_telemetry(
        _causal(routed_role=False), expected_tick=77,
        require_action_generation=True,
    ) is None
    assert parse_causal_telemetry(
        _causal(
            flags=(
                CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
                | CausalFlags.TRANSITION_TRAINABLE
                | CausalFlags.ROLE_PLAYING
            ),
            routed_role=False,
        ),
        expected_tick=77,
        require_action_generation=True,
    ) is None
    forged_normal = _causal(
        flags=CausalFlags.ROLE_PUBLIC_PM_NORMAL,
        routed_role=False,
        echo_tick=0,
        action_generation=0,
    )
    assert parse_causal_telemetry(
        forged_normal, expected_tick=77, require_action_generation=False,
    ) is None


@pytest.mark.parametrize("version", (0, 1, 3))
def test_routed_parser_rejects_old_or_unknown_causal_version(version):
    packet = bytearray(_causal())
    struct.pack_into("<I", packet, 4, version)
    assert parse_causal_telemetry(
        bytes(packet), expected_tick=77, require_action_generation=True,
    ) is None


@pytest.mark.parametrize(
    "malformed_flags",
    (
        CausalFlags.TRANSITION_TRAINABLE,
        CausalFlags.ECHO_VALID | CausalFlags.TRANSITION_TRAINABLE,
        CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE,
    ),
)
def test_byte_parser_rejects_trainable_without_both_prerequisites(
    malformed_flags,
):
    packet = bytearray(_causal())
    struct.pack_into("<I", packet, 12, int(malformed_flags))

    assert parse_causal_telemetry(
        bytes(packet), expected_tick=77, require_action_generation=True,
    ) is None


def test_server_client_and_python_causal_field_names_are_identical():
    lithium_header = (ROOT.parent / "q2-lithium-3zb2" / "ml_bridge.h").read_text()
    client_source = (
        ROOT.parent / "q2-ml-client" / "src" / "client" / "cl_ml_harness.c"
    ).read_text()

    def names(source: str) -> tuple[str, ...]:
        blocks = re.findall(
            r"typedef struct\s*\{([^}]+)\}\s*ml_causal_telemetry_t\s*;",
            source,
            flags=re.DOTALL,
        )
        assert len(blocks) == 1
        result = []
        for name, count in re.findall(
            r"uint32_t\s+([a-z_][a-z0-9_]*)(?:\[(\d+)\])?\s*;",
            blocks[0],
        ):
            if count:
                result.extend(f"{name}_{index}" for index in range(int(count)))
            else:
                result.append(name)
        return tuple(result)

    assert names(lithium_header) == names(client_source) == CAUSAL_FIELD_NAMES
    for flag in CausalFlags:
        match = re.search(
            rf"#define\s+ML_CAUSAL_{flag.name}\s+\(1u\s*<<\s*(\d+)\)",
            lithium_header,
        )
        assert match is not None, flag.name
        assert 1 << int(match.group(1)) == int(flag)


def test_life_epoch_cannot_alias_after_disconnected_slot_reuse():
    source = (ROOT.parent / "q2-lithium-3zb2" / "p_client.c").read_text()
    assert "static uint32_t ml_life_epoch_by_slot[MAX_CLIENTS]" in source
    assert "ml_life_epoch_by_slot[index]++" in source
    assert (
        "client->resp.ml_life_epoch = ml_life_epoch_by_slot[index]" in source
    )
    assert "client->resp.ml_life_epoch++" not in source


def test_hook_attempt_authority_is_after_engine_availability_checks_only():
    lithium = ROOT.parent / "q2-lithium-3zb2"
    assert "ML_CausalHookAttempt(ent);" not in (lithium / "ml_obs.c").read_text()
    assert "ML_CausalHookAttempt(ent);" not in (
        lithium / "ml_client_telemetry.c"
    ).read_text()
    hook_source = (lithium / "l_hook.c").read_text()
    function = hook_source.split("void Weapon_Hook_Fire(edict_t *ent)", 1)[1]
    function = function.split("\n}", 1)[0]
    authority = function.index("ML_CausalHookAttempt(ent);")
    assert function.index("ML_CausalHookFireAccepted(") < authority


def test_persistent_target_and_environmental_source_stay_stable_then_change():
    flags = (
        CausalFlags.TARGET_VALID | CausalFlags.TARGET_HIT
        | CausalFlags.ENV_SOURCE_ACTIVE | CausalFlags.ENV_SOURCE_EVIDENCE
        | CausalFlags.ENV_DAMAGE | CausalFlags.ECHO_VALID
        | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE
    )
    first = parse_causal_telemetry(_causal(
        flags=flags, target_id=3, target_epoch=9, source_id=0xA0B0C0D0,
        source_epoch=2, environmental_mod=19, environmental_damage=8,
    ), expected_tick=77, require_action_generation=True)
    second = parse_causal_telemetry(_causal(
        flags=flags, tick=78, target_id=3, target_epoch=9,
        source_id=0xA0B0C0D0, source_epoch=2, environmental_mod=19,
        environmental_damage=4, echo_tick=77,
    ), expected_tick=78, require_action_generation=True)
    next_life = parse_causal_telemetry(_causal(
        flags=flags, tick=79, target_id=3, target_epoch=10,
        source_id=0xA0B0C0D0, source_epoch=3, environmental_mod=19,
        environmental_damage=4, echo_tick=78,
    ), expected_tick=79, require_action_generation=True)
    assert first is not None and second is not None and next_life is not None
    assert (first.target_id, first.target_epoch) == (second.target_id, second.target_epoch)
    assert (
        first.environmental_source_id,
        first.environmental_source_epoch,
    ) == (second.environmental_source_id, second.environmental_source_epoch)
    assert next_life.target_epoch != second.target_epoch
    assert (
        next_life.environmental_source_epoch
        != second.environmental_source_epoch
    )


def test_true_hook_necessity_and_environmental_clear_are_explicit():
    flags = (
        CausalFlags.ENV_SOURCE_EVIDENCE
        | CausalFlags.ENV_SOURCE_CLEARED | CausalFlags.HOOK_ATTEMPTED
        | CausalFlags.HOOK_ATTACHED | CausalFlags.HOOK_VALID
        | CausalFlags.HOOK_NECESSITY_KNOWN
        | CausalFlags.HOOK_WAS_NECESSARY | CausalFlags.ECHO_VALID
        | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE
    )
    parsed = parse_causal_telemetry(_causal(
        flags=flags, source_id=55, source_epoch=7, environmental_mod=22,
        hook_zone_id=4, hook_attempt_tick=76, hook_action_generation=12,
    ), expected_tick=77, require_action_generation=True)
    assert parsed is not None
    assert parsed.hook_was_necessary
    assert parsed.has(CausalFlags.ENV_SOURCE_CLEARED)


def test_environmental_event_requires_nonzero_source_identity():
    flags = (
        CausalFlags.ENV_SOURCE_EVIDENCE | CausalFlags.ENV_DAMAGE
        | CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
        | CausalFlags.TRANSITION_TRAINABLE
    )
    assert parse_causal_telemetry(_causal(
        flags=flags, environmental_mod=19, environmental_damage=8,
    ), expected_tick=77, require_action_generation=True) is None

    remembered_source = parse_causal_telemetry(_causal(
        source_id=99, source_epoch=4, environmental_mod=19,
    ), expected_tick=77, require_action_generation=True)
    assert remembered_source is not None
    assert not remembered_source.has(CausalFlags.ENV_SOURCE_EVIDENCE)


def test_delayed_hook_attach_invalid_and_necessity_keep_origin_identity():
    attach = parse_causal_telemetry(_causal(
        flags=(
            CausalFlags.HOOK_ATTEMPTED | CausalFlags.HOOK_ATTACHED
            | CausalFlags.HOOK_VALID | CausalFlags.HOOK_NECESSITY_KNOWN
            | CausalFlags.HOOK_WAS_NECESSARY | CausalFlags.ECHO_VALID
            | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE
        ),
        hook_zone_id=3, hook_attempt_tick=72, hook_action_generation=8,
    ), expected_tick=77, require_action_generation=True)
    assert attach is not None and attach.hook_was_necessary
    assert (attach.hook_attempt_tick, attach.hook_action_generation) == (72, 8)

    invalid = parse_causal_telemetry(_causal(
        flags=(
            CausalFlags.HOOK_ATTEMPTED | CausalFlags.HOOK_INVALID
            | CausalFlags.HOOK_NECESSITY_KNOWN | CausalFlags.ECHO_VALID
            | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE
        ),
        hook_attempt_tick=62, hook_action_generation=4,
    ), expected_tick=77, require_action_generation=True)
    assert invalid is not None
    assert invalid.has(CausalFlags.HOOK_INVALID)


def test_hook_origin_rejects_missing_future_and_same_tick_generation_mismatch():
    flags = (
        CausalFlags.HOOK_ATTEMPTED | CausalFlags.HOOK_INVALID
        | CausalFlags.HOOK_NECESSITY_KNOWN | CausalFlags.ECHO_VALID
        | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE
    )
    for packet in (
        _causal(flags=flags),
        _causal(
            flags=flags, hook_attempt_tick=78, hook_action_generation=13,
        ),
        _causal(
            flags=flags, hook_attempt_tick=77, hook_action_generation=12,
        ),
    ):
        assert parse_causal_telemetry(
            packet, expected_tick=77, require_action_generation=True,
        ) is None

    assert parse_causal_telemetry(_causal(
        hook_attempt_tick=72, hook_action_generation=8,
    ), expected_tick=77, require_action_generation=True) is None


def test_c_authority_proves_hook_denial_and_true_necessity(tmp_path):
    cc = shutil.which("cc")
    lithium = ROOT.parent / "q2-lithium-3zb2"
    if not cc or not (lithium / "ml_bridge.c").is_file():
        pytest.skip("Lithium C source/compiler unavailable")
    probe = tmp_path / "hook_necessity.c"
    probe.write_text(
        '#include "ml_bridge.h"\n'
        'int main(void) {\n'
        '  if (ML_CausalHookFireAccepted(1, 2.0f, 0.0f, 1.0f)) return 1;\n'
        '  if (ML_CausalHookFireAccepted(0, 1.0f, 0.5f, 1.0f)) return 2;\n'
        '  if (!ML_CausalHookFireAccepted(0, 1.5f, 0.5f, 1.0f)) return 3;\n'
        '  if (!ML_HookNecessityBudgetProven(481.0f, 1.5f, 1)) return 4;\n'
        '  if (ML_HookNecessityBudgetProven(480.0f, 1.5f, 1)) return 5;\n'
        '  if (ML_HookNecessityBudgetProven(481.0f, 1.51f, 1)) return 6;\n'
        '  if (ML_HookNecessityBudgetProven(481.0f, 1.5f, 0)) return 7;\n'
        '  if (!ML_CausalHookOriginValid(77, 72, 8, 1)) return 8;\n'
        '  if (ML_CausalHookOriginValid(77, 0, 8, 1)) return 9;\n'
        '  if (ML_CausalHookOriginValid(77, 78, 8, 1)) return 10;\n'
        '  if (ML_CausalHookOriginValid(77, 72, 193, 1)) return 11;\n'
        '  if (!ML_CausalHookOriginValid(77, 72, 0, 0)) return 12;\n'
        '  if (ML_CausalEnvironmentalSourceEpoch(2, 55, 55, 1, 0) != 2) return 13;\n'
        '  if (ML_CausalEnvironmentalSourceEpoch(2, 55, 55, 0, 29) != 2) return 14;\n'
        '  if (ML_CausalEnvironmentalSourceEpoch(2, 55, 55, 0, 30) != 3) return 15;\n'
        '  if (ML_CausalEnvironmentalSourceEpoch(2, 55, 56, 1, 0) != 3) return 16;\n'
        '  if (ML_CausalEnvironmentalSourceEpoch(0, 0, 55, 0, 0) != 1) return 17;\n'
        '  return 0;\n}\n',
        encoding="ascii",
    )
    bridge = tmp_path / "ml_bridge.o"
    subprocess.run([
        cc, "-std=gnu99", "-O1", "-DNEED_STRLCAT", "-DNEED_STRLCPY",
        "-fPIC", "-ffunction-sections", "-fdata-sections", "-c",
        str(lithium / "ml_bridge.c"), "-o", str(bridge),
    ], check=True, cwd=lithium)
    executable = tmp_path / "hook_necessity"
    subprocess.run([
        cc, "-std=gnu99", f"-I{lithium}", "-Wl,--gc-sections",
        str(probe), str(bridge), "-lm", "-o", str(executable),
    ], check=True)
    subprocess.run([str(executable)], check=True)


def test_unresolved_hook_verdict_is_trainable_but_cannot_claim_necessity():
    pending = _causal(flags=(
        CausalFlags.HOOK_ATTEMPTED | CausalFlags.HOOK_ATTACHED
        | CausalFlags.HOOK_VALID | CausalFlags.ECHO_VALID
        | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE
    ), hook_zone_id=0, hook_attempt_tick=77, hook_action_generation=13)
    parsed = parse_causal_telemetry(
        pending, expected_tick=77, require_action_generation=True
    )
    assert parsed is not None
    assert parsed.facts_complete
    assert parsed.transition_trainable
    assert not parsed.has(CausalFlags.HOOK_NECESSITY_KNOWN)
    assert not parsed.hook_was_necessary

    forged_necessary = bytearray(pending)
    forged = int(
        CausalFlags.HOOK_ATTEMPTED | CausalFlags.HOOK_ATTACHED
        | CausalFlags.HOOK_VALID | CausalFlags.ECHO_VALID
        | CausalFlags.HOOK_WAS_NECESSARY
        | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE
    )
    struct.pack_into("<I", forged_necessary, 12, forged)
    assert parse_causal_telemetry(
        bytes(forged_necessary), expected_tick=77,
        require_action_generation=True,
    ) is None


def test_rejects_unknown_flags_reserved_words_and_incoherent_event_identity():
    packet = bytearray(_causal())
    struct.pack_into("<I", packet, 12, 1 << 31)
    assert parse_causal_telemetry(
        bytes(packet), expected_tick=77, require_action_generation=True
    ) is None

    packet = bytearray(_causal())
    struct.pack_into("<I", packet, 76, 1)
    assert parse_causal_telemetry(
        bytes(packet), expected_tick=77, require_action_generation=True
    ) is None

    packet = _causal(flags=(
        CausalFlags.TARGET_HIT | CausalFlags.ECHO_VALID
        | CausalFlags.FACTS_COMPLETE | CausalFlags.TRANSITION_TRAINABLE
    ))
    assert parse_causal_telemetry(
        packet, expected_tick=77, require_action_generation=True
    ) is None

    impossible_packets = (
        _causal(
            flags=(
                CausalFlags.TARGET_VALID | CausalFlags.TARGET_KILLED
                | CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
                | CausalFlags.TRANSITION_TRAINABLE
            ), target_id=2, target_epoch=3,
        ),
        _causal(
            flags=(
                CausalFlags.ENV_SOURCE_EVIDENCE
                | CausalFlags.ENV_SOURCE_CLEARED | CausalFlags.ENV_DAMAGE
                | CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
                | CausalFlags.TRANSITION_TRAINABLE
            ), source_id=4, source_epoch=2, environmental_mod=19,
            environmental_damage=1,
        ),
        _causal(
            flags=(
                CausalFlags.HOOK_ATTEMPTED | CausalFlags.HOOK_INVALID
                | CausalFlags.ECHO_VALID | CausalFlags.FACTS_COMPLETE
                | CausalFlags.TRANSITION_TRAINABLE
            ), hook_attempt_tick=77, hook_action_generation=13,
        ),
    )
    for impossible in impossible_packets:
        assert parse_causal_telemetry(
            impossible, expected_tick=77, require_action_generation=True,
        ) is None
