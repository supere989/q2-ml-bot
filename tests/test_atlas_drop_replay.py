from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness import atlas_drop_replay as drop
from harness import atlas_exact_drops as exact


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def state(
    index: int, velocity_z: float, grounded: bool, *, waterlevel: int = 0,
    origin_fixed: tuple[int, int, int] = (0, 0, 193), gravity: int = 800,
) -> dict:
    velocity_fixed = [0, 0, int(velocity_z * 8)]
    return {
        "command_index": index,
        "origin": [component * 0.125 for component in origin_fixed],
        "velocity": [component * 0.125 for component in velocity_fixed],
        "origin_fixed": list(origin_fixed),
        "velocity_fixed": velocity_fixed,
        "pm_type": 0,
        "pm_flags": 4 if grounded else 0,
        "pm_time": 0,
        "gravity": gravity,
        "viewangles": [0.0, 0.0, 0.0],
        "viewheight": 22.0,
        "mins": [-16.0, -16.0, -24.0],
        "maxs": [16.0, 16.0, 32.0],
        "grounded": grounded,
        "waterlevel": waterlevel,
        "watertype": 0,
        "touch_count": 1 if grounded else 0,
    }


class OracleFixture:
    pmove_tool_identity = digest("pmove-tool")
    pmove_physics_identity = digest("pmove-physics")
    fall_tool_identity = digest("fall-tool")
    fall_constants = "test-exact-fall-constants-v1"

    def __init__(
        self, frames: list[dict], *, pmove_sha256: str, fall_sha256: str,
        map_sha256: str,
    ) -> None:
        self.frames = frames
        self.pmove_sha256 = pmove_sha256
        self.fall_sha256 = fall_sha256
        self.map_sha256 = map_sha256
        self.calls: list[tuple[str, list[dict]]] = []
        self.fall_source = {
            "shared_c_sha256": digest("fall-c"),
            "shared_h_sha256": digest("fall-h"),
            "integration_sha256": digest("fall-integration"),
            "game_header_sha256": digest("fall-game-header"),
            "constants_sha256": hashlib.sha256(self.fall_constants.encode()).hexdigest(),
            "build_contract": "test-linux-c99-f32-fall-v1",
            "tool_closure_sha256": self.fall_tool_identity,
        }

    def fall_physics_identity(self, fall: dict) -> str:
        parameters = {
            "fall_damagemod": drop._f32(fall["fall_damagemod"]),
            "deathmatch": fall["deathmatch"],
            "dmflags": fall["dmflags"],
        }
        return drop._fall_physics_identity(parameters, self.fall_source, self.fall_constants)

    def manifest(
        self, map_path: Path, *, replay_id: str = "drop", fall_damagemod: float = 1.0,
        deathmatch: bool = True, dmflags: int = 0, health: int = 100,
        cadence_msec: int = 100, command_msec: int | None = None,
        dynamic_movers: bool = False,
    ) -> dict:
        if command_msec is None:
            command_msec = cadence_msec
        fall = {
            "fall_damagemod": fall_damagemod,
            "deathmatch": deathmatch,
            "dmflags": dmflags,
            "health": health,
        }
        return {
            "schema": drop.MANIFEST_SCHEMA,
            "id": replay_id,
            "map_path": str(map_path),
            "horizon_frames": len(self.frames),
            "cadence_msec": cadence_msec,
            "dynamic_movers": dynamic_movers,
            "pmove": {
                "origin": [0.0, 0.0, 96.0],
                "velocity": [0.0, 0.0, self.frames[0]["velocity"][2]],
                "pm_type": 0,
                "pm_flags": 0,
                "pm_time": 0,
                "gravity": 800,
                "airaccelerate": 0.0,
                "delta_angles_short": [0, 0, 0],
                "snapinitial": False,
                "commands": [{"msec": command_msec} for _ in self.frames],
            },
            "fall": fall,
            "authorities": {
                "pmove": {
                    "executable_sha256": self.pmove_sha256,
                    "tool_identity": self.pmove_tool_identity,
                    "physics_identity": self.pmove_physics_identity,
                    "map_sha256": self.map_sha256,
                },
                "fall": {
                    "executable_sha256": self.fall_sha256,
                    "tool_identity": self.fall_tool_identity,
                    "physics_identity": self.fall_physics_identity(fall),
                },
            },
        }

    def _pmove_identity(self, request: dict) -> dict:
        provenance = {
            "schema": "q2-oracle-tool-identity-v1",
            "tool_identity": self.pmove_tool_identity,
            "source_closure_sha256": digest("pmove-closure"),
            "source_closure_count": 12,
            "build_identity_sha256": digest("pmove-build"),
            "compiler": {
                "command": "cc", "version": "fixture cc 1", "target": "fixture-linux",
                "executable_sha256": digest("cc"),
            },
            "archiver": {
                "command": "ar", "version": "fixture ar 1",
                "executable_sha256": digest("ar"),
            },
            "build": {"cflags": "-O1", "ldflags": "-lm"},
        }
        return {
            "ok": True,
            "id": request["id"],
            "op": "identity",
            "schema": drop.PMOVE_SCHEMA,
            "tool_identity": self.pmove_tool_identity,
            "physics_identity": self.pmove_physics_identity,
            "map_sha256": self.map_sha256,
            "map_checksum": 424242,
            "parameters": {
                "gravity": request["gravity"],
                "airaccelerate": request["airaccelerate"],
                "constants": "fixture-pmove-constants-v1",
            },
            "provenance": provenance,
            "source": {
                "collision_sha256": digest("collision-c"),
                "pmove_sha256": digest("pmove-c"),
                "shared_header_sha256": digest("shared-h"),
                "shared_source_sha256": digest("shared-c"),
            },
        }

    def _pmove_response(self, request: dict) -> dict:
        frames = deepcopy(self.frames)
        return {
            "ok": True,
            "id": request["id"],
            "op": "simulate",
            "schema": drop.PMOVE_SCHEMA,
            "tool_identity": self.pmove_tool_identity,
            "physics_identity": self.pmove_physics_identity,
            "map_sha256": self.map_sha256,
            "map_checksum": 424242,
            "frames": frames,
            "final": deepcopy(frames[-1]),
            "command_count": len(frames),
        }

    def _fall_identity(self, request: dict) -> dict:
        parameters = {
            "fall_damagemod": request["fall_damagemod"],
            "deathmatch": request["deathmatch"],
            "dmflags": request["dmflags"],
        }
        return {
            "ok": True,
            "id": request["id"],
            "op": "identity",
            "schema": drop.FALL_SCHEMA,
            "physics_identity": drop._fall_physics_identity(
                parameters, self.fall_source, self.fall_constants,
            ),
            "tool_identity": self.fall_tool_identity,
            "parameters": parameters,
            "constants": self.fall_constants,
            "source": deepcopy(self.fall_source),
        }

    @staticmethod
    def _fall_law(request: dict) -> dict:
        delta = request["velocity_z"] - request["old_velocity_z"]
        delta = delta * delta * 0.0001
        result = {
            "suppression": "none",
            "severity": "none",
            "delta": delta,
            "fall_value": 0.0,
            "fall_time_offset": 0.0,
            "emit_event": False,
            "set_fall_state": False,
            "set_pain_debounce": False,
            "damage": 0,
            "apply_damage": False,
            "unmitigated_health_after": request["health"],
            "unmitigated_lethal": False,
        }
        if request["waterlevel"] == 3:
            result["suppression"] = "underwater"
            return result
        if request["waterlevel"] == 2:
            delta *= 0.25
        if request["waterlevel"] == 1:
            delta *= 0.5
        result["delta"] = delta
        if delta < 1:
            result["suppression"] = "below_threshold"
            return result
        if delta < 15:
            result["severity"] = "footstep"
            result["emit_event"] = True
            return result
        result["fall_value"] = min(delta * 0.5, 40.0)
        result["fall_time_offset"] = 0.3
        result["set_fall_state"] = True
        if delta <= 30:
            result["severity"] = "short"
            result["emit_event"] = True
            return result
        result["severity"] = "far" if delta >= 55 else "fall"
        result["emit_event"] = request["health"] > 0
        result["set_pain_debounce"] = True
        damage = max(int((delta - 30) / 2), 1)
        damage = int(damage * request["fall_damagemod"])
        apply_damage = (not request["deathmatch"]) or not (request["dmflags"] & 8)
        health_after = request["health"] - damage if apply_damage else request["health"]
        result.update({
            "damage": damage,
            "apply_damage": apply_damage,
            "unmitigated_health_after": health_after,
            "unmitigated_lethal": apply_damage and damage > 0 and health_after <= 0,
        })
        return result

    def _fall_response(self, request: dict) -> dict:
        parameters = {
            "fall_damagemod": request["fall_damagemod"],
            "deathmatch": request["deathmatch"],
            "dmflags": request["dmflags"],
        }
        return {
            "ok": True,
            "id": request["id"],
            "op": "evaluate",
            "schema": drop.FALL_SCHEMA,
            "physics_identity": drop._fall_physics_identity(
                parameters, self.fall_source, self.fall_constants,
            ),
            "tool_identity": self.fall_tool_identity,
            "input": {key: value for key, value in request.items() if key not in {"id", "op"}},
            **self._fall_law(request),
        }

    def __call__(
        self, kind: str, executable: Path, requests: list[dict], map_path: Path | None,
    ) -> list[dict]:
        del executable, map_path
        self.calls.append((kind, deepcopy(requests)))
        if kind == "pmove":
            return [self._pmove_identity(requests[0]), self._pmove_response(requests[1])]
        if kind == "fall":
            return [self._fall_identity(requests[0]), self._fall_response(requests[1])]
        raise AssertionError(kind)


class DropReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="atlas-drop-replay-")
        self.root = Path(self.temporary.name)
        self.map = self.root / "fixture.bsp"
        self.pmove = self.root / "q2-pmove-oracle"
        self.fall = self.root / "q2-fall-oracle"
        self.map.write_bytes(b"fixture IBSP bytes")
        self.pmove.write_bytes(b"fixture pmove executable")
        self.fall.write_bytes(b"fixture fall executable")
        self.map_sha256 = hashlib.sha256(self.map.read_bytes()).hexdigest()
        self.pmove_sha256 = hashlib.sha256(self.pmove.read_bytes()).hexdigest()
        self.fall_sha256 = hashlib.sha256(self.fall.read_bytes()).hexdigest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def fixture(self, velocity: float, *, waterlevel: int = 0) -> OracleFixture:
        return OracleFixture(
            [
                state(0, velocity, False, origin_fixed=(0, 0, 512)),
                state(1, 0, True, waterlevel=waterlevel),
            ],
            pmove_sha256=self.pmove_sha256,
            fall_sha256=self.fall_sha256,
            map_sha256=self.map_sha256,
        )

    def replay(self, fixture: OracleFixture, **manifest_changes: object) -> dict:
        manifest = fixture.manifest(self.map, **manifest_changes)
        return drop.replay_drop(
            manifest, pmove_oracle=self.pmove, fall_oracle=self.fall, runner=fixture,
        )

    def test_tesla_vectors_and_injected_session_launch_no_subprocesses(self) -> None:
        expected = {
            -300: ("footstep", True, False, 0),
            -400: ("short", True, False, 0),
            -600: ("fall", True, False, 3),
            -1000: ("far", True, False, 35),
            -1600: ("far", False, True, 113),
        }
        with patch("harness.atlas_drop_replay.subprocess.run") as subprocess_run:
            for velocity, (severity, safe, lethal, damage) in expected.items():
                with self.subTest(velocity=velocity):
                    fixture = self.fixture(velocity)
                    result = self.replay(fixture, replay_id=f"tesla-{abs(velocity)}")
                    self.assertEqual(result["classification"], "Exact")
                    self.assertEqual(result["severity"], severity)
                    self.assertEqual(result["safe"], safe)
                    self.assertEqual(result["lethal"], lethal)
                    self.assertEqual(result["authorities"]["fall"]["response"]["damage"], damage)
                    self.assertEqual(result["landing"]["old_velocity_z"], velocity)
                    self.assertEqual(result["landing"]["velocity_z"], 0)
                    self.assertEqual(result["landing"]["origin_fixed"], [0, 0, 193])
                    self.assertEqual(len(result["trajectory_sha256"]), 64)
                    self.assertFalse(result["fall_request"]["hook_out"])
                    self.assertFalse(result["fall_request"]["grapple_present"])
            subprocess_run.assert_not_called()

    def test_water_mitigation_zero_through_three(self) -> None:
        expected = {
            0: ("far", 35),
            1: ("fall", 10),
            2: ("short", 0),
            3: ("none", 0),
        }
        for waterlevel, (severity, damage) in expected.items():
            with self.subTest(waterlevel=waterlevel):
                fixture = self.fixture(-1000, waterlevel=waterlevel)
                result = self.replay(fixture, replay_id=f"water-{waterlevel}")
                self.assertEqual(result["classification"], "Exact")
                self.assertEqual(result["severity"], severity)
                self.assertEqual(result["authorities"]["fall"]["response"]["damage"], damage)
                self.assertEqual(result["fall_request"]["waterlevel"], waterlevel)

    def test_df_no_falling_preserves_severity_but_is_safe(self) -> None:
        fixture = self.fixture(-1000)
        result = self.replay(fixture, dmflags=8, health=10)
        self.assertEqual(result["classification"], "Exact")
        self.assertEqual(result["severity"], "far")
        self.assertTrue(result["safe"])
        self.assertFalse(result["lethal"])
        self.assertFalse(result["authorities"]["fall"]["response"]["apply_damage"])

    def test_no_landing_and_same_frame_initial_ground_are_unknown(self) -> None:
        scenarios = (
            [state(0, -600, False), state(1, -650, False)],
            [state(0, 0, True), state(1, 0, True)],
        )
        for frames in scenarios:
            with self.subTest(grounded=[frame["grounded"] for frame in frames]):
                fixture = OracleFixture(
                    frames, pmove_sha256=self.pmove_sha256,
                    fall_sha256=self.fall_sha256, map_sha256=self.map_sha256,
                )
                result = self.replay(fixture)
                self.assertEqual(result["classification"], "Unknown")
                self.assertEqual(result["reason"], "no_landing")
                self.assertTrue(result["omit_controlled_drop_edge"])
                self.assertNotIn("safe", result)
                self.assertEqual([kind for kind, _ in fixture.calls], ["pmove"])

    def test_batched_no_landing_has_zero_stage_evidence(self) -> None:
        fixture = OracleFixture(
            [state(0, -600, False), state(1, -650, False)],
            pmove_sha256=self.pmove_sha256,
            fall_sha256=self.fall_sha256,
            map_sha256=self.map_sha256,
        )

        class PmoveProcess:
            binary = self.pmove
            requests = 0
            max_requests = 100
            identity = {
                "parameters": {"gravity": 800, "airaccelerate": 0.0},
                "tool_identity": fixture.pmove_tool_identity,
                "physics_identity": fixture.pmove_physics_identity,
                "map_sha256": self.map_sha256,
            }

            def call(process_self, requests):
                process_self.requests += len(requests)
                return [fixture._pmove_identity(request) for request in requests]

        default_fall = {
            "fall_damagemod": 1.0, "deathmatch": True, "dmflags": 0,
        }

        class FallProcess:
            binary = self.fall
            requests = 0
            max_requests = 100
            identity = {
                "tool_identity": fixture.fall_tool_identity,
                "physics_identity": fixture.fall_physics_identity(default_fall),
            }

            def call(process_self, requests):
                process_self.requests += len(requests)
                if not requests:
                    return []
                raise AssertionError("no-landing trajectory reached fall authority")

        pmove_process = PmoveProcess()
        fall_process = FallProcess()
        request = fixture.manifest(
            self.map, replay_id="batched-no-landing",
        )["pmove"]
        provisional = exact.DropTrajectory(
            identifier="batched-no-landing", source_l1=(0, 0, 0),
            direction=(1, 0), mode="ground", request=request, response={},
        )
        replay_manifest = exact._drop_manifest(
            provisional, bsp=self.map, pmove_process=pmove_process,
            fall_process=fall_process,
        )
        stage = drop.pmove_requests_for_drop_manifest(replay_manifest)
        trajectory = exact.DropTrajectory(
            identifier=provisional.identifier, source_l1=provisional.source_l1,
            direction=provisional.direction, mode=provisional.mode,
            request=request,
            response=fixture._pmove_response(stage["simulate_request"]),
        )
        result = exact.classify_drop_trajectories(
            [trajectory], bsp=self.map, pmove_process=pmove_process,
            fall_process=fall_process,
        )[0]
        self.assertEqual(result["classification"]["reason"], "no_landing")
        self.assertEqual(result["evidence"], 0)
        self.assertEqual(result["validation_version"], 0)
        self.assertEqual(fall_process.requests, 0)
        summary = exact.summarize_drop_classifications([result])
        self.assertEqual(summary["evidence"], 0)
        self.assertEqual(summary["validation_version"], 0)

    def test_first_landing_is_authoritative_when_final_reaches_another_floor(self) -> None:
        frames = [
            state(0, -600, False, origin_fixed=(0, 0, 512)),
            state(1, 0, True, origin_fixed=(0, 0, 193)),
            state(2, -200, False, origin_fixed=(128, 0, 256)),
            state(3, 0, True, origin_fixed=(256, 0, 64)),
        ]
        fixture = OracleFixture(
            frames, pmove_sha256=self.pmove_sha256,
            fall_sha256=self.fall_sha256, map_sha256=self.map_sha256,
        )
        result = self.replay(fixture, replay_id="first-landing-only")
        self.assertEqual(result["classification"], "Exact")
        self.assertEqual(result["landing"]["command_index"], 1)
        self.assertEqual(result["landing"]["origin_fixed"], [0, 0, 193])
        self.assertNotEqual(
            result["landing"]["origin_fixed"], frames[-1]["origin_fixed"]
        )

    def test_invalid_water_cadence_dynamic_mover_and_missing_fall_fail_closed(self) -> None:
        invalid_water = self.fixture(-1000, waterlevel=4)
        water_result = self.replay(invalid_water)
        self.assertEqual(water_result["classification"], "Unknown")
        self.assertEqual(water_result["reason"], "invalid_water")

        invalid_cadence = self.fixture(-1000)
        cadence_result = self.replay(invalid_cadence, cadence_msec=100, command_msec=50)
        self.assertEqual(cadence_result["classification"], "Unknown")
        self.assertEqual(cadence_result["reason"], "invalid_cadence")
        self.assertEqual(invalid_cadence.calls, [])

        mover = self.fixture(-1000)
        mover_result = self.replay(mover, dynamic_movers=True)
        self.assertEqual(mover_result["reason"], "unsupported_dynamic_mover")
        self.assertEqual(mover.calls, [])

        missing = self.fixture(-1000)
        missing_result = drop.replay_drop(
            missing.manifest(self.map), pmove_oracle=self.pmove,
            fall_oracle=None, runner=missing,
        )
        self.assertEqual(missing_result["reason"], "missing_fall_authority")
        self.assertNotIn("severity", missing_result)

    def test_out_of_order_and_tampered_frames_fail_closed(self) -> None:
        out_of_order = self.fixture(-1000)
        out_of_order.frames[0]["command_index"] = 1
        result = self.replay(out_of_order)
        self.assertEqual(result["classification"], "Unknown")
        self.assertTrue(result["omit_controlled_drop_edge"])

        tampered = self.fixture(-1000)
        tampered.frames[1]["origin"] = [1.0, 0.0, 24.125]
        result = self.replay(tampered)
        self.assertEqual(result["classification"], "Unknown")
        self.assertIn("tampered", result["detail"])

    def test_pure_prepare_and_evaluate_accept_prebatched_mappings(self) -> None:
        fixture = self.fixture(-600)
        manifest = fixture.manifest(self.map)
        request_stage = drop.pmove_requests_for_drop_manifest(manifest)
        self.assertEqual(request_stage["classification"], "NeedsPmoveAuthority")
        pmove_requests = (
            request_stage["identity_request"], request_stage["simulate_request"],
        )
        pmove_identity, pmove_response = fixture(
            "pmove", self.pmove, list(pmove_requests), self.map,
        )
        prepared = drop.prepare_drop_fall_request(
            manifest,
            pmove_identity=pmove_identity,
            pmove_response=pmove_response,
            pmove_executable_sha256=self.pmove_sha256,
            map_sha256=self.map_sha256,
        )
        self.assertEqual(prepared["classification"], "NeedsFallAuthority")
        self.assertEqual(prepared["fall_request"]["old_velocity_z"], -600)
        self.assertEqual(prepared["fall_request"]["velocity_z"], 0)
        fall_identity, fall_response = fixture(
            "fall", self.fall,
            [prepared["fall_identity_request"], prepared["fall_request"]], None,
        )
        with patch("harness.atlas_drop_replay.subprocess.run") as subprocess_run:
            result = drop.evaluate_drop_evidence(
                manifest,
                pmove_identity=pmove_identity,
                pmove_response=pmove_response,
                fall_identity=fall_identity,
                fall_response=fall_response,
                pmove_executable_sha256=self.pmove_sha256,
                fall_executable_sha256=self.fall_sha256,
                map_sha256=self.map_sha256,
            )
            subprocess_run.assert_not_called()
        self.assertEqual(result["classification"], "Exact")
        self.assertEqual(result["severity"], "fall")
        self.assertEqual(result["fall_request"], prepared["fall_request"])

        tampered_fall = deepcopy(fall_response)
        tampered_fall["input"]["old_velocity_z"] = -599
        rejected = drop.evaluate_drop_evidence(
            manifest,
            pmove_identity=pmove_identity,
            pmove_response=pmove_response,
            fall_identity=fall_identity,
            fall_response=tampered_fall,
            pmove_executable_sha256=self.pmove_sha256,
            fall_executable_sha256=self.fall_sha256,
            map_sha256=self.map_sha256,
        )
        self.assertEqual(rejected["classification"], "Unknown")
        self.assertNotIn("safe", rejected)


if __name__ == "__main__":
    unittest.main()
