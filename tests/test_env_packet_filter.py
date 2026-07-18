from types import SimpleNamespace

from harness.env import (
    _is_duplicate_obs_tick,
    _is_fresh_obs_transition,
    _observations_share_tick,
)
from harness.spatial import VoxelSpatialReward


def _accepted_ticks(ticks, previous_tick, previous_is_terminal=False):
    accepted = []
    for tick, is_terminal in ticks:
        if _is_duplicate_obs_tick(
            tick,
            previous_tick,
            is_terminal=is_terminal,
            previous_is_terminal=previous_is_terminal,
        ):
            continue
        accepted.append((tick, is_terminal))
    return accepted


def test_same_tick_duplicates_are_ignored():
    # A duplicate arriving one step late matches the already-returned obs and
    # must not replay its reward fields or terminal bit.
    assert _accepted_ticks([(100, False), (100, False)], previous_tick=100) == []
    assert _accepted_ticks(
        [(100, True)], previous_tick=100, previous_is_terminal=True
    ) == []


def test_same_new_tick_packets_remain_mergeable():
    # Multiple packets first observed in one drain can split reward deltas or
    # carry a same-frame death after the lockstep pre-pass. They are merged by
    # step_all rather than mistaken for a replay of the previous transition.
    assert _accepted_ticks(
        [(101, False), (101, True)], previous_tick=100
    ) == [(101, False), (101, True)]


def test_late_same_tick_terminal_promotes_previous_nonterminal():
    # Python can return the lockstep pre-pass's live packet before a later bot
    # kills this slot in the same engine frame. The terminal/reward packet is
    # a new transition boundary even though its numeric tick is unchanged.
    assert _accepted_ticks(
        [(100, True)], previous_tick=100, previous_is_terminal=False
    ) == [(100, True)]
    previous = SimpleNamespace(tick=100, is_terminal=False)
    terminal = SimpleNamespace(tick=100, is_terminal=True)
    assert _is_fresh_obs_transition(terminal, previous)

    replayed_terminal = SimpleNamespace(tick=100, is_terminal=True)
    assert not _is_fresh_obs_transition(replayed_terminal, terminal)


def test_lower_tick_after_map_reload_is_accepted():
    # Quake resets level.framenum on map load. Freshness is therefore based on
    # exact duplication, not numerical ordering.
    assert _accepted_ticks([(3, False)], previous_tick=9000) == [(3, False)]


def test_bootstrap_alignment_requires_every_slot_on_same_tick():
    assert not _observations_share_tick([])
    assert not _observations_share_tick([SimpleNamespace(tick=101), None])
    assert not _observations_share_tick([
        SimpleNamespace(tick=123), SimpleNamespace(tick=112),
    ])
    assert _observations_share_tick([
        SimpleNamespace(tick=145), SimpleNamespace(tick=145),
    ])


def test_spatial_reward_rng_is_local_and_repeatable():
    first = VoxelSpatialReward.from_env(seed=1234)
    second = VoxelSpatialReward.from_env(seed=1234)

    first._reset_episode_state()
    # Advancing another reward instance cannot steal this instance's draws,
    # which matters when separate servers update concurrently.
    distraction = VoxelSpatialReward.from_env(seed=99)
    for _ in range(10):
        distraction._reset_episode_state()
    second._reset_episode_state()
    assert first.aggression == second.aggression

    first._reset_episode_state()
    second._reset_episode_state()
    assert first.aggression == second.aggression
