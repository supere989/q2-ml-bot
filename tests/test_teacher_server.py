"""Tests for the wrapper-driven teacher rotation (tools/teacher_server.py)."""

from __future__ import annotations

import io
import threading

from tools.map_farm_client import ShuffledStockRotation
from tools.teacher_server import (
    _load_stock_draws,
    _save_stock_draws,
    _select_next_map,
    _watch_stdout,
)


def test_stock_draws_roundtrip(tmp_path):
    assert _load_stock_draws(tmp_path) == 0
    _save_stock_draws(tmp_path, 7)
    assert _load_stock_draws(tmp_path) == 7


def test_stock_draws_corrupt_state_defaults_to_zero(tmp_path):
    (tmp_path / ".teacher_stock_draws").write_text("not-a-number")
    assert _load_stock_draws(tmp_path) == 0


def test_stock_draws_replay_reproduces_rotation(tmp_path):
    reference = ShuffledStockRotation(["q2dm2", "q2dm4", "q2dm6", "q2dm8"], seed=2204)
    expected = [reference.next() for _ in range(12)]

    _save_stock_draws(tmp_path, 5)
    resumed = ShuffledStockRotation(["q2dm2", "q2dm4", "q2dm6", "q2dm8"], seed=2204)
    for _ in range(_load_stock_draws(tmp_path)):
        resumed.next()
    assert [resumed.next() for _ in range(7)] == expected[5:]


def test_select_next_map_prefers_staged_generated():
    calls = []
    next_map, staged = _select_next_map(
        "q2dm2", "mlteacher_12345678", lambda: calls.append(1) or "q2dm4")
    assert (next_map, staged) == ("mlteacher_12345678", None)
    assert calls == []  # stock rotation untouched while a generated map stages


def test_select_next_map_stock_after_generated():
    next_map, staged = _select_next_map(
        "mlteacher_12345678", "mlteacher_87654321", lambda: "q2dm6")
    assert (next_map, staged) == ("q2dm6", "mlteacher_87654321")


def test_select_next_map_stock_fallback_during_farm_outage():
    next_map, staged = _select_next_map("q2dm2", None, lambda: "q2dm8")
    assert (next_map, staged) == ("q2dm8", None)


def test_watch_stdout_flags_round_end_markers(capsys):
    stream = io.BytesIO(b"Korn was blasted by Sodom.\nTimelimit hit.\n")
    hit = threading.Event()
    _watch_stdout(stream, hit)
    assert hit.is_set()
    assert "Timelimit hit." in capsys.readouterr().out


def test_watch_stdout_ignores_normal_lines():
    stream = io.BytesIO(b"Evil Zeep entered the game\n")
    hit = threading.Event()
    _watch_stdout(stream, hit)
    assert not hit.is_set()
