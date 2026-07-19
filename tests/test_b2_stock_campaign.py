"""Atomic exact-membership tests for the one-shot B2 stock campaign."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_b2_stock_campaign as stock_tool
from tools.run_b2_stock_campaign import (
    B2StockCampaignError,
    STOCK_IDS,
    run_stock_campaign,
)
from tools.run_generator_cohort import STAGE_SUFFIXES, canonical_bytes


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: bytes = b"fixture\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    client = tmp_path / "client"
    lithium = tmp_path / "lithium"
    return {
        "repo_root": ROOT,
        "python": _write(tmp_path / "python", b"#!/bin/sh\n"),
        "stock_pak": _write(tmp_path / "pak0.pak"),
        "provenance": _write(tmp_path / "provenance.json"),
        "stock_inventory": _write(tmp_path / "stock-inventory.json"),
        "b1_gate": _write(tmp_path / "B1-GATE.json"),
        "client_root": client,
        "lithium_root": lithium,
        "hook_attestation": _write(tmp_path / "hook-attestation.json"),
        "fall_oracle": _write(tmp_path / "fall-oracle"),
        "packer": _write(tmp_path / "packer"),
        "verifier": _write(tmp_path / "verifier"),
        "output_root": tmp_path / "workspace/stock",
        "report_path": tmp_path / "workspace/reports/stock-campaign.json",
        "cm_oracle": _write(client / "release/q2-cm-oracle"),
        "pmove_oracle": _write(client / "release/q2-pmove-oracle"),
        "hook_oracle": _write(lithium / "tools/q2-hook-oracle"),
    }


def _install_successful_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository_bindings: list[dict[str, object]] | None = None,
) -> None:
    stable = {
        "repository_commit": "1" * 40,
        "repository_tree": "2" * 40,
        "git_clean": True,
    }
    bindings = iter(repository_bindings or [stable, stable])
    monkeypatch.setattr(stock_tool, "repository_binding", lambda _repo: next(bindings))

    def builder(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        for map_id in STOCK_IDS:
            for suffix in STAGE_SUFFIXES["analysis"]:
                _write(output / f"{map_id}{suffix}", f"{map_id}{suffix}\n".encode())
        summary = {
            "schema": "q2-atlas-stock-build-v1",
            "maps": [{"canonical_map_id": map_id} for map_id in STOCK_IDS],
        }
        return SimpleNamespace(
            returncode=0,
            stdout=canonical_bytes(summary),
            stderr=b"",
        )

    def extract(_pak: Path, output: Path):
        for map_id in STOCK_IDS:
            _write(output / f"{map_id}.bsp", f"bsp:{map_id}\n".encode())
        return {map_id: output / f"{map_id}.bsp" for map_id in STOCK_IDS}

    def validate(
        bsp: Path,
        analysis: Path,
        *,
        b1_gate_path: Path,
        stock_provenance_path: Path,
        stock_inventory_path: Path,
    ):
        assert b1_gate_path.is_file()
        assert stock_provenance_path.is_file()
        assert stock_inventory_path.is_file()
        assert analysis == analysis.parent / f"{bsp.stem}.analysis.manifest.json"
        assert analysis.is_file()
        return {
            "schema": "fixture-stock-validation",
            "map": bsp.stem,
            "passed": True,
        }

    monkeypatch.setattr(stock_tool.subprocess, "run", builder)
    monkeypatch.setattr(stock_tool, "extract_stock", extract)
    monkeypatch.setattr(stock_tool, "validate_stock_analysis", validate)


def test_stock_campaign_publishes_one_exact_root_and_canonical_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    paths["output_root"].parent.mkdir(parents=True)
    paths["report_path"].parent.mkdir(parents=True)
    _install_successful_fakes(monkeypatch)

    report = run_stock_campaign(**{
        name: value
        for name, value in paths.items()
        if name not in {"cm_oracle", "pmove_oracle", "hook_oracle"}
    })

    assert report["passed"] is True
    assert report["map_count"] == 8
    assert [row["map"] for row in report["maps"]] == list(STOCK_IDS)
    assert paths["report_path"].read_bytes() == canonical_bytes(report)
    output = paths["output_root"]
    assert {path.name for path in output.iterdir()} == {
        "bsp",
        "analysis",
        "validation",
    }
    assert {path.name for path in (output / "bsp").iterdir()} == {
        f"{map_id}.bsp" for map_id in STOCK_IDS
    }
    assert {path.name for path in (output / "validation").iterdir()} == {
        f"{map_id}.stock-validation.json" for map_id in STOCK_IDS
    }
    assert not list(output.parent.glob(".b2-stock-campaign-*"))


def test_stock_builder_failure_publishes_no_partial_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    paths["output_root"].parent.mkdir(parents=True)
    paths["report_path"].parent.mkdir(parents=True)
    monkeypatch.setattr(
        stock_tool,
        "repository_binding",
        lambda _repo: {"git_clean": True},
    )
    monkeypatch.setattr(
        stock_tool.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=7, stdout=b"", stderr=b"failed"
        ),
    )

    with pytest.raises(B2StockCampaignError, match="failed with exit 7"):
        run_stock_campaign(**{
            name: value
            for name, value in paths.items()
            if name not in {"cm_oracle", "pmove_oracle", "hook_oracle"}
        })

    assert not paths["output_root"].exists()
    assert paths["report_path"].is_file()
    assert paths["report_path"].read_bytes() == b""
    assert not list(paths["output_root"].parent.glob(".b2-stock-campaign-*"))


def test_repository_drift_refuses_stock_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    paths["output_root"].parent.mkdir(parents=True)
    paths["report_path"].parent.mkdir(parents=True)
    _install_successful_fakes(
        monkeypatch,
        repository_bindings=[
            {"repository_commit": "1" * 40, "git_clean": True},
            {"repository_commit": "2" * 40, "git_clean": True},
        ],
    )

    with pytest.raises(B2StockCampaignError, match="repository changed"):
        run_stock_campaign(**{
            name: value
            for name, value in paths.items()
            if name not in {"cm_oracle", "pmove_oracle", "hook_oracle"}
        })

    assert not paths["output_root"].exists()
    assert paths["report_path"].read_bytes() == b""


def test_stock_campaign_refuses_existing_output_or_report(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["output_root"].mkdir(parents=True)
    paths["report_path"].parent.mkdir(parents=True)
    paths["report_path"].write_bytes(b"occupied\n")

    with pytest.raises(B2StockCampaignError, match="output root already exists"):
        run_stock_campaign(**{
            name: value
            for name, value in paths.items()
            if name not in {"cm_oracle", "pmove_oracle", "hook_oracle"}
        })

    paths["output_root"].rmdir()
    with pytest.raises(B2StockCampaignError, match="report already exists"):
        run_stock_campaign(**{
            name: value
            for name, value in paths.items()
            if name not in {"cm_oracle", "pmove_oracle", "hook_oracle"}
        })
