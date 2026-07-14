import argparse
import stat
from pathlib import Path

import pytest

from tools.network_public_server import (
    TelemetrySettings,
    _build_command,
    _telemetry_from_env,
    _write_config,
)


SECRET = "A" * 48


def _args(**updates):
    values = {
        "port": 28000,
        "maxclients": 6,
        "timelimit": 15.0,
        "fraglimit": 20,
        "dlserver": "http://5.78.204.86:32494",
    }
    values.update(updates)
    return argparse.Namespace(**values)


def test_telemetry_contract_is_env_only_and_repr_redacts_secret():
    settings = _telemetry_from_env({
        "Q2_ML_CLIENT_TELEMETRY": "1",
        "Q2_ML_CLIENT_TELEMETRY_PORT": "28049",
        "Q2_ML_CLIENT_TELEMETRY_TOKEN": SECRET,
    })

    assert settings.port == 28049
    assert settings.token == SECRET
    assert SECRET not in repr(settings)


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"Q2_ML_CLIENT_TELEMETRY": "0"},
        {
            "Q2_ML_CLIENT_TELEMETRY": "1",
            "Q2_ML_CLIENT_TELEMETRY_PORT": "28049",
            "Q2_ML_CLIENT_TELEMETRY_TOKEN": "unsafe;set rcon_password stolen",
        },
    ],
)
def test_telemetry_contract_fails_closed_without_disclosing_value(environment):
    with pytest.raises(ValueError) as failure:
        _telemetry_from_env(environment)
    assert environment.get("Q2_ML_CLIENT_TELEMETRY_TOKEN", "absent") not in str(
        failure.value
    )


def test_secret_config_is_0600_and_enables_conduit_before_map(tmp_path):
    runtime = tmp_path / "runtime"
    settings = TelemetrySettings(port=28049, token=SECRET)

    config = _write_config(runtime, _args(), "q2dm1", settings)
    text = config.read_text()
    mode = stat.S_IMODE(config.stat().st_mode)

    assert mode == 0o600
    assert "set autospawn 0" in text
    assert 'set botlist ""' in text
    assert "set maxclients 6" in text
    assert "set ml_enabled 0" in text
    assert "set ml_client_telemetry 1" in text
    assert "set ml_client_telemetry_port 28049" in text
    assert text.count(SECRET) == 1
    assert text.index("set ml_client_telemetry_token") < text.index("map q2dm1")
    assert 'set sv_downloadserver "http://5.78.204.86:32494"' in text
    assert not list(config.parent.glob("*.tmp"))


def test_q2ded_command_never_contains_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("Q2_BIND_IP", "")
    args = _args()
    runtime = tmp_path / "runtime"
    q2ded = runtime / "q2ded"
    config = runtime / "lithium" / "ml_network_public_28000.cfg"

    command = _build_command(q2ded, runtime, args, config)

    assert SECRET not in " ".join(command)
    assert command[-2:] == ["+exec", config.name]
    assert command[command.index("port") + 1] == "28000"


def test_service_uses_protected_environment_file_and_network_launcher():
    service = (
        Path(__file__).resolve().parents[1]
        / "ops"
        / "q2mlbot-network-server.service"
    ).read_text()

    assert "EnvironmentFile=/etc/q2mlbot/network-client-harness.env" in service
    assert "tools/network_public_server.py" in service
    assert "--port 28000" in service
    assert "--maxclients 6" in service
    assert "Q2_ML_CLIENT_TELEMETRY_TOKEN" not in service
    assert "<secret>" not in service


def test_live_trainer_service_is_four_client_warm_start_and_has_no_secret():
    ops = Path(__file__).resolve().parents[1] / "ops"
    service = (ops / "q2-network-live-trainer.service").read_text()
    environment = (ops / "q2-network-live-trainer.env.example").read_text()

    assert "EnvironmentFile=%h/.config/q2-network-live-trainer.env" in service
    assert "test -n" in service
    assert "Q2_NETWORK_CLIENTS=4" in environment
    assert "Q2_RESUME_DIR=/home/raymond/q2-ml-bot/checkpoints/movement_reset_v2" in environment
    assert "Q2_RUNS_DIR=/home/raymond/q2-ml-bot/runs" in environment
    assert "python3 -u -m train.ppo" in environment
    assert "--timescale 1" in environment
    assert "--resume" in environment
    assert "/usr/bin/false" not in environment
    assert "Q2_ML_CLIENT_TELEMETRY_TOKEN=\n" in environment
