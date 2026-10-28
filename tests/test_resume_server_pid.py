"""Guard rails for the server's one-command PID restart launcher."""

import json

import pytest

from scripts import resume_server_pid as launcher

run_state = launcher.run_state
session_budget = launcher.session_budget


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_crash_uses_newer_checkpoint_than_stale_result(tmp_path):
    config = {"training": {"max_steps": 500}}
    write_json(tmp_path / "run_config.json", config)
    write_json(tmp_path / "run_result.json", {"status": "paused", "global_step": 60})
    write_json(tmp_path / "progress.json", {"status": "running", "global_step": 89})
    write_json(tmp_path / "restart_checkpoints/latest.json", {"global_step": 89})
    assert run_state(tmp_path, config) == ("resume", 89)


@pytest.mark.parametrize("status", ["complete", "stopped_hard"])
def test_terminal_status_blocks_resume(tmp_path, status):
    write_json(tmp_path / "run_config.json", {"seed": 42})
    write_json(tmp_path / "run_result.json", {"status": status, "global_step": 100})
    assert run_state(tmp_path, {"seed": 42}) == (status, 100)


def test_mismatched_saved_config_is_rejected(tmp_path):
    write_json(tmp_path / "run_config.json", {"seed": 42})
    with pytest.raises(RuntimeError, match="differs"):
        run_state(tmp_path, {"seed": 43})


def test_budget_reserves_one_hour_before_platform_restart():
    assert session_budget(0) == 10800
    assert session_budget(3600) == 7200
    assert session_budget(10500) == 300


def test_endpoint_probe_uses_mirror_if_official_data_fails(monkeypatch):
    checked = []

    def probe(url, *, byte_range):
        checked.append((url, byte_range))
        return url.startswith("https://hf-mirror.com")

    monkeypatch.setattr(launcher, "probe_url", probe)
    assert launcher.probe_endpoint() == "https://hf-mirror.com"
    assert len(checked) == 3
    assert checked[1][1] is True
    assert checked[2][1] is False
