"""Guard rails for the server's one-command PID restart launcher."""

import json

import pytest
import yaml

from scripts import resume_server_pid as launcher
from src.evaluation.online_data import materialize_roles
from src.evaluation.selection_cache import SelectedSourceCache

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
    assert len(checked) == 1 + len(launcher.SOURCE_PROBES) + 1
    assert checked[1][1] is True
    assert checked[2][1] is False


def test_endpoint_probe_checks_mmlu_and_honors_override(monkeypatch):
    checked = []

    def probe(url, *, byte_range):
        checked.append(url)
        return True

    monkeypatch.setattr(launcher, "probe_url", probe)
    assert launcher.probe_endpoint("https://huggingface.co") == "https://huggingface.co"
    assert len(checked) == 1 + len(launcher.SOURCE_PROBES)
    assert any("/cais/mmlu/" in url for url in checked)
    assert all(url.startswith("https://huggingface.co") for url in checked)


def test_complete_frozen_selection_skips_future_dataset_probes(tmp_path, monkeypatch):
    suite = {"version": 1, "seed": 42, "protocol": "pid-grpo-zero-shot-v1",
             "aggregation": "sample_weighted", "sources": {
                 "mmlu": {"path": "test/mmlu", "name": "all", "revision": "a" * 40,
                          "split": "test", "kind": "mmlu", "selection": "reservoir",
                          "allocations": {"control": 2, "forgetting": 2}}}}
    raw = [{"question": f"q{i}", "subject": "science", "choices": ["A", "B"],
            "answer": i % 2} for i in range(8)]
    _, manifest = materialize_roles(suite, ["control"], row_loader=lambda _: raw)
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(suite))
    write_json(tmp_path / "manifest.json", manifest)
    config = {"evaluation": {"suite_config": "suite.yaml", "manifest_file": "manifest.json"},
              "storage": {"cache_dir": str(tmp_path / "cache")},
              "data": {"backend": "online_suite"}}
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    assert not launcher.training_cache_complete(config)
    cache = SelectedSourceCache(tmp_path / "cache/online_selections", expected_manifest=manifest)
    materialize_roles(suite, ["control"], row_loader=lambda _: raw, source_cache=cache)
    assert launcher.training_cache_complete(config)


def test_endpoint_probe_only_checks_uncached_sources(monkeypatch):
    checked = []
    monkeypatch.setattr(launcher, "probe_url", lambda url, **kwargs: checked.append(url) or True)
    launcher.probe_endpoint("https://huggingface.co", required_tasks=["mmlu"])
    assert len(checked) == 2
    assert "/cais/mmlu/" in checked[0]
