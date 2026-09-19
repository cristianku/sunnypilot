from __future__ import annotations

import json
from pathlib import Path

from openpilot.sunnypilot.speed_vision import runner


def test_missing_model_produces_unavailable_runner(tmp_path: Path) -> None:
  lock = tmp_path / "artifacts.lock.json"
  lock.write_text(json.dumps({
    "schema_version": 1,
    "feature_id": "nnslr-speed-vision",
    "state": "bootstrap",
    "core": {"tree_digest": "x"},
    "model": {"state": "unavailable", "bundle_digest": None},
    "rulepack": {"state": "unavailable", "digest": None},
    "capability_profile": "core-contract-only",
  }), encoding="utf-8")
  r = runner.load_runner(lock)
  assert isinstance(r, runner.UnavailableRunner)
  assert r.health().available is False
  assert r.health().reason == "model_unavailable"


def test_no_mock_backend_is_selectable_onroad(tmp_path: Path) -> None:
  lock = tmp_path / "artifacts.lock.json"
  lock.write_text(json.dumps({
    "schema_version": 1,
    "feature_id": "nnslr-speed-vision",
    "state": "bootstrap",
    "core": {"tree_digest": "x"},
    "model": {"state": "ready", "bundle_digest": "abc", "backend_profiles": []},
    "rulepack": {"state": "unavailable", "digest": None},
    "capability_profile": "observation",
  }), encoding="utf-8")
  r = runner.load_runner(lock, requested_backend="mock")
  assert isinstance(r, runner.UnavailableRunner)
  assert r.health().reason == "unsupported_backend"


def test_unavailable_runner_never_accepts_frames(tmp_path: Path) -> None:
  r = runner.UnavailableRunner("not_ready")
  assert r.submit_latest(object()) is False
  assert r.poll() is None
  assert r.health().available is False
