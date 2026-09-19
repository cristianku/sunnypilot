from __future__ import annotations

from pathlib import Path

from openpilot.sunnypilot.speed_vision import publisher

REPO = Path(__file__).resolve().parents[4]


def test_unavailable_state_carries_no_speed_value() -> None:
  payload = publisher.build_unavailable_state(
    session_id="session",
    producer_epoch=3,
    sequence=4,
    backend="unavailable",
    reason="model_unavailable",
    processed_mono_ns=100,
  )
  assert payload["observations"] == []
  assert payload["hypothesis"]["state"] == "unavailable"
  assert payload["hypothesis"]["hasValue"] is False
  assert payload["hypothesis"]["valueKph"] == 0
  assert payload["hypothesis"]["usableForAdvisory"] is False
  assert payload["health"]["available"] is False


def test_process_is_default_off_and_separate_from_sla() -> None:
  params = (REPO / "openpilot/common/params_keys.h").read_text(encoding="utf-8")
  process_config = (REPO / "openpilot/system/manager/process_config.py").read_text(encoding="utf-8")
  assert '{"VisionSpeedLimitMode", {PERSISTENT, INT, "0"}}' in params
  assert '{"VisionSpeedLimitWarnings", {PERSISTENT, BOOL, "0"}}' in params
  assert '{"VisionSpeedLimitDiagnostics", {PERSISTENT, BOOL, "0"}}' in params
  assert 'PythonProcess("speedvisiond", "openpilot.sunnypilot.speed_vision.speedvisiond", speed_vision)' in process_config
  assert 'def speed_vision(' in process_config
