from __future__ import annotations

from pathlib import Path

import pytest

from openpilot.sunnypilot.speed_vision import publisher
from openpilot.sunnypilot.speed_vision._vendor.speed_vision_core import types

REPO = Path(__file__).resolve().parents[5]


def _frame() -> types.FrameRef:
  return types.FrameRef(
    session_id="session-a",
    stream=types.StreamId.NARROW_ROAD,
    frame_id=42,
    capture_mono_ns=1_000_000_000,
    capture_reference=types.CaptureReference.SOF,
    native_width=1928,
    native_height=1208,
  )


def _batch(*, value: int | None = 50, detections: int = 1) -> types.ObservationBatch:
  frame = _frame()
  rows = []
  for i in range(detections):
    state = types.ValueState.VALUE if value is not None else types.ValueState.UNREADABLE
    rows.append(types.Detection(
      frame=frame,
      bbox_xyxy=(10.0 + i, 20.0, 30.0 + i, 40.0),
      sign_family=types.SignFamily.MAX_SPEED,
      value_state=state,
      value_kph=value,
      detection_score=0.95,
      classification_score=0.90,
      supported_domain=True,
      observation_id=f"obs-{i}",
    ))
  return types.ObservationBatch(
    frame=frame,
    model_hash="model",
    config_hash="config",
    rulepack_hash="rules",
    processed_mono_ns=1_050_000_000,
    backend="reference",
    detections=tuple(rows),
  )


def _health() -> types.PerceptionHealth:
  return types.PerceptionHealth(
    session_id="session-a",
    backend="reference",
    backend_available=True,
    last_processed_capture_mono_ns=1_000_000_000,
    last_successful_completion_mono_ns=1_050_000_000,
  )


def test_schema_uses_pinned_reserved_binding_and_nonlogged_service() -> None:
  log_schema = (REPO / "openpilot/cereal/log.capnp").read_text(encoding="utf-8")
  custom_schema = (REPO / "openpilot/cereal/custom.capnp").read_text(encoding="utf-8")
  services = (REPO / "openpilot/cereal/services.py").read_text(encoding="utf-8")
  assert "speedVisionState @136 :Custom.SpeedVisionState;" in log_schema
  assert "struct SpeedVisionState @0xcb9fd56c7057593a" in custom_schema
  assert '"speedVisionState": (False, 10., None)' in services


def test_absent_observation_value_stays_absent_on_wire() -> None:
  payload = publisher.build_speed_vision_state(
    _batch(value=None),
    _health(),
    types.LimitHypothesis(False, None, types.HypothesisState.OBSERVED),
    producer_epoch=7,
    sequence=9,
  )
  observation = payload["observations"][0]
  assert observation["hasValue"] is False
  assert observation["valueKph"] == 0
  assert payload["hypothesis"]["hasValue"] is False
  assert payload["hypothesis"]["valueKph"] == 0


def test_wire_payload_has_no_control_request_fields() -> None:
  payload = publisher.build_speed_vision_state(
    _batch(),
    _health(),
    types.LimitHypothesis(True, 50, types.HypothesisState.CURRENT, usable_for_advisory=True),
    producer_epoch=1,
    sequence=1,
  )
  forbidden = {"targetSpeed", "targetAcceleration", "actuatorRequest", "cruiseSetSpeed"}
  stack = [payload]
  keys = set()
  while stack:
    value = stack.pop()
    if isinstance(value, dict):
      keys.update(value)
      stack.extend(value.values())
    elif isinstance(value, list):
      stack.extend(value)
  assert not (keys & forbidden)


def test_more_than_eight_observations_is_rejected() -> None:
  with pytest.raises(publisher.SpeedVisionWireError, match="eight"):
    publisher.build_speed_vision_state(
      _batch(detections=9),
      _health(),
      types.LimitHypothesis(False, None, types.HypothesisState.NONE),
      producer_epoch=1,
      sequence=1,
    )
