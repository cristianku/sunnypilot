from __future__ import annotations

from enum import Enum
from typing import Any

MAX_WIRE_OBSERVATIONS = 8


class SpeedVisionWireError(ValueError):
  """Portable-to-Cap'n-Proto wire conversion failed."""


_STREAM = {
  "narrow_road": "narrowRoad",
  "wide_road": "wideRoad",
  "cabin": "cabin",
  "synthetic": "synthetic",
}
_CAPTURE = {"sof": "sof", "eof": "eof", "unknown": "unknown"}
_SIGN_FAMILY = {
  "max_speed": "maxSpeed",
  "cancellation": "cancellation",
  "zone": "zone",
  "variable_display": "variableDisplay",
  "other_sign": "otherSign",
  "not_a_sign": "notASign",
  "unreadable": "unreadable",
}
_VALUE_STATE = {
  "value": "value",
  "unknown": "unknown",
  "unreadable": "unreadable",
  "not_applicable": "notApplicable",
  "unavailable": "unavailable",
}
_HYPOTHESIS = {
  "none": "none",
  "observed": "observed",
  "ahead": "ahead",
  "current": "current",
  "uncertain": "uncertain",
  "unavailable": "unavailable",
}


def _enum_value(value: Any) -> str:
  if isinstance(value, Enum):
    return str(value.value)
  return str(value)


def _wire_enum(value: Any, table: dict[str, str], field: str) -> str:
  key = _enum_value(value)
  try:
    return table[key]
  except KeyError as exc:
    raise SpeedVisionWireError(f"unsupported {field}: {key!r}") from exc


def _optional_u64(value: int | None) -> tuple[bool, int]:
  if value is None:
    return False, 0
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise SpeedVisionWireError(f"optional timestamp must be a non-negative int or None, got {value!r}")
  return True, value


def _wire_value(has_value: bool, value: int | None) -> tuple[bool, int]:
  if has_value:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > 65535:
      raise SpeedVisionWireError(f"present speed value must be a positive uint16, got {value!r}")
    return True, value
  if value is not None:
    raise SpeedVisionWireError("absent speed value must be None")
  return False, 0


def build_speed_vision_state(
  batch: Any,
  health: Any,
  hypothesis: Any,
  *,
  producer_epoch: int,
  sequence: int,
) -> dict[str, Any]:
  """Build a strict JSON-like representation matching SpeedVisionState."""
  if isinstance(producer_epoch, bool) or not isinstance(producer_epoch, int) or producer_epoch < 0:
    raise SpeedVisionWireError("producer_epoch must be a non-negative integer")
  if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
    raise SpeedVisionWireError("sequence must be a non-negative integer")

  detections = tuple(batch.detections)
  if len(detections) > MAX_WIRE_OBSERVATIONS:
    raise SpeedVisionWireError("wire contract permits at most eight observations")

  frame = batch.frame
  has_last_processed, last_processed = _optional_u64(health.last_processed_capture_mono_ns)
  has_last_success, last_success = _optional_u64(health.last_successful_completion_mono_ns)
  has_last_context, last_context = _optional_u64(hypothesis.last_context_mono_ns)
  has_activation_start, activation_start = _optional_u64(hypothesis.activation_start_mono_ns)
  has_activation_end, activation_end = _optional_u64(hypothesis.activation_end_mono_ns)
  hyp_has_value, hyp_value = _wire_value(bool(hypothesis.has_value), hypothesis.value_kph)

  observations: list[dict[str, Any]] = []
  for detection in detections:
    has_value = _enum_value(detection.value_state) == "value"
    obs_has_value, obs_value = _wire_value(has_value, detection.value_kph)
    x1, y1, x2, y2 = detection.bbox_xyxy
    observations.append({
      "observationId": str(detection.observation_id),
      "frameId": int(detection.frame.frame_id),
      "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
      "signFamily": _wire_enum(detection.sign_family, _SIGN_FAMILY, "sign family"),
      "valueState": _wire_enum(detection.value_state, _VALUE_STATE, "value state"),
      "hasValue": obs_has_value,
      "valueKph": obs_value,
      "detectionScore": float(detection.detection_score),
      "classificationScore": float(detection.classification_score),
      "supportedDomain": bool(detection.supported_domain),
    })

  last_observation = frame.capture_mono_ns if observations else None
  has_last_observation, last_observation_ns = _optional_u64(last_observation)

  return {
    "schemaVersion": int(batch.schema_version),
    "producerEpoch": producer_epoch,
    "sequence": sequence,
    "modelHash": str(batch.model_hash),
    "configHash": str(batch.config_hash),
    "rulepackHash": str(batch.rulepack_hash),
    "processedMonoNs": int(batch.processed_mono_ns),
    "frame": {
      "sessionId": str(frame.session_id),
      "stream": _wire_enum(frame.stream, _STREAM, "stream"),
      "frameId": int(frame.frame_id),
      "timestampSof": int(frame.capture_mono_ns) if _enum_value(frame.capture_reference) == "sof" else 0,
      "timestampEof": int(frame.capture_mono_ns) if _enum_value(frame.capture_reference) == "eof" else 0,
      "nativeWidth": int(frame.native_width),
      "nativeHeight": int(frame.native_height),
      "preprocessingIdentity": str(frame.preprocessing_identity),
      "captureReference": _wire_enum(frame.capture_reference, _CAPTURE, "capture reference"),
    },
    "health": {
      "backend": str(health.backend),
      "available": bool(health.backend_available),
      "status": str(batch.backend_status),
      "hasLastProcessedCapture": has_last_processed,
      "lastProcessedCaptureMonoNs": last_processed,
      "hasLastSuccessfulCompletion": has_last_success,
      "lastSuccessfulCompletionMonoNs": last_success,
      "overflow": bool(health.overflow),
      "faultCode": "" if health.fault_code is None else str(health.fault_code),
      "droppedFrames": int(health.dropped_frames),
    },
    "observations": observations,
    "hypothesis": {
      "state": _wire_enum(hypothesis.state, _HYPOTHESIS, "hypothesis state"),
      "hasValue": hyp_has_value,
      "valueKph": hyp_value,
      "usableForAdvisory": bool(hypothesis.usable_for_advisory),
      "trackId": "" if hypothesis.track_id is None else str(hypothesis.track_id),
      "eventId": "" if hypothesis.event_id is None else str(hypothesis.event_id),
      "hasActivationStart": has_activation_start,
      "activationStartMonoNs": activation_start,
      "hasActivationEnd": has_activation_end,
      "activationEndMonoNs": activation_end,
      "hasLastContext": has_last_context,
      "lastContextMonoNs": last_context,
      "unavailableReason": "" if hypothesis.unavailable_reason is None else str(hypothesis.unavailable_reason),
    },
    "hasLastObservationMonoNs": has_last_observation,
    "lastObservationMonoNs": last_observation_ns,
  }


def fill_cereal_state(builder: Any, payload: dict[str, Any]) -> None:
  """Assign a validated payload to a generated SpeedVisionState builder."""
  for key in ("schemaVersion", "producerEpoch", "sequence", "modelHash", "configHash",
              "rulepackHash", "processedMonoNs", "hasLastObservationMonoNs", "lastObservationMonoNs"):
    setattr(builder, key, payload[key])

  for key, value in payload["frame"].items():
    setattr(builder.frame, key, value)
  for key, value in payload["health"].items():
    setattr(builder.health, key, value)
  for key, value in payload["hypothesis"].items():
    setattr(builder.hypothesis, key, value)

  rows = builder.init("observations", len(payload["observations"]))
  for row, source in zip(rows, payload["observations"], strict=True):
    for key, value in source.items():
      if key == "bbox":
        for bbox_key, bbox_value in value.items():
          setattr(row.bbox, bbox_key, bbox_value)
      else:
        setattr(row, key, value)
