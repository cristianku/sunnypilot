from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._vendor.speed_vision_core import types


class FrameAdapterError(ValueError):
  """Camera/context data cannot be represented without guessing."""


@dataclass(frozen=True)
class OwnedFrame:
  frame: types.FrameRef
  pixels: bytes
  timestamp_sof: int
  timestamp_eof: int
  stride: int
  pixel_format: str
  buffer_len: int


def _strict_nonnegative_int(value: int, field: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise FrameAdapterError(f"{field} must be a non-negative integer")
  return value


def copy_vipc_frame(
  buffer: Any,
  *,
  session_id: str,
  stream: types.StreamId,
  frame_id: int,
  timestamp_sof: int,
  timestamp_eof: int,
  width: int,
  height: int,
  stride: int,
  pixel_format: str = "nv12",
  preprocessing_identity: str = "none",
) -> OwnedFrame:
  """Copy a VisionIPC frame into bounded owned memory.

  Runtime code must not keep an unowned VisionIPC buffer after the producer can
  recycle it. This adapter copies first and only then constructs portable
  metadata. It never fabricates timing from frame rate.
  """
  if not session_id:
    raise FrameAdapterError("session_id must be non-empty")
  frame_id = _strict_nonnegative_int(frame_id, "frame_id")
  timestamp_sof = _strict_nonnegative_int(timestamp_sof, "timestamp_sof")
  timestamp_eof = _strict_nonnegative_int(timestamp_eof, "timestamp_eof")
  width = _strict_nonnegative_int(width, "width")
  height = _strict_nonnegative_int(height, "height")
  stride = _strict_nonnegative_int(stride, "stride")
  if width <= 0 or height <= 0 or stride < width:
    raise FrameAdapterError(f"invalid frame geometry width={width} height={height} stride={stride}")
  if pixel_format.lower() != "nv12":
    raise FrameAdapterError(f"unsupported pixel format: {pixel_format}")

  try:
    pixels = memoryview(buffer).tobytes()
  except TypeError as exc:
    try:
      pixels = bytes(buffer)
    except Exception as fallback_exc:
      raise FrameAdapterError("VisionIPC buffer does not expose copyable bytes") from fallback_exc

  # NV12: full-height Y plane plus half-height interleaved UV plane, with the
  # same row stride. Padding is retained so later preprocessing can honor it.
  expected_min = stride * height + stride * ((height + 1) // 2)
  if len(pixels) < expected_min:
    raise FrameAdapterError(
      f"short NV12 buffer: got {len(pixels)} bytes, need at least {expected_min}"
    )

  if timestamp_sof > 0:
    capture = timestamp_sof
    reference = types.CaptureReference.SOF
  elif timestamp_eof > 0:
    capture = timestamp_eof
    reference = types.CaptureReference.EOF
  else:
    raise FrameAdapterError("frame has neither SOF nor EOF capture timestamp")

  frame = types.FrameRef(
    session_id=session_id,
    stream=stream,
    frame_id=frame_id,
    capture_mono_ns=capture,
    capture_reference=reference,
    native_width=width,
    native_height=height,
    preprocessing_identity=preprocessing_identity,
  )
  return OwnedFrame(
    frame=frame,
    pixels=pixels,
    timestamp_sof=timestamp_sof,
    timestamp_eof=timestamp_eof,
    stride=stride,
    pixel_format="nv12",
    buffer_len=len(pixels),
  )


def build_road_context(
  owned_frame: OwnedFrame,
  *,
  context_mono_ns: int,
  max_context_age_ns: int,
  calibration_available: bool,
  ego_motion_available: bool,
  road_continuity_id: str | None,
  at_junction: bool,
  country_ambiguous: bool,
  provenance: str,
) -> types.RoadContext:
  """Bind read-only context to a frame only when timing is explicitly close."""
  context_mono_ns = _strict_nonnegative_int(context_mono_ns, "context_mono_ns")
  max_context_age_ns = _strict_nonnegative_int(max_context_age_ns, "max_context_age_ns")
  delta = abs(context_mono_ns - owned_frame.frame.capture_mono_ns)
  if delta > max_context_age_ns:
    return types.RoadContext(
      session_id=owned_frame.frame.session_id,
      context_mono_ns=context_mono_ns,
      calibration_available=False,
      ego_motion_available=False,
      road_continuity_id=None,
      at_junction=False,
      country_ambiguous=True,
      provenance="unresolved_stale_context",
    )

  return types.RoadContext(
    session_id=owned_frame.frame.session_id,
    context_mono_ns=context_mono_ns,
    calibration_available=bool(calibration_available),
    ego_motion_available=bool(ego_motion_available),
    road_continuity_id=road_continuity_id,
    at_junction=bool(at_junction),
    country_ambiguous=bool(country_ambiguous),
    provenance=provenance,
  )
