from __future__ import annotations

import pytest

from openpilot.sunnypilot.speed_vision import adapters
from openpilot.sunnypilot.speed_vision._vendor.speed_vision_core import types


def test_reused_vipc_buffer_does_not_change_owned_snapshot() -> None:
  raw = bytearray(range(24))
  owned = adapters.copy_vipc_frame(
    raw,
    session_id="s",
    stream=types.StreamId.NARROW_ROAD,
    frame_id=7,
    timestamp_sof=1_000,
    timestamp_eof=1_100,
    width=4,
    height=4,
    stride=4,
  )
  before = owned.pixels
  raw[:] = b"\xff" * len(raw)
  assert owned.pixels == before
  assert owned.frame.capture_mono_ns == 1_000
  assert owned.frame.capture_reference == types.CaptureReference.SOF
  assert owned.timestamp_eof == 1_100


def test_nv12_stride_padding_is_preserved() -> None:
  # width=4, stride=8, height=4 -> 8*4 Y + 8*2 UV = 48 bytes
  raw = bytes(range(48))
  owned = adapters.copy_vipc_frame(
    raw, session_id="s", stream=types.StreamId.NARROW_ROAD,
    frame_id=1, timestamp_sof=10, timestamp_eof=20,
    width=4, height=4, stride=8,
  )
  assert owned.stride == 8
  assert len(owned.pixels) == 48


def test_short_nv12_buffer_is_rejected() -> None:
  with pytest.raises(adapters.FrameAdapterError, match="short"):
    adapters.copy_vipc_frame(
      b"x" * 20, session_id="s", stream=types.StreamId.NARROW_ROAD,
      frame_id=1, timestamp_sof=10, timestamp_eof=20,
      width=4, height=4, stride=8,
    )


def test_stale_context_becomes_unknown_instead_of_reusing_latest() -> None:
  frame = adapters.copy_vipc_frame(
    bytes(range(24)), session_id="s", stream=types.StreamId.NARROW_ROAD,
    frame_id=1, timestamp_sof=1_000_000_000, timestamp_eof=1_000_010_000,
    width=4, height=4, stride=4,
  )
  context = adapters.build_road_context(
    frame,
    context_mono_ns=1_500_000_000,
    max_context_age_ns=100_000_000,
    calibration_available=True,
    ego_motion_available=True,
    road_continuity_id="road-a",
    at_junction=True,
    country_ambiguous=False,
    provenance="test",
  )
  assert context.calibration_available is False
  assert context.ego_motion_available is False
  assert context.road_continuity_id is None
  assert context.provenance == "unresolved_stale_context"


def test_fresh_context_preserves_read_only_context() -> None:
  frame = adapters.copy_vipc_frame(
    bytes(range(24)), session_id="s", stream=types.StreamId.NARROW_ROAD,
    frame_id=1, timestamp_sof=1_000_000_000, timestamp_eof=1_000_010_000,
    width=4, height=4, stride=4,
  )
  context = adapters.build_road_context(
    frame,
    context_mono_ns=1_020_000_000,
    max_context_age_ns=100_000_000,
    calibration_available=True,
    ego_motion_available=True,
    road_continuity_id="road-a",
    at_junction=False,
    country_ambiguous=False,
    provenance="local-messages",
  )
  assert context.session_id == "s"
  assert context.road_continuity_id == "road-a"
  assert context.provenance == "local-messages"
