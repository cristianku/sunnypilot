from __future__ import annotations

from openpilot.sunnypilot.speed_vision.lifecycle import LatestFrameSlot, VisionSpeedLimitMode


def test_latest_frame_slot_replaces_pending_frame() -> None:
  slot = LatestFrameSlot()
  assert slot.submit("frame-1") is True
  assert slot.take() == "frame-1"
  slot.mark_in_flight(True)
  assert slot.submit("frame-2") is True
  assert slot.submit("frame-3") is True
  assert slot.pending == "frame-3"
  assert slot.replaced_frames == 1
  slot.mark_in_flight(False)
  assert slot.take() == "frame-3"


def test_off_mode_is_zero_and_has_no_work() -> None:
  assert VisionSpeedLimitMode.OFF.value == 0
  slot = LatestFrameSlot(enabled=False)
  assert slot.submit("frame") is False
  assert slot.pending is None


def test_producer_epoch_reset_drops_pending_state() -> None:
  slot = LatestFrameSlot()
  slot.submit("old")
  first = slot.producer_epoch
  slot.reset_epoch()
  assert slot.producer_epoch == first + 1
  assert slot.pending is None
  assert slot.in_flight is False
