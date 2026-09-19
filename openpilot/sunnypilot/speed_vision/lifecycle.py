from __future__ import annotations

from enum import IntEnum
from typing import Any


class VisionSpeedLimitMode(IntEnum):
  OFF = 0
  SHADOW = 1
  OBSERVATION = 2
  ADVISORY = 3


class LatestFrameSlot:
  """One in-flight frame and one replaceable pending frame."""

  def __init__(self, *, enabled: bool = True) -> None:
    self.enabled = enabled
    self.pending: Any | None = None
    self.in_flight = False
    self.replaced_frames = 0
    self.producer_epoch = 0

  def submit(self, frame: Any) -> bool:
    if not self.enabled:
      return False
    if self.in_flight:
      if self.pending is not None:
        self.replaced_frames += 1
      self.pending = frame
      return True
    if self.pending is not None:
      self.replaced_frames += 1
    self.pending = frame
    return True

  def take(self) -> Any | None:
    if not self.enabled or self.in_flight:
      return None
    frame = self.pending
    self.pending = None
    if frame is not None:
      self.in_flight = True
    return frame

  def mark_in_flight(self, active: bool) -> None:
    self.in_flight = bool(active)

  def reset_epoch(self) -> None:
    self.producer_epoch += 1
    self.pending = None
    self.in_flight = False
    self.replaced_frames = 0

  def disable(self) -> None:
    self.enabled = False
    self.pending = None
    self.in_flight = False

  def enable(self) -> None:
    self.enabled = True
