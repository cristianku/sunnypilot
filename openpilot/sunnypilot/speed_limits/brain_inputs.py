"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

VISION AND MAP ARE TWO SEPARATE SOURCES: NOTHING HERE READS THE MAP (liveMapDataSP). THE BRAIN'S INPUTS COME FROM THE CAMERA
AND THE CAR'S OWN SIGNALS ONLY; THE COUNTRY IS THE ONLY THING GPS GIVES, AS IT DOES TO THE SIGNS' UNIT.

What the brain is given beside the picture, frame by frame, all from what comes at or before the frame. One file for the
car and for the training: the server loads this very file by its path (numpy only, nothing of openpilot), so that what the
network learned on is what it gets on the car, and the checkpoint keeps the hash of the file it was trained with. Nothing
on the car imports it yet: the brain has no network there.
"""
import math

import numpy as np

COUNTRIES = ("CH", "DE", "FR", "IT", "AT", "US")
SIGN_KPH_SCALE = 130.0
SIGN_AGE_SCALE = math.log1p(3600.0)
LEAD_METRES_SCALE = 100.0
LEAD_METRES_MAX = 200.0
MEMORY_GAP_SECONDS = 5.0  # longer without a frame: its memory starts again from zeros
INPUT_SIZES = {"signs": 6, "lead": 3, "country": len(COUNTRIES) + 1}


class SignInputs:
  """The signs group from the sure readings of a drive: add() them in order, features() at a moment after them."""

  def __init__(self):
    self.number, self.display, self.read_at, self.ended = None, False, None, False

  def add(self, now: float, signs: list) -> None:
    """signs: (class, confidence) of a frame."""
    for name, _confidence in signs:
      kind, _, value = name.partition(":")
      if kind in ("maximum_speed", "variable_display") and value.isdigit():
        self.number, self.display, self.read_at, self.ended = int(value), kind == "variable_display", now, False
      elif kind == "cancellation":
        self.ended = True

  def features(self, now: float) -> np.ndarray:
    """The last number read / SIGN_KPH_SCALE, known, it came from a display, its age, an end read since it, a number read in
    the last second."""
    row = np.zeros(INPUT_SIZES["signs"], dtype=np.float32)
    if self.number is not None:
      row[:] = (self.number / SIGN_KPH_SCALE, 1.0, float(self.display), math.log1p(now - self.read_at) / SIGN_AGE_SCALE,
                float(self.ended), float(now - self.read_at < 1))
    return row


def lead_row(metres: float | None, kph: float | None) -> np.ndarray:
  """Distance and speed of the car ahead, known."""
  row = np.zeros(INPUT_SIZES["lead"], dtype=np.float32)
  if metres is not None and kph is not None:
    row[:] = min(metres, LEAD_METRES_MAX) / LEAD_METRES_SCALE, kph / SIGN_KPH_SCALE, 1.0
  return row


def country_row(name: str | None) -> np.ndarray:
  """One of COUNTRIES, the last place for any other and for none."""
  row = np.zeros(INPUT_SIZES["country"], dtype=np.float32)
  row[COUNTRIES.index(name) if name in COUNTRIES else len(COUNTRIES)] = 1.0
  return row


def pack_inputs(inputs: dict, groups: tuple) -> np.ndarray:
  """The extra input of the network: the groups side by side, in the order the network was trained with."""
  return np.concatenate([inputs[name] for name in groups], axis=-1)
