"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

A stop the brain says (a red light, a stop sign), as a braking of its own among the planner's candidates.

openpilot has no "stop in N metres": the planner takes the hardest braking of the lead car, the set speed and the driving
model, and shouldStop only says the car stands. The brain says a stop as it says a limit: the limit ahead at the lowest
speeds (STOP_SPEED and under) with the metres to it, then that speed as the one to hold. With the Stop for Smartspeed
toggle (SmartspeedStop) in Smartspeed, the braking that stops the car there, -v² / 2d, is one more candidate: the planner
takes it when it is the hardest, and the car stands while the brain says stop. Nothing of the radar or the lead is
touched, and without the toggle nothing changes.
"""
import time

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

# A speed this low is a stop: the brain's table starts at 1 km/h, and a limit of 0 is "none".
STOP_SPEED = 2. * CV.KPH_TO_MS
# The hardest braking asked for a stop, and how fast it may build up (m/s³).
STOP_ACCEL_MIN = -3.0
STOP_JERK = 2.0
# The last metres: the braking is worked out for a metre at least, so it fades as the car stops.
STOP_MIN_DISTANCE = 1.0


class StopAhead:
  def __init__(self):
    self.params = Params()
    self.frame = -1
    self.enabled = self.params.get_bool("SmartspeedStop")
    self.active = False
    self.distance = 0.
    self.output_a_target = 0.

  def update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("SmartspeedStop")

  def update(self, sm: messaging.SubMaster, smartspeed_engaged: bool, v_ego: float) -> None:
    """Every plan: is there a stop, and how far. smartspeed_engaged: openpilot drives the speed and the mode is
    Smartspeed. A stale speedVisionSP is never used."""
    self.frame += 1
    self.update_params()
    self.active, self.distance = False, 0.
    if not (self.enabled and smartspeed_engaged):
      return
    try:
      alive = sm.alive['speedVisionSP']
      vision = sm['speedVisionSP']
    except KeyError:
      return
    if not alive:
      return
    if 0. < vision.speedLimit <= STOP_SPEED:
      self.active = True
    elif 0. < vision.speedLimitAhead <= STOP_SPEED:
      # The metres were measured when speedvisiond sent them
      since = v_ego * (time.monotonic() - sm.recv_time['speedVisionSP'])
      self.active, self.distance = True, max(0., vision.speedLimitAheadDistance - since)

  def a_target(self, v_ego: float, a_prev: float, dt: float) -> float | None:
    """The braking that stops the car at the stop, None without one: no harder than STOP_ACCEL_MIN, building up from the
    plan's last acceleration no faster than STOP_JERK."""
    if not self.active:
      return None
    a_target = -v_ego ** 2 / (2. * max(self.distance, STOP_MIN_DISTANCE))
    self.output_a_target = float(np.clip(max(a_target, a_prev - STOP_JERK * dt), STOP_ACCEL_MIN, 0.))
    return self.output_a_target
