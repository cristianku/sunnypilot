"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time
from types import SimpleNamespace

import pytest

from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit import stop_ahead
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.stop_ahead import STOP_ACCEL_MIN, STOP_JERK, StopAhead

KPH = CV.KPH_TO_MS


class SM(dict):
  """A SubMaster with speedVisionSP alone."""
  def __init__(self, limit=0., ahead=0., distance=0., alive=True, age=0.):
    super().__init__(speedVisionSP=SimpleNamespace(speedLimit=limit, speedLimitAhead=ahead, speedLimitAheadDistance=distance))
    self.alive = {'speedVisionSP': alive}
    self.recv_time = {'speedVisionSP': time.monotonic() - age}


@pytest.fixture
def stop():
  found = StopAhead()
  found.enabled = True
  found.update_params = lambda: None
  return found


def test_off_by_default_and_then_no_candidate():
  stop = StopAhead()
  stop.update(SM(ahead=1 * KPH, distance=60.), True, 14.)
  assert not stop.enabled and not stop.active and stop.a_target(14., 0., DT_MDL) is None


def test_only_while_openpilot_drives_the_speed_in_smartspeed(stop):
  stop.update(SM(ahead=1 * KPH, distance=60.), False, 14.)
  assert not stop.active and stop.a_target(14., 0., DT_MDL) is None


def test_a_limit_ahead_is_no_stop(stop):
  stop.update(SM(limit=50 * KPH, ahead=30 * KPH, distance=60.), True, 14.)
  assert not stop.active and stop.a_target(14., 0., DT_MDL) is None


def test_a_stale_or_missing_reading_is_never_used(stop):
  stop.update(SM(ahead=1 * KPH, distance=60., alive=False), True, 14.)
  assert not stop.active
  stop.update(SimpleNamespace(alive={}), True, 14.)
  assert not stop.active


def test_a_stop_ahead_brakes_to_stand_there(stop):
  stop.update(SM(limit=50 * KPH, ahead=1 * KPH, distance=60.), True, 14.)
  assert stop.active and stop.distance == pytest.approx(60., abs=0.1)
  # Already braking as hard: the braking that stops in 60 m from 14 m/s
  assert stop.a_target(14., -2., DT_MDL) == pytest.approx(-14. ** 2 / 120., abs=0.01)


def test_the_metres_shrink_with_the_way_made_since_the_reading(stop):
  stop.update(SM(ahead=1 * KPH, distance=60., age=1.), True, 14.)
  assert stop.distance == pytest.approx(46., abs=0.1)


def test_the_braking_builds_up_and_has_a_floor(stop):
  stop.update(SM(ahead=1 * KPH, distance=5.), True, 14.)
  assert stop.a_target(14., 0., DT_MDL) == pytest.approx(-STOP_JERK * DT_MDL)
  assert stop.a_target(14., -5., DT_MDL) == STOP_ACCEL_MIN


def test_a_speed_to_hold_that_low_is_a_stop_here(stop):
  stop.update(SM(limit=1 * KPH), True, 5.)
  assert stop.active and stop.distance == 0.
  assert stop.a_target(5., -3., DT_MDL) == STOP_ACCEL_MIN


def test_the_car_stands_while_the_brain_says_stop(stop):
  stop.update(SM(limit=1 * KPH), True, 0.)
  a_target = stop.a_target(0., 0., DT_MDL)
  assert a_target == 0. and should_stop(0., a_target)
  # The stop goes (green): no candidate, the others move the car off
  stop.update(SM(limit=50 * KPH), True, 0.)
  assert stop.a_target(0., 0., DT_MDL) is None


def test_it_never_accelerates(stop):
  stop.update(SM(ahead=1 * KPH, distance=500.), True, 1.)
  assert stop.a_target(1., 1.5, DT_MDL) <= 0.
  assert stop_ahead.STOP_SPEED < 5 * KPH
