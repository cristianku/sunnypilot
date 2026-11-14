"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

pipeline.SpeedVision: what speedvisiond does between its sockets, and what a replay of a recorded drive runs. The
daemon's own behaviour around it is in international/test_daemon.py.
"""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.sunnypilot.speed_limits import pipeline, vision
from openpilot.sunnypilot.speed_limits.country import GpsFix
from openpilot.sunnypilot.speed_limits.tests.test_speed_vision_sp import FIELDS

WIDTH, HEIGHT = 1344, 760
Y, UV = np.zeros((HEIGHT, WIDTH), dtype=np.uint8), np.zeros((HEIGHT // 2, WIDTH), dtype=np.uint8)


def sign(name="maximum_speed:50", confidence=0.95, box=(600, 200, 640, 240)):
  return {"class": name, "box": list(box), "confidence": confidence, "speed": vision.class_speed(name)}


def files(**settings):
  return NS(name="synthetic", settings=settings, classes=["maximum_speed:50"])


@pytest.fixture
def frames(monkeypatch):
  """The signs each frame read gives, in order: what vision.read_signs returns."""
  given = []
  monkeypatch.setattr(vision, "read_signs", lambda *args: given.pop(0) if given else [])
  return given


class Brain:
  """A brain.BrainReader saying 80 km/h from its first step, with what it was given."""
  def __init__(self):
    self.speed_limit = 0.0
    self.road_type = ""
    self.signs = NS(read_at=None)
    self.given, self.steps, self.enabled, self.map = [], [], [], []

  def add_signs(self, now, signs):
    self.given.append(signs)
    if signs:
      self.signs.read_at = now

  def add_pedals(self, now, gas, brake):
    pass

  def add_car_state(self, now, steering_deg, left, right, yaw_rate):
    pass

  def add_device_motion(self, now, z, valid):
    pass

  def add_enabled(self, now, enabled):
    self.enabled.append(enabled)

  def add_map_limit(self, now, metres_per_second, valid):
    self.map.append((now, metres_per_second, valid))

  def due(self, now):
    return True

  def step(self, now, y, uv, lead, unix_seconds, country, v_ego, sign_distance, position=None):
    self.steps.append({"lead": lead, "unix": unix_seconds, "country": country, "v_ego": v_ego})
    self.speed_limit = 80 / 3.6


def test_a_release_is_read_from_its_directory(tmp_path):
  manifest = {"format": 2, "version": "2026-11-10", "finder": {"file": "signnet.onnx", "contract": "signnet.json"},
              "classifier": {"file": "classifier.onnx", "contract": "classifier.json"},
              "decider": {"file": "decider.onnx", "contract": "decider.json"}, "settings": {"min_confidence": 0.8}}
  (tmp_path / "speed_limits.json").write_text(json.dumps(manifest))
  (tmp_path / "classifier.json").write_text(json.dumps({"output": {"classes": ["maximum_speed:50", "cancellation"]}}))
  (tmp_path / "decider.json").write_text(json.dumps({"task": "decider"}))
  found = pipeline.release(tmp_path)
  assert found.name == "speed-limits 2026-11-10" and found.classes == ["maximum_speed:50", "cancellation"]
  assert (found.finder, found.classifier, found.decider) == tuple(tmp_path / name for name in
                                                                 ("signnet.onnx", "classifier.onnx", "decider.onnx"))
  assert found.brain is None and found.road_type is None and found.decider_contract == {"task": "decider"}
  assert found.onnx() == [tmp_path / "signnet.onnx", tmp_path / "classifier.onnx", tmp_path / "decider.onnx"]
  assert found.settings == {"min_confidence": 0.8}


def test_no_release_in_an_older_format_or_an_empty_directory(tmp_path):
  assert pipeline.release(tmp_path) is None
  (tmp_path / "speed_limits.json").write_text(json.dumps({"format": 1}))
  assert pipeline.release(tmp_path) is None


def test_the_maps_limit_goes_to_the_brain_and_to_nothing_else(frames):
  """Only a brain trained with the map's limit uses it (BrainReader.add_map_limit); the rules never see it."""
  frames.extend([[sign()], [sign()]])
  brain = Brain()
  with_brain = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None, reader=brain)
  with_brain.map_limit(1.0, 80 / 3.6, True)
  assert brain.map == [(1.0, 80 / 3.6, True)]
  rules = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None)
  rules.map_limit(1.0, 80 / 3.6, True)
  rules.frame(1.0, Y, UV)
  assert rules.frame(1.2, Y, UV)["speedLimit"] == pytest.approx(50 / 3.6)


def test_a_frame_is_read_a_period_after_the_last_one(frames):
  speed_vision = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None)
  assert speed_vision.due(10.0)
  speed_vision.frame(10.0, Y, UV)
  assert not speed_vision.due(10.0 + pipeline.PERIOD - 0.01)
  assert speed_vision.due(10.0 + pipeline.PERIOD)


def test_the_message_has_only_fields_of_speed_vision_sp(frames):
  frames.extend([[sign()], [sign()]])
  memory = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None)
  memory.frame(1.0, Y, UV)
  sent = memory.frame(1.2, Y, UV)
  assert set(sent) <= set(FIELDS)
  assert sent["speedLimit"] == pytest.approx(50 / 3.6) and sent["signRead"] and sent["rawSignSeen"]
  assert (sent["modelName"], sent["rawSignClass"], sent["speedLimitReason"]) == ("synthetic", "maximum_speed:50", "sign")
  frames.append([sign()])
  with_brain = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None, reader=Brain())
  sent = with_brain.frame(1.0, Y, UV)
  assert set(sent) <= set(FIELDS)
  assert sent["speedLimit"] == pytest.approx(80 / 3.6) and sent["signRead"]
  assert (sent["speedLimitState"], sent["speedLimitReason"], sent["rawSignTaken"]) == ("limit", "brain", True)
  assert "speedLimitAhead" not in sent and "speedLimitAheadDistance" not in sent and "roadType" not in sent


def test_the_limit_ahead_is_the_rules_and_never_a_brains(frames):
  # A 50 held, then a 30 on a sign ahead: the rules hold the 50 up to the sign and say the 30 with the metres to it
  ahead = [sign("maximum_speed:30", box=(600, 200, 620 + n, 220 + n)) for n in range(3)]
  frames.extend([[sign()], [sign()], *([one] for one in ahead)])
  rules = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None)
  sent = [rules.frame(1.0 + n * pipeline.PERIOD, Y, UV) for n in range(5)][-1]
  assert sent["speedLimit"] == pytest.approx(50 / 3.6) and sent["speedLimitAhead"] == pytest.approx(30 / 3.6)
  assert sent["speedLimitAheadDistance"] == pytest.approx(sent["signDistance"]) and sent["signDistance"] > 0
  # The same road with a brain: its speed alone, and the distance to the sign for the screen's reading square
  frames.extend([[sign()], [sign()], *([one] for one in ahead)])
  with_brain = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None, reader=Brain())
  sent = [with_brain.frame(1.0 + n * pipeline.PERIOD, Y, UV) for n in range(5)][-1]
  assert sent["speedLimit"] == pytest.approx(80 / 3.6) and sent["signDistance"] > 0
  assert "speedLimitAhead" not in sent and "speedLimitAheadDistance" not in sent


def test_the_brain_gets_the_cars_signals_at_its_step(frames):
  brain = Brain()
  speed_vision = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None, reader=brain)
  speed_vision.car_state(5.0, 4.99, 20.0, 0.0, False, False, 0.0, False, False)
  speed_vision.enabled(5.0, True)
  speed_vision.radar(True, 30.0, 10.0)
  speed_vision.gps(GpsFix(4.9, 47.0, 8.0, 5.0, True), 1_770_000_000_000)
  speed_vision.frame(5.0, Y, UV)
  speed_vision.radar(False, 0.0, 0.0)
  speed_vision.frame(6.0, Y, UV)
  assert brain.enabled == [True]
  assert brain.steps[0] == {"lead": (30.0, 36.0), "unix": pytest.approx(1_770_000_000), "country": None, "v_ego": 20.0}
  assert brain.steps[1]["lead"] is None and brain.steps[1]["unix"] == pytest.approx(1_770_000_001)  # the clock runs on


def test_only_the_last_gps_message_before_a_frame_counts(frames):
  brain = Brain()
  speed_vision = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None, reader=brain)
  speed_vision.gps(GpsFix(0.8, 47.0, 8.0, 5.0, True), 1_770_000_000_000)
  speed_vision.gps(GpsFix(0.9, 47.0, 8.0, 5.0, False), 1_770_000_000_100)  # the fix lost: speedvisiond's socket keeps this one
  speed_vision.frame(1.0, Y, UV)
  assert brain.steps[0]["unix"] is None


def test_a_sign_of_another_road_shows_and_is_not_taken(frames):
  class SideRoad:
    def rule(self, y, uv, signs, v_ego):
      for one in signs:
        one["road"] = "other_road"

  frames.extend([[sign()], [sign()]])
  brain = Brain()
  speed_vision = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 1141.5, None, None, reader=brain, ruler=SideRoad())
  speed_vision.frame(1.0, Y, UV)
  sent = speed_vision.frame(1.2, Y, UV)
  assert brain.given == [[], []]
  assert (sent["rawSignClass"], sent["rawSignRoad"], sent["rawSignTaken"], sent["rawSignSeen"]) == \
         ("maximum_speed:50", "other_road", False, True)
  assert [one["road"] for one in speed_vision.signs] == ["other_road"]  # a replay draws its box too


def test_the_road_type_network_only_without_a_brain(frames):
  road_types = NS(classes=["town", "garage"], due=lambda now: False, road_type="town")
  assert pipeline.SpeedVision(files(), WIDTH, HEIGHT, 0.0, None, None, reader=Brain(), road_types=road_types).road_types is None
  alone = pipeline.SpeedVision(files(), WIDTH, HEIGHT, 0.0, None, None, road_types=road_types)
  assert alone.memory.garage_known and alone.frame(1.0, Y, UV)["roadType"] == "town"
