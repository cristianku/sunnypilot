import pytest

from openpilot.selfdrive.ui.sunnypilot.mici.onroad.sign_reading import ROAD_SIGNS, distance_text, reading_text
from openpilot.sunnypilot.speed_limits import vision


@pytest.mark.parametrize('sign_class, value, shown', [
  ("", 0., ("", False)),
  ("maximum_speed:80", 80., ("80", False)),
  ("maximum_speed:120", 120., ("120", False)),
  ("maximum_speed_hours:30:22-6", 30., ("30", False)),
  ("variable_display:100", 100., ("100", False)),
  ("cancellation", 0., ("END", False)),
  ("other_sign:motorway_start", 0., ("MWY", False)),
  ("other_sign:town_exit", 0., ("TOWN", True)),
])
def test_the_square_says_the_sign_as_read(sign_class, value, shown):
  assert reading_text(sign_class, value) == shown


def test_every_road_sign_a_reading_can_be_has_a_name():
  assert set(ROAD_SIGNS) == set(vision.ROAD_SIGNS)


@pytest.mark.parametrize('metres, metric, shown', [
  (0., True, ""),
  (0.4, True, ""),
  (42.4, True, "42 m"),
  (142., True, "142 m"),
  (30.48, False, "100 ft"),
  (0., False, ""),
])
def test_the_square_says_the_distance_to_the_sign(metres, metric, shown):
  assert distance_text(metres, metric) == shown
