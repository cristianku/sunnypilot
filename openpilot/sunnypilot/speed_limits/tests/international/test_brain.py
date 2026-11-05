"""The brain's reader on a fake network: what it sends in, what it says, what it refuses."""

import hashlib
from pathlib import Path

import numpy as np
import pytest
from openpilot.sunnypilot.speed_limits import brain, brain_inputs

GROUPS = ['signs', 'plate', 'hour', 'lead', 'country']
SIZE = sum(brain_inputs.INPUT_SIZES[name] for name in GROUPS)


def contract(**changes):
  inputs = hashlib.sha256(Path(brain_inputs.__file__).read_bytes()).hexdigest()
  step = {'rate_hz': 1.0, 'groups': GROUPS, 'memory_gap_seconds': 5.0}
  return {'inputs_sha256': inputs, 'output': {'kph': [5, 10, 15, 20]}, 'step': step, **changes}


class Network:
  """Says class 2 (15 km/h); its memory is how many steps it has made since zeros."""
  def __init__(self):
    self.calls = []

  def input_shape(self, name):
    return (1, 4)

  def __call__(self, image, extra, hidden):
    self.calls.append((image, extra, hidden))
    return {'speed': np.array([[0.0, 0.1, 0.9, 0.0]], dtype=np.float32), 'hidden_out': hidden + 1}


@pytest.fixture
def reader(vision):
  return brain.BrainReader(Network(), contract(), vision, 64, 32)


def frame():
  return np.zeros((32, 64), dtype=np.uint8), np.full((16, 64), 128, dtype=np.uint8)


def test_refuses_a_network_made_with_other_inputs(vision):
  with pytest.raises(ValueError, match='another brain_inputs'):
    brain.BrainReader(Network(), contract(inputs_sha256='0' * 64), vision, 64, 32)


def test_refuses_a_network_asking_for_what_the_car_does_not_give(vision):
  wrong = contract(step={'rate_hz': 1.0, 'groups': ['signs', 'map'], 'memory_gap_seconds': 5.0})
  with pytest.raises(ValueError, match='does not give'):
    brain.BrainReader(Network(), wrong, vision, 64, 32)


def test_says_the_class_in_metres_per_second(reader):
  assert reader.speed_limit == 0.0
  assert reader.step(1.0, *frame(), None, None, None) == pytest.approx(15 / 3.6)
  assert reader.speed_limit == pytest.approx(15 / 3.6)


def test_steps_once_a_second(reader):
  assert reader.due(0.0)
  reader.step(0.0, *frame(), None, None, None)
  assert not reader.due(0.9)
  assert reader.due(1.0)


def test_the_groups_go_in_side_by_side(reader):
  reader.step(1.0, *frame(), (30.0, 36.0), 1_770_000_000.0, 'DE')
  _, extra, _ = reader.network.calls[0]
  assert extra.shape == (1, SIZE)
  at = sum(brain_inputs.INPUT_SIZES[name] for name in GROUPS[:3])
  assert np.array_equal(extra[0, at:at + 3], brain_inputs.lead_row(30.0, 36.0))
  assert np.array_equal(extra[0, -brain_inputs.INPUT_SIZES['country']:], brain_inputs.country_row('DE'))


def test_a_sign_read_reaches_the_network(reader):
  for at in (0.0, 0.2, 0.4, 0.6):
    reader.add_signs(at, [{'class': 'maximum_speed:80', 'confidence': 0.99}])
  reader.step(1.0, *frame(), None, None, None)
  assert reader.signs.number == 80
  assert reader.network.calls[0][1][0, 0] == pytest.approx(80 / brain_inputs.SIGN_KPH_SCALE)


def test_the_memory_goes_on_and_starts_again_after_a_gap(reader):
  reader.step(0.0, *frame(), None, None, None)
  reader.step(1.0, *frame(), None, None, None)
  assert reader.network.calls[1][2][0, 0] == 1
  reader.step(7.0, *frame(), None, None, None)
  assert reader.network.calls[2][2][0, 0] == 0
