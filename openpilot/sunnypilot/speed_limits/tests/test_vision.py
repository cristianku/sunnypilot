"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The expected values come from ffmpeg 7.1 (yuv420p to rgb24, as the dataset's frames were decoded) and Pillow 12
(Image.resize BILINEAR, as the networks' inputs were resized in training).
"""
import numpy as np
import pytest

from openpilot.sunnypilot.speed_limits import vision

IMAGE = np.array([[139, 241, 6, 106, 188, 229, 10], [86, 66, 213, 167, 81, 153, 76], [250, 50, 112, 233, 89, 6, 143],
                  [210, 166, 189, 33, 30, 202, 220], [122, 77, 115, 230, 240, 184, 63]], dtype=np.uint8)
PIL_DOWN_4X3 = [[136, 120, 150, 111], [148, 143, 104, 115], [140, 140, 157, 157]]
PIL_UP_9X8 = [[139, 207, 137, 28, 106, 170, 211, 156, 10], [116, 148, 134, 105, 133, 139, 172, 143, 39],
              [96, 76, 128, 199, 171, 101, 116, 122, 80], [199, 103, 95, 159, 212, 114, 67, 75, 122],
              [238, 137, 109, 144, 171, 93, 69, 101, 167], [213, 177, 170, 153, 46, 37, 121, 198, 215],
              [161, 131, 130, 147, 144, 147, 173, 172, 132], [122, 92, 94, 141, 230, 238, 209, 144, 63]]
FFMPEG_RGB = {(16, 128, 128): (0, 0, 0), (235, 128, 128): (255, 255, 255), (128, 128, 128): (130, 130, 130),
              (17, 128, 129): (2, 0, 1), (120, 77, 150): (156, 122, 18), (90, 128, 100): (41, 108, 86),
              (60, 160, 77): (0, 79, 115), (200, 100, 77): (132, 255, 157), (81, 90, 240): (253, 0, 0)}


def nv12(y: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  uv = np.empty((u.shape[0], u.shape[1] * 2), dtype=np.uint8)
  uv[:, 0::2], uv[:, 1::2] = u, v
  return y.astype(np.uint8), uv


def test_resize_is_pillow():
  planes = IMAGE[None]
  assert vision.resize(planes, 4, 3)[0].tolist() == PIL_DOWN_4X3
  assert vision.resize(planes, 9, 8)[0].tolist() == PIL_UP_9X8
  assert np.array_equal(vision.resize(planes, 7, 5), planes)


def test_resample_coefficients_drop_only_zero_taps():
  index, fixed = vision.resample_coefficients(1344, 896)
  assert index.shape == (896, 3)
  assert np.all(np.abs(fixed.sum(axis=1) - (1 << vision.PRECISION_BITS)) <= 3)
  assert index.min() >= 0 and index.max() <= 1343


@pytest.mark.parametrize("in_size, out_size", [(1344, 896), (1928, 896), (300, 200), (96, 64), (90, 60), (64, 48), (9, 6)])
def test_column_resampler_is_the_general_way(in_size, out_size):
  planes = np.random.default_rng(in_size).integers(0, 256, (3, 5, in_size), dtype=np.uint8)
  resampler = vision.ColumnResampler(in_size, out_size)
  general = vision.transpose(vision.resample_rows(vision.transpose(planes), resampler.index, resampler.fixed))
  assert np.array_equal(resampler(planes), general)
  if (in_size, out_size) == (1344, 896):
    assert len(resampler.phases) == 2 and len(resampler.edges) <= 4


def test_yuv_to_rgb_is_ffmpeg():
  for (y, u, v), rgb in FFMPEG_RGB.items():
    planes = vision.yuv_to_rgb(*nv12(np.full((2, 2), y), np.full((1, 1), u), np.full((1, 1), v)), 0, 2, 0, 2)
    assert planes[:, 0, 0].tolist() == list(rgb), (y, u, v)


def test_yuv_to_rgb_region_is_the_frame_cut():
  rng = np.random.default_rng(0)
  y, uv = nv12(rng.integers(0, 256, (10, 12)), rng.integers(0, 256, (5, 6)), rng.integers(0, 256, (5, 6)))
  whole = vision.yuv_to_rgb(y, uv, 0, 10, 0, 12)
  for top, bottom, left, right in ((1, 8, 3, 12), (0, 1, 0, 1), (9, 10, 11, 12), (3, 7, 1, 6)):
    assert np.array_equal(vision.yuv_to_rgb(y, uv, top, bottom, left, right), whole[:, top:bottom, left:right])


def test_nv12_planes_with_stride():
  data = np.arange(256 * 6, dtype=np.int64).astype(np.uint8)
  y, uv = vision.nv12_planes(data, 6, 4, 128, 128 * 4)
  assert y.shape == (4, 6) and uv.shape == (2, 6)
  assert y[1, 0] == 128 % 256 and uv[1, 2] == (128 * 4 + 128 + 2) % 256


def test_signnet_input_shape():
  height, width = 760, 1344
  y, uv = nv12(np.full((height, width), 235), np.full((height // 2, width // 2), 128), np.full((height // 2, width // 2), 128))
  signnet_input = vision.SignNetInput(width, height)
  x = signnet_input(y, uv)
  assert x.shape == (1, 3, vision.SIGNNET_HEIGHT, vision.SIGNNET_WIDTH) and x.dtype == np.float32
  assert np.all(x == 1.0)
  assert signnet_input.used_rows == 432 and signnet_input.scale == 1.5


def test_signnet_boxes():
  output = np.zeros((1, 5, 36, 112), dtype=np.float32)
  output[0, 0, 10, 20] = 0.9
  output[0, 0, 10, 21] = 0.5  # beside a higher peak: not a local maximum
  output[0, 0, 30, 100] = 0.4
  output[0, 0, 5, 5] = 0.2  # under MIN_BOX_SCORE
  output[0, 1:, 10, 20] = [np.log(16), np.log(24), 0.5, 0.25]
  boxes = vision.signnet_boxes(output, 1.5)
  assert [round(box["score"], 2) for box in boxes] == [0.9, 0.4]
  # centre (20.5 * 8, 10.25 * 8) = (164, 82), 16 x 24 in band pixels, times 1.5
  assert np.allclose(boxes[0]["box"], [234, 105, 258, 141])


def test_crop_rectangle():
  assert vision.crop_rectangle([10.2, 20.7, 30.1, 40.0], 100, 50) == (10, 20, 31, 40)
  assert vision.crop_rectangle([-5, -5, 0.5, 0.2], 100, 50) == (0, 0, 2, 2)
  assert vision.crop_rectangle([99.6, 49.9, 120, 80], 100, 50) == (98, 48, 100, 50)


def test_classifier_input():
  y, uv = nv12(np.full((20, 30), 128), np.full((10, 15), 128), np.full((10, 15), 128))
  x = vision.classifier_input(y, uv, [3, 4, 13, 9])
  assert x.shape == (1, 3, vision.CLASSIFIER_SIZE, vision.CLASSIFIER_SIZE)
  expected = (130 / 255 - vision.CLASSIFIER_MEAN.reshape(3)) / vision.CLASSIFIER_STD.reshape(3)
  assert np.allclose(x[0, :, 80, 80], expected, atol=1e-6)


@pytest.mark.parametrize("name, speed", [("maximum_speed:80", 80), ("variable_display:100", 100), ("zone:30", 30),
                                         ("road_marking:30", 30), ("cancellation", 0), ("maximum_speed:unknown", None),
                                         ("zone:unknown", None), ("other_sign:give_way", None), ("not_a_sign", None),
                                         ("unreadable", None)])
def test_class_speed(name, speed):
  assert vision.class_speed(name) == speed


def test_read_signs():
  output = np.zeros((1, 5, 36, 112), dtype=np.float32)
  output[0, 0, 10, 20], output[0, 0, 20, 50] = 0.9, 0.8
  output[0, 1:3] = np.log(20)
  classes = ["maximum_speed:50", "maximum_speed:80", "not_a_sign"]
  answers = iter([np.array([[5.0, 0.0, 0.0]]), np.array([[0.0, 0.5, 0.0]])])
  y, uv = nv12(np.full((760, 1344), 128), np.full((380, 672), 128), np.full((380, 672), 128))
  signs = vision.read_signs(y, uv, vision.SignNetInput(1344, 760), lambda x: output, lambda x: next(answers), classes)
  assert [(sign["class"], sign["speed"]) for sign in signs] == [("maximum_speed:50", 50), ("maximum_speed:80", None)]


def test_memory_confirms_and_holds():
  memory = vision.SpeedLimitMemory()
  assert memory.update(0.0, [60]) == 0  # one frame is not enough
  assert memory.update(0.2, [60]) == 60 and memory.sign_read
  assert memory.update(0.4, []) == 60 and not memory.sign_read  # held without a sign
  assert memory.update(10.0, [80]) == 60
  assert memory.update(12.0, [80]) == 60  # too far apart
  assert memory.update(12.4, [80]) == 80
  assert memory.update(15.0, [50, 50]) == 80  # a value counts once a frame
  assert memory.update(20.0, [0]) == 80
  assert memory.update(20.2, [0]) == 0  # the end of the limit clears it


def test_memory_two_signs_at_once():
  memory = vision.SpeedLimitMemory()
  memory.update(0.0, [80, 60])
  assert memory.update(0.2, [80, 60]) == 60  # a tie: the lower
  memory.update(5.0, [100, 120])
  memory.update(5.2, [120])
  assert memory.update(5.4, [100, 120]) == 120  # the one seen most
  memory.update(9.0, [0, 60])
  assert memory.update(9.2, [0, 60]) == 60  # the end of a limit beside roadworks: not the general limit


def confirm(memory, at, readings=(), signs=(), **context):
  memory.update(at, list(readings), list(signs), **context)
  return memory.update(at + 0.2, list(readings), list(signs), **context)


def test_memory_road_signs_set_the_general_limit():
  memory = vision.SpeedLimitMemory()
  assert confirm(memory, 0.0, signs=["other_sign:town_entry"]) == 50
  assert confirm(memory, 5.0, [30]) == 30
  assert confirm(memory, 9.0, [0]) == 50  # the end of the 30 in a town: the town's 50
  assert confirm(memory, 20.0, signs=["other_sign:town_exit"]) == 80
  assert confirm(memory, 30.0, signs=["other_sign:give_way"]) == 80  # not a sign of the road's kind
  assert not memory.sign_read
  assert confirm(memory, 40.0, signs=["other_sign:town_exit"]) == 80 and memory.sign_read  # the same limit read again
  assert vision.SpeedLimitMemory("IT").update(0.0, [], ["other_sign:motorway_start"]) == 0
  assert confirm(vision.SpeedLimitMemory("IT"), 0.0, signs=["other_sign:motorway_start"]) == 130


def test_memory_ramp_limit_ends_by_the_road_type_network():
  memory = vision.SpeedLimitMemory()
  assert confirm(memory, 0.0, signs=["other_sign:motorway_start"], road_type="motorway") == 120
  assert confirm(memory, 10.0, [60], road_type="motorway") == 60  # the ramp's 60, by the sign, whatever the network
  for t in range(12, 20):
    assert memory.update(float(t), [], road_type="ramp") == 60  # still on the ramp
  for t in range(20, 27):
    assert memory.update(float(t), [], road_type="motorway") == 60  # on the carriageway, not long enough
  assert memory.update(27.0, [], road_type="town") == 60  # a doubt starts the count again
  for t in range(28, 36):
    memory.update(float(t), [], road_type="motorway")
  assert memory.update(36.0, [], road_type="motorway") == 120
  assert memory.road == "motorway"


def test_memory_ramp_limit_far_from_the_sign_by_the_network():
  memory = vision.SpeedLimitMemory()
  confirm(memory, 0.0, signs=["other_sign:motorway_start"])
  assert confirm(memory, 100.0, [60], road_type="ramp") == 60  # far from the sign, the network says ramp
  for t in range(102, 110):
    memory.update(float(t), [], road_type="expressway")
  assert memory.update(110.0, [], road_type="expressway") == 100


def test_memory_ramp_limit_holds_without_the_network():
  memory = vision.SpeedLimitMemory()
  confirm(memory, 0.0, signs=["other_sign:motorway_start"])
  assert confirm(memory, 10.0, [60]) == 60
  for t in range(12, 200):
    assert memory.update(float(t), []) == 60  # nothing says the ramp is over: the 60 holds until a sign


def test_memory_motorway_limit_is_not_a_ramp_limit():
  memory = vision.SpeedLimitMemory()
  confirm(memory, 0.0, signs=["other_sign:motorway_start"])
  assert confirm(memory, 300.0, [80], road_type="motorway") == 80  # roadworks long after the entry
  for t in range(301, 320):
    assert memory.update(float(t), [], road_type="motorway") == 80  # the motorway does not end it


def test_memory_follows_the_map_on_another_road():
  memory = vision.SpeedLimitMemory()
  assert confirm(memory, 0.0, [60]) == 60  # the ramp's 60, no limit on the ramp in the map
  for t in range(2, 5):
    assert memory.update(float(t), [], map_limit=120) == 60  # the map's 120 not held long enough
  assert memory.update(5.0, [], map_limit=120) == 120  # on the motorway: the map's limit
  assert confirm(memory, 20.0, [80], map_limit=120) == 80  # roadworks: the sign wins
  for t in range(22, 60):
    assert memory.update(float(t), [], map_limit=120) == 80  # the same road in the map: the 80 holds


def test_memory_a_wrong_branch_sign_ends_with_the_road():
  memory = vision.SpeedLimitMemory()
  assert confirm(memory, 0.0, [40], map_limit=60) == 40  # the other branch's 40 on our ramp
  memory.update(3.0, [], map_limit=120)
  memory.update(4.0, [], map_limit=60)  # the GPS beside the motorway for a second
  assert memory.update(7.0, [], map_limit=120) == 40
  assert memory.update(10.0, [], map_limit=120) == 120


def test_memory_without_a_limit_leaves_the_map_alone():
  memory = vision.SpeedLimitMemory()
  for t in range(10):
    assert memory.update(float(t), [], map_limit=50 if t < 5 else 80) == 0  # nothing read: the map is used as it is
  assert confirm(memory, 20.0, [60], map_limit=80) == 60
  for t in range(22, 26):
    memory.update(float(t), [], map_limit=None)  # the map has no limit here: nothing changes
  assert memory.speed_limit == 60


def test_memory_germany_has_no_general_motorway_limit():
  memory = vision.SpeedLimitMemory("DE")
  confirm(memory, 0.0, signs=["other_sign:motorway_start"])
  assert confirm(memory, 5.0, [80]) == 80
  for t in range(7, 16):
    memory.update(float(t), [], road_type="motorway")
  assert memory.speed_limit == 0  # no limit
