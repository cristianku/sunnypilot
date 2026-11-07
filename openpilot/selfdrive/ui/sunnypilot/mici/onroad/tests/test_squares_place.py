import pyray as rl
import pytest

from openpilot.selfdrive.ui.sunnypilot.mici.onroad.final_speed_limit import final_box
from openpilot.selfdrive.ui.sunnypilot.mici.onroad.vision_speed_limit import GAP, MARGIN, READING_SIZE, SIZE, \
  reading_box, vision_box

RECT = rl.Rectangle(0, 0, 500, 240)
# The GPU icon of the big model (HudRenderer._draw_model_source), the orange one, the widest.
GPU_ICON = rl.Rectangle(RECT.width - 10 - 75, RECT.height - 14 - (50 + 44) / 2, 75, 44)


def _overlap(a: rl.Rectangle, b: rl.Rectangle) -> bool:
  return a.x < b.x + b.width and b.x < a.x + a.width and a.y < b.y + b.height and b.y < a.y + a.height


@pytest.mark.parametrize('box', [vision_box(RECT), vision_box(RECT, True), final_box(RECT), reading_box(RECT),
                                 reading_box(RECT, True)])
def test_no_square_covers_the_gpu_icon(box):
  assert not _overlap(box, GPU_ICON)


def test_the_road_type_under_the_squares_stays_off_the_gpu_icon():
  box = reading_box(RECT)
  assert box.y + READING_SIZE + GAP + 20 <= GPU_ICON.y


@pytest.mark.parametrize('beside_final', [False, True])
def test_the_reading_goes_under_the_vision_square(beside_final):
  vision, reading = vision_box(RECT, beside_final), reading_box(RECT, beside_final)
  assert reading.y == vision.y + SIZE + GAP and reading.x - vision.x == vision.x + SIZE - reading.x - READING_SIZE
  assert not _overlap(reading, final_box(RECT))


def test_the_final_limit_takes_the_corner_and_the_vision_square_goes_beside_it():
  final, vision = final_box(RECT), vision_box(RECT, True)
  assert (final.x, final.y) == (RECT.width - MARGIN - SIZE, MARGIN) == (vision_box(RECT).x, vision_box(RECT).y)
  assert not _overlap(final, vision)
  assert vision.x + SIZE + GAP == final.x and vision.y == final.y
