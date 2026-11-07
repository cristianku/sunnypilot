"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.mici.onroad.vision_speed_limit import READING_SIZE, ROUNDNESS, SEGMENTS, BORDER, \
  DARK_GREY, GREY, reading_box
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

CANCELLATION = "cancellation"
# The signs of the kind of road a reading can be (speed_limits/vision.py: ROAD_SIGNS), by their short name, crossed out
# at their end as on the sign.
ROAD_SIGNS = {
  "other_sign:motorway_start": ("MWY", False),
  "other_sign:motorway_end": ("MWY", True),
  "other_sign:expressway_start": ("EXP", False),
  "other_sign:expressway_end": ("EXP", True),
  "other_sign:town_entry": ("TOWN", False),
  "other_sign:town_exit": ("TOWN", True),
}
# The text's size by its length: a number of two digits, of three, a word.
FONT_SIZES = {1: 30, 2: 30, 3: 24, 4: 19}
CONFIDENCE_FONT_SIZE = 16
CONFIDENCE_TOP = 5
TEXT_TOP = 21  # where the text's space starts, under the confidence
CROSS = 3  # the line through a road sign's end
# Red, the ring of a round sign, while the sign is in view.
RED = rl.Color(235, 60, 60, 235)


def reading_text(sign_class: str, value: float) -> tuple[str, bool]:
  """What the square says of a raw reading (speedVisionSP.rawSignClass and rawSignValue): its text and whether it is
  crossed out. The number as written on the sign, in its own unit; END for the end of a limit; a road sign's short
  name; "" before the first sign."""
  if not sign_class:
    return "", False
  if sign_class == CANCELLATION:
    return "END", False
  if sign_class in ROAD_SIGNS:
    return ROAD_SIGNS[sign_class]
  return (str(round(value)), False) if value > 0 else ("?", False)


class SignReading(Widget):
  """The last sign the classifier read (speedVisionSP.rawSign*), alone, in a small square under the vision square, for
  the driver to see the sign reader work apart from the limit the vision square shows: the sign as it was read (its
  number as written, END, a road sign's short name, crossed out at its end) with the classifier's confidence on top, sure
  enough or not, whatever the memory or the brain made of it. Its border is red while that sign is in view
  (rawSignSeen) and grey once it is passed, as the reading stays until another sign is read. A dash before the first
  sign or without speedvisiond. Drawn with the vision square; the drawing changes nothing."""
  def __init__(self):
    super().__init__()
    self._font: rl.Font = gui_app.font(FontWeight.BOLD)
    self._text: str = ""
    self._crossed: bool = False
    self._confidence: str = ""
    self._seen: bool = False

  def _update_state(self) -> None:
    sm = ui_state.sm
    alive = sm.alive['speedVisionSP']
    vision = sm['speedVisionSP']
    self._text, self._crossed = reading_text(vision.rawSignClass, vision.rawSignValue) if alive else ("", False)
    self._confidence = f"{round(vision.rawSignConfidence * 100)}%" if self._text else ""
    self._seen = alive and vision.rawSignSeen

  def _render(self, rect: rl.Rectangle) -> None:
    if not ui_state.speed_limit_vision_display or ui_state.sm.recv_frame["carState"] < ui_state.started_frame:
      return

    box = reading_box(rect, ui_state.speed_limit_final_display)
    rl.draw_rectangle_rounded(box, ROUNDNESS, SEGMENTS, DARK_GREY)
    rl.draw_rectangle_rounded_lines_ex(box, ROUNDNESS, SEGMENTS, BORDER, RED if self._seen else GREY)

    color = rl.WHITE if self._text else GREY
    if self._confidence:
      size = measure_text_cached(self._font, self._confidence, CONFIDENCE_FONT_SIZE)
      rl.draw_text_ex(self._font, self._confidence, rl.Vector2(box.x + (READING_SIZE - size.x) / 2, box.y + CONFIDENCE_TOP),
                      CONFIDENCE_FONT_SIZE, 0, color)

    text = self._text or "-"
    font_size = FONT_SIZES.get(len(text), FONT_SIZES[4])
    size = measure_text_cached(self._font, text, font_size)
    text_top = box.y + TEXT_TOP
    position = rl.Vector2(box.x + (READING_SIZE - size.x) / 2, text_top + (box.y + READING_SIZE - text_top - size.y) / 2)
    rl.draw_text_ex(self._font, text, position, font_size, 0, color)
    if self._crossed:
      middle = position.y + size.y / 2
      rl.draw_line_ex(rl.Vector2(position.x - 2, middle + size.y / 3), rl.Vector2(position.x + size.x + 2, middle - size.y / 3),
                      CROSS, color)
