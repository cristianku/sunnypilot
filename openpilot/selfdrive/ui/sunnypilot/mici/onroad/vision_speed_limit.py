"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.common.constants import CV
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

SIZE = 72
MARGIN = 12
ROUNDNESS = 0.3
SEGMENTS = 10
BORDER = 3
EYE_WIDTH = 24
EYE_HEIGHT = 17
EYE_TOP = 9
FONT_SIZE = 34
FONT_SIZE_3_DIGITS = 28
# How far up the square moves while the steering wheel shows in the bottom left corner: above it.
LIFT = 62

# Fuchsia, not the red and white of a road sign: the limit is read by the network, not a sign from the map or the car.
FUCHSIA = rl.Color(176, 38, 212, 165)  # see-through: the road shows under it
FUCHSIA_LIGHT = rl.Color(226, 120, 248, 255)
DARK_GREY = rl.Color(40, 40, 40, 165)
GREY = rl.Color(145, 155, 149, 255)


class VisionSpeedLimit(Widget):
  """The speed limit read by the speed-limits neural network (speedVisionSP), drawn in the bottom left corner when
  Show Vision Speed Limit is on or Speed Limit Source is Map Data + Vision: a fuchsia square with an eye and the limit
  in white, always visible (alerts are drawn at the top). The drawing does not change the speed. A dark square with a
  dash means no reading: the network is not sure, or speedvisiond is not running."""
  def __init__(self):
    super().__init__()
    self._font: rl.Font = gui_app.font(FontWeight.BOLD)
    self._eye: rl.Texture = gui_app.texture("icons_mici/onroad/eye_fill.png", EYE_WIDTH, EYE_HEIGHT)
    self._speed_limit: float = 0.
    self._lift: float = 0.

  def set_lift(self, lift: float) -> None:
    """0 in the corner, 1 above the steering wheel (while it shows); in between while it fades."""
    self._lift = min(1., max(0., lift))

  def _update_state(self) -> None:
    sm = ui_state.sm
    speed_limit = sm['speedVisionSP'].speedLimit if sm.alive['speedVisionSP'] else 0.
    self._speed_limit = max(0., speed_limit) * (CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH)

  def _render(self, rect: rl.Rectangle) -> None:
    if not ui_state.speed_limit_vision_display or ui_state.sm.recv_frame["carState"] < ui_state.started_frame:
      return

    box = rl.Rectangle(rect.x + MARGIN, rect.y + rect.height - MARGIN - SIZE - LIFT * self._lift, SIZE, SIZE)
    has_limit = round(self._speed_limit) > 0
    rl.draw_rectangle_rounded(box, ROUNDNESS, SEGMENTS, FUCHSIA if has_limit else DARK_GREY)
    rl.draw_rectangle_rounded_lines_ex(box, ROUNDNESS, SEGMENTS, BORDER, FUCHSIA_LIGHT if has_limit else FUCHSIA)

    eye_x = box.x + (SIZE - self._eye.width) / 2
    rl.draw_texture_v(self._eye, rl.Vector2(eye_x, box.y + EYE_TOP), rl.WHITE if has_limit else GREY)

    text = str(round(self._speed_limit)) if has_limit else "-"
    font_size = FONT_SIZE if len(text) < 3 else FONT_SIZE_3_DIGITS
    size = measure_text_cached(self._font, text, font_size)
    text_top = box.y + EYE_TOP + self._eye.height
    text_y = text_top + (box.y + SIZE - text_top - size.y) / 2
    rl.draw_text_ex(self._font, text, rl.Vector2(box.x + (SIZE - size.x) / 2, text_y), font_size, 0,
                    rl.WHITE if has_limit else GREY)
