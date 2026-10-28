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

RADIUS = 40
RING_WIDTH = 8
MARGIN = 12
FONT_SIZE = 40
FONT_SIZE_3_DIGITS = 32

RED = rl.Color(228, 0, 18, 255)
GREY = rl.Color(145, 155, 149, 255)
DARK_GREY = rl.Color(40, 40, 40, 230)


class VisionSpeedLimit(Widget):
  """The speed limit read by the speed-limits neural network (speedVisionSP), drawn in the top right corner when
  Show Vision Speed Limit is on or Speed Limit Source is Map Data + Vision. The drawing does not change the speed. A
  grey ring with a dash means no reading: the network is not sure, or speedvisiond is not running."""
  def __init__(self):
    super().__init__()
    self._font: rl.Font = gui_app.font(FontWeight.BOLD)
    self._speed_limit: float = 0.

  def _update_state(self) -> None:
    sm = ui_state.sm
    speed_limit = sm['speedVisionSP'].speedLimit if sm.alive['speedVisionSP'] else 0.
    self._speed_limit = max(0., speed_limit) * (CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH)

  def _render(self, rect: rl.Rectangle) -> None:
    if not ui_state.speed_limit_vision_display or ui_state.sm.recv_frame["carState"] < ui_state.started_frame:
      return

    center = rl.Vector2(rect.x + rect.width - MARGIN - RADIUS, rect.y + MARGIN + RADIUS)
    has_limit = round(self._speed_limit) > 0
    rl.draw_circle_v(center, RADIUS, RED if has_limit else GREY)
    rl.draw_circle_v(center, RADIUS - RING_WIDTH, rl.WHITE if has_limit else DARK_GREY)

    text = str(round(self._speed_limit)) if has_limit else "-"
    font_size = FONT_SIZE if len(text) < 3 else FONT_SIZE_3_DIGITS
    size = measure_text_cached(self._font, text, font_size)
    rl.draw_text_ex(self._font, text, rl.Vector2(center.x - size.x / 2, center.y - size.y / 2), font_size, 0,
                    rl.BLACK if has_limit else GREY)
