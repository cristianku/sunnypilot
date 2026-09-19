from __future__ import annotations

import pyray as rl

from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.onroad.hud_renderer import UI_CONFIG
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

KM_TO_MILE = 0.621371
WIDTH = 190
HEIGHT = 92


class VisionSpeedLimitRenderer(Widget):
  """Separate advisory-only VISION badge. Never replaces the SLA badge."""

  def __init__(self) -> None:
    super().__init__()
    self.visible = False
    self.available = False
    self.usable = False
    self.has_value = False
    self.value_kph = 0
    self.state = "unavailable"
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi = gui_app.font(FontWeight.SEMI_BOLD)

  def update(self) -> None:
    sm = ui_state.sm
    self.visible = bool(sm.seen["speedVisionState"])
    if not self.visible:
      return

    state = sm["speedVisionState"]
    self.available = bool(state.health.available)
    self.usable = bool(state.hypothesis.usableForAdvisory)
    self.has_value = bool(state.hypothesis.hasValue)
    self.value_kph = int(state.hypothesis.valueKph) if self.has_value else 0
    self.state = str(state.hypothesis.state)

  def _render(self, rect: rl.Rectangle) -> None:
    if not self.visible:
      return

    set_width = UI_CONFIG.set_speed_width_metric if ui_state.is_metric else UI_CONFIG.set_speed_width_imperial
    x = rect.x + 60 + set_width + 30 + UI_CONFIG.set_speed_width_metric + 35
    y = rect.y + 52
    box = rl.Rectangle(x, y, WIDTH, HEIGHT)

    bg = rl.Color(0, 0, 0, 165)
    border = rl.Color(120, 120, 120, 180)
    value_color = rl.Color(150, 150, 150, 255)
    if self.available and self.usable and self.has_value:
      border = rl.Color(80, 190, 220, 220)
      value_color = rl.WHITE
    elif self.available:
      border = rl.Color(180, 160, 80, 200)

    rl.draw_rectangle_rounded(box, 0.25, 8, bg)
    rl.draw_rectangle_rounded_lines_ex(box, 0.25, 8, 3, border)

    label = "VISION"
    label_size = 28
    label_w = measure_text_cached(self._font_semi, label, label_size).x
    rl.draw_text_ex(self._font_semi, label, rl.Vector2(x + (WIDTH - label_w) / 2, y + 8), label_size, 0, border)

    if self.has_value:
      value = self.value_kph if ui_state.is_metric else round(self.value_kph * KM_TO_MILE)
      text = str(value)
    else:
      text = "---"
    value_size = 45
    text_w = measure_text_cached(self._font_bold, text, value_size).x
    rl.draw_text_ex(self._font_bold, text, rl.Vector2(x + (WIDTH - text_w) / 2, y + 39), value_size, 0, value_color)
