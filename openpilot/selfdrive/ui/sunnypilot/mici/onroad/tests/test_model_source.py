from types import SimpleNamespace

import pyray as rl
import pytest

from openpilot.selfdrive.ui.sunnypilot.mici.onroad import hud_renderer
from openpilot.selfdrive.ui.mici.onroad import hud_renderer as factory
from openpilot.selfdrive.ui.sunnypilot.mici.onroad.hud_renderer import HudRendererSP
from openpilot.selfdrive.ui.sunnypilot.mici.onroad.vision_speed_limit import vision_box
from openpilot.selfdrive.ui.ui_state import ChestnutState

RECT = rl.Rectangle(0, 0, 500, 240)
WIDE = SimpleNamespace(width=75, height=44)  # the orange icon, the widest


def _icon_at(monkeypatch, *, auto=False, final=False, vision=False) -> rl.Rectangle:
  """Where the factory drawing puts the GPU icon, with these on the screen."""
  drawn = []
  state = SimpleNamespace(chestnut_state=ChestnutState.FAILED, speed_limit_final_display=final,
                          speed_limit_vision_display=vision)
  monkeypatch.setattr(hud_renderer, 'ui_state', state)
  monkeypatch.setattr(factory, 'ui_state', state)
  monkeypatch.setattr(factory.rl, 'get_time', lambda: 0.)
  monkeypatch.setattr(factory.rl, 'draw_texture_ex', lambda icon, pos, *_: drawn.append(pos))

  hud = HudRendererSP.__new__(HudRendererSP)  # the place needs no window
  hud._auto_speed = auto
  hud._txt_wheel = SimpleNamespace(width=50, height=50)
  hud._txt_chestnut_orange = WIDE
  hud._chestnut_icon = None
  hud._chestnut_fade_time = 0.
  hud._chestnut_alpha_filter = SimpleNamespace(update=lambda visible: 1.)
  hud._draw_model_source(RECT)
  return rl.Rectangle(drawn[0].x, drawn[0].y, WIDE.width, WIDE.height)


def _overlap(a: rl.Rectangle, b: rl.Rectangle) -> bool:
  return a.x < b.x + b.width and b.x < a.x + a.width and a.y < b.y + b.height and b.y < a.y + a.height


@pytest.mark.parametrize('shown', [{'auto': True, 'vision': True}, {'vision': True}, {'final': True}, {'final': True, 'vision': True}])
def test_no_square_goes_over_the_gpu_icon(monkeypatch, shown):
  icon = _icon_at(monkeypatch, **shown)
  assert not _overlap(icon, vision_box(RECT))
  assert not _overlap(icon, vision_box(RECT, True))


def test_the_gpu_icon_stays_above_the_word_smart(monkeypatch):
  icon = _icon_at(monkeypatch, auto=True, vision=True)
  box = vision_box(RECT)
  assert icon.y == box.y
  assert icon.y + icon.height <= box.y + box.height - 28  # the capitals of SMART, 26 px, stand on the square's bottom


def test_without_squares_the_icon_stays_where_it_was(monkeypatch):
  icon = _icon_at(monkeypatch)
  assert (icon.x, icon.y) == (RECT.width - 10 - WIDE.width, RECT.height - 14 - (50 + WIDE.height) / 2)
