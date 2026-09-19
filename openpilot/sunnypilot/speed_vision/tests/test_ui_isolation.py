from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def test_vision_widget_is_separate_from_operational_sla_renderer() -> None:
  hud = (REPO / "openpilot/selfdrive/ui/sunnypilot/onroad/hud_renderer.py").read_text(encoding="utf-8")
  sla = (REPO / "openpilot/selfdrive/ui/sunnypilot/onroad/speed_limit.py").read_text(encoding="utf-8")
  vision = (REPO / "openpilot/selfdrive/ui/sunnypilot/onroad/speed_vision.py").read_text(encoding="utf-8")
  state = (REPO / "openpilot/selfdrive/ui/sunnypilot/ui_state.py").read_text(encoding="utf-8")

  assert "VisionSpeedLimitRenderer" in hud
  assert "self.speed_vision_renderer.render(rect)" in hud
  assert "self.speed_limit_renderer.render(rect)" in hud
  assert "speedVisionState" in state
  assert "speedVisionState" in vision
  assert "speedVisionState" not in sla


def test_widget_labels_source_as_vision() -> None:
  vision = (REPO / "openpilot/selfdrive/ui/sunnypilot/onroad/speed_vision.py").read_text(encoding="utf-8")
  assert '"VISION"' in vision
