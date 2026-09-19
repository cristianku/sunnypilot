from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SPEED_VISION = REPO / "openpilot/sunnypilot/speed_vision"


def test_operational_speed_limit_resolver_does_not_consume_vision() -> None:
  resolver = (REPO / "openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py").read_text(encoding="utf-8")
  assist = (REPO / "openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_assist.py").read_text(encoding="utf-8")
  assert "speedVisionState" not in resolver
  assert "speedVisionState" not in assist
  assert "SpeedLimitSource.vision" not in resolver
  assert "SpeedLimitSource.vision" not in assist


def test_operational_source_enum_remains_car_map_only() -> None:
  custom = (REPO / "openpilot/cereal/custom.capnp").read_text(encoding="utf-8")
  start = custom.index("enum Source {", custom.index("struct SpeedLimit {"))
  end = custom.index("}", start)
  enum = custom[start:end]
  assert "none @0;" in enum
  assert "car @1;" in enum
  assert "map @2;" in enum
  assert "vision" not in enum.lower()


def test_speed_vision_production_code_has_no_vehicle_control_surface() -> None:
  forbidden = [
    "sendcan", "CarControl", "carControl", "panda", "safetyParam",
    "SpeedLimitResolver", "SpeedLimitAssist", "cruiseControl.override",
  ]
  production = [
    p for p in SPEED_VISION.rglob("*.py")
    if "tests" not in p.parts and "_vendor" not in p.parts
  ]
  for path in production:
    text = path.read_text(encoding="utf-8")
    for token in forbidden:
      assert token not in text, (path, token)
