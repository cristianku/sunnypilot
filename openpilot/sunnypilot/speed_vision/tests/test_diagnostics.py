from __future__ import annotations

import json
from pathlib import Path

import pytest

from openpilot.sunnypilot.speed_vision.diagnostics import DiagnosticError, DiagnosticJournal


def test_diagnostics_default_disabled_writes_nothing(tmp_path: Path) -> None:
  journal = DiagnosticJournal(tmp_path, enabled=False, max_files=2)
  assert journal.append({"status": "ok"}) is None
  assert list(tmp_path.glob("*")) == []


def test_diagnostics_reject_raw_frame_like_fields(tmp_path: Path) -> None:
  journal = DiagnosticJournal(tmp_path, enabled=True, max_files=2)
  for key in ("pixels", "image", "frame_bytes", "raw_frame"):
    with pytest.raises(DiagnosticError, match="raw frame"):
      journal.append({key: "forbidden"})


def test_diagnostics_are_bounded_and_structured(tmp_path: Path) -> None:
  journal = DiagnosticJournal(tmp_path, enabled=True, max_files=2)
  for seq in range(4):
    journal.append({"sequence": seq, "health": {"available": False}})
  files = sorted(tmp_path.glob("*.json"))
  assert len(files) == 2
  payloads = [json.loads(p.read_text(encoding="utf-8")) for p in files]
  assert {p["sequence"] for p in payloads} == {2, 3}
