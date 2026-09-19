from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


class DiagnosticError(ValueError):
  """Private diagnostics payload or storage policy is invalid."""


_RAW_FRAME_KEYS = {"pixels", "image", "frame_bytes", "raw_frame"}


def _contains_raw_frame_field(value: Any) -> bool:
  if isinstance(value, Mapping):
    for key, child in value.items():
      if str(key) in _RAW_FRAME_KEYS:
        return True
      if _contains_raw_frame_field(child):
        return True
  elif isinstance(value, (list, tuple)):
    return any(_contains_raw_frame_field(v) for v in value)
  return False


class DiagnosticJournal:
  """Explicit, bounded local JSON diagnostics.

  The caller chooses the root after privacy review. This module does not know
  uploader/logger paths and cannot enable itself.
  """

  def __init__(self, root: Path, *, enabled: bool, max_files: int = 100) -> None:
    if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0:
      raise DiagnosticError("max_files must be a positive integer")
    self.root = Path(root)
    self.enabled = bool(enabled)
    self.max_files = max_files
    self._sequence = 0

  def append(self, payload: Mapping[str, Any]) -> Path | None:
    if not self.enabled:
      return None
    if not isinstance(payload, Mapping):
      raise DiagnosticError("diagnostic payload must be a mapping")
    if _contains_raw_frame_field(payload):
      raise DiagnosticError("raw frame payloads are forbidden in diagnostics")

    self.root.mkdir(parents=True, exist_ok=True)
    sequence = self._sequence
    self._sequence += 1
    target = self.root / f"{sequence:020d}.json"
    temp = self.root / f".{sequence:020d}.{os.getpid()}.tmp"
    document = dict(payload)
    document.setdefault("sequence", sequence)
    temp.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temp, target)

    files = sorted(self.root.glob("*.json"))
    excess = len(files) - self.max_files
    for old in files[:max(0, excess)]:
      old.unlink()
    return target
