from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class RunnerHealth:
  available: bool
  backend: str
  reason: str = ""


class SpeedVisionRunner(Protocol):
  def prepare(self) -> RunnerHealth: ...
  def submit_latest(self, frame: Any) -> bool: ...
  def poll(self) -> Any | None: ...
  def health(self) -> RunnerHealth: ...
  def close(self) -> None: ...


class UnavailableRunner:
  """Fail-closed runtime used when no verified model/backend exists."""

  def __init__(self, reason: str, backend: str = "unavailable") -> None:
    self._health = RunnerHealth(False, backend, reason)

  def prepare(self) -> RunnerHealth:
    return self._health

  def submit_latest(self, frame: Any) -> bool:
    return False

  def poll(self) -> None:
    return None

  def health(self) -> RunnerHealth:
    return self._health

  def close(self) -> None:
    return None


def load_runner(artifacts_lock: Path, *, requested_backend: str | None = None) -> SpeedVisionRunner:
  """Select only explicitly verified runtime backends.

  No mock/test backend is selectable from the on-road artifact lock. Until a
  target profile is implemented and verified, the loader returns a fail-closed
  UnavailableRunner rather than changing the driving runner or importing GPU
  frameworks opportunistically.
  """
  try:
    payload = json.loads(Path(artifacts_lock).read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return UnavailableRunner("artifact_lock_unreadable")

  model = payload.get("model")
  if not isinstance(model, dict) or model.get("state") != "ready":
    return UnavailableRunner("model_unavailable")

  if requested_backend is not None:
    profiles = model.get("backend_profiles")
    if not isinstance(profiles, list) or requested_backend not in profiles:
      return UnavailableRunner("unsupported_backend", backend=requested_backend)

  # Source bootstrap deliberately ships no on-road inference backend. Adding a
  # real backend requires an exact capability profile plus target verification.
  return UnavailableRunner("no_verified_runtime_backend", backend=requested_backend or "unavailable")
