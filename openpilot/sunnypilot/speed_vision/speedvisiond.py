#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path

import openpilot.cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper

from .lifecycle import VisionSpeedLimitMode
from .publisher import build_unavailable_state, fill_cereal_state
from .runner import load_runner

SERVICE_HZ = 10.0
ARTIFACT_LOCK = Path(__file__).resolve().with_name("artifacts.lock.json")


def read_mode(params: Params) -> VisionSpeedLimitMode:
  try:
    return VisionSpeedLimitMode(int(params.get("VisionSpeedLimitMode", return_default=True)))
  except (TypeError, ValueError):
    return VisionSpeedLimitMode.OFF


def main() -> None:
  params = Params()
  if read_mode(params) == VisionSpeedLimitMode.OFF:
    return

  pm = messaging.PubMaster(["speedVisionState"])
  producer_epoch = time.monotonic_ns()
  session_id = f"speedvisiond-{producer_epoch}"
  sequence = 0
  rk = Ratekeeper(SERVICE_HZ, print_delay_threshold=None)
  runtime = load_runner(ARTIFACT_LOCK)
  health = runtime.prepare()

  try:
    while read_mode(params) != VisionSpeedLimitMode.OFF:
      now = time.monotonic_ns()

      # Bootstrap behavior is deliberately fail-closed. A real backend is
      # connected in the target-specific task after its artifacts/capability
      # profile and runtime budget have been verified.
      if not health.available:
        payload = build_unavailable_state(
          session_id=session_id,
          producer_epoch=producer_epoch,
          sequence=sequence,
          backend=health.backend,
          reason=health.reason,
          processed_mono_ns=now,
        )
        msg = messaging.new_message("speedVisionState")
        msg.valid = True
        fill_cereal_state(msg.speedVisionState, payload)
        pm.send("speedVisionState", msg)
        sequence += 1
        rk.keep_time()
        health = runtime.health()
        continue

      # No unverified backend is allowed to reach this path.
      payload = build_unavailable_state(
        session_id=session_id,
        producer_epoch=producer_epoch,
        sequence=sequence,
        backend=health.backend,
        reason="runtime_backend_not_connected",
        processed_mono_ns=now,
      )
      msg = messaging.new_message("speedVisionState")
      msg.valid = True
      fill_cereal_state(msg.speedVisionState, payload)
      pm.send("speedVisionState", msg)
      sequence += 1
      rk.keep_time()
      health = runtime.health()
  finally:
    runtime.close()


if __name__ == "__main__":
  main()
