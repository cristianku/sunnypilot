#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

speedvisiond: the speed limit read on the road camera by the speed-limits networks (speedVisionSP, m/s, 0 when there
is no reading), a few times a second. SignNet finds the signs in the top of the frame, the classifier reads each,
and a reading becomes the limit when two frames agree; it stays until another sign. A sign of the kind of road (a
motorway, a town, their ends) gives the road's general limit, and the limit of a motorway ramp ends once the car is
on the carriageway (vision.SpeedLimitMemory). It runs when Show Vision Speed Limit is on or Speed Limit Source is Map
Data + Vision.
"""
import os
os.environ['GMMU'] = '0'
import time
from typing import NoReturn

import numpy as np
from msgq.visionipc import VisionIpcClient
from openpilot.cereal import messaging
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.constants import CV
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.speed_limits import runner, vision

PERIOD = 0.2  # seconds between the frames read
NICENESS = 10  # the frame's preparation takes 100 ms on a little core: the driving processes come first


def idle(reason: str) -> NoReturn:
  cloudlog.warning(f"speedvisiond: {reason}, no reading")
  while True:
    time.sleep(60)


def main() -> None:
  os.nice(NICENESS)
  files = runner.networks()
  if files is None:
    idle("the models repository has no speed-limits networks")
  try:
    signnet = runner.Network(runner.compile_network(files.finder))
    classifier = runner.Network(runner.compile_network(files.classifier))
  except Exception:
    cloudlog.exception("speedvisiond: the networks did not compile or load")
    idle("no networks")

  # SignNet takes the GPU for about 20 ms, which the driving model shares at the same priority: it runs right after
  # the driving model has given its output, when the GPU is free until the next frame (or after 100 ms without one).
  sm = messaging.SubMaster(["modelV2"])

  def finder(x):
    sm.update(0)
    sm.update(100)
    return signnet(x)

  client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD, True)
  while not client.connect(False):
    time.sleep(0.1)
  cloudlog.warning(f"speedvisiond: {files.name} on the road camera, {client.width} x {client.height}")

  car = messaging.SubMaster(["carState"])  # apart from sm, whose update waits for the driving model
  pm = messaging.PubMaster(["speedVisionSP"])
  signnet_input = vision.SignNetInput(client.width, client.height)
  memory = vision.SpeedLimitMemory()
  next_frame = 0.0
  while True:
    buf = client.recv()
    if buf is None:
      continue
    now = time.monotonic()
    if now < next_frame:
      continue
    next_frame = now + PERIOD

    data = np.frombuffer(buf.data, dtype=np.uint8)  # buf.data is a memoryview of the camera's buffer
    y, uv = vision.nv12_planes(data, buf.width, buf.height, buf.stride, buf.uv_offset)
    signs = vision.read_signs(y, uv, signnet_input, finder, classifier, files.classes)
    car.update(0)
    speed_kph = car["carState"].vEgo * CV.MS_TO_KPH if car.seen["carState"] else None
    sure = [sign for sign in signs if sign["confidence"] >= vision.MIN_CONFIDENCE]
    speed_limit = memory.update(now, [sign["speed"] for sign in sure if sign["speed"] is not None],
                                [sign["class"] for sign in sure], speed_kph=speed_kph)

    msg = messaging.new_message("speedVisionSP", valid=True)
    msg.speedVisionSP.speedLimit = float(speed_limit * CV.KPH_TO_MS)
    msg.speedVisionSP.modelName = files.name
    msg.speedVisionSP.signRead = memory.sign_read
    pm.send("speedVisionSP", msg)


if __name__ == "__main__":
  main()
