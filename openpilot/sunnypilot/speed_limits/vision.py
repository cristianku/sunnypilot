"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The speed-limits networks' inputs and outputs, in numpy: the road camera's NV12 frame to SignNet's band and the
classifier's crops, SignNet's boxes, the classifier's class to a speed, and the memory that turns the readings of
single frames into the limit shown. The networks were trained on frames that ffmpeg decoded to RGB (swscale's BT.601
limited range in 16-bit fixed point, each chroma sample on 2 x 2 pixels) and that PIL resized (bilinear,
antialiased, in 8-bit fixed point): the same arithmetic is done here, bit for bit, so the networks see on the car what
they saw in training.
"""
import math

import numpy as np

SIGNNET_WIDTH = 896
SIGNNET_HEIGHT = 288  # the top of the frame scaled to SIGNNET_WIDTH: 97 % of the signs end above row 420 of 760
SIGNNET_STRIDE = 8
MAX_BOXES = 5  # the classifier takes about 6 ms a box on the Adreno
MIN_BOX_SCORE = 0.3

CLASSIFIER_SIZE = 160
CLASSIFIER_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
CLASSIFIER_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
MIN_CROP_SIDE = 2
MIN_CONFIDENCE = 0.7

# A reading becomes the limit when it is in CONFIRM_FRAMES frames within CONFIRM_SECONDS; it stays until another does.
CONFIRM_FRAMES = 2
CONFIRM_SECONDS = 1.5

SPEED_KINDS = ("maximum_speed", "variable_display", "zone", "road_marking")
CANCELLATION = "cancellation"

# The general limits that hold where no number is signed (km/h; None: no general limit), by country, for the kind of
# road: in a town, outside towns, on an expressway (Autostrasse), on a motorway.
IMPLICIT_LIMITS = {
  "CH": {"town": 50, "rural": 80, "expressway": 100, "motorway": 120},
  "AT": {"town": 50, "rural": 100, "expressway": 100, "motorway": 130},
  "IT": {"town": 50, "rural": 90, "expressway": 110, "motorway": 130},
  "FR": {"town": 50, "rural": 80, "expressway": 110, "motorway": 130},
  "DE": {"town": 50, "rural": 100, "expressway": None, "motorway": None},
}
DEFAULT_COUNTRY = "CH"
# The signs that say which kind of road begins: their general limit replaces the limit held.
ROAD_SIGNS = {
  "other_sign:motorway_start": "motorway",
  "other_sign:motorway_end": "rural",
  "other_sign:expressway_start": "expressway",
  "other_sign:expressway_end": "rural",
  "other_sign:town_entry": "town",
  "other_sign:town_exit": "rural",
}
# A number read within RAMP_SECONDS of a motorway or expressway sign is the limit of the ramp: it ends when the car is
# on the main carriageway, which the road-type network sees (MERGE_SECONDS of "motorway" or "expressway" in a row) or,
# without it, the car tells (MERGE_SECONDS above the ramp's limit by MERGE_MARGIN km/h).
RAMP_SECONDS = 45.0
MERGE_SECONDS = 8.0
MERGE_MARGIN = 20.0
# Below this speed (km/h) the car is not on a carriageway (a jam, a toll): the time above the ramp limit starts again.
MERGE_MIN_SPEED = 70.0

PRECISION_BITS = 22  # PIL's fixed point for 8-bit images (Resample.c)


def _yuv_term(coefficient: int, offset: int) -> np.ndarray:
  """swscale's term of a channel for each 8-bit value (x86 yuv2rgb: the value * 8 less the offset, times the
  coefficient in 13-bit fixed point, the high 16 bits)."""
  return ((np.arange(256) * 8 - offset) * coefficient >> 16).astype(np.int16)


# BT.601 limited range: 255 / 219 for luma, 1.596, -0.392, -0.813 and 2.017 for chroma, times 8192.
LUMA = _yuv_term(9539, 16 * 8)
RED_V = _yuv_term(13075, 128 * 8)
GREEN_U = _yuv_term(-3209, 128 * 8)
GREEN_V = _yuv_term(-6660, 128 * 8)
BLUE_U = _yuv_term(16525, 128 * 8)


def resample_coefficients(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
  """PIL's bilinear resampling of in_size pixels to out_size (Resample.c precompute_coeffs and
  normalize_coeffs_8bpc): for each output pixel the input pixels it reads (out_size, taps) and their weights in
  fixed point. Taps of weight 0 are left out where they can be (they add nothing), the others have weight 0."""
  scale = in_size / out_size
  filterscale = max(scale, 1.0)
  support = filterscale  # the bilinear filter's support is 1
  ss = 1.0 / filterscale
  first, weights = [], []
  for xx in range(out_size):
    center = (xx + 0.5) * scale
    xmin = max(int(center - support + 0.5), 0)
    xmax = min(int(center + support + 0.5), in_size) - xmin
    k = [max(0.0, 1.0 - abs((x + xmin - center + 0.5) * ss)) for x in range(xmax)]
    ww = 0.0
    for w in k:
      ww += w
    fixed = [int(0.5 + (w / ww if ww != 0.0 else w) * (1 << PRECISION_BITS)) for w in k]
    used = [x for x, w in enumerate(fixed) if w != 0] or [0]
    first.append(xmin + used[0])
    weights.append(fixed[used[0]:used[-1] + 1])
  taps = max(len(w) for w in weights)
  index = np.minimum(np.array(first)[:, None] + np.arange(taps), in_size - 1)
  fixed = np.array([w + [0] * (taps - len(w)) for w in weights], dtype=np.int32)
  return index, fixed


class ColumnResampler:
  """PIL's resampling of the columns of uint8 planes (C, H, in_size) to out_size, fast where the scale is a ratio
  of small numbers p / q (1344 to 896 is 3 / 2): away from the edges every q-th output reads the same taps of every
  p-th input with the same weights, so the inputs are split by their place in p and each tap is one multiply-add of
  whole planes. The edges, and sizes without such a ratio, take the general way."""
  def __init__(self, in_size: int, out_size: int):
    self.in_size, self.out_size = in_size, out_size
    self.index, self.fixed = resample_coefficients(in_size, out_size)
    g = math.gcd(in_size, out_size)
    self.p, self.q = in_size // g, out_size // g
    self.phases: list[tuple[int, int, list[tuple[int, int, int]]]] = []  # (first, end, [(plane, offset, weight)])
    covered = np.zeros(out_size, dtype=bool)
    if self.q <= 4 and out_size >= 4 * self.q:
      cycles = out_size // self.q
      for r in range(self.q):
        middle = (cycles // 2) * self.q + r
        base = self.index[middle] - self.p * (cycles // 2)
        weights = self.fixed[middle]
        outputs = np.arange(r, out_size, self.q)
        m = (outputs - r) // self.q
        same = np.all((self.index[outputs] == self.p * m[:, None] + base) & (self.fixed[outputs] == weights), axis=1)
        taps = [(int(b) % self.p, int(b) // self.p, int(w)) for b, w in zip(base, weights, strict=True) if w != 0]
        # the longest run of matching outputs around the middle whose inputs are all inside the frame
        same &= np.all([(m + offset >= 0) & (m + offset < in_size // self.p) for _, offset, _ in taps], axis=0)
        first = end = cycles // 2
        while first > 0 and same[first - 1]:
          first -= 1
        while end < len(m) and same[end]:
          end += 1
        if end > first:
          self.phases.append((first, end, taps))
          covered[outputs[first:end]] = True
    if in_size % self.p:
      self.phases, covered[:] = [], False
    self.edges = np.flatnonzero(~covered)

  def __call__(self, planes: np.ndarray) -> np.ndarray:
    channels, height, width = planes.shape
    out = np.empty((channels, height, self.out_size), dtype=np.uint8)
    if self.phases:
      split = np.moveaxis(planes.reshape(channels, height, width // self.p, self.p), 3, 0).astype(np.int32)
      for r, (first, end, taps) in enumerate(self.phases):
        total = np.full((channels, height, end - first), 1 << (PRECISION_BITS - 1), dtype=np.int32)
        for plane, offset, weight in taps:
          total += split[plane, :, :, first + offset:end + offset] * weight
        total >>= PRECISION_BITS
        out[:, :, first * self.q + r:(end - 1) * self.q + r + 1:self.q] = np.minimum(total, 255)
    if len(self.edges) > self.out_size // 4:  # the general way: the columns as rows, whole rows are fast to gather
      out[:, :, self.edges] = transpose(resample_rows(transpose(planes), self.index[self.edges], self.fixed[self.edges]))
    elif len(self.edges):  # a few columns at the edges
      total = np.full((channels, height, len(self.edges)), 1 << (PRECISION_BITS - 1), dtype=np.int32)
      for tap in range(self.index.shape[1]):
        total += planes[:, :, self.index[self.edges, tap]] * self.fixed[self.edges, tap]
      out[:, :, self.edges] = np.minimum(total >> PRECISION_BITS, 255)
    return out


def resample_rows(planes: np.ndarray, index: np.ndarray, fixed: np.ndarray) -> np.ndarray:
  """One pass of PIL's resampling of uint8 planes (C, H, W) along their rows (axis 1: whole rows, the fast way in
  numpy)."""
  total = np.full((planes.shape[0], index.shape[0], planes.shape[2]), 1 << (PRECISION_BITS - 1), dtype=np.int32)
  for tap in range(index.shape[1]):
    total += planes[:, index[:, tap]] * fixed[:, tap].reshape(1, -1, 1)
  return np.minimum(total >> PRECISION_BITS, 255).astype(np.uint8)


def transpose(planes: np.ndarray) -> np.ndarray:
  return np.ascontiguousarray(planes.transpose(0, 2, 1))


def resize(planes: np.ndarray, width: int, height: int) -> np.ndarray:
  """PIL's Image.resize((width, height), BILINEAR) of uint8 planes (C, H, W): columns first, then rows."""
  if planes.shape[2] != width:
    planes = transpose(resample_rows(transpose(planes), *resample_coefficients(planes.shape[2], width)))
  if planes.shape[1] != height:
    planes = resample_rows(planes, *resample_coefficients(planes.shape[1], height))
  return planes


def nv12_planes(data: np.ndarray, width: int, height: int, stride: int, uv_offset: int) -> tuple[np.ndarray, np.ndarray]:
  """Copies of the luma (height, width) and the interleaved chroma (height / 2, width) of an NV12 buffer: the
  camera may write the buffer again while the networks read it."""
  y = data[:stride * height].reshape(height, stride)[:, :width].copy()
  uv = data[uv_offset:uv_offset + stride * (height // 2)].reshape(height // 2, stride)[:, :width].copy()
  return y, uv


def yuv_to_rgb(y: np.ndarray, uv: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
  """The RGB planes (uint8, 3 x rows x columns) of rows top..bottom and columns left..right, as ffmpeg decodes the
  road camera's video to the frames of the dataset: each chroma sample on its 2 x 2 pixels, each channel the sum of
  swscale's terms."""
  # The pairs of rows that hold the region (pairs, 2, columns): each chroma row, its terms on 2 columns, added to both.
  r0, c0, r1, c1 = top // 2, left // 2, (bottom + 1) // 2, (right + 1) // 2
  u, v = uv[r0:r1, 2 * c0:2 * c1:2], uv[r0:r1, 2 * c0 + 1:2 * c1:2]
  luma = LUMA[y[2 * r0:2 * r1, 2 * c0:2 * c1]].reshape(r1 - r0, 2, 2 * (c1 - c0))
  pairs = np.empty((3, r1 - r0, 2, 2 * (c1 - c0)), dtype=np.uint8)
  for plane, term in zip(pairs, (RED_V[v], GREEN_U[u] + GREEN_V[v], BLUE_U[u]), strict=True):
    np.clip(luma + np.repeat(term, 2, axis=1)[:, None, :], 0, 255, out=plane, casting="unsafe")
  dy, dx = top - 2 * r0, left - 2 * c0
  return pairs.reshape(3, 2 * (r1 - r0), 2 * (c1 - c0))[:, dy:dy + bottom - top, dx:dx + right - left]


class SignNetInput:
  """SignNet's input for frames of one size: the frame scaled to SIGNNET_WIDTH and its top SIGNNET_HEIGHT rows,
  as float 0..1 (1, 3, SIGNNET_HEIGHT, SIGNNET_WIDTH)."""
  def __init__(self, width: int, height: int):
    self.width, self.height = width, height
    self.scale = width / SIGNNET_WIDTH  # a band pixel in frame pixels
    self.columns = ColumnResampler(width, SIGNNET_WIDTH) if width != SIGNNET_WIDTH else None
    rows_index, rows_fixed = resample_coefficients(height, round(height * SIGNNET_WIDTH / width))
    self.rows = rows_index[:SIGNNET_HEIGHT], rows_fixed[:SIGNNET_HEIGHT]
    self.used_rows = int(self.rows[0].max()) + 1

  def band(self, y: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """The band as uint8 planes (3, SIGNNET_HEIGHT, SIGNNET_WIDTH)."""
    planes = yuv_to_rgb(y, uv, 0, self.used_rows, 0, self.width)
    if self.columns is not None:
      planes = self.columns(planes)
    return resample_rows(planes, *self.rows)

  def __call__(self, y: np.ndarray, uv: np.ndarray) -> np.ndarray:
    pixels = self.band(y, uv)[None].astype(np.float32)
    return np.divide(pixels, 255.0, out=pixels)


def signnet_boxes(output: np.ndarray, scale: float, limit: int = MAX_BOXES, min_score: float = MIN_BOX_SCORE) -> list[dict]:
  """SignNet's boxes (frame pixels, the best first): the local maxima (3 x 3) of the heat map of its output
  (1, 5, H, W: heat, log width, log height, offset x, offset y) of at least min_score, at most limit."""
  heat, shape = output[0, 0], output[0, 1:5]
  padded = np.pad(heat, 1, constant_values=-np.inf)
  height, width = heat.shape
  pooled = np.max([padded[dy:dy + height, dx:dx + width] for dy in range(3) for dx in range(3)], axis=0)
  peaks = np.where(pooled == heat, heat, 0.0).ravel()
  boxes = []
  for index in np.argsort(-peaks, kind="stable")[:limit]:
    score = float(peaks[index])
    if score < min_score:
      break
    iy, ix = divmod(int(index), width)
    log_w, log_h, ox, oy = (float(value) for value in shape[:, iy, ix])
    w, h = math.exp(min(log_w, 8.0)), math.exp(min(log_h, 8.0))
    cx, cy = (ix + ox) * SIGNNET_STRIDE, (iy + oy) * SIGNNET_STRIDE
    box = [(cx - w / 2) * scale, (cy - h / 2) * scale, (cx + w / 2) * scale, (cy + h / 2) * scale]
    boxes.append({"box": box, "score": score})
  return boxes


def crop_rectangle(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
  """The pixels of a box to cut from a frame of width x height: rounded outwards, inside the frame, at least
  MIN_CROP_SIDE pixels on each side (nnslr_tools.training.crop_rectangle)."""
  x1, y1, x2, y2 = (float(value) for value in box)
  left, top = math.floor(min(x1, x2)), math.floor(min(y1, y2))
  right, bottom = math.ceil(max(x1, x2)), math.ceil(max(y1, y2))

  def widen(low: int, high: int, limit: int) -> tuple[int, int]:
    side = min(MIN_CROP_SIDE, limit)
    if high - low < side:
      middle = (low + high) / 2
      low = int(math.floor(middle - side / 2))
      high = low + side
    low = max(0, min(low, limit - side))
    high = min(limit, max(high, low + side))
    return low, high

  left, right = widen(max(0, left), min(int(width), right), int(width))
  top, bottom = widen(max(0, top), min(int(height), bottom), int(height))
  return left, top, right, bottom


def classifier_input(y: np.ndarray, uv: np.ndarray, box: list[float]) -> np.ndarray:
  """The classifier's input for a box: the box cut from the full frame, resized to CLASSIFIER_SIZE and normalised,
  float (1, 3, CLASSIFIER_SIZE, CLASSIFIER_SIZE)."""
  left, top, right, bottom = crop_rectangle(box, y.shape[1], y.shape[0])
  crop = resize(yuv_to_rgb(y, uv, top, bottom, left, right), CLASSIFIER_SIZE, CLASSIFIER_SIZE)
  pixels = crop.astype(np.float32) / 255.0
  return ((pixels - CLASSIFIER_MEAN) / CLASSIFIER_STD)[None]


def softmax(logits: np.ndarray) -> np.ndarray:
  exp = np.exp(logits - logits.max())
  return exp / exp.sum()


def class_speed(name: str) -> float | None:
  """The speed (km/h) a class gives: its number for a speed sign, a variable display, a zone or a marking on the
  road; 0 for the end of a limit; None for the others (other signs, unreadable, not a sign)."""
  if name == CANCELLATION:
    return 0.0
  kind, _, value = name.partition(":")
  if kind in SPEED_KINDS and value.isdigit():
    return float(value)
  return None


def read_signs(y: np.ndarray, uv: np.ndarray, signnet_input: SignNetInput, finder, classifier,
               classes: list[str]) -> list[dict]:
  """The signs of a frame: SignNet's boxes, each read by the classifier (its class, how sure, and the speed it gives
  when it is sure enough: km/h, 0 for the end of a limit, None for no speed). finder and classifier take the
  networks' inputs and give their outputs as numpy arrays."""
  signs = signnet_boxes(finder(signnet_input(y, uv)), signnet_input.scale)
  for sign in signs:
    probabilities = softmax(classifier(classifier_input(y, uv, sign["box"])).reshape(-1))
    index = int(np.argmax(probabilities))
    sign["class"], sign["confidence"] = classes[index], float(probabilities[index])
    sign["speed"] = class_speed(classes[index]) if sign["confidence"] >= MIN_CONFIDENCE else None
  return signs


class SpeedLimitMemory:
  """The limit shown (km/h, 0: none). A number becomes the limit when it is read in CONFIRM_FRAMES frames within
  CONFIRM_SECONDS, and stays until another one is. A sign of the kind of road (ROAD_SIGNS), confirmed the same way,
  sets the general limit of that road (IMPLICIT_LIMITS of the country); so does the end of a limit, on a road whose
  kind is known (else 0, no limit).

  The limit of a ramp ends on the main carriageway, where no sign ends it: a number read within RAMP_SECONDS of a
  motorway or expressway sign is taken as the ramp's, and once the car is on the carriageway (road_type from the
  road-type network, or the car's speed when there is none) the general limit of the road comes back."""
  def __init__(self, country: str = DEFAULT_COUNTRY):
    self.speed_limit = 0.0
    self.country = country if country in IMPLICIT_LIMITS else DEFAULT_COUNTRY
    self.road: str | None = None  # the kind of road from the last road sign or the road-type network
    self._seen: list[tuple[float, float | str]] = []
    self._road_sign_at = -math.inf  # when the last motorway or expressway sign was confirmed
    self._ramp_limit = False  # the limit held is a ramp's
    self._merge_since: float | None = None
    self.sign_read = False  # a number or a road sign was confirmed in the last update (the limit held again too)

  def _confirmed(self, now: float, values: list) -> list:
    self._seen = [(t, value) for t, value in self._seen if now - t <= CONFIRM_SECONDS]
    self._seen += [(now, value) for value in set(values)]
    counts: dict = {}
    for _, value in self._seen:
      counts[value] = counts.get(value, 0) + 1
    confirmed = [value for value in set(values) if counts[value] >= CONFIRM_FRAMES]
    # Two at once (a gantry, a side road): the one seen most, the lower number on a tie.
    return sorted(confirmed, key=lambda value: (-counts[value], value if isinstance(value, float) else 0.0))

  def _general(self, road: str | None) -> float:
    limit = IMPLICIT_LIMITS[self.country].get(road) if road else None
    return float(limit or 0)

  def update(self, now: float, readings: list[float], signs: list[str] | tuple[str, ...] = (), road_type: str | None = None,
             speed_kph: float | None = None) -> float:
    """readings: the speeds read in this frame (class_speed); signs: the classes of the other signs read sure enough;
    road_type: what the road-type network sees (town, rural, expressway, motorway, motorway_ramp...), None without it;
    speed_kph: the car's speed, None when unknown."""
    roads = [ROAD_SIGNS[name] for name in signs if name in ROAD_SIGNS]
    confirmed = self._confirmed(now, [float(value) for value in readings] + roads)
    numbers = [value for value in confirmed if isinstance(value, float)]
    road_signs = [value for value in confirmed if isinstance(value, str)]
    self.sign_read = bool(confirmed)

    if road_signs:
      self.road = road_signs[0]
      self.speed_limit = self._general(self.road)
      self._ramp_limit = False
      if self.road in ("motorway", "expressway"):
        self._road_sign_at = now
    if road_type in ("town", "rural", "expressway", "motorway") and not road_signs:
      self.road = road_type
    if numbers:
      value = numbers[0]
      if value == 0:
        self.speed_limit = self._general(self.road)
        self._ramp_limit = False
      else:
        self.speed_limit = value
        on_ramp = road_type is not None and road_type.endswith("_ramp")
        self._ramp_limit = on_ramp or (now - self._road_sign_at <= RAMP_SECONDS and road_type is None)
      self._merge_since = None
      return self.speed_limit

    if self._ramp_limit:
      if road_type is not None:
        merged = road_type in ("motorway", "expressway")
      else:
        merged = (speed_kph is not None and speed_kph >= MERGE_MIN_SPEED
                  and speed_kph >= self.speed_limit + MERGE_MARGIN)
      if not merged:
        self._merge_since = None
      elif self._merge_since is None:
        self._merge_since = now
      elif now - self._merge_since >= MERGE_SECONDS:
        road = road_type if road_type in ("motorway", "expressway") else (self.road or "motorway")
        self.road = road
        self.speed_limit = self._general(road)
        self._ramp_limit = False
        self._merge_since = None
    return self.speed_limit
