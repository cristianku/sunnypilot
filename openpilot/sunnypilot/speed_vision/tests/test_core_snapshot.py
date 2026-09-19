from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "_vendor" / "speed_vision_core"
MANIFEST = ROOT / "_vendor" / "speed_vision_core.snapshot.json"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "valid_batch.json"

FORBIDDEN = {"torch", "numpy", "cv2", "onnx", "onnxruntime", "tinygrad", "cereal", "opendbc", "pandas", "cupy", "pycuda", "triton"}


def _sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _tree_digest(hashes: dict[str, str]) -> str:
  h = hashlib.sha256()
  for name in sorted(hashes):
    h.update(name.encode())
    h.update(b"\0")
    h.update(hashes[name].encode())
    h.update(b"\n")
  return h.hexdigest()


def test_snapshot_files_match_manifest() -> None:
  payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
  expected = {str(k): str(v) for k, v in payload["file_sha256"].items()}
  actual = {name: _sha256(CORE / name) for name in payload["files"]}
  assert actual == expected
  assert _tree_digest(actual) == payload["tree_digest"]


def test_snapshot_has_no_training_or_vehicle_framework_imports() -> None:
  for path in CORE.glob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
      names = []
      if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
      elif isinstance(node, ast.ImportFrom) and node.module:
        names = [node.module]
      for name in names:
        assert name.split(".", 1)[0] not in FORBIDDEN, (path.name, name)


def test_golden_batch_round_trip() -> None:
  from openpilot.sunnypilot.speed_vision._vendor.speed_vision_core import types
  payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
  batch = types.batch_from_dict(payload)
  assert types.batch_to_dict(batch) == payload
