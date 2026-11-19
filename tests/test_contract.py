"""Infrastructure tests use a synthetic ONNX fixture, never a trained mask model."""
import io
from pathlib import Path

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from detector import Detector, NAMES, decode, letterbox
from prepare_data import group_items, split_groups


def fixture_model(path, names=NAMES):
    # Three spatially separate detections + an overlapping lower-score duplicate.
    raw = np.array([[[80, 208, 335, 81], [208, 208, 208, 208],
                     [50, 50, 50, 50], [60, 60, 60, 60],
                     [.9, .05, .05, .7], [.05, .9, .05, .1], [.05, .05, .9, .1]]], np.float32)
    graph = helper.make_graph([helper.make_node("Constant", [], ["output"],
        value=numpy_helper.from_array(raw))], "synthetic_contract_test",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 416, 416])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 7, 4])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    helper.set_model_props(model, {"names": repr(dict(enumerate(names)))})
    onnx.save(model, str(path))


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "fixture.onnx"
    fixture_model(path)
    return path


def test_non_square_preprocess():
    tensor, ratio, left, top = letterbox(Image.new("RGB", (640, 480), (255, 0, 0)), 416)
    assert tensor.shape == (1, 3, 416, 416) and tensor.dtype == np.float32
    assert (ratio, left, top) == (.65, 0, 52)
    assert tensor[0, 0, 100, 100] == 1 and tensor[0, 2, 100, 100] == 0


def test_real_onnx_runtime_and_three_labels(model_path):
    detector = Detector(model_path)
    detections, timing = detector.predict(Image.new("RGB", (640, 480)))
    assert [d["label"] for d in detections] == NAMES
    assert len(detections) == 3  # lower-confidence overlapping duplicate removed
    assert all(0 <= d["box"][0] < d["box"][2] <= 640 for d in detections)
    assert all(v >= 0 for v in timing.values())


def test_reject_wrong_model_classes(tmp_path):
    path = tmp_path / "wrong.onnx"
    fixture_model(path, ["cat", "dog", "person"])
    with pytest.raises(ValueError, match="Wrong model labels"):
        Detector(path)


def test_empty_frame_outputs():
    raw = np.zeros((1, 7, 4), np.float32)
    assert decode(raw, 640, 480, .65, 0, 52) == []
    with pytest.raises(ValueError, match="Expected raw"):
        decode(np.zeros((1, 300, 6)), 640, 480, .65, 0, 52)


def test_api_errors_auth_and_timing(monkeypatch, model_path):
    import app
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("API_KEY", "contract-test-key")
    monkeypatch.delenv("MODEL_URL", raising=False)
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    image = io.BytesIO()
    Image.new("RGB", (640, 480)).save(image, format="JPEG")
    headers = {"X-API-Key": "contract-test-key"}
    with TestClient(app.app) as client:
        assert client.get("/health").json()["classes"] == NAMES
        assert client.get("/").status_code == 200
        assert client.post("/predict", files={"file": ("x.jpg", image.getvalue(), "image/jpeg")}).status_code == 401
        assert client.post("/predict", headers=headers,
                           files={"file": ("x.txt", b"bad", "text/plain")}).status_code == 415
        assert client.post("/predict", headers=headers,
                           files={"file": ("x.jpg", b"bad", "image/jpeg")}).status_code == 400
        assert client.post("/predict", headers=headers,
                           files={"file": ("x.jpg", b"x" * (2*1024*1024+1), "image/jpeg")}).status_code == 413
        result = client.post("/predict", headers=headers,
                             files={"file": ("x.jpg", image.getvalue(), "image/jpeg")})
        assert result.status_code == 200
        data = result.json()
        assert (data["width"], data["height"]) == (640, 480)
        assert len(data["detections"]) == 3
        assert data["timing"]["cloud_pipeline_ms"] >= data["timing"]["inference_ms"]
        cors = client.options("/predict", headers={"Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-api-key"})
        assert cors.status_code == 200


def test_checksum_mismatch(monkeypatch, model_path):
    from app import load_model_file
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_SHA256", "0" * 64)
    monkeypatch.delenv("MODEL_URL", raising=False)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        load_model_file()


def test_group_split_keeps_duplicates_and_sessions():
    items = [{"path": Path(f"{i}.png"), "sha256": str(i), "dhash": i*65537,
              "counts": [int(i % 3 == c) for c in range(3)]} for i in range(36)]
    items[1]["sha256"] = items[0]["sha256"]
    groups, pairs = group_items(items, 0)
    assert pairs and any(0 in g and 1 in g for g in groups)
    splits = split_groups(items, groups)
    assert len(splits) == len(groups) and set(splits) == {0, 1, 2}


def test_latency_summary_excludes_warmup_and_errors(tmp_path, monkeypatch):
    import csv
    import json
    import sys
    import summarize_latency
    path = tmp_path / "latency.csv"
    fields = ["status", "phase", "error", "scene", "model_sha256", "detected_labels",
              "inference_ms", "rtt_ms", "e2e_ms", "cloud_pipeline_ms", "queue_ms"]
    base = {"status": "ok", "phase": "steady", "error": "", "scene": "with_mask",
            "model_sha256": "test", "detected_labels": "with_mask", "inference_ms": "10",
            "rtt_ms": "30", "e2e_ms": "40", "cloud_pipeline_ms": "15", "queue_ms": "0"}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([base, {**base, "inference_ms": "20"},
                         {**base, "phase": "warmup", "inference_ms": "999"},
                         {**base, "status": "error", "error": "timeout"}])
    output = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["summarize_latency.py", str(path), "--out", str(output)])
    summarize_latency.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["metrics_ms"]["inference_ms"]["mean"] == 15
    assert result["steady_successes"] == 2 and result["error_rate"] == .25
