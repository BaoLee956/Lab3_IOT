"""Unit and contract tests for Part 3: Latency Measurement (Local -> Cloud)."""
import csv
import io
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app
import summarize_latency
from detector import Detector, NAMES
from tests.test_contract import fixture_model


@pytest.fixture
def test_model_path(tmp_path):
    path = tmp_path / "latency_fixture.onnx"
    fixture_model(path)
    return path


def test_detector_latency_metrics(test_model_path):
    detector = Detector(test_model_path)
    detections, timing = detector.predict(Image.new("RGB", (640, 480)))
    assert [d["label"] for d in detections] == NAMES
    for metric in ["preprocess_ms", "inference_ms", "cloud_inference_ms", "postprocess_ms"]:
        assert metric in timing, f"Missing {metric} in detector timing"
        assert isinstance(timing[metric], (float, int))
        assert timing[metric] >= 0
    assert timing["cloud_inference_ms"] == timing["inference_ms"]


def test_api_latency_response_contract(monkeypatch, test_model_path):
    monkeypatch.setenv("MODEL_PATH", str(test_model_path))
    monkeypatch.setenv("API_KEY", "test-latency-key")
    monkeypatch.delenv("MODEL_URL", raising=False)
    monkeypatch.delenv("MODEL_SHA256", raising=False)

    image_buffer = io.BytesIO()
    Image.new("RGB", (640, 480)).save(image_buffer, format="JPEG")
    headers = {"X-API-Key": "test-latency-key"}

    with TestClient(app.app) as client:
        response = client.post(
            "/predict",
            headers=headers,
            files={"file": ("frame.jpg", image_buffer.getvalue(), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        # Check top-level cloud_inference_ms
        assert "cloud_inference_ms" in data
        assert isinstance(data["cloud_inference_ms"], (float, int))
        assert data["cloud_inference_ms"] >= 0

        # Check timing object
        timing = data["timing"]
        expected_timing_keys = [
            "preprocess_ms",
            "inference_ms",
            "cloud_inference_ms",
            "postprocess_ms",
            "decode_ms",
            "queue_ms",
            "cloud_pipeline_ms",
            "cloud_total_ms",
        ]
        for key in expected_timing_keys:
            assert key in timing, f"Missing key {key} in timing"
            assert isinstance(timing[key], (float, int)), f"{key} is not numeric"
            assert timing[key] >= 0, f"{key} is negative"

        # Check relationships
        assert timing["cloud_inference_ms"] == timing["inference_ms"]
        assert timing["cloud_pipeline_ms"] >= timing["cloud_inference_ms"]
        assert timing["cloud_total_ms"] >= timing["cloud_inference_ms"]


def test_summarizer_with_extended_latency_csv(tmp_path, monkeypatch):
    csv_file = tmp_path / "extended_latency.csv"
    fields = [
        "status", "phase", "error", "scene", "model_sha256", "detected_labels",
        "inference_ms", "rtt_ms", "e2e_ms", "cloud_pipeline_ms", "queue_ms",
        "cloud_inference_ms", "local_cloud_rtt_ms", "end_to_end_ms"
    ]
    rows = [
        {
            "status": "ok", "phase": "warmup", "error": "", "scene": "with_mask",
            "model_sha256": "sha256demo", "detected_labels": "with_mask",
            "inference_ms": "99.0", "rtt_ms": "150.0", "e2e_ms": "200.0",
            "cloud_pipeline_ms": "110.0", "queue_ms": "1.0",
            "cloud_inference_ms": "99.0", "local_cloud_rtt_ms": "150.0", "end_to_end_ms": "200.0"
        },
        {
            "status": "ok", "phase": "steady", "error": "", "scene": "with_mask",
            "model_sha256": "sha256demo", "detected_labels": "with_mask",
            "inference_ms": "15.5", "rtt_ms": "45.0", "e2e_ms": "60.0",
            "cloud_pipeline_ms": "20.0", "queue_ms": "0.5",
            "cloud_inference_ms": "15.5", "local_cloud_rtt_ms": "45.0", "end_to_end_ms": "60.0"
        },
        {
            "status": "ok", "phase": "steady", "error": "", "scene": "with_mask",
            "model_sha256": "sha256demo", "detected_labels": "with_mask",
            "inference_ms": "25.5", "rtt_ms": "55.0", "e2e_ms": "70.0",
            "cloud_pipeline_ms": "30.0", "queue_ms": "0.5",
            "cloud_inference_ms": "25.5", "local_cloud_rtt_ms": "55.0", "end_to_end_ms": "70.0"
        },
        {
            "status": "error", "phase": "steady", "error": "timeout", "scene": "with_mask",
            "model_sha256": "sha256demo", "detected_labels": "",
            "inference_ms": "", "rtt_ms": "", "e2e_ms": "",
            "cloud_pipeline_ms": "", "queue_ms": "",
            "cloud_inference_ms": "", "local_cloud_rtt_ms": "", "end_to_end_ms": ""
        }
    ]

    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    out_json = tmp_path / "summary_result.json"
    monkeypatch.setattr(sys, "argv", ["summarize_latency.py", str(csv_file), "--out", str(out_json)])
    summarize_latency.main()

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert summary["attempts_in_csv"] == 4
    assert summary["errors"] == 1
    assert summary["warmup_successes"] == 1
    assert summary["steady_successes"] == 2
    assert summary["metrics_ms"]["inference_ms"]["mean"] == 20.5
    assert summary["metrics_ms"]["rtt_ms"]["min"] == 45.0
    assert summary["metrics_ms"]["e2e_ms"]["max"] == 70.0
