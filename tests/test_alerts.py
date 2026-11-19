"""Alert tests use fake transports; they never contact Supabase or Telegram."""
import io
import json
import threading
import time
import uuid
from email import policy
from email.parser import BytesParser

import pytest
from PIL import Image

from alerts import AlertConfig, AlertService, DeliveryError, annotate_photo, photo_form


@pytest.fixture
def config():
    return AlertConfig(enabled=True, supabase_url="https://example.supabase.co",
                       supabase_key="sb_secret_" + "test_only", bot_token="12345:" + "x" * 25,
                       chat_id="12345")


@pytest.fixture
def result():
    return {"width": 640, "height": 480, "model_sha256": "fixture-model",
            "timing": {"inference_ms": 12, "cloud_pipeline_ms": 15}, "detections": [
                {"label": "without_mask", "class_id": 1, "confidence": .9, "box": [50, 50, 180, 200]}]}


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (640, 480), "#cccccc").save(buffer, "JPEG")
    return buffer.getvalue()


class FakeTransport:
    def __init__(self, failure=None):
        self.calls, self.failure = [], failure

    def __call__(self, method, url, headers, body, timeout, service):
        self.calls.append((service, url, headers, body))
        if self.failure:
            self.failure(service, url)
        return b'{"ok":true,"result":{"message_id":42}}' if service == "telegram" else b""


def confirm(service, result, session=None):
    session = session or str(uuid.uuid4())
    outcomes = [service.observe(session, result, image_bytes(), True) for _ in range(3)]
    return session, outcomes[-1]


def test_opt_in_and_configuration_default_off(monkeypatch, result):
    monkeypatch.delenv("ALERTS_ENABLED", raising=False)
    config = AlertConfig.from_env()
    assert not config.enabled
    service = AlertService(config)
    assert service.observe(None, result, b"", True)["state"] == "disabled"
    monkeypatch.setenv("ALERTS_ENABLED", "1")
    for key in ["SUPABASE_URL", "SUPABASE_SECRET_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "API_KEY"]:
        monkeypatch.delenv(key, raising=False)
    config = AlertConfig.from_env()
    assert not config.enabled and "SUPABASE_URL" in config.error


def test_configuration_requires_auth_and_accepts_new_keys(monkeypatch, config):
    for key, value in {"ALERTS_ENABLED": "1", "SUPABASE_URL": config.supabase_url,
                       "SUPABASE_SECRET_KEY": config.supabase_key, "TELEGRAM_BOT_TOKEN": config.bot_token,
                       "TELEGRAM_CHAT_ID": config.chat_id, "API_KEY": "demo-api-key-at-least-16"}.items():
        monkeypatch.setenv(key, value)
    assert AlertConfig.from_env().enabled
    monkeypatch.setenv("API_KEY", "short")
    assert "API_KEY" in AlertConfig.from_env().error
    monkeypatch.setenv("API_KEY", "demo-api-key-at-least-16")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co:invalid")
    assert not AlertConfig.from_env().enabled


def test_temporal_votes_same_label_and_current_evidence(config, result):
    service = AlertService(config)
    session = str(uuid.uuid4())
    empty = {**result, "detections": []}
    assert service.observe(session, result, b"", False)["state"] == "paused"
    assert service.observe(session, result, b"", True)["state"] == "observing"
    assert service.observe(session, empty, b"", True)["state"] == "no_violation"
    assert service.observe(session, result, b"", True)["state"] == "queued"
    assert service.jobs.qsize() == 1
    # A current all-correct/empty image must not send a stale violation photo.
    assert service.observe(session, empty, b"", True)["state"] == "no_violation"


def test_different_violation_types_are_not_combined_into_two_votes(config, result):
    service = AlertService(config)
    session = str(uuid.uuid4())
    wrong = {**result, "detections": [{**result["detections"][0],
             "label": "mask_weared_incorrect", "class_id": 2}]}
    service.observe(session, result, b"", True)
    service.observe(session, {**result, "detections": []}, b"", True)
    assert service.observe(session, wrong, b"", True)["state"] == "observing"
    assert service.jobs.empty()


def test_session_isolation_expiry_and_global_cooldown(config, result):
    now = [0.0]
    service = AlertService(config, clock=lambda: now[0])
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    service.observe(a, result, b"", True)
    service.observe(a, result, b"", True)
    assert service.observe(b, result, b"", True)["state"] == "observing"
    now[0] = 9
    assert service.observe(a, result, b"", True)["state"] == "observing"
    service.observe(a, result, b"", True)
    assert service.observe(a, result, b"", True)["state"] == "queued"
    # Replacing the browser session does not bypass the global cooldown.
    _, decision = confirm(service, result, b)
    assert decision["state"] == "cooldown"
    now[0] = 40
    _, decision = confirm(service, result, b)
    assert decision["state"] == "queued"


def test_bounded_queue_does_not_claim_acceptance_when_full(config, result):
    now = [0.0]
    service = AlertService(config, clock=lambda: now[0])
    for value in [0, 31]:
        now[0] = value
        assert confirm(service, result)[1]["state"] == "queued"
    now[0] = 62
    assert confirm(service, result)[1]["state"] == "queue_full"
    assert len(service.events) == 2


def test_successful_delivery_uses_private_upload_and_direct_telegram_file(config, result):
    fake = FakeTransport()
    service = AlertService(config, transport=fake)
    session, decision = confirm(service, result)
    service._process(*service.jobs.get_nowait())
    event = service.snapshot(session)["recent"][0]
    assert event["id"] == decision["event_id"]
    assert (event["log_status"], event["storage_status"], event["telegram_status"]) == ("saved", "saved", "sent")
    assert event["telegram_message_id"] == 42 and event["telegram_ack_ms"] >= 0
    database = [call for call in fake.calls if "/rest/v1/" in call[1]]
    assert len(database) == 2
    assert json.loads(database[-1][3])["telegram_status"] == "sent"
    assert all("Authorization" not in call[2] for call in fake.calls if call[0] == "supabase")
    upload = next(call for call in fake.calls if "/storage/v1/object/" in call[1])
    assert "/public/" not in upload[1] and upload[2]["Content-Type"] == "image/jpeg"
    telegram = next(call for call in fake.calls if call[0] == "telegram")
    message = BytesParser(policy=policy.default).parsebytes(
        ("Content-Type: " + telegram[2]["Content-Type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + telegram[3])
    parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
    assert parts["chat_id"].get_payload(decode=True) == b"12345"
    assert "Không đeo" in parts["caption"].get_payload(decode=True).decode()
    assert parts["photo"].get_payload(decode=True).startswith(b"\xff\xd8")
    assert config.supabase_key not in json.dumps(service.snapshot())
    assert config.bot_token not in json.dumps(service.snapshot())


@pytest.mark.parametrize("failure_code,ambiguous,expected", [
    ("telegram_http_403", False, "failed"), ("telegram_http_429", False, "failed"),
    ("telegram_network_error", True, "unknown")])
def test_telegram_failure_persists_and_ambiguous_send_is_not_retried(config, result, failure_code, ambiguous, expected):
    def fail(service, url):
        if service == "telegram":
            raise DeliveryError(failure_code, ambiguous)
    fake = FakeTransport(fail)
    service = AlertService(config, transport=fake)
    confirm(service, result)
    service._process(*service.jobs.get_nowait())
    event = service.snapshot()["recent"][0]
    assert event["telegram_status"] == expected and event["log_status"] == "saved"
    assert event["telegram_message_id"] is None
    assert sum(call[0] == "telegram" for call in fake.calls) == 1
    assert json.loads(fake.calls[-1][3])["telegram_status"] == expected


def test_supabase_failure_does_not_suppress_telegram(config, result):
    def fail(service, url):
        if service == "supabase":
            raise DeliveryError("supabase_http_401")
    service = AlertService(config, transport=FakeTransport(fail))
    confirm(service, result)
    service._process(*service.jobs.get_nowait())
    event = service.snapshot()["recent"][0]
    assert (event["log_status"], event["storage_status"], event["telegram_status"]) == ("failed", "failed", "sent")


def test_image_encoding_failure_is_logged_without_sending(config, result):
    fake = FakeTransport()
    service = AlertService(config, transport=fake)
    _, decision = confirm(service, result)
    row, raw, detection, accepted_at = service.jobs.get_nowait()
    service._process(row, b"not-an-image", detection, accepted_at)
    event = service.snapshot()["recent"][0]
    assert event["id"] == decision["event_id"]
    assert event["log_status"] == "saved" and event["telegram_status"] == "failed"
    assert event["storage_status"] == "failed" and not any(call[0] == "telegram" for call in fake.calls)


def test_observe_does_not_wait_for_delivery_network(config, result):
    entered, release = threading.Event(), threading.Event()
    fake = FakeTransport()
    def slow(*args):
        entered.set()
        assert release.wait(2)
        return fake(*args)
    service = AlertService(config, transport=slow)
    service.start()
    try:
        _, decision = confirm(service, result)
        assert decision["state"] == "queued" and entered.wait(1)
        assert service.snapshot()["recent"][0]["telegram_status"] == "pending"
        release.set()
        deadline = time.monotonic()+2
        while service.snapshot()["recent"][0]["telegram_status"] == "pending" and time.monotonic() < deadline:
            time.sleep(.01)
        assert service.snapshot()["recent"][0]["telegram_status"] == "sent"
    finally:
        release.set()
        service.close()


def test_annotated_photo_and_multipart_unicode(result):
    photo = annotate_photo(image_bytes(), result)
    with Image.open(io.BytesIO(photo)) as image:
        assert image.size == (640, 480)
        # Rectangle is visibly red instead of the original gray.
        red, green, blue = image.getpixel((50, 100))
        assert red > green+40 and red > blue+40
    body, content_type = photo_form("123", "Cảnh báo đeo sai", photo)
    assert b"Content-Type: image/jpeg" in body and "Cảnh báo".encode() in body
    assert "multipart/form-data; boundary=" in content_type


def test_api_auth_session_and_notify_are_enforced(monkeypatch, tmp_path, result, config):
    import app
    from fastapi.testclient import TestClient
    class FakeDetector:
        size = 416
        def __init__(self, path):
            pass
        def predict(self, image):
            return result["detections"], {"inference_ms": 12, "preprocess_ms": 1, "postprocess_ms": 1}
    model = tmp_path / "model.onnx"
    model.write_bytes(b"stub-model-contract-only")
    monkeypatch.setenv("MODEL_PATH", str(model))
    monkeypatch.delenv("MODEL_URL", raising=False)
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    monkeypatch.setenv("API_KEY", "test-api-key-at-least-16")
    monkeypatch.setattr(app, "Detector", FakeDetector)
    # No background worker here: inspect queue acceptance directly.
    service = AlertService(config, transport=FakeTransport())
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(app, "AlertService", lambda cfg: service)
    headers = {"X-API-Key": "test-api-key-at-least-16", "X-Session-ID": str(uuid.uuid4())}
    with TestClient(app.app) as client:
        assert client.get("/alerts/status").status_code == 401
        assert client.get("/alerts/status", headers=headers).status_code == 200
        files = {"file": ("image.jpg", image_bytes(), "image/jpeg")}
        assert client.post("/predict", headers={**headers, "X-Session-ID": "bad"}, files=files).status_code == 400
        assert client.post("/predict", headers=headers, files=files).json()["alert"]["state"] == "paused"
        for _ in range(3):
            response = client.post("/predict", headers=headers, files=files, data={"notify": "true"})
        assert response.status_code == 200 and response.json()["alert"]["state"] == "queued"
        assert service.jobs.qsize() == 1
        cors = client.options("/predict", headers={"Origin": "http://localhost:5500",
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "x-session-id,x-api-key"})
        assert cors.status_code == 200
