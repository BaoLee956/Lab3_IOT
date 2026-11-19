"""Persisted history and private photo regressions, with fake Supabase only."""
import io
import json
import re
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import pytest
from PIL import Image

from alerts import AlertConfig, AlertService, DeliveryError


class Database:
    def __init__(self, count=25):
        self.calls = []
        self.rows = [{"id": str(uuid.UUID(int=n+1)),
                      "occurred_at": (datetime(2026, 10, 13, tzinfo=timezone.utc)+timedelta(seconds=n//2)).isoformat(),
                      "session_id": str(uuid.UUID(int=(n % 3)+100)),
                      "violation_labels": ["without_mask"], "counts": {"without_mask": 1},
                      "storage_status": "saved", "telegram_status": "sent", "errors": [],
                      "image_path": "2026-10-13/"+str(uuid.UUID(int=n+1))+".jpg"}
                     for n in range(count)]
        photo = io.BytesIO()
        Image.new("RGB", (640, 480), "green").save(photo, "JPEG")
        self.photo = photo.getvalue()

    def __call__(self, method, url, headers, body, timeout, service):
        self.calls.append((method, url, headers))
        assert method == "GET" and body is None and service == "supabase"
        if "/storage/v1/object/authenticated/" in url:
            return self.photo
        params = parse_qs(urlsplit(url).query)
        assert "session_id" not in params  # all camera sessions, not just the current one
        rows = sorted(self.rows, key=lambda row: (row["occurred_at"], row["id"]), reverse=True)
        if "or" in params:
            match = re.fullmatch(r"\(occurred_at.lt.(.*),and\(occurred_at.eq.(.*),id.lt.(.*)\)\)", params["or"][0])
            assert match
            moment, same_moment, identifier = match.groups()
            assert moment == same_moment
            rows = [row for row in rows if (row["occurred_at"], row["id"]) < (moment, identifier)]
        if "id" in params:
            rows = [row for row in rows if row["id"] == params["id"][0][3:]]
        rows = rows[:int(params["limit"][0])]
        columns = params["select"][0].split(",")
        return json.dumps([{key: value for key, value in row.items() if key in columns} for row in rows]).encode()


@pytest.fixture
def database():
    return Database()


@pytest.fixture
def service(database):
    # Outgoing alerts OFF. Supabase history should still be available.
    config = AlertConfig(enabled=False, supabase_url="https://example.supabase.co",
                         supabase_key="sb_secret_test_only")
    return AlertService(config, transport=database)


def test_all_persisted_history_after_restart_and_across_sessions(service, database):
    assert not service.events
    identifiers, cursor, pages = [], None, []
    while True:
        result = service.read_history(10, cursor)
        pages.append(len(result["items"]))
        assert result["source"] == "supabase"
        assert all(row["log_status"] == "saved" for row in result["items"])
        identifiers.extend(row["id"] for row in result["items"])
        if not result["has_more"]:
            assert result["next_cursor"] is None
            break
        cursor = result["next_cursor"]
    assert pages == [10, 10, 5]
    assert len(identifiers) == len(set(identifiers)) == 25
    assert set(identifiers) == {row["id"] for row in database.rows}
    assert len({row["session_id"] for row in database.rows}) == 3
    assert all("Authorization" not in call[2] for call in database.calls)


def test_new_insert_does_not_shift_cursor_or_duplicate_old_rows(service, database):
    first = service.read_history(10)
    original_ids = {row["id"] for row in database.rows}
    database.rows.append({**database.rows[0], "id": str(uuid.uuid4()), "occurred_at": "2026-10-14T00:00:00+00:00"})
    second = service.read_history(10, first["next_cursor"])
    third = service.read_history(10, second["next_cursor"])
    ids = [row["id"] for page in [first, second, third] for row in page["items"]]
    assert len(ids) == len(set(ids)) == 25
    assert set(ids) == original_ids


@pytest.mark.parametrize("cursor", ["not-a-cursor!", "x"*513, "e30", "W10"])
def test_invalid_cursors_never_reach_database(service, database, cursor):
    with pytest.raises(ValueError):
        service.read_history(10, cursor)
    assert database.calls == []


def test_disabled_alerts_keep_read_credentials(monkeypatch):
    monkeypatch.setenv("ALERTS_ENABLED", "0")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_test_only")
    config = AlertConfig.from_env()
    assert not config.enabled and config.supabase_key == "sb_secret_test_only"
    assert config.supabase_url == "https://example.supabase.co"


def test_private_photo_resolved_by_record_id(service, database):
    record = database.rows[0]
    photo = service.read_image(record["id"])
    assert photo == database.photo
    assert len(database.calls) == 2
    assert "/storage/v1/object/authenticated/mask-alerts/" in database.calls[-1][1]
    assert "/public/" not in database.calls[-1][1]


@pytest.mark.parametrize("path", ["../../secret.jpg", "https://attacker.invalid/photo.jpg", "2026-10-13/not-the-event-id.jpg"])
def test_photo_path_cannot_escape_bucket_or_reference_another_record(service, database, path):
    database.rows[0]["image_path"] = path
    with pytest.raises(DeliveryError, match="alert_image_path_invalid"):
        service.read_image(database.rows[0]["id"])
    assert len(database.calls) == 1


def test_missing_image_and_invalid_upstream_data(service, database):
    database.rows[0]["image_path"] = None
    with pytest.raises(DeliveryError, match="alert_image_not_found"):
        service.read_image(database.rows[0]["id"])
    database.photo = b"not-a-jpeg"
    database.rows[0]["image_path"] = "2026-10-13/"+database.rows[0]["id"]+".jpg"
    with pytest.raises(DeliveryError, match="alert_image_invalid"):
        service.read_image(database.rows[0]["id"])


def test_bad_history_configuration_or_response_is_not_empty_success(service):
    broken = AlertService(replace(service.config, supabase_key=""))
    with pytest.raises(DeliveryError, match="history_not_configured"):
        broken.read_history()
    broken = AlertService(service.config, transport=lambda *args: b'{"error":"failed"}')
    with pytest.raises(DeliveryError, match="supabase_invalid_response"):
        broken.read_history()


def test_history_and_photo_routes_require_api_key(monkeypatch, tmp_path, service, database):
    import app
    from fastapi.testclient import TestClient
    class DetectorStub:
        size = 416
        def __init__(self, path):
            pass
    model = tmp_path / "stub.onnx"
    model.write_bytes(b"not-used-for-inference")
    monkeypatch.setenv("MODEL_PATH", str(model))
    monkeypatch.delenv("MODEL_URL", raising=False)
    monkeypatch.delenv("MODEL_SHA256", raising=False)
    monkeypatch.setenv("API_KEY", "history-test-key-16")
    monkeypatch.setattr(app, "Detector", DetectorStub)
    monkeypatch.setattr(app, "AlertService", lambda config: service)
    photo_path = "/alerts/"+database.rows[0]["id"]+"/image"
    headers = {"X-API-Key": "history-test-key-16"}
    with TestClient(app.app) as client:
        for route in ["/alerts/history", photo_path]:
            assert client.get(route).status_code == 401
            assert client.get(route, headers={"X-API-Key": "wrong"}).status_code == 401
        result = client.get("/alerts/history", headers=headers)
        assert result.status_code == 200 and len(result.json()["items"]) == 10
        assert client.get("/alerts/history?limit=51", headers=headers).status_code == 422
        assert client.get("/alerts/history?cursor=invalid", headers=headers).status_code == 400
        photo = client.get(photo_path, headers=headers)
        assert photo.status_code == 200 and photo.content == database.photo
        assert photo.headers["content-type"] == "image/jpeg"
        assert photo.headers["cache-control"] == "private, no-store"
        monkeypatch.delenv("API_KEY")
        assert client.get("/alerts/history").status_code == 503
        assert client.get(photo_path).status_code == 503
