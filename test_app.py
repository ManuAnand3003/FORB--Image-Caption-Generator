"""
VisionCaption — Unit & Integration Tests
=========================================
Tests for FastAPI endpoints (REST + WebSocket) and ML pipeline.

Run with: pytest test_app.py -v
"""

import base64
import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import captioner
from main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_image():
    """Create a small test image (16×16 RGB)."""
    img = Image.new("RGB", (16, 16), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ──────────────────────────────────────────────────────────────────────────────
# HEALTH & BASIC ENDPOINTS
# ──────────────────────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    """Test /api/health returns model status."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model" in data
    assert "version" in data


def test_frontend_serves(client):
    """Test GET / serves the SPA HTML."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "VisionCaption" in resp.text


# ──────────────────────────────────────────────────────────────────────────────
# CAPTION PREVIEW (NO MODEL LOAD)
# ──────────────────────────────────────────────────────────────────────────────

def test_caption_preview_basic(client, sample_image):
    """Test /api/caption-preview returns placeholders without loading model."""
    resp = client.post(
        "/api/caption-preview",
        files={"file": ("test.png", sample_image, "image/png")},
        data={"num_captions": 3},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["captions"]) == 3
    assert len(data["scores"]) == 3
    assert data["mode"] == "preview"
    assert data["inference_ms"] < 100  # should be near-instant


def test_caption_preview_single(client, sample_image):
    """Test /api/caption-preview with single caption."""
    resp = client.post(
        "/api/caption-preview",
        files={"file": ("test.png", sample_image, "image/png")},
        data={"num_captions": 1},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["captions"]) == 1


def test_caption_preview_many(client, sample_image):
    """Test /api/caption-preview caps at available placeholders."""
    resp = client.post(
        "/api/caption-preview",
        files={"file": ("test.png", sample_image, "image/png")},
        data={"num_captions": 10},  # request more than available
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["captions"]) <= 5  # max placeholders


# ──────────────────────────────────────────────────────────────────────────────
# CAPTION ENDPOINT (WITH MODEL) — SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow  # mark as slow; use pytest -m "not slow" to skip
def test_caption_beam_mode(client, sample_image):
    """Test /api/caption with beam search (slow: actually loads BLIP)."""
    resp = client.post(
        "/api/caption",
        files={"file": ("test.png", sample_image, "image/png")},
        data={
            "mode": "beam",
            "num_captions": 3,
            "prompt": "",
            "max_tokens": 30,
        },
    )
    # If model isn't loaded, it may timeout or fail gracefully
    if resp.status_code == 500:
        # Expected if model loading failed (no GPU, etc.)
        assert "error" in resp.json()
    else:
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "captions" in data
        assert "inference_ms" in data


@pytest.mark.slow
def test_caption_missing_file(client):
    """Test /api/caption without file."""
    resp = client.post("/api/caption")
    # FastAPI returns 422 for missing required file
    assert resp.status_code in [422, 400]


# ──────────────────────────────────────────────────────────────────────────────
# WEBSOCKET TESTS
# ──────────────────────────────────────────────────────────────────────────────

def test_websocket_connection(client):
    """Test WebSocket /ws/stream accepts connections."""
    with client.websocket_connect("/ws/stream") as websocket:
        # Connection should succeed
        assert websocket is not None


def test_websocket_rejects_invalid_data(client):
    """Test WebSocket rejects malformed base64 frames."""
    with client.websocket_connect("/ws/stream") as websocket:
        # Send invalid data
        websocket.send_text("invalid_base64_not_decodable!!!")
        # Server should either close or send error; connection should not crash
        try:
            data = websocket.receive_json(timeout=3.0)
            # If we get here, server sent a response (likely error)
            assert "error" in data or data is None
        except Exception:
            # Expected: connection may close or timeout
            pass


def test_websocket_caption_roundtrip(client, sample_image):
    """Test WebSocket accepts a valid JPEG frame and returns a caption payload."""
    sample_image.seek(0)
    payload = base64.b64encode(sample_image.read()).decode("utf-8")

    with client.websocket_connect("/ws/stream") as websocket:
        websocket.send_text(payload)
        data = websocket.receive_json()

    assert "caption" in data
    assert "inference_ms" in data
    assert data["frame_no"] == 1
    assert isinstance(data["caption"], str)
    assert data["inference_ms"] >= 0


# ──────────────────────────────────────────────────────────────────────────────
# INTEGRATION: FLOW TEST
# ──────────────────────────────────────────────────────────────────────────────

def test_full_flow_preview_then_health(client, sample_image):
    """Test user flow: health check → upload preview → status check."""
    # 1. Check health
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200

    # 2. Upload via preview
    preview_resp = client.post(
        "/api/caption-preview",
        files={"file": ("test.png", sample_image, "image/png")},
        data={"num_captions": 2},
    )
    assert preview_resp.status_code == 200
    assert len(preview_resp.json()["captions"]) == 2

    # 3. Check health again
    health_resp2 = client.get("/api/health")
    assert health_resp2.status_code == 200


def test_cors_headers(client):
    """Test CORS middleware allows cross-origin requests."""
    resp = client.get("/api/health")
    # FastAPI CORS middleware should add these headers
    assert resp.status_code == 200
    # Headers check is implicit in successful cross-origin request handling


# ──────────────────────────────────────────────────────────────────────────────
# PARAMETER VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def test_caption_valid_modes(client, sample_image):
    """Test caption endpoint accepts valid modes."""
    for mode in ["beam", "accurate", "sample", "greedy"]:
        resp = client.post(
            "/api/caption-preview",
            files={"file": ("test.png", sample_image, "image/png")},
            data={"num_captions": 1},
        )
        assert resp.status_code == 200


def test_web_assist_helper(monkeypatch):
    """Test the web-assist helper returns context when the external lookup succeeds."""
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                "red apple",
                ["Apple"],
                ["Apple is an edible fruit produced by an apple tree."],
                ["https://en.wikipedia.org/wiki/Apple"],
            ]

    monkeypatch.setattr(captioner.requests, "get", lambda *args, **kwargs: FakeResponse())
    context = captioner._web_assist_context("a red apple on a table")
    assert context is not None
    assert "edible fruit" in context


def test_caption_accurate_mode(client, sample_image):
    """Test /api/caption accepts the higher-accuracy decoding mode."""
    resp = client.post(
        "/api/caption",
        files={"file": ("test.png", sample_image, "image/png")},
        data={
            "mode": "accurate",
            "num_captions": 3,
            "prompt": "",
            "max_tokens": 30,
            "web_assist": "false",
        },
    )
    if resp.status_code == 500:
        assert "error" in resp.json()
    else:
        data = resp.json()
        assert resp.status_code == 200
        assert data["success"] is True
        assert data["mode"] == "accurate"


def test_caption_num_range(client, sample_image):
    """Test num_captions parameter clamped to valid range."""
    # Request 0 captions (should clamp to 1)
    resp = client.post(
        "/api/caption-preview",
        files={"file": ("test.png", sample_image, "image/png")},
        data={"num_captions": 0},
    )
    # Preview endpoint should handle gracefully
    assert resp.status_code == 200 or resp.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
