"""
VisionCaption — FastAPI Backend
================================
REST endpoints for image captioning + WebSocket for live webcam stream.

Endpoints:
  GET  /                    → Frontend SPA
  POST /api/caption         → Caption uploaded image (beam / sample / greedy)
  WS   /ws/stream           → Real-time webcam captioning
  GET  /api/health          → System status + model info
"""

import base64
import io
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from engine import caption_image, caption_frame, model_info as get_model_info

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VisionCaption API",
    description="Image captioning powered by BLIP (CNN + Transformer)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ─── Startup: pre-load model ──────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """
    Pre-load the BLIP model at startup so the first request isn't slow.
    In production, this runs once when the container boots.
    """
    print("[VisionCaption] Pre-loading BLIP model at startup…")
    try:
        from engine.captioner import _load
        _load()
    except Exception as e:
        print(f"[VisionCaption] Startup pre-load failed (will retry on first request): {e}")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html = Path("static/index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.post("/api/caption")
async def api_caption(
    file:         UploadFile = File(...),
    mode:         str        = Form(default="beam"),
    num_captions: int        = Form(default=3),
    prompt:       str        = Form(default=""),
    max_tokens:   int        = Form(default=60),
):
    """
    Caption an uploaded image.

    Form fields:
      file          — image file (JPG/PNG/WEBP)
      mode          — "beam" | "sample" | "greedy"
      num_captions  — 1–5
      prompt        — optional conditioning text (e.g. "a photograph of")
      max_tokens    — max caption length in tokens
    """
    try:
        raw = await file.read()
        result = caption_image(
            image_bytes   = raw,
            num_captions  = min(max(num_captions, 1), 5),
            mode          = mode,
            prompt        = prompt.strip(),
            max_new_tokens= min(max(max_tokens, 20), 120),
        )
        return {"success": True, **result}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    """
    WebSocket endpoint for live webcam captioning.

    Protocol:
      Client → Server: base64-encoded JPEG frame  (optionally prefixed "data:image/jpeg;base64,")
      Server → Client: JSON { caption, inference_ms, frame_no }
    """
    await websocket.accept()
    frame_no = 0

    try:
        while True:
            data = await websocket.receive_text()

            # Strip data URL prefix if present
            if "," in data:
                data = data.split(",", 1)[1]

            img_bytes = base64.b64decode(data)
            nparr     = np.frombuffer(img_bytes, np.uint8)
            frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            frame_no += 1
            result = caption_frame(frame)

            await websocket.send_json({
                "caption":      result["caption"],
                "inference_ms": result["inference_ms"],
                "frame_no":     frame_no,
            })

    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@app.get("/api/health")
async def health():
    return {
        "status":  "ok",
        "model":   get_model_info(),
        "version": "1.0.0",
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
