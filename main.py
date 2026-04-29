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
from contextlib import asynccontextmanager
from fastapi.concurrency import run_in_threadpool

from captioner import caption_image, caption_frame, model_info as get_model_info


# ─── App & Lifespan ───────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler that pre-loads the model without blocking the event loop."""
    print("[VisionCaption] Pre-loading BLIP model at startup (threaded)…")
    try:
        from captioner import _load
        # Load model in a thread to avoid blocking the event loop
        await run_in_threadpool(_load)
    except Exception as e:
        print(f"[VisionCaption] Startup pre-load failed (will retry on first request): {e}")
    yield


app = FastAPI(
    title="VisionCaption API",
    description="Image captioning powered by BLIP (CNN + Transformer)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html = Path("index.html").read_text(encoding="utf-8")
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
        # Run synchronous ML inference in a thread to avoid blocking the event loop
        result = await run_in_threadpool(
            caption_image,
            raw,
            min(max(num_captions, 1), 5),
            mode,
            prompt.strip(),
            min(max(max_tokens, 20), 120),
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
            # Run synchronous frame captioning in threadpool
            result = await run_in_threadpool(caption_frame, frame)

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


@app.post("/api/caption-preview")
async def api_caption_preview(
    file:         UploadFile = File(...),
    num_captions: int        = Form(default=3),
):
    """
    Quick preview: return placeholder captions without loading BLIP model.
    Useful for testing UI, fallback when GPU unavailable, or demo mode.

    Returns generic captions based on detected image properties (size, content hint).
    """
    try:
        import io
        from PIL import Image
        
        raw = await file.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = pil.size
        
        # Generate deterministic placeholders based on image dimensions
        aspect = w / h if h > 0 else 1.0
        placeholders = [
            f"A scene with dimensions {w}×{h} pixels, captured with artistic composition.",
            f"An image in {'landscape' if aspect > 1.2 else 'portrait' if aspect < 0.8 else 'square'} orientation, rich in detail and color.",
            f"A photograph showcasing interesting visual elements and atmospheric lighting.",
            f"A vibrant image full of texture and visual interest, {w}px wide.",
            f"An evocative scene captured with careful attention to composition.",
        ]
        
        selected = placeholders[:min(num_captions, len(placeholders))]
        return {
            "success": True,
            "captions": selected,
            "scores": [0.88 - i*0.05 for i in range(len(selected))],
            "inference_ms": 2,  # instant
            "image_size": f"{w}×{h}",
            "mode": "preview",
            "attention_b64": None,
            "model_info": {"status": "preview_mode", "note": "Placeholder captions for UI testing"},
        }
    
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


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
