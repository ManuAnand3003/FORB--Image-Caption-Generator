# 👁 VisionCaption — AI Image Captioning System

**IBM Innovation Centre for Education · AI & ML Major Project**

Generate natural language captions for any image using Salesforce BLIP (Vision Transformer + Causal Transformer). Upload photos, explore multiple caption candidates ranked by confidence, visualize model attention with heatmaps, or stream live webcam captions via WebSocket.

**Stack:** FastAPI + BLIP (HuggingFace) + WebSocket + Vanilla JS SPA  
**Model:** Salesforce/blip-image-captioning-large (470M params, ViT-L/16 + 12-layer decoder)  
**Aesthetic:** Editorial amber/cream on deep charcoal — distinct from typical tech UI.

---

## ✨ Core Features

| Feature | Details |
|---------|---------|
| 📸 **Image Upload** | Drag & drop or click to upload JPG/PNG/WEBP. Instant preview. |
| 🔭 **Beam Search** | Generate 3–5 caption candidates ranked by confidence. Deterministic, high-quality. |
| 🎯 **High Accuracy Mode** | Uses a wider beam and stronger decoding constraints for better caption quality. |
| 🎲 **Nucleus Sampling** | Temperature-based sampling for creative, diverse captions. Different each run. |
| ⚡ **Greedy Decode** | Fastest mode: single caption per image. Real-time constrained environments. |
| 🔥 **Attention Heatmap** | Extract ViT cross-attention, overlay on image. See where model "looked". |
| 💬 **Text Prompting** | Condition decoder with optional prefix (e.g., "a photograph of"). |
| 🌐 **Web Assist** | Optional internet-backed context lookup for extra detail when the network is available. |
| 📊 **Confidence Scores** | Log-probability normalized (0–1) per caption. |
| 📷 **Live Webcam** | WebSocket stream, captions every ~3s. Typewriter effect. Session log. |
| 🗂 **Session History** | Thumbnails + captions of all uploads. Click to replay. |
| 🎭 **Preview Mode** | Fallback when BLIP unavailable. Fast UI testing. |
| ⚠ **Progressive Loading** | Status indicator. Auto-retry with backoff. Graceful timeout + fallback. |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+, 4GB+ RAM (CPU) or GPU with CUDA (2x faster)
- ~2GB disk for model cache (downloads on first run)

### Installation & Run

```bash
cd VisionCaption
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

The browser opens automatically after the server is ready. If you want to disable that, set `OPEN_BROWSER=0` before running `python main.py`.
If you want the BLIP model preloaded before the UI appears, set `PRELOAD_MODEL=1`.

**First run:** Downloads BLIP-Large (~1.9 GB). Cached in `~/.cache/huggingface/`. Subsequent runs: instant.

---

## 📋 API Reference

### `POST /api/caption` — Caption with BLIP
Form: `file`, `mode` ("beam"|"accurate"|"sample"|"greedy"), `num_captions` (1–5), `prompt`, `max_tokens`, `web_assist`

`web_assist=true` is opt-in. When the network is available, the app will try to look up a short context snippet from Wikipedia and show it beside the caption results. If the lookup fails, the local caption still works normally.

### `POST /api/caption-preview` — Quick preview (no GPU)
Instant deterministic captions based on image dimensions.

### `GET /api/health` — Model status
Returns device, params, readiness.

### `WS /ws/stream` — Webcam captions
Client sends base64 JPEG → Server returns JSON caption.

### `GET /` — SPA frontend

---

## 🏗 Architecture

**Encoder (Vision Transformer):**
- Input: 384×384 RGB image
- Split into 576 patches (16×16 px each)
- 24 self-attention layers → 577 feature vectors (dim 1024)

**Decoder (Causal Language Model):**
- 12-layer transformer with cross-attention to image features
- Autoregressive: generates one token at a time
- Output: natural language caption

**Why Transformers?** Global self-attention captures long-range spatial relationships. CNNs have local receptive fields.

**Decoding Strategies:**
- **Beam Search:** k=5 beams, deterministic, high-quality
- **Nucleus Sampling:** Temperature=0.85, top_p=0.92, creative & diverse
- **Greedy:** Fastest, but repetitive

**Attention Heatmap:**
1. Extract ViT last-layer attention (24 heads × 577×577)
2. Average across heads
3. CLS token attention to patches (576,)
4. Reshape to 24×24 grid
5. Upsample to original resolution
6. JET colormap overlay (blue=low, red=high)

---

## 📊 Project Structure

```
VisionCaption/
├── main.py           # FastAPI (REST + WebSocket)
├── captioner.py      # BLIP inference engine
├── index.html        # SPA frontend
├── test_app.py       # Unit & integration tests
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🎭 Demo Script (30s)

1. Pre-load model before presentation
2. Upload complex scene (landscape/crowd)
3. **Beam search:** Show 3 candidates, explain scores
4. **Attention heatmap** 🔥 → "Red = model focus"
5. **Nucleus sampling:** Different output, explain diversity
6. **Webcam:** Point at audience/object → live captions
7. **Optional:** Text prompt to condition output

**Pitch:**
> "Our system generates natural language descriptions of images using BLIP, Salesforce's vision-language model. The encoder is a Vision Transformer—same tech as GPT-4V. It splits images into patches and applies self-attention globally. The decoder generates words one-at-a-time, cross-attending to image features. The heatmap shows which pixels influenced each word. We support beam search for quality, nucleus sampling for creativity, and greedy for speed. Live webcam streams captions every 3 seconds."

---

## 🧪 Testing

```bash
# Fast tests (skip BLIP load)
pytest test_app.py -v -m "not slow"

# Preview endpoint only
pytest test_app.py -k preview -v

# Full suite (with BLIP; slow)
pytest test_app.py -v
```

---

## 🐳 Docker

```bash
docker build -t visioncaption:latest .
docker run -p 8000:8000 visioncaption:latest

# With GPU
docker run --gpus all -p 8000:8000 visioncaption:latest
```

---

## 📈 Performance

| Scenario | Recommendation |
|----------|---|
| CPU-only | Greedy: ~5–10s/image |
| GPU (CUDA) | Beam search: ~1–2s/image |
| Mobile/Edge | Preview (instant) or greedy |
| Real-time | Webcam + greedy |

---

## 🔬 Model Info

| Attribute | Value |
|-----------|-------|
| **Model** | Salesforce/blip-image-captioning-large |
| **Encoder** | ViT-L/16 |
| **Decoder** | 12-layer causal Transformer |
| **Params** | ~470M |
| **Training Data** | 129M pairs (LAION + COCO + CC3M/CC12M) |
| **Input Resolution** | 384×384 px |
| **Vocabulary** | 30,522 tokens |
| **COCO CIDEr** | 136.7 (SoTA @ 2022 release) |

---

## 📝 Extend It

**Custom model:** Edit `_load()` in captioner.py:
```python
MODEL_ID = "Salesforce/blip2-opt-6.7b"  # BLIP-2
```

**Fine-tune:** Use HuggingFace Trainer + custom dataset

**Deploy:** AWS SageMaker, GCP Cloud Run, Azure Containers, or HF Spaces

**Auth:** Add API keys + rate limiting via FastAPI Depends

---

## 📚 References

1. Li et al. (2022). **BLIP: Bootstrapping Language-Image Pre-training...** ICML 2022. [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)
2. Dosovitskiy et al. (2020). **An Image is Worth 16×16 Words.** [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
3. [Salesforce BLIP on HuggingFace](https://huggingface.co/Salesforce/blip-image-captioning-large)

---

## 🤝 License

BLIP: HuggingFace hub (free for research & commercial). Transformers: MIT.

---

## 📧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model not found" | Check internet; cache saves to `~/.cache/huggingface/` |
| CUDA OOM | Use CPU or reduce max_tokens |
| WebSocket refused | Check firewall; ensure port 8000 open |
| Slow first inference | Normal; model pre-loads. Subsequent: cached in memory |
| Timeout | Frontend auto-falls back to preview mode |
| Web assist unavailable | Check connectivity. The local caption still works without internet. |

---

**Built with ❤️ for Education · 2026**
