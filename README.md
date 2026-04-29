# 👁 VisionCaption — AI Image Captioning System
**IBM Innovation Centre for Education · AI & ML Major Project**

> Describe any image in natural English using a state-of-the-art CNN + Transformer pipeline.
> Built with FastAPI · BLIP · ViT-L/16 · WebSockets · Vanilla JS

---

## ✨ Features

| Feature | Details |
|---|---|
| 📸 **Image Upload** | Drag & drop any photo → instant natural language description |
| 🔭 **Beam Search** | Returns 3 caption candidates ranked by model confidence |
| 🎲 **Nucleus Sampling** | Temperature-based sampling for creative, diverse captions |
| 🔥 **Attention Heatmap** | Overlays ViT cross-attention on the image — see what the model "looked at" |
| 📷 **Live Webcam** | WebSocket stream with rolling captions every 3 seconds |
| 💬 **Text Prompting** | Condition the decoder with a text prefix (e.g. "a photograph of") |
| 📊 **Confidence Scores** | Log-probability normalized score per caption candidate |

---

## 🚀 Quick Start

```bash
# 1. Clone / download this folder
cd imagecap

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

Open **http://localhost:8000** in your browser.

> **Note:** First run downloads the BLIP-Large model (~1.9 GB) from HuggingFace.
> It is cached in `~/.cache/huggingface/`. Subsequent starts are instant.

---

## 🏗 Project Structure

```
imagecap/
│
├── main.py                 # FastAPI app — REST + WebSocket endpoints
├── engine/
│   ├── __init__.py
│   └── captioner.py        # Core ML: BLIP inference, beam search, attention heatmap
├── static/
│   └── index.html          # Full SPA frontend (zero build step)
├── requirements.txt
├── Dockerfile
└── README.md               # This file
```

---

## 🧠 How It Works — The Science

### Problem Statement
Given an image I, generate a natural language sentence C that describes its content:
```
C* = argmax P(C | I)
```

### Architecture: BLIP (Bootstrapping Language-Image Pre-training)
**Paper:** Li et al., 2022 — "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation"

#### 1. Image Encoder — Vision Transformer (ViT-L/16)
```
Input: 384×384 RGB image
  │
  ▼
Split into 576 patches (24×24 grid, 16×16 px each)
  │
  ▼
Linear projection → patch embeddings + positional encoding
  │
  ▼
[CLS] token + 576 patch tokens → self-attention layers (24 blocks)
  │
  ▼
Output: 577 feature vectors of dim 1024 (rich visual representations)
```

**Why Transformers over CNN?**
- CNNs have local receptive fields — they struggle to capture long-range relationships ("the dog on the LEFT side of the bridge in the BACKGROUND")
- ViT applies attention globally — every patch can attend to every other patch
- Result: richer contextual understanding of the full scene

#### 2. Text Decoder — Causal Language Model (BertLMHead)
```
Visual features (from encoder)
  │
  ▼  cross-attention
Token embeddings  →  [self-attn] → [cross-attn] → [FFN]  × 12 layers
  │                                    ↑
  │                          attends to image patches
  ▼
Softmax over 30,522 vocab tokens → P(next word | all previous words, image)
```

**Autoregressive generation:** At each step t, the model predicts one token:
```
P(w_t | w_1, ..., w_{t-1}, Image)
```
This continues until `[SEP]` token or max length.

#### 3. Decoding Strategies

**Beam Search (mode="beam")**
```
Maintains k=5 partial hypotheses simultaneously.
At each step, expand all k → k×V candidates → keep top k by total log-prob.
Final output: top 3 complete sequences by score/length normalized log-prob.

Pros:  Deterministic, high quality, grammatically sound
Cons:  Can produce generic / "safe" captions
```

**Nucleus Sampling (mode="sample")**
```
At each step, sort vocab by P(w_t | context).
Find smallest set S such that ΣP(w ∈ S) ≥ p (top_p=0.92).
Sample from S with temperature T=0.85.

Pros:  Creative, diverse — different result every run
Cons:  Occasionally produces unusual phrasing
```

**Greedy (mode="greedy")**
```
At each step: w_t = argmax P(w_t | context)
Fastest, but often produces repetitive / suboptimal captions.
```

#### 4. Attention Heatmap
```
ViT produces attention weights at every layer: shape (heads, N+1, N+1)

1. Extract last-layer attention: (24 heads, 577 tokens, 577 tokens)
2. Average across heads → (577, 577)
3. Take CLS token's row → [0, 1:] = (576,) attention to each patch
4. Reshape to (24, 24) spatial grid
5. Bilinear resize to original image resolution
6. Apply JET colormap (blue=low, red=high attention)
7. Blend 55% original + 45% heatmap
```

This answers: **"Where did the model look to generate this caption?"**

#### 5. Confidence Scores
For each generated sequence S = (w_1, ..., w_T):
```
log P(S | Image) = Σ log P(w_t | w_1,...,w_{t-1}, Image)
score = exp(log P(S) / T)   — length-normalized
```
Scores are normalized to [0, 1] for display.

---

## 📊 Model Information

| Attribute | Value |
|---|---|
| Model | Salesforce/blip-image-captioning-large |
| Encoder | Vision Transformer ViT-L/16 |
| Decoder | 12-layer causal Transformer |
| Parameters | ~470M |
| Training Data | 129M image-text pairs (LAION, COCO, CC3M, CC12M) |
| Input Resolution | 384×384 px |
| Vocabulary | 30,522 tokens (WordPiece) |
| COCO CIDEr | 136.7 (state-of-the-art at release) |

---

## 🌐 API Reference

### `POST /api/caption`
Caption an uploaded image.

**Form fields:**
| Field | Type | Default | Description |
|---|---|---|---|
| `file` | File | required | Image file |
| `mode` | str | `"beam"` | `"beam"` \| `"sample"` \| `"greedy"` |
| `num_captions` | int | `3` | Number of captions (1–5) |
| `prompt` | str | `""` | Optional text prefix |
| `max_tokens` | int | `60` | Max caption length |

**Response:**
```json
{
  "success": true,
  "captions": ["A dog sitting...", "A golden retriever...", "A fluffy dog..."],
  "scores": [0.82, 0.71, 0.63],
  "inference_ms": 1240,
  "image_size": "800×600",
  "mode": "beam",
  "attention_b64": "data:image/jpeg;base64,...",
  "model_info": {...}
}
```

### `WS /ws/stream`
WebSocket live webcam captioning.

```
Client → Server: base64-encoded JPEG string
Server → Client: { "caption": "...", "inference_ms": 1100, "frame_no": 5 }
```

### `GET /api/health`
System status and model info.

---

## 🎭 Demo Tips (For Maximum Audience Impact)

### Demo Flow
1. **Open the app before your presentation** and pre-load the model (first request caches it)
2. **Start with a vivid photo** — landscape, street scene, group of people
3. **Run beam search first** — show 3 different candidates, explain ranking
4. **Toggle the attention heatmap** — "the red areas are what the model focused on"
5. **Try nucleus sampling** on the same image — show how it generates different output
6. **Switch to webcam** — point at audience, at a prop, at yourself
7. **Use a prompt** — type "a painting of" and show how it conditions output

### 30-Second Explanation for Judges
> "Traditional image recognition tells you *what* is in a photo. Image captioning goes further
> — it generates a complete sentence in natural language, combining computer vision with NLP.
> Our system uses BLIP, which processes images through a Vision Transformer — the same
> technology behind GPT-4's vision capabilities. The model splits the image into patches,
> runs attention across them to understand spatial relationships, then feeds those features
> into a language decoder that generates words one at a time. The heatmap you see shows
> exactly which pixels influenced each word in the caption."

### Key Terms to Know
| Term | Meaning |
|---|---|
| **Cross-attention** | Mechanism allowing text decoder to "look at" image features while generating each word |
| **Patch embedding** | Converting each 16×16 image tile into a vector the transformer can process |
| **Beam search** | Keeping multiple partial hypotheses alive in parallel; picks the statistically best sequence |
| **Perplexity** | Measure of how "surprised" the model is by a sequence; lower = more confident |
| **COCO CIDEr** | Standard benchmark metric for image captioning quality |
| **Autoregressive** | Generating one token at a time, each conditioned on all previous tokens |

---

## 🔬 Extending This Project

```python
# 1. Visual Question Answering — ask questions about the image
from transformers import BlipForQuestionAnswering
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# 2. Image-Text Matching — check if a caption matches an image
from transformers import BlipForImageTextRetrieval

# 3. Fine-tune on specific domain (medical images, satellite imagery, art)
# The model architecture is the same — just swap the training data

# 4. Multilingual captions — translate output with Helsinki-NLP/opus-mt-en-LANG
from transformers import pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
hindi_caption = translator(english_caption)[0]["translation_text"]
```

---

## 📚 References

1. Li, J., Li, D., Xiong, C., & Hoi, S. (2022). **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.** ICML 2022. [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)
2. Dosovitskiy, A., et al. (2020). **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.** [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
3. Vinyals, O., et al. (2015). **Show and Tell: A Neural Image Caption Generator.** CVPR 2015. (Original CNN+RNN approach)
4. [Salesforce BLIP on HuggingFace](https://huggingface.co/Salesforce/blip-image-captioning-large)
5. [BLIP GitHub](https://github.com/salesforce/BLIP)

---

*IBM Innovation Centre for Education · Yenepoya Deemed to be University · AI & ML Major*
