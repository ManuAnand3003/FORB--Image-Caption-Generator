"""
VisionCaption — ML Captioning Engine
======================================
Uses Salesforce BLIP (Bootstrapping Language-Image Pre-training).

Architecture recap for your presentation:
  - ENCODER: Vision Transformer (ViT-L/16) splits image into 16×16 patches,
    runs self-attention across them → rich visual feature vectors.
  - DECODER: Causal language transformer attends to those features via
    cross-attention, then generates words one by one (autoregressive).
  - TRAINING: Contrastive + captioning + matching losses on 129M image-text pairs.

Why BLIP over just CNN+LSTM?
  - Transformers capture global context (not just local CNN receptive fields).
  - Pre-trained on internet-scale data → zero-shot captioning quality.
  - Same architecture powers GPT-4V, Gemini, Claude's vision.
"""

import io
import time
import warnings
from pathlib import Path
from typing import Optional
import re

import numpy as np
import requests
from PIL import Image, ImageOps

warnings.filterwarnings("ignore")
warnings.filterwarnings(
    "ignore",
    message=r"numpy\.core is deprecated and has been renamed to numpy\._core.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)


# ══════════════════════════════════════════════════════════════════════════════
# Lazy imports — avoids crashing at startup if GPU/CPU is loading
# ══════════════════════════════════════════════════════════════════════════════
_model    = None
_processor = None
_device   = None


def _load():
    """Load BLIP model once, cache globally. Thread-safe via FastAPI startup."""
    global _model, _processor, _device
    if _model is not None:
        return

    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[VisionCaption] Device: {_device}")
    print(f"[VisionCaption] Loading BLIP large model (~1.9GB first run)…")

    t0 = time.time()
    MODEL_ID = "Salesforce/blip-image-captioning-large"

    _processor = BlipProcessor.from_pretrained(MODEL_ID)
    _model     = BlipForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
    ).to(_device)
    _model.eval()

    print(f"[VisionCaption] Model ready in {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _pil_from_bytes(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_from_frame(frame_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_rgb)


# ══════════════════════════════════════════════════════════════════════════════
# Core Inference
# ══════════════════════════════════════════════════════════════════════════════

def caption_image(
    image_bytes: bytes,
    num_captions: int = 3,
    mode: str = "beam",          # "beam" | "sample" | "greedy" | "accurate"
    prompt: str = "",             # optional text prompt to condition on
    max_new_tokens: int = 60,
    web_assist: bool = False,
) -> dict:
    """
    Generate captions for an uploaded image.

    Args:
        image_bytes:   Raw file bytes (JPEG / PNG / WEBP)
        num_captions:  How many caption candidates to return
        mode:          "beam"   → best-first beam search (precise, consistent)
                       "sample" → nucleus sampling (creative, diverse)
                       "greedy" → single greedy decode (fastest)
        prompt:        Optional text prefix to condition the decoder
        max_new_tokens: Max words in output caption

    Returns:
        dict with keys: captions, scores, inference_ms, model_info, attention_b64
    """
    _load()

    import torch

    pil = _pil_from_bytes(image_bytes)
    t0  = time.perf_counter()

    # ── Preprocess ─────────────────────────────────────────────────────────
    inputs = _processor(
        images=pil,
        text=prompt if prompt else None,
        return_tensors="pt",
    ).to(_device)

    # ── Generate ───────────────────────────────────────────────────────────
    with torch.no_grad():
        if mode == "accurate":
            out = _model.generate(
                **inputs,
                num_beams=max(num_captions, 8),
                num_return_sequences=num_captions,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                length_penalty=1.05,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                output_scores=True,
                return_dict_in_generate=True,
            )
        elif mode == "beam":
            out = _model.generate(
                **inputs,
                num_beams=max(num_captions, 5),
                num_return_sequences=num_captions,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                length_penalty=1.2,     # slightly favor longer captions
                repetition_penalty=1.3,
                output_scores=True,
                return_dict_in_generate=True,
            )
        elif mode == "sample":
            out = _model.generate(
                **inputs,
                do_sample=True,
                temperature=0.85,
                top_p=0.92,
                num_return_sequences=num_captions,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.3,
                output_scores=True,
                return_dict_in_generate=True,
            )
        else:  # greedy
            out = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    # ── Decode ─────────────────────────────────────────────────────────────
    captions = _processor.batch_decode(out.sequences, skip_special_tokens=True)
    captions = [c.strip().capitalize() for c in captions]

    # Compute sequence-level log-probability scores for ranking
    scores = _compute_scores(out, inputs)

    # ── Attention heatmap ──────────────────────────────────────────────────
    attn_b64 = _attention_heatmap(pil, inputs)
    web_context = _web_assist_context(captions[0], prompt) if web_assist else None

    return {
        "captions":     captions,
        "scores":       scores,
        "inference_ms": elapsed_ms,
        "image_size":   f"{pil.width}×{pil.height}",
        "mode":         mode,
        "attention_b64": attn_b64,
        "web_context":  web_context,
        "model_info":   {
            "name":     "BLIP-Large",
            "encoder":  "ViT-L/16 (Vision Transformer)",
            "decoder":  "Causal Transformer (BertLMHead)",
            "device":   _device,
            "params":   "470M",
        },
    }


def caption_frame(frame_bgr: np.ndarray, prompt: str = "") -> dict:
    """
    Fast single greedy caption for a live webcam frame.
    Skips attention and multi-beam for speed.
    """
    _load()
    import torch

    pil = _pil_from_frame(frame_bgr)
    t0  = time.perf_counter()

    inputs = _processor(
        images=pil,
        text=prompt if prompt else None,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        out = _model.generate(
            **inputs,
            num_beams=3,
            max_new_tokens=40,
            repetition_penalty=1.3,
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    caption    = _processor.decode(out[0], skip_special_tokens=True).strip().capitalize()

    return {
        "caption":      caption,
        "inference_ms": elapsed_ms,
    }


def _web_assist_context(caption: str, prompt: str = "") -> Optional[str]:
    """Try to enrich a caption with a short external context snippet.

    This is intentionally opt-in and best-effort: if the network is unavailable
    or the query is too generic, the function returns None.
    """

    stopwords = {
        "a", "an", "and", "are", "at", "be", "been", "being", "but", "for",
        "from", "in", "into", "is", "it", "near", "of", "on", "or", "the",
        "this", "that", "to", "with", "woman", "man", "person", "people",
        "scene", "image", "photo", "photograph", "picture", "there", "has",
        "have", "standing", "sitting", "looking", "front", "background",
    }
    text = f"{prompt} {caption}".strip().lower()
    tokens = [token for token in re.findall(r"[a-z][a-z0-9'-]{2,}", text) if token not in stopwords]
    query = " ".join(tokens[:6]).strip() or caption.strip()
    if not query:
        return None

    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "opensearch",
                "search": query,
                "limit": 1,
                "namespace": 0,
                "format": "json",
            },
            timeout=3.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        summaries = payload[2] if isinstance(payload, list) and len(payload) > 2 else []
        if summaries and summaries[0]:
            context = str(summaries[0]).strip()
            return context[:220].rsplit(" ", 1)[0] + "..." if len(context) > 220 else context
    except Exception as e:
        print(f"[WebAssist] {e}")

    return None
# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

def _compute_scores(out, inputs) -> list[float]:
    """
    Compute a normalized confidence score per generated sequence.
    Uses the mean log-probability of generated tokens (excluding prompt).

    Why log-prob? The model outputs a probability distribution over vocabulary
    at each step. The joint probability of a sequence = product of step probs.
    We use log-sum for numerical stability, then normalize by length.
    """
    import torch

    if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
        raw_scores = out.sequences_scores.detach().float().cpu().tolist()
        if len(raw_scores) == 1:
            return [1.0]
        low = min(raw_scores)
        high = max(raw_scores)
        if abs(high - low) < 1e-8:
            return [0.5] * len(raw_scores)
        return [round((score - low) / (high - low), 4) for score in raw_scores]

    if not hasattr(out, "scores") or not out.scores:
        return [1.0] * len(out.sequences)

    try:
        # Stack scores: shape (seq_len, batch*beams, vocab_size)
        stacked = torch.stack(out.scores, dim=0)   # (T, B, V)
        log_probs = torch.log_softmax(stacked, dim=-1)  # per-step log dist

        prompt_len = inputs.get("input_ids", None)
        prompt_len = prompt_len.shape[-1] if prompt_len is not None else 0

        scores = []
        for i, seq in enumerate(out.sequences):
            if i >= log_probs.shape[1]:
                scores.append(0.5)
                continue
            # Gather log-prob of actually chosen token at each step
            generated = seq[prompt_len:]  # remove prompt tokens when available
            lp = 0.0
            for t, tok in enumerate(generated):
                if t >= log_probs.shape[0]:
                    break
                lp += log_probs[t, i % log_probs.shape[1], tok].item()
            length = max(len(generated), 1)
            # Normalize by length, map to [0,1]
            norm = lp / length
            score = float(np.clip(np.exp(norm), 0, 1))
            scores.append(round(score, 4))
        return scores
    except Exception as e:
        print(f"[Scoring] {e}")
        return [round(1.0 / (i + 1), 3) for i in range(len(out.sequences))]


# ══════════════════════════════════════════════════════════════════════════════
# Attention Heatmap
# ══════════════════════════════════════════════════════════════════════════════

def _attention_heatmap(pil: Image.Image, inputs: dict) -> Optional[str]:
    """
    Extract cross-attention from the ViT encoder's last layer and overlay
    on the original image as a colour heatmap.

    How it works:
    - ViT splits image into N patches (e.g. 196 patches for 224×224 with 16×16 patch)
    - Plus 1 [CLS] token that aggregates global image meaning
    - The attention matrix (H×N×N per layer) shows how much each patch
      attends to every other patch
    - We take the CLS→patch attention = "which patches matter most"
    - Reshape (14×14) → resize to image → apply jet colormap

    Returns base64-encoded JPEG of the overlay, or None on failure.
    """
    import torch
    import base64

    try:
        with torch.no_grad():
            vision_out = _model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_attentions=True,
                return_dict=True,
            )

        # Get last-layer attention: (batch, heads, seq, seq)
        attn = vision_out.attentions[-1]  # shape: (1, H, N+1, N+1)
        attn = attn.squeeze(0)            # (H, N+1, N+1)

        # Average across heads, get CLS token's attention to all patches
        attn_mean = attn.mean(dim=0)       # (N+1, N+1)
        cls_attn  = attn_mean[0, 1:]       # (N,)  — CLS→patches, skip CLS itself

        # Reshape to spatial grid
        n_patches = cls_attn.shape[0]
        grid_size = int(n_patches ** 0.5)  # e.g. 14 for 196 patches
        heatmap   = cls_attn.reshape(grid_size, grid_size).cpu().float().numpy()

        # Normalize to [0,255]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)

        # Resize to original image and colorize with a simple blue→red ramp.
        img_w, img_h = pil.size
        heatmap_img = Image.fromarray(heatmap).resize((img_w, img_h), Image.Resampling.BICUBIC)
        colored = ImageOps.colorize(heatmap_img.convert("L"), black="#1f4b99", white="#f5a623")

        # Blend: 55% original image, 45% attention heatmap.
        overlay = Image.blend(pil.convert("RGB"), colored.convert("RGB"), alpha=0.45)

        # Encode as base64 JPEG.
        buf = io.BytesIO()
        overlay.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    except Exception as e:
        print(f"[Attention] {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Model Info
# ══════════════════════════════════════════════════════════════════════════════

def model_info() -> dict:
    if _model is None:
        return {"status": "not_loaded", "device": "unknown"}
    return {
        "status":   "ready",
        "name":     "Salesforce/blip-image-captioning-large",
        "encoder":  "ViT-L/16 (Vision Transformer)",
        "decoder":  "Causal Transformer (BertLMHead)",
        "device":   _device,
        "params":   "~470M",
        "training_data": "129M image-text pairs (LAION + COCO + CC3M + CC12M)",
    }
