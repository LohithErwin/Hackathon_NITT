

import base64
import csv
import sys
import time
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------- config ----------------
OLLAMA_URL = "http://localhost:11434"
VLM_MODEL = "qwen2.5vl:3b"
CLIP_MODEL = "openai/clip-vit-large-patch14"   # standard model for CLIPScore
IMAGE_DIR = Path(__file__).parent / "images"
OUT_CSV = Path(__file__).parent / "clip_results.csv"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# CLIP's text encoder only reads 77 tokens, so ask for a short caption.
PROMPT = ("Describe this image in one detailed sentence of at most 30 words. "
          "Mention the main objects, setting and colors. No preamble.")
# ----------------------------------------


def check_ollama():
    """Make sure Ollama is up and the model is pulled."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        sys.exit("Ollama is not reachable at localhost:11434. Start it with:  ollama serve")
    names = [m["name"] for m in r.json().get("models", [])]
    if not any(n.startswith(VLM_MODEL) for n in names):
        sys.exit(f"{VLM_MODEL} not found. Run:  ollama pull {VLM_MODEL}\nInstalled: {names}")
    print(f"[ok] Ollama running, {VLM_MODEL} available")


def caption_image(path: Path) -> str:
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": VLM_MODEL,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"].strip().replace("\n", " ")


def main():
    check_ollama()

    images = sorted(p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in EXTS)
    if not images:
        sys.exit(f"No images found in {IMAGE_DIR}")
    print(f"[ok] Found {len(images)} image(s) in {IMAGE_DIR}\n")

    # ---- 1. Generate captions with Qwen2.5-VL ----
    captions = []
    for p in images:
        t0 = time.time()
        cap = caption_image(p)
        captions.append(cap)
        print(f"{p.name}  ({time.time() - t0:.1f}s)\n  -> {cap}\n")

    # ---- 2. CLIP score ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ({CLIP_MODEL}) on {device} ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    pil_imgs = [Image.open(p).convert("RGB") for p in images]
    with torch.no_grad():
        inputs = proc(text=captions, images=pil_imgs, return_tensors="pt",
                      padding=True, truncation=True, max_length=77).to(device)
        out = model(**inputs)
        # image_embeds / text_embeds are the projected embeddings
        # (works on both old and new transformers versions)
        img_emb = out.image_embeds
        txt_emb = out.text_embeds
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

    sim = (img_emb @ txt_emb.T).cpu()          # [n_images, n_captions] cosine
    matched = sim.diag()

    # CLIPScore (Hessel et al. 2021): 2.5 * max(cos, 0)  -> roughly 0..1
    clip_score = 2.5 * matched.clamp(min=0)

    # Sanity check: does each image pick its own caption over the others?
    n = len(images)
    if n > 1:
        top1 = (sim.argmax(dim=1) == torch.arange(n)).float().mean().item()
        mismatched = (sim.sum() - matched.sum()) / (n * n - n)

    # ---- 3. Report ----
    print("\n" + "=" * 78)
    print(f"{'image':<28}{'cosine':>10}{'CLIPScore':>12}{'x100':>10}")
    print("-" * 78)
    rows = []
    for p, cap, c, s in zip(images, captions, matched, clip_score):
        print(f"{p.name[:27]:<28}{c.item():>10.4f}{s.item():>12.4f}{c.item()*100:>10.2f}")
        rows.append([p.name, cap, f"{c.item():.4f}", f"{s.item():.4f}"])
    print("-" * 78)
    print(f"{'MEAN':<28}{matched.mean().item():>10.4f}{clip_score.mean().item():>12.4f}"
          f"{matched.mean().item()*100:>10.2f}")
    if n > 1:
        print(f"\nMean cosine with WRONG captions (baseline): {mismatched.item():.4f}")
        print(f"Image->caption top-1 retrieval accuracy:   {top1*100:.1f}%")
    print("=" * 78)
    print("Guide (ViT-L/14 cosine): ~0.30+ good, 0.25-0.30 ok, <0.20 poor match")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "caption", "cosine", "clip_score"])
        w.writerows(rows)
    print(f"\nSaved results to {OUT_CSV}")


if __name__ == "__main__":
    main()