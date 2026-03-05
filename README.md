# SyntheticImageGen 🖼️→🧬
> Real Dataset → Synthetic Look-Alike Image Generator

Give it a folder of real images → it learns their visual distribution → generates a synthetic fake dataset that looks real.

---

## The Problem It Solves

You need 500 training images. You have 50.

Traditional augmentation (flip, crop, rotate) reuses the same pixels. **SyntheticImageGen manufactures new ones** — statistically indistinguishable from your real data, generated from scratch by a diffusion model fine-tuned on your specific visual domain.

---

## How It Works (4 Phases)

```
[Your 50 Real Images]
        ↓
  Phase 1: Fingerprint
  Extract color, brightness, texture distribution
        ↓
  Phase 2: LoRA Fine-Tune (SD-Turbo, ~20 min on T4)
  Teach the model your visual domain
        ↓
  Phase 3: Generate + Quality Filter
  Make N synthetic images → discard outliers via cosine similarity
        ↓
  Phase 4: Download
  ZIP with real/ + synthetic/ folders + metadata JSON per image
```

---

## MVP Scope

| Feature | Status |
|---|---|
| Image upload (folder or ZIP) | ✅ MVP |
| Auto-resize to 256×256 | ✅ MVP |
| Metadata fingerprint (color, brightness, texture) | ✅ MVP |
| LoRA fine-tune on SD-Turbo | ✅ MVP |
| Synthetic image generation (N configurable) | ✅ MVP |
| Cosine similarity quality filter | ✅ MVP |
| Side-by-side real vs synthetic preview grid | ✅ MVP |
| ZIP export with labels + metadata JSON | ✅ MVP |
| Authentication / user accounts | ❌ Post-MVP |
| Multiple domain support per session | ❌ Post-MVP |
| Cloud GPU provisioning | ❌ Post-MVP |

---

## Stack

| Layer | Tech |
|---|---|
| UI | Gradio |
| Diffusion backbone | SD-Turbo (`stabilityai/sd-turbo`) |
| Fine-tuning | LoRA via PEFT |
| Diffusion pipeline | 🤗 diffusers |
| Image analysis | OpenCV + Pillow |
| Numerics | NumPy |
| Training runtime | Google Colab (T4 GPU) |

---

## Project Structure

```
synthetic-image-gen/
├── app.py                  # Gradio UI entry point
├── pipeline/
│   ├── ingestor.py         # Upload handling, resize, fingerprint extraction
│   ├── trainer.py          # LoRA fine-tune on SD-Turbo
│   ├── generator.py        # Synthetic image generation loop
│   ├── filter.py           # Cosine similarity quality filter
│   └── exporter.py         # ZIP builder + metadata JSON writer
├── utils/
│   ├── fingerprint.py      # Color histogram, brightness, texture variance
│   └── similarity.py       # Cosine distance between fingerprint vectors
├── notebooks/
│   └── train_colab.ipynb   # Full LoRA training on Colab T4
├── requirements.txt
└── README.md
```

---

## Quickstart (Local)

```bash
# 1. Clone and install
git clone https://github.com/yourname/synthetic-image-gen
cd synthetic-image-gen
pip install -r requirements.txt

# 2. Launch UI
python app.py

# 3. Open browser → upload your image folder → set N → generate
```

**For LoRA training**, use the Colab notebook (`notebooks/train_colab.ipynb`) — local training requires ~8GB VRAM minimum.

---

## Requirements

```txt
gradio>=4.0
diffusers>=0.27
peft>=0.10
transformers>=4.40
torch>=2.1
opencv-python
Pillow
numpy
accelerate
```

---

## Example Use Cases

- **Plant disease detection** — 50 leaf photos → 500 synthetic diseased leaves
- **Medical imaging** — augment rare condition scans for classifier training
- **Industrial defect detection** — generate fake-but-realistic defect samples
- **Wildlife cameras** — expand small camera trap datasets

---

## Output Format

```
synthetic_dataset.zip
├── real/
│   ├── img_001.jpg
│   └── img_002.jpg
└── synthetic/
    ├── syn_001.jpg
    ├── syn_001_meta.json    ← { seed, similarity_score, base_image }
    ├── syn_002.jpg
    └── syn_002_meta.json
```

---

## Quality Filter Logic

Each synthetic image is fingerprinted (same pipeline as real images). Its feature vector is compared to the **centroid of the real distribution** via cosine distance.

```
similarity_score = cosine_similarity(synthetic_fingerprint, real_centroid)
threshold = 0.80  # configurable
```

Images below threshold are discarded and regenerated. Only statistically close images make it into your dataset.

---

## Limitations (Be Honest in Your Portfolio)

- LoRA fine-tune quality scales with dataset size — fewer than ~20 real images may produce noisy results
- SD-Turbo is fast but lower fidelity than SDXL; swap backbone for higher quality at cost of speed
- Quality filter uses shallow features (color, texture) — semantic accuracy not guaranteed
- Not a replacement for real data; works best as a **complement** to real collection

---

## Roadmap

- [ ] SDXL backbone option for higher fidelity
- [ ] FID score reporting (Fréchet Inception Distance)
- [ ] Prompt-guided generation ("add rain," "change lighting")
- [ ] Hugging Face Spaces deployment
- [ ] Batch LoRA training across multiple domains

---

## Why This Project Matters

Data augmentation is a **core production skill** every ML team needs. This project demonstrates:

1. Understanding of the full ML data pipeline — not just model training
2. Practical use of diffusion models beyond text-to-image demos
3. Statistical thinking about data distributions
4. End-to-end product thinking: ingest → process → generate → filter → export

Pair with **PlantDoc** or **TruthScan** in your portfolio for a complete ML engineering story.

---

*Built with SD-Turbo + LoRA + Gradio | ~2 week build | Colab T4 compatible*
