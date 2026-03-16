# SyntheticImageGen — System Architecture
> RTX 3050 Ti (4 GB VRAM) · 4-bit Quantized · Research-Grounded Redesign

---

## 0. Design Constraints & Guiding Principles

| Constraint | Value | Impact |
|---|---|---|
| VRAM budget | 4 GB (RTX 3050 Ti) | Every component must be VRAM-profiled |
| Target resolution | 256 × 256 (train), up to 512 × 512 (inference) | Caps UNet memory per forward pass |
| LoRA adapter rank | r = 4–8 | Low-rank subspace theory (Aghajanyan et al., 2020) |
| Quantization | INT4 (bitsandbytes `nf4`) | ~60 % VRAM reduction on UNet weights |
| Attention | xformers memory-efficient attention | Eliminates O(n²) attention map materialization |
| Batch size | 1 (train), 4 (inference with tiling) | Stable under 4 GB ceiling |

**Core philosophy:** every upgrade must survive a 4 GB ceiling. If it doesn't fit, it doesn't ship.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SyntheticImageGen                            │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────────┐    │
│  │  Gradio  │───▶│   Ingestor   │───▶│  DINOv2 Fingerprinter  │    │
│  │   UI     │    │  (Phase 1)   │    │  (replaces OpenCV)     │    │
│  └──────────┘    └──────────────┘    └────────────┬───────────┘    │
│                                                   │                 │
│                                         Real Distribution           │
│                                         Centroid (L2-normed)        │
│                                                   │                 │
│  ┌────────────────────────────────────────────────▼───────────┐    │
│  │               LoRA Fine-Tuner  (Phase 2)                   │    │
│  │                                                            │    │
│  │  SD 1.5 UNet (INT4, nf4)  ◀── bitsandbytes quantization   │    │
│  │  + LCM-LoRA adapter (r=8) ◀── Trajectory Consistency      │    │
│  │  + VAE (fp16, CPU-offload)                                 │    │
│  │  + Text encoder (fp16, CPU-offload)                        │    │
│  │  Gradient checkpointing ✓  xformers attention ✓            │    │
│  └────────────────────────────────────────────────┬───────────┘    │
│                                                   │                 │
│  ┌────────────────────────────────────────────────▼───────────┐    │
│  │           Generator + CLIP Quality Filter  (Phase 3)       │    │
│  │                                                            │    │
│  │  Generate N images (batches of 4 with VAE tiling)         │    │
│  │  Score each via CLIP ViT-B/32 cosine sim vs real set      │    │
│  │  FID proxy scoring (InceptionV3 features, 64-d pool)      │    │
│  │  Discard score < τ (default 0.78), regenerate             │    │
│  └────────────────────────────────────────────────┬───────────┘    │
│                                                   │                 │
│  ┌────────────────────────────────────────────────▼───────────┐    │
│  │                    Exporter  (Phase 4)                     │    │
│  │  ZIP: real/ + synthetic/  +  per-image metadata JSON       │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Ingestor (`pipeline/ingestor.py`)

**Responsibilities:** Accept folder/ZIP upload → validate → resize → hand off to fingerprinter.

```
Input  : folder or .zip of JPG/PNG images
Output : List[PIL.Image]  (256×256, RGB, normalized 0–1)

Steps:
  1. Unpack ZIP if provided (tempfile, auto-cleanup)
  2. Filter to valid image extensions (.jpg .jpeg .png .webp)
  3. Resize: Lanczos downscale → center-crop to 256×256
  4. Reject images < 64×64 (too small to fingerprint reliably)
  5. Return image list + ingest report (count, rejected, avg size before resize)
```

**VRAM impact:** Zero — runs on CPU.

---

### 2.2 DINOv2 Fingerprinter (`utils/fingerprint.py`)

**Replaces:** OpenCV color histogram + brightness + texture variance.

**Why DINOv2 over OpenCV features:**  
Self-supervised ViT features (Oquab et al., *DINOv2*, 2023) encode *semantic* content — not just pixel statistics. Two images of the same disease spot in different lighting will have similar DINOv2 embeddings but very different color histograms. This gives the quality filter real teeth.

```
Model    : facebook/dinov2-small  (ViT-S/14)
          → 384-dim CLS token embedding per image
          → model loaded in fp16, runs on GPU during fingerprint pass only
          → offloaded to CPU after fingerprint extraction

VRAM cost: ~330 MB (fp16 ViT-S)  — loaded briefly, then freed

Pipeline:
  1. Batch images through ViT-S/14 (batch=8, 256×256 input)
  2. Extract CLS token → 384-dim vector
  3. L2-normalize each vector
  4. Compute real_centroid = mean(all real vectors)
  5. Compute real_std  = std(all real vectors)  [for adaptive threshold]
  6. Persist centroid + std as numpy .npy in session dir

Output: dict { "centroid": np.ndarray(384,),
               "std": float,
               "per_image_embeddings": np.ndarray(N, 384) }
```

**Fallback (CPU-only mode):** If VRAM unavailable during fingerprint, use CLIP ViT-B/32 on CPU — slower (~3 s/image) but identical interface.

---

### 2.3 LoRA Fine-Tuner (`pipeline/trainer.py`)

**Backbone selection — why SD 1.5 over SD-Turbo for 4 GB VRAM:**

| Model | UNet params | FP16 VRAM | INT4 VRAM | Training feasibility @ 4 GB |
|---|---|---|---|---|
| SD-Turbo | 865 M | ~3.4 GB | ~1.8 GB | Tight (VAE + text enc push OOM) |
| SD 1.5 | 860 M | ~3.2 GB | ~1.7 GB | ✅ Comfortable with offloads |
| SDXL | 2.6 B | ~9.8 GB | ~5.1 GB | ❌ OOM |
| LCM-SD 1.5 | 860 M | ~3.2 GB | ~1.7 GB | ✅ **Selected** |

**Selected stack: SD 1.5 + LCM-LoRA (Luo et al., 2023)**  
LCM-LoRA (Latent Consistency Model LoRA) distills the multi-step DDPM process into 4-step inference while preserving fine-tuning capability. Training a domain LoRA on top of an LCM backbone means you get both fast inference *and* domain adaptation.

#### 4-Bit Quantization Setup

```python
from transformers import BitsAndBytesConfig
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4: optimal for normally distributed weights
    bnb_4bit_use_double_quant=True,     # nested quantization: quantize the quantization constants
    bnb_4bit_compute_dtype=torch.float16
)

unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
    quantization_config=nf4_config,
    torch_dtype=torch.float16
)
```

**NF4 vs INT8 choice rationale:**  
NormalFloat4 (Dettmers et al., *QLoRA*, 2023) is information-theoretically optimal for weights drawn from a normal distribution — which neural network weights approximate. It outperforms INT4 by ~1 dB PSNR at the same bit-width.

#### LoRA Adapter Config

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                          # rank: low enough for 4 GB, high enough for domain shift
    lora_alpha=16,                # scaling = alpha / r = 2.0
    target_modules=[              # inject only into attention projections
        "to_q", "to_k", "to_v", "to_out.0",
        "ff.net.0.proj", "ff.net.2"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="diffusion"
)
```

**Why r=8 specifically:**  
Aghajanyan et al. (2020) show that pre-trained models have a very low *intrinsic dimensionality* — the minimum dimension needed to express 90% of fine-tuning. For vision tasks on SD 1.5-scale models, empirical results place this at r ≤ 16. Rank 8 captures the domain shift while keeping the adapter at ~6 MB (vs. 3 GB full fine-tune).

#### Training Loop (Memory-Optimized)

```
Gradient checkpointing : torch.utils.checkpoint  → trades compute for VRAM
                         Saves ~40% activation memory during backward pass
xformers attention     : xformers.ops.memory_efficient_attention
                         Eliminates materialization of full N×N attention maps
VAE                    : fp16, moved to CPU after encode step
Text encoder           : fp16, CPU-offloaded after prompt embedding
Optimizer              : bitsandbytes 8-bit AdamW (bnb.optim.AdamW8bit)
                         Optimizer state in INT8 → ~75% optimizer memory reduction
LR schedule            : Cosine with warmup (500 steps)
Learning rate          : 1e-4
Max train steps        : 1000  (sufficient for <100 real images)
Batch size             : 1
Mixed precision        : fp16  (via accelerate)
```

**Estimated peak VRAM during training:**

| Component | VRAM |
|---|---|
| UNet (INT4 nf4) | ~1.7 GB |
| LoRA adapter (r=8) | ~6 MB |
| Activations (grad ckpt) | ~0.6 GB |
| 8-bit AdamW optimizer | ~0.3 GB |
| VAE encode (brief) | ~0.2 GB |
| **Total peak** | **~2.8 GB** ✅ |

---

### 2.4 Generator (`pipeline/generator.py`)

```
Inference stack:
  - LCMScheduler (4 denoising steps — deterministic, fast)
  - Guidance scale: 1.0–2.0  (LCM operates at low CFG)
  - Attention slicing: pipe.enable_attention_slicing(slice_size=1)
  - VAE tiling: pipe.enable_vae_tiling()  → allows 512×512 at 4 GB
  - Sequential CPU offload for text encoder between batches

Generation loop:
  For each batch of 4 images:
    1. Sample seed from RNG (logged in metadata)
    2. Generate latent via LCM (4 steps)
    3. Decode via VAE (fp16)
    4. Extract DINOv2 embedding (batched, GPU)
    5. Score vs real_centroid
    6. Accept if score ≥ τ, else queue for regen
    7. Log: seed, score, generation_time_ms

Adaptive threshold:
  τ = max(0.65, real_mean_sim - 2 * real_std)
  This self-calibrates to the tightness of your real distribution.
  A uniform-looking dataset (e.g., all X-rays) gets a tighter τ than
  a diverse one (e.g., mixed lighting conditions).
```

**Peak VRAM during inference (no training loaded):**

| Component | VRAM |
|---|---|
| UNet (INT4) | ~1.7 GB |
| LCM scheduler tensors | ~0.1 GB |
| VAE decode (tiled) | ~0.3 GB |
| DINOv2-S scoring | ~0.3 GB |
| **Total peak** | **~2.4 GB** ✅ |

---

### 2.5 Quality Filter (`pipeline/filter.py`)

**Two-tier filtering — shallow + semantic:**

```
Tier 1 — Fast pre-filter (CPU, <1 ms/image):
  • HSV histogram correlation  (OpenCV compareHist, CORREL method)
  • Reject if correlation < 0.5 with any real image
  • Purpose: catch color-domain failures immediately (e.g., all-gray images)

Tier 2 — Semantic filter (GPU, ~15 ms/image):
  • DINOv2 CLS cosine similarity vs real_centroid
  • Reject if sim < τ (adaptive threshold, see §2.4)
  • Purpose: catch semantic drift (e.g., wrong object class generated)

Optional Tier 3 — FID proxy (every 50 images, CPU):
  • Extract InceptionV3 pool_3 features (2048-d)
  • Estimate μ, Σ of real vs synthetic batches
  • Report running FID estimate in UI (not used as hard filter — informational)
```

**Why add FID proxy:**  
Fréchet Inception Distance (Heusel et al., 2017) is the industry standard for measuring distributional similarity between real and generated image sets. Reporting it in the UI gives users an objective quality signal without requiring them to understand cosine distance on DINOv2 embeddings.

---

### 2.6 Exporter (`pipeline/exporter.py`)

```
Output structure:
  synthetic_dataset_{timestamp}.zip
  ├── real/
  │   ├── img_001.jpg
  │   └── img_001_fingerprint.json   ← DINOv2 embedding (base64 + norm)
  ├── synthetic/
  │   ├── syn_001.jpg
  │   └── syn_001_meta.json
  └── dataset_report.json            ← global stats

syn_NNN_meta.json schema:
{
  "seed": 42,
  "similarity_score": 0.834,
  "tier1_histogram_corr": 0.71,
  "fid_estimate": 18.4,             ← null if < 50 images generated
  "base_image": "img_007.jpg",      ← nearest real image by DINOv2 sim
  "generation_time_ms": 1240,
  "lora_checkpoint": "checkpoint-800",
  "model": "sd-1.5-lcm-lora-r8",
  "resolution": [256, 256],
  "prompt": "photo of [V]",         ← textual inversion token if used
  "threshold_used": 0.78
}

dataset_report.json:
{
  "real_count": 50,
  "synthetic_count": 450,
  "rejected_count": 23,
  "acceptance_rate": 0.951,
  "fid_estimate": 18.4,
  "mean_similarity": 0.821,
  "std_similarity": 0.044,
  "training_steps": 800,
  "total_generation_time_s": 184
}
```

---

## 3. VRAM Budget Planner

Full lifecycle — which components live in VRAM at each phase:

```
Phase 1 (Ingest + Fingerprint):
  DINOv2-S (fp16)         = 330 MB
  Image batch (8×256×256) = ~50 MB
  ─────────────────────────────────
  Peak                    ≈ 380 MB  ✅

Phase 2 (Training):
  UNet INT4 nf4           = 1,700 MB
  LoRA adapter            = 6 MB
  Activations (grad ckpt) = 600 MB
  8-bit AdamW state       = 300 MB
  Brief VAE encode        = 200 MB
  ─────────────────────────────────
  Peak                    ≈ 2,806 MB ✅  (margin: ~1.2 GB)

Phase 3 (Inference):
  UNet INT4 nf4           = 1,700 MB
  VAE decode (tiled)      = 300 MB
  DINOv2-S scoring        = 330 MB
  LCM scheduler           = 100 MB
  ─────────────────────────────────
  Peak                    ≈ 2,430 MB ✅  (margin: ~1.6 GB)
```

> **Note:** Training and inference do not overlap. After training completes, the optimizer state and gradient buffers are freed before generation begins. `torch.cuda.empty_cache()` is called at every phase boundary.

---

## 4. Dependency Stack

```
# Core diffusion
diffusers>=0.29          # LCMScheduler, pipeline CPU offload, VAE tiling
peft>=0.11               # LoraConfig, get_peft_model
transformers>=4.41       # DINOv2, CLIP, BitsAndBytesConfig
accelerate>=0.30         # mixed precision trainer, device placement

# Quantization
bitsandbytes>=0.43       # NF4, 8-bit AdamW
                         # REQUIRES: CUDA 11.8+ driver

# Memory-efficient attention
xformers>=0.0.26         # memory_efficient_attention kernel
                         # REQUIRES: torch matching CUDA version

# Feature extraction  
torch>=2.2               # compile() support for DINOv2 speedup
torchvision>=0.17        # InceptionV3 for FID proxy

# Image processing
opencv-python>=4.9
Pillow>=10.2
numpy>=1.26

# UI + export
gradio>=4.25
```

---

## 5. Research Basis

| Component | Paper | Key Insight Used |
|---|---|---|
| LoRA adapter design | Hu et al., *LoRA* (2021) | Decompose weight delta as BA; r=8 sufficient for domain shift |
| Rank selection theory | Aghajanyan et al. (2020) | Pre-trained models fine-tune in low-dim subspaces; r≤16 captures it |
| 4-bit quantization | Dettmers et al., *QLoRA* (2023) | NF4 is info-theoretically optimal for Gaussian weight distributions |
| Fast inference | Luo et al., *LCM-LoRA* (2023) | Consistency distillation enables 4-step inference without quality collapse |
| Semantic fingerprinting | Oquab et al., *DINOv2* (2023) | ViT self-supervised features generalize across domains without task-specific labels |
| Quality measurement | Heusel et al., *FID* (2017) | Fréchet distance on Inception features = distributional realism metric |
| Latent space foundation | Kingma & Welling, *VAE* (2013) | Defines the ℝᵈ latent space LoRA operates within |
| Denoising process | Ho et al., *DDPM* (2020) | Forward/reverse diffusion process SD 1.5 is built on |

---

## 6. File Structure (Updated)

```
synthetica/
├── app.py                        # Gradio entry point
├── pipeline/
│   ├── ingestor.py               # Upload → resize → validate
│   ├── trainer.py                # SD 1.5 + LCM-LoRA + INT4 fine-tune
│   ├── generator.py              # LCM inference loop + adaptive threshold
│   ├── filter.py                 # Tier-1 histogram + Tier-2 DINOv2 filter
│   └── exporter.py               # ZIP builder + metadata JSON
├── utils/
│   ├── fingerprint.py            # DINOv2-S embedding extractor
│   ├── similarity.py             # Cosine sim, adaptive τ, FID proxy
│   ├── vram_guard.py             # VRAM monitor + phase-boundary cache flush
│   └── quantize.py               # NF4 config factory + model loader
├── config/
│   └── defaults.yaml             # τ=0.78, r=8, steps=1000, batch=1
├── notebooks/
│   └── train_local.ipynb         # Full local training walkthrough
├── requirements.txt
├── ARCHITECTURE.md               # ← this file
├── STATES.md
└── .github/
    └── copilot-instructions.md
```

---

## 7. Known Limitations

- **Minimum dataset size:** DINOv2 centroid is unreliable below ~15 real images. Below this threshold, fall back to Tier-1 (OpenCV) filtering only and warn the user.
- **Resolution ceiling:** 512×512 is achievable via VAE tiling at 4 GB, but training at 512 will OOM. Train at 256, upsample at inference with `enable_vae_tiling()`.
- **NF4 + xformers compatibility:** Some xformers versions conflict with bitsandbytes on Windows. Provide a `--no-xformers` flag fallback that uses PyTorch SDPA (`F.scaled_dot_product_attention`) instead.
- **LCM guidance range:** LCM is trained at guidance_scale ≤ 2.0. Values above this degrade output. The UI must clamp the slider.
- **FID reliability:** FID below 50 images is noisy. Display FID only when synthetic count ≥ 50.
