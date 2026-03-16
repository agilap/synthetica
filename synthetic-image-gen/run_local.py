"""
run_local.py — End-to-end pipeline runner (no UI).

Steps:
  0. Download a subset of Oxford 102 Flowers from HuggingFace into data/real_raw/
  1. Ingest & resize to 256x256  →  data/real/
  2. Build real distribution fingerprint (centroid vector)
  3. Load SD-Turbo + LoRA weights from ./lora/
  4. Generate N synthetic images  →  data/synthetic/
  5. Quality-filter against real centroid
  6. Export ZIP  →  data/dataset.zip

Edit the CONFIG block below, then run:
  PYTHONPATH=. /home/alex/synthetica/.venv/bin/python run_local.py
"""
from __future__ import annotations

import os
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
LORA_DIR        = "./lora_output/"            # folder containing adapter_model.safetensors
REAL_RAW_DIR    = "data/real_raw/"    # where downloaded Oxford Flowers images land
REAL_DIR        = "data/real/"        # resized 256x256 copies
SYNTHETIC_DIR   = "data/synthetic/"  # generated images
EXPORT_ZIP      = "data/dataset.zip"

MAX_REAL_IMAGES = 200     # how many Oxford Flowers to download (None = all 8 189)
N_GENERATE      = 50      # synthetic images to generate
QUALITY_THRESHOLD = 0.80  # cosine similarity cutoff
SEED            = 42
PROMPT          = "a high quality photo of a flower"
# ─────────────────────────────────────────────────────────────────────────────


def download_oxford_flowers(dest_dir: str, max_images: int | None) -> None:
    """Download Oxford Flowers subset from HuggingFace datasets to dest_dir."""
    from datasets import load_dataset
    from PIL import Image

    print(f"\n{'='*60}")
    print("Step 0: Downloading Oxford Flowers …")
    print(f"{'='*60}")

    os.makedirs(dest_dir, exist_ok=True)

    ds = load_dataset("nelorth/oxford-flowers", split="train")
    if max_images is not None:
        ds = ds.select(range(min(max_images, len(ds))))

    print(f"Saving {len(ds)} images to {dest_dir} …")
    for i, sample in enumerate(ds):
        img = sample.get("image") or sample.get("img")
        if img is None:
            continue
        img.convert("RGB").save(
            Path(dest_dir) / f"flower_{i:05d}.jpg",
            format="JPEG", quality=95,
        )
        if (i + 1) % 50 == 0 or (i + 1) == len(ds):
            print(f"  {i + 1}/{len(ds)}")

    print(f"✓ Download complete — {len(list(Path(dest_dir).glob('*.jpg')))} images")


def main() -> None:
    # ── Step 0: Download Oxford Flowers ──────────────────────────────────────
    if not any(Path(REAL_RAW_DIR).glob("*.jpg")):
        download_oxford_flowers(REAL_RAW_DIR, MAX_REAL_IMAGES)
    else:
        existing = len(list(Path(REAL_RAW_DIR).glob("*.jpg")))
        print(f"\nStep 0: Skipped — {existing} images already in {REAL_RAW_DIR}")

    # ── Step 1: Ingest & resize ───────────────────────────────────────────────
    from pipeline.ingestor import RealDataIngestor

    existing_real = sorted(Path(REAL_DIR).glob("*.jpg")) if Path(REAL_DIR).exists() else []
    if existing_real:
        real_paths = [str(p) for p in existing_real]
        print(f"\nStep 1: Skipped — {len(real_paths)} resized images already in {REAL_DIR}")
    else:
        print(f"\n{'='*60}")
        print("Step 1: Ingesting & resizing real images …")
        print(f"{'='*60}")
        ingestor   = RealDataIngestor(upload_path=REAL_RAW_DIR, output_dir=REAL_DIR)
        real_paths = ingestor.load_images()
        print(f"✓ {len(real_paths)} images saved to {REAL_DIR}")

    # ── Step 2: Distribution fingerprint ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("Step 2: Building real distribution fingerprint …")
    print(f"{'='*60}")
    _ingestor    = RealDataIngestor(upload_path=REAL_RAW_DIR, output_dir=REAL_DIR)
    dist         = _ingestor.build_distribution_fingerprint(real_paths)
    centroid     = dist["centroid_vector"]
    fingerprints = dist["individual_fingerprints"]
    print(f"✓ Centroid vector shape: {centroid.shape}")

    # ── Step 3: Load pipeline ─────────────────────────────────────────────────
    from pipeline.generator import SyntheticGenerator

    print(f"\n{'='*60}")
    print(f"Step 3: Loading SD-Turbo + LoRA from '{LORA_DIR}' …")
    print(f"{'='*60}")
    generator = SyntheticGenerator(
        lora_weights_dir=LORA_DIR,
        output_dir=SYNTHETIC_DIR,
        seed=SEED,
    )
    generator.load_pipeline()

    # ── Step 4: Generate ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 4: Generating {N_GENERATE} synthetic images …")
    print(f"{'='*60}")
    generated_records = generator.generate(n=N_GENERATE, prompt=PROMPT)
    print(f"✓ {len(generated_records)} images saved to {SYNTHETIC_DIR}")

    # ── Step 5: Quality filter ────────────────────────────────────────────────
    from pipeline.filter import QualityFilter

    print(f"\n{'='*60}")
    print(f"Step 5: Filtering (threshold={QUALITY_THRESHOLD}) …")
    print(f"{'='*60}")
    qf = QualityFilter(real_centroid_vector=centroid, threshold=QUALITY_THRESHOLD)
    accepted, rejected = qf.filter_batch(generated_records)

    # ── Step 6: Export ────────────────────────────────────────────────────────
    from pipeline.exporter import DatasetExporter

    print(f"\n{'='*60}")
    print("Step 6: Exporting ZIP …")
    print(f"{'='*60}")
    exporter  = DatasetExporter(output_path=EXPORT_ZIP)
    zip_path  = exporter.export(real_paths, fingerprints, accepted)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Real images     : {len(real_paths)}")
    print(f"  Generated       : {len(generated_records)}")
    print(f"  Accepted        : {len(accepted)}")
    print(f"  Rejected        : {len(rejected)}")
    print(f"  Dataset ZIP     : {zip_path.resolve()}")


if __name__ == "__main__":
    main()
