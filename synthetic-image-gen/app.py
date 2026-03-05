"""
app.py — Gradio web interface for the synthetic-image-gen pipeline.

Connects the full pipeline:
  Upload → Ingest → Generate (LoRA) → Filter → Export → Download

Assumes LoRA weights are already trained (e.g. via notebooks/train_colab.ipynb)
and available on the local filesystem.  No training UI is provided here.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import gradio as gr

# ── Pipeline imports ──────────────────────────────────────────────────────
from pipeline.ingestor import RealDataIngestor
from pipeline.generator import SyntheticGenerator
from pipeline.filter import QualityFilter
from pipeline.exporter import DatasetExporter

# ── Working directories ───────────────────────────────────────────────────
_REAL_DIR      = "data/real/"
_SYNTHETIC_DIR = "data/synthetic/"
_EXPORT_ZIP    = "data/dataset.zip"


# ---------------------------------------------------------------------------
# Core pipeline function (generator — yields log lines for streaming UI)
# ---------------------------------------------------------------------------

def run_pipeline(
    upload_file,          # Gradio file object (path to uploaded file/ZIP)
    lora_weights_path: str,
    n_images: int,
    quality_threshold: float,
    generation_seed: int,
    prompt: str,
):
    """Run the full pipeline and yield (log, real_imgs, syn_imgs, summary, zip_path) tuples."""

    def _emit(msg: str, log_so_far: list[str]):
        log_so_far.append(msg)
        return "\n".join(log_so_far)

    log: list[str] = []
    real_gallery: list[str] = []
    syn_gallery:  list[str] = []
    summary = ""
    zip_path = None

    # ── Validate LoRA weights path ─────────────────────────────────────────
    lora_dir = Path(lora_weights_path.strip())
    if not lora_dir.exists():
        gr.Warning(
            f"LoRA weights folder not found: '{lora_dir}'. "
            "Train on Colab first and paste the correct local path."
        )
        yield _emit(f"✗ LoRA weights directory not found: {lora_dir}", log), \
              real_gallery, syn_gallery, summary, None
        return

    if upload_file is None:
        gr.Warning("Please upload a folder ZIP or image files first.")
        yield _emit("✗ No upload provided.", log), real_gallery, syn_gallery, summary, None
        return

    upload_path = upload_file.name if hasattr(upload_file, "name") else str(upload_file)

    # ── Step 1: Ingest real images ────────────────────────────────────────
    try:
        yield _emit("▶ [1/6] Ingesting real images …", log), \
              real_gallery, syn_gallery, summary, None

        ingestor = RealDataIngestor(upload_path=upload_path, output_dir=_REAL_DIR)
        real_paths = ingestor.load_images()

        log.append(f"  ✓ Loaded {len(real_paths)} image(s) → {_REAL_DIR}")
        real_gallery = real_paths
        yield "\n".join(log), real_gallery, syn_gallery, summary, None

    except Exception:
        yield _emit(f"✗ Ingest failed:\n{traceback.format_exc()}", log), \
              real_gallery, syn_gallery, summary, None
        return

    # ── Step 2: Build real distribution fingerprint ───────────────────────
    try:
        yield _emit("▶ [2/6] Building distribution fingerprint …", log), \
              real_gallery, syn_gallery, summary, None

        dist = ingestor.build_distribution_fingerprint(real_paths)
        centroid      = dist["centroid_vector"]
        fingerprints  = dist["individual_fingerprints"]

        log.append("  ✓ Fingerprint centroid computed.")
        yield "\n".join(log), real_gallery, syn_gallery, summary, None

    except Exception:
        yield _emit(f"✗ Fingerprint failed:\n{traceback.format_exc()}", log), \
              real_gallery, syn_gallery, summary, None
        return

    # ── Step 3: Load SD-Turbo + LoRA pipeline ────────────────────────────
    try:
        yield _emit(f"▶ [3/6] Loading SD-Turbo pipeline + LoRA from '{lora_dir}' …", log), \
              real_gallery, syn_gallery, summary, None

        generator = SyntheticGenerator(
            lora_weights_dir=str(lora_dir),
            output_dir=_SYNTHETIC_DIR,
            seed=generation_seed,
        )
        generator.load_pipeline()

        log.append("  ✓ Pipeline ready.")
        yield "\n".join(log), real_gallery, syn_gallery, summary, None

    except Exception:
        yield _emit(f"✗ Pipeline load failed:\n{traceback.format_exc()}", log), \
              real_gallery, syn_gallery, summary, None
        return

    # ── Step 4: Generate synthetic images ────────────────────────────────
    try:
        yield _emit(f"▶ [4/6] Generating {n_images} synthetic images …", log), \
              real_gallery, syn_gallery, summary, None

        generated_records = generator.generate(n=n_images, prompt=prompt)

        log.append(f"  ✓ Generated {len(generated_records)} image(s).")
        yield "\n".join(log), real_gallery, syn_gallery, summary, None

    except Exception:
        yield _emit(f"✗ Generation failed:\n{traceback.format_exc()}", log), \
              real_gallery, syn_gallery, summary, None
        return

    # ── Step 5: Quality filter ────────────────────────────────────────────
    try:
        yield _emit(f"▶ [5/6] Filtering with threshold={quality_threshold:.2f} …", log), \
              real_gallery, syn_gallery, summary, None

        qf = QualityFilter(real_centroid_vector=centroid, threshold=quality_threshold)
        accepted, rejected = qf.filter_batch(generated_records)

        syn_gallery = [r["image_path"] for r in accepted]
        summary = (
            f"{len(real_paths)} real  |  "
            f"{len(accepted)} synthetic accepted  |  "
            f"{len(rejected)} rejected"
        )
        log.append(
            f"  ✓ {len(accepted)}/{len(generated_records)} passed quality filter."
        )
        yield "\n".join(log), real_gallery, syn_gallery, summary, None

    except Exception:
        yield _emit(f"✗ Filter failed:\n{traceback.format_exc()}", log), \
              real_gallery, syn_gallery, summary, None
        return

    # ── Step 6: Export ZIP ────────────────────────────────────────────────
    try:
        yield _emit("▶ [6/6] Building ZIP archive …", log), \
              real_gallery, syn_gallery, summary, None

        exporter = DatasetExporter(output_path=_EXPORT_ZIP)
        zip_out  = exporter.export(
            real_image_paths=real_paths,
            real_fingerprints=fingerprints,
            accepted_records=accepted,
        )
        zip_path = str(zip_out)

        log.append(f"  ✓ Dataset saved to {zip_out}")
        log.append("✅ Pipeline complete! Download your dataset below.")
        yield "\n".join(log), real_gallery, syn_gallery, summary, zip_path

    except Exception:
        yield _emit(f"✗ Export failed:\n{traceback.format_exc()}", log), \
              real_gallery, syn_gallery, summary, None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SyntheticImageGen", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            "# 🖼️ SyntheticImageGen\n"
            "Generate synthetic training images by fine-tuning SD-Turbo with LoRA "
            "on your own real images."
        )

        # Shared state passed between tabs via Gradio state
        _real_imgs_state = gr.State([])
        _syn_imgs_state  = gr.State([])
        _summary_state   = gr.State("")

        with gr.Tabs():

            # ── Tab 1: Generate ──────────────────────────────────────────
            with gr.Tab("Generate"):

                with gr.Row():
                    with gr.Column(scale=1):
                        upload = gr.File(
                            label="Upload Real Images (ZIP of images or single folder ZIP)",
                            file_types=[".zip", ".jpg", ".jpeg", ".png"],
                        )

                        lora_path = gr.Textbox(
                            label="LoRA Weights Path",
                            value="./lora_output/",
                            placeholder="Path to your downloaded lora_output/ folder",
                        )
                        gr.Markdown(
                            "> **How to get LoRA weights:** Train on Colab using "
                            "`notebooks/train_colab.ipynb`, then download the "
                            "`lora_output/` folder and paste its local path above."
                        )

                        prompt_input = gr.Textbox(
                            label="Generation Prompt",
                            value="a high quality photo",
                        )

                    with gr.Column(scale=1):
                        n_slider = gr.Slider(
                            minimum=10, maximum=500, step=10, value=100,
                            label="Number of images to generate",
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.50, maximum=0.99, step=0.01, value=0.80,
                            label="Quality threshold (cosine similarity)",
                        )
                        seed_input = gr.Number(
                            label="Random seed", value=42, precision=0,
                        )

                        run_btn = gr.Button("🚀 Generate Dataset", variant="primary")

                status_log = gr.Textbox(
                    label="Status log",
                    lines=10,
                    interactive=False,
                    placeholder="Pipeline output will appear here …",
                )

                download_file = gr.File(
                    label="⬇️ Download Dataset ZIP",
                    interactive=False,
                    visible=True,
                )

            # ── Tab 2: Preview ───────────────────────────────────────────
            with gr.Tab("Preview"):
                summary_text = gr.Textbox(
                    label="Summary",
                    interactive=False,
                    placeholder="Run the pipeline to see results …",
                )
                with gr.Row():
                    real_gallery = gr.Gallery(
                        label="Real images",
                        columns=4,
                        height=400,
                        object_fit="contain",
                    )
                    syn_gallery = gr.Gallery(
                        label="Accepted synthetic images",
                        columns=4,
                        height=400,
                        object_fit="contain",
                    )

        # ── Wire up the button ───────────────────────────────────────────
        run_btn.click(
            fn=run_pipeline,
            inputs=[
                upload,
                lora_path,
                n_slider,
                threshold_slider,
                seed_input,
                prompt_input,
            ],
            outputs=[
                status_log,
                real_gallery,
                syn_gallery,
                summary_text,
                download_file,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo = build_ui()
    demo.launch(share=False)


if __name__ == "__main__":
    main()
