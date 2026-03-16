"""SyntheticImageGen Gradio app layout (no event wiring logic yet)."""

from __future__ import annotations

import uuid
from pathlib import Path

import gradio as gr
import yaml


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "defaults.yaml"
DEFAULT_CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}


def show_only(panel_name: str) -> list:
	names = [
		"upload",
		"fingerprint",
		"train_config",
		"training",
		"gen_config",
		"generating",
		"preview",
		"exporting",
		"error",
	]
	return [gr.update(visible=(n == panel_name)) for n in names]


with gr.Blocks(theme=gr.themes.Soft()) as demo:
	gr.Markdown("# SyntheticImageGen 🧬")
	gr.Markdown("> Real Dataset → Synthetic Look-Alike Generator")
	vram_badge = gr.Textbox(label="GPU Status", interactive=False, value="Checking...")

	app_state = gr.State({"phase": "IDLE", "session_id": str(uuid.uuid4())})

	with gr.Column(visible=True) as upload_panel:
		gr.Markdown("Upload at least 15 images (.jpg/.jpeg/.png/.webp) or a .zip archive.")
		upload_input = gr.File(
			file_count="multiple",
			file_types=[".jpg", ".jpeg", ".png", ".webp", ".zip"],
		)
		analyze_btn = gr.Button("🔍 Analyze Images", variant="primary")

	with gr.Column(visible=False) as fingerprint_panel:
		gr.Markdown("## Analyzing your images...")
		fingerprint_progress = gr.Textbox(interactive=False)

	with gr.Column(visible=False) as train_config_panel:
		gr.Markdown("## Dataset Fingerprint")
		dataset_stats = gr.Markdown("")
		lora_rank_slider = gr.Slider(4, 16, step=4, value=8, label="LoRA Rank")
		train_steps_slider = gr.Slider(200, 2000, step=100, value=1000, label="Training Steps")
		lr_slider = gr.Slider(0.00001, 0.0005, step=0.00001, value=0.0001, label="Learning Rate")
		vram_estimate_box = gr.Textbox(label="Estimated VRAM", interactive=False)
		train_btn = gr.Button("🚀 Start Fine-Tuning", variant="primary")

	with gr.Column(visible=False) as training_panel:
		gr.Markdown("## Training LoRA...")
		loss_plot = gr.LinePlot(x="step", y="loss", label="Training Loss")
		training_status = gr.Textbox(interactive=False)
		cancel_train_btn = gr.Button("⏹ Cancel", variant="stop")

	with gr.Column(visible=False) as gen_config_panel:
		gr.Markdown("## Generate Synthetic Images")
		n_slider = gr.Slider(10, 500, step=10, value=100, label="Images to Generate")
		threshold_slider = gr.Slider(0.65, 0.95, step=0.01, value=0.78, label="Similarity Threshold")
		guidance_slider = gr.Slider(1.0, 2.0, step=0.1, value=1.5, label="Guidance Scale (max 2.0)")
		time_estimate_box = gr.Textbox(label="Estimated Time", interactive=False)
		generate_btn = gr.Button("⚡ Generate", variant="primary")

	with gr.Column(visible=False) as generating_panel:
		gr.Markdown("## Generating...")
		gen_progress = gr.Textbox(interactive=False)
		live_gallery = gr.Gallery(columns=4, rows=2, label="Preview")
		cancel_gen_btn = gr.Button("⏹ Cancel", variant="stop")

	with gr.Column(visible=False) as preview_panel:
		gr.Markdown("## Results")
		results_gallery = gr.Gallery(columns=4, rows=4)
		report_box = gr.Markdown("")
		fid_box = gr.Textbox(label="FID Score", interactive=False)
		download_btn = gr.DownloadButton("⬇ Download ZIP")
		with gr.Row():
			more_btn = gr.Button("Generate More")
			reset_btn = gr.Button("Reset")

	with gr.Column(visible=False) as exporting_panel:
		gr.Markdown("## Preparing download...")
		export_status = gr.Textbox(interactive=False)

	with gr.Column(visible=False) as error_panel:
		error_title = gr.Markdown("## ❌ Something went wrong")
		error_msg = gr.Textbox(label="Error", interactive=False)
		error_suggestion = gr.Textbox(label="Suggestion", interactive=False)
		with gr.Row():
			retry_btn = gr.Button("Try Again")
			reset_btn_2 = gr.Button("Reset")

	_noop = lambda *args, **kwargs: None

	analyze_btn.click(fn=_noop)
	train_btn.click(fn=_noop)
	cancel_train_btn.click(fn=_noop)
	generate_btn.click(fn=_noop)
	cancel_gen_btn.click(fn=_noop)
	more_btn.click(fn=_noop)
	reset_btn.click(fn=_noop)
	retry_btn.click(fn=_noop)
	reset_btn_2.click(fn=_noop)


if __name__ == "__main__":
	demo.launch(share=False, server_port=7860, server_name="127.0.0.1")
