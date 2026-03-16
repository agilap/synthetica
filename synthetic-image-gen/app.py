"""SyntheticImageGen Gradio app with pipeline handler wiring."""

from __future__ import annotations

import queue
import shutil
import threading
import uuid
import zipfile
from itertools import zip_longest
from pathlib import Path

import gradio as gr
import numpy as np
import yaml

from pipeline.exporter import export_dataset
from pipeline.generator import generate
from pipeline.ingestor import ingest
from pipeline.trainer import train
from utils.fingerprint import extract_fingerprints
from utils.quantize import estimate_vram_mb
from utils.similarity import compute_adaptive_threshold, compute_fid_proxy
from utils.vram_guard import get_vram_status


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "defaults.yaml"
SESSIONS_DIR = BASE_DIR / "sessions"
DEFAULT_CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}


def _new_state(session_id: str | None = None) -> dict:
	sid = session_id or str(uuid.uuid4())
	return {
		"phase": "IDLE",
		"session_id": sid,
		"session_dir": str(SESSIONS_DIR / sid),
		"cancel_flag": None,
		"last_failed_phase": "",
	}


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


def _vram_badge_text() -> str:
	status = get_vram_status()
	if not status["available"]:
		return "CPU (no CUDA) | Used: 0 MB | Free: 0 MB | Peak: 0 MB"
	return (
		f"{status['device_name']} | Used: {status['used_mb']} MB | "
		f"Free: {status['free_mb']} MB | Peak: {status['peak_mb']} MB"
	)


def _set_error_state(state: dict, error) -> dict:
	state["phase"] = "ERROR"
	state["last_failed_phase"] = error.phase
	state["last_error_message"] = error.error_message
	state["last_error_suggestion"] = error.recovery_suggestion
	return state


def _as_paths(files) -> list[Path]:
	if files is None:
		return []
	file_items = files if isinstance(files, list) else [files]
	paths: list[Path] = []
	for item in file_items:
		name = getattr(item, "name", item)
		if name:
			paths.append(Path(name))
	return paths


def _interleave_gallery(real_images: list, synthetic_images: list) -> list:
	gallery: list = []
	for real_img, synthetic_img in zip_longest(real_images, synthetic_images):
		if real_img is not None:
			gallery.append(real_img)
		if synthetic_img is not None:
			gallery.append(synthetic_img)
	return gallery


def handle_upload(files, state: dict):
	state = dict(state or _new_state())
	session_dir = Path(state.get("session_dir", SESSIONS_DIR / state["session_id"]))
	real_dir = session_dir / "real"
	real_dir.mkdir(parents=True, exist_ok=True)

	file_paths = _as_paths(files)
	if not file_paths:
		state["last_failed_phase"] = "INGESTING"
		return (
			*show_only("error"),
			"",
			gr.update(),
			state,
			_vram_badge_text(),
			"No files uploaded.",
			"Upload at least 15 images or a ZIP archive.",
		)

	for src in file_paths:
		if src.suffix.lower() == ".zip":
			with zipfile.ZipFile(src, "r") as zf:
				zf.extractall(real_dir)
		else:
			dst = real_dir / src.name
			shutil.copy2(src, dst)

	ingest_result, ingest_error = ingest(real_dir)
	if ingest_error is not None:
		state = _set_error_state(state, ingest_error)
		return (
			*show_only("error"),
			"",
			gr.update(),
			state,
			_vram_badge_text(),
			ingest_error.error_message,
			ingest_error.recovery_suggestion,
		)

	fingerprint_result, fp_error = extract_fingerprints(ingest_result.images)
	if fp_error is not None:
		state = _set_error_state(state, fp_error)
		return (
			*show_only("error"),
			"",
			gr.update(),
			state,
			_vram_badge_text(),
			fp_error.error_message,
			fp_error.recovery_suggestion,
		)

	centroid_path = session_dir / "centroid.npy"
	session_dir.mkdir(parents=True, exist_ok=True)
	np.save(centroid_path, fingerprint_result.centroid)

	adaptive_threshold = compute_adaptive_threshold(fingerprint_result.embeddings)
	state.update(
		{
			"phase": "READY_TO_TRAIN",
			"session_dir": str(session_dir),
			"real_count": ingest_result.count,
			"centroid_path": str(centroid_path),
			"real_paths": [str(p) for p in ingest_result.paths],
			"real_images": ingest_result.images,
			"fingerprint_result": fingerprint_result,
			"adaptive_threshold": float(adaptive_threshold),
			"model_id": DEFAULT_CONFIG.get("model_id", "runwayml/stable-diffusion-v1-5"),
		}
	)

	stats = (
		f"**Valid images:** {ingest_result.count}  \n"
		f"**Rejected:** {ingest_result.rejected}  \n"
		f"**Avg original size:** {ingest_result.avg_original_size[0]}×{ingest_result.avg_original_size[1]}  \n"
		f"**Centroid std:** {fingerprint_result.centroid_std:.4f}  \n"
		f"**Intra-set mean sim:** {fingerprint_result.intra_set_mean_sim:.4f}"
	)

	return (
		*show_only("train_config"),
		stats,
		gr.update(value=float(adaptive_threshold)),
		state,
		_vram_badge_text(),
		"",
		"",
	)


def handle_start_training(rank, steps, lr, state: dict, progress=gr.Progress()):
	state = dict(state or _new_state())
	session_dir = Path(state.get("session_dir", SESSIONS_DIR / state["session_id"]))
	train_out = session_dir / "training"
	train_out.mkdir(parents=True, exist_ok=True)

	cancel_event = threading.Event()
	state["cancel_flag"] = cancel_event
	state["phase"] = "TRAINING"

	model_id = state.get("model_id", DEFAULT_CONFIG.get("model_id", "runwayml/stable-diffusion-v1-5"))
	images = state.get("real_images", [])

	q: queue.Queue[tuple[int, float]] = queue.Queue()
	holder: dict[str, object] = {"result": None, "error": None}
	loss_points: list[dict[str, float]] = []

	def _on_progress(step: int, loss: float) -> None:
		q.put((step, loss))

	def _worker() -> None:
		result, error = train(
			images=images,
			model_id=model_id,
			output_dir=train_out,
			steps=int(steps),
			lr=float(lr),
			rank=int(rank),
			alpha=int(rank) * 2,
			progress_callback=_on_progress,
			cancel_flag=cancel_event,
		)
		holder["result"] = result
		holder["error"] = error

	worker = threading.Thread(target=_worker, daemon=True)
	worker.start()

	yield (
		*show_only("training"),
		gr.update(value=[]),
		"Training started...",
		gr.update(value=float(state.get("adaptive_threshold", DEFAULT_CONFIG.get("similarity_threshold", 0.78)))),
		state,
		_vram_badge_text(),
		"",
		"",
	)

	while worker.is_alive() or not q.empty():
		try:
			step, loss = q.get(timeout=0.2)
			loss_points.append({"step": int(step), "loss": float(loss)})
			progress(min((int(step) + 1) / max(int(steps), 1), 1.0), desc=f"Training step {int(step) + 1}/{int(steps)}")
			yield (
				*show_only("training"),
				gr.update(value=loss_points),
				f"Step {int(step) + 1}/{int(steps)}  |  Loss: {float(loss):.6f}",
				gr.update(value=float(state.get("adaptive_threshold", DEFAULT_CONFIG.get("similarity_threshold", 0.78)))),
				state,
				_vram_badge_text(),
				"",
				"",
			)
		except queue.Empty:
			continue

	error = holder.get("error")
	if error is not None:
		state = _set_error_state(state, error)
		yield (
			*show_only("error"),
			gr.update(value=loss_points),
			"Training failed.",
			gr.update(),
			state,
			_vram_badge_text(),
			error.error_message,
			error.recovery_suggestion,
		)
		return

	result = holder.get("result")
	state.update(
		{
			"phase": "READY_TO_GENERATE",
			"checkpoint_path": str(result.checkpoint_path),
			"training_steps": int(result.steps_completed),
			"lora_rank": int(rank),
			"training_result": result,
			"cancel_flag": None,
		}
	)
	threshold_default = float(state.get("adaptive_threshold", DEFAULT_CONFIG.get("similarity_threshold", 0.78)))

	yield (
		*show_only("gen_config"),
		gr.update(value=loss_points),
		f"Training complete. Final loss: {float(result.final_loss):.6f}",
		gr.update(value=threshold_default),
		state,
		_vram_badge_text(),
		"",
		"",
	)


def handle_cancel_training(state: dict):
	state = dict(state or _new_state())
	cancel_flag = state.get("cancel_flag")
	if isinstance(cancel_flag, threading.Event):
		cancel_flag.set()
	return "Cancellation requested.", state, _vram_badge_text()


def handle_start_generation(n, threshold, guidance, state: dict, progress=gr.Progress()):
	state = dict(state or _new_state())
	session_dir = Path(state.get("session_dir", SESSIONS_DIR / state["session_id"]))
	synthetic_dir = session_dir / "synthetic"
	synthetic_dir.mkdir(parents=True, exist_ok=True)

	cancel_event = threading.Event()
	state["cancel_flag"] = cancel_event
	state["phase"] = "GENERATING"

	q: queue.Queue[tuple[int, int]] = queue.Queue()
	holder: dict[str, object] = {"result": None, "error": None}

	def _on_progress(accepted: int, generated: int) -> None:
		q.put((accepted, generated))

	def _worker() -> None:
		result, error = generate(
			checkpoint_path=Path(state["checkpoint_path"]),
			model_id=state.get("model_id", DEFAULT_CONFIG.get("model_id", "runwayml/stable-diffusion-v1-5")),
			fingerprint_result=state.get("fingerprint_result"),
			n_target=int(n),
			guidance_scale=float(guidance),
			inference_steps=int(DEFAULT_CONFIG.get("inference_steps", 4)),
			similarity_threshold=float(threshold),
			output_dir=synthetic_dir,
			progress_callback=_on_progress,
			cancel_flag=cancel_event,
		)
		holder["result"] = result
		holder["error"] = error

	worker = threading.Thread(target=_worker, daemon=True)
	worker.start()

	last_gallery_emit = 0
	yield (
		*show_only("generating"),
		gr.update(value=[]),
		"Generating started...",
		gr.update(),
		gr.update(),
		gr.update(value=""),
		state,
		_vram_badge_text(),
		"",
		"",
	)

	while worker.is_alive() or not q.empty():
		try:
			accepted, generated = q.get(timeout=0.2)
			progress(min(int(accepted) / max(int(n), 1), 1.0), desc=f"Accepted {int(accepted)}/{int(n)}")
			gallery_update = gr.update()
			if int(accepted) - last_gallery_emit >= 10:
				current = sorted(synthetic_dir.glob("syn_*.jpg"))[-8:]
				gallery_update = gr.update(value=current)
				last_gallery_emit = int(accepted)
			yield (
				*show_only("generating"),
				gallery_update,
				f"Accepted {int(accepted)} / {int(n)} | Generated {int(generated)}",
				gr.update(),
				gr.update(),
				gr.update(value=""),
				state,
				_vram_badge_text(),
				"",
				"",
			)
		except queue.Empty:
			continue

	error = holder.get("error")
	if error is not None:
		state = _set_error_state(state, error)
		yield (
			*show_only("error"),
			gr.update(value=[]),
			"Generation failed.",
			gr.update(value=[]),
			gr.update(value=""),
			gr.update(value=""),
			state,
			_vram_badge_text(),
			error.error_message,
			error.recovery_suggestion,
		)
		return

	result = holder.get("result")
	if int(n) >= 50:
		fid = compute_fid_proxy(state.get("real_images", []), result.accepted_images)
		result.fid_estimate = fid

	state.update(
		{
			"phase": "READY_TO_EXPORT",
			"generation_result": result,
			"cancel_flag": None,
		}
	)

	interleaved = _interleave_gallery(state.get("real_images", []), result.accepted_images)
	report_md = (
		f"**Accepted:** {result.n_accepted}  \n"
		f"**Generated:** {result.n_generated}  \n"
		f"**Rejected:** {result.n_rejected}  \n"
		f"**Mean similarity:** {result.mean_similarity:.4f}"
	)
	fid_text = "N/A" if result.fid_estimate is None else f"{float(result.fid_estimate):.4f}"

	yield (
		*show_only("preview"),
		gr.update(value=[]),
		f"Generation complete. Accepted {result.n_accepted} images.",
		gr.update(value=interleaved),
		gr.update(value=report_md),
		gr.update(value=fid_text),
		state,
		_vram_badge_text(),
		"",
		"",
	)


def handle_cancel_generation(state: dict):
	state = dict(state or _new_state())
	cancel_flag = state.get("cancel_flag")
	if isinstance(cancel_flag, threading.Event):
		cancel_flag.set()
	return "Cancellation requested.", state, _vram_badge_text()


def handle_download(state: dict):
	state = dict(state or _new_state())
	generation_result = state.get("generation_result")
	if generation_result is None:
		return (
			gr.update(),
			"No generation results available for export.",
			gr.update(),
			state,
			_vram_badge_text(),
			*show_only("preview"),
			"",
			"",
		)

	session_dir = Path(state["session_dir"])
	zip_path, dataset_report, error = export_dataset(
		session_dir=session_dir,
		generation_result=generation_result,
		real_image_paths=[Path(p) for p in state.get("real_paths", [])],
		training_steps=int(state.get("training_steps", 0)),
		lora_rank=int(state.get("lora_rank", DEFAULT_CONFIG.get("lora_rank", 8))),
		model_id=state.get("model_id", DEFAULT_CONFIG.get("model_id", "runwayml/stable-diffusion-v1-5")),
		fingerprint_result=state.get("fingerprint_result"),
	)

	if error is not None:
		state = _set_error_state(state, error)
		return (
			gr.update(),
			"Export failed.",
			gr.update(),
			state,
			_vram_badge_text(),
			*show_only("error"),
			error.error_message,
			error.recovery_suggestion,
		)

	state["zip_path"] = str(zip_path)
	state["dataset_report"] = dataset_report.to_dict()

	report_md = (
		f"**Export ready:** {zip_path.name}  \n"
		f"**Synthetic:** {dataset_report.synthetic_count}  \n"
		f"**Acceptance rate:** {dataset_report.acceptance_rate:.2%}"
	)

	return (
		gr.update(value=str(zip_path)),
		"Export complete.",
		gr.update(value=report_md),
		state,
		_vram_badge_text(),
		*show_only("preview"),
		"",
		"",
	)


def handle_generate_more(state: dict):
	state = dict(state or _new_state())
	if not state.get("checkpoint_path"):
		return *show_only("error"), state, _vram_badge_text(), "No checkpoint found.", "Train a model first."
	state["phase"] = "READY_TO_GENERATE"
	return *show_only("gen_config"), state, _vram_badge_text(), "", ""


def handle_reset(state: dict):
	old_state = dict(state or {})
	old_session_dir = old_state.get("session_dir")
	if old_session_dir:
		shutil.rmtree(Path(old_session_dir), ignore_errors=True)

	new_state = _new_state()
	return (
		*show_only("upload"),
		new_state,
		_vram_badge_text(),
		gr.update(value=""),
		gr.update(value=[]),
		gr.update(value=[]),
		gr.update(value=""),
		gr.update(value=""),
		gr.update(value=""),
		"",
		"",
	)


def handle_try_again(state: dict):
	state = dict(state or _new_state())
	phase = state.get("last_failed_phase", "")
	route = {
		"INGESTING": "upload",
		"FINGERPRINTING": "upload",
		"TRAINING": "train_config",
		"GENERATING": "gen_config",
		"EXPORTING": "preview",
	}
	return *show_only(route.get(phase, "upload")), state, _vram_badge_text(), "", ""


with gr.Blocks(theme=gr.themes.Soft()) as demo:
	gr.Markdown("# SyntheticImageGen 🧬")
	gr.Markdown("> Real Dataset → Synthetic Look-Alike Generator")
	vram_badge = gr.Textbox(label="GPU Status", interactive=False, value=_vram_badge_text())

	app_state = gr.State(_new_state())

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
		lora_rank_slider = gr.Slider(4, 16, step=4, value=int(DEFAULT_CONFIG.get("lora_rank", 8)), label="LoRA Rank")
		train_steps_slider = gr.Slider(
			200,
			2000,
			step=100,
			value=int(DEFAULT_CONFIG.get("train_steps", 1000)),
			label="Training Steps",
		)
		lr_slider = gr.Slider(
			0.00001,
			0.0005,
			step=0.00001,
			value=float(DEFAULT_CONFIG.get("learning_rate", 0.0001)),
			label="Learning Rate",
		)
		vram_estimate_box = gr.Textbox(
			label="Estimated VRAM",
			interactive=False,
			value=str(estimate_vram_mb(int(DEFAULT_CONFIG.get("lora_rank", 8)))),
		)
		train_btn = gr.Button("🚀 Start Fine-Tuning", variant="primary")

	with gr.Column(visible=False) as training_panel:
		gr.Markdown("## Training LoRA...")
		loss_plot = gr.LinePlot(x="step", y="loss", label="Training Loss")
		training_status = gr.Textbox(interactive=False)
		cancel_train_btn = gr.Button("⏹ Cancel", variant="stop")

	with gr.Column(visible=False) as gen_config_panel:
		gr.Markdown("## Generate Synthetic Images")
		n_slider = gr.Slider(10, 500, step=10, value=100, label="Images to Generate")
		threshold_slider = gr.Slider(
			0.65,
			0.95,
			step=0.01,
			value=float(DEFAULT_CONFIG.get("similarity_threshold", 0.78)),
			label="Similarity Threshold",
		)
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

	analyze_btn.click(
		fn=handle_upload,
		inputs=[upload_input, app_state],
		outputs=[
			upload_panel,
			fingerprint_panel,
			train_config_panel,
			training_panel,
			gen_config_panel,
			generating_panel,
			preview_panel,
			exporting_panel,
			error_panel,
			dataset_stats,
			threshold_slider,
			app_state,
			vram_badge,
			error_msg,
			error_suggestion,
		],
	)

	train_btn.click(
		fn=handle_start_training,
		inputs=[lora_rank_slider, train_steps_slider, lr_slider, app_state],
		outputs=[
			upload_panel,
			fingerprint_panel,
			train_config_panel,
			training_panel,
			gen_config_panel,
			generating_panel,
			preview_panel,
			exporting_panel,
			error_panel,
			loss_plot,
			training_status,
			threshold_slider,
			app_state,
			vram_badge,
			error_msg,
			error_suggestion,
		],
	)

	cancel_train_btn.click(
		fn=handle_cancel_training,
		inputs=[app_state],
		outputs=[training_status, app_state, vram_badge],
	)

	generate_btn.click(
		fn=handle_start_generation,
		inputs=[n_slider, threshold_slider, guidance_slider, app_state],
		outputs=[
			upload_panel,
			fingerprint_panel,
			train_config_panel,
			training_panel,
			gen_config_panel,
			generating_panel,
			preview_panel,
			exporting_panel,
			error_panel,
			live_gallery,
			gen_progress,
			results_gallery,
			report_box,
			fid_box,
			app_state,
			vram_badge,
			error_msg,
			error_suggestion,
		],
	)

	cancel_gen_btn.click(
		fn=handle_cancel_generation,
		inputs=[app_state],
		outputs=[gen_progress, app_state, vram_badge],
	)

	download_btn.click(
		fn=handle_download,
		inputs=[app_state],
		outputs=[
			download_btn,
			export_status,
			report_box,
			app_state,
			vram_badge,
			upload_panel,
			fingerprint_panel,
			train_config_panel,
			training_panel,
			gen_config_panel,
			generating_panel,
			preview_panel,
			exporting_panel,
			error_panel,
			error_msg,
			error_suggestion,
		],
	)

	more_btn.click(
		fn=handle_generate_more,
		inputs=[app_state],
		outputs=[
			upload_panel,
			fingerprint_panel,
			train_config_panel,
			training_panel,
			gen_config_panel,
			generating_panel,
			preview_panel,
			exporting_panel,
			error_panel,
			app_state,
			vram_badge,
			error_msg,
			error_suggestion,
		],
	)

	reset_btn.click(
		fn=handle_reset,
		inputs=[app_state],
		outputs=[
			upload_panel,
			fingerprint_panel,
			train_config_panel,
			training_panel,
			gen_config_panel,
			generating_panel,
			preview_panel,
			exporting_panel,
			error_panel,
			app_state,
			vram_badge,
			dataset_stats,
			loss_plot,
			results_gallery,
			report_box,
			fid_box,
			export_status,
			error_msg,
			error_suggestion,
		],
	)

	retry_btn.click(
		fn=handle_try_again,
		inputs=[app_state],
		outputs=[
			upload_panel,
			fingerprint_panel,
			train_config_panel,
			training_panel,
			gen_config_panel,
			generating_panel,
			preview_panel,
			exporting_panel,
			error_panel,
			app_state,
			vram_badge,
			error_msg,
			error_suggestion,
		],
	)

	reset_btn_2.click(
		fn=handle_reset,
		inputs=[app_state],
		outputs=[
			upload_panel,
			fingerprint_panel,
			train_config_panel,
			training_panel,
			gen_config_panel,
			generating_panel,
			preview_panel,
			exporting_panel,
			error_panel,
			app_state,
			vram_badge,
			dataset_stats,
			loss_plot,
			results_gallery,
			report_box,
			fid_box,
			export_status,
			error_msg,
			error_suggestion,
		],
	)

	upload_input.upload(fn=lambda: _vram_badge_text(), inputs=None, outputs=[vram_badge])


if __name__ == "__main__":
	demo.launch(share=False, server_port=7860, server_name="127.0.0.1")
