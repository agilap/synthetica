"""Synthetic image generation pipeline for SyntheticImageGen."""

from __future__ import annotations

import gc
import logging
import random
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, cast

import torch
from diffusers import LCMScheduler, StableDiffusionPipeline

from pipeline.filter import filter_images
from utils.types import ErrorState, GenerationResult


logger = logging.getLogger(__name__)


def generate(
	checkpoint_path: Path,
	model_id: str,
	fingerprint_result,
	n_target: int = 100,
	guidance_scale: float = 1.5,
	inference_steps: int = 4,
	similarity_threshold: float = 0.78,
	output_dir: Path | None = None,
	progress_callback: Callable[[int, int], None] | None = None,
	cancel_flag: threading.Event | None = None,
) -> tuple[GenerationResult, ErrorState | None]:
	if not (1 <= inference_steps <= 8):
		return cast(GenerationResult, None), ErrorState(
			phase="GENERATING",
			error_type="validation",
			error_message="inference_steps must be in the LCM range 1-8.",
			recoverable=True,
			recovery_suggestion="Use inference_steps=4 (recommended for LCM).",
			traceback=None,
		)
	if not (1.0 <= guidance_scale <= 2.0):
		return cast(GenerationResult, None), ErrorState(
			phase="GENERATING",
			error_type="validation",
			error_message="guidance_scale must be in the LCM-safe range 1.0-2.0.",
			recoverable=True,
			recovery_suggestion="Use guidance_scale between 1.0 and 2.0.",
			traceback=None,
		)
	if n_target <= 0:
		return cast(GenerationResult, None), ErrorState(
			phase="GENERATING",
			error_type="validation",
			error_message="n_target must be greater than zero.",
			recoverable=True,
			recovery_suggestion="Set n_target to a positive integer.",
			traceback=None,
		)

	checkpoint_path = Path(checkpoint_path)
	if output_dir is not None:
		output_dir = Path(output_dir)
		output_dir.mkdir(parents=True, exist_ok=True)

	pipe = None
	accepted: list = []
	accepted_paths: list[Path] = []
	metadata: list[dict] = []
	n_generated = 0
	consecutive_reject_batches = 0
	start_time = time.time()

	try:
		pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
		pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
		pipe.unet.load_attn_procs(checkpoint_path)
		pipe.enable_attention_slicing(slice_size=1)
		pipe.enable_vae_tiling()
		pipe.enable_model_cpu_offload()

		while len(accepted) < n_target:
			if cancel_flag is not None and cancel_flag.is_set():
				logger.info("Generation cancelled at %d accepted / %d generated", len(accepted), n_generated)
				break

			seeds = [random.randint(0, 2**32 - 1) for _ in range(4)]
			generators = [torch.Generator("cpu").manual_seed(seed) for seed in seeds]
			t0 = time.time()

			images = pipe(
				prompt="photo of [V]",
				num_inference_steps=inference_steps,
				guidance_scale=guidance_scale,
				num_images_per_prompt=4,
				generator=generators,
			).images

			batch_ms = int((time.time() - t0) * 1000)
			n_generated += 4

			passed_imgs, passed_scores, passed_seeds = filter_images(
				images,
				seeds,
				fingerprint_result,
				similarity_threshold,
			)

			consecutive_reject_batches = 0 if passed_imgs else consecutive_reject_batches + 1
			if consecutive_reject_batches > 50:
				return cast(GenerationResult, None), ErrorState(
					phase="GENERATING",
					error_type="rejection_overflow",
					error_message="50+ consecutive batches rejected.",
					recoverable=False,
					recovery_suggestion="Re-train with more steps or lower learning rate.",
					traceback=None,
				)

			for img, score, seed in zip(passed_imgs, passed_scores, passed_seeds):
				if len(accepted) >= n_target:
					break

				idx = len(accepted)
				fname = f"syn_{idx:04d}.jpg"

				if output_dir is not None:
					save_path = output_dir / fname
					img.save(save_path)
				else:
					save_path = Path(fname)

				accepted.append(img)
				accepted_paths.append(save_path)
				metadata.append(
					{
						"seed": seed,
						"similarity_score": round(float(score), 4),
						"generation_time_ms": batch_ms,
						"model": model_id,
						"lora_checkpoint": str(checkpoint_path),
						"resolution": [256, 256],
						"prompt": "photo of [V]",
						"threshold_used": similarity_threshold,
						"filename": fname,
					}
				)

			if progress_callback is not None:
				progress_callback(len(accepted), n_generated)

		scores = [float(item["similarity_score"]) for item in metadata if "similarity_score" in item]
		mean_similarity = float(sum(scores) / len(scores)) if scores else 0.0

		return GenerationResult(
			accepted_images=accepted,
			accepted_paths=accepted_paths,
			metadata=metadata,
			n_generated=n_generated,
			n_accepted=len(accepted),
			n_rejected=max(0, n_generated - len(accepted)),
			mean_similarity=mean_similarity,
			fid_estimate=None,
			total_time_s=float(time.time() - start_time),
		), None

	except torch.cuda.OutOfMemoryError:
		return cast(GenerationResult, None), ErrorState(
			phase="GENERATING",
			error_type="OOM",
			error_message="Ran out of VRAM during generation.",
			recoverable=True,
			recovery_suggestion="Reduce target count per run or retry after clearing GPU memory.",
			traceback=traceback.format_exc(),
		)
	except Exception:
		return cast(GenerationResult, None), ErrorState(
			phase="GENERATING",
			error_type="io_error",
			error_message="Unexpected generation failure.",
			recoverable=False,
			recovery_suggestion="Check checkpoint path and runtime environment, then retry.",
			traceback=traceback.format_exc(),
		)
	finally:
		del pipe
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		gc.collect()
