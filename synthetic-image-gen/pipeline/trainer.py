"""LoRA trainer pipeline for SyntheticImageGen."""

from __future__ import annotations

import gc
import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, cast

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from peft import get_peft_model
from torch.nn.utils import clip_grad_norm_
from transformers import CLIPTokenizer, get_cosine_schedule_with_warmup

from utils.quantize import (
	get_lora_config,
	load_cpu_offloaded_text_encoder,
	load_cpu_offloaded_vae,
	load_quantized_unet,
)
from utils.types import ErrorState, TrainingResult


logger = logging.getLogger(__name__)


def _preprocess_image(image: object) -> torch.Tensor:
	pil_image = image.convert("RGB")
	if pil_image.size != (256, 256):
		pil_image = pil_image.resize((256, 256))

	array = np.asarray(pil_image, dtype=np.float32)
	tensor = torch.from_numpy(array).permute(2, 0, 1)
	tensor = tensor / 127.5 - 1.0
	return tensor.contiguous()


def train(
	images: list,
	model_id: str,
	output_dir: Path,
	steps: int = 1000,
	lr: float = 1e-4,
	rank: int = 8,
	alpha: int = 16,
	progress_callback: Callable[[int, float], None] | None = None,
	cancel_flag: threading.Event | None = None,
) -> tuple[TrainingResult, ErrorState | None]:
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	if not images:
		return cast(TrainingResult, None), ErrorState(
			phase="TRAINING",
			error_type="io_error",
			error_message="No images provided for training.",
			recoverable=False,
			recovery_suggestion="Provide at least one valid image before training.",
			traceback=None,
		)

	if steps <= 0:
		return cast(TrainingResult, None), ErrorState(
			phase="TRAINING",
			error_type="validation",
			error_message="Training steps must be greater than zero.",
			recoverable=True,
			recovery_suggestion="Set steps to a positive integer value.",
			traceback=None,
		)

	training_device = "cuda" if torch.cuda.is_available() else "cpu"
	logger.info("Training device: %s", training_device)

	unet = None
	vae = None
	text_encoder = None
	optimizer = None
	lr_scheduler = None
	final_checkpoint = output_dir / "final"
	loss_history: list[tuple[int, float]] = []
	nan_streak = 0
	last_loss = 0.0
	steps_completed = 0
	start = time.perf_counter()

	try:
		if torch.cuda.is_available():
			torch.cuda.reset_peak_memory_stats()

		unet = load_quantized_unet(model_id)
		unet = get_peft_model(unet, get_lora_config(rank=rank, alpha=alpha))
		unet.train()

		tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
		text_encoder = load_cpu_offloaded_text_encoder(model_id)
		text_encoder.eval()
		vae = load_cpu_offloaded_vae(model_id)
		vae.eval()

		scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

		prompt = "photo of [V]"
		tokenized = tokenizer(
			prompt,
			padding="max_length",
			truncation=True,
			max_length=tokenizer.model_max_length,
			return_tensors="pt",
		)
		input_ids = tokenized.input_ids.squeeze(0)

		dataset: list[dict[str, torch.Tensor]] = [
			{
				"pixel_values": _preprocess_image(image),
				"input_ids": input_ids.clone(),
			}
			for image in images
		]

		trainable_params = [param for param in unet.parameters() if param.requires_grad]
		optimizer = bnb.optim.AdamW8bit(trainable_params, lr=lr, weight_decay=0.01)

		warmup_steps = min(100, steps // 10)
		lr_scheduler = get_cosine_schedule_with_warmup(
			optimizer=optimizer,
			num_warmup_steps=warmup_steps,
			num_training_steps=steps,
		)

		for step in range(steps):
			if cancel_flag is not None and cancel_flag.is_set():
				cancel_dir = output_dir / f"cancelled_step_{step}"
				cancel_dir.mkdir(parents=True, exist_ok=True)
				unet.save_pretrained(cancel_dir)
				logger.info("Training cancelled at step %d", step)
				break

			sample = dataset[step % len(dataset)]
			pixel_values = sample["pixel_values"].unsqueeze(0)
			input_ids_step = sample["input_ids"].unsqueeze(0)

			pixel_values = pixel_values.to(training_device, dtype=torch.float16)

			if training_device == "cuda":
				vae = vae.to("cuda")
			latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
			if training_device == "cuda":
				vae = vae.to("cpu")
				torch.cuda.empty_cache()

			noise = torch.randn_like(latents)
			timesteps = torch.randint(
				low=0,
				high=scheduler.config.num_train_timesteps,
				size=(latents.shape[0],),
				device=latents.device,
				dtype=torch.long,
			)
			noisy_latents = scheduler.add_noise(latents, noise, timesteps)

			if training_device == "cuda":
				text_encoder = text_encoder.to("cuda")
				encoder_hidden_states = text_encoder(input_ids_step.to("cuda"))[0]
				text_encoder = text_encoder.to("cpu")
				torch.cuda.empty_cache()
			else:
				encoder_hidden_states = text_encoder(input_ids_step)[0]

			noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
			loss = F.mse_loss(noise_pred.float(), noise.float())

			loss_value = float(loss.item())
			loss_history.append((step, loss_value))

			if torch.isnan(loss):
				nan_streak += 1
				logger.warning("NaN loss at step %d (streak=%d)", step, nan_streak)
				optimizer.zero_grad(set_to_none=True)
				if nan_streak >= 3:
					return cast(TrainingResult, None), ErrorState(
						phase="TRAINING",
						error_type="NaN_loss",
						error_message="Encountered 3 consecutive NaN losses.",
						recoverable=True,
						recovery_suggestion="Lower learning rate or reduce LoRA rank and retry.",
						traceback=None,
					)
				continue
			nan_streak = 0

			loss.backward()
			clip_grad_norm_(trainable_params, 1.0)
			optimizer.step()
			lr_scheduler.step()
			optimizer.zero_grad(set_to_none=True)

			if step % 200 == 0:
				step_dir = output_dir / f"step_{step}"
				step_dir.mkdir(parents=True, exist_ok=True)
				unet.save_pretrained(step_dir)

			if progress_callback is not None:
				progress_callback(step, loss_value)

			last_loss = loss_value
			steps_completed = step + 1

		final_checkpoint.mkdir(parents=True, exist_ok=True)
		unet.save_pretrained(final_checkpoint)

		peak_vram_mb = (
			int(torch.cuda.max_memory_allocated() // (1024**2)) if torch.cuda.is_available() else 0
		)

		return TrainingResult(
			checkpoint_path=final_checkpoint,
			final_loss=last_loss,
			steps_completed=steps_completed,
			peak_vram_mb=peak_vram_mb,
			training_time_s=float(time.perf_counter() - start),
			loss_history=loss_history,
		), None

	except torch.cuda.OutOfMemoryError:
		logger.exception("CUDA OOM during training")
		return cast(TrainingResult, None), ErrorState(
			phase="TRAINING",
			error_type="OOM",
			error_message="Ran out of VRAM during training.",
			recoverable=True,
			recovery_suggestion="Reduce LoRA rank to 4.",
			traceback=traceback.format_exc(),
		)
	except Exception:
		logger.exception("Unhandled training failure")
		return cast(TrainingResult, None), ErrorState(
			phase="TRAINING",
			error_type="io_error",
			error_message="Unexpected training failure.",
			recoverable=False,
			recovery_suggestion="Check inputs and environment, then retry.",
			traceback=traceback.format_exc(),
		)
	finally:
		del unet, vae, text_encoder, optimizer, lr_scheduler
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		gc.collect()
