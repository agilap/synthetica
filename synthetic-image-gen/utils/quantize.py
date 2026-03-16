"""Quantization and model-loading helpers for low-VRAM execution."""

from __future__ import annotations

import logging

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from peft import LoraConfig
from transformers import BitsAndBytesConfig, CLIPTextModel


logger = logging.getLogger(__name__)


def get_nf4_config() -> BitsAndBytesConfig:
	return BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type="nf4",
		bnb_4bit_use_double_quant=True,
		bnb_4bit_compute_dtype=torch.float16,
	)


def load_quantized_unet(model_id: str) -> UNet2DConditionModel:
	try:
		unet = UNet2DConditionModel.from_pretrained(
			model_id,
			subfolder="unet",
			quantization_config=get_nf4_config(),
			torch_dtype=torch.float16,
			device_map="auto",
		)
		unet.enable_gradient_checkpointing()

		try:
			unet.enable_xformers_memory_efficient_attention()
		except Exception as exc:
			logger.warning(
				"xformers attention unavailable, falling back to AttnProcessor2_0: %s",
				exc,
			)
			unet.set_attn_processor(AttnProcessor2_0())

		return unet
	except torch.cuda.OutOfMemoryError:
		logger.exception("OOM while loading quantized UNet for model_id=%s", model_id)
		raise


def load_cpu_offloaded_vae(model_id: str) -> AutoencoderKL:
	return AutoencoderKL.from_pretrained(
		model_id,
		subfolder="vae",
		torch_dtype=torch.float16,
		device_map="cpu",
	)


def load_cpu_offloaded_text_encoder(model_id: str) -> CLIPTextModel:
	return CLIPTextModel.from_pretrained(
		model_id,
		subfolder="text_encoder",
		torch_dtype=torch.float16,
		device_map="cpu",
	)


def get_lora_config(rank: int = 8, alpha: int = 16) -> LoraConfig:
	if rank > 16:
		raise ValueError("LoRA rank > 16 risks OOM on 4 GB VRAM")

	return LoraConfig(
		r=rank,
		lora_alpha=alpha,
		target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
		lora_dropout=0.05,
		bias="none",
	)


def estimate_vram_mb(rank: int = 8) -> dict[str, int | bool]:
	unet_nf4 = 1700
	lora_adapter = rank * 6 // 8 * 2
	activations_grad_ckpt = 600
	optimizer_8bit = 300
	vae_brief = 200
	total_estimate = unet_nf4 + lora_adapter + activations_grad_ckpt + optimizer_8bit + vae_brief
	vram_budget = 4096

	return {
		"unet_nf4": unet_nf4,
		"lora_adapter": lora_adapter,
		"activations_grad_ckpt": activations_grad_ckpt,
		"optimizer_8bit": optimizer_8bit,
		"vae_brief": vae_brief,
		"total_estimate": total_estimate,
		"vram_budget": vram_budget,
		"fits": total_estimate < 3800,
	}
