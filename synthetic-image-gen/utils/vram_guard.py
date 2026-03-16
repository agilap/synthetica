"""VRAM guard utilities for CUDA-aware memory monitoring and cleanup."""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager

import torch


logger = logging.getLogger(__name__)


def vram_used_mb() -> int:
	if not torch.cuda.is_available():
		return 0
	return int(torch.cuda.memory_allocated() // (1024**2))


def vram_peak_mb() -> int:
	if not torch.cuda.is_available():
		return 0
	return int(torch.cuda.max_memory_allocated() // (1024**2))


def reset_peak() -> None:
	if torch.cuda.is_available():
		torch.cuda.reset_peak_memory_stats()


def flush() -> None:
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
	gc.collect()


@contextmanager
def phase_boundary(phase_name: str):
	reset_peak()
	logger.info("[%s] start | free_vram_mb=%d", phase_name, get_vram_status()["free_mb"])
	try:
		yield
	except Exception as exc:
		flush()
		logger.exception("[%s] failed | exception=%s", phase_name, type(exc).__name__)
		raise
	else:
		flush()
		logger.info("[%s] complete | peak_vram_mb=%d", phase_name, vram_peak_mb())


def get_vram_status() -> dict[str, bool | int | str]:
	if not torch.cuda.is_available():
		return {
			"available": False,
			"used_mb": 0,
			"peak_mb": 0,
			"total_mb": 0,
			"free_mb": 0,
			"device_name": "CPU (no CUDA)",
		}

	used_mb = vram_used_mb()
	peak_mb = vram_peak_mb()
	total_mb = int(torch.cuda.get_device_properties(0).total_memory // (1024**2))
	free_mb = max(0, total_mb - used_mb)
	device_name = torch.cuda.get_device_name(0)

	return {
		"available": True,
		"used_mb": used_mb,
		"peak_mb": peak_mb,
		"total_mb": total_mb,
		"free_mb": free_mb,
		"device_name": device_name,
	}


def assert_vram_fits(estimated_mb: int, label: str = "") -> None:
	if not torch.cuda.is_available():
		return

	status = get_vram_status()
	free_mb = int(status["free_mb"])
	threshold_warn = free_mb - 500
	threshold_fail = free_mb - 200
	tag = f"[{label}] " if label else ""

	if estimated_mb > threshold_warn:
		logger.warning(
			"%sEstimated VRAM (%d MB) is close to free VRAM headroom (%d MB free).",
			tag,
			estimated_mb,
			free_mb,
		)

	if estimated_mb > threshold_fail:
		raise RuntimeError(
			f"{tag}Estimated VRAM {estimated_mb} MB exceeds safe limit with current free VRAM {free_mb} MB"
		)
