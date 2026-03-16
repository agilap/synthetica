"""DINOv2 fingerprint extraction utilities."""

from __future__ import annotations

import gc
import logging
import traceback
from typing import cast

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from utils.types import ErrorState, FingerprintResult


logger = logging.getLogger(__name__)


def _chunked(items: list, size: int) -> list[list]:
	return [items[i : i + size] for i in range(0, len(items), size)]


def _l2_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
	norms = np.where(norms == 0.0, 1.0, norms)
	return embeddings / norms


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
	norm = np.linalg.norm(vector)
	if norm == 0.0:
		return vector
	return vector / norm


def _extract_on_device(images: list, device: str) -> FingerprintResult:
	processor = None
	model = None
	inputs = None
	batch_embeddings: list[np.ndarray] = []

	try:
		logger.info("Fingerprint extraction device: %s", device)

		if device == "cuda":
			torch.cuda.reset_peak_memory_stats()

		processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
		model = AutoModel.from_pretrained("facebook/dinov2-small", torch_dtype=torch.float16)
		model = model.to(device)
		model.eval()

		batches = _chunked(images, 8)
		for idx, batch in enumerate(batches, start=1):
			logger.info("Fingerprint batch %d/%d", idx, len(batches))
			inputs = processor(images=batch, return_tensors="pt").to(device)
			with torch.no_grad():
				cls = model(**inputs).last_hidden_state[:, 0]
			batch_embeddings.append(cls.cpu().float().numpy())

		embeddings = np.vstack(batch_embeddings)
		embeddings = _l2_normalize_rows(embeddings)

		centroid = embeddings.mean(axis=0)
		centroid = _l2_normalize_vector(centroid)

		centroid_std = float(embeddings.std())

		sim_matrix = np.dot(embeddings, embeddings.T)
		n = sim_matrix.shape[0]
		mask = ~np.eye(n, dtype=bool)
		if mask.any():
			intra_set_mean_sim = float(sim_matrix[mask].mean())
		else:
			intra_set_mean_sim = 0.0

		if device == "cuda":
			peak_vram_mb = int(torch.cuda.max_memory_allocated() // (1024**2))
			logger.info("Fingerprint peak VRAM: %d MB", peak_vram_mb)

		return FingerprintResult(
			embeddings=embeddings,
			centroid=centroid,
			centroid_std=centroid_std,
			intra_set_mean_sim=intra_set_mean_sim,
			image_count=int(embeddings.shape[0]),
		)
	finally:
		del model, inputs
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		gc.collect()
		logger.info("DINOv2 freed")


def extract_fingerprints(images: list) -> tuple[FingerprintResult, ErrorState | None]:
	"""Extract DINOv2 CLS embeddings and summary statistics from 256x256 RGB PIL images."""
	if not images:
		return cast(FingerprintResult, None), ErrorState(
			phase="FINGERPRINTING",
			error_type="validation",
			error_message="No images were provided for fingerprint extraction.",
			recoverable=True,
			recovery_suggestion="Provide at least one valid image.",
			traceback=None,
		)

	device = "cuda" if torch.cuda.is_available() else "cpu"

	try:
		return _extract_on_device(images, device), None
	except torch.cuda.OutOfMemoryError:
		if device != "cuda":
			return cast(FingerprintResult, None), ErrorState(
				phase="FINGERPRINTING",
				error_type="OOM",
				error_message="Out of memory during fingerprint extraction.",
				recoverable=True,
				recovery_suggestion="Reduce input image count and retry.",
				traceback=traceback.format_exc(),
			)

		logger.warning("CUDA OOM during fingerprint extraction; retrying on CPU.")
		try:
			return _extract_on_device(images, "cpu"), None
		except Exception:
			return cast(FingerprintResult, None), ErrorState(
				phase="FINGERPRINTING",
				error_type="OOM",
				error_message="Out of memory on CUDA and CPU retry failed.",
				recoverable=True,
				recovery_suggestion="Retry with fewer images or lower concurrent workload.",
				traceback=traceback.format_exc(),
			)
	except ConnectionError:
		return cast(FingerprintResult, None), ErrorState(
			phase="FINGERPRINTING",
			error_type="model_download",
			error_message="Failed to download DINOv2 model resources.",
			recoverable=True,
			recovery_suggestion="Check internet connection or pre-download the model.",
			traceback=traceback.format_exc(),
		)
	except Exception:
		return cast(FingerprintResult, None), ErrorState(
			phase="FINGERPRINTING",
			error_type="io_error",
			error_message="Unexpected fingerprint extraction failure.",
			recoverable=False,
			recovery_suggestion="Check input images and environment setup, then retry.",
			traceback=traceback.format_exc(),
		)
