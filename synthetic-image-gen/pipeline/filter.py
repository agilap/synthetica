"""Two-tier quality filter for generated synthetic images."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from utils.types import FingerprintResult


logger = logging.getLogger(__name__)


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	norms = np.where(norms == 0.0, 1.0, norms)
	return matrix / norms


def _compute_hsv_hist_rgb_array(rgb_array: np.ndarray) -> np.ndarray:
	bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
	hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
	return cv2.normalize(hist, hist).flatten()


def _tier1_histogram_check(img: Image.Image, real_hists: list[np.ndarray]) -> bool:
	rgb = np.array(img.convert("RGB"))
	gen_hist = _compute_hsv_hist_rgb_array(rgb)
	return any(cv2.compareHist(gen_hist, real_hist, cv2.HISTCMP_CORREL) >= 0.5 for real_hist in real_hists)


def _build_real_hists(paths: list[Path]) -> list[np.ndarray]:
	hists: list[np.ndarray] = []
	for path in paths:
		try:
			with Image.open(path) as image:
				rgb = np.array(image.convert("RGB"))
			hists.append(_compute_hsv_hist_rgb_array(rgb))
		except Exception:
			logger.warning("Skipping histogram reference image: %s", path)
	return hists


def _extract_dinov2_batch(images: list) -> np.ndarray:
	if not images:
		return np.empty((0, 384), dtype=np.float32)

	processor = None
	model = None
	inputs = None
	cls_embeddings: list[np.ndarray] = []

	try:
		device = "cuda" if torch.cuda.is_available() else "cpu"
		processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
		model = AutoModel.from_pretrained("facebook/dinov2-small", torch_dtype=torch.float16)
		model = model.to(device)
		model.eval()

		batch_size = 8
		for start_idx in range(0, len(images), batch_size):
			batch = images[start_idx : start_idx + batch_size]
			inputs = processor(images=batch, return_tensors="pt").to(device)
			with torch.no_grad():
				cls = model(**inputs).last_hidden_state[:, 0]
			cls_embeddings.append(cls.cpu().float().numpy())

		embeddings = np.vstack(cls_embeddings).astype(np.float32)
		return _l2_normalize_rows(embeddings)
	finally:
		del model, inputs
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		gc.collect()
		logger.info("DINOv2 filter model freed")


def filter_images(
	images: list,
	seeds: list[int],
	fingerprint_result: FingerprintResult,
	threshold: float,
	real_image_paths: list[Path] | None = None,
) -> tuple[list, list[float], list[int]]:
	if not images:
		return [], [], []

	real_hists: list[np.ndarray] = []
	if real_image_paths:
		real_hists = _build_real_hists(real_image_paths)

	tier1_images: list = []
	tier1_seeds: list[int] = []

	for img, seed in zip(images, seeds):
		if real_hists and not _tier1_histogram_check(img, real_hists):
			continue
		tier1_images.append(img)
		tier1_seeds.append(seed)

	if not tier1_images:
		return [], [], []

	embeddings = _extract_dinov2_batch(tier1_images)

	centroid = np.asarray(fingerprint_result.centroid, dtype=np.float32)
	centroid_norm = np.linalg.norm(centroid)
	if centroid_norm == 0.0:
		centroid = centroid
	else:
		centroid = centroid / centroid_norm

	sims = np.dot(embeddings, centroid)

	accepted_images: list = []
	accepted_scores: list[float] = []
	accepted_seeds: list[int] = []

	for img, sim, seed in zip(tier1_images, sims, tier1_seeds):
		score = float(sim)
		if score >= threshold:
			accepted_images.append(img)
			accepted_scores.append(score)
			accepted_seeds.append(seed)

	return accepted_images, accepted_scores, accepted_seeds
