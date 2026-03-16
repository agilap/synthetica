"""Stateless similarity and distribution-matching utilities."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from torchvision import transforms
from torchvision.models import inception_v3


logger = logging.getLogger(__name__)


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
	return matrix @ query


def compute_adaptive_threshold(embeddings: np.ndarray) -> float:
	if embeddings.shape[0] < 2:
		return 0.65

	sim_matrix = embeddings @ embeddings.T
	mask = ~np.eye(len(embeddings), dtype=bool)
	sims = sim_matrix[mask]
	return float(max(0.65, sims.mean() - 2 * sims.std()))


def _extract_inception_features(
	model: torch.nn.Module,
	images: list,
	device: str,
	batch_size: int = 8,
) -> np.ndarray:
	preprocess = transforms.Compose(
		[
			transforms.Resize((299, 299)),
			transforms.ToTensor(),
		]
	)

	features: list[np.ndarray] = []
	with torch.no_grad():
		for start in range(0, len(images), batch_size):
			batch_images = images[start : start + batch_size]

			tensor_batch = []
			for img in batch_images:
				pil_img = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
				tensor_batch.append(preprocess(pil_img.convert("RGB")))

			inputs = torch.stack(tensor_batch, dim=0).to(device)
			out = model(inputs)
			features.append(out.detach().cpu().float().numpy())

	return np.vstack(features).astype(np.float64)


def compute_fid_proxy(real_images: list, fake_images: list, device: str = "cuda") -> float | None:
	if len(real_images) < 10 or len(fake_images) < 10:
		return None
	if len(real_images) < 50 or len(fake_images) < 50:
		logger.warning("FID proxy computed with fewer than 50 images per set; estimate may be noisy.")

	runtime_device = device
	if runtime_device == "cuda" and not torch.cuda.is_available():
		runtime_device = "cpu"

	model = None
	try:
		model = inception_v3(pretrained=True, transform_input=False)
		model.fc = torch.nn.Identity()
		model.eval()
		model = model.to(runtime_device)

		real_feats = _extract_inception_features(model, real_images, runtime_device, batch_size=8)
		fake_feats = _extract_inception_features(model, fake_images, runtime_device, batch_size=8)

		mu_real = np.mean(real_feats, axis=0)
		mu_fake = np.mean(fake_feats, axis=0)
		sigma_real = np.cov(real_feats, rowvar=False)
		sigma_fake = np.cov(fake_feats, rowvar=False)

		mean_diff = mu_real - mu_fake
		mean_norm = float(mean_diff @ mean_diff)

		covmean = sqrtm(sigma_real @ sigma_fake)
		if np.iscomplexobj(covmean):
			covmean = covmean.real

		fid = mean_norm + np.trace(sigma_real + sigma_fake - 2 * covmean)
		return float(np.real(fid))
	finally:
		del model
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		gc.collect()


def nearest_real_image(
	synthetic_embedding: np.ndarray,
	real_embeddings: np.ndarray,
	real_paths: list[Path],
) -> tuple[Path, float]:
	sims = cosine_similarity_batch(synthetic_embedding, real_embeddings)
	idx = int(np.argmax(sims))
	return real_paths[idx], float(sims[idx])
