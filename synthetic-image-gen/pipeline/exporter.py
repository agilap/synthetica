"""Dataset export pipeline for SyntheticImageGen."""

from __future__ import annotations

import json
import logging
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np

from utils.similarity import nearest_real_image
from utils.types import DatasetReport, ErrorState, GenerationResult


logger = logging.getLogger(__name__)


def _resolve_resolution(metadata: list[dict]) -> tuple[int, int]:
	if not metadata:
		return (256, 256)

	value = metadata[0].get("resolution")
	if isinstance(value, (list, tuple)) and len(value) == 2:
		return (int(value[0]), int(value[1]))
	return (256, 256)


def _materialize_synthetic_paths(
	session_dir: Path,
	generation_result: GenerationResult,
) -> list[Path]:
	synthetic_dir = session_dir / "synthetic"
	synthetic_dir.mkdir(parents=True, exist_ok=True)

	materialized: list[Path] = []
	for idx, image in enumerate(generation_result.accepted_images):
		default_name = f"syn_{idx:04d}.jpg"
		provided_path = (
			generation_result.accepted_paths[idx]
			if idx < len(generation_result.accepted_paths)
			else Path(default_name)
		)

		target = provided_path if provided_path.is_absolute() else synthetic_dir / provided_path.name
		target.parent.mkdir(parents=True, exist_ok=True)

		if not target.exists():
			image.save(target)

		materialized.append(target)

	return materialized


def export_dataset(
	session_dir: Path,
	generation_result: GenerationResult,
	real_image_paths: list[Path],
	training_steps: int,
	lora_rank: int,
	model_id: str,
	fingerprint_result=None,
) -> tuple[Path, DatasetReport, ErrorState | None]:
	try:
		session_dir = Path(session_dir)
		session_dir.mkdir(parents=True, exist_ok=True)

		synthetic_paths = _materialize_synthetic_paths(session_dir, generation_result)

		acceptance_rate = generation_result.n_accepted / max(generation_result.n_generated, 1)
		sim_values = [float(meta.get("similarity_score", 0.0)) for meta in generation_result.metadata]
		std_similarity = float(np.std(sim_values)) if sim_values else 0.0

		dataset_report = DatasetReport(
			real_count=len(real_image_paths),
			synthetic_count=generation_result.n_accepted,
			rejected_count=generation_result.n_rejected,
			acceptance_rate=float(acceptance_rate),
			fid_estimate=generation_result.fid_estimate,
			mean_similarity=float(generation_result.mean_similarity),
			std_similarity=std_similarity,
			training_steps=int(training_steps),
			total_generation_time_s=float(generation_result.total_time_s),
			lora_rank=int(lora_rank),
			model_id=model_id,
			resolution=_resolve_resolution(generation_result.metadata),
		)

		meta_paths: list[Path] = []
		for path, meta in zip(synthetic_paths, generation_result.metadata):
			meta_payload = dict(meta)

			if (
				fingerprint_result is not None
				and real_image_paths
				and getattr(fingerprint_result, "embeddings", None) is not None
				and getattr(fingerprint_result, "centroid", None) is not None
			):
				synthetic_embedding = np.asarray(
					meta_payload.get("synthetic_embedding", fingerprint_result.centroid),
					dtype=np.float32,
				)
				base_path, base_score = nearest_real_image(
					synthetic_embedding=synthetic_embedding,
					real_embeddings=np.asarray(fingerprint_result.embeddings, dtype=np.float32),
					real_paths=real_image_paths,
				)
				meta_payload["base_image"] = base_path.name
				meta_payload["base_similarity"] = round(float(base_score), 4)

			meta_path = path.with_name(f"{path.stem}_meta.json")
			meta_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
			meta_paths.append(meta_path)

		report_path = session_dir / "dataset_report.json"
		report_path.write_text(json.dumps(dataset_report.to_dict(), indent=2), encoding="utf-8")

		zip_name = f"synthetic_dataset_{session_dir.name[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
		zip_path = session_dir / "export" / zip_name
		zip_path.parent.mkdir(parents=True, exist_ok=True)

		with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
			for real_path in real_image_paths:
				real_path = Path(real_path)
				if real_path.exists():
					zf.write(real_path, arcname=f"real/{real_path.name}")

			for syn_path in synthetic_paths:
				if syn_path.exists():
					zf.write(syn_path, arcname=f"synthetic/{syn_path.name}")

			for meta_path in meta_paths:
				if meta_path.exists():
					zf.write(meta_path, arcname=f"synthetic/{meta_path.name}")

			zf.write(report_path, arcname="dataset_report.json")

		with zipfile.ZipFile(zip_path, "r") as verify_zip:
			names = verify_zip.namelist()

		expected = len(real_image_paths) + len(synthetic_paths) * 2 + 1
		if len(names) < expected:
			return cast(Path, None), cast(DatasetReport, None), ErrorState(
				phase="EXPORTING",
				error_type="io_error",
				error_message="ZIP verification failed: archive is missing expected files.",
				recoverable=False,
				recovery_suggestion="Retry export and ensure output files are writable.",
				traceback=None,
			)

		return zip_path, dataset_report, None

	except OSError:
		logger.exception("OS error while exporting dataset")
		return cast(Path, None), cast(DatasetReport, None), ErrorState(
			phase="EXPORTING",
			error_type="io_error",
			error_message="Check disk space.",
			recoverable=False,
			recovery_suggestion="Free disk space and retry export.",
			traceback=traceback.format_exc(),
		)
