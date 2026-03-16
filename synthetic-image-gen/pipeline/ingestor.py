"""Image ingestion pipeline for preparing inputs for fingerprinting."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path

from PIL import Image

from utils.types import ErrorState, IngestResult, VALID_EXTENSIONS


def _center_crop(img: Image.Image, size: int) -> Image.Image:
	"""Return a centered square crop of the requested size."""
	width, height = img.size
	left = max((width - size) // 2, 0)
	top = max((height - size) // 2, 0)
	right = left + size
	bottom = top + size
	return img.crop((left, top, right, bottom))


def _extract_zip(source_zip: Path) -> Path:
	"""Extract a ZIP source into a temporary directory and register cleanup."""
	temp_dir = Path(tempfile.mkdtemp(prefix="syntheticimagegen_ingest_"))
	atexit.register(shutil.rmtree, temp_dir, True)
	with zipfile.ZipFile(source_zip, "r") as archive:
		archive.extractall(temp_dir)
	return temp_dir


def ingest(source: Path | str) -> tuple[IngestResult, ErrorState | None]:
	"""Ingest image files from a directory or ZIP and normalize to 256x256 RGB."""
	try:
		source_path = Path(source).expanduser().resolve()
	except Exception:
		return None, ErrorState(
			phase="INGESTING",
			error_type="io_error",
			error_message="Invalid source path.",
			recoverable=False,
			recovery_suggestion="Provide a valid directory path or .zip file.",
			traceback=traceback.format_exc(),
		)

	try:
		if source_path.is_file() and source_path.suffix.lower() == ".zip":
			root_dir = _extract_zip(source_path)
		elif source_path.is_dir():
			root_dir = source_path
		else:
			return None, ErrorState(
				phase="INGESTING",
				error_type="io_error",
				error_message="Source must be an existing directory or .zip file.",
				recoverable=False,
				recovery_suggestion="Provide a valid directory path or .zip file.",
				traceback=None,
			)
	except Exception:
		return None, ErrorState(
			phase="INGESTING",
			error_type="io_error",
			error_message="Failed to read or extract source input.",
			recoverable=False,
			recovery_suggestion="Check ZIP integrity and file permissions, then retry.",
			traceback=traceback.format_exc(),
		)

	images: list[Image.Image] = []
	paths: list[Path] = []
	rejection_reasons: list[str] = []
	rejected = 0
	width_sum = 0
	height_sum = 0

	try:
		all_files: list[Path] = []
		for walk_root, _, filenames in os.walk(root_dir):
			for filename in filenames:
				all_files.append(Path(walk_root) / filename)

		for file_path in all_files:
			extension = file_path.suffix.lower()
			if extension not in VALID_EXTENSIONS:
				continue

			try:
				if file_path.stat().st_size == 0:
					continue
			except Exception:
				rejected += 1
				rejection_reasons.append("io_error")
				continue

			try:
				with Image.open(file_path) as opened:
					width, height = opened.size
					if min(width, height) < 64:
						rejected += 1
						rejection_reasons.append("too_small")
						continue

					rgb = opened.convert("RGB")
			except Exception:
				rejected += 1
				rejection_reasons.append("corrupt")
				continue

			scale = 256 / min(width, height)
			resized_w = max(1, int(width * scale))
			resized_h = max(1, int(height * scale))
			resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

			resized = rgb.resize((resized_w, resized_h), resample=resampling)
			cropped = _center_crop(resized, 256)

			images.append(cropped)
			paths.append(file_path)
			width_sum += width
			height_sum += height

	except Exception:
		return None, ErrorState(
			phase="INGESTING",
			error_type="io_error",
			error_message="Unexpected I/O failure while scanning source images.",
			recoverable=False,
			recovery_suggestion="Verify source readability and retry ingestion.",
			traceback=traceback.format_exc(),
		)

	count = len(images)
	if count > 0:
		avg_original_size = (int(width_sum / count), int(height_sum / count))
	else:
		avg_original_size = (0, 0)

	result = IngestResult(
		images=images,
		paths=paths,
		count=count,
		rejected=rejected,
		rejection_reasons=rejection_reasons,
		avg_original_size=avg_original_size,
	)

	if count < 15:
		return result, ErrorState(
			phase="INGESTING",
			error_type="validation",
			error_message="Not enough valid images for reliable fingerprinting.",
			recoverable=True,
			recovery_suggestion="Add more images. Minimum 15 required for reliable fingerprinting.",
			traceback=None,
		)

	return result, None
