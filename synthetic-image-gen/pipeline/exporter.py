"""
pipeline/exporter.py — ZIP archive creation for real and synthetic datasets.

Responsibilities:
- Accept the real image records (from the ingestor) and the filtered
  synthetic image records (from the filter stage).
- Organise images into a two-folder layout inside a ZIP archive:
    real/       — original uploaded images (resized to 256×256)
    synthetic/  — filtered generated images
- Write a per-image JSON sidecar file containing metadata such as
  filename, fingerprint vector, generation parameters, and filter score.
- Return the path of the produced ZIP file for download or further use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def export(
    real_records: list[dict[str, Any]],
    synthetic_records: list[dict[str, Any]],
    output_path: str | Path = "dataset.zip",
) -> Path:
    """Build a ZIP archive containing real and synthetic image sets.

    Args:
        real_records:      Pre-processed records for the real images.
        synthetic_records: Filtered synthetic image records.
        output_path:       Destination path for the output ZIP file.

    Returns:
        Path to the created ZIP archive.
    """
    pass
