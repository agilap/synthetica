"""
pipeline/ingestor.py — Upload handling and image preprocessing.

Responsibilities:
- Accept one or more uploaded image files (paths or byte streams).
- Auto-resize every image to a canonical 256×256 resolution while
  preserving aspect ratio via centre-crop or padding.
- Extract a metadata fingerprint for each image (colour histogram,
  brightness, texture variance) by delegating to utils.fingerprint.
- Return a normalised dataset dict ready for the trainer stage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def ingest(sources: list[str | Path]) -> list[dict[str, Any]]:
    """Ingest and preprocess a list of image sources.

    Args:
        sources: File paths (or upload byte objects) for input images.

    Returns:
        A list of records, each containing the resized image array and
        its extracted metadata fingerprint.
    """
    pass
