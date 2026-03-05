"""
pipeline/ingestor.py — Upload handling and image preprocessing.

Exposes RealDataIngestor, which:
  1. Accepts a folder or ZIP file of raw images.
  2. Finds every .jpg/.jpeg/.png recursively (extracts ZIP first if needed).
  3. Resizes each image to 256×256 using Lanczos resampling and saves the
     result to a configurable output directory.
  4. Computes a fingerprint for every resized image via utils.fingerprint
     and aggregates them into a centroid vector via utils.similarity.
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from utils.fingerprint import extract_fingerprint
from utils.similarity import fingerprint_to_vector

# Canonical output size.
_TARGET_SIZE = (256, 256)
# Image extensions considered valid inputs.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class RealDataIngestor:
    """Ingest raw real images, resize them, and build a distribution fingerprint.

    Args:
        upload_path: Path to either a folder of images or a ``.zip`` archive.
        output_dir:  Directory where 256×256 resized copies will be saved.
                     Created automatically if it does not exist.
    """

    def __init__(self, upload_path: str, output_dir: str = "data/real/") -> None:
        self.upload_path = Path(upload_path)
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_images(self) -> list[str]:
        """Locate, resize, and save all images from the upload source.

        If ``upload_path`` is a ZIP archive the contents are extracted to a
        temporary directory first, then all ``.jpg``/``.jpeg``/``.png`` files
        are discovered recursively.  Each discovered image is resized to
        256×256 using Lanczos resampling and saved to ``output_dir``.

        Returns:
            Sorted list of absolute paths to the saved (resized) images.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        source_dir, _tmp = self._resolve_source_dir()
        try:
            raw_paths = sorted(
                p
                for p in source_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            )

            total = len(raw_paths)
            if total == 0:
                print("No images found.")
                return []

            saved: list[str] = []
            for idx, src in enumerate(raw_paths, start=1):
                dest = self._unique_dest(src)
                self._resize_and_save(src, dest)
                saved.append(str(dest))
                print(f"Processed {idx}/{total}: {src.name} → {dest.name}")

            return saved
        finally:
            if _tmp is not None:
                shutil.rmtree(_tmp, ignore_errors=True)

    def build_distribution_fingerprint(self, image_paths: list[str]) -> dict:
        """Compute per-image fingerprints and their mean centroid vector.

        Calls :func:`utils.fingerprint.extract_fingerprint` on every path in
        *image_paths*, converts each fingerprint to a unit-length vector via
        :func:`utils.similarity.fingerprint_to_vector`, then averages the
        vectors to produce a centroid that represents the real distribution.

        Args:
            image_paths: List of paths to resized images, as returned by
                         :meth:`load_images`.

        Returns:
            A dict with keys:

            * ``centroid_vector`` – :class:`numpy.ndarray` of shape ``(26,)``,
              the mean (un-renormalised) of all unit fingerprint vectors.
            * ``individual_fingerprints`` – list[dict], one raw fingerprint
              dict per image.
            * ``image_paths`` – the input *image_paths* list, preserved for
              downstream traceability.
        """
        fingerprints: list[dict] = []
        vectors: list[np.ndarray] = []

        total = len(image_paths)
        for idx, path in enumerate(image_paths, start=1):
            fp = extract_fingerprint(path)
            fingerprints.append(fp)
            vectors.append(fingerprint_to_vector(fp))
            print(f"Fingerprinted {idx}/{total}: {Path(path).name}")

        centroid = np.mean(np.stack(vectors), axis=0) if vectors else np.zeros(26)

        return {
            "centroid_vector": centroid,
            "individual_fingerprints": fingerprints,
            "image_paths": image_paths,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_source_dir(self) -> tuple[Path, str | None]:
        """Return the directory to scan and an optional temp dir to clean up."""
        if zipfile.is_zipfile(self.upload_path):
            tmp = tempfile.mkdtemp(prefix="ingestor_")
            with zipfile.ZipFile(self.upload_path, "r") as zf:
                zf.extractall(tmp)
            return Path(tmp), tmp

        if self.upload_path.is_dir():
            return self.upload_path, None

        raise ValueError(
            f"upload_path must be a directory or a ZIP file, got: {self.upload_path}"
        )

    def _unique_dest(self, src: Path) -> Path:
        """Return a collision-free destination path inside output_dir."""
        dest = self.output_dir / src.name
        stem, suffix = src.stem, src.suffix
        counter = 1
        while dest.exists():
            dest = self.output_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        return dest

    @staticmethod
    def _resize_and_save(src: Path, dest: Path) -> None:
        """Open *src*, resize to 256×256 (Lanczos), and save to *dest*."""
        with Image.open(src) as img:
            img = img.convert("RGB")
            img = img.resize(_TARGET_SIZE, Image.LANCZOS)
            img.save(dest)


if __name__ == "__main__":
    import sys
    import os
    import random

    # ------------------------------------------------------------------
    # Create a small synthetic test dataset (5 random 300×200 PNG images)
    # in a temporary folder, then run the full ingestor pipeline on it.
    # ------------------------------------------------------------------
    rng = random.Random(42)
    tmp_input = tempfile.mkdtemp(prefix="ingestor_test_input_")
    tmp_output = tempfile.mkdtemp(prefix="ingestor_test_output_")

    print("=== Building dummy input images ===")
    for i in range(5):
        pixels = np.array(
            [[[rng.randint(0, 255) for _ in range(3)] for _ in range(300)] for _ in range(200)],
            dtype=np.uint8,
        )
        p = Path(tmp_input) / f"test_{i:02d}.png"
        Image.fromarray(pixels, "RGB").save(p)
        print(f"  Created {p.name}")

    print(f"\n=== Running load_images on {tmp_input} ===")
    ingestor = RealDataIngestor(upload_path=tmp_input, output_dir=tmp_output)
    saved = ingestor.load_images()

    print(f"\n=== Running build_distribution_fingerprint on {len(saved)} images ===")
    dist = ingestor.build_distribution_fingerprint(saved)

    centroid = dist["centroid_vector"]
    print(f"\nCentroid vector shape : {centroid.shape}")
    print(f"Centroid vector norm  : {np.linalg.norm(centroid):.6f}")
    print(f"Individual fingerprints collected: {len(dist['individual_fingerprints'])}")

    # Cleanup
    shutil.rmtree(tmp_input, ignore_errors=True)
    shutil.rmtree(tmp_output, ignore_errors=True)
    print("\nDone — temp directories cleaned up.")
