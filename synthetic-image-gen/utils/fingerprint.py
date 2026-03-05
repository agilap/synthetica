"""
utils/fingerprint.py — Low-dimensional feature extraction from a single image.

Exposes a single public function, extract_fingerprint(), that loads an image
from disk and returns a dict containing:
  - color_histogram : normalized 8-bin histogram per RGB channel (24 floats).
  - avg_brightness  : mean grayscale intensity in [0, 255].
  - texture_variance: variance of the Laplacian — a standard sharpness measure.

Dependencies: Pillow (image I/O + histograms), OpenCV (grayscale + Laplacian),
              NumPy (array arithmetic).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# Number of histogram bins per channel.
_BINS = 8


def extract_fingerprint(image_path: str) -> dict:
    """Extract a fingerprint from an image file.

    Loads the image at *image_path*, then computes:

    * **color_histogram** – An 8-bin normalised histogram for each of the
      R, G, and B channels, flattened into a single list of 24 floats.
      Each channel's bins sum to 1.0.
    * **avg_brightness** – Mean pixel intensity of the grayscale image,
      in the range [0, 255].
    * **texture_variance** – Variance of the pixel-wise Laplacian applied
      to the grayscale image.  Higher values indicate more texture / edges;
      lower values indicate smoother / blurrier images.

    Args:
        image_path: Absolute or relative path to any image format supported
                    by Pillow (JPEG, PNG, BMP, TIFF, …).

    Returns:
        A dict with keys ``color_histogram`` (list[float], length 24),
        ``avg_brightness`` (float), and ``texture_variance`` (float).
    """
    # --- Load with Pillow (handles a wide variety of formats) --------------
    pil_img = Image.open(image_path).convert("RGB")
    rgb_array = np.array(pil_img, dtype=np.uint8)  # shape (H, W, 3)

    # --- Color histogram (Pillow) -------------------------------------------
    # Pillow's histogram() returns 256 counts per channel concatenated.
    raw_hist = pil_img.histogram()  # length 768 for an RGB image

    color_histogram: list[float] = []
    for ch in range(3):
        # Extract the 256 counts for this channel.
        counts_256 = np.array(raw_hist[ch * 256 : (ch + 1) * 256], dtype=np.float64)

        # Rebin from 256 buckets down to _BINS by summing groups.
        group = 256 // _BINS
        counts_8 = counts_256.reshape(_BINS, group).sum(axis=1)

        # Normalise so the channel sums to 1.0 (guard against all-zero).
        total = counts_8.sum()
        if total > 0:
            counts_8 /= total

        color_histogram.extend(counts_8.tolist())

    # --- Grayscale (OpenCV) -------------------------------------------------
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)  # shape (H, W), uint8

    # --- Average brightness -------------------------------------------------
    avg_brightness: float = float(gray.mean())

    # --- Texture variance via Laplacian (OpenCV) ----------------------------
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance: float = float(laplacian.var())

    return {
        "color_histogram": color_histogram,
        "avg_brightness": avg_brightness,
        "texture_variance": texture_variance,
    }


if __name__ == "__main__":
    import io
    import tempfile
    import os

    # Build a random 256×256 RGB image and save it to a temporary PNG file.
    rng = np.random.default_rng(seed=42)
    dummy_pixels = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(dummy_pixels, mode="RGB")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        dummy_pil.save(tmp_path)

    try:
        fp = extract_fingerprint(tmp_path)
    finally:
        os.unlink(tmp_path)

    print("color_histogram  (24 values):")
    hist = fp["color_histogram"]
    for ch_idx, ch_name in enumerate(("R", "G", "B")):
        bins = hist[ch_idx * _BINS : (ch_idx + 1) * _BINS]
        formatted = ", ".join(f"{v:.4f}" for v in bins)
        print(f"  {ch_name}: [{formatted}]  sum={sum(bins):.6f}")

    print(f"\navg_brightness   : {fp['avg_brightness']:.4f}")
    print(f"texture_variance : {fp['texture_variance']:.4f}")
