"""
utils/fingerprint.py — Low-dimensional feature extraction from a single image.

Responsibilities:
- Compute a normalised colour histogram (per channel) from an input image
  represented as a NumPy array in HWC / RGB format.
- Calculate the average brightness of the image (mean luminance across all
  pixels after converting to greyscale).
- Measure texture variance using the variance of the Laplacian of the
  greyscale image, which correlates with edge richness / sharpness.
- Concatenate all extracted features into a single 1-D float32 vector
  suitable for cosine distance comparison in utils.similarity.
"""
from __future__ import annotations

import numpy as np


def extract(image: np.ndarray) -> np.ndarray:
    """Extract a fingerprint feature vector from an image.

    Args:
        image: Input image as a NumPy array with shape (H, W, 3) and
               dtype uint8, in RGB channel order.

    Returns:
        A 1-D float32 numpy array encoding the colour histogram,
        average brightness, and texture variance of the image.
    """
    pass
