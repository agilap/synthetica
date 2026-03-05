"""
utils/similarity.py — Vector conversion and cosine similarity for image fingerprints.

Exposes two public functions:

  fingerprint_to_vector(fingerprint)
      Converts a fingerprint dict (produced by utils.fingerprint) into a
      unit-length 1-D numpy array of length 26 (24 histogram + 1 brightness
      + 1 texture variance), suitable for cosine similarity comparison.

  cosine_similarity(vec_a, vec_b)
      Returns the cosine similarity between two fingerprint vectors in [-1, 1].
      Returns 0.0 if either vector is a zero vector.
"""
from __future__ import annotations

import numpy as np


def fingerprint_to_vector(fingerprint: dict) -> np.ndarray:
    """Convert a fingerprint dict into a normalised 1-D numpy array.

    Concatenates the following fields in order:
      1. ``color_histogram``  — 24 floats (8 bins × 3 RGB channels)
      2. ``avg_brightness``   — 1 float   (mean grayscale intensity, 0–255)
      3. ``texture_variance`` — 1 float   (variance of Laplacian)

    The resulting length-26 vector is then L2-normalised to unit length so
    that cosine similarity is equivalent to the dot product.

    Args:
        fingerprint: Dict with keys ``color_histogram`` (list[float], len 24),
                     ``avg_brightness`` (float), and ``texture_variance`` (float),
                     as returned by ``utils.fingerprint.extract_fingerprint``.

    Returns:
        A unit-length float64 numpy array of shape (26,).  If the raw vector
        is all zeros the zero vector is returned unchanged.
    """
    raw = np.array(
        fingerprint["color_histogram"]
        + [fingerprint["avg_brightness"], fingerprint["texture_variance"]],
        dtype=np.float64,
    )

    norm = np.linalg.norm(raw)
    if norm == 0.0:
        return raw

    return raw / norm


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute the cosine similarity between two fingerprint vectors.

    Both vectors should already be unit-length (i.e. produced by
    :func:`fingerprint_to_vector`), in which case the similarity is simply
    their dot product.  For non-unit inputs the norms are computed on the fly.

    Args:
        vec_a: First fingerprint vector (1-D float64, length 26).
        vec_b: Second fingerprint vector (1-D float64, length 26).

    Returns:
        Cosine similarity in the range [-1, 1].  Returns 0.0 if either
        vector is a zero vector (similarity undefined).
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


if __name__ == "__main__":
    import random

    rng = random.Random(0)

    def _random_fingerprint(rng: random.Random) -> dict:
        """Create a plausible random fingerprint dict."""
        # 24 histogram values: 8 bins per channel, each channel sums to 1.
        histogram: list[float] = []
        for _ in range(3):
            raw_bins = [rng.random() for _ in range(8)]
            total = sum(raw_bins)
            histogram.extend(b / total for b in raw_bins)

        return {
            "color_histogram": histogram,
            "avg_brightness": rng.uniform(0.0, 255.0),
            "texture_variance": rng.uniform(0.0, 60000.0),
        }

    fp_a = _random_fingerprint(rng)
    fp_b = _random_fingerprint(rng)

    vec_a = fingerprint_to_vector(fp_a)
    vec_b = fingerprint_to_vector(fp_b)

    similarity = cosine_similarity(vec_a, vec_b)

    print(f"Vector A (len={len(vec_a)}, norm={np.linalg.norm(vec_a):.6f}):")
    print(f"  histogram[:6] = {vec_a[:6].tolist()}")
    print(f"  avg_brightness component = {vec_a[24]:.6f}")
    print(f"  texture_variance component = {vec_a[25]:.6f}")
    print()
    print(f"Vector B (len={len(vec_b)}, norm={np.linalg.norm(vec_b):.6f}):")
    print(f"  histogram[:6] = {vec_b[:6].tolist()}")
    print(f"  avg_brightness component = {vec_b[24]:.6f}")
    print(f"  texture_variance component = {vec_b[25]:.6f}")
    print()
    print(f"Cosine similarity(A, B) = {similarity:.6f}")

    # Sanity checks
    assert len(vec_a) == 26, "vector length must be 26"
    assert len(vec_b) == 26, "vector length must be 26"
    assert abs(np.linalg.norm(vec_a) - 1.0) < 1e-9, "vec_a must be unit length"
    assert abs(np.linalg.norm(vec_b) - 1.0) < 1e-9, "vec_b must be unit length"
    assert -1.0 <= similarity <= 1.0, "similarity must be in [-1, 1]"
    assert cosine_similarity(np.zeros(26), vec_b) == 0.0, "zero-vector edge case"
    print("\nAll assertions passed.")
