"""
utils/similarity.py — Distance metric between two image fingerprint vectors.

Responsibilities:
- Accept two 1-D float32 fingerprint vectors produced by utils.fingerprint.
- Compute the cosine distance between them (1 − cosine_similarity), yielding
  a value in [0, 2] where 0 means identical and 2 means maximally dissimilar.
- Guard against zero-norm vectors to avoid division-by-zero errors.
- Optionally accept a batch of vectors and return pairwise distances for use
  in the filter stage when comparing synthetic images to the real centroid.
"""
from __future__ import annotations

import numpy as np


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the cosine distance between two fingerprint vectors.

    Args:
        a: First fingerprint vector (1-D float32).
        b: Second fingerprint vector (1-D float32), same length as *a*.

    Returns:
        Cosine distance in the range [0, 2].  Returns 1.0 if either
        vector has zero norm (undefined similarity).
    """
    pass


def pairwise_distances(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute cosine distances between each row of *matrix* and *vector*.

    Args:
        matrix: Shape (N, D) array of N fingerprint vectors.
        vector: Shape (D,) reference fingerprint vector.

    Returns:
        Shape (N,) float32 array of cosine distances.
    """
    pass
