"""
pipeline/filter.py — Quality filtering of synthetic images against the real distribution.

Exposes QualityFilter, which compares every synthetic image's fingerprint
vector to the centroid of the real distribution (produced by
RealDataIngestor.build_distribution_fingerprint) using cosine similarity.
Images whose similarity score falls below a configurable threshold are
rejected as distribution outliers.
"""
from __future__ import annotations

import numpy as np

from utils.fingerprint import extract_fingerprint
from utils.similarity import cosine_similarity, fingerprint_to_vector


class QualityFilter:
    """Filter synthetic images by similarity to the real image distribution.

    Args:
        real_centroid_vector: The mean unit-length fingerprint vector of the
                              real dataset, as returned in
                              ``RealDataIngestor.build_distribution_fingerprint``
                              under the key ``"centroid_vector"``.
        threshold:            Minimum cosine similarity required for an image
                              to be accepted.  Range [−1, 1]; a value of
                              ``0.80`` means the synthetic image must be at
                              least 80 % similar (in fingerprint space) to
                              the real centroid.
    """

    def __init__(self, real_centroid_vector: np.ndarray, threshold: float = 0.80) -> None:
        self.real_centroid_vector = np.asarray(real_centroid_vector, dtype=np.float64)
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_image(self, image_path: str) -> float:
        """Compute the similarity score of a single image against the real centroid.

        Extracts the fingerprint of the image at *image_path*, converts it to
        a unit-length vector, and returns the cosine similarity between that
        vector and ``self.real_centroid_vector``.

        Args:
            image_path: Path to a saved image file (JPEG, PNG, …).

        Returns:
            Cosine similarity in [−1, 1].  Values closer to 1.0 indicate
            the image is statistically similar to the real distribution.
        """
        fingerprint = extract_fingerprint(image_path)
        image_vector = fingerprint_to_vector(fingerprint)
        return cosine_similarity(image_vector, self.real_centroid_vector)

    def filter_batch(
        self,
        generated: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Score every image in *generated* and split into accepted / rejected.

        Each dict in *generated* must contain at least an ``"image_path"`` key
        (as produced by ``SyntheticGenerator.generate``).  A
        ``"similarity_score"`` key is added in-place to every dict before the
        split is performed.

        Args:
            generated: List of generation record dicts from
                       ``SyntheticGenerator.generate``.

        Returns:
            A 2-tuple ``(accepted, rejected)`` where:

            * ``accepted`` – records whose ``similarity_score >= threshold``.
            * ``rejected`` – records whose ``similarity_score < threshold``.
        """
        accepted: list[dict] = []
        rejected: list[dict] = []

        total = len(generated)
        for record in generated:
            score = self.score_image(record["image_path"])
            record["similarity_score"] = score

            if score >= self.threshold:
                accepted.append(record)
            else:
                rejected.append(record)

        print(
            f"Quality filter: {len(accepted)}/{total} passed "
            f"(threshold={self.threshold:.2f})"
        )
        return accepted, rejected

