"""Shared dataclasses for SyntheticImageGen pipeline phases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

_ALLOWED_ERROR_TYPES = {
    "OOM",
    "NaN_loss",
    "rejection_overflow",
    "io_error",
    "model_download",
    "validation",
}


def _validate_not_none(instance: object, field_names: list[str]) -> None:
    for field_name in field_names:
        if getattr(instance, field_name) is None:
            raise ValueError(f"{field_name} must not be None")


@dataclass
class IngestResult:
    images: list[Image.Image] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)
    count: int = 0
    rejected: int = 0
    rejection_reasons: list[str] = field(default_factory=list)
    avg_original_size: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        _validate_not_none(
            self,
            [
                "images",
                "paths",
                "count",
                "rejected",
                "rejection_reasons",
                "avg_original_size",
            ],
        )


@dataclass
class FingerprintResult:
    embeddings: np.ndarray
    centroid: np.ndarray
    centroid_std: float
    intra_set_mean_sim: float
    image_count: int

    def __post_init__(self) -> None:
        _validate_not_none(
            self,
            [
                "embeddings",
                "centroid",
                "centroid_std",
                "intra_set_mean_sim",
                "image_count",
            ],
        )


@dataclass
class TrainingResult:
    checkpoint_path: Path
    final_loss: float
    steps_completed: int
    peak_vram_mb: int
    training_time_s: float
    loss_history: list[tuple[int, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_not_none(
            self,
            [
                "checkpoint_path",
                "final_loss",
                "steps_completed",
                "peak_vram_mb",
                "training_time_s",
                "loss_history",
            ],
        )


@dataclass
class GenerationResult:
    accepted_images: list[Image.Image] = field(default_factory=list)
    accepted_paths: list[Path] = field(default_factory=list)
    metadata: list[dict] = field(default_factory=list)
    n_generated: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    mean_similarity: float = 0.0
    fid_estimate: float | None = None
    total_time_s: float = 0.0

    def __post_init__(self) -> None:
        _validate_not_none(
            self,
            [
                "accepted_images",
                "accepted_paths",
                "metadata",
                "n_generated",
                "n_accepted",
                "n_rejected",
                "mean_similarity",
                "total_time_s",
            ],
        )


@dataclass
class ErrorState:
    phase: str
    error_type: str
    error_message: str
    recoverable: bool
    recovery_suggestion: str
    traceback: str | None = None

    def __post_init__(self) -> None:
        _validate_not_none(
            self,
            ["phase", "error_type", "error_message", "recoverable", "recovery_suggestion"],
        )
        if self.error_type not in _ALLOWED_ERROR_TYPES:
            allowed = ", ".join(sorted(_ALLOWED_ERROR_TYPES))
            raise ValueError(f"error_type must be one of: {allowed}")


@dataclass
class DatasetReport:
    real_count: int
    synthetic_count: int
    rejected_count: int
    acceptance_rate: float
    fid_estimate: float | None
    mean_similarity: float
    std_similarity: float
    training_steps: int
    total_generation_time_s: float
    lora_rank: int
    model_id: str
    resolution: tuple[int, int]

    def __post_init__(self) -> None:
        _validate_not_none(
            self,
            [
                "real_count",
                "synthetic_count",
                "rejected_count",
                "acceptance_rate",
                "mean_similarity",
                "std_similarity",
                "training_steps",
                "total_generation_time_s",
                "lora_rank",
                "model_id",
                "resolution",
            ],
        )

    def to_dict(self) -> dict:
        return asdict(self)
