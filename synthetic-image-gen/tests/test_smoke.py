from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pipeline.exporter import export_dataset
from pipeline.ingestor import ingest
from utils.fingerprint import extract_fingerprints
from utils.similarity import compute_adaptive_threshold, compute_fid_proxy, cosine_similarity_batch
from utils.types import ErrorState, GenerationResult
from utils.vram_guard import flush, get_vram_status, vram_used_mb


def _write_jpg(path: Path, size: tuple[int, int] = (64, 64), color=(100, 50, 25)) -> None:
    img = Image.new("RGB", size, color=color)
    img.save(path, format="JPEG")


def _normalized_embeddings(n: int, d: int = 384) -> np.ndarray:
    arr = np.random.randn(n, d).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def test_ingest_valid_folder(tmp_path):
    data_dir = tmp_path / "images"
    data_dir.mkdir(parents=True)
    for i in range(20):
        _write_jpg(data_dir / f"img_{i:02d}.jpg")

    result, error = ingest(data_dir)

    assert error is None
    assert result.count == 20
    assert all(img.size == (256, 256) for img in result.images)
    assert all(img.mode == "RGB" for img in result.images)


def test_ingest_too_few_images(tmp_path):
    data_dir = tmp_path / "few"
    data_dir.mkdir(parents=True)
    for i in range(5):
        _write_jpg(data_dir / f"img_{i:02d}.jpg")

    result, error = ingest(data_dir)

    assert result.count == 5
    assert error is not None
    assert error.error_type == "validation"
    assert error.recoverable is True


def test_ingest_zip_input(tmp_path):
    source_dir = tmp_path / "zip_src"
    source_dir.mkdir(parents=True)
    for i in range(20):
        _write_jpg(source_dir / f"img_{i:02d}.jpg")

    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.glob("*.jpg"):
            zf.write(file_path, arcname=file_path.name)

    result, error = ingest(zip_path)

    assert error is None
    assert result.count == 20


def test_ingest_rejects_corrupt(tmp_path):
    data_dir = tmp_path / "corrupt"
    data_dir.mkdir(parents=True)
    for i in range(20):
        _write_jpg(data_dir / f"ok_{i:02d}.jpg")
    (data_dir / "bad.jpg").write_bytes(b"not-a-real-image")

    result, error = ingest(data_dir)

    assert error is None
    assert result.rejected >= 1
    assert "corrupt" in result.rejection_reasons


def test_ingest_rejects_too_small(tmp_path):
    data_dir = tmp_path / "small"
    data_dir.mkdir(parents=True)
    for i in range(20):
        _write_jpg(data_dir / f"ok_{i:02d}.jpg")
    _write_jpg(data_dir / "tiny.jpg", size=(32, 32))

    result, error = ingest(data_dir)

    assert error is None
    assert result.rejected >= 1
    assert "too_small" in result.rejection_reasons


def test_fingerprint_output_shape(monkeypatch, dummy_images):
    class DummyInputs(dict):
        def to(self, _device):
            return self

    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            bs = len(images)
            return DummyInputs({"pixel_values": torch.zeros(bs, 3, 224, 224)})

    class DummyOutput:
        def __init__(self, batch_size: int):
            self.last_hidden_state = torch.randn(batch_size, 2, 384, dtype=torch.float32)

    class DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            bs = kwargs["pixel_values"].shape[0]
            return DummyOutput(bs)

    monkeypatch.setattr("utils.fingerprint.AutoImageProcessor.from_pretrained", lambda *_args, **_kwargs: DummyProcessor())
    import utils.fingerprint as fp_module

    class MockAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyModel()

    monkeypatch.setattr(fp_module, "AutoModel", MockAutoModel)
    monkeypatch.setattr("utils.fingerprint.torch.cuda.is_available", lambda: False)

    result, error = extract_fingerprints(dummy_images)

    assert error is None
    assert result.embeddings.shape == (20, 384)
    norms = np.linalg.norm(result.embeddings, axis=1)
    assert np.allclose(norms, np.ones_like(norms), atol=1e-5)
    assert result.centroid.shape == (384,)


def test_fingerprint_frees_gpu(monkeypatch, dummy_images):
    class DummyInputs(dict):
        def to(self, _device):
            return self

    class DummyProcessor:
        def __call__(self, images, return_tensors="pt"):
            bs = len(images)
            return DummyInputs({"pixel_values": torch.zeros(bs, 3, 224, 224)})

    class DummyOutput:
        def __init__(self, batch_size: int):
            self.last_hidden_state = torch.randn(batch_size, 2, 384, dtype=torch.float32)

    class DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            bs = kwargs["pixel_values"].shape[0]
            return DummyOutput(bs)

    alloc = {"value": 123 * 1024 * 1024}

    monkeypatch.setattr("utils.fingerprint.AutoImageProcessor.from_pretrained", lambda *_args, **_kwargs: DummyProcessor())
    import utils.fingerprint as fp_module

    class MockAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyModel()

    monkeypatch.setattr(fp_module, "AutoModel", MockAutoModel)
    monkeypatch.setattr("utils.fingerprint.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("utils.fingerprint.torch.cuda.memory_allocated", lambda: alloc["value"])
    monkeypatch.setattr("utils.fingerprint.torch.cuda.reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr("utils.fingerprint.torch.cuda.max_memory_allocated", lambda: alloc["value"])
    monkeypatch.setattr("utils.fingerprint.torch.cuda.empty_cache", lambda: None)

    before = alloc["value"]
    result, error = extract_fingerprints(dummy_images)
    after = alloc["value"]

    assert error is None
    assert result.image_count == 20
    assert before == after


def test_adaptive_threshold_range():
    embeddings = _normalized_embeddings(20)
    threshold = compute_adaptive_threshold(embeddings)
    assert 0.65 <= threshold <= 1.0


def test_fid_proxy_small_set(dummy_images):
    assert compute_fid_proxy([], []) is None
    assert compute_fid_proxy(dummy_images[:9], dummy_images[:9]) is None


def test_cosine_similarity_self():
    matrix = _normalized_embeddings(5)
    query = matrix[0].copy()
    sims = cosine_similarity_batch(query, matrix)
    assert pytest.approx(1.0, abs=1e-6) == float(sims[0])


def test_exporter_zip_structure(session_dir, dummy_images):
    real_paths: list[Path] = []
    for i in range(5):
        p = session_dir / "real" / f"real_{i:02d}.jpg"
        dummy_images[i].save(p)
        real_paths.append(p)

    synthetic_paths: list[Path] = []
    metadata: list[dict] = []
    for i in range(5):
        p = session_dir / "synthetic" / f"syn_{i:04d}.jpg"
        dummy_images[i + 5].save(p)
        synthetic_paths.append(p)
        metadata.append({"similarity_score": 0.8 + i * 0.01, "resolution": [256, 256]})

    generation_result = GenerationResult(
        accepted_images=dummy_images[5:10],
        accepted_paths=synthetic_paths,
        metadata=metadata,
        n_generated=20,
        n_accepted=5,
        n_rejected=15,
        mean_similarity=0.82,
        fid_estimate=None,
        total_time_s=1.2,
    )

    zip_path, report, error = export_dataset(
        session_dir=session_dir,
        generation_result=generation_result,
        real_image_paths=real_paths,
        training_steps=1000,
        lora_rank=8,
        model_id="runwayml/stable-diffusion-v1-5",
        fingerprint_result=None,
    )

    assert error is None
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()

    assert any(name.startswith("real/") for name in names)
    assert any(name.startswith("synthetic/") for name in names)
    assert sum(name.startswith("synthetic/") and name.endswith("_meta.json") for name in names) == 5
    assert "dataset_report.json" in names
    assert report.synthetic_count == 5


def test_vram_guard_cpu_safe(monkeypatch):
    monkeypatch.setattr("utils.vram_guard.torch.cuda.is_available", lambda: False)

    assert vram_used_mb() == 0
    flush()
    status = get_vram_status()
    assert status["available"] is False


def test_error_state_invalid_type():
    with pytest.raises(ValueError):
        ErrorState(
            phase="TEST",
            error_type="NOT_VALID",
            error_message="bad",
            recoverable=False,
            recovery_suggestion="none",
            traceback=None,
        )
