from __future__ import annotations

import pytest
from PIL import Image


@pytest.fixture
def dummy_images():
    return [
        Image.new("RGB", (256, 256), color=(r * 10, g * 10, 50))
        for r, g in [(i // 5, i % 5) for i in range(20)]
    ]


@pytest.fixture
def session_dir(tmp_path):
    d = tmp_path / "session_test"
    (d / "real").mkdir(parents=True)
    (d / "synthetic").mkdir()
    return d
