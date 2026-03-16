"""Verify SyntheticImageGen scaffold structure."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "synthetic-image-gen"

EXPECTED_DIRS = [
    PROJECT,
    PROJECT / "config",
    PROJECT / "pipeline",
    PROJECT / "utils",
    PROJECT / "tests",
    PROJECT / "scripts",
    PROJECT / "sessions",
]

EXPECTED_FILES = [
    PROJECT / "app.py",
    PROJECT / "config" / "defaults.yaml",
    PROJECT / "pipeline" / "__init__.py",
    PROJECT / "pipeline" / "ingestor.py",
    PROJECT / "pipeline" / "trainer.py",
    PROJECT / "pipeline" / "generator.py",
    PROJECT / "pipeline" / "filter.py",
    PROJECT / "pipeline" / "exporter.py",
    PROJECT / "utils" / "__init__.py",
    PROJECT / "utils" / "fingerprint.py",
    PROJECT / "utils" / "similarity.py",
    PROJECT / "utils" / "vram_guard.py",
    PROJECT / "utils" / "quantize.py",
    PROJECT / "tests" / "__init__.py",
    PROJECT / "sessions" / ".gitkeep",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check_dir(path: Path) -> bool:
    ok = path.is_dir()
    print(f"{'✅' if ok else '❌'} {rel(path)}")
    return ok


def check_file(path: Path) -> bool:
    ok = path.is_file()
    print(f"{'✅' if ok else '❌'} {rel(path)}")
    return ok


def main() -> int:
    all_ok = True

    print("Checking expected directories:")
    for path in EXPECTED_DIRS:
        all_ok = check_dir(path) and all_ok

    print("\nChecking expected files:")
    for path in EXPECTED_FILES:
        all_ok = check_file(path) and all_ok

    if not all_ok:
        print("\nStructure verification failed.")
        return 1

    print("\nStructure verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
