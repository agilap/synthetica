"""
pipeline/exporter.py — ZIP archive creation for real and synthetic datasets.

Exposes DatasetExporter, which bundles:
  real/           — resized real images (256×256)
  synthetic/      — accepted synthetic images
  real/<name>.json       — per-image metadata for each real image
  synthetic/<name>.json  — per-image metadata for each synthetic image
  manifest.json          — summary of the full export

into a single ZIP archive ready for download.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path


class DatasetExporter:
    """Build a ZIP dataset archive from real and accepted synthetic images.

    Args:
        output_path: Destination path for the ZIP file.
    """

    def __init__(self, output_path: str | Path = "dataset.zip") -> None:
        self.output_path = Path(output_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        real_image_paths: list[str],
        real_fingerprints: list[dict],
        accepted_records: list[dict],
    ) -> Path:
        """Write a ZIP archive organised into real/ and synthetic/ folders.

        Each image is accompanied by a JSON sidecar file containing its
        metadata. A top-level ``manifest.json`` summarises the export.

        Args:
            real_image_paths:  Paths to the resized real images (from
                               ``RealDataIngestor.load_images``).
            real_fingerprints: Parallel list of fingerprint dicts for each
                               real image (from
                               ``build_distribution_fingerprint``
                               [``"individual_fingerprints"``]).
            accepted_records:  List of generation dicts from
                               ``QualityFilter.filter_batch`` accepted list.
                               Each must have ``image_path``, ``seed``,
                               ``prompt``, and ``similarity_score``.

        Returns:
            Path to the created ZIP archive.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(self.output_path, "w", zipfile.ZIP_DEFLATED) as zf:

            # ── Real images ──────────────────────────────────────────────
            for img_path, fp in zip(real_image_paths, real_fingerprints):
                p = Path(img_path)
                zf.write(p, arcname=f"real/{p.name}")

                meta = {
                    "filename": p.name,
                    "source": "real",
                    "fingerprint": {
                        k: (v if not hasattr(v, "tolist") else v.tolist())
                        for k, v in fp.items()
                    },
                }
                zf.writestr(f"real/{p.stem}.json", json.dumps(meta, indent=2))

            # ── Accepted synthetic images ────────────────────────────────
            for record in accepted_records:
                p = Path(record["image_path"])
                zf.write(p, arcname=f"synthetic/{p.name}")

                meta = {
                    "filename": p.name,
                    "source": "synthetic",
                    "seed": record["seed"],
                    "prompt": record["prompt"],
                    "similarity_score": record["similarity_score"],
                }
                zf.writestr(f"synthetic/{p.stem}.json", json.dumps(meta, indent=2))

            # ── Top-level manifest ───────────────────────────────────────
            manifest = {
                "real_count": len(real_image_paths),
                "synthetic_count": len(accepted_records),
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        print(
            f"Exported {len(real_image_paths)} real + {len(accepted_records)} "
            f"synthetic images → {self.output_path}"
        )
        return self.output_path
