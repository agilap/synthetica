"""
pipeline/generator.py — Synthetic image generation with a fine-tuned LoRA model.

Assumes LoRA weights have already been trained (e.g. on Colab via
notebooks/train_colab.ipynb) and downloaded to a local directory.
No training takes place here.

Exposes SyntheticGenerator, which:
  1. Loads the SD-Turbo base pipeline from HuggingFace.
  2. Injects the local LoRA attention processors via load_attn_procs().
  3. Generates N images with unique per-image seeds and saves them to disk.
"""
from __future__ import annotations

from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

_MODEL_ID = "stabilityai/sd-turbo"


class SyntheticGenerator:
    """Generate synthetic images using SD-Turbo + a local LoRA adapter.

    Args:
        lora_weights_dir: Path to the downloaded ``lora_output/`` folder
                          produced by the training notebook.
        output_dir:       Directory where generated images will be saved.
                          Created automatically if it does not exist.
        seed:             Base RNG seed.  Image *i* uses seed ``seed + i``
                          so every image is reproducible independently.
    """

    def __init__(
        self,
        lora_weights_dir: str,
        output_dir: str = "data/synthetic/",
        seed: int = 42,
    ) -> None:
        self.lora_weights_dir = Path(lora_weights_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self._pipeline: StableDiffusionPipeline | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_pipeline(self) -> None:
        """Load SD-Turbo and inject LoRA attention processors.

        Moves the pipeline to CUDA when a GPU is available, otherwise
        falls back to CPU (functional but significantly slower).

        Raises:
            FileNotFoundError: If ``lora_weights_dir`` does not exist.
        """
        if not self.lora_weights_dir.exists():
            raise FileNotFoundError(
                f"LoRA weights directory not found: {self.lora_weights_dir}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        print(f"Loading {_MODEL_ID} base pipeline (device={device}) ...")
        pipeline = StableDiffusionPipeline.from_pretrained(
            _MODEL_ID,
            torch_dtype=torch_dtype,
        )
        pipeline = pipeline.to(device)

        # Inject the LoRA weights into the UNet attention layers.
        pipeline.unet.load_attn_procs(str(self.lora_weights_dir))

        # Disable the safety checker so it doesn't block synthetic outputs.
        pipeline.safety_checker = None

        self._pipeline = pipeline
        print(f"Pipeline loaded with LoRA weights from {self.lora_weights_dir}")

    def generate(
        self,
        n: int,
        prompt: str = "a high quality photo",
    ) -> list[dict]:
        """Generate *n* synthetic images and save them to ``output_dir``.

        Each image is generated with seed ``self.seed + i`` to ensure
        full reproducibility while keeping every image distinct.

        Args:
            n:      Number of images to generate.
            prompt: Text prompt passed to the SD-Turbo pipeline.

        Returns:
            A list of *n* dicts, each containing:

            * ``image_path`` – absolute path to the saved JPEG file.
            * ``seed``       – integer seed used for that image.
            * ``prompt``     – the prompt string used.

        Raises:
            RuntimeError: If :meth:`load_pipeline` has not been called first.
        """
        if self._pipeline is None:
            raise RuntimeError(
                "Pipeline is not loaded. Call load_pipeline() before generate()."
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []

        for i in range(n):
            image_seed = self.seed + i
            generator = torch.Generator(
                device=self._pipeline.device
            ).manual_seed(image_seed)

            image = self._pipeline(
                prompt=prompt,
                num_inference_steps=4,   # SD-Turbo is designed for 1-4 steps
                guidance_scale=0.0,      # SD-Turbo uses CFG-free generation
                generator=generator,
                height=256,
                width=256,
            ).images[0]

            filename = f"syn_{i:04d}.jpg"
            save_path = self.output_dir / filename
            image.save(save_path, format="JPEG", quality=95)

            records.append({
                "image_path": str(save_path.resolve()),
                "seed": image_seed,
                "prompt": prompt,
            })

            if (i + 1) % 10 == 0 or (i + 1) == n:
                print(f"Generated {i + 1}/{n}: {filename}  (seed={image_seed})")

        return records
