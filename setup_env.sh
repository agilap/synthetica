#!/usr/bin/env bash
# RTX 3050 Ti 4GB — CUDA 11.8 — pinned for VRAM stability

set -euo pipefail

if command -v py >/dev/null 2>&1; then
  PY_BOOTSTRAP=(py -3.10)
elif command -v python >/dev/null 2>&1; then
  PY_BOOTSTRAP=(python)
else
  echo "Python 3.10 was not found in PATH. Install Python 3.10 and retry."
  exit 1
fi

echo "Creating virtual environment: .venv"
"${PY_BOOTSTRAP[@]}" -m venv .venv

VENV_PY=".venv/Scripts/python.exe"

echo "Upgrading pip"
"$VENV_PY" -m pip install --upgrade pip

echo "Installing torch 2.2.0 CUDA 11.8 wheels (separate step)"
"$VENV_PY" -m pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118

echo "Installing remaining dependencies in pinned order"
"$VENV_PY" -m pip install xformers==0.0.26.post1

# bitsandbytes fallback for Windows: if standard install fails, use bitsandbytes-windows fork
if ! "$VENV_PY" -m pip install bitsandbytes==0.43.3; then
  echo "bitsandbytes install failed on Windows, falling back to bitsandbytes-windows"
  "$VENV_PY" -m pip install bitsandbytes-windows==0.43.3
fi

"$VENV_PY" -m pip install diffusers==0.29.2
"$VENV_PY" -m pip install peft==0.11.1
"$VENV_PY" -m pip install transformers==4.41.2
"$VENV_PY" -m pip install accelerate==0.30.1
"$VENV_PY" -m pip install torchvision==0.17.0
"$VENV_PY" -m pip install gradio==4.25.0
"$VENV_PY" -m pip install opencv-python==4.9.0.80
"$VENV_PY" -m pip install Pillow==10.3.0
"$VENV_PY" -m pip install numpy==1.26.4
"$VENV_PY" -m pip install "pyyaml>=6.0"
"$VENV_PY" -m pip install "pytest>=8.0"
"$VENV_PY" -m pip install "pytest-mock>=3.12"
"$VENV_PY" -m pip install "scipy>=1.12"

echo "Running verification imports"
"$VENV_PY" - <<'PY'
import importlib


def check_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:
        return None, exc


torch_mod, torch_err = check_import("torch")
if torch_err:
    print(f"[FAIL] torch import failed: {torch_err}")
else:
    cuda_available = torch_mod.cuda.is_available()
    if cuda_available:
        device_name = torch_mod.cuda.get_device_name(0)
    else:
        device_name = "N/A"
    print(f"[OK] torch CUDA available: {cuda_available}")
    print(f"[OK] torch CUDA device: {device_name}")

bnb_mod, bnb_err = check_import("bitsandbytes")
if bnb_err:
    print(f"[FAIL] bitsandbytes import failed: {bnb_err}")
else:
    print(f"[OK] bitsandbytes version: {getattr(bnb_mod, '__version__', 'unknown')}")

xformers_mod, xformers_err = check_import("xformers")
if xformers_err:
    print(f"[FAIL] xformers import failed: {xformers_err}")
else:
    print(f"[OK] xformers version: {getattr(xformers_mod, '__version__', 'unknown')}")
PY

echo "Setup complete. Activate with: source .venv/Scripts/activate"
