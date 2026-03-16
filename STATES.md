# SyntheticImageGen — Application States
> State machine, data contracts, and transition rules for every pipeline phase

---

## State Diagram

```
                            ┌─────────────┐
                            │    IDLE     │◀──────────────────────────────┐
                            │  (startup)  │                               │
                            └──────┬──────┘                               │
                                   │  user uploads folder/ZIP              │
                                   ▼                                       │
                            ┌─────────────┐                               │
                     ┌─────▶│  INGESTING  │                               │
                     │      │  (Phase 1)  │                               │
                     │      └──────┬──────┘                               │
                     │             │  images validated + resized           │
                     │             ▼                                       │
                     │      ┌─────────────────┐                           │
                     │      │  FINGERPRINTING  │                          │
                     │      │  (DINOv2 pass)   │                          │
                     │      └────────┬─────────┘                          │
                     │               │  centroid + std computed            │
                     │               ▼                                     │
                     │      ┌─────────────┐                               │
                     │      │   READY_TO  │◀─────────────────────┐        │
                     │      │    TRAIN    │                       │        │
                     │      └──────┬──────┘                       │        │
                     │             │  user clicks "Fine-Tune"     │        │
                     │             ▼                               │        │
                     │      ┌─────────────┐                       │        │
         re-upload   │      │  TRAINING   │──── user cancels ────▶│        │
                     │      │  (Phase 2)  │                       │        │
                     │      └──────┬──────┘                       │        │
                     │             │  LoRA checkpoint saved        │        │
                     │             ▼                               │        │
                     │      ┌─────────────────┐                   │        │
                     │      │  READY_TO_GEN   │◀── adjust N ──────┘        │
                     │      │  (awaiting N)   │                             │
                     │      └──────┬──────────┘                             │
                     │             │  user clicks "Generate"                │
                     │             ▼                                        │
                     │      ┌─────────────┐                                │
                     │      │  GENERATING │                                │
                     │      │  (Phase 3)  │                                │
                     │      └──────┬──────┘                                │
                     │             │  N accepted images collected          │
                     │             ▼                                        │
                     │      ┌─────────────┐                                │
                     │      │  PREVIEWING │                                │
                     │      │  (grid UI)  │                                │
                     │      └──────┬──────┘                                │
                     │             │  user clicks "Download ZIP"           │
                     │             ▼                                        │
                     │      ┌─────────────┐                                │
                     │      │  EXPORTING  │──── export complete ──────────▶│
                     │      │  (Phase 4)  │                                │
                     │      └─────────────┘                                │
                     │                                                      │
                     └───────────────────────── ERROR ─────────────────────┘
                                              (any phase)
```

---

## State Definitions

### `IDLE`

The application has launched but no data has been loaded yet.

```python
@dataclass
class IdleState:
    session_id: str           # UUID, created at app launch
    ui_message: str = "Upload a folder or ZIP of real images to begin."
    gpu_available: bool       # checked at startup
    vram_mb: int              # read from torch.cuda.get_device_properties()
    quantization_mode: str    # "nf4" | "fp16" | "cpu" — auto-selected at startup
```

**Entry conditions:** App launch, or user clicks "Reset / Upload New Dataset".  
**Exit conditions:** User uploads files → transition to `INGESTING`.  
**Side effects on entry:**
- Call `torch.cuda.empty_cache()`
- Delete any temp session directory from previous run
- Log GPU info to console

---

### `INGESTING`

Files are being unpacked, validated, and resized.

```python
@dataclass
class IngestingState:
    session_id: str
    upload_path: Path           # temp directory with raw uploads
    files_found: int = 0
    files_valid: int = 0
    files_rejected: int = 0
    rejection_reasons: list[str] = field(default_factory=list)
    progress: float = 0.0       # 0.0 – 1.0
```

**Entry conditions:** Files uploaded by user.  
**Exit conditions:**
- `files_valid >= 15` → transition to `FINGERPRINTING`
- `files_valid < 15` → back to `IDLE` with warning: "Need at least 15 valid images for reliable fingerprinting."
- Any I/O error → transition to `ERROR`

**Validation rules:**
- File extension in `{.jpg, .jpeg, .png, .webp}`
- File size > 0 bytes and < 50 MB
- Decodable by Pillow without exception
- Minimum dimension ≥ 64 px (either axis)

**Side effects:**
- Create session directory: `./sessions/{session_id}/real/`
- Save resized 256×256 images as PNG
- Log rejected files with reasons

---

### `FINGERPRINTING`

DINOv2-S ViT extracts semantic embeddings from all real images. Runs on GPU if available, CPU otherwise.

```python
@dataclass
class FingerprintingState:
    session_id: str
    image_paths: list[Path]
    embeddings: np.ndarray | None = None      # shape (N, 384)
    centroid: np.ndarray | None = None        # shape (384,)
    centroid_std: float | None = None
    intra_set_mean_sim: float | None = None   # avg cosine sim within real set
    progress: float = 0.0
    device_used: str = "cuda"                 # or "cpu"
```

**Entry conditions:** `INGESTING` succeeded.  
**Exit conditions:**
- Embeddings computed → transition to `READY_TO_TRAIN`
- GPU OOM → retry on CPU, then proceed
- Model download failure → transition to `ERROR`

**Side effects:**
- Save `centroid.npy`, `embeddings.npy` to `./sessions/{session_id}/`
- Free DINOv2 model from VRAM immediately after: `del model; torch.cuda.empty_cache()`
- Log `intra_set_mean_sim` — if > 0.95, warn: "Dataset is very homogeneous; threshold will be tight."

---

### `READY_TO_TRAIN`

Fingerprinting is done. User reviews dataset stats and configures training before starting.

```python
@dataclass
class ReadyToTrainState:
    session_id: str
    real_count: int
    centroid: np.ndarray
    centroid_std: float
    intra_set_mean_sim: float
    # UI-configurable training parameters (with defaults)
    lora_rank: int = 8
    lora_alpha: int = 16
    train_steps: int = 1000
    learning_rate: float = 1e-4
    # Display
    recommended_steps: int = 0    # computed: max(500, real_count * 20)
```

**Entry conditions:** `FINGERPRINTING` succeeded.  
**Exit conditions:**
- User clicks "Start Fine-Tune" → transition to `TRAINING`
- User uploads new folder → transition to `INGESTING`

**UI shown:** dataset stats card (image count, mean similarity, recommended steps), training hyperparameter sliders, VRAM estimate badge.

---

### `TRAINING`

LoRA fine-tuning is running. GPU is fully occupied. No other GPU operations occur.

```python
@dataclass
class TrainingState:
    session_id: str
    current_step: int = 0
    total_steps: int = 1000
    current_loss: float | None = None
    loss_history: list[tuple[int, float]] = field(default_factory=list)
    checkpoint_path: Path | None = None      # saved every 200 steps
    eta_seconds: int | None = None
    cancelled: bool = False
    peak_vram_mb: int = 0                    # polled via torch.cuda.max_memory_allocated()
```

**Entry conditions:** User confirmed training config in `READY_TO_TRAIN`.  
**Exit conditions:**
- `current_step == total_steps` → save final checkpoint → transition to `READY_TO_GEN`
- `cancelled == True` → save last checkpoint → transition to `READY_TO_TRAIN`
- OOM error → transition to `ERROR` with message: "OOM during training. Try reducing LoRA rank to 4 or enabling CPU offload."
- NaN loss for 3+ consecutive steps → transition to `ERROR`

**Checkpointing:** Save adapter weights every 200 steps to `./sessions/{session_id}/checkpoints/step_{N}/`. On crash recovery, the UI offers to resume from last checkpoint.

**Side effects on exit:**
- `torch.cuda.empty_cache()`
- Log final loss + peak VRAM to `training_log.json`

---

### `READY_TO_GEN`

Training is complete. User selects how many synthetic images to generate.

```python
@dataclass
class ReadyToGenState:
    session_id: str
    checkpoint_path: Path
    training_loss_final: float
    # UI-configurable generation parameters
    n_generate: int = 100          # target accepted images
    similarity_threshold: float    # auto-computed adaptive τ; user can override
    guidance_scale: float = 1.5    # LCM range: 1.0 – 2.0
    inference_steps: int = 4       # LCM default; 1–8 range
    output_resolution: tuple[int, int] = (256, 256)
```

**Entry conditions:** `TRAINING` completed or cancelled with checkpoint.  
**Exit conditions:**
- User clicks "Generate" → transition to `GENERATING`
- User adjusts N / threshold → stays in `READY_TO_GEN`, updates preview stats

**UI shown:** estimated generation time (N × ~1.2s), acceptance rate prediction, VRAM usage badge, threshold slider with real distribution overlay chart.

---

### `GENERATING`

Synthetic images are being generated and filtered.

```python
@dataclass
class GeneratingState:
    session_id: str
    n_target: int                        # how many accepted images user wants
    n_generated: int = 0                 # total attempts (accepted + rejected)
    n_accepted: int = 0                  # passed both filter tiers
    n_rejected: int = 0
    accepted_images: list[Path] = field(default_factory=list)
    accepted_metadata: list[dict] = field(default_factory=list)
    fid_estimate: float | None = None    # computed every 50 accepted images
    current_mean_sim: float = 0.0
    progress: float = 0.0               # n_accepted / n_target
    eta_seconds: int | None = None
    cancelled: bool = False
```

**Entry conditions:** User clicked "Generate" in `READY_TO_GEN`.  
**Exit conditions:**
- `n_accepted == n_target` → transition to `PREVIEWING`
- `cancelled == True` → transition to `PREVIEWING` with partial results (if n_accepted ≥ 1)
- Rejection rate > 80% for 50 consecutive attempts → transition to `ERROR` with message: "Model may have diverged. Try re-training with more steps or lower LR."
- OOM → transition to `ERROR`

**Side effects:**
- Stream preview grid to Gradio UI every 10 accepted images
- Log each image's metadata to `./sessions/{session_id}/synthetic/`

---

### `PREVIEWING`

Generation is done. User reviews side-by-side real vs synthetic grid and dataset stats.

```python
@dataclass
class PreviewingState:
    session_id: str
    real_images: list[Path]
    synthetic_images: list[Path]
    dataset_report: DatasetReport
    # DatasetReport fields:
    #   real_count, synthetic_count, rejected_count
    #   acceptance_rate, fid_estimate, mean_similarity
    #   std_similarity, training_steps, total_generation_time_s
```

**Entry conditions:** `GENERATING` completed or cancelled with results.  
**Exit conditions:**
- User clicks "Download ZIP" → transition to `EXPORTING`
- User clicks "Generate More" → transition to `READY_TO_GEN` with existing checkpoint
- User clicks "Reset" → transition to `IDLE`

**UI shown:** 4×4 grid (alternating real/synthetic), dataset report card, FID score badge (if ≥ 50 synthetic images), download button.

---

### `EXPORTING`

ZIP archive is being assembled.

```python
@dataclass
class ExportingState:
    session_id: str
    zip_path: Path | None = None
    progress: float = 0.0    # 0.0 – 1.0
    bytes_written: int = 0
```

**Entry conditions:** User clicked "Download ZIP" in `PREVIEWING`.  
**Exit conditions:**
- ZIP written to `./sessions/{session_id}/export/` → serve as Gradio file download → transition to `IDLE`
- I/O error → transition to `ERROR`

**ZIP structure enforced:**
```
synthetic_dataset_{session_id[:8]}_{timestamp}.zip
├── real/
│   └── *.jpg  (resized 256×256 originals)
├── synthetic/
│   ├── syn_*.jpg
│   └── syn_*_meta.json
└── dataset_report.json
```

---

### `ERROR`

A non-recoverable failure has occurred in any phase.

```python
@dataclass
class ErrorState:
    session_id: str
    phase: str                  # which phase failed
    error_type: str             # "OOM" | "NaN_loss" | "rejection_overflow" | "io_error" | "model_download"
    error_message: str          # human-readable description
    recoverable: bool           # if True, show "Try Again" button
    recovery_suggestion: str    # e.g., "Reduce LoRA rank to 4"
    traceback: str | None       # stored in session log, not shown to user
```

**Entry conditions:** Any phase raises an unhandled exception or hits a defined failure condition.  
**Exit conditions:**
- `recoverable == True` + user clicks "Try Again" → return to the failed phase's preceding state
- User clicks "Reset" → transition to `IDLE`

**Specific error handling:**

| Error | Phase | Recovery |
|---|---|---|
| CUDA OOM during training | `TRAINING` | Lower LoRA rank (4 → 2), enable CPU offload, retry |
| CUDA OOM during inference | `GENERATING` | Enable attention slicing slice_size=1, batch=1, retry |
| NaN loss | `TRAINING` | Lower LR (1e-4 → 5e-5), reset to checkpoint, retry |
| Rejection rate > 80% | `GENERATING` | Prompt user to re-train with more steps |
| Model download failure | `FINGERPRINTING` / `TRAINING` | Retry with cached model; if no cache, show offline instructions |
| Corrupt upload | `INGESTING` | Skip file, continue with rest |
| Disk full | `EXPORTING` | Show disk usage; ask user to free space |

---

## State Transition Table

| From | Event | To | Guard |
|---|---|---|---|
| `IDLE` | files_uploaded | `INGESTING` | files not empty |
| `INGESTING` | validation_complete | `FINGERPRINTING` | valid_count ≥ 15 |
| `INGESTING` | validation_complete | `IDLE` | valid_count < 15 |
| `INGESTING` | error | `ERROR` | — |
| `FINGERPRINTING` | embeddings_ready | `READY_TO_TRAIN` | centroid computed |
| `FINGERPRINTING` | error | `ERROR` | — |
| `READY_TO_TRAIN` | start_training | `TRAINING` | — |
| `READY_TO_TRAIN` | re_upload | `INGESTING` | — |
| `TRAINING` | step == total_steps | `READY_TO_GEN` | loss not NaN |
| `TRAINING` | cancelled | `READY_TO_TRAIN` | checkpoint saved |
| `TRAINING` | nan_loss | `ERROR` | consecutive NaN ≥ 3 |
| `TRAINING` | oom | `ERROR` | — |
| `READY_TO_GEN` | start_generate | `GENERATING` | n_generate ≥ 1 |
| `READY_TO_GEN` | re_upload | `INGESTING` | — |
| `GENERATING` | n_accepted == target | `PREVIEWING` | — |
| `GENERATING` | cancelled | `PREVIEWING` | n_accepted ≥ 1 |
| `GENERATING` | rejection_overflow | `ERROR` | reject_rate > 0.8 for 50 steps |
| `GENERATING` | oom | `ERROR` | — |
| `PREVIEWING` | download_clicked | `EXPORTING` | — |
| `PREVIEWING` | generate_more | `READY_TO_GEN` | checkpoint exists |
| `PREVIEWING` | reset | `IDLE` | — |
| `EXPORTING` | zip_ready | `IDLE` | file served |
| `EXPORTING` | error | `ERROR` | — |
| `ERROR` | try_again | prev_entry_state | recoverable == True |
| `ERROR` | reset | `IDLE` | — |

---

## Session Persistence

Each pipeline run is isolated under `./sessions/{session_id}/`:

```
sessions/{session_id}/
├── real/                   # resized real images (written at INGESTING)
├── embeddings/
│   ├── centroid.npy        # written at FINGERPRINTING
│   └── embeddings.npy
├── checkpoints/
│   ├── step_200/           # LoRA adapter weights (written at TRAINING)
│   ├── step_400/
│   └── final/
├── synthetic/              # generated images + metadata (written at GENERATING)
├── export/                 # final ZIP (written at EXPORTING)
├── training_log.json       # step-by-step loss + VRAM
└── state.json              # last known state (for crash recovery)
```

`state.json` is written atomically (write-to-temp + rename) at every state transition so that a hard crash can be recovered by reloading the session:

```json
{
  "session_id": "abc12345-...",
  "current_state": "TRAINING",
  "timestamp": "2025-03-15T14:23:01Z",
  "last_checkpoint": "checkpoints/step_600",
  "real_count": 50,
  "lora_rank": 8,
  "train_steps_target": 1000,
  "train_steps_done": 600
}
```

---

## Gradio UI State Mapping

| App State | Gradio components visible | Components interactive |
|---|---|---|
| `IDLE` | Upload widget, VRAM badge | Upload only |
| `INGESTING` | Progress bar, log output | Cancel |
| `FINGERPRINTING` | Progress bar, "Analyzing images…" | Cancel |
| `READY_TO_TRAIN` | Stats card, hyperparameter sliders | Sliders, Start button |
| `TRAINING` | Loss curve (live), VRAM gauge, step counter | Cancel |
| `READY_TO_GEN` | Stats card, N slider, threshold slider, time estimate | Sliders, Generate button |
| `GENERATING` | Progress bar, live preview grid, acceptance rate | Cancel |
| `PREVIEWING` | Full grid, dataset report, FID badge | Download, Generate More, Reset |
| `EXPORTING` | Spinner, progress bar | — |
| `ERROR` | Error card, suggestion text | Try Again (if recoverable), Reset |
