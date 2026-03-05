"""
app.py — Entry point for the synthetic-image-gen application.

Launches the Gradio web interface that wires together the full pipeline:
upload → ingest → train (LoRA) → generate → filter → export.
Users interact with this file to configure pipeline parameters and
trigger end-to-end synthetic dataset generation.
"""


def main() -> None:
    """Launch the Gradio UI."""
    pass


if __name__ == "__main__":
    main()
