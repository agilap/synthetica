from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROMPTS: list[tuple[str, str]] = [
    ("0-STATE", "Build tracker script and BUILD_STATE.md"),
    ("0-ENV", "Environment setup script + requirements.txt"),
    ("0-STRUCT", "Project skeleton and directory structure"),
    ("0-TYPES", "Shared dataclasses (utils/types.py)"),
    ("1-INGESTOR", "Image ingestor (pipeline/ingestor.py)"),
    ("1-FINGERPRINT", "DINOv2 fingerprinter (utils/fingerprint.py)"),
    ("2-QUANTIZE", "Quantization utilities (utils/quantize.py)"),
    ("2-TRAINER", "LoRA fine-tuner (pipeline/trainer.py)"),
    ("3-GENERATOR", "Synthetic image generator (pipeline/generator.py)"),
    ("3-FILTER", "Two-tier quality filter (pipeline/filter.py)"),
    ("3-SIMILARITY", "Similarity math + FID proxy (utils/similarity.py)"),
    ("4-EXPORTER", "Dataset exporter (pipeline/exporter.py)"),
    ("4-VRAM", "VRAM guard utilities (utils/vram_guard.py)"),
    ("5-APP", "Gradio app layout (app.py)"),
    ("5-APP-WIRE", "Gradio event handler wiring (app.py update)"),
    ("6-TEST-SMOKE", "Smoke test suite (tests/test_smoke.py)"),
]

PENDING = "⬜ pending"
DONE = "✅ done"


def build_state_path() -> Path:
    return Path(__file__).resolve().parent.parent / "BUILD_STATE.md"


def default_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for prompt_id, description in PROMPTS:
        rows[prompt_id] = {
            "id": prompt_id,
            "description": description,
            "status": PENDING,
            "completed_at": "",
            "commit_hash": "",
        }
    return rows


def render_markdown(rows: dict[str, dict[str, str]]) -> str:
    lines = [
        "| ID | Description | Status | Completed At | Commit Hash |",
        "| --- | --- | --- | --- | --- |",
    ]
    for prompt_id, description in PROMPTS:
        row = rows[prompt_id]
        lines.append(
            f"| {prompt_id} | {description} | {row['status']} | {row['completed_at']} | {row['commit_hash']} |"
        )
    return "\n".join(lines) + "\n"


def parse_table(content: str) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [part.strip() for part in line.split("|")[1:-1]]
        if len(cells) != 5:
            continue
        if cells[0] == "ID":
            continue
        if all(cell and all(ch in "-:" for ch in cell) for cell in cells):
            continue

        prompt_id, description, status, completed_at, commit_hash = cells
        parsed[prompt_id] = {
            "id": prompt_id,
            "description": description,
            "status": status,
            "completed_at": completed_at,
            "commit_hash": commit_hash,
        }
    return parsed


def ensure_build_state_exists(path: Path) -> None:
    if path.exists():
        return
    rows = default_rows()
    path.write_text(render_markdown(rows), encoding="utf-8")


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    ensure_build_state_exists(path)
    content = path.read_text(encoding="utf-8")
    parsed = parse_table(content)
    rows = default_rows()

    for prompt_id in rows:
        if prompt_id in parsed:
            rows[prompt_id]["status"] = parsed[prompt_id].get("status", rows[prompt_id]["status"])
            rows[prompt_id]["completed_at"] = parsed[prompt_id].get("completed_at", "")
            rows[prompt_id]["commit_hash"] = parsed[prompt_id].get("commit_hash", "")

    return rows


def write_rows(path: Path, rows: dict[str, dict[str, str]]) -> None:
    path.write_text(render_markdown(rows), encoding="utf-8")


def latest_git_hash() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "—"


def done_count(rows: dict[str, dict[str, str]]) -> int:
    return sum(1 for row in rows.values() if row["status"] == DONE)


def next_pending(rows: dict[str, dict[str, str]]) -> str:
    for prompt_id, _ in PROMPTS:
        if rows[prompt_id]["status"] != DONE:
            return prompt_id
    return "None"


def print_status(rows: dict[str, dict[str, str]]) -> None:
    headers = ["ID", "Status", "Completed At", "Commit Hash"]
    table_rows: list[list[str]] = []
    for prompt_id, _ in PROMPTS:
        row = rows[prompt_id]
        table_rows.append(
            [
                prompt_id,
                row["status"] or "—",
                row["completed_at"] or "—",
                row["commit_hash"] or "—",
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in table_rows:
        print(fmt(row))

    total = len(PROMPTS)
    complete = done_count(rows)
    print()
    print(f"Complete: {complete}/{total}")
    print(f"Next pending: {next_pending(rows)}")


def mark_done(prompt_id: str, rows: dict[str, dict[str, str]], path: Path) -> int:
    if prompt_id not in rows:
        print(f"Unknown prompt ID: {prompt_id}")
        return 1

    rows[prompt_id]["status"] = DONE
    rows[prompt_id]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    rows[prompt_id]["commit_hash"] = latest_git_hash()
    write_rows(path, rows)

    complete = done_count(rows)
    print(f"✅ Marked {prompt_id} as done. ({complete}/{len(PROMPTS)} complete)")
    return 0


def reset_prompt(prompt_id: str, rows: dict[str, dict[str, str]], path: Path) -> int:
    if prompt_id not in rows:
        print(f"Unknown prompt ID: {prompt_id}")
        return 1

    rows[prompt_id]["status"] = PENDING
    rows[prompt_id]["completed_at"] = ""
    rows[prompt_id]["commit_hash"] = ""
    write_rows(path, rows)

    print(f"Reset {prompt_id} to pending.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mark SyntheticImageGen build prompts done/pending.")
    parser.add_argument("prompt_id", nargs="?", help="Prompt ID to mark done, e.g., 0-STATE")
    parser.add_argument("--status", action="store_true", help="Print full build status table")
    parser.add_argument("--reset", metavar="ID", help="Reset one prompt back to pending")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    action_count = int(bool(args.prompt_id)) + int(args.status) + int(bool(args.reset))

    if action_count != 1:
        print(
            "Usage:\n"
            "  python scripts/mark_done.py <PROMPT_ID>\n"
            "  python scripts/mark_done.py --status\n"
            "  python scripts/mark_done.py --reset <ID>"
        )
        return 1

    path = build_state_path()
    rows = load_rows(path)

    if args.status:
        print_status(rows)
        return 0

    if args.reset:
        return reset_prompt(args.reset, rows, path)

    return mark_done(args.prompt_id, rows, path)


if __name__ == "__main__":
    sys.exit(main())