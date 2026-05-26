from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT_DIR / "dataset" / "webqa_final_category_difficulty_sample_70.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "defenses" / "sample"
RANDOM_SEED = 6
SAMPLE_PER_CATEGORY_PER_DIFFICULTY = 10
DIFFICULTIES = ["easy", "hard"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def get_category(record: dict[str, Any]) -> str:
    return (
        str(record.get("counterfactual_edit", {}).get("category", "")).strip()
        or "unknown"
    )


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(row: list[str]) -> str:
        cells = []
        for i, cell in enumerate(row):
            cells.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        return " | ".join(cells)

    sep = "-+-".join("-" * width for width in widths)
    return "\n".join([render(headers), sep] + [render(row) for row in rows])


def sample_records(data: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in data:
        if not isinstance(record, dict):
            continue
        difficulty = str(record.get("difficulty", "")).strip().lower()
        if difficulty not in DIFFICULTIES:
            continue
        grouped[(get_category(record), difficulty)].append(record)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for category in sorted({category for category, _ in grouped}):
        for difficulty in DIFFICULTIES:
            bucket = grouped[(category, difficulty)]
            if len(bucket) < SAMPLE_PER_CATEGORY_PER_DIFFICULTY:
                raise ValueError(
                    f"Not enough records for category={category}, difficulty={difficulty}: need {SAMPLE_PER_CATEGORY_PER_DIFFICULTY}, got {len(bucket)}"
                )
            sampled.extend(rng.sample(bucket, SAMPLE_PER_CATEGORY_PER_DIFFICULTY))
    return sampled


def main() -> None:
    data = load_json(DEFAULT_INPUT)
    if not isinstance(data, list):
        raise TypeError(f"{DEFAULT_INPUT}: expected JSON list")
    sampled = sample_records(data, RANDOM_SEED)
    output_json = DEFAULT_OUTPUT_DIR / "sample_180.json"
    report_json = DEFAULT_OUTPUT_DIR / "sample_180_report.json"
    report_md = DEFAULT_OUTPUT_DIR / "sample_180_report.md"
    save_json(output_json, sampled)

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in sampled:
        counts[
            (get_category(record), str(record.get("difficulty", "")).strip().lower())
        ] += 1
    rows = []
    for category in sorted({key[0] for key in counts}):
        rows.append(
            [
                category,
                str(counts[(category, "easy")]),
                str(counts[(category, "hard")]),
                str(counts[(category, "easy")] + counts[(category, "hard")]),
            ]
        )
    table = format_table(["category", "easy", "hard", "total"], rows)
    payload = {
        "input_path": str(DEFAULT_INPUT),
        "sample_output_path": str(output_json),
        "random_seed": RANDOM_SEED,
        "sample_per_category_per_difficulty": SAMPLE_PER_CATEGORY_PER_DIFFICULTY,
        "num_samples": len(sampled),
        "rows": rows,
    }
    save_json(report_json, payload)
    save_text(report_md, "# Defense Sample\n\n```\n" + table + "\n```\n")
    print(f"Saved sample to: {output_json}")
    print(f"Saved report to: {report_json}")


if __name__ == "__main__":
    main()
