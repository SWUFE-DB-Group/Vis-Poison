from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[3]
BASE_DIR = ROOT_DIR / "results" / "defenses" / "is_valid"
INPUTS = {
    "ours_factcheck": BASE_DIR / "ours" / "caption_180_ours_factcheck.json",
    "ours_jailbreak": BASE_DIR
    / "ours"
    / "caption_180_ours_jailbreak_after_factcheck.json",
    "mm_factcheck": BASE_DIR / "mm" / "caption_180_mm_factcheck.json",
    "mm_jailbreak": BASE_DIR / "mm" / "caption_180_mm_jailbreak_after_factcheck.json",
    "poisoned_factcheck": BASE_DIR / "poisoned" / "caption_180_poisoned_factcheck.json",
    "poisoned_jailbreak": BASE_DIR
    / "poisoned"
    / "caption_180_poisoned_jailbreak_after_factcheck.json",
    "eye_factcheck": BASE_DIR / "eye" / "caption_180_eye_factcheck.json",
}
OUTPUT_JSON = BASE_DIR / "stats_report.json"
OUTPUT_MD = BASE_DIR / "stats_report.md"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(cells: list[str]) -> str:
        return "  ".join(
            cell.rjust(widths[i]) if i else cell.ljust(widths[i])
            for i, cell in enumerate(cells)
        )

    return "\n".join(
        [render(headers), "  ".join("-" * width for width in widths)]
        + [render(row) for row in rows]
    )


def load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = load_json(path)
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected JSON object")
    return data


def summarize_factcheck_then_jailbreak(
    group: str, factcheck_path: Path, jailbreak_path: Path
) -> dict[str, Any]:
    factcheck = load_json(factcheck_path)
    total = int(factcheck.get("factcheck_total", 0))
    factual = int(factcheck.get("factcheck_factual_count", 0))
    errors = int(factcheck.get("factcheck_error_count", 0))
    jailbreak = load_optional(jailbreak_path) or {}
    jailbreak_pass = (
        int(jailbreak.get("jailbreak_pass_count", factual)) if factual else 0
    )
    return {
        "group": group,
        "pipeline_order": ["factcheck", "jailbreak"],
        "total": total,
        "jailbreak_pass_count": jailbreak_pass,
        "jailbreak_pass_rate": (jailbreak_pass / total) if total else 0.0,
        "factcheck_pass_count": factual,
        "factcheck_pass_rate": (factual / total) if total else 0.0,
        "error_count": errors + int(jailbreak.get("jailbreak_error_count", 0)),
    }


def summarize_eye(group: str, factcheck_path: Path) -> dict[str, Any]:
    factcheck = load_json(factcheck_path)
    total = int(factcheck.get("factcheck_total", 0))
    factual = int(factcheck.get("factcheck_factual_count", 0))
    errors = int(factcheck.get("factcheck_error_count", 0))
    return {
        "group": group,
        "pipeline_order": ["jailbreak", "factcheck"],
        "total": total,
        "jailbreak_pass_count": total,
        "jailbreak_pass_rate": 1.0 if total else 0.0,
        "factcheck_pass_count": factual,
        "factcheck_pass_rate": (factual / total) if total else 0.0,
        "error_count": errors,
    }


def main() -> None:
    summaries = [
        summarize_factcheck_then_jailbreak(
            "ours", INPUTS["ours_factcheck"], INPUTS["ours_jailbreak"]
        ),
        summarize_factcheck_then_jailbreak(
            "mm", INPUTS["mm_factcheck"], INPUTS["mm_jailbreak"]
        ),
        summarize_factcheck_then_jailbreak(
            "poisoned", INPUTS["poisoned_factcheck"], INPUTS["poisoned_jailbreak"]
        ),
        summarize_eye("eye", INPUTS["eye_factcheck"]),
    ]
    rows = [
        [
            item["group"],
            str(item["total"]),
            str(item["jailbreak_pass_count"]),
            pct(float(item["jailbreak_pass_rate"])),
            str(item["factcheck_pass_count"]),
            pct(float(item["factcheck_pass_rate"])),
            str(item["error_count"]),
        ]
        for item in summaries
    ]
    table = fmt_table(
        [
            "Group",
            "Total",
            "Jailbreak Pass",
            "Jailbreak Rate",
            "Factcheck Pass",
            "Factcheck Rate",
            "Errors",
        ],
        rows,
    )
    save_json(OUTPUT_JSON, {"summaries": summaries, "table_rows": rows})
    OUTPUT_MD.write_text(
        "# isValid Stats\n\n```\n" + table + "\n```\n", encoding="utf-8"
    )
    print(table)
    print(f"Saved JSON report to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
