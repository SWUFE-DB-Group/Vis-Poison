import argparse
import json
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT_DIR / "results" / "retrieval_p2"
BACKEND_ORDER = ["clip", "siglip", "qwen"]
DATASET_ORDER = ["COCO", "Flickr30k"]
KB_SIZE_ORDER = ["1k", "10k", "30k"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def pair(top1: float | None, top3: float | None) -> str:
    return f"{pct(top1)} / {pct(top3)}"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return " | ".join(
            cell.ljust(widths[i]) if i < 2 else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        )

    divider = "-+-".join("-" * width for width in widths)
    lines = [fmt_row(headers), divider]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def extract_result(
    result_dir: Path,
    backend_name: str,
    dataset_name: str,
    kb_size: str,
) -> str:
    result_path = (
        result_dir
        / backend_name
        / f"retrieval_{backend_name}_{dataset_name.lower()}_{kb_size}.json"
    )
    if not result_path.exists():
        return "-"
    loaded = load_json(result_path)
    if not isinstance(loaded, dict):
        return "-"
    summary = loaded.get("summary", {})
    if not isinstance(summary, dict):
        return "-"
    return pair(summary.get("top1_rate"), summary.get("top3_rate"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print P2 retrieval stats for all backends, datasets, and KB sizes."
    )
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    headers = ["backend", "dataset"]
    for kb_size in KB_SIZE_ORDER:
        headers.append(f"{kb_size} t1/t3")

    rows: list[list[str]] = []
    for backend_name in BACKEND_ORDER:
        for dataset_name in DATASET_ORDER:
            row = [backend_name, dataset_name]
            for kb_size in KB_SIZE_ORDER:
                row.append(extract_result(result_dir, backend_name, dataset_name, kb_size))
            rows.append(row)

    print(format_table(headers, rows))


if __name__ == "__main__":
    main()
